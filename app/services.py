"""领域服务：登记、校准、撤回与小时等效声级重算。

所有写函数都要求调用方已持有写事务（BEGIN IMMEDIATE），
原始数据变更与小时结果重算在同一事务内提交或一起回滚。
"""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

from .schemas import CalibrationIn, ReadingIn, Role
from .time_utils import SECONDS_PER_HOUR, format_epoch_second, hours_touching

# dB 转声能时的参考基准，公共因子在平均后会抵消。
_ENERGY_REF_FACTOR = 0.1

# 有限 REAL 的绝对值上限远在 1e307 之上（双精度约 1.8e308），
# 用该界做 SQL 预筛找出 ±Inf，再用 math.isfinite 复核，避免误伤有限的极大值。
_FINITE_ABS_BOUND = 1.0e307


class ServiceError(Exception):
    """服务层业务错误基类。"""


class NotFoundError(ServiceError):
    """编号不存在。"""


class ConflictError(ServiceError):
    """编号冲突 / 区间重叠 / 状态不允许变更。"""


class ValidationError(ServiceError):
    """入参语义不合法。"""


@dataclass(frozen=True)
class Reading:
    number: str
    point: str
    role: str
    start_ts: int
    end_ts: int
    db_value: float


@dataclass(frozen=True)
class Calibration:
    number: str
    point: str
    role: str
    start_ts: int
    end_ts: int
    offset_db: float
    withdrawn: bool


# --------------------------------------------------------------------------- #
# 行序列化
# --------------------------------------------------------------------------- #
def reading_out(row: sqlite3.Row) -> dict:
    return {
        "number": row["number"],
        "point": row["point"],
        "role": Role(row["role"]),
        "start": format_epoch_second(row["start_ts"]),
        "end": format_epoch_second(row["end_ts"]),
        "db": row["db_value"],
        "quarantined": bool(row["quarantined"]),
    }


def calibration_out(row: sqlite3.Row) -> dict:
    return {
        "number": row["number"],
        "point": row["point"],
        "role": Role(row["role"]),
        "start": format_epoch_second(row["start_ts"]),
        "end": format_epoch_second(row["end_ts"]),
        "offset": row["offset_db"],
        "withdrawn": bool(row["withdrawn"]),
        "quarantined": bool(row["quarantined"]),
    }


# --------------------------------------------------------------------------- #
# 通用校验
# --------------------------------------------------------------------------- #
def _validate_interval(start_ts: int, end_ts: int) -> None:
    if start_ts >= end_ts:
        raise ValidationError("区间必须满足 start < end，且为左闭右开的整秒区间")


def _validate_finite(value: float, label: str) -> None:
    """分贝值 / 校准偏移必须是有限数值，NaN 与 ±Inf 一律拒收。"""
    if not math.isfinite(value):
        raise ValidationError(f"{label}必须是有限数值，不能为 NaN 或 ±Inf")


def _find_reading_overlap(
    conn: sqlite3.Connection,
    point: str,
    role: Role,
    start_ts: int,
    end_ts: int,
    exclude_number: str | None = None,
) -> str | None:
    """返回与给定区间重叠的已有读数编号，没有则返回 None。

    已隔离的读数不参与重叠判定，其时段可用新编号补录。
    """
    sql = (
        "SELECT number FROM readings "
        "WHERE point = ? AND role = ? AND quarantined = 0 "
        "AND start_ts < ? AND end_ts > ?"
    )
    params: list[object] = [point, role.value, end_ts, start_ts]
    if exclude_number is not None:
        sql += " AND number != ?"
        params.append(exclude_number)
    row = conn.execute(sql, params).fetchone()
    return row["number"] if row else None


def _find_calibration_overlap(
    conn: sqlite3.Connection,
    point: str,
    role: Role,
    start_ts: int,
    end_ts: int,
    exclude_number: str | None = None,
) -> str | None:
    """返回与给定区间重叠的有效（未撤回、未隔离）校准单编号。"""
    sql = (
        "SELECT number FROM calibrations "
        "WHERE point = ? AND role = ? AND withdrawn = 0 AND quarantined = 0 "
        "AND start_ts < ? AND end_ts > ?"
    )
    params: list[object] = [point, role.value, end_ts, start_ts]
    if exclude_number is not None:
        sql += " AND number != ?"
        params.append(exclude_number)
    row = conn.execute(sql, params).fetchone()
    return row["number"] if row else None


# --------------------------------------------------------------------------- #
# 读数登记
# --------------------------------------------------------------------------- #
def register_reading(conn: sqlite3.Connection, number: str, payload: ReadingIn) -> tuple[dict, bool]:
    """登记读数。

    返回 (响应体, 是否新建)。同号原样重传视为幂等成功；同号内容不同则冲突。
    读数可补交（任意历史时刻），没有时效限制。
    """
    _validate_interval(payload.start, payload.end)
    _validate_finite(payload.db, "读数分贝值")
    existing = conn.execute(
        "SELECT * FROM readings WHERE number = ?", (number,)
    ).fetchone()
    now = int(time.time())

    if existing is not None:
        if existing["quarantined"]:
            # 隔离编号不得再用；其时段可按现有规则用新编号补录。
            raise ConflictError(
                f"读数编号 {number} 已因非有限值被隔离，不能复用，请换用新编号"
            )
        same = (
            existing["point"] == payload.point
            and existing["role"] == payload.role.value
            and existing["start_ts"] == payload.start
            and existing["end_ts"] == payload.end
            and float(existing["db_value"]) == float(payload.db)
        )
        if not same:
            raise ConflictError(
                f"读数编号 {number} 已存在且内容不同；同号只接受原样重传"
            )
        return reading_out(existing), False

    overlap = _find_reading_overlap(
        conn, payload.point, payload.role, payload.start, payload.end
    )
    if overlap is not None:
        raise ConflictError(
            f"与同仪器读数 {overlap} 的区间重叠，同一仪器读数区间不得重叠"
        )

    conn.execute(
        "INSERT INTO readings (number, point, role, start_ts, end_ts, db_value, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            number,
            payload.point,
            payload.role.value,
            payload.start,
            payload.end,
            payload.db,
            now,
        ),
    )
    row = conn.execute("SELECT * FROM readings WHERE number = ?", (number,)).fetchone()
    recompute_hours(
        conn, payload.point, hours_touching(payload.start, payload.end)
    )
    return reading_out(row), True


# --------------------------------------------------------------------------- #
# 校准单登记 / 修改 / 撤回
# --------------------------------------------------------------------------- #
def upsert_calibration(
    conn: sqlite3.Connection, number: str, payload: CalibrationIn
) -> tuple[dict, bool]:
    """登记新校准单，或修改已有校准单的起止时刻与偏移。

    已撤回的编号不可复活，需使用新编号。
    """
    _validate_interval(payload.start, payload.end)
    _validate_finite(payload.offset, "校准偏移")
    existing = conn.execute(
        "SELECT * FROM calibrations WHERE number = ?", (number,)
    ).fetchone()
    now = int(time.time())

    if existing is not None:
        if existing["quarantined"]:
            raise ConflictError(
                f"校准单 {number} 已因非有限值被隔离，不能修改或复活，请使用新编号"
            )
        if existing["withdrawn"]:
            raise ConflictError(f"校准单 {number} 已撤回，不能修改或复活，请使用新编号")
        if payload.point != existing["point"] or payload.role.value != existing["role"]:
            raise ConflictError(
                f"校准单 {number} 的测点与主备角色不可修改，只能修改起止时刻和偏移"
            )

        overlap = _find_calibration_overlap(
            conn,
            payload.point,
            payload.role,
            payload.start,
            payload.end,
            exclude_number=number,
        )
        if overlap is not None:
            raise ConflictError(
                f"与有效校准单 {overlap} 的区间重叠，同一仪器校准区间不得重叠"
            )

        conn.execute(
            "UPDATE calibrations SET point = ?, role = ?, start_ts = ?, end_ts = ?, "
            "offset_db = ?, updated_at = ? WHERE number = ?",
            (
                payload.point,
                payload.role.value,
                payload.start,
                payload.end,
                payload.offset,
                now,
                number,
            ),
        )
        affected = set(hours_touching(existing["start_ts"], existing["end_ts"]))
        affected.update(hours_touching(payload.start, payload.end))
        recompute_hours(conn, payload.point, sorted(affected))
        row = conn.execute(
            "SELECT * FROM calibrations WHERE number = ?", (number,)
        ).fetchone()
        return calibration_out(row), False

    overlap = _find_calibration_overlap(
        conn, payload.point, payload.role, payload.start, payload.end
    )
    if overlap is not None:
        raise ConflictError(
            f"与有效校准单 {overlap} 的区间重叠，同一仪器校准区间不得重叠"
        )

    conn.execute(
        "INSERT INTO calibrations (number, point, role, start_ts, end_ts, offset_db, "
        "withdrawn, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (
            number,
            payload.point,
            payload.role.value,
            payload.start,
            payload.end,
            payload.offset,
            now,
            now,
        ),
    )
    row = conn.execute(
        "SELECT * FROM calibrations WHERE number = ?", (number,)
    ).fetchone()
    recompute_hours(conn, payload.point, hours_touching(payload.start, payload.end))
    return calibration_out(row), True


def withdraw_calibration(conn: sqlite3.Connection, number: str) -> dict:
    """撤回校准单（软删除）。撤回后其偏移不再参与任何小时计算。"""
    existing = conn.execute(
        "SELECT * FROM calibrations WHERE number = ?", (number,)
    ).fetchone()
    if existing is None:
        raise NotFoundError(f"校准单 {number} 不存在")
    if existing["withdrawn"]:
        # 幂等：重复撤回直接返回当前状态。
        return calibration_out(existing)

    now = int(time.time())
    conn.execute(
        "UPDATE calibrations SET withdrawn = 1, updated_at = ? WHERE number = ?",
        (now, number),
    )
    row = conn.execute(
        "SELECT * FROM calibrations WHERE number = ?", (number,)
    ).fetchone()
    recompute_hours(
        conn, existing["point"], hours_touching(existing["start_ts"], existing["end_ts"])
    )
    return calibration_out(row)


# --------------------------------------------------------------------------- #
# 小时等效声级计算
# --------------------------------------------------------------------------- #
def _active_calibrations(
    conn: sqlite3.Connection, point: str, role: str
) -> list[Calibration]:
    rows = conn.execute(
        "SELECT * FROM calibrations WHERE point = ? AND role = ? "
        "AND withdrawn = 0 AND quarantined = 0 ORDER BY start_ts",
        (point, role),
    ).fetchall()
    return [
        Calibration(
            number=r["number"],
            point=r["point"],
            role=r["role"],
            start_ts=r["start_ts"],
            end_ts=r["end_ts"],
            offset_db=r["offset_db"],
            withdrawn=False,
        )
        for r in rows
    ]


def _adjust_reading(
    reading: Reading, calibrations: list[Calibration]
) -> list[tuple[int, int, float]]:
    """把一条读数按校准区间切成若干 (起点, 终点, 叠加偏移后的分贝) 片段。

    校准单同仪器互不重叠，因此任一秒最多命中一个偏移；未命中校准的片段偏移为 0。
    """
    segments: list[tuple[int, int, float]] = []
    cursor = reading.start_ts
    for cal in calibrations:
        if cal.end_ts <= cursor:
            continue
        if cal.start_ts >= reading.end_ts:
            break
        overlap_start = max(cursor, cal.start_ts)
        overlap_end = min(reading.end_ts, cal.end_ts)
        if overlap_start < overlap_end:
            if cursor < overlap_start:
                segments.append((cursor, overlap_start, reading.db_value))
            segments.append(
                (overlap_start, overlap_end, reading.db_value + cal.offset_db)
            )
            cursor = overlap_end
    if cursor < reading.end_ts:
        segments.append((cursor, reading.end_ts, reading.db_value))
    return segments


def _compute_hour(
    conn: sqlite3.Connection, point: str, hour_start: int
) -> dict | None:
    """计算单个测点单个 UTC 小时的结果；无任何覆盖返回 None。"""
    hour_end = hour_start + SECONDS_PER_HOUR
    rows = conn.execute(
        "SELECT * FROM readings WHERE point = ? AND quarantined = 0 "
        "AND start_ts < ? AND end_ts > ? ORDER BY start_ts, role",
        (point, hour_end, hour_start),
    ).fetchall()
    if not rows:
        return None

    calibrations = {
        Role.PRIMARY.value: _active_calibrations(conn, point, Role.PRIMARY.value),
        Role.BACKUP.value: _active_calibrations(conn, point, Role.BACKUP.value),
    }

    # chosen_energy[秒在小时内的偏移] = 该秒计入的声能；-1 表示尚未覆盖。
    chosen_energy: list[float] = [-1.0] * SECONDS_PER_HOUR
    chosen_role: list[int] = [0] * SECONDS_PER_HOUR  # 1=primary, 2=backup

    for row in rows:
        reading = Reading(
            number=row["number"],
            point=row["point"],
            role=row["role"],
            start_ts=row["start_ts"],
            end_ts=row["end_ts"],
            db_value=row["db_value"],
        )
        role_rank = 1 if reading.role == Role.PRIMARY.value else 2
        for seg_start, seg_end, adjusted_db in _adjust_reading(
            reading, calibrations[reading.role]
        ):
            lo = max(seg_start, hour_start)
            hi = min(seg_end, hour_end)
            energy = 10.0 ** (_ENERGY_REF_FACTOR * adjusted_db)
            for sec in range(lo - hour_start, hi - hour_start):
                # 主表优先：主表已覆盖的秒不再被备表替换；备表只补缺口。
                if chosen_energy[sec] < 0:
                    chosen_energy[sec] = energy
                    chosen_role[sec] = role_rank
                elif chosen_role[sec] > role_rank:
                    chosen_energy[sec] = energy
                    chosen_role[sec] = role_rank

    covered = [e for e in chosen_energy if e >= 0.0]
    if not covered:
        return None

    primary_seconds = sum(1 for r in chosen_role if r == 1)
    backup_seconds = sum(1 for r in chosen_role if r == 2)
    mean_energy = sum(covered) / len(covered)
    leq_db = 10.0 * math.log10(mean_energy)

    return {
        "point": point,
        "hour_start": hour_start,
        "leq_db": leq_db,
        "covered_seconds": len(covered),
        "primary_seconds": primary_seconds,
        "backup_seconds": backup_seconds,
    }


def recompute_hours(
    conn: sqlite3.Connection, point: str, hour_starts: list[int] | range
) -> None:
    """重算指定测点受影响的小时，并以新结果替换旧结果（无覆盖则删除旧行）。"""
    now = int(time.time())
    for hour_start in hour_starts:
        conn.execute(
            "DELETE FROM hourly_results WHERE point = ? AND hour_start = ?",
            (point, hour_start),
        )
        result = _compute_hour(conn, point, hour_start)
        if result is not None:
            conn.execute(
                "INSERT INTO hourly_results (point, hour_start, leq_db, covered_seconds, "
                "primary_seconds, backup_seconds, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result["point"],
                    result["hour_start"],
                    result["leq_db"],
                    result["covered_seconds"],
                    result["primary_seconds"],
                    result["backup_seconds"],
                    now,
                ),
            )


# --------------------------------------------------------------------------- #
# 启动隔离：清理旧库残存的非有限单据
# --------------------------------------------------------------------------- #
def quarantine_non_finite(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """把库中残存的非有限读数 / 校准单标记为隔离，并只重算受影响的小时。

    仅供启动流程在写事务内调用：隔离标记与受影响小时的重算在同一事务里
    提交或一起回滚；失败时保留原有记录与结果。本函数幂等，下次启动可重试。
    返回被隔离的编号清单（按单据类型分组）。
    """
    quarantined_readings: list[str] = []
    quarantined_calibrations: list[str] = []
    affected: dict[str, set[int]] = {}

    # SQLite 把 NaN 存为 NULL，±Inf 超出一切有限 REAL；预筛后再用
    # math.isfinite 复核，有限的极大值不误伤。
    reading_rows = conn.execute(
        "SELECT * FROM readings WHERE quarantined = 0 "
        "AND (db_value IS NULL OR db_value > ? OR db_value < ?)",
        (_FINITE_ABS_BOUND, -_FINITE_ABS_BOUND),
    ).fetchall()
    for row in reading_rows:
        value = row["db_value"]
        if value is not None and math.isfinite(value):
            continue
        conn.execute(
            "UPDATE readings SET quarantined = 1 WHERE number = ?", (row["number"],)
        )
        affected.setdefault(row["point"], set()).update(
            hours_touching(row["start_ts"], row["end_ts"])
        )
        quarantined_readings.append(row["number"])

    calibration_rows = conn.execute(
        "SELECT * FROM calibrations WHERE quarantined = 0 "
        "AND (offset_db IS NULL OR offset_db > ? OR offset_db < ?)",
        (_FINITE_ABS_BOUND, -_FINITE_ABS_BOUND),
    ).fetchall()
    for row in calibration_rows:
        value = row["offset_db"]
        if value is not None and math.isfinite(value):
            continue
        conn.execute(
            "UPDATE calibrations SET quarantined = 1 WHERE number = ?", (row["number"],)
        )
        if not row["withdrawn"]:
            # 已撤回的单据本就不参与核算，无需重算其窗口。
            affected.setdefault(row["point"], set()).update(
                hours_touching(row["start_ts"], row["end_ts"])
            )
        quarantined_calibrations.append(row["number"])

    for point, hours in affected.items():
        recompute_hours(conn, point, sorted(hours))

    return {"readings": quarantined_readings, "calibrations": quarantined_calibrations}


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #
def get_hour_result(
    conn: sqlite3.Connection, point: str, hour_start: int
) -> dict | None:
    row = conn.execute(
        "SELECT * FROM hourly_results WHERE point = ? AND hour_start = ?",
        (point, hour_start),
    ).fetchone()
    if row is None:
        return None
    return {
        "point": row["point"],
        "hour_start": row["hour_start"],
        "leq_db": row["leq_db"],
        "covered_seconds": row["covered_seconds"],
        "primary_seconds": row["primary_seconds"],
        "backup_seconds": row["backup_seconds"],
    }


def list_hour_results(
    conn: sqlite3.Connection, point: str, start_hour: int, end_hour: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM hourly_results WHERE point = ? AND hour_start >= ? AND hour_start < ? "
        "ORDER BY hour_start",
        (point, start_hour, end_hour),
    ).fetchall()
    return [
        {
            "point": r["point"],
            "hour_start": r["hour_start"],
            "leq_db": r["leq_db"],
            "covered_seconds": r["covered_seconds"],
            "primary_seconds": r["primary_seconds"],
            "backup_seconds": r["backup_seconds"],
        }
        for r in rows
    ]


def list_readings(conn: sqlite3.Connection, point: str | None) -> list[dict]:
    if point is None:
        rows = conn.execute("SELECT * FROM readings ORDER BY point, start_ts").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM readings WHERE point = ? ORDER BY start_ts", (point,)
        ).fetchall()
    return [reading_out(r) for r in rows]


def list_calibrations(conn: sqlite3.Connection, point: str | None) -> list[dict]:
    if point is None:
        rows = conn.execute(
            "SELECT * FROM calibrations ORDER BY point, start_ts"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM calibrations WHERE point = ? ORDER BY start_ts", (point,)
        ).fetchall()
    return [calibration_out(r) for r in rows]
