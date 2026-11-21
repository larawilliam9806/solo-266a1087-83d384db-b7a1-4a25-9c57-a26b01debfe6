"""非有限值防护：写入拒收、启动隔离、编号禁用与受影响小时重算。"""
from __future__ import annotations

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

HOUR = "2026-11-21T08:00:00Z"
HALF = "2026-11-21T08:30:00Z"
NEXT = "2026-11-21T09:00:00Z"


def _put_raw(client, url: str, body: str):
    """httpx 的 json= 不允许非有限值，这里直接发原始 JSON 体。"""
    return client.put(url, content=body, headers={"Content-Type": "application/json"})


def _reading_json(point: str, role: str, start: str, end: str, db_literal: str) -> str:
    return (
        f'{{"point":"{point}","role":"{role}",'
        f'"start":"{start}","end":"{end}","db":{db_literal}}}'
    )


def _cal_json(point: str, role: str, start: str, end: str, offset_literal: str) -> str:
    return (
        f'{{"point":"{point}","role":"{role}",'
        f'"start":"{start}","end":"{end}","offset":{offset_literal}}}'
    )


@pytest.fixture()
def db_env(monkeypatch: pytest.MonkeyPatch):
    """独立临时库；返回 (db_path, seed, start)，可先写脏数据再启动应用。"""
    tmpdir = tempfile.mkdtemp(prefix="noise-quar-")
    db_path = os.path.join(tmpdir, "test.db")
    monkeypatch.setenv("NOISE_DB_PATH", db_path)

    from app import database
    from app.time_utils import parse_epoch_second

    database.init_db(db_path)

    def seed_reading(number, point, role, start, end, db):
        conn = database.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO readings (number, point, role, start_ts, end_ts, db_value, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (number, point, role, parse_epoch_second(start),
                 parse_epoch_second(end), db, int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    def seed_calibration(number, point, role, start, end, offset, withdrawn=0):
        conn = database.connect(db_path)
        try:
            now = int(time.time())
            conn.execute(
                "INSERT INTO calibrations (number, point, role, start_ts, end_ts, "
                "offset_db, withdrawn, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (number, point, role, parse_epoch_second(start),
                 parse_epoch_second(end), offset, withdrawn, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def seed_hour(point, hour, leq_db, covered=3600, primary=3600, backup=0):
        conn = database.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO hourly_results (point, hour_start, leq_db, "
                "covered_seconds, primary_seconds, backup_seconds, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (point, parse_epoch_second(hour), leq_db, covered, primary, backup,
                 int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    def start():
        from app.main import app

        return TestClient(app)

    seed = {"reading": seed_reading, "calibration": seed_calibration, "hour": seed_hour}
    return db_path, seed, start


# --------------------------------------------------------------------------- #
# 写入拒收：登记 / 修改时遇到非有限值报错，单据与小时结果保持原样
# --------------------------------------------------------------------------- #
def test_register_reading_rejects_non_finite_db(client):
    for literal in ("-Infinity", "Infinity", "NaN"):
        resp = _put_raw(client, f"/readings/{literal}", _reading_json(
            "P1", "primary", HOUR, NEXT, literal))
        assert resp.status_code == 400, (literal, resp.text)
        assert "有限" in resp.json()["detail"]

    # 单据与小时结果都保持原样：没有任何写入痕迹。
    assert client.get("/readings").json() == []
    assert client.get(f"/points/P1/hours/{HOUR}").status_code == 404


def test_calibration_rejects_non_finite_offset(client):
    # 新增校准单带非有限偏移 -> 400
    for literal in ("-Infinity", "Infinity", "NaN"):
        resp = _put_raw(client, f"/calibrations/{literal}", _cal_json(
            "P1", "primary", HOUR, NEXT, literal))
        assert resp.status_code == 400, (literal, resp.text)
    assert client.get("/calibrations").json() == []

    # 先落一条合法校准单与读数，再尝试把偏移改成非有限值 -> 400，原单据与小时结果不变。
    client.put("/readings/R1", json={
        "point": "P1", "role": "primary", "start": HOUR, "end": NEXT, "db": 60.0,
    })
    client.put("/calibrations/C1", json={
        "point": "P1", "role": "primary", "start": HOUR, "end": HALF, "offset": 3.0,
    })
    before = client.get(f"/points/P1/hours/{HOUR}").json()

    resp = _put_raw(client, "/calibrations/C1", _cal_json(
        "P1", "primary", HOUR, HALF, "-Infinity"))
    assert resp.status_code == 400

    cal = client.get("/calibrations").json()
    assert len(cal) == 1 and cal[0]["offset"] == 3.0 and cal[0]["quarantined"] is False
    assert client.get(f"/points/P1/hours/{HOUR}").json() == before


def test_failed_validation_keeps_records_and_hours(client):
    client.put("/readings/R1", json={
        "point": "P1", "role": "primary", "start": HOUR, "end": NEXT, "db": 60.0,
    })
    # 非有限值校验优先于重叠检查：报 400 而非 409，且 R1 与小时结果不受影响。
    resp = _put_raw(client, "/readings/R2", _reading_json(
        "P1", "primary", HOUR, NEXT, "-Infinity"))
    assert resp.status_code == 400

    rows = client.get("/readings").json()
    assert [r["number"] for r in rows] == ["R1"]
    result = client.get(f"/points/P1/hours/{HOUR}").json()
    assert abs(result["leq_db"] - 60.0) < 1e-9
    assert result["covered_seconds"] == 3600


def test_finite_edge_values_still_accepted(client):
    # 有限的极端值（如远低于听觉阈的负分贝）仍是合法读数。
    resp = client.put("/readings/R1", json={
        "point": "P1", "role": "primary", "start": HOUR, "end": NEXT, "db": -150.0,
    })
    assert resp.status_code == 200
    assert abs(client.get(f"/points/P1/hours/{HOUR}").json()["leq_db"] + 150.0) < 1e-9


# --------------------------------------------------------------------------- #
# 启动隔离：旧库残存的非有限单据被标记、跳过核算、只重算受影响小时
# --------------------------------------------------------------------------- #
def test_startup_quarantines_inf_reading_and_restores_backup(db_env):
    _db_path, seed, start = db_env
    # 旧库遗留：备表整小时 55 dB 有效，主表前半小时为 -inf（修复前入库的脏数据）。
    seed["reading"]("B1", "P1", "backup", HOUR, NEXT, 55.0)
    seed["reading"]("INF1", "P1", "primary", HOUR, HALF, float("-inf"))

    with start() as client:
        # 明细仍可查到原编号、异常值与隔离状态。
        rows = {r["number"]: r for r in client.get("/readings?point=P1").json()}
        assert rows["INF1"]["quarantined"] is True
        assert rows["INF1"]["db"] == float("-inf")
        assert rows["B1"]["quarantined"] is False

        # 核算跳过隔离读数：有效备表照常补缺，主表不再挤占。
        result = client.get(f"/points/P1/hours/{HOUR}").json()
        assert result["covered_seconds"] == 3600
        assert result["primary_seconds"] == 0
        assert result["backup_seconds"] == 3600
        assert abs(result["leq_db"] - 55.0) < 1e-9


def test_startup_quarantines_inf_calibration_and_backfill_works(db_env):
    _db_path, seed, start = db_env
    seed["reading"]("R1", "P1", "primary", HOUR, NEXT, 60.0)
    seed["calibration"]("C-OLD", "P1", "primary", HOUR, NEXT, float("-inf"))

    with start() as client:
        cal = client.get("/calibrations").json()
        assert len(cal) == 1
        assert cal[0]["quarantined"] is True
        assert cal[0]["offset"] == float("-inf")

        # 隔离偏移不再叠加，小时结果回到原始分贝。
        result = client.get(f"/points/P1/hours/{HOUR}").json()
        assert abs(result["leq_db"] - 60.0) < 1e-9

        # 正常补报不再被脏校准单拖垮。
        resp = client.put("/readings/R2", json={
            "point": "P1", "role": "backup", "start": HOUR, "end": NEXT, "db": 50.0,
        })
        assert resp.status_code == 200


def test_quarantined_numbers_cannot_be_reused(db_env):
    _db_path, seed, start = db_env
    seed["reading"]("INF1", "P1", "primary", HOUR, HALF, float("-inf"))
    seed["calibration"]("C-OLD", "P1", "primary", HOUR, HALF, float("-inf"))

    with start() as client:
        # 隔离编号不得再用（读数 409 / 校准单 409）。
        resp = client.put("/readings/INF1", json={
            "point": "P1", "role": "primary", "start": HOUR, "end": HALF, "db": 58.0,
        })
        assert resp.status_code == 409
        resp = client.put("/calibrations/C-OLD", json={
            "point": "P1", "role": "primary", "start": HOUR, "end": HALF, "offset": 2.0,
        })
        assert resp.status_code == 409

        # 其时段可按现有规则用新编号补录。
        resp = client.put("/readings/R-NEW", json={
            "point": "P1", "role": "primary", "start": HOUR, "end": HALF, "db": 58.0,
        })
        assert resp.status_code == 200
        resp = client.put("/calibrations/C-NEW", json={
            "point": "P1", "role": "primary", "start": HOUR, "end": HALF, "offset": 2.0,
        })
        assert resp.status_code == 200

        result = client.get(f"/points/P1/hours/{HOUR}").json()
        assert result["primary_seconds"] == 1800
        assert abs(result["leq_db"] - 60.0) < 1e-9


def test_quarantine_recomputes_only_affected_hours(db_env):
    _db_path, seed, start = db_env
    seed["reading"]("INF1", "P1", "primary", HOUR, HALF, float("-inf"))
    seed["reading"]("B1", "P1", "backup", HOUR, NEXT, 55.0)
    # 与隔离记录无关的小时结果（即使是异常值）不得被启动流程触碰。
    seed["hour"]("P2", HOUR, 999.0)

    with start() as client:
        affected = client.get(f"/points/P1/hours/{HOUR}").json()
        assert abs(affected["leq_db"] - 55.0) < 1e-9
        untouched = client.get(f"/points/P2/hours/{HOUR}").json()
        assert untouched["leq_db"] == 999.0


def test_quarantined_records_remain_listed(db_env):
    _db_path, seed, start = db_env
    seed["reading"]("INF1", "P1", "primary", HOUR, HALF, float("inf"))
    seed["calibration"]("C-OLD", "P1", "primary", HOUR, HALF, float("-inf"))

    with start() as client:
        readings = client.get("/readings").json()
        assert len(readings) == 1
        assert readings[0]["number"] == "INF1"
        assert readings[0]["db"] == float("inf")
        assert readings[0]["quarantined"] is True

        # include_withdrawn=false 也不隐藏隔离单据（隔离与撤回是两个状态）。
        cals = client.get("/calibrations?include_withdrawn=false").json()
        assert len(cals) == 1
        assert cals[0]["number"] == "C-OLD"
        assert cals[0]["offset"] == float("-inf")
        assert cals[0]["quarantined"] is True
        assert cals[0]["withdrawn"] is False


def test_startup_quarantine_is_idempotent(db_env):
    _db_path, seed, start = db_env
    seed["reading"]("INF1", "P1", "primary", HOUR, HALF, float("-inf"))
    seed["reading"]("B1", "P1", "backup", HOUR, NEXT, 55.0)

    with start() as client:
        first = client.get(f"/points/P1/hours/{HOUR}").json()
    # 第二次启动重复执行隔离：结果不变。
    with start() as client:
        rows = client.get("/readings").json()
        assert all(r["quarantined"] == (r["number"] == "INF1") for r in rows)
        assert client.get(f"/points/P1/hours/{HOUR}").json() == first


def test_quarantine_failure_rolls_back_and_next_start_retries(db_env, monkeypatch):
    db_path, seed, start = db_env
    seed["reading"]("INF1", "P1", "primary", HOUR, HALF, float("-inf"))
    seed["reading"]("B1", "P1", "backup", HOUR, NEXT, 55.0)
    seed["hour"]("P1", HOUR, 42.0)  # 启动前的旧小时结果

    from app import database, services

    original = services.recompute_hours

    def boom(conn, point, hours):
        raise RuntimeError("模拟启动隔离失败")

    monkeypatch.setattr(services, "recompute_hours", boom)
    with pytest.raises(RuntimeError):
        with start():
            pass

    # 失败时保留原有记录和结果：隔离标记未落地，旧小时结果原样。
    from app.time_utils import parse_epoch_second

    conn = database.connect(db_path)
    try:
        row = conn.execute(
            "SELECT quarantined FROM readings WHERE number = 'INF1'"
        ).fetchone()
        assert row["quarantined"] == 0
        kept = services.get_hour_result(conn, "P1", parse_epoch_second(HOUR))
        assert kept is not None and kept["leq_db"] == 42.0
    finally:
        conn.close()

    # 下次启动可重试：隔离与重算一并生效。
    monkeypatch.setattr(services, "recompute_hours", original)
    with start() as client:
        rows = {r["number"]: r for r in client.get("/readings").json()}
        assert rows["INF1"]["quarantined"] is True
        result = client.get(f"/points/P1/hours/{HOUR}").json()
        assert abs(result["leq_db"] - 55.0) < 1e-9
        assert result["backup_seconds"] == 3600


def test_old_schema_database_is_migrated_and_quarantined(db_env):
    """模拟修复前的旧库（无 quarantined 列）：启动时迁移并隔离。"""
    db_path, _seed, start = db_env
    from app import database
    from app.time_utils import parse_epoch_second

    # 把库回退成旧结构：删掉隔离列，重建一张旧版 readings。
    conn = database.connect(db_path)
    try:
        conn.execute("ALTER TABLE readings DROP COLUMN quarantined")
        conn.execute("ALTER TABLE calibrations DROP COLUMN quarantined")
        now = int(time.time())
        conn.execute(
            "INSERT INTO readings (number, point, role, start_ts, end_ts, db_value, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("INF-OLD", "P9", "primary", parse_epoch_second(HOUR),
             parse_epoch_second(HALF), float("-inf"), now),
        )
        conn.commit()
    finally:
        conn.close()

    with start() as client:
        rows = client.get("/readings?point=P9").json()
        assert len(rows) == 1
        assert rows[0]["number"] == "INF-OLD"
        assert rows[0]["quarantined"] is True
        assert rows[0]["db"] == float("-inf")
        # 旧库中没有其它覆盖，该小时无数据。
        assert client.get(f"/points/P9/hours/{HOUR}").status_code == 404
