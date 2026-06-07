from __future__ import annotations

from pathlib import Path

from codex_benchmark.checkpoint import SCHEMA_VERSION, CallRecord, CheckpointStore, utc_now
from codex_benchmark.statistics import (
    _attempt_timeout_count,
    _classification_wrong,
    _metadata_bool,
    compute_summary,
)


def test_checkpoint_records_calls_and_attempt_timeouts(tmp_path: Path) -> None:
    store = CheckpointStore(str(tmp_path / "benchmark.sqlite3"))
    run_id = "unit_run"
    now = utc_now()
    try:
        store.start_run(
            run_id=run_id,
            config_hash="hash",
            config_json="{}",
            codex_version="unit",
            resume=False,
        )
        store.record_call(
            CallRecord(
                run_id=run_id,
                module="structured",
                scenario="schema_json",
                call_index=0,
                status="success",
                success=True,
                started_at=now,
                ended_at=now,
                latency_ms=100.0,
                retries=1,
                attempt_count=2,
                attempt_json_failures=1,
                attempt_schema_failures=1,
                invalid_json=False,
                schema_valid=True,
                metadata={
                    "attempts": [
                        {
                            "attempt": 0,
                            "success": False,
                            "error_kind": "timeout",
                            "invalid_json": True,
                            "schema_valid": False,
                        },
                        {
                            "attempt": 1,
                            "success": True,
                            "error_kind": None,
                            "invalid_json": False,
                            "schema_valid": True,
                        },
                    ]
                },
            )
        )

        assert store.call_exists(run_id, "structured", "schema_json", 0)
        summary = compute_summary(store, run_id)
    finally:
        store.close()

    scenario = summary["scenarios"][0]
    assert scenario["calls"] == 1
    assert scenario["success_rate"] == 1.0
    assert scenario["attempt_timeout_count"] == 1
    assert scenario["attempt_json_failure_rate"] == 0.5
    assert scenario["attempt_schema_failure_rate"] == 0.5


def test_attempt_timeout_count_ignores_invalid_metadata() -> None:
    assert _attempt_timeout_count({"metadata_json": "{not-json"}) == 0
    assert _attempt_timeout_count({"metadata_json": "[1, 2, 3]"}) == 0


def test_summary_metadata_flags_are_parsed_from_json() -> None:
    row = {
        "metadata_json": (
            '{"classification":{"classification_wrong":true},'
            '"insufficient_dataset_size":true}'
        )
    }

    assert _classification_wrong(row) == 1
    assert _metadata_bool(row, "insufficient_dataset_size") is True
    assert _classification_wrong({"metadata_json": '{"classification": []}'}) == 0


def test_checkpoint_migration_records_schema_version(tmp_path: Path) -> None:
    store = CheckpointStore(str(tmp_path / "benchmark.sqlite3"))
    try:
        row = store.conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        ).fetchone()
    finally:
        store.close()

    assert row["value"] == str(SCHEMA_VERSION)
