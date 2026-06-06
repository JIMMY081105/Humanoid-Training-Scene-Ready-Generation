from __future__ import annotations

from codex_benchmark.report import REPORT_TITLE, SUMMARY_TABLE_HEADERS, _summary_table


def test_summary_table_uses_declared_headers() -> None:
    table = _summary_table(
        [
            {
                "module": "structured",
                "scenario": "schema_json",
                "calls": 1,
                "success_rate": 1.0,
            }
        ]
    )

    assert table.splitlines()[0] == "| " + " | ".join(SUMMARY_TABLE_HEADERS) + " |"
    assert "| structured | schema_json | 1 | 1.0000 |" in table


def test_summary_table_handles_empty_rows() -> None:
    assert _summary_table([]) == "_No data._"


def test_report_title_is_plain_text() -> None:
    assert REPORT_TITLE == "Codex CLI Autonomous Pipeline Benchmark"
