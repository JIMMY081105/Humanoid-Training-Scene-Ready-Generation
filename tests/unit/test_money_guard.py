import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "sage_scene_checker"
sys.path.insert(0, str(TOOLS_DIR))

from money_guard import BackendUsage, MoneyGuardError, check_money_guard  # noqa: E402


def test_money_guard_catches_paid_api_env_vars():
    with pytest.raises(MoneyGuardError) as exc_info:
        check_money_guard(
            no_paid_api=True,
            environ={"OPENAI_API_KEY": "sk-test", "PATH": "ignored"},
        )

    assert exc_info.value.result.allowed is False
    assert exc_info.value.result.present_env_vars == ["OPENAI_API_KEY"]


def test_money_guard_reports_backend_usage_without_paid_calls():
    usage = BackendUsage(codex_cli_calls=2)

    result = check_money_guard(no_paid_api=True, environ={}, usage=usage)

    assert result.allowed is True
    assert result.to_report_dict() == {
        "openai_api_calls": 0,
        "gemini_api_calls": 0,
        "anthropic_api_calls": 0,
        "external_paid_api_calls": 0,
        "codex_cli_calls": 2,
    }
