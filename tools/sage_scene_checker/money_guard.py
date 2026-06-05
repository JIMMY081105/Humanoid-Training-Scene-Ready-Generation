"""No-paid-API guard for SceneSmith validation tools."""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Mapping


PAID_API_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "NVIDIA_API_KEY",
    "ARK_API_KEY",
)


@dataclass
class BackendUsage:
    """Counters reported by the checker."""

    openai_api_calls: int = 0
    gemini_api_calls: int = 0
    anthropic_api_calls: int = 0
    external_paid_api_calls: int = 0
    codex_cli_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "openai_api_calls": self.openai_api_calls,
            "gemini_api_calls": self.gemini_api_calls,
            "anthropic_api_calls": self.anthropic_api_calls,
            "external_paid_api_calls": self.external_paid_api_calls,
            "codex_cli_calls": self.codex_cli_calls,
        }


@dataclass
class MoneyGuardResult:
    """Result of checking the local environment for paid API access."""

    allowed: bool
    no_paid_api: bool
    present_env_vars: list[str] = field(default_factory=list)
    usage: BackendUsage = field(default_factory=BackendUsage)

    def to_report_dict(self) -> dict[str, int]:
        return self.usage.to_dict()


class MoneyGuardError(RuntimeError):
    """Raised when no-paid-api mode detects a paid API credential."""

    def __init__(self, result: MoneyGuardResult) -> None:
        self.result = result
        names = ", ".join(result.present_env_vars)
        super().__init__(f"Paid API environment variables are present: {names}")


def paid_api_env_vars_present(environ: Mapping[str, str]) -> list[str]:
    """Return configured paid API credential names that are present."""

    return [name for name in PAID_API_ENV_VARS if environ.get(name)]


def check_money_guard(
    *,
    no_paid_api: bool,
    environ: Mapping[str, str] | None = None,
    usage: BackendUsage | None = None,
) -> MoneyGuardResult:
    """Check whether paid API credentials are present.

    The checker itself never calls paid APIs. In strict mode, the presence of a
    paid API key is treated as a hard failure to prevent hidden fallback paths.
    """

    env = environ if environ is not None else os.environ
    present = paid_api_env_vars_present(env)
    result = MoneyGuardResult(
        allowed=not (no_paid_api and present),
        no_paid_api=no_paid_api,
        present_env_vars=present,
        usage=usage or BackendUsage(),
    )
    if no_paid_api and present:
        raise MoneyGuardError(result)
    return result

