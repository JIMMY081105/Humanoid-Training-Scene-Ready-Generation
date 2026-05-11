from __future__ import annotations

import importlib

from types import SimpleNamespace

from omegaconf import OmegaConf

# Match the production import order that loads Google GenAI after the OpenAI
# timeout alias and mutates the shared timeout class's display module.
importlib.import_module("scenesmith.experiments.indoor_scene_generation")

from agents import ModelSettings  # noqa: E402
from agents.models._trace import model_config_for_trace  # noqa: E402
from openai import Timeout  # noqa: E402

from scenesmith.agent_utils.base_stateful_agent import (  # noqa: E402
    BaseStatefulAgent,
    _TraceSafeModelSettings,
)


def _subject(*, include_timeout: bool = True) -> SimpleNamespace:
    config = {
        "openai": {
            "service_tier": "default" if include_timeout else None,
            "reasoning_effort": {"planner": "low"},
            "verbosity": {"planner": "medium"},
        }
    }
    if include_timeout:
        config["api_timeout"] = {
            "connect": 30.0,
            "read": 1800.0,
            "write": 1800.0,
            "pool": 1800.0,
        }
    return SimpleNamespace(cfg=OmegaConf.create(config))


def test_provider_timeout_survives_request_settings_but_not_trace_encoding() -> None:
    settings = BaseStatefulAgent._get_model_settings(_subject(), "planner")

    assert isinstance(settings, _TraceSafeModelSettings)
    assert settings.extra_args is not None
    timeout = settings.extra_args["timeout"]
    assert isinstance(timeout, Timeout)
    assert timeout.connect == 30.0
    assert timeout.read == 1800.0
    assert timeout.write == 1800.0
    assert timeout.pool == 1800.0
    assert settings.extra_args["service_tier"] == "default"

    trace = model_config_for_trace(
        settings,
        base_url="https://api.openai.com/v1",
    )
    assert trace["verbosity"] == "medium"
    assert "extra_args" not in trace


def test_runner_settings_merge_preserves_trace_safe_timeout_identity() -> None:
    settings = BaseStatefulAgent._get_model_settings(_subject(), "planner")
    assert settings is not None
    assert settings.extra_args is not None

    resolved = settings.resolve(ModelSettings(tool_choice="auto"))

    assert isinstance(resolved, _TraceSafeModelSettings)
    assert resolved.extra_args is not None
    assert resolved.extra_args["timeout"] is settings.extra_args["timeout"]
    trace = resolved.to_traceable_dict()
    assert trace["tool_choice"] == "auto"
    assert "extra_args" not in trace


def test_absent_timeout_configuration_keeps_existing_none_behavior() -> None:
    assert BaseStatefulAgent._get_model_settings(_subject(include_timeout=False)) is None
