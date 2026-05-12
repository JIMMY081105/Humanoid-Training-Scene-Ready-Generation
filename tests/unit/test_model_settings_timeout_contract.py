import importlib

from types import SimpleNamespace

import pytest


pytest.importorskip("agents")
pytest.importorskip("omegaconf")
pytest.importorskip("openai")

from agents import ModelSettings  # noqa: E402
from agents.models._trace import model_config_for_trace  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from openai import Timeout  # noqa: E402

try:
    # Match production import order. The Google GenAI import chain mutates the
    # shared timeout class's display module, which is why the original failure
    # named google.genai even though SceneSmith imported openai.Timeout.
    importlib.import_module("scenesmith.experiments.indoor_scene_generation")
    from scenesmith.agent_utils.base_stateful_agent import (  # noqa: E402
        BaseStatefulAgent,
        _TraceSafeModelSettings,
    )
except ModuleNotFoundError as exc:
    pytest.skip(
        f"SceneSmith runtime dependencies are unavailable: {exc.name}",
        allow_module_level=True,
    )


def _configured_model_settings() -> ModelSettings:
    owner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "api_timeout": {
                    "connect": 30.0,
                    "read": 1800.0,
                    "write": 1800.0,
                    "pool": 1800.0,
                },
                "openai": {
                    "service_tier": "default",
                    "reasoning_effort": {"planner": "low"},
                    "verbosity": {"planner": "medium"},
                },
            }
        )
    )
    settings = BaseStatefulAgent._get_model_settings(owner, "planner")
    assert settings is not None
    return settings


def test_provider_timeout_survives_request_settings_but_not_trace_encoding() -> None:
    settings = _configured_model_settings()

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


def test_resolving_model_settings_preserves_trace_safe_subclass() -> None:
    settings = _configured_model_settings()

    resolved = settings.resolve(ModelSettings(tool_choice="auto"))

    assert isinstance(resolved, _TraceSafeModelSettings)
    assert resolved.extra_args == settings.extra_args
    assert resolved.extra_args["timeout"] is settings.extra_args["timeout"]
    trace = resolved.to_traceable_dict()
    assert trace["tool_choice"] == "auto"
    assert "extra_args" not in trace


def test_absent_timeout_configuration_keeps_existing_none_behavior() -> None:
    owner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "openai": {
                    "service_tier": None,
                    "reasoning_effort": {},
                    "verbosity": {},
                }
            }
        )
    )

    assert BaseStatefulAgent._get_model_settings(owner) is None
