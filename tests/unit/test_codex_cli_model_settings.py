from __future__ import annotations

import importlib.util
import json
import sys
import types

from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CODEX_CLI_PATH = (
    REPOSITORY_ROOT / "scenesmith" / "agent_utils" / "codex_cli.py"
)
HANDOFF_OVERLAY_CODEX_CLI_PATH = (
    REPOSITORY_ROOT / "_remote_patch" / "scenesmith" / "agent_utils" / "codex_cli.py"
)
CODEX_CLI_PATH = (
    CANONICAL_CODEX_CLI_PATH
    if CANONICAL_CODEX_CLI_PATH.is_file()
    else HANDOFF_OVERLAY_CODEX_CLI_PATH
)


def _stub_module(monkeypatch, name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_codex_cli(monkeypatch):
    class StubValue:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class StubBase:
        pass

    _stub_module(monkeypatch, "agents", ModelSettings=StubBase)
    _stub_module(
        monkeypatch,
        "agents.agent_output",
        AgentOutputSchemaBase=StubBase,
    )
    _stub_module(monkeypatch, "agents.items", ModelResponse=StubValue)
    _stub_module(monkeypatch, "agents.models")
    _stub_module(
        monkeypatch,
        "agents.models.interface",
        Model=StubBase,
        ModelProvider=StubBase,
        ModelTracing=StubBase,
    )
    _stub_module(monkeypatch, "agents.usage", Usage=StubValue)
    _stub_module(monkeypatch, "openai")
    _stub_module(monkeypatch, "openai.types")
    _stub_module(
        monkeypatch,
        "openai.types.responses",
        ResponseFunctionToolCall=StubValue,
        ResponseOutputMessage=StubValue,
        ResponseOutputText=StubValue,
    )

    module_name = "_test_scenesmith_codex_cli"
    specification = importlib.util.spec_from_file_location(module_name, CODEX_CLI_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, module_name, module)
    specification.loader.exec_module(module)
    return module


def test_agents_prompt_omits_provider_timeout_without_pydantic_serialization(
    monkeypatch,
) -> None:
    module = _load_codex_cli(monkeypatch)

    provider_timeout_type = type(
        "Timeout",
        (),
        {"__module__": "google.genai._interactions"},
    )

    class Reasoning:
        model_fields = {"effort": object()}

        def __init__(self) -> None:
            self.effort = "high"

    class PydanticLikeModelSettings:
        model_fields = {
            "extra_args": object(),
            "reasoning": object(),
            "verbosity": object(),
        }

        def __init__(self) -> None:
            self.extra_args = {
                "timeout": provider_timeout_type(),
                "service_tier": "priority",
            }
            self.reasoning = Reasoning()
            self.verbosity = "medium"

        def model_dump(self, **_kwargs):
            raise AssertionError("Pydantic serialization must not be invoked")

    prompt = module._build_agents_prompt(
        system_instructions="system",
        input_payload=[{"role": "user", "content": "design a floor"}],
        tools=[],
        output_schema=None,
        model_name="gpt-5.2",
        model_settings=PydanticLikeModelSettings(),
    )

    payload = json.loads(prompt.split("SCENESMITH_MODEL_PAYLOAD:\n", 1)[1])
    assert payload["model_settings"] == {
        "extra_args": {"service_tier": "priority"},
        "reasoning": {"effort": "high"},
        "verbosity": "medium",
    }
    assert "google.genai" not in prompt


def test_agents_prompt_rejects_other_unknown_model_setting_objects(monkeypatch) -> None:
    module = _load_codex_cli(monkeypatch)

    class PydanticLikeModelSettings:
        model_fields = {"extra_args": object()}

        def __init__(self) -> None:
            self.extra_args = {"unsupported": object()}

    with pytest.raises(
        TypeError,
        match=r"Unsupported model-setting value at model_settings\.extra_args\.unsupported",
    ):
        module._safe_model_dump(PydanticLikeModelSettings())
