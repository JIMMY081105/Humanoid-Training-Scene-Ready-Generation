from __future__ import annotations

import asyncio
import base64
import json

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.utils.openai import encode_image_to_base64


def test_todo_complete_advances_the_same_fifo_task_as_get_next() -> None:
    tools = WorkflowTools()
    tools._designer_todo_manager_impl("add", "plan the room")
    tools._designer_todo_manager_impl("add", "generate assets")
    tools._designer_todo_manager_impl("add", "validate final scene")

    next_task = json.loads(tools._designer_todo_manager_impl("get_next"))
    completed = json.loads(tools._designer_todo_manager_impl("complete"))
    following = json.loads(tools._designer_todo_manager_impl("get_next"))

    assert next_task["task"]["id"] == "todo_0"
    assert completed["task"]["id"] == "todo_0"
    assert completed["task"]["task"] == "plan the room"
    assert following["task"]["id"] == "todo_1"


@dataclass
class _RoomGeometry:
    length: float = 9.0
    width: float = 7.5


class _Scene:
    def __init__(self) -> None:
        self.room_geometry = _RoomGeometry()
        self.version = "empty"

    def content_hash(self) -> str:
        return self.version


class _RenderingManager:
    def __init__(self, render_dir: Path) -> None:
        self.render_dir = render_dir
        self.calls = 0

    def render_scene(self, *args: object, **kwargs: object) -> Path:
        self.calls += 1
        return self.render_dir


class _ResetTracker:
    def __init__(self) -> None:
        self.calls = 0

    def reset_observation_guard(self) -> None:
        self.calls += 1


def test_observe_scene_blocks_repeated_unchanged_render_and_rearms_after_change(
    tmp_path: Path,
) -> None:
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    Image.new("RGB", (2, 2), color=(240, 230, 210)).save(render_dir / "0_top.png")
    scene = _Scene()
    rendering = _RenderingManager(render_dir)
    tools = VisionTools(
        scene=scene,  # type: ignore[arg-type]
        rendering_manager=rendering,  # type: ignore[arg-type]
        cfg=None,  # type: ignore[arg-type]
        blender_server=object(),  # type: ignore[arg-type]
    )

    first = tools._observe_scene_impl()
    repeated = tools._observe_scene_impl()

    assert rendering.calls == 1
    assert len(first) == 2
    assert len(repeated) == 1
    assert "has not changed" in repeated[0].text
    assert "Do not call observe_scene again" in repeated[0].text

    tools.reset_observation_guard()
    new_turn_same_scene = tools._observe_scene_impl()

    assert rendering.calls == 2
    assert len(new_turn_same_scene) == 2

    scene.version = "one-object-added"
    after_change = tools._observe_scene_impl()

    assert rendering.calls == 3
    assert len(after_change) == 2
    assert tools._unchanged_observation_blocks == 0


def test_each_new_furniture_agent_run_resets_the_correct_observation_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(StatefulFurnitureAgent)
    designer = _ResetTracker()
    critic = _ResetTracker()
    agent._designer_vision_tools = designer  # type: ignore[assignment]
    agent._critic_vision_tools = critic  # type: ignore[assignment]

    async def fake_initial(self: BaseStatefulAgent) -> str:
        return "initial"

    async def fake_change(self: BaseStatefulAgent, instruction: str) -> str:
        return instruction

    async def fake_critique(
        self: BaseStatefulAgent, update_checkpoint: bool = True
    ) -> str:
        return f"critique:{update_checkpoint}"

    monkeypatch.setattr(BaseStatefulAgent, "_request_initial_design_impl", fake_initial)
    monkeypatch.setattr(BaseStatefulAgent, "_request_design_change_impl", fake_change)
    monkeypatch.setattr(BaseStatefulAgent, "_request_critique_impl", fake_critique)

    assert asyncio.run(agent._request_initial_design_impl()) == "initial"
    assert asyncio.run(agent._request_design_change_impl("change")) == "change"
    assert asyncio.run(agent._request_critique_impl(update_checkpoint=False)) == (
        "critique:False"
    )
    assert designer.calls == 2
    assert critic.calls == 1


def test_room_worker_forces_responses_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents

    from scripts import run_single_room_worker as worker

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        agents,
        "set_default_openai_api",
        lambda value: calls.append(("api", value)),
    )
    monkeypatch.setattr(
        agents,
        "set_tracing_disabled",
        lambda value: calls.append(("tracing_disabled", value)),
    )

    assert worker._configure_agents_transport() == "responses"
    assert calls == [("api", "responses"), ("tracing_disabled", True)]


def test_responses_transport_preserves_images_in_function_tool_output() -> None:
    from agents import ModelSettings
    from agents.models.openai_responses import OpenAIResponsesModel
    from openai import AsyncOpenAI

    model = OpenAIResponsesModel(
        "gpt-5.2",
        AsyncOpenAI(api_key="test-only"),
        model_is_explicit=True,
    )
    items = [
        {
            "type": "function_call",
            "call_id": "call_observe",
            "name": "observe_scene",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_observe",
            "output": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                },
                {"type": "input_text", "text": "visual ready"},
            ],
        },
    ]

    request = model._build_response_create_kwargs(
        system_instructions="system",
        input=items,  # type: ignore[arg-type]
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
    )

    output = request["input"][1]["output"]
    assert output[0] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,AAAA",
    }
    assert output[1] == {"type": "input_text", "text": "visual ready"}


def test_png_data_url_encoder_emits_bytes_matching_its_mime_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (3, 2), color=(23, 117, 201)).save(source, format="JPEG")

    encoded = encode_image_to_base64(source)
    raw = base64.b64decode(encoded, validate=True)

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(raw)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.mode == "RGB"
        assert decoded.size == (3, 2)
