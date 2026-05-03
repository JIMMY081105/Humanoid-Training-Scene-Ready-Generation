"""Codex CLI adapters for SceneSmith model calls.

This module keeps Codex integration inside the production SceneSmith pipeline:
the OpenAI Agents SDK still owns sessions, tool execution, structured-output
validation, retries, and logging. Codex CLI is used only as the model backend
that emits either assistant text or function-call response items.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator

from agents import ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider, ModelTracing
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

console_logger = logging.getLogger(__name__)


_DATA_IMAGE_RE = re.compile(r"^data:image/([^;]+);base64,(.*)$", re.DOTALL)
_CODEX_USAGE_LIMIT_RE = re.compile(r"\busage limit\b", re.IGNORECASE)
_CODEX_TRY_AGAIN_AT_RE = re.compile(
    r"try again at\s+([0-9]{1,2})(?::([0-9]{2}))?\s*([AP]M)",
    re.IGNORECASE,
)


@dataclass
class CodexCLIConfig:
    """Configuration for non-interactive `codex exec` calls."""

    enabled: bool = False
    executable: str = "codex"
    model: str | None = None
    profile: str | None = None
    cwd: str = "."
    sandbox: str = "read-only"
    approval_policy: str = "never"
    timeout_seconds: int = 1800
    skip_git_repo_check: bool = True
    ignore_user_config: bool = False
    ignore_rules: bool = True
    ephemeral: bool = True
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: Any | None) -> "CodexCLIConfig":
        if cfg is None:
            return cls()

        def get(name: str, default: Any) -> Any:
            return getattr(cfg, name, default)

        return cls(
            enabled=bool(get("enabled", False)),
            executable=str(get("executable", "codex")),
            model=_none_if_empty(get("model", None)),
            profile=_none_if_empty(get("profile", None)),
            cwd=str(get("cwd", ".")),
            sandbox=str(get("sandbox", "read-only")),
            approval_policy=str(get("approval_policy", "never")),
            timeout_seconds=int(get("timeout_seconds", 1800)),
            skip_git_repo_check=bool(get("skip_git_repo_check", True)),
            ignore_user_config=bool(get("ignore_user_config", False)),
            ignore_rules=bool(get("ignore_rules", True)),
            ephemeral=bool(get("ephemeral", True)),
            extra_args=list(get("extra_args", []) or []),
            env=dict(get("env", {}) or {}),
        )


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value or None


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float_optional(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _codex_usage_limit_retry_delay(output: str) -> float | None:
    """Return a sleep duration for Codex CLI usage-limit errors, if present."""
    if not _CODEX_USAGE_LIMIT_RE.search(output):
        return None

    explicit_delay = _env_float_optional("SCENESMITH_CODEX_CLI_USAGE_LIMIT_SLEEP_SECONDS")
    if explicit_delay is not None:
        return explicit_delay

    padding_s = _env_float(
        "SCENESMITH_CODEX_CLI_USAGE_LIMIT_PADDING_SECONDS", default=120.0
    )
    max_sleep_s = _env_float(
        "SCENESMITH_CODEX_CLI_USAGE_LIMIT_MAX_SLEEP_SECONDS", default=7200.0
    )
    fallback_s = _env_float(
        "SCENESMITH_CODEX_CLI_USAGE_LIMIT_FALLBACK_SECONDS", default=1800.0
    )

    match = _CODEX_TRY_AGAIN_AT_RE.search(output)
    if not match:
        return min(fallback_s, max_sleep_s)

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = match.group(3).upper()
    if period == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    now = datetime.now()
    reset_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset_at <= now:
        reset_at += timedelta(days=1)

    return min((reset_at - now).total_seconds() + padding_s, max_sleep_s)


def _write_attempt_logs(
    *, call_dir: Path, attempt: int, stdout: str, stderr: str
) -> None:
    (call_dir / f"stdout_attempt_{attempt}.txt").write_text(
        stdout, encoding="utf-8"
    )
    (call_dir / f"stderr_attempt_{attempt}.txt").write_text(
        stderr, encoding="utf-8"
    )


def is_codex_enabled(cfg: Any) -> bool:
    """Return whether a config subtree enables Codex model calls."""
    return bool(getattr(getattr(cfg, "codex", None), "enabled", False))


class CodexCLIClient:
    """Thin subprocess wrapper around `codex exec`."""

    def __init__(self, config: CodexCLIConfig, artifact_dir: Path):
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def reserve_call_dir(self, call_name: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        call_id = f"{stamp}-{call_name}-{uuid.uuid4().hex[:8]}"
        call_dir = self.artifact_dir / _sanitize_filename(call_id)
        call_dir.mkdir(parents=True, exist_ok=False)
        return call_dir

    def run(
        self,
        prompt: str,
        *,
        output_schema: dict[str, Any] | None = None,
        image_paths: list[Path] | None = None,
        exec_config_args: list[str] | None = None,
        call_name: str = "call",
        call_dir: Path | None = None,
    ) -> str:
        call_dir = call_dir or self.reserve_call_dir(call_name)
        call_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = call_dir / "prompt.txt"
        stdout_path = call_dir / "stdout.txt"
        stderr_path = call_dir / "stderr.txt"
        last_message_path = call_dir / "last_message.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        schema_path = None
        if output_schema is not None:
            schema_path = call_dir / "output_schema.json"
            schema_path.write_text(json.dumps(output_schema, indent=2), encoding="utf-8")

        command = self._build_command(
            schema_path=schema_path,
            image_paths=[Path(p) for p in (image_paths or [])],
            exec_config_args=exec_config_args or [],
            last_message_path=last_message_path,
        )
        (call_dir / "command.json").write_text(
            json.dumps(command, indent=2), encoding="utf-8"
        )

        max_attempts = _env_int("SCENESMITH_CODEX_CLI_ATTEMPTS", default=1)
        retry_delay = _env_float("SCENESMITH_CODEX_CLI_RETRY_DELAY_SECONDS", 20.0)
        max_usage_limit_waits = _env_int(
            "SCENESMITH_CODEX_CLI_USAGE_LIMIT_WAITS", default=3
        )
        stdout = ""
        stderr = ""
        returncode: int | None = None
        usage_limit_waits = 0
        invocation = 0

        attempt = 1
        while attempt <= max_attempts:
            invocation += 1
            console_logger.info(
                "Calling Codex CLI model backend: %s (attempt %d/%d, invocation %d)",
                call_dir,
                attempt,
                max_attempts,
                invocation,
            )
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.config.timeout_seconds,
                    cwd=self.config.cwd,
                    env=self._environment(),
                )
            except subprocess.TimeoutExpired as exc:
                stdout = _safe_process_text(exc.stdout)
                stderr = _safe_process_text(exc.stderr)
                stdout_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                _write_attempt_logs(
                    call_dir=call_dir,
                    attempt=invocation,
                    stdout=stdout,
                    stderr=stderr,
                )
                if attempt < max_attempts:
                    console_logger.warning(
                        "Codex CLI timed out after %ss; retrying in %.1fs. "
                        "Artifacts: %s",
                        self.config.timeout_seconds,
                        retry_delay,
                        call_dir,
                    )
                    time.sleep(retry_delay)
                    attempt += 1
                    continue
                raise TimeoutError(
                    f"Codex CLI timed out after {self.config.timeout_seconds}s. "
                    f"Artifacts: {call_dir}"
                ) from exc

            returncode = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            _write_attempt_logs(
                call_dir=call_dir,
                attempt=invocation,
                stdout=stdout,
                stderr=stderr,
            )

            if completed.returncode == 0:
                break

            usage_limit_delay = _codex_usage_limit_retry_delay(f"{stdout}\n{stderr}")
            if (
                usage_limit_delay is not None
                and usage_limit_waits < max_usage_limit_waits
            ):
                usage_limit_waits += 1
                console_logger.warning(
                    "Codex CLI usage limit hit; waiting %.1fs before retrying "
                    "without consuming retry budget (%d/%d). Artifacts: %s. "
                    "stderr tail: %s",
                    usage_limit_delay,
                    usage_limit_waits,
                    max_usage_limit_waits,
                    call_dir,
                    stderr[-500:],
                )
                time.sleep(usage_limit_delay)
                continue

            if attempt < max_attempts:
                console_logger.warning(
                    "Codex CLI model call failed (returncode=%s); retrying in %.1fs. "
                    "Artifacts: %s. stderr tail: %s",
                    completed.returncode,
                    retry_delay,
                    call_dir,
                    stderr[-500:],
                )
                time.sleep(retry_delay)
                attempt += 1
                continue

            raise RuntimeError(
                "Codex CLI model call failed "
                f"(returncode={returncode}). Artifacts: {call_dir}. "
                f"stderr tail: {stderr[-1000:]}"
            )

        if last_message_path.exists():
            return last_message_path.read_text(encoding="utf-8").strip()

        # Fallback for older CLI behavior.
        return stdout.strip()

    def _build_command(
        self,
        *,
        schema_path: Path | None,
        image_paths: list[Path],
        exec_config_args: list[str],
        last_message_path: Path,
    ) -> list[str]:
        cfg = self.config
        command = [
            cfg.executable,
            "-C",
            cfg.cwd,
            "-a",
            cfg.approval_policy,
            "-s",
            cfg.sandbox,
        ]
        if cfg.model:
            command.extend(["--model", cfg.model])
        if cfg.profile:
            command.extend(["--profile", cfg.profile])
        command.extend(
            [
                "exec",
                "--color",
                "never",
                "--output-last-message",
                str(last_message_path),
            ]
        )
        if cfg.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if cfg.ephemeral:
            command.append("--ephemeral")
        if cfg.ignore_user_config:
            command.append("--ignore-user-config")
        if cfg.ignore_rules:
            command.append("--ignore-rules")
        for image_path in image_paths:
            command.extend(["--image", str(image_path)])
        command.extend(exec_config_args)
        if schema_path is not None:
            command.extend(["--output-schema", str(schema_path)])
        command.extend(cfg.extra_args)
        command.append("-")
        return command

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config.env)
        if env.get("SCENESMITH_CODEX_ISOLATE_HOME", "1") != "0":
            env["CODEX_HOME"] = str(self._prepare_isolated_codex_home(env))
        return env

    def _prepare_isolated_codex_home(self, env: dict[str, str]) -> Path:
        """Use a room-local Codex runtime home to avoid shared SQLite corruption."""
        room_dir = self.artifact_dir
        for parent in [self.artifact_dir, *self.artifact_dir.parents]:
            if parent.name == "codex_model_calls":
                room_dir = parent.parent
                break

        codex_home = room_dir / ".codex_runtime"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "log").mkdir(parents=True, exist_ok=True)

        source_home = Path(env.get("CODEX_HOME") or (Path.home() / ".codex"))
        for name in ("auth.json", "config.toml", "installation_id", "models_cache.json"):
            source = source_home / name
            target = codex_home / name
            if source.exists() and not target.exists():
                shutil.copy2(source, target)

        return codex_home


class CodexModelProvider(ModelProvider):
    """Agents SDK model provider backed by Codex CLI."""

    def __init__(self, client: CodexCLIClient):
        self.client = client

    def get_model(self, model_name: str | None) -> Model:
        return CodexAgentsModel(client=self.client, model_name=model_name)


class CodexAgentsModel(Model):
    """Agents SDK Model implementation that delegates turns to Codex CLI."""

    def __init__(self, client: CodexCLIClient, model_name: str | None = None):
        self.client = client
        self.model_name = model_name

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        call_dir = self.client.reserve_call_dir("agents")
        image_dir = call_dir / "images"
        image_paths: list[Path] = []
        sanitized_input = _sanitize_for_prompt(
            input, image_dir=image_dir, image_paths=image_paths
        )

        codex_prompt = _build_agents_prompt(
            system_instructions=system_instructions,
            input_payload=sanitized_input,
            tools=tools,
            output_schema=output_schema,
            model_name=self.model_name,
            model_settings=model_settings,
        )

        result_text = await asyncio.to_thread(
            self.client.run,
            codex_prompt,
            output_schema=_agent_response_schema(),
            image_paths=image_paths,
            exec_config_args=_reasoning_effort_exec_args(model_settings),
            call_name="agents",
            call_dir=call_dir,
        )
        result = _parse_json_object(result_text, "Codex Agents response")

        return ModelResponse(
            output=[_codex_result_to_response_item(result)],
            usage=Usage(requests=1),
            response_id=f"codex_{uuid.uuid4().hex}",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("CodexAgentsModel does not support streaming")
        yield


def create_codex_model_provider(cfg: Any, artifact_dir: Path) -> CodexModelProvider:
    """Create a Codex model provider from a SceneSmith config subtree."""
    client = CodexCLIClient(
        config=CodexCLIConfig.from_cfg(getattr(cfg, "codex", None)),
        artifact_dir=artifact_dir,
    )
    return CodexModelProvider(client=client)


def create_codex_client(cfg: Any, artifact_dir: Path) -> CodexCLIClient:
    """Create a Codex CLI client from a SceneSmith config subtree."""
    return CodexCLIClient(
        config=CodexCLIConfig.from_cfg(getattr(cfg, "codex", None)),
        artifact_dir=artifact_dir,
    )


def codex_text_completion(
    *,
    client: CodexCLIClient,
    messages: list[dict[str, Any]] | str,
    instructions: str | None = None,
    response_format: dict[str, str] | None = None,
    call_name: str = "text",
) -> str:
    """Run a direct text/VLM-style completion through Codex CLI."""
    call_dir = client.reserve_call_dir(call_name)
    image_dir = call_dir / "images"
    image_paths: list[Path] = []
    sanitized_messages = _sanitize_for_prompt(
        messages, image_dir=image_dir, image_paths=image_paths
    )
    prompt = _build_text_prompt(
        messages=sanitized_messages,
        instructions=instructions,
        json_required=bool(response_format and response_format.get("type") == "json_object"),
    )
    return client.run(
        prompt,
        output_schema=None,
        image_paths=image_paths,
        call_name=call_name,
        call_dir=call_dir,
    )


def _build_agents_prompt(
    *,
    system_instructions: str | None,
    input_payload: Any,
    tools: list[Any],
    output_schema: AgentOutputSchemaBase | None,
    model_name: str | None,
    model_settings: ModelSettings,
) -> str:
    tool_specs = []
    for tool in tools:
        if getattr(tool, "name", None):
            tool_specs.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "params_json_schema", {}),
                }
            )

    output_contract: dict[str, Any] = {"plain_text": True}
    if output_schema and not output_schema.is_plain_text():
        output_contract = {
            "plain_text": False,
            "name": output_schema.name(),
            "json_schema": output_schema.json_schema(),
        }

    payload = {
        "model_name_requested_by_scenesmith": model_name,
        "model_settings": _safe_model_dump(model_settings),
        "system_instructions": system_instructions,
        "conversation_input": input_payload,
        "available_tools": tool_specs,
        "final_output_contract": output_contract,
    }

    return (
        "You are serving as the model backend inside the SceneSmith production "
        "pipeline. Do not inspect files, run shell commands, edit files, or bypass "
        "the provided tool interface. Use only the conversation and tool schemas "
        "below.\n\n"
        "Return exactly one JSON object matching the configured output schema:\n"
        "- To ask SceneSmith to run a tool: "
        '{"kind":"tool_call","content":"","name":"tool_name","arguments_json":"{...}"}\n'
        "- To answer the agent turn: "
        '{"kind":"message","content":"...","name":"","arguments_json":"{}"}\n\n'
        "If a final_output_contract includes a json_schema, the message content "
        "must itself be a JSON string matching that schema, with no markdown. "
        "Tool-call arguments_json must be a JSON object string matching the "
        "selected tool's parameters exactly. Use \"{}\" for tools without "
        "arguments.\n\n"
        f"SCENESMITH_MODEL_PAYLOAD:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def _build_text_prompt(
    *,
    messages: Any,
    instructions: str | None,
    json_required: bool,
) -> str:
    json_instruction = (
        "Return only a valid JSON object. Do not wrap it in markdown.\n\n"
        if json_required
        else ""
    )
    return (
        "You are serving as a direct SceneSmith text/VLM completion backend. "
        "Do not inspect files, run shell commands, edit files, or add unrelated "
        "commentary. Answer only the provided request.\n\n"
        f"{json_instruction}"
        f"INSTRUCTIONS:\n{instructions or ''}\n\n"
        f"MESSAGES:\n{json.dumps(messages, indent=2, ensure_ascii=False)}"
    )


def _agent_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "content", "name", "arguments_json"],
        "properties": {
            "kind": {"type": "string", "enum": ["message", "tool_call"]},
            "content": {"type": "string"},
            "name": {"type": "string"},
            "arguments_json": {"type": "string"},
        },
    }


def _codex_result_to_response_item(result: dict[str, Any]) -> Any:
    kind = result.get("kind")
    if kind == "message":
        content = result.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Codex message response missing string 'content'")
        return ResponseOutputMessage(
            id=f"msg_{uuid.uuid4().hex}",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=content,
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )

    if kind == "tool_call":
        name = result.get("name")
        arguments = result.get("arguments")
        arguments_json = result.get("arguments_json")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Codex tool_call response missing string 'name'")
        if arguments is None and arguments_json is None:
            arguments = {}
        elif arguments is None:
            if not isinstance(arguments_json, str):
                raise RuntimeError("Codex tool_call 'arguments_json' must be a string")
            arguments = _parse_json_object(
                arguments_json or "{}",
                "Codex tool_call arguments_json",
            )
        if not isinstance(arguments, dict):
            raise RuntimeError("Codex tool_call arguments must be an object")
        return ResponseFunctionToolCall(
            arguments=json.dumps(arguments, separators=(",", ":")),
            call_id=f"call_{uuid.uuid4().hex}",
            name=name,
            type="function_call",
            id=f"fc_{uuid.uuid4().hex}",
            status="completed",
        )

    raise RuntimeError(f"Unknown Codex response kind: {kind!r}")


def _sanitize_for_prompt(value: Any, *, image_dir: Path, image_paths: list[Path]) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_unset=True)

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            sanitized[key] = _sanitize_for_prompt(
                item, image_dir=image_dir, image_paths=image_paths
            )
        return sanitized

    if isinstance(value, list):
        return [
            _sanitize_for_prompt(item, image_dir=image_dir, image_paths=image_paths)
            for item in value
        ]

    if isinstance(value, str):
        image_path = _maybe_write_data_image(value, image_dir, len(image_paths))
        if image_path is not None:
            image_paths.append(image_path)
            return f"[attached_image:{image_path.name}]"
        return value

    return value


def _maybe_write_data_image(value: str, image_dir: Path, index: int) -> Path | None:
    match = _DATA_IMAGE_RE.match(value)
    if not match:
        return None
    extension = match.group(1).split("+")[0].lower()
    if extension == "jpeg":
        extension = "jpg"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"image_{index:03d}.{extension}"
    image_path.write_bytes(base64.b64decode(match.group(2)))
    return image_path


def _safe_model_dump(value: Any) -> Any:
    """Return a deterministic JSON-only view of Agents SDK model settings.

    ``ModelSettings.extra_args`` can contain provider transport objects such as
    ``httpx.Timeout`` or ``google.genai._interactions.Timeout``.  Those objects
    are neither part of the Codex prompt contract nor used by the Codex CLI
    backend (which has its own ``CodexCLIConfig.timeout_seconds``).  Asking
    Pydantic to serialize the complete settings object therefore fails before
    Codex can run.  Read declared fields directly, omit that transport-only
    timeout, and reject every other non-JSON value instead of silently falling
    back to a potentially nondeterministic ``repr``.
    """

    return _json_safe_model_setting(value, path="model_settings")


def _json_safe_model_setting(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"Non-finite model setting at {path}")
        return value

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Non-string model-setting key at {path}: {key!r}")
            # API-provider timeout objects are transport configuration.  The
            # Codex subprocess timeout is enforced independently by the client.
            if path == "model_settings.extra_args" and key == "timeout":
                continue
            result[key] = _json_safe_model_setting(item, path=f"{path}.{key}")
        return result

    if isinstance(value, (list, tuple)):
        return [
            _json_safe_model_setting(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if is_dataclass(value) and not isinstance(value, type):
        declared = {
            item.name: getattr(value, item.name)
            for item in fields(value)
            if getattr(value, item.name) is not None
        }
        return _json_safe_model_setting(declared, path=path)

    # Pydantic v2 models expose their declared fields on the class.  Reading
    # them avoids invoking a serializer that cannot handle a nested provider
    # timeout object.
    model_fields = getattr(type(value), "model_fields", None)
    if isinstance(model_fields, Mapping):
        declared = {
            name: getattr(value, name)
            for name in model_fields
            if getattr(value, name, None) is not None
        }
        return _json_safe_model_setting(declared, path=path)

    raise TypeError(
        f"Unsupported model-setting value at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _reasoning_effort_exec_args(model_settings: ModelSettings) -> list[str]:
    reasoning = getattr(model_settings, "reasoning", None)
    effort = getattr(reasoning, "effort", None)
    if not effort:
        return []
    return ["-c", f'model_reasoning_effort="{effort}"']


def _parse_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} was not valid JSON: {text[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _safe_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
