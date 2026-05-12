from __future__ import annotations

import ast
import threading

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scenesmith"
    / "agent_utils"
    / "physical_feasibility.py"
)
if not SOURCE_PATH.is_file():
    pytest.skip(
        "SceneSmith physical_feasibility.py is not installed in this handoff checkout",
        allow_module_level=True,
    )


class _RecordingLogger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)


class _RecordingSignal:
    SIGALRM = 14

    def __init__(self) -> None:
        self.old_handler = object()
        self.current_handler = self.old_handler
        self.signal_calls: list[tuple[object, object]] = []
        self.alarm_calls: list[int] = []

    def signal(self, signum, handler):
        if threading.current_thread() is not threading.main_thread():
            raise AssertionError("signal.signal must never run in a worker thread")
        previous = self.current_handler
        self.current_handler = handler
        self.signal_calls.append((signum, handler))
        return previous

    def alarm(self, seconds: int) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise AssertionError("signal.alarm must never run in a worker thread")
        self.alarm_calls.append(seconds)


class _Solver:
    def __init__(self, *, trigger_alarm: _RecordingSignal | None = None) -> None:
        self.trigger_alarm = trigger_alarm
        self.calls: list[tuple[object, object, int]] = []

    def Solve(self, prog, initial_guess, options):  # noqa: N802 - Drake API
        self.calls.append((prog, options, threading.get_ident()))
        assert initial_guess is None
        if self.trigger_alarm is not None:
            self.trigger_alarm.current_handler(self.trigger_alarm.SIGALRM, None)
        return "solver-result"


def _load_timeout_helper(
    fake_signal: _RecordingSignal,
) -> tuple[object, _RecordingLogger]:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_solve_projection_with_timeout"
    )
    logger = _RecordingLogger()
    namespace = {
        "signal": fake_signal,
        "threading": threading,
        "console_logger": logger,
    }
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            filename=str(SOURCE_PATH),
            mode="exec",
        ),
        namespace,
    )
    return namespace["_solve_projection_with_timeout"], logger


def test_worker_thread_runs_solver_without_installing_signal_handler() -> None:
    fake_signal = _RecordingSignal()
    solve_with_timeout, logger = _load_timeout_helper(fake_signal)
    solver = _Solver()
    prog = object()
    options = object()
    main_thread_id = threading.get_ident()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            solve_with_timeout,
            solver=solver,
            prog=prog,
            options=options,
            time_limit_s=360.0,
        ).result(timeout=5)

    assert result == "solver-result"
    assert len(solver.calls) == 1
    called_prog, called_options, worker_thread_id = solver.calls[0]
    assert called_prog is prog
    assert called_options is options
    assert worker_thread_id != main_thread_id
    assert fake_signal.signal_calls == []
    assert fake_signal.alarm_calls == []
    assert logger.debug_messages
    assert "solver's configured time and iteration limits" in logger.debug_messages[0]


def test_main_thread_keeps_hard_alarm_and_restores_previous_handler() -> None:
    fake_signal = _RecordingSignal()
    solve_with_timeout, _ = _load_timeout_helper(fake_signal)
    solver = _Solver()

    result = solve_with_timeout(
        solver=solver,
        prog="program",
        options="options",
        time_limit_s=360.0,
    )

    assert result == "solver-result"
    assert fake_signal.alarm_calls == [420, 0]
    assert len(fake_signal.signal_calls) == 2
    assert fake_signal.signal_calls[0][0] == fake_signal.SIGALRM
    assert callable(fake_signal.signal_calls[0][1])
    assert fake_signal.signal_calls[1] == (
        fake_signal.SIGALRM,
        fake_signal.old_handler,
    )
    assert fake_signal.current_handler is fake_signal.old_handler


def test_main_thread_timeout_still_cancels_alarm_and_propagates() -> None:
    fake_signal = _RecordingSignal()
    solve_with_timeout, _ = _load_timeout_helper(fake_signal)
    solver = _Solver(trigger_alarm=fake_signal)

    with pytest.raises(TimeoutError, match="Projection hard timeout"):
        solve_with_timeout(
            solver=solver,
            prog="program",
            options="options",
            time_limit_s=5.0,
        )

    assert fake_signal.alarm_calls == [65, 0]
    assert fake_signal.current_handler is fake_signal.old_handler


def test_projection_uses_helper_and_both_solvers_have_internal_limits() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    solve = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "solve_non_penetration_ik"
    )
    called_names = {
        node.func.id
        for node in ast.walk(solve)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    direct_signal_calls = {
        node.func.attr
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "signal"
    }

    assert "_solve_projection_with_timeout" in called_names
    assert direct_signal_calls == set()
    assert '"Time limit", time_limit_s' in source
    assert '"max_cpu_time", time_limit_s' in source
