from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
import types
import uuid

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import flask
import pytest
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGERS = (
    (
        "geometry_generation_server",
        "GeometryGenerationServer",
        "GeometryGenerationApp",
        {"backend": "sam3d", "sam3d_config": {"checkpoint": "unused"}},
    ),
    (
        "objaverse_retrieval_server",
        "ObjaverseRetrievalServer",
        "ObjaverseRetrievalApp",
        {},
    ),
    (
        "articulated_retrieval_server",
        "ArticulatedRetrievalServer",
        "ArticulatedRetrievalApp",
        {},
    ),
    (
        "materials_retrieval_server",
        "MaterialsRetrievalServer",
        "MaterialsRetrievalApp",
        {},
    ),
    (
        "hssd_retrieval_server",
        "HssdRetrievalServer",
        "HssdRetrievalApp",
        {},
    ),
)


class _FakeProcessingApp(flask.Flask):
    instances: list[_FakeProcessingApp] = []

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__(f"test-server-{uuid.uuid4().hex}")
        self.processing_started = False
        self.processing_stopped = False
        self._processing_stop = threading.Event()
        self.processing_thread: threading.Thread | None = None
        self._active_lock = threading.Lock()
        self._active_requests = 0
        self.max_active_requests = 0
        self.add_url_rule("/health", "health", lambda: {"status": "ok"})
        self.add_url_rule("/slow", "slow", self._slow_request)
        type(self).instances.append(self)

    def _slow_request(self):
        with self._active_lock:
            self._active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests, self._active_requests
            )
        try:
            time.sleep(0.2)
            return {"status": "ok"}
        finally:
            with self._active_lock:
                self._active_requests -= 1

    def start_processing(self) -> None:
        self.processing_started = True
        self.processing_thread = threading.Thread(
            target=self._processing_stop.wait,
            name=f"test-processing-{uuid.uuid4().hex}",
            daemon=False,
        )
        self.processing_thread.start()

    def stop_processing(self) -> None:
        self.processing_stopped = True
        self._processing_stop.set()
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2)
            if self.processing_thread.is_alive():
                raise RuntimeError("test processing thread did not stop")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stub_absolute_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    packages = (
        "scenesmith",
        "scenesmith.utils",
        "scenesmith.agent_utils",
        "scenesmith.agent_utils.articulated_retrieval_server",
        "scenesmith.agent_utils.materials_retrieval_server",
    )
    for package_name in packages:
        module = types.ModuleType(package_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, module)

    network_utils = types.ModuleType("scenesmith.utils.network_utils")
    network_utils.is_port_available = lambda host, port: True
    monkeypatch.setitem(
        sys.modules, "scenesmith.utils.network_utils", network_utils
    )

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = type("DictConfig", (), {})
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    articulated_config = types.ModuleType(
        "scenesmith.agent_utils.articulated_retrieval_server.config"
    )
    articulated_config.ArticulatedConfig = type("ArticulatedConfig", (), {})
    monkeypatch.setitem(
        sys.modules,
        "scenesmith.agent_utils.articulated_retrieval_server.config",
        articulated_config,
    )

    materials_config = types.ModuleType(
        "scenesmith.agent_utils.materials_retrieval_server.config"
    )
    materials_config.MaterialsConfig = type("MaterialsConfig", (), {})
    monkeypatch.setitem(
        sys.modules,
        "scenesmith.agent_utils.materials_retrieval_server.config",
        materials_config,
    )


def _load_manager(
    monkeypatch: pytest.MonkeyPatch,
    server_package: str,
    manager_class: str,
    app_class: str,
):
    _stub_absolute_imports(monkeypatch)
    source_dir = REPO_ROOT / "scenesmith" / "agent_utils" / server_package
    package_name = f"_shutdown_test_{server_package}_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    app_module = types.ModuleType(f"{package_name}.server_app")
    setattr(app_module, app_class, _FakeProcessingApp)
    monkeypatch.setitem(sys.modules, app_module.__name__, app_module)

    module_name = f"{package_name}.server_manager"
    module_path = source_dir / "server_manager.py"
    if not module_path.is_file():
        pytest.skip("requires a full SceneSmith source checkout")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return getattr(module, manager_class)


@pytest.mark.parametrize(
    "server_package,manager_class,app_class,manager_kwargs", MANAGERS
)
def test_managed_server_shutdown_owns_and_joins_every_thread(
    monkeypatch: pytest.MonkeyPatch,
    server_package: str,
    manager_class: str,
    app_class: str,
    manager_kwargs: dict,
) -> None:
    manager_type = _load_manager(
        monkeypatch, server_package, manager_class, app_class
    )
    port = _free_port()
    server = manager_type(host="127.0.0.1", port=port, **manager_kwargs)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: pytest.fail(
            "managed shutdown must not call the broken /shutdown endpoint"
        ),
    )

    server.start()
    app = server._app
    serving_thread = server._server_thread
    processing_thread = app.processing_thread
    assert server._http_server.multithread is True
    assert serving_thread is not None and serving_thread.daemon is False
    assert processing_thread is not None and processing_thread.daemon is False

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(
                lambda _: requests.get(
                    f"http://127.0.0.1:{port}/slow", timeout=2
                ),
                range(2),
            )
        )
    assert [response.status_code for response in responses] == [200, 200]
    assert app.max_active_requests == 2

    server.stop()

    assert not serving_thread.is_alive()
    assert not processing_thread.is_alive()
    assert app.processing_stopped is True
    assert server.is_running() is False
    assert server._http_server is None
    assert server._server_thread is None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))

    # A cleanly stopped manager can bind the same port again and stop twice.
    server.start()
    restarted_thread = server._server_thread
    server.stop()
    server.stop()
    assert restarted_thread is not None and not restarted_thread.is_alive()


@pytest.mark.parametrize(
    "server_package,manager_class,app_class,manager_kwargs", MANAGERS
)
def test_startup_failure_rolls_back_http_and_processing_threads(
    monkeypatch: pytest.MonkeyPatch,
    server_package: str,
    manager_class: str,
    app_class: str,
    manager_kwargs: dict,
) -> None:
    manager_type = _load_manager(
        monkeypatch, server_package, manager_class, app_class
    )
    port = _free_port()
    server = manager_type(host="127.0.0.1", port=port, **manager_kwargs)
    captured: dict[str, object] = {}
    real_cleanup = server._cleanup

    def capture_cleanup() -> None:
        captured["app"] = server._app
        captured["server_thread"] = server._server_thread
        real_cleanup()

    monkeypatch.setattr(server, "_cleanup", capture_cleanup)
    monkeypatch.setattr(
        server,
        "_wait_until_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic readiness failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic readiness failure"):
        server.start()

    app = captured["app"]
    serving_thread = captured["server_thread"]
    assert app.processing_stopped is True
    assert not app.processing_thread.is_alive()
    assert not serving_thread.is_alive()
    assert server._app is None
    assert server._http_server is None
    assert server._server_thread is None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))
