from __future__ import annotations

import sys
import weakref

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from PIL import Image

# The handoff repository mirrors only the patched manager; the clean upstream
# checkout supplies cuda_env_setup.py.  Install a no-CUDA collection stub on
# platforms where that upstream file is intentionally absent.
try:
    from scenesmith.agent_utils.geometry_generation_server import (  # noqa: F401
        cuda_env_setup,
    )
except ImportError:
    cuda_env_setup = ModuleType(
        "scenesmith.agent_utils.geometry_generation_server.cuda_env_setup"
    )
    cuda_env_setup.ensure_cuda_env = lambda: False  # type: ignore[attr-defined]
    sys.modules[cuda_env_setup.__name__] = cuda_env_setup

try:
    from scenesmith.agent_utils import mesh_utils  # noqa: F401
except ImportError:
    mesh_utils = ModuleType("scenesmith.agent_utils.mesh_utils")
    mesh_utils.load_mesh_as_trimesh = lambda _path: None  # type: ignore[attr-defined]
    sys.modules[mesh_utils.__name__] = mesh_utils

from scenesmith.agent_utils.geometry_generation_server import sam3d_pipeline_manager
from scripts import run_single_room_worker


def _install_cuda_cleanup_spies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    real_collect = sam3d_pipeline_manager.gc.collect
    monkeypatch.setattr(
        sam3d_pipeline_manager.gc,
        "collect",
        lambda: events.append("gc") or real_collect(),
    )
    monkeypatch.setattr(
        sam3d_pipeline_manager.torch.cuda,
        "is_available",
        lambda: events.append("cuda_available") or True,
    )
    monkeypatch.setattr(
        sam3d_pipeline_manager.torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    return events


def test_sam3d_request_releases_transient_cuda_memory_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _install_cuda_cleanup_spies(monkeypatch)
    exported: list[str] = []
    run_calls: list[dict] = []
    transient_refs: list[weakref.ReferenceType] = []

    class GLB:
        def export(self, path: str) -> None:
            events.append("export")
            exported.append(path)

    class Pipeline:
        def run(self, image: np.ndarray, **kwargs) -> dict:
            events.append("run")
            run_calls.append({"image": image, **kwargs})
            transient = TransientOutput()
            transient.cycle = transient
            transient_refs.append(weakref.ref(transient))
            return {"glb": GLB(), "gaussian": transient, "mesh": object()}

    class TransientOutput:
        cycle: object | None = None

    cached_pipeline = Pipeline()
    monkeypatch.setattr(
        sam3d_pipeline_manager.SAM3DPipelineManager,
        "_sam3d_pipeline",
        cached_pipeline,
    )
    output = tmp_path / "asset.glb"
    sam3d_pipeline_manager.generate_3d_from_mask(
        image=Image.new("RGB", (4, 3), "white"),
        mask=np.ones((3, 4), dtype=np.uint8),
        sam3d_pipeline=cached_pipeline,
        output_path=output,
    )

    assert events == [
        "gc",
        "cuda_available",
        "empty_cache",
        "run",
        "export",
        "gc",
        "cuda_available",
        "empty_cache",
    ]
    assert exported == [str(output)]
    assert len(transient_refs) == 1
    assert transient_refs[0]() is None
    assert len(run_calls) == 1
    assert run_calls[0]["image"].shape == (3, 4, 4)
    assert run_calls[0] | {"image": None} == {
        "image": None,
        "mask": None,
        "with_mesh_postprocess": True,
        "with_texture_baking": True,
        "with_layout_postprocess": True,
        "use_vertex_color": False,
    }
    assert (
        sam3d_pipeline_manager.SAM3DPipelineManager._sam3d_pipeline
        is cached_pipeline
    )


def test_sam3d_request_releases_transient_cuda_memory_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _install_cuda_cleanup_spies(monkeypatch)

    class Pipeline:
        def run(self, _image: np.ndarray, **_kwargs) -> dict:
            events.append("run")
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        sam3d_pipeline_manager.generate_3d_from_mask(
            image=Image.new("RGB", (2, 2), "white"),
            mask=np.ones((2, 2), dtype=np.uint8),
            sam3d_pipeline=Pipeline(),
            output_path=tmp_path / "never.glb",
        )

    assert events == [
        "gc",
        "cuda_available",
        "empty_cache",
        "run",
        "gc",
        "cuda_available",
        "empty_cache",
    ]


def test_cuda_cleanup_failure_does_not_replace_generation_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sam3d_pipeline_manager.gc, "collect", lambda: 0)
    monkeypatch.setattr(
        sam3d_pipeline_manager.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(
        sam3d_pipeline_manager.torch.cuda,
        "empty_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    class Pipeline:
        def run(self, _image: np.ndarray, **_kwargs) -> dict:
            raise RuntimeError("original generation failure")

    with pytest.raises(RuntimeError, match="original generation failure"):
        sam3d_pipeline_manager.generate_3d_from_mask(
            image=Image.new("RGB", (2, 2), "white"),
            mask=np.ones((2, 2), dtype=np.uint8),
            sam3d_pipeline=Pipeline(),
            output_path=tmp_path / "never.glb",
        )


def test_room_worker_sets_expandable_segments_and_preserves_other_settings() -> None:
    empty: dict[str, str] = {}
    assert (
        run_single_room_worker._configure_pytorch_cuda_allocator(empty)
        == "expandable_segments:True"
    )
    assert empty == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    existing = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
    assert run_single_room_worker._configure_pytorch_cuda_allocator(existing) == (
        "max_split_size_mb:256,expandable_segments:True"
    )
    assert existing["PYTORCH_CUDA_ALLOC_CONF"] == (
        "max_split_size_mb:256,expandable_segments:True"
    )

    already_valid = {
        "PYTORCH_CUDA_ALLOC_CONF": (
            "garbage_collection_threshold:0.8,expandable_segments:True"
        )
    }
    assert run_single_room_worker._configure_pytorch_cuda_allocator(
        already_valid
    ) == already_valid["PYTORCH_CUDA_ALLOC_CONF"]


@pytest.mark.parametrize(
    "value",
    (
        "expandable_segments:False",
        "expandable_segments:0",
        "expandable_segments",
        "expandable_segments:True,expandable_segments:True",
    ),
)
def test_room_worker_rejects_conflicting_cuda_allocator_settings(value: str) -> None:
    environment = {"PYTORCH_CUDA_ALLOC_CONF": value}
    with pytest.raises(RuntimeError, match="expandable_segments"):
        run_single_room_worker._configure_pytorch_cuda_allocator(environment)
    assert environment == {"PYTORCH_CUDA_ALLOC_CONF": value}
