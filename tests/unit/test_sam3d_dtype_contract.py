from contextlib import contextmanager
import sys
from types import ModuleType

import numpy as np
import torch
from PIL import Image

# The handoff repository mirrors only the patched manager; the clean upstream
# checkout supplies cuda_env_setup.py. Keep CPU-only local collection possible.
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


def test_generate_mask_uses_sam3_bfloat16_autocast_and_converts_scores(
    monkeypatch,
) -> None:
    autocast_state = {"active": False, "calls": []}

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        autocast_state["calls"].append((device_type, dtype))
        assert autocast_state["active"] is False
        autocast_state["active"] = True
        try:
            yield
        finally:
            autocast_state["active"] = False

    monkeypatch.setattr(sam3d_pipeline_manager.torch, "autocast", fake_autocast)

    class Processor:
        def set_image(self, image: Image.Image) -> dict:
            assert image.mode == "RGB"
            assert autocast_state["active"] is True
            return {"image_encoded": True}

        def set_text_prompt(self, *, state: dict, prompt: str) -> dict:
            assert state == {"image_encoded": True}
            assert prompt == "student desk"
            assert autocast_state["active"] is True
            mask = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
            mask[:, :, 1:3, 1:3] = True
            return {
                "masks": mask,
                # NumPy cannot consume a bfloat16 tensor directly.  This
                # mirrors the real SAM3 output under its required autocast.
                "scores": torch.tensor([0.75], dtype=torch.bfloat16),
            }

    mask = sam3d_pipeline_manager.generate_mask(
        image=Image.new("RGB", (4, 4), "white"),
        sam3_processor=Processor(),
        mode="object_description",
        object_description="student desk",
    )

    assert autocast_state["calls"] == [("cuda", torch.bfloat16)]
    assert autocast_state["active"] is False
    assert mask.dtype == np.uint8
    assert mask.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
