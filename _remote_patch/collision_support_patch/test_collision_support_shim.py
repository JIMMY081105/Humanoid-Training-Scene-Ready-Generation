"""Focused integration test for the visual-free shelf collision repair."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, RigidTransform

from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("collision_support_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate agent: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.StatefulManipulandAgent


def main() -> None:
    candidate = Path(sys.argv[1]).resolve(strict=True)
    Agent = _load_agent(candidate)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        assets_root = root / "generated_assets" / "manipuland"
        registry = AssetRegistry(
            auto_save_path=assets_root / "asset_registry.json",
            required_root=assets_root,
            allowed_object_types=frozenset({ObjectType.MANIPULAND}),
        )
        scene = RoomScene(room_geometry=None, scene_dir=root)
        surface = SupportSurface(
            surface_id=UniqueID("S_shelf"),
            bounding_box_min=np.array([-0.40, -0.15, 0.0]),
            bounding_box_max=np.array([0.40, 0.15, 0.0]),
            transform=RigidTransform(p=[1.0, 2.0, 1.5]),
        )
        furniture = SceneObject(
            object_id=UniqueID("bookcase_0"),
            object_type=ObjectType.FURNITURE,
            name="bookcase",
            description="tall library bookcase with interior shelves",
            transform=RigidTransform(),
            support_surfaces=[surface],
        )
        scene.add_object(furniture)
        holder = SimpleNamespace(
            scene=scene,
            asset_manager=SimpleNamespace(registry=registry),
            _requires_collision_support_shims=Agent._requires_collision_support_shims,
            _collision_support_sdf=Agent._collision_support_sdf,
            _collision_support_gltf=Agent._collision_support_gltf,
        )

        installed = Agent._install_collision_support_shims(
            holder, furniture=furniture, surfaces=[surface]
        )
        assert installed == 1
        installed_again = Agent._install_collision_support_shims(
            holder, furniture=furniture, surfaces=[surface]
        )
        assert installed_again == 0

        shims = [
            obj
            for obj in scene.objects.values()
            if obj.metadata.get("collision_support_shim") is True
        ]
        assert len(shims) == 1
        shim = shims[0]
        assert shim.placement_info is not None
        assert shim.placement_info.parent_surface_id == surface.surface_id
        assert np.allclose(shim.transform.translation(), [1.0, 2.0, 1.495])
        assert shim.sdf_path is not None
        sdf = shim.sdf_path.read_text(encoding="utf-8")
        assert "<static>true</static>" in sdf
        assert "0.796 0.296 0.01" in sdf
        builder = DiagramBuilder()
        plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(plant).AddModels(str(shim.sdf_path))
        plant.Finalize()
        directive = scene.to_drake_directive(
            include_objects=[shim.object_id], exclude_room_geometry=True
        )
        assert "collision_support_bookcase_0_S_shelf" in directive
        assert "default_free_body_pose" in directive
    print("COLLISION_SUPPORT_SHIM_TEST=PASS")


if __name__ == "__main__":
    main()
