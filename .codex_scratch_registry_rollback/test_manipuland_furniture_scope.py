"""Per-furniture manipuland ownership and surface-ID scope regressions."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenesmith.agent_utils import physical_feasibility
from scenesmith.agent_utils.asset_manager import AssetManager
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils import asset_registry as asset_registry_module
from scenesmith.agent_utils.asset_router import router as asset_router_module
from scenesmith.agent_utils.house import RoomGeometry
from scenesmith.agent_utils.physics_validation import (
    CollisionPair,
    filter_collisions_by_agent,
)
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
    serialize_rigid_transform,
)
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenesmith.manipuland_agents.tools.stacking import StackSimulationResult
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)


def _surface(surface_id: str, z: float = 1.0) -> SupportSurface:
    return SupportSurface(
        surface_id=UniqueID(surface_id),
        bounding_box_min=np.array([-1.0, -1.0, 0.0]),
        bounding_box_max=np.array([1.0, 1.0, 2.0]),
        transform=RigidTransform(p=[0.0, 0.0, z]),
    )


def _furniture(object_id: str, surface: SupportSurface) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.FURNITURE,
        name=object_id,
        description=object_id,
        transform=RigidTransform(),
        support_surfaces=[surface],
        bbox_min=np.array([-1.0, -1.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
    )


def _manipuland(object_id: str, surface_id: str, geometry_path: Path) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.MANIPULAND,
        name=object_id,
        description=object_id,
        transform=RigidTransform(p=[0.0, 0.0, 1.0]),
        geometry_path=geometry_path,
        placement_info=PlacementInfo(
            parent_surface_id=UniqueID(surface_id),
            position_2d=np.array([0.0, 0.0]),
            rotation_2d=0.0,
        ),
        bbox_min=np.array([-0.05, -0.05, 0.0]),
        bbox_max=np.array([0.05, 0.05, 0.1]),
    )


def _asset_files(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    geometry_path = tmp_path / f"{stem}.gltf"
    sdf_path = tmp_path / f"{stem}.sdf"
    geometry_path.write_text("{}", encoding="utf-8")
    sdf_path.write_text(
        "<sdf version='1.7'><model name='asset'><link name='base'/></model></sdf>",
        encoding="utf-8",
    )
    return geometry_path, sdf_path


def _registry_asset(namespace: Path, object_id: str) -> SceneObject:
    asset_dir = namespace / "sdf" / f"{object_id}_unique"
    asset_dir.mkdir(parents=True, exist_ok=True)
    geometry, sdf = _asset_files(asset_dir, object_id)
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.MANIPULAND,
        name=object_id,
        description=f"reusable {object_id}",
        transform=RigidTransform(),
        geometry_path=geometry,
        sdf_path=sdf,
        metadata={"asset_source": "objaverse", "nested": {"value": 1}},
        bbox_min=np.array([-0.1, -0.1, 0.0]),
        bbox_max=np.array([0.1, 0.1, 0.1]),
    )


def _tools(scene: RoomScene, current: SceneObject, protected: set[str]) -> ManipulandTools:
    cfg = OmegaConf.load(
        Path(__file__).parents[2]
        / "configurations/manipuland_agent/base_manipuland_agent.yaml"
    )
    return ManipulandTools(
        scene=scene,
        asset_manager=MagicMock(),
        cfg=cfg,
        current_furniture_id=current.object_id,
        support_surfaces={
            str(surface.surface_id): surface for surface in current.support_surfaces
        },
        protected_object_ids=protected,
    )


def test_restore_advances_surface_allocator_and_rejects_duplicate_owners(
    tmp_path: Path,
) -> None:
    scene = RoomScene(room_geometry=None, scene_dir=tmp_path)
    teacher = _furniture("teacher_desk_0", _surface("S_a"))
    scene.add_object(teacher)
    state = scene.to_state_dict()

    restored = RoomScene(room_geometry=None, scene_dir=tmp_path)
    restored.restore_from_state_dict(state)
    assert restored.generate_surface_id() == UniqueID("S_b")

    duplicate = json.loads(json.dumps(state))
    cabinet = json.loads(json.dumps(state["objects"]["teacher_desk_0"]))
    cabinet["object_id"] = "filing_cabinet_0"
    cabinet["name"] = "filing_cabinet_0"
    duplicate["objects"]["filing_cabinet_0"] = cabinet
    with pytest.raises(RuntimeError, match="Duplicate support surface ID S_a"):
        restored.restore_from_state_dict(duplicate)


def test_real_room_geometry_floor_roundtrip_reconciles_surface_ids(
    tmp_path: Path,
) -> None:
    sdf_path = tmp_path / "room.sdf"
    sdf_path.write_text(
        "<sdf version='1.7'><model name='room'><link name='base'/></model></sdf>",
        encoding="utf-8",
    )
    floor = SceneObject(
        object_id=UniqueID("floor_classroom_01"),
        object_type=ObjectType.FLOOR,
        name="floor",
        description="floor",
        transform=RigidTransform(),
        sdf_path=sdf_path,
        support_surfaces=[_surface("S_4", z=0.0)],
    )
    geometry = RoomGeometry(
        sdf_tree=ET.parse(sdf_path),
        sdf_path=sdf_path,
        floor=floor,
    )
    state = RoomScene(room_geometry=geometry, scene_dir=tmp_path).to_state_dict()
    restored = RoomScene(room_geometry=None, scene_dir=tmp_path)
    restored.restore_from_state_dict(state)
    assert restored.room_geometry.floor.support_surfaces[0].surface_id == "S_4"
    assert restored.generate_surface_id() == "S_5"


def test_distinct_floor_object_alias_is_rejected(tmp_path: Path) -> None:
    floor = SceneObject(
        object_id=UniqueID("floor_classroom_01"),
        object_type=ObjectType.FLOOR,
        name="floor",
        description="floor",
        transform=RigidTransform(),
        support_surfaces=[_surface("S_floor")],
    )
    alias = SceneObject(
        object_id=floor.object_id,
        object_type=ObjectType.FLOOR,
        name="alias",
        description="alias",
        transform=RigidTransform(),
        support_surfaces=[_surface("S_alias")],
    )
    scene = RoomScene(
        room_geometry=SimpleNamespace(floor=floor),
        scene_dir=tmp_path,
        objects={alias.object_id: alias},
    )
    with pytest.raises(RuntimeError, match="aliases a distinct scene object"):
        scene.generate_surface_id()


def test_active_scope_binds_every_protected_direct_and_composite_asset(
    tmp_path: Path,
) -> None:
    furniture_geometry, furniture_sdf = _asset_files(tmp_path, "cabinet")
    member_geometry, member_sdf = _asset_files(tmp_path, "prior_member")
    container_geometry, container_sdf = _asset_files(tmp_path, "prior_container")
    fill_geometry, fill_sdf = _asset_files(tmp_path, "prior_fill")
    furniture = _furniture("filing_cabinet_0", _surface("S_0"))
    furniture.geometry_path = furniture_geometry
    furniture.sdf_path = furniture_sdf
    prior = _manipuland("prior_stack_0", "S_0", member_geometry)
    prior.geometry_path = None
    prior.sdf_path = None
    prior.metadata = {
        "composite_type": "stack",
        "member_assets": [
            {
                "asset_id": "prior_registry_asset",
                "name": "prior member",
                "geometry_path": str(member_geometry),
                "sdf_path": str(member_sdf),
                "transform": serialize_rigid_transform(prior.transform),
            }
        ],
        "num_members": 1,
    }
    prior_filled = _manipuland("prior_filled_0", "S_0", fill_geometry)
    prior_filled.geometry_path = None
    prior_filled.sdf_path = None
    prior_filled.metadata = {
        "composite_type": "filled_container",
        "container_asset": {
            "asset_id": "prior_container_asset",
            "name": "container",
            "geometry_path": str(container_geometry),
            "sdf_path": str(container_sdf),
            "transform": serialize_rigid_transform(prior_filled.transform),
        },
        "fill_assets": [
            {
                "asset_id": "prior_fill_asset",
                "name": "fill",
                "geometry_path": str(fill_geometry),
                "sdf_path": str(fill_sdf),
                "transform": serialize_rigid_transform(prior_filled.transform),
            }
        ],
        "num_fill_objects": 1,
    }
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={
            obj.object_id: obj for obj in (furniture, prior, prior_filled)
        },
    )
    agent = StatefulManipulandAgent.__new__(StatefulManipulandAgent)
    agent.scene = scene
    agent.cfg = OmegaConf.create({"checkpoint_test": True})
    agent._protected_object_ids = frozenset()
    agent._furniture_scope_input_scene_state = None
    agent._furniture_scope_target_id = None
    agent._protected_asset_object_ids = frozenset()
    agent._protected_asset_files = None
    agent._begin_furniture_scope(
        furniture_id=furniture.object_id,
        furniture_index=0,
        input_scene_state=scene.to_state_dict(),
    )
    assert {record["path"] for record in agent._protected_asset_files} == {
        path.name
        for path in (
            furniture_geometry,
            furniture_sdf,
            member_geometry,
            member_sdf,
            container_geometry,
            container_sdf,
            fill_geometry,
            fill_sdf,
        )
    }

    # Only support extraction on the active target is a legal baseline delta.
    furniture.support_surfaces.append(_surface("S_1"))
    agent._verify_active_furniture_scope()

    original_member_bytes = member_geometry.read_bytes()
    member_geometry.write_bytes(b'{"mutated":true}')
    with pytest.raises(RuntimeError, match="asset inventory mismatch"):
        agent._verify_active_furniture_scope()
    member_geometry.write_bytes(original_member_bytes)

    prior.description = "mutated prior composite"
    with pytest.raises(RuntimeError, match="mutated protected object"):
        agent._verify_active_furniture_scope()


@pytest.mark.parametrize("mutation", ["transform", "metadata", "room_width"])
def test_floor_semantics_allow_only_target_support_surface_delta(mutation: str) -> None:
    floor = {
        "object_id": "floor_classroom_01",
        "object_type": "floor",
        "transform": {
            "translation": [0, 0, 0],
            "rotation_wxyz": [1, 0, 0, 0],
        },
        "support_surfaces": [],
        "metadata": {"authority": "room"},
    }
    before = {
        "objects": {},
        "text_description": "classroom",
        "room_geometry": {"floor": floor, "walls": [], "width": 10.0},
    }
    after = json.loads(json.dumps(before))
    after["room_geometry"]["floor"]["support_surfaces"] = [
        {"surface_id": "S_floor"}
    ]
    StatefulManipulandAgent._verify_input_object_semantics(
        input_scene_state=before,
        output_scene_state=after,
        target_furniture_id="floor_classroom_01",
    )
    if mutation == "transform":
        after["room_geometry"]["floor"]["transform"]["translation"][0] = 1
    elif mutation == "metadata":
        after["room_geometry"]["floor"]["metadata"]["authority"] = "changed"
    else:
        after["room_geometry"]["width"] = 9.0
    with pytest.raises(RuntimeError, match="changed|mutated"):
        StatefulManipulandAgent._verify_input_object_semantics(
            input_scene_state=before,
            output_scene_state=after,
            target_furniture_id="floor_classroom_01",
        )


def test_room_geometry_objects_compare_itemwise_with_quaternion_sign_equivalence() -> None:
    def geometry_object(object_id: str, object_type: str) -> dict:
        return {
            "object_id": object_id,
            "object_type": object_type,
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "support_surfaces": [],
            "metadata": {"authority": "room"},
        }

    before = {
        "objects": {},
        "text_description": "classroom",
        "room_geometry": {
            "floor": geometry_object("floor_classroom_01", "floor"),
            "walls": [geometry_object("north_wall_0", "wall")],
            "width": 10.0,
        },
    }
    after = json.loads(json.dumps(before))
    after["room_geometry"]["floor"]["transform"]["rotation_wxyz"] = [
        -1.0,
        0.0,
        0.0,
        0.0,
    ]
    after["room_geometry"]["walls"][0]["transform"]["rotation_wxyz"] = [
        -1.0,
        0.0,
        0.0,
        0.0,
    ]
    StatefulManipulandAgent._verify_input_object_semantics(
        input_scene_state=before,
        output_scene_state=after,
        target_furniture_id="unrelated_target",
    )
    after["room_geometry"]["walls"][0]["metadata"]["authority"] = "changed"
    with pytest.raises(RuntimeError, match="mutated protected object"):
        StatefulManipulandAgent._verify_input_object_semantics(
            input_scene_state=before,
            output_scene_state=after,
            target_furniture_id="unrelated_target",
        )


def test_duplicate_surface_id_does_not_expose_or_mutate_prior_manipuland(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    teacher = _furniture("teacher_desk_0", _surface("S_0"))
    cabinet = _furniture("filing_cabinet_0", _surface("S_0", z=1.2))
    prior = _manipuland("gradebook_0", "S_0", geometry)
    current = _manipuland("folder_0", "S_0", geometry)
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (teacher, cabinet, prior, current)},
    )
    tools = _tools(scene, cabinet, {str(prior.object_id), str(teacher.object_id)})

    state = json.loads(tools._get_current_scene_state_impl())
    visible_ids = {
        item["object_id"]
        for surface in state["surfaces"]
        for item in surface["manipulands"]
    }
    assert visible_ids == {"folder_0"}

    before = prior.transform.GetAsMatrix4().copy()
    removed = json.loads(tools._remove_manipuland_impl("gradebook_0"))
    moved = json.loads(
        tools._move_manipuland_impl(
            object_id="gradebook_0",
            surface_id="S_0",
            position_x=0.2,
            position_z=0.1,
        )
    )
    resolved = json.loads(tools._resolve_penetrations_impl(["gradebook_0"]))
    assert removed["error_type"] == "immutable_object"
    assert moved["error_type"] == "immutable_object"
    assert resolved["error_type"] == "immutable_object"
    assert scene.get_object(prior.object_id) is prior
    np.testing.assert_allclose(prior.transform.GetAsMatrix4(), before)

    monkeypatch.setattr(tools, "_is_top_surface", lambda _: True)
    monkeypatch.setattr(
        tools, "_validate_convex_hull_footprint", lambda **_: (True, None)
    )
    monkeypatch.setattr(tools, "_remove_if_placement_collides", lambda **_: None)
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.tools.manipuland_tools.apply_placement_noise",
        lambda transform, **_: transform,
    )
    edited = json.loads(
        tools._move_manipuland_impl(
            object_id="folder_0",
            surface_id="S_0",
            position_x=0.2,
            position_z=0.1,
        )
    )
    assert edited["success"] is True
    np.testing.assert_allclose(current.placement_info.position_2d, [0.2, 0.1])


def test_non_manipuland_mutations_and_mixed_resolution_are_atomic(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    target = _furniture("filing_cabinet_0", _surface("S_0"))
    unrelated = _furniture("unrelated_cabinet_0", _surface("S_1"))
    current = _manipuland("folder_0", "S_0", geometry)
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (target, unrelated, current)},
    )
    tools = _tools(scene, target, set())
    unrelated_before = unrelated.transform.GetAsMatrix4().copy()
    current_before = current.transform.GetAsMatrix4().copy()

    moved = json.loads(
        tools._move_manipuland_impl(
            object_id=str(unrelated.object_id),
            surface_id="S_0",
            position_x=0.2,
            position_z=0.1,
        )
    )
    removed = json.loads(tools._remove_manipuland_impl(str(unrelated.object_id)))
    mixed = json.loads(
        tools._resolve_penetrations_impl(
            [str(current.object_id), str(unrelated.object_id)]
        )
    )
    assert moved["success"] is False
    assert removed["success"] is False
    assert mixed["success"] is False
    assert scene.get_object(unrelated.object_id) is unrelated
    np.testing.assert_allclose(unrelated.transform.GetAsMatrix4(), unrelated_before)
    np.testing.assert_allclose(current.transform.GetAsMatrix4(), current_before)


def test_manipuland_target_surface_extraction_does_not_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = tmp_path / "shared.gltf"
    geometry.write_text("{}", encoding="utf-8")
    first = _furniture("student_desk_0", _surface("unused_0"))
    second = _furniture("student_desk_1", _surface("unused_1"))
    first.geometry_path = geometry
    second.geometry_path = geometry
    first.support_surfaces = []
    second.support_surfaces = []
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={first.object_id: first, second.object_id: second},
    )
    local_surface = _surface("local")
    monkeypatch.setattr(
        "scenesmith.agent_utils.room.extract_support_surfaces_from_mesh",
        lambda **_: [local_surface],
    )
    surfaces = extract_and_propagate_support_surfaces(
        scene=scene,
        furniture_object=first,
        config=SimpleNamespace(recompute_hssd_surfaces=False),
        propagate_to_identical=False,
    )
    assert surfaces == first.support_surfaces
    assert len(first.support_surfaces) == 1
    assert second.support_surfaces == []


def test_current_stack_remains_visible_and_editable_with_reused_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = tmp_path / "reused.gltf"
    geometry.write_text("{}", encoding="utf-8")
    furniture = _furniture("filing_cabinet_0", _surface("S_0"))
    prior = _manipuland("prior_book_0", "S_0", geometry)
    stack = _manipuland("stack_0", "S_0", geometry)
    stack.metadata = {
        "composite_type": "stack",
        "member_assets": [
            {
                "asset_id": "prior_book_asset",
                "geometry_path": str(geometry),
                "sdf_path": str(tmp_path / "reused.sdf"),
                "transform": serialize_rigid_transform(stack.transform),
            }
        ],
    }
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (furniture, prior, stack)},
    )
    tools = _tools(scene, furniture, {str(prior.object_id)})
    visible = json.loads(tools._get_current_scene_state_impl())
    assert visible["surfaces"][0]["manipulands"][0]["object_id"] == "stack_0"
    member_before = stack.metadata["member_assets"][0]["transform"]
    monkeypatch.setattr(tools, "_is_top_surface", lambda _: True)
    monkeypatch.setattr(
        tools, "_validate_convex_hull_footprint", lambda **_: (True, None)
    )
    monkeypatch.setattr(tools, "_remove_if_placement_collides", lambda **_: None)
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.tools.manipuland_tools.apply_placement_noise",
        lambda transform, **_: transform,
    )
    result = json.loads(
        tools._move_manipuland_impl(
            object_id="stack_0",
            surface_id="S_0",
            position_x=0.25,
            position_z=0.15,
        )
    )
    assert result["success"] is True
    assert stack.metadata["member_assets"][0]["transform"] != member_before
    assert scene.get_object(prior.object_id) is prior


@pytest.mark.parametrize("composite_kind", ["stack", "filled_container", "pile"])
def test_composite_tools_reuse_prior_registry_assets_without_mutating_prior_object(
    composite_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, sdf = _asset_files(tmp_path, "prior_registry_asset")
    furniture = _furniture("filing_cabinet_0", _surface("S_0"))
    prior = _manipuland("prior_registry_asset", "S_0", geometry)
    prior.sdf_path = sdf
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (furniture, prior)},
    )
    asset_manager = MagicMock()
    asset_manager.get_asset_by_id.side_effect = (
        lambda object_id: prior if object_id == prior.object_id else None
    )
    asset_manager.list_available_assets.return_value = [prior]
    tools = _tools(scene, furniture, {str(prior.object_id)})
    tools.asset_manager = asset_manager
    monkeypatch.setattr(tools, "_remove_if_placement_collides", lambda **_: None)
    prior_before = json.dumps(
        scene.to_state_dict()["objects"][str(prior.object_id)], sort_keys=True
    )
    geometry_before = geometry.read_bytes()
    sdf_before = sdf.read_bytes()

    if composite_kind == "stack":
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.stack_tools."
            "load_collision_bounds_for_scene_object",
            lambda _: (-0.05, 0.05),
        )
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.stack_tools.simulate_stack_stability",
            lambda **kwargs: StackSimulationResult(
                is_stable=True,
                final_transforms=kwargs["initial_transforms"],
                stable_indices=[0, 1],
                unstable_indices=[],
            ),
        )
        result = json.loads(
            tools._create_stack_impl(
                asset_ids=[str(prior.object_id), str(prior.object_id)],
                surface_id="S_0",
                position_x=0.0,
                position_z=0.0,
            )
        )
        composite_id = result["stack_object_id"]
        member_records = scene.get_object(UniqueID(composite_id)).metadata[
            "member_assets"
        ]
    elif composite_kind == "pile":
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.pile_tools."
            "compute_pile_spawn_transforms",
            lambda **kwargs: [
                kwargs["base_transform"],
                RigidTransform(
                    p=kwargs["base_transform"].translation()
                    + np.array([0.05, 0.0, 0.0])
                ),
            ],
        )
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.pile_tools.simulate_pile_physics",
            lambda **kwargs: (
                [0, 1],
                [],
                kwargs["initial_transforms"],
                None,
            ),
        )
        result = json.loads(
            tools._create_pile_impl(
                asset_ids=[str(prior.object_id), str(prior.object_id)],
                surface_id="S_0",
                position_x=0.0,
                position_z=0.0,
            )
        )
        composite_id = result["pile_object_id"]
        member_records = scene.get_object(UniqueID(composite_id)).metadata[
            "member_assets"
        ]
    else:
        monkeypatch.setattr(
            SupportSurface, "contains_point_2d", lambda self, position: True
        )
        monkeypatch.setattr(tools, "_is_top_surface", lambda _: True)
        monkeypatch.setattr(
            tools, "_validate_convex_hull_footprint", lambda **_: (True, None)
        )
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.fill_tools."
            "load_collision_meshes_from_sdf",
            lambda _: [SimpleNamespace(vertices=np.zeros((1, 3)))],
        )
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.fill_tools."
            "compute_container_interior_bounds",
            lambda **_: SimpleNamespace(),
        )
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.fill_tools.run_fill_simulation_loop",
            lambda **kwargs: ([0], [kwargs["container_transform"]]),
        )
        monkeypatch.setattr(
            "scenesmith.manipuland_agents.tools.fill_tools."
            "compute_composite_bbox_in_local_frame",
            lambda **_: (np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2])),
        )
        result = json.loads(
            tools._fill_container_impl(
                container_asset_id=str(prior.object_id),
                fill_asset_ids=[str(prior.object_id)],
                surface_id="S_0",
                position_x=0.0,
                position_z=0.0,
            )
        )
        composite_id = result["filled_container_id"]
        composite = scene.get_object(UniqueID(composite_id))
        member_records = [
            composite.metadata["container_asset"],
            *composite.metadata["fill_assets"],
        ]

    assert result["success"] is True
    assert len(member_records) == 2
    assert {record["asset_id"] for record in member_records} == {
        str(prior.object_id)
    }
    composite = scene.get_object(UniqueID(composite_id))
    if composite_kind == "filled_container":
        assert composite.metadata["num_fill_objects"] == 1
    else:
        assert composite.metadata["num_members"] == 2
    assert scene.get_object(prior.object_id) is prior
    assert json.dumps(
        scene.to_state_dict()["objects"][str(prior.object_id)], sort_keys=True
    ) == prior_before
    assert geometry.read_bytes() == geometry_before
    assert sdf.read_bytes() == sdf_before


@pytest.mark.parametrize("protected_as_composite", [False, True])
def test_rescale_rejects_sdf_shared_with_protected_scope(
    tmp_path: Path,
    protected_as_composite: bool,
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    shared_sdf = tmp_path / "shared.sdf"
    original = b"<sdf version='1.7'/>"
    shared_sdf.write_bytes(original)
    furniture = _furniture("cabinet_0", _surface("S_0"))
    prior = _manipuland("prior_0", "S_0", geometry)
    current = _manipuland("current_0", "S_0", geometry)
    current.sdf_path = shared_sdf
    if protected_as_composite:
        prior.sdf_path = tmp_path / "prior_composite.sdf"
        prior.metadata = {
            "composite_type": "stack",
            "member_assets": [{"sdf_path": str(shared_sdf)}],
        }
    else:
        prior.sdf_path = shared_sdf
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (furniture, prior, current)},
    )
    tools = _tools(scene, furniture, {str(prior.object_id)})
    result = json.loads(tools._rescale_manipuland_impl("current_0", 1.2))
    assert result["error_type"] == "immutable_object"
    assert shared_sdf.read_bytes() == original


def test_unshared_current_manipuland_rescale_remains_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    sdf = tmp_path / "current.sdf"
    sdf.write_text("<sdf version='1.7'/>", encoding="utf-8")
    furniture = _furniture("cabinet_0", _surface("S_0"))
    current = _manipuland("current_0", "S_0", geometry)
    current.sdf_path = sdf
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (furniture, current)},
    )
    tools = _tools(scene, furniture, set())
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.tools.manipuland_tools.rescale_object_common",
        lambda **_: SimpleNamespace(
            to_json=lambda: json.dumps({"success": True, "object_id": "current_0"})
        ),
    )
    assert json.loads(tools._rescale_manipuland_impl("current_0", 1.1))["success"]


def test_floor_vision_keeps_furniture_context_but_only_current_floor_items(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    floor = SceneObject(
        object_id=UniqueID("floor_classroom_01"),
        object_type=ObjectType.FLOOR,
        name="floor",
        description="floor",
        transform=RigidTransform(),
        support_surfaces=[_surface("S_floor", z=0.0)],
    )
    desk = _furniture("desk_0", _surface("S_desk"))
    cabinet = _furniture("cabinet_0", _surface("S_cabinet"))
    prior = _manipuland("prior_floor_item_0", "S_floor", geometry)
    current = _manipuland("current_floor_item_0", "S_floor", geometry)
    scene = RoomScene(
        room_geometry=SimpleNamespace(floor=floor),
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (desk, cabinet, prior, current)},
    )
    rendering_manager = MagicMock()
    rendering_manager.render_scene.return_value = tmp_path
    cfg = SimpleNamespace(
        rendering=SimpleNamespace(
            annotations=SimpleNamespace(enable_support_surface_debug=False)
        )
    )
    vision = ManipulandVisionTools(
        scene=scene,
        rendering_manager=rendering_manager,
        cfg=cfg,
        current_furniture_id=floor.object_id,
        blender_server=MagicMock(),
        protected_object_ids={str(prior.object_id)},
    )
    vision._observe_scene_impl()
    included = set(rendering_manager.render_scene.call_args.kwargs["include_objects"])
    assert included == {desk.object_id, cabinet.object_id, current.object_id}
    assert prior.object_id not in included


def test_floor_postprocessing_uses_furniture_context_and_exact_new_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    floor = SceneObject(
        object_id=UniqueID("floor_classroom_01"),
        object_type=ObjectType.FLOOR,
        name="floor",
        description="floor",
        transform=RigidTransform(),
        support_surfaces=[_surface("S_floor", z=0.0)],
    )
    desk = _furniture("desk_0", _surface("S_desk"))
    cabinet = _furniture("cabinet_0", _surface("S_cabinet"))
    prior = _manipuland("prior_floor_item_0", "S_floor", geometry)
    current = _manipuland("current_floor_item_0", "S_floor", geometry)
    scene = RoomScene(
        room_geometry=SimpleNamespace(floor=floor),
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (desk, cabinet, prior, current)},
    )
    captured: dict[str, RoomScene] = {}

    def fake_postprocess(*, scene: RoomScene, **kwargs):
        captured["scene"] = scene
        return scene, True, []

    monkeypatch.setattr(
        physical_feasibility,
        "apply_physical_feasibility_postprocessing",
        fake_postprocess,
    )
    config = SimpleNamespace(
        projection=SimpleNamespace(
            enabled=False,
            influence_distance=0.1,
            solver_name="snopt",
            iteration_limit=1,
            time_limit_s=1.0,
            xy_only=True,
            fix_rotation=True,
        ),
        simulation=SimpleNamespace(
            enabled=False,
            simulation_time_s=0.0,
            time_step_s=0.01,
            timeout_s=1.0,
            remove_fallen_manipulands=False,
            fallen_manipuland_floor_z=-1.0,
            fallen_manipuland_near_floor_z=0.01,
            fallen_manipuland_z_displacement=1.0,
        ),
    )
    physical_feasibility.apply_per_furniture_postprocessing(
        full_scene=scene,
        furniture_id=floor.object_id,
        manipuland_ids=[current.object_id],
        config=config,
    )
    scoped_ids = set(captured["scene"].objects)
    assert scoped_ids == {desk.object_id, cabinet.object_id, current.object_id}
    assert floor.object_id not in scoped_ids
    assert prior.object_id not in scoped_ids


def test_physics_filter_uses_explicit_current_scope_not_shared_surface_id(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "item.gltf"
    geometry.write_text("{}", encoding="utf-8")
    furniture = _furniture("filing_cabinet_0", _surface("S_0"))
    prior = _manipuland("prior_0", "S_0", geometry)
    current = _manipuland("current_0", "S_0", geometry)
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (furniture, prior, current)},
    )
    prior_collision = CollisionPair(
        object_a_name=prior.name,
        object_a_id=str(prior.object_id),
        object_b_name=furniture.name,
        object_b_id=str(furniture.object_id),
        penetration_depth=0.1,
    )
    current_collision = CollisionPair(
        object_a_name=current.name,
        object_a_id=str(current.object_id),
        object_b_name=furniture.name,
        object_b_id=str(furniture.object_id),
        penetration_depth=0.1,
    )
    filtered = filter_collisions_by_agent(
        collisions=[prior_collision, current_collision],
        scene=scene,
        agent_type=AgentType.MANIPULAND,
        current_furniture_id=furniture.object_id,
        manipuland_scope_ids={current.object_id},
    )
    assert filtered == [current_collision]


def test_asset_router_forced_candidate_collision_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asset_router_module.time, "time_ns", lambda: 123)
    monkeypatch.setattr(
        asset_router_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="fixed"),
    )
    existing = tmp_path / "pencil_primitive_supply_123_fixed.glb"
    original = b"protected prior asset"
    existing.write_bytes(original)
    router = asset_router_module.AssetRouter.__new__(asset_router_module.AssetRouter)
    result = router._create_primitive_manipuland_fallback_geometry(
        item=SimpleNamespace(
            short_name="pencil",
            description="yellow pencil",
            dimensions=[0.18, 0.01, 0.01],
        ),
        geometry_dir=tmp_path,
    )
    assert result is None
    assert existing.read_bytes() == original


def test_asset_manager_repeated_candidate_uses_distinct_reserved_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scenesmith.agent_utils.asset_manager as asset_manager_module

    manager = AssetManager.__new__(AssetManager)
    manager.images_dir = tmp_path / "images"
    manager.geometry_dir = tmp_path / "geometry"
    manager.sdf_dir = tmp_path / "sdf"
    uuid_values = iter(
        [SimpleNamespace(hex="first"), SimpleNamespace(hex="second")]
    )
    monkeypatch.setattr(asset_manager_module.time, "time_ns", lambda: 456)
    monkeypatch.setattr(asset_manager_module.uuid, "uuid4", lambda: next(uuid_values))
    first = manager._create_asset_paths(["book"], ["book"])[0]
    second = manager._create_asset_paths(["book"], ["book"])[0]
    assert first.sdf_dir != second.sdf_dir
    assert first.sdf_dir.is_dir()
    assert second.sdf_dir.is_dir()


def _strict_registry(tmp_path: Path) -> tuple[AssetRegistry, Path, Path]:
    namespace = tmp_path / "generated_assets" / "manipuland"
    namespace.mkdir(parents=True, exist_ok=True)
    path = namespace / "asset_registry.json"
    registry = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    return registry, namespace, path


def test_strict_registry_fresh_save_restart_snapshot_and_head_tamper(
    tmp_path: Path,
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    pristine = registry.snapshot()
    registry.verify_snapshot(pristine)
    asset = _registry_asset(namespace, "book_0")
    registry.register(asset)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["assets"]["book_0"]["sdf_path"].startswith("sdf/")
    assert not Path(document["assets"]["book_0"]["sdf_path"]).is_absolute()

    restarted = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    restarted.load_from_file(path)
    assert restarted.get(UniqueID("book_0")).sdf_path == asset.sdf_path
    baseline = restarted.snapshot()
    restarted.verify_snapshot(baseline)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["revision"] += 1
    payload = {key: value for key, value in tampered.items() if key != "attestation"}
    tampered["attestation"] = asset_registry_module._canonical_payload_sha256(payload)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="differs from memory|lineage|predecessor"
    ):
        restarted.verify_persisted()


def _scope_journal(tmp_path: Path) -> Path:
    scene_states = tmp_path / "scene_states"
    scene_states.mkdir(exist_ok=True)
    return scene_states / ".manipuland_registry_scope_transaction.json"


def _scope_context() -> dict[str, object]:
    return {"furniture_id": "filing_cabinet_0", "furniture_index": 1}


def test_registry_scope_failure_restores_exact_registry_tree_and_action_log(
    tmp_path: Path,
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    baseline_asset = _registry_asset(namespace, "teacher_book_0")
    registry.register(baseline_asset)
    unrelated = namespace / "debug" / "preexisting.txt"
    unrelated.parent.mkdir(exist_ok=True)
    unrelated.write_bytes(b"preserve me")
    action_log = tmp_path / "action_log.json"
    action_before = b'[{"step_number": 1, "action": "teacher"}]\n'
    action_log.write_bytes(action_before)
    registry_before = path.read_bytes()
    snapshot_before = registry.snapshot()

    transaction = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
        action_log_path=action_log,
    )
    failed_asset = _registry_asset(namespace, "failed_folder_0")
    failed_namespace = failed_asset.sdf_path.parent
    registry.register(failed_asset)
    (namespace / "debug" / "scope-only.txt").write_bytes(b"discard me")
    action_log.write_bytes(
        b'[{"step_number": 1}, {"step_number": 2, "action": "failed"}]\n'
    )

    transaction.rollback()

    assert path.read_bytes() == registry_before
    assert registry.snapshot() == snapshot_before
    assert registry.get(UniqueID("failed_folder_0")) is None
    assert not failed_namespace.exists()
    assert unrelated.read_bytes() == b"preserve me"
    assert action_log.read_bytes() == action_before
    assert not _scope_journal(tmp_path).exists()


def test_registry_scope_commit_retains_assets_and_action_log(tmp_path: Path) -> None:
    registry, namespace, _ = _strict_registry(tmp_path)
    registry.register(_registry_asset(namespace, "teacher_book_0"))
    action_log = tmp_path / "action_log.json"
    action_log.write_text("[]\n", encoding="utf-8")
    transaction = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
        action_log_path=action_log,
    )
    retained = _registry_asset(namespace, "retained_folder_0")
    registry.register(retained)
    action_after = b'[{"step_number": 1}]\n'
    action_log.write_bytes(action_after)

    transaction.commit()

    assert registry.get(retained.object_id) is not None
    assert retained.sdf_path.parent.is_dir()
    assert action_log.read_bytes() == action_after
    assert not _scope_journal(tmp_path).exists()


def test_registry_scope_crash_journal_rolls_back_before_retry(tmp_path: Path) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    registry.register(_registry_asset(namespace, "teacher_book_0"))
    registry_before = path.read_bytes()
    action_log = tmp_path / "action_log.json"
    action_before = b'[{"step_number": 1}]\n'
    action_log.write_bytes(action_before)
    crashed = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
        action_log_path=action_log,
    )
    failed = _registry_asset(namespace, "crashed_folder_0")
    registry.register(failed)
    action_log.write_text('[{"step_number": 1}, {"step_number": 2}]\n')
    # Model abrupt process loss: the kernel releases the lock, but the durable
    # active journal remains for the next process.
    crashed._release()

    restarted = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    restarted.load_from_file(path)
    retry = restarted.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
        action_log_path=action_log,
    )
    assert path.read_bytes() == registry_before
    assert restarted.get(failed.object_id) is None
    assert not failed.sdf_path.parent.exists()
    assert action_log.read_bytes() == action_before
    retry.rollback()


def test_registry_scope_keyboard_interrupt_like_cleanup_is_transactional(
    tmp_path: Path,
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    registry.register(_registry_asset(namespace, "teacher_book_0"))
    before = path.read_bytes()
    transaction = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
    )
    failed = _registry_asset(namespace, "interrupted_folder_0")
    registry.register(failed)
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        finally:
            transaction.rollback()
    assert path.read_bytes() == before
    assert registry.get(failed.object_id) is None


def test_registry_scope_refuses_to_erase_changed_preexisting_file(
    tmp_path: Path,
) -> None:
    registry, namespace, _ = _strict_registry(tmp_path)
    registry.register(_registry_asset(namespace, "teacher_book_0"))
    protected = namespace / "debug" / "preexisting.txt"
    protected.parent.mkdir(exist_ok=True)
    protected.write_bytes(b"before")
    transaction = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
    )
    protected.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed preexisting file"):
        transaction.rollback()
    assert protected.read_bytes() == b"tampered"
    assert _scope_journal(tmp_path).is_file()


def test_revalidated_checkpoint_can_commit_crash_journal(tmp_path: Path) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    registry.register(_registry_asset(namespace, "teacher_book_0"))
    transaction = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
    )
    retained = _registry_asset(namespace, "checkpoint_folder_0")
    registry.register(retained)
    transaction._release()

    restarted = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    restarted.load_from_file(path)
    assert restarted.commit_recovered_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
    )
    assert restarted.get(retained.object_id) is not None
    assert not _scope_journal(tmp_path).exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_registry_scope_rollback_refuses_link_hazards(
    link_kind: str,
    tmp_path: Path,
) -> None:
    registry, namespace, _ = _strict_registry(tmp_path)
    registry.register(_registry_asset(namespace, "teacher_book_0"))
    transaction = registry.begin_scope_transaction(
        journal_path=_scope_journal(tmp_path),
        context=_scope_context(),
    )
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside must survive")
    hazard = namespace / "debug" / "scope-hazard.txt"
    hazard.parent.mkdir(exist_ok=True)
    try:
        if link_kind == "symlink":
            hazard.symlink_to(outside)
        else:
            os.link(outside, hazard)
    except OSError as exc:
        transaction._release()
        pytest.skip(f"{link_kind} unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|linked or special"):
        transaction.rollback()
    assert outside.read_bytes() == b"outside must survive"
    assert hazard.exists() or hazard.is_symlink()
    assert _scope_journal(tmp_path).is_file()


def test_registry_snapshot_and_loaded_definition_are_deeply_immutable(
    tmp_path: Path,
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    asset = _registry_asset(namespace, "book_0")
    registry.register(asset)
    baseline = registry.snapshot()
    registry._assets[asset.object_id].metadata["nested"]["value"] = 2
    with pytest.raises(RuntimeError, match="mutated baseline definition"):
        registry.verify_snapshot(baseline)
    registry._assets[asset.object_id].metadata["nested"]["value"] = 1

    restarted = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    restarted.load_from_file(path)
    restarted._assets[asset.object_id].metadata["nested"]["value"] = 3
    with pytest.raises(RuntimeError, match="Immutable registry entry changed"):
        restarted.save_to_file(path)
    restarted._assets[asset.object_id].metadata["nested"]["value"] = 1
    asset.geometry_path.write_bytes(b"changed")
    with pytest.raises(
        RuntimeError,
        match="asset bytes changed|file inventory|glTF is invalid",
    ):
        restarted.verify_snapshot(baseline)


def test_registry_register_write_failure_rolls_back_memory_and_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    first = _registry_asset(namespace, "book_0")
    registry.register(first)
    before = path.read_bytes()
    second = _registry_asset(namespace, "folder_0")
    monkeypatch.setattr(asset_registry_module.os, "replace", MagicMock(side_effect=OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        registry.register(second)
    assert registry.get(second.object_id) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["duplicate_key", "nan", "outside", "glb"])
def test_strict_registry_rejects_malformed_or_unsafe_documents(
    corruption: str, tmp_path: Path
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    asset = _registry_asset(namespace, "book_0")
    registry.register(asset)
    raw = path.read_text(encoding="utf-8")
    if corruption == "duplicate_key":
        raw = raw.replace('"revision": 1,', '"revision": 1, "revision": 1,')
    else:
        document = json.loads(raw)
        entry = document["assets"]["book_0"]
        if corruption == "nan":
            entry["metadata"]["bad"] = float("nan")
        elif corruption == "outside":
            entry["geometry_path"] = "../../outside.gltf"
        else:
            entry["geometry_path"] = entry["geometry_path"].replace(".gltf", ".glb")
        payload = {key: value for key, value in document.items() if key != "attestation"}
        document["attestation"] = asset_registry_module._canonical_payload_sha256(payload)
        raw = json.dumps(document, allow_nan=True)
    path.write_text(raw, encoding="utf-8")
    candidate = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    with pytest.raises(RuntimeError):
        candidate.load_from_file(path)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink", "empty"])
def test_strict_registry_rejects_linked_or_empty_asset_files(
    link_kind: str, tmp_path: Path
) -> None:
    registry, namespace, _ = _strict_registry(tmp_path)
    asset = _registry_asset(namespace, "book_0")
    geometry = asset.geometry_path
    original = geometry.read_bytes()
    geometry.unlink()
    if link_kind == "symlink":
        target = geometry.with_name("target.gltf")
        target.write_bytes(original)
        try:
            geometry.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"symlink unavailable: {exc}")
    elif link_kind == "hardlink":
        target = geometry.with_name("target.gltf")
        target.write_bytes(original)
        try:
            import os

            os.link(target, geometry)
        except OSError as exc:
            pytest.skip(f"hardlink unavailable: {exc}")
    else:
        geometry.write_bytes(b"")
    with pytest.raises(RuntimeError):
        registry.register(asset)
    assert registry.size() == 0


def test_loaded_registry_template_sdf_cannot_be_rescaled_through_placed_clone(
    tmp_path: Path,
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    template = _registry_asset(namespace, "book_0")
    registry.register(template)
    restarted = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    restarted.load_from_file(path)
    furniture = _furniture("cabinet_0", _surface("S_0"))
    clone = _manipuland("book_instance_0", "S_0", template.geometry_path)
    clone.sdf_path = template.sdf_path
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (furniture, clone)},
    )
    tools = _tools(scene, furniture, set())
    tools.asset_manager = SimpleNamespace(registry=restarted)
    before = template.sdf_path.read_bytes()
    result = json.loads(tools._rescale_manipuland_impl(str(clone.object_id), 1.2))
    assert result["error_type"] == "immutable_object"
    assert template.sdf_path.read_bytes() == before


def test_asset_manager_constructor_reloads_validated_unbound_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scenesmith.agent_utils.asset_manager as asset_manager_module

    cfg = OmegaConf.load(
        Path(__file__).parents[2]
        / "configurations/manipuland_agent/base_manipuland_agent.yaml"
    )
    cfg.asset_manager.router.enabled = False
    cfg.asset_manager.router.strategies.articulated.enabled = False
    cfg.asset_manager.router.strategies.thin_covering.enabled = False
    monkeypatch.setattr(asset_manager_module, "create_image_generator", MagicMock())
    monkeypatch.setattr(asset_manager_module, "HssdRetrievalClient", MagicMock())
    logger = SimpleNamespace(output_dir=tmp_path)
    first = AssetManager(
        logger=logger,
        vlm_service=MagicMock(),
        blender_server=None,
        collision_client=None,
        cfg=cfg,
        agent_type=AgentType.MANIPULAND,
    )
    namespace = tmp_path / "generated_assets" / "manipuland"
    template = _registry_asset(namespace, "book_0")
    first.registry.register(template)

    restarted = AssetManager(
        logger=logger,
        vlm_service=MagicMock(),
        blender_server=None,
        collision_client=None,
        cfg=cfg,
        agent_type=AgentType.MANIPULAND,
    )
    assert restarted.get_asset_by_id(template.object_id).sdf_path == template.sdf_path
    assert [asset.object_id for asset in restarted.list_available_assets()] == [
        template.object_id
    ]


def test_checkpoint_leaf_reconciliation_is_template_only_and_deduplicates_instances(
    tmp_path: Path,
) -> None:
    registry, namespace, _ = _strict_registry(tmp_path)
    source = _registry_asset(namespace, "source_book_0")
    first_leaf = _manipuland("book_instance_0", "S_0", source.geometry_path)
    first_leaf.sdf_path = source.sdf_path
    first_leaf.name = source.name
    first_leaf.description = source.description
    first_leaf.metadata = json.loads(json.dumps(source.metadata))
    first_leaf.bbox_min = source.bbox_min.copy()
    first_leaf.bbox_max = source.bbox_max.copy()
    second_leaf = _manipuland("book_instance_1", "S_0", source.geometry_path)
    second_leaf.sdf_path = source.sdf_path
    second_leaf.name = source.name
    second_leaf.description = source.description
    second_leaf.metadata = json.loads(json.dumps(source.metadata))
    second_leaf.bbox_min = source.bbox_min.copy()
    second_leaf.bbox_max = source.bbox_max.copy()
    # Live checkpoint state stores portable paths relative to scene_dir.  Registry
    # definitions are absolute after strict restart, so reconciliation must bind
    # these against the scene directory rather than the process working directory.
    first_leaf.geometry_path = first_leaf.geometry_path.relative_to(tmp_path)
    first_leaf.sdf_path = first_leaf.sdf_path.relative_to(tmp_path)
    second_leaf.geometry_path = second_leaf.geometry_path.relative_to(tmp_path)
    second_leaf.sdf_path = second_leaf.sdf_path.relative_to(tmp_path)
    scene = RoomScene(
        room_geometry=None,
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in (first_leaf, second_leaf)},
    )
    manager = AssetManager.__new__(AssetManager)
    manager.agent_type = AgentType.MANIPULAND
    manager.registry = registry
    before = scene.to_state_dict()
    manager.reconcile_registry_with_scene(
        scene=scene,
        leaf_object_ids={first_leaf.object_id, second_leaf.object_id},
    )
    assert registry.size() == 1
    assert scene.to_state_dict() == before
    template = registry.list_all()[0]
    assert template.transform.GetAsMatrix4().tolist() == RigidTransform().GetAsMatrix4().tolist()
    assert template.placement_info is None
    assert template.geometry_path == source.geometry_path.resolve()
    assert template.sdf_path == source.sdf_path.resolve()


@pytest.mark.parametrize("source", ["objathor", "artiverse"])
def test_articulated_manipuland_registers_one_exclusive_sdf_namespace(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scenesmith.agent_utils.asset_manager as asset_manager_module

    registry, namespace, path = _strict_registry(tmp_path)
    source_dir = tmp_path / "publisher_articulated"
    source_dir.mkdir()
    source_sdf = source_dir / "cabinet.sdf"
    source_sdf.write_text(
        "<sdf version='1.7'><model name='cabinet'><link name='base'/></model></sdf>",
        encoding="utf-8",
    )

    class _CombinedMesh:
        def export(self, output: Path) -> None:
            output.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        asset_manager_module,
        "combine_sdf_meshes_at_joint_angles",
        lambda *_args, **_kwargs: _CombinedMesh(),
    )
    monkeypatch.setattr(
        asset_manager_module,
        "normalize_copied_artiverse_visuals",
        lambda _sdf, directory: {
            "copied_tree_sha256_after": (
                asset_manager_module._sha256_directory_tree(directory)
            )
        },
    )
    manager = AssetManager.__new__(AssetManager)
    manager.agent_type = AgentType.MANIPULAND
    manager.geometry_dir = namespace / "geometry"
    manager.sdf_dir = namespace / "sdf"
    manager.geometry_dir.mkdir(exist_ok=True)
    manager.registry = registry
    manager.cfg = SimpleNamespace(
        asset_manager=SimpleNamespace(
            articulated=SimpleNamespace(enable_self_collision_filtering=False)
        )
    )
    scene_object = manager._convert_articulated_to_scene_object(
        articulated=SimpleNamespace(
            item=SimpleNamespace(
                short_name="filing_cabinet",
                description="articulated filing cabinet",
            ),
            sdf_path=source_sdf,
            source=source,
            object_id="publisher-cabinet-0",
            bounding_box_min=[-0.5, -0.3, 0.0],
            bounding_box_max=[0.5, 0.3, 1.2],
        ),
        request=SimpleNamespace(object_type=ObjectType.MANIPULAND),
    )
    assert scene_object.geometry_path.parent == scene_object.sdf_path.parent
    assert scene_object.sdf_path.parent.parent == namespace / "sdf"
    restarted = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    restarted.load_from_file(path)
    assert restarted.get(scene_object.object_id).geometry_path == (
        scene_object.geometry_path.resolve()
    )


@pytest.mark.parametrize(
    ("manager_agent_type", "request_object_type"),
    [
        (None, ObjectType.MANIPULAND),
        (AgentType.FURNITURE, ObjectType.MANIPULAND),
        (AgentType.MANIPULAND, ObjectType.FURNITURE),
    ],
    ids=[
        "missing-manager-manipuland-request",
        "furniture-manager-manipuland-request",
        "manipuland-manager-generic-request",
    ],
)
def test_articulated_manager_request_scope_mismatch_has_no_filesystem_effect(
    manager_agent_type: AgentType | None,
    request_object_type: ObjectType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scenesmith.agent_utils.asset_manager as asset_manager_module

    manager = AssetManager.__new__(AssetManager)
    if manager_agent_type is not None:
        manager.agent_type = manager_agent_type
    manager.geometry_dir = tmp_path / "geometry"
    manager.sdf_dir = tmp_path / "sdf"
    manager.geometry_dir.mkdir()
    manager.sdf_dir.mkdir()
    manager.registry = MagicMock()
    copytree = MagicMock(side_effect=AssertionError("copytree must not run"))
    monkeypatch.setattr(asset_manager_module.shutil, "copytree", copytree)

    with pytest.raises(RuntimeError, match="manipuland scope mismatch"):
        manager._convert_articulated_to_scene_object(
            articulated=SimpleNamespace(
                item=SimpleNamespace(short_name="cabinet"),
                sdf_path=tmp_path / "source" / "cabinet.sdf",
            ),
            request=SimpleNamespace(object_type=request_object_type),
        )

    copytree.assert_not_called()
    manager.registry.register.assert_not_called()
    assert list(manager.geometry_dir.iterdir()) == []
    assert list(manager.sdf_dir.iterdir()) == []


@pytest.mark.parametrize("failure", [False, True], ids=["success", "exception"])
def test_manipuland_stage_always_cleans_up_helper_servers(failure: bool) -> None:
    from scenesmith.experiments.indoor_scene_generation import (
        _run_manipuland_agent_with_cleanup,
    )

    events: list[str] = []

    class _Agent:
        async def add_manipulands(self, *, scene) -> None:
            assert scene == "scene"
            events.append("run")
            if failure:
                raise RuntimeError("injected generation failure")

        def cleanup(self) -> None:
            events.append("cleanup")

    if failure:
        with pytest.raises(RuntimeError, match="injected generation failure"):
            _run_manipuland_agent_with_cleanup(_Agent(), "scene")
    else:
        _run_manipuland_agent_with_cleanup(_Agent(), "scene")
    assert events == ["run", "cleanup"]


def test_exact_legacy_registry_migrates_once_and_pins_baseline_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, namespace, path = _strict_registry(tmp_path)
    ids = sorted(asset_registry_module._LEGACY_INTERRUPTED_REGISTRY_IDS)
    assets = {object_id: _registry_asset(namespace, object_id) for object_id in ids}
    legacy = {
        object_id: registry._serialize_asset(asset)
        for object_id, asset in assets.items()
    }
    path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
    raw_sha = asset_registry_module._sha256_file(path)
    normalized = {
        object_id: registry._serialize_asset(asset, root=namespace.resolve())
        for object_id, asset in assets.items()
    }
    monkeypatch.setattr(
        asset_registry_module, "_LEGACY_INTERRUPTED_REGISTRY_SHA256", raw_sha
    )
    monkeypatch.setattr(
        asset_registry_module,
        "_LEGACY_INTERRUPTED_ENTRY_SHA256",
        {
            object_id: asset_registry_module._canonical_payload_sha256(record)
            for object_id, record in normalized.items()
        },
    )
    monkeypatch.setattr(
        asset_registry_module,
        "_LEGACY_INTERRUPTED_AGGREGATE_SHA256",
        asset_registry_module._canonical_payload_sha256(normalized),
    )
    monkeypatch.setattr(
        asset_registry_module,
        "_LEGACY_INTERRUPTED_DIRECTORY_MANIFESTS",
        {
            object_id: registry._legacy_directory_manifest(
                asset.sdf_path.parent, root=namespace.resolve()
            )
            for object_id, asset in assets.items()
        },
    )
    registry.load_from_file(path)
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 2
    assert migrated["lineage_root"] == raw_sha
    assert set(migrated["assets"]) == set(ids)
    assert registry.size() == 4

    migrated["assets"][ids[0]]["description"] = "forged"
    payload = {key: value for key, value in migrated.items() if key != "attestation"}
    migrated["attestation"] = asset_registry_module._canonical_payload_sha256(payload)
    path.write_text(json.dumps(migrated), encoding="utf-8")
    candidate = AssetRegistry(
        auto_save_path=path,
        required_root=namespace,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    with pytest.raises(RuntimeError, match="baseline semantics changed"):
        candidate.load_from_file(path)
