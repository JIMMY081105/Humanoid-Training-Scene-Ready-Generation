from __future__ import annotations

import json
import importlib.util
import hashlib
import os

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from scripts.validate_articulated_motion import (
    ROLE_MOTION_REQUIREMENTS,
    evaluate,
    run,
    verify_output,
)


ROLE_FIXTURES = {
    "library": {
        "object_id": "library_glass_bookcase",
        "description": "openable library bookcase cabinet with hinged glass doors",
        "source": "artiverse",
        "articulated_id": "artiverse/bookcase/fpModel/100",
        "joints": [("left_glass_door", "revolute")],
    },
    "storage_room": {
        "object_id": "school_supply_utility",
        "description": "school supply utility cabinet with two hinged doors",
        "source": "artvip",
        "articulated_id": "large_furniture/model_storage_utility",
        "joints": [
            ("left_utility_door", "revolute"),
            ("right_utility_door", "revolute"),
        ],
    },
    "classroom_01": {
        "object_id": "teacher_filing_cabinet",
        "description": "teacher filing cabinet with operable drawers",
        "source": "artvip",
        "articulated_id": "large_furniture/model_teacher_filing",
        "joints": [("upper_drawer", "prismatic"), ("lower_drawer", "prismatic")],
    },
}


class FakeDrakeLoader:
    def __init__(self, *, move: bool = True):
        self.move = move
        self.calls: list[tuple[Path, list[dict[str, Any]]]] = []

    def __call__(
        self, sdf_path: Path, requests: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        copied = [dict(request) for request in requests]
        self.calls.append((sdf_path, copied))
        exercises = []
        for request in requests:
            magnitude = 0.25 if self.move else 0.0
            exercises.append(
                {
                    "joint_name": request["joint_name"],
                    "joint_type": request["joint_type"],
                    "model_name": sdf_path.parent.name,
                    "parent_body_name": request["parent_link"],
                    "child_body_name": request["child_link"],
                    "limits": {
                        "lower": request["limits"]["lower"],
                        "upper": request["limits"]["upper"],
                    },
                    "tested_positions": list(request["tested_positions"]),
                    "transform_delta": {
                        "max_abs_matrix_delta": magnitude,
                        "translation_norm": (
                            magnitude if request["joint_type"] == "prismatic" else 0.0
                        ),
                        "rotation_frobenius_norm": (
                            magnitude if request["joint_type"] == "revolute" else 0.0
                        ),
                    },
                    "child_body_pose_changed": self.move,
                }
            )
        return {
            "status": "pass",
            "model_names": [sdf_path.parent.name],
            "num_model_instances": 3,
            "num_bodies": len(exercises) + 2,
            "num_joints": len(exercises),
            "num_positions": len(exercises),
            "joint_exercises": exercises,
        }


def _joint_xml(name: str, joint_type: str, ordinal: int) -> str:
    lower, upper = ("-1.2", "1.2") if joint_type == "revolute" else ("0.0", "0.4")
    axis = "0 0 1" if joint_type == "revolute" else "1 0 0"
    return f"""
      <joint name="{name}" type="{joint_type}">
        <parent>base</parent>
        <child>moving_{ordinal}</child>
        <axis><xyz>{axis}</xyz><limit><lower>{lower}</lower><upper>{upper}</upper></limit></axis>
      </joint>"""


def _write_sdf(asset_dir: Path, joints: list[tuple[str, str]]) -> Path:
    asset_dir.mkdir(parents=True)
    (asset_dir / "mesh.obj").write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n", encoding="ascii"
    )
    inertia = (
        "<inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy>"
        "<izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"
    )
    links = [
        "<link name='base'>"
        + inertia
        + "<visual name='mesh'><geometry><mesh><uri>mesh.obj</uri></mesh></geometry></visual>"
        + "</link>"
    ] + [
        f"<link name='moving_{ordinal}'>{inertia}</link>"
        for ordinal in range(len(joints))
    ]
    joint_xml = "".join(
        _joint_xml(name, joint_type, ordinal)
        for ordinal, (name, joint_type) in enumerate(joints)
    )
    sdf_path = asset_dir / "asset.sdf"
    sdf_path.write_text(
        "<sdf version='1.9'><model name='role_asset'>"
        + "".join(links)
        + joint_xml
        + "</model></sdf>",
        encoding="utf-8",
    )
    return sdf_path


def _setup_scene(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    scene_dir = tmp_path / "scene"
    sdf_paths: dict[str, Path] = {}
    state_paths: dict[str, Path] = {}
    for room_id, fixture in ROLE_FIXTURES.items():
        room_dir = scene_dir / f"room_{room_id}"
        sdf_path = _write_sdf(
            room_dir / "generated_assets" / "sdf" / fixture["object_id"],
            list(fixture["joints"]),
        )
        state_path = room_dir / "scene_states" / "final_scene" / "scene_state.json"
        state_path.parent.mkdir(parents=True)
        state = {
            "objects": {
                fixture["object_id"]: {
                    "object_id": fixture["object_id"],
                    "name": fixture["description"],
                    "description": fixture["description"],
                    "object_type": "FURNITURE",
                    "sdf_path": str(sdf_path.relative_to(room_dir)),
                    "metadata": {
                        "asset_source": "articulated",
                        "is_articulated": True,
                        "articulated_source": fixture["source"],
                        "articulated_id": fixture["articulated_id"],
                    },
                }
            }
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        sdf_paths[room_id] = sdf_path
        state_paths[room_id] = state_path
    return scene_dir, sdf_paths, state_paths


def test_all_three_roles_are_hash_bound_and_really_exercised(tmp_path: Path) -> None:
    scene_dir, _sdfs, _states = _setup_scene(tmp_path)
    output = tmp_path / "articulated_motion.json"
    loader = FakeDrakeLoader()

    result = run(scene_dir, output, drake_loader=loader)

    assert result["status"] == "pass"
    assert set(result["roles"]) == set(ROLE_MOTION_REQUIREMENTS)
    assert result["artiverse_roles"] == ["library_glass_door_bookcase"]
    assert result["drake_motion_exercised"] is True
    assert len(loader.calls) == 3
    storage = result["roles"]["school_supply_two_door_utility_cabinet"]
    assert len(storage["drake_motion"]["joint_exercises"]) == 2
    assert storage["resource_tree"]["file_count"] == 2
    for role in result["roles"].values():
        assert role["sdf"]["sha256"]
        assert role["resource_tree"]["sha256"]
        assert all(
            joint["child_body_pose_changed"]
            for joint in role["drake_motion"]["joint_exercises"]
        )

    verification_output = tmp_path / "articulated_motion_verification.json"
    verification = verify_output(
        scene_dir,
        output,
        drake_loader=loader,
        verification_output=verification_output,
    )
    assert verification["status"] == "pass"
    assert verification["drake_motion_repeated"] is True
    assert verification["source_gate"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert verification["source_gate"]["evidence_sha256"] == result["evidence_sha256"]
    assert verification["recomputed_gate"]["evidence_sha256"] == result["evidence_sha256"]
    assert verification["recomputed_gate"]["role_count"] == 3
    assert json.loads(verification_output.read_text(encoding="utf-8")) == verification
    assert not list(tmp_path.glob(".articulated_motion_verification.json.*.tmp"))
    assert len(loader.calls) == 6


@pytest.mark.skipif(
    importlib.util.find_spec("pydrake") is None,
    reason="pydrake is available only in the SQZ acceptance environment",
)
def test_real_pydrake_loader_changes_child_body_poses(tmp_path: Path) -> None:
    scene_dir, _sdfs, _states = _setup_scene(tmp_path)

    result = evaluate(scene_dir)

    assert result["status"] == "pass"
    assert all(
        exercise["child_body_pose_changed"]
        for role in result["roles"].values()
        for exercise in role["drake_motion"]["joint_exercises"]
    )


def test_role_specific_wrong_joint_type_and_zero_range_fail_closed(tmp_path: Path) -> None:
    scene_dir, sdf_paths, _states = _setup_scene(tmp_path)
    output = tmp_path / "motion.json"
    classroom_sdf = sdf_paths["classroom_01"]
    classroom_sdf.write_text(
        classroom_sdf.read_text(encoding="utf-8").replace(
            'type="prismatic"', 'type="revolute"'
        ),
        encoding="utf-8",
    )

    wrong_type = run(scene_dir, output, drake_loader=FakeDrakeLoader())
    assert wrong_type["status"] == "fail"
    assert "wrong joint type" in wrong_type["critical_issues"][0]

    scene_dir, sdf_paths, _states = _setup_scene(tmp_path / "second")
    library_sdf = sdf_paths["library"]
    library_sdf.write_text(
        library_sdf.read_text(encoding="utf-8").replace(
            "<upper>1.2</upper>", "<upper>-1.2</upper>"
        ),
        encoding="utf-8",
    )
    zero_range = run(scene_dir, output, drake_loader=FakeDrakeLoader())
    assert zero_range["status"] == "fail"
    assert "positive joint range" in zero_range["critical_issues"][0]


def test_drake_load_without_child_pose_motion_fails(tmp_path: Path) -> None:
    scene_dir, _sdfs, _states = _setup_scene(tmp_path)

    with pytest.raises(RuntimeError, match="no child-body motion"):
        evaluate(scene_dir, drake_loader=FakeDrakeLoader(move=False))


def test_sdf_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    scene_dir, sdf_paths, state_paths = _setup_scene(tmp_path)
    state_path = state_paths["library"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    outside = scene_dir / "outside.sdf"
    outside.write_text(sdf_paths["library"].read_text(encoding="utf-8"), encoding="utf-8")
    state["objects"]["library_glass_bookcase"]["sdf_path"] = "../outside.sdf"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes"):
        evaluate(scene_dir, drake_loader=FakeDrakeLoader())

    scene_dir, sdf_paths, state_paths = _setup_scene(tmp_path / "linked")
    target = sdf_paths["library"]
    link = target.with_name("linked.sdf")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable for this Windows test account")
    state_path = state_paths["library"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["objects"]["library_glass_bookcase"]["sdf_path"] = str(
        link.relative_to(scene_dir / "room_library")
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="symlink"):
        evaluate(scene_dir, drake_loader=FakeDrakeLoader())


def test_verify_only_rehashes_assets_and_repeats_motion(tmp_path: Path) -> None:
    scene_dir, sdf_paths, _states = _setup_scene(tmp_path)
    output = tmp_path / "motion.json"
    loader = FakeDrakeLoader()
    assert run(scene_dir, output, drake_loader=loader)["status"] == "pass"
    (sdf_paths["library"].parent / "mesh.obj").write_bytes(b"mutated-resource")

    receipt = tmp_path / "motion_verification.json"
    verification = verify_output(
        scene_dir,
        output,
        drake_loader=loader,
        verification_output=receipt,
    )

    assert verification["status"] == "fail"
    assert verification["drake_motion_repeated"] is True
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "fail"
    assert len(loader.calls) == 6
    assert any("differ" in issue for issue in verification["critical_issues"])


def test_verify_rejects_forged_saved_attestation(tmp_path: Path) -> None:
    scene_dir, _sdfs, _states = _setup_scene(tmp_path)
    output = tmp_path / "motion.json"
    assert run(scene_dir, output, drake_loader=FakeDrakeLoader())["status"] == "pass"
    saved = json.loads(output.read_text(encoding="utf-8"))
    saved["roles"]["library_glass_door_bookcase"]["articulated_source"] = "forged"
    output.write_text(json.dumps(saved), encoding="utf-8")

    verification = verify_output(scene_dir, output, drake_loader=FakeDrakeLoader())

    assert verification["status"] == "fail"
    assert any("attestation" in issue for issue in verification["critical_issues"])
