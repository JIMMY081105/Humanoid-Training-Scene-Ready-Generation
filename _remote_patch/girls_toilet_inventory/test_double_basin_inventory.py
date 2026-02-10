from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.school_room_contract import ROOM_REQUIREMENTS, evaluate_room_inventory


def _mesh_backed(name: str, description: str) -> dict:
    return {
        "object_id": name,
        "object_type": "furniture",
        "name": name,
        "description": description,
        "geometry_path": "asset.gltf",
        "sdf_path": "asset.sdf",
        "transform": {
            "translation": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "bbox_min": [-0.7, -0.3, 0.0],
        "bbox_max": [0.7, 0.3, 1.0],
        "immutable": False,
        "scale_factor": 1.0,
        "metadata": {},
    }


def _state(tmp_path: Path) -> dict:
    for name in ("asset.gltf", "asset.sdf"):
        (tmp_path / name).write_bytes(b"fixture")
    objects = {
        "toilet_0": _mesh_backed("toilet_0", "school toilet fixture"),
        "toilet_1": _mesh_backed("toilet_1", "school toilet fixture"),
        "mirror_0": _mesh_backed("mirror_0", "portrait restroom mirror"),
        "mirror_1": _mesh_backed("mirror_1", "portrait restroom mirror"),
        "partition_0": _mesh_backed("partition_0", "restroom stall partition"),
        "partition_1": _mesh_backed("partition_1", "restroom stall partition"),
        "soap_0": _mesh_backed("soap_0", "soap dispenser"),
        "dryer_0": _mesh_backed("dryer_0", "hand dryer"),
        "trash_0": _mesh_backed("trash_0", "trash bin"),
        "sanitary_0": _mesh_backed("sanitary_0", "sanitary disposal bin"),
        "sink_vanity_0": _mesh_backed(
            "sink_vanity_0", "light wood double-basin sink vanity"
        ),
    }
    objects["sink_vanity_0"]["metadata"]["dimension_contract"] = {
        "status": "pass",
        "measured_final_scene_dimensions_m": [1.44, 0.60, 1.06],
    }
    return {"objects": objects}


def test_double_basin_counts_as_two_verified_sink_components(tmp_path: Path) -> None:
    state = _state(tmp_path)
    result = evaluate_room_inventory(
        "girls_toilet",
        state,
        ROOM_REQUIREMENTS["girls_toilet"],
        asset_root=tmp_path,
    )
    assert result["counts"]["sinks"] == 2
    assert not any("Multi-basin fixture" in issue for issue in result["critical_issues"])
    record = result["fixture_multiplicity_evidence"]["sink_vanity_0"]
    assert record["component_count"] == 2
    assert record["policy"] == "verified_double_basin_fixture_components"


def test_double_basin_claim_fails_closed_without_valid_dimensions(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["objects"]["sink_vanity_0"]["metadata"]["dimension_contract"]["status"] = "fail"
    result = evaluate_room_inventory(
        "girls_toilet",
        state,
        ROOM_REQUIREMENTS["girls_toilet"],
        asset_root=tmp_path,
    )
    assert result["status"] == "fail"
    assert result["counts"]["sinks"] == 1
    assert any("lacks a passing" in issue for issue in result["critical_issues"])


def test_single_basin_does_not_receive_multiplicity(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["objects"]["sink_vanity_0"]["description"] = "single-basin sink vanity"
    result = evaluate_room_inventory(
        "girls_toilet",
        state,
        ROOM_REQUIREMENTS["girls_toilet"],
        asset_root=tmp_path,
    )
    assert result["status"] == "fail"
    assert result["counts"]["sinks"] == 1
    assert result["fixture_multiplicity_evidence"] == {}
