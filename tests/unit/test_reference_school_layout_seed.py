from __future__ import annotations

from copy import deepcopy
import json
import shutil

from scripts import seed_reference_school_layout as seed


EXPECTED_IDS = {
    "classroom_01",
    "classroom_02",
    "classroom_03",
    "classroom_04",
    "classroom_05",
    "classroom_06",
    "library",
    "boys_toilet",
    "girls_toilet",
    "storage_room",
    "main_corridor",
}


def _overlap(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def test_reference_seed_has_exact_nonoverlapping_room_contract() -> None:
    evidence = seed._seed_spec_evidence()
    assert evidence["profile"] == "school_reference_20260710"
    assert len(evidence["room_ids"]) == 11
    assert set(evidence["room_ids"]) == EXPECTED_IDS
    bounds = evidence["room_bounds"]
    ids = sorted(bounds)
    assert all(
        _overlap(bounds[first], bounds[second]) == 0.0
        for index, first in enumerate(ids)
        for second in ids[index + 1 :]
    )
    assert bounds["library"] == [10.0, -9.0, 20.0, 0.0]
    assert bounds["main_corridor"] == [9.0, 0.0, 21.0, 22.5]
    assert bounds["boys_toilet"] == [1.0, 7.5, 5.0, 11.5]
    assert bounds["girls_toilet"] == [5.0, 7.5, 9.0, 11.5]


def test_reference_seed_binds_every_real_opening_role() -> None:
    evidence = seed._seed_spec_evidence()
    assert len(evidence["interior_door_ids"]) == 10
    assert len(set(evidence["interior_door_ids"])) == 10
    assert evidence["window_rooms"] == [
        "classroom_01",
        "classroom_02",
        "classroom_03",
        "classroom_04",
        "classroom_05",
        "classroom_06",
        "library",
    ]
    assert evidence["entrance"] == {
        "id": "main_entrance",
        "room": "library",
        "wall": "south",
        "position_exact": 4.1,
        "width": 1.8,
        "height": 2.2,
        "leaf_count": 2,
    }
    assert seed._publication_names()[-1] == "house_layout.json"
    assert {
        f"room_{room_id}" for room_id in EXPECTED_IDS
    } <= set(seed._publication_names())


def test_transaction_preserves_final_scene_depth(tmp_path) -> None:
    scene = tmp_path / "repo" / "outputs" / "date" / "run" / "scene_000"
    transaction = seed._transaction_path(scene, pid=123, nonce="abc")
    staged_scene = transaction / scene.name

    assert transaction.parent == scene.parent.parent
    assert staged_scene.parent != scene.parent
    assert len(staged_scene.parts) == len(scene.parts)


def test_gltf_local_uri_manifest_rejects_staging_depth_regression(tmp_path) -> None:
    repo = tmp_path / "repo"
    texture = repo / "materials" / "wood" / "color.jpg"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"texture")
    final_scene = repo / "outputs" / "date" / "run" / "scene_000"
    final_floor = final_scene / "floor_plans" / "classroom_01" / "floors"
    final_floor.mkdir(parents=True)
    (final_scene / "room_geometry").mkdir()
    good_uri = "../../../../../../../materials/wood/color.jpg"
    gltf = final_floor / "floor.gltf"
    gltf.write_text(
        json.dumps({"images": [{"uri": good_uri}], "buffers": []}),
        encoding="utf-8",
    )

    manifest = seed._gltf_local_uri_manifest(repo, final_scene)
    assert manifest[0]["uri"] == good_uri
    assert manifest[0]["target_scope"] == "repo"
    assert manifest[0]["target"] == "materials/wood/color.jpg"

    gltf.write_text(
        json.dumps(
            {
                "images": [
                    {"uri": "../../../../../../../../materials/wood/color.jpg"}
                ],
                "buffers": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        seed._gltf_local_uri_manifest(repo, final_scene)
    except seed.SeedError as exc:
        assert "unresolved" in str(exc) or "escapes the repo" in str(exc)
    else:
        raise AssertionError("one-level staging-depth regression was accepted")


def test_gltf_local_uri_manifest_is_publication_path_stable(tmp_path) -> None:
    repo = tmp_path / "repo"
    texture = repo / "materials" / "wood" / "color.jpg"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"texture")

    final_scene = repo / "outputs" / "date" / "run" / "scene_000"
    transaction = seed._transaction_path(final_scene, pid=123, nonce="abc")
    staged_scene = transaction / final_scene.name
    staged_floor = staged_scene / "floor_plans" / "classroom_01" / "floors"
    staged_floor.mkdir(parents=True)
    (staged_scene / "room_geometry").mkdir()
    (staged_floor / "floor.bin").write_bytes(b"mesh")
    staged_gltf = staged_floor / "floor.gltf"
    staged_gltf.write_text(
        json.dumps(
            {
                "buffers": [{"uri": "floor.bin"}],
                "images": [
                    {"uri": "../../../../../../../materials/wood/color.jpg"}
                ],
            }
        ),
        encoding="utf-8",
    )

    staged_manifest = seed._gltf_local_uri_manifest(repo, staged_scene)
    shutil.copytree(staged_scene, final_scene)
    final_manifest = seed._gltf_local_uri_manifest(repo, final_scene)

    assert staged_manifest == final_manifest
    assert staged_manifest[0]["target_scope"] == "scene"
    assert staged_manifest[0]["target"] == (
        "floor_plans/classroom_01/floors/floor.bin"
    )
    assert staged_manifest[1]["target_scope"] == "repo"
    assert staged_manifest[1]["target"] == "materials/wood/color.jpg"


def test_structural_receipt_allows_only_expected_prompt_binding_mutation() -> None:
    document = {
        "house_prompt": "immutable",
        "rooms": [{"id": "classroom_01", "prompt": "planner", "width": 7.5}],
        "placed_rooms": [
            {
                "room_id": "classroom_01",
                "prompt": "planner",
                "position": [0.0, 0.0],
                "width": 9.0,
                "depth": 7.5,
            }
        ],
        "doors": [{"id": "door"}],
    }
    original = seed._structural_layout_sha256(document)
    prompt_bound = deepcopy(document)
    prompt_bound["rooms"][0]["prompt"] = "canonical"
    prompt_bound["placed_rooms"][0]["prompt"] = "canonical"
    assert seed._structural_layout_sha256(prompt_bound) == original

    substituted = deepcopy(prompt_bound)
    substituted["placed_rooms"][0]["width"] = 8.0
    assert seed._structural_layout_sha256(substituted) != original
