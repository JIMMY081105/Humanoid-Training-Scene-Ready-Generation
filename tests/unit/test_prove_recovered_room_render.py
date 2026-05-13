from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

from pathlib import Path

import pytest

from scripts import prove_recovered_room_render as proof


def _write(path: Path, value: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_copy_regular_tree_is_byte_exact_and_rejects_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write(source / "a" / "one.bin", b"one")
    _write(source / "two.bin", b"two")
    destination = tmp_path / "destination"

    proof._copy_regular_tree(source, destination)

    before = proof._tree_record(source, "source")
    after = proof._tree_record(destination, "destination")
    assert {
        key: before[key] for key in ("file_count", "total_bytes", "sha256")
    } == {key: after[key] for key in ("file_count", "total_bytes", "sha256")}
    with pytest.raises(proof.ProofError, match="already exists"):
        proof._copy_regular_tree(source, destination)


def test_shadow_output_is_the_exact_production_run_depth(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "outputs").mkdir(parents=True)
    expected = repo / "outputs" / "2026-07-12" / "proof_unique"

    assert proof._validate_shadow_run_destination(repo, expected) == expected
    with pytest.raises(proof.ProofError, match="must itself be a run root"):
        proof._validate_shadow_run_destination(repo, expected / "shadow_run")
    with pytest.raises(proof.ProofError, match="YYYY-MM-DD"):
        proof._validate_shadow_run_destination(
            repo, repo / "outputs" / "preflight" / "proof_unique"
        )


def _write_gltf(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_recursive_gltf_manifest_binds_same_depth_repo_resources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    shadow_run = repo / "outputs" / "2026-07-12" / "proof_unique"
    scene = shadow_run / "scene_000"
    generated = scene / "room_classroom_01" / "generated_assets"
    geometry = scene / "room_geometry"
    floors = scene / "floor_plans"
    texture = repo / "materials" / "wood" / "color.jpg"
    _write(texture, b"texture")

    model = generated / "model.gltf"
    nested = generated / "parts" / "nested.gltf"
    mesh = nested.parent / "nested.bin"
    _write(mesh, b"mesh")
    _write_gltf(model, {"extensions": {"TEST_external": {"uri": "parts/nested.gltf"}}})
    _write_gltf(nested, {"buffers": [{"uri": "nested.bin"}]})
    _write_gltf(geometry / "room.gltf", {"buffers": [{"uri": "data:application/octet-stream;base64,eA=="}]})
    floor = floors / "classroom_01" / "floors" / "floor.gltf"
    texture_uri = os.path.relpath(texture, floor.parent).replace("\\", "/")
    _write_gltf(floor, {"images": [{"uri": texture_uri}]})
    forbidden = repo / "outputs" / "2026-07-12" / "live" / "scene_000"
    forbidden.mkdir(parents=True)

    manifest = proof._gltf_external_uri_manifest(
        repo=repo,
        shadow_run=shadow_run,
        bound_roots=(generated, geometry, floors),
        forbidden_source_roots=(forbidden,),
    )

    assert manifest["gltf_count"] == 4
    assert manifest["external_uri_count"] == 3
    assert {item["uri"] for item in manifest["external_uris"]} == {
        "parts/nested.gltf",
        "nested.bin",
        texture_uri,
    }
    texture_record = next(
        item for item in manifest["external_uris"] if item["uri"] == texture_uri
    )
    assert texture_record["target_scope"] == "repo"
    assert texture_record["target"] == "materials/wood/color.jpg"


def test_gltf_manifest_rejects_extra_shadow_depth_and_live_source_target(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    correct_run = repo / "outputs" / "2026-07-12" / "proof_unique"
    nested_run = correct_run / "shadow_run"
    scene = nested_run / "scene_000"
    generated = scene / "room_classroom_01" / "generated_assets"
    geometry = scene / "room_geometry"
    floors = scene / "floor_plans"
    source_scene = repo / "outputs" / "2026-07-12" / "live" / "scene_000"
    source_mesh = source_scene / "room_geometry" / "live.bin"
    _write(source_mesh, b"live")
    live_uri = os.path.relpath(source_mesh, generated).replace("\\", "/")
    _write_gltf(generated / "model.gltf", {"buffers": [{"uri": live_uri}]})
    _write_gltf(geometry / "room.gltf", {})
    _write_gltf(floors / "floor.gltf", {})

    with pytest.raises(proof.ProofError, match="unbound live source"):
        proof._gltf_external_uri_manifest(
            repo=repo,
            shadow_run=nested_run,
            bound_roots=(generated, geometry, floors),
            forbidden_source_roots=(source_scene,),
        )
    with pytest.raises(proof.ProofError, match="must itself be a run root"):
        proof._validate_shadow_run_destination(repo, nested_run)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation needs privileges")
def test_tree_rejects_symbolic_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "target", b"target")
    (source / "link").symlink_to(source / "target")
    with pytest.raises(proof.ProofError, match="symbolic link"):
        proof._tree_record(source, "source")


def test_rebase_preserves_prompt_and_only_moves_copied_markers(tmp_path: Path) -> None:
    old = tmp_path / "old"
    state = {
        "text_description": "EXACT prompt, do not shorten.",
        "room_geometry": {
            "sdf_path": str(old / "scene_000" / "room_geometry" / "room.sdf"),
            "floor": {
                "geometry_path": str(
                    old
                    / "scene_000"
                    / "floor_plans"
                    / "classroom_01"
                    / "floors"
                    / "floor.gltf"
                )
            },
        },
        "objects": {
            "cabinet_0": {
                "sdf_path": str(
                    old
                    / "scene_000"
                    / "room_classroom_01"
                    / "generated_assets"
                    / "furniture"
                    / "cabinet.sdf"
                ),
                "geometry_path": "generated_assets/furniture/cabinet.glb",
                "metadata": {
                    "articulated_source_sdf_path": "/immutable/data/artiverse/source.sdf"
                },
            }
        },
    }
    shadow_scene = tmp_path / "repo" / "outputs" / "date" / "proof" / "scene_000"
    shadow_room = shadow_scene / "room_classroom_01"

    result = proof._rebase_state_paths(
        state, shadow_room=shadow_room, shadow_scene=shadow_scene
    )

    assert result["text_description"] == state["text_description"]
    assert result["room_geometry"]["sdf_path"] == str(
        shadow_scene / "room_geometry" / "room.sdf"
    )
    assert result["room_geometry"]["floor"]["geometry_path"] == str(
        shadow_scene
        / "floor_plans"
        / "classroom_01"
        / "floors"
        / "floor.gltf"
    )
    assert result["objects"]["cabinet_0"]["sdf_path"] == str(
        shadow_room / "generated_assets" / "furniture" / "cabinet.sdf"
    )
    assert (
        result["objects"]["cabinet_0"]["metadata"][
            "articulated_source_sdf_path"
        ]
        == "/immutable/data/artiverse/source.sdf"
    )
    assert state["objects"]["cabinet_0"]["sdf_path"].startswith(str(old))


def test_sanitized_environment_removes_keys_and_blocks_external_proxy() -> None:
    source = {
        "PATH": "/bin",
        "OPENAI_API_KEY": "secret",
        "ANTHROPIC_API_KEY": "secret2",
        "HTTPS_PROXY": "http://real-proxy:1234",
        "NO_PROXY": "example.com",
    }
    result = proof._sanitized_environment(source)

    assert result["PATH"] == "/bin"
    assert "OPENAI_API_KEY" not in result
    assert "ANTHROPIC_API_KEY" not in result
    assert result["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert result["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert result["HF_HUB_OFFLINE"] == "1"


def test_used_artiverse_visuals_must_be_derived_external_gltf(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    shadow_room = (
        repo
        / "outputs"
        / "2026-07-12"
        / "proof_unique"
        / "scene_000"
        / "room_classroom_01"
    )
    model = shadow_room / "generated_assets" / "furniture" / "cabinet"
    gltf = model / "gltfs" / "door.gltf"
    _write_gltf(gltf, {"buffers": [{"uri": "door.bin"}]})
    _write(gltf.parent / "door.bin", b"mesh")
    _write(model / "objs" / "door.obj", b"collision")
    sdf = model / "model.sdf"
    sdf.write_text(
        "<sdf version='1.9'><model name='cabinet'><link name='door'>"
        "<visual name='visual'><geometry><mesh><uri>./gltfs/door.gltf</uri>"
        "</mesh></geometry></visual><collision name='collision'><geometry><mesh>"
        "<uri>./objs/door.obj</uri></mesh></geometry></collision>"
        "</link></model></sdf>",
        encoding="utf-8",
    )
    state = {
        "objects": {
            "storage_cabinet_0": {
                "sdf_path": sdf.relative_to(shadow_room).as_posix(),
                "metadata": {"articulated_source": "artiverse"},
            }
        }
    }

    records = proof._validate_artiverse_visual_gltf_bindings(
        state, shadow_room=shadow_room, repo=repo, source_roots=()
    )

    assert records == [
        {
            "object_id": "storage_cabinet_0",
            "sdf": sdf.relative_to(shadow_room).as_posix(),
            "visual_index": 0,
            "uri": "./gltfs/door.gltf",
            "target": gltf.relative_to(shadow_room).as_posix(),
            "size_bytes": gltf.stat().st_size,
            "sha256": proof._sha256_file(gltf),
        }
    ]

    document = ET.parse(sdf)
    uri = next(
        element
        for element in document.getroot().iter()
        if str(element.tag).rsplit("}", 1)[-1] == "uri"
    )
    uri.text = "./glbs/door.glb"
    document.write(sdf, encoding="utf-8")
    with pytest.raises(proof.ProofError, match="not the derived external GLTF"):
        proof._validate_artiverse_visual_gltf_bindings(
            state, shadow_room=shadow_room, repo=repo, source_roots=()
        )


def test_render_log_rejects_ignored_glb_mesh_message(tmp_path: Path) -> None:
    clean = tmp_path / "clean.log"
    clean.write_text("loaded derived external mesh door.gltf\n", encoding="utf-8")
    proof._assert_no_ignored_glb_meshes(clean)

    failed = tmp_path / "failed.log"
    failed.write_text(
        "warning: ignored unsupported mesh ./glbs/door.glb\n", encoding="utf-8"
    )
    with pytest.raises(proof.ProofError, match="ignored GLB meshes"):
        proof._assert_no_ignored_glb_meshes(failed)

    drake_warning = tmp_path / "drake-warning.log"
    drake_warning.write_text(
        "RenderEngineGltfClient only supports Mesh specifications which use "
        ".obj or .gltf files\n",
        encoding="utf-8",
    )
    with pytest.raises(proof.ProofError, match="ignored GLB meshes"):
        proof._assert_no_ignored_glb_meshes(drake_warning)


def test_source_path_audit_rejects_nested_live_floor_reference(tmp_path: Path) -> None:
    live_scene = tmp_path / "live_run" / "scene_000"
    shadow_scene = tmp_path / "repo" / "outputs" / "date" / "proof" / "scene_000"
    state = {
        "room_geometry": {
            "sdf_path": str(shadow_scene / "room_geometry" / "room.sdf"),
            "floor": {
                "geometry_path": str(
                    live_scene
                    / "floor_plans"
                    / "classroom_01"
                    / "floors"
                    / "floor.gltf"
                )
            },
        }
    }
    with pytest.raises(proof.ProofError, match="unbound source"):
        proof._audit_no_source_scene_paths(state, source_roots=[live_scene])


def test_source_path_audit_allows_shadow_and_external_immutable_data(
    tmp_path: Path,
) -> None:
    live_scene = tmp_path / "live_run" / "scene_000"
    shadow_scene = tmp_path / "repo" / "outputs" / "date" / "proof" / "scene_000"
    state = {
        "room_geometry": {
            "sdf_path": str(shadow_scene / "room_geometry" / "room.sdf")
        },
        "metadata": {
            "articulated_source_sdf_path": "/immutable/data/artiverse/model.sdf"
        },
    }
    proof._audit_no_source_scene_paths(state, source_roots=[live_scene])


def test_validate_review_evidence_binds_state_blend_and_three_images(
    tmp_path: Path,
) -> None:
    state = tmp_path / "scene_state.json"
    blend = tmp_path / "scene.blend"
    _write(state, b"state")
    _write(blend, b"blend")
    views = []
    for index, name in enumerate(proof.VIEW_NAMES):
        image = tmp_path / f"classroom_01_{name}.png"
        _write(image, f"image-{index}".encode())
        views.append(
            {
                "view_name": name,
                "image": str(image),
                "image_sha256": proof._sha256_file(image),
            }
        )
    evidence = {
        "status": "pass",
        "rendered_view_count": 3,
        "views": views,
        "derivation_receipt": {
            "source_state": proof._file_record(state, "state"),
            "source_blend": proof._file_record(blend, "blend"),
            "attestation": {"sha256": "a" * 64},
        },
    }
    evidence_path = tmp_path / "classroom_01_cutaway_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    result = proof._validate_review_evidence(
        evidence_path, expected_state=state, expected_blend=blend
    )

    assert result["document"]["status"] == "pass"


def test_discover_source_tree_requires_one_root(tmp_path: Path) -> None:
    room = tmp_path / "scene_000" / "room_classroom_01"
    first = room / "generated_assets"
    second = tmp_path / "other" / "generated_assets"
    _write(first / "one", b"one")
    _write(second / "two", b"two")
    state = {
        "objects": {
            "a": {"sdf_path": str(first / "one")},
            "b": {"sdf_path": str(second / "two")},
        }
    }
    with pytest.raises(proof.ProofError, match="exactly one"):
        proof._discover_source_tree(
            state, marker="generated_assets", source_room=room
        )
