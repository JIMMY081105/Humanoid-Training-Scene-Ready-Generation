"""Strict restart/authority regressions for the manipuland asset registry.

These tests deliberately exercise persistence boundaries that ordinary in-process
registry tests do not reach.  Non-manipuland registries retain their historical
session-only compatibility; the strict contract is enabled only by supplying an
explicit MANIPULAND namespace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from pydrake.all import RigidTransform

from scenesmith.agent_utils import asset_registry as registry_module
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID


def _strict_registry(tmp_path: Path) -> tuple[AssetRegistry, Path, Path]:
    root = tmp_path / "generated_assets" / "manipuland"
    (root / "sdf").mkdir(parents=True, exist_ok=True)
    path = root / "asset_registry.json"
    return (
        AssetRegistry(
            auto_save_path=path,
            required_root=root,
            allowed_object_types=frozenset({ObjectType.MANIPULAND}),
        ),
        root,
        path,
    )


def _asset(root: Path, object_id: str) -> SceneObject:
    directory = root / "sdf" / f"{object_id}_namespace"
    directory.mkdir(parents=True, exist_ok=False)
    geometry = directory / f"{object_id}.gltf"
    sdf = directory / f"{object_id}.sdf"
    geometry.write_text("{}\n", encoding="utf-8")
    sdf.write_text(
        "<sdf version='1.7'><model name='asset'><link name='base'/></model></sdf>\n",
        encoding="utf-8",
    )
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.MANIPULAND,
        name=object_id,
        description=f"reusable {object_id}",
        transform=RigidTransform(),
        geometry_path=geometry,
        sdf_path=sdf,
        metadata={"asset_source": "generated", "nested": {"value": 1}},
        bbox_min=np.array([-0.1, -0.1, 0.0]),
        bbox_max=np.array([0.1, 0.1, 0.2]),
    )


def _candidate(root: Path, path: Path) -> AssetRegistry:
    return AssetRegistry(
        auto_save_path=path,
        required_root=root,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )


def _rewrite_attested(path: Path, mutate) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    payload = {key: value for key, value in document.items() if key != "attestation"}
    document["attestation"] = registry_module._canonical_payload_sha256(payload)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


@pytest.mark.parametrize(
    "spelling",
    [
        lambda relative, absolute: absolute,
        lambda relative, absolute: f"./{relative}",
        lambda relative, absolute: relative.replace("/", "//", 1),
        lambda relative, absolute: relative.replace("/", "\\", 1),
    ],
    ids=["absolute", "dot-prefix", "double-separator", "backslash"],
)
def test_schema2_asset_paths_are_exact_namespace_relative_posix(
    tmp_path: Path, spelling
) -> None:
    registry, root, path = _strict_registry(tmp_path)
    asset = _asset(root, "book_0")
    registry.register(asset)
    relative = asset.geometry_path.resolve().relative_to(root.resolve()).as_posix()
    _rewrite_attested(
        path,
        lambda document: document["assets"]["book_0"].__setitem__(
            "geometry_path", spelling(relative, str(asset.geometry_path.resolve()))
        ),
    )
    with pytest.raises(RuntimeError):
        _candidate(root, path).load_from_file(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_type",
        "nonidentity",
        "composite",
        "support_surface",
        "placement",
        "immutable",
    ],
)
def test_strict_registry_accepts_only_manipuland_identity_templates(
    tmp_path: Path, mutation: str
) -> None:
    registry, root, _ = _strict_registry(tmp_path)
    asset = _asset(root, "book_0")
    if mutation == "wrong_type":
        asset.object_type = ObjectType.FURNITURE
    elif mutation == "nonidentity":
        asset.transform = RigidTransform(p=[0.1, 0.0, 0.0])
    elif mutation == "composite":
        asset.metadata["composite_type"] = "stack"
    elif mutation == "support_surface":
        asset.support_surfaces = [object()]
    elif mutation == "placement":
        asset.placement_info = object()
    else:
        asset.immutable = True
    with pytest.raises(RuntimeError, match="template|object type|identity"):
        registry.register(asset)
    assert registry.size() == 0


def test_strict_registry_detaches_template_from_returned_scene_instance(
    tmp_path: Path,
) -> None:
    registry, root, _ = _strict_registry(tmp_path)
    asset = _asset(root, "book_0")
    registry.register(asset)
    asset.transform = RigidTransform(p=[1.0, 0.0, 0.0])
    asset.metadata["nested"]["value"] = 99

    stored = registry.get(asset.object_id)
    assert stored is not asset
    assert stored.transform.translation().tolist() == [0.0, 0.0, 0.0]
    assert stored.metadata["nested"]["value"] == 1


def test_fresh_registry_allows_room_local_id_matching_legacy_name(
    tmp_path: Path,
) -> None:
    registry, root, path = _strict_registry(tmp_path)
    registry.register(_asset(root, "binder_clip_0"))

    candidate = _candidate(root, path)
    candidate.load_from_file(path)

    assert candidate.size() == 1
    assert candidate.get(UniqueID("binder_clip_0")).description == (
        "reusable binder_clip_0"
    )


@pytest.mark.parametrize("failure", ["escape", "symlink", "hardlink", "cycle"])
def test_strict_registry_rejects_unsafe_transitive_dependencies(
    tmp_path: Path, failure: str
) -> None:
    registry, root, _ = _strict_registry(tmp_path)
    asset = _asset(root, "book_0")
    directory = asset.sdf_path.parent
    if failure == "escape":
        asset.geometry_path.write_text(
            json.dumps({"buffers": [{"uri": "../../outside.bin"}]}),
            encoding="utf-8",
        )
    elif failure in {"symlink", "hardlink"}:
        target = tmp_path / "outside.bin"
        target.write_bytes(b"outside")
        dependency = directory / "payload.bin"
        try:
            if failure == "symlink":
                dependency.symlink_to(target)
            else:
                os.link(target, dependency)
        except OSError as exc:
            pytest.skip(f"{failure} creation unavailable: {exc}")
        asset.geometry_path.write_text(
            json.dumps({"buffers": [{"uri": "payload.bin"}]}),
            encoding="utf-8",
        )
    else:
        asset.geometry_path.write_text(
            json.dumps({"buffers": [{"uri": asset.sdf_path.name}]}),
            encoding="utf-8",
        )
        asset.sdf_path.write_text(
            "<sdf version='1.7'><model name='asset'><link name='base'>"
            f"<visual name='v'><geometry><mesh><uri>{asset.geometry_path.name}</uri>"
            "</mesh></geometry></visual></link></model></sdf>\n",
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError):
        registry.register(asset)
    assert registry.size() == 0


@pytest.mark.parametrize("replacement", [None, "f" * 64], ids=["missing", "forged"])
def test_revision_two_requires_the_exact_revision_one_predecessor(
    tmp_path: Path, replacement: str | None
) -> None:
    registry, root, path = _strict_registry(tmp_path)
    registry.register(_asset(root, "book_0"))
    revision_one = json.loads(path.read_text(encoding="utf-8"))
    registry.register(_asset(root, "folder_0"))
    revision_two = json.loads(path.read_text(encoding="utf-8"))
    assert revision_two["previous_attestation"] == revision_one["attestation"]
    assert replacement != revision_one["attestation"]

    _rewrite_attested(
        path,
        lambda document: document.__setitem__("previous_attestation", replacement),
    )
    with pytest.raises(RuntimeError, match="lineage|predecessor"):
        _candidate(root, path).load_from_file(path)


def test_snapshot_pins_registry_head_while_allowing_attested_descendants(
    tmp_path: Path,
) -> None:
    registry, root, _ = _strict_registry(tmp_path)
    registry.register(_asset(root, "book_0"))
    checkpoint_snapshot = registry.snapshot()
    assert checkpoint_snapshot["head"]["revision"] == 1
    assert checkpoint_snapshot["head"]["attestation"] == (
        registry._previous_attestation
    )
    assert len(checkpoint_snapshot["head"]["persisted_file_sha256"]) == 64

    registry.register(_asset(root, "folder_0"))
    registry.verify_snapshot(checkpoint_snapshot)

    forged = json.loads(json.dumps(checkpoint_snapshot))
    forged["head"]["attestation"] = "f" * 64
    with pytest.raises(RuntimeError, match="head|ancestor"):
        registry.verify_snapshot(forged)

    missing_file_head = json.loads(json.dumps(checkpoint_snapshot))
    missing_file_head["head"]["persisted_file_sha256"] = None
    with pytest.raises(RuntimeError, match="head"):
        registry.verify_snapshot(missing_file_head)


def test_post_replace_directory_fsync_failure_keeps_memory_and_disk_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root, path = _strict_registry(tmp_path)
    registry.register(_asset(root, "book_0"))
    second = _asset(root, "folder_0")
    monkeypatch.setattr(
        registry_module,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("directory fsync failed")),
    )
    with pytest.raises(OSError, match="directory fsync failed"):
        registry.register(second)
    assert registry._last_save_committed is True
    assert registry.get(second.object_id) is not None
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert "folder_0" in disk["assets"]
    assert registry._revision == disk["revision"]
    assert registry._previous_attestation == disk["attestation"]


def _authorize_synthetic_legacy(
    registry: AssetRegistry,
    root: Path,
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    assets = {
        object_id: _asset(root, object_id)
        for object_id in sorted(registry_module._LEGACY_INTERRUPTED_REGISTRY_IDS)
    }
    legacy = {
        object_id: registry._serialize_asset(asset)
        for object_id, asset in assets.items()
    }
    raw = (json.dumps(legacy, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    normalized = {
        object_id: registry._serialize_asset(asset, root=root.resolve())
        for object_id, asset in assets.items()
    }
    monkeypatch.setattr(
        registry_module,
        "_LEGACY_INTERRUPTED_REGISTRY_SHA256",
        registry_module.hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        registry_module,
        "_LEGACY_INTERRUPTED_ENTRY_SHA256",
        {
            object_id: registry_module._canonical_payload_sha256(record)
            for object_id, record in normalized.items()
        },
    )
    monkeypatch.setattr(
        registry_module,
        "_LEGACY_INTERRUPTED_AGGREGATE_SHA256",
        registry_module._canonical_payload_sha256(normalized),
    )
    monkeypatch.setattr(
        registry_module,
        "_LEGACY_INTERRUPTED_DIRECTORY_MANIFESTS",
        {
            object_id: registry._legacy_directory_manifest(
                asset.sdf_path.parent, root=root.resolve()
            )
            for object_id, asset in assets.items()
        },
    )
    return raw


def test_failed_legacy_migration_rolls_back_memory_and_preserves_legacy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, root, path = _strict_registry(tmp_path)
    raw = _authorize_synthetic_legacy(registry, root, path, monkeypatch)
    monkeypatch.setattr(
        registry_module.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        registry.load_from_file(path)
    assert path.read_bytes() == raw
    assert registry.size() == 0
    assert registry._revision == 0
    assert registry._previous_attestation is None
    assert registry._persisted_file_sha256 is None


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_registry_document_link_is_rejected(
    tmp_path: Path, link_kind: str
) -> None:
    registry, root, path = _strict_registry(tmp_path)
    registry.register(_asset(root, "book_0"))
    linked = root / "linked_registry.json"
    try:
        if link_kind == "symlink":
            linked.symlink_to(path)
        else:
            os.link(path, linked)
    except OSError as exc:
        pytest.skip(f"{link_kind} creation unavailable: {exc}")
    with pytest.raises(RuntimeError):
        _candidate(root, linked).load_from_file(linked)


def test_symlinked_strict_registry_root_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real" / "manipuland"
    (real_root / "sdf").mkdir(parents=True)
    linked_root = tmp_path / "linked_manipuland"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    registry = AssetRegistry(
        auto_save_path=linked_root / "asset_registry.json",
        required_root=linked_root,
        allowed_object_types=frozenset({ObjectType.MANIPULAND}),
    )
    with pytest.raises(RuntimeError, match="symlink"):
        registry.register(_asset(linked_root, "book_0"))


def test_unreferenced_namespace_file_is_bound_and_revalidated(tmp_path: Path) -> None:
    registry, root, path = _strict_registry(tmp_path)
    asset = _asset(root, "book_0")
    extra = asset.sdf_path.parent / "unreferenced-but-authoritative.dat"
    extra.write_bytes(b"publisher sidecar")
    registry.register(asset)
    restarted = _candidate(root, path)
    restarted.load_from_file(path)
    extra.write_bytes(b"mutated sidecar")
    with pytest.raises(RuntimeError, match="inventory|bytes|differs"):
        restarted.verify_persisted()


def test_generic_registry_preserves_session_only_path_compatibility(
    tmp_path: Path,
) -> None:
    """Strict restart validation must not broaden into furniture/wall registries."""
    path = tmp_path / "generated_assets" / "furniture" / "asset_registry.json"
    registry = AssetRegistry(auto_save_path=path)
    asset = SceneObject(
        object_id=UniqueID("chair_0"),
        object_type=ObjectType.FURNITURE,
        name="chair",
        description="mocked furniture asset",
        transform=RigidTransform(p=[1.0, 2.0, 0.0]),
        geometry_path=Path("/test/nonexistent_asset.gltf"),
        sdf_path=Path("/test/nonexistent_asset.sdf"),
        image_path=Path("/test/nonexistent_image.png"),
        bbox_min=np.array([-0.5, -0.5, 0.0]),
        bbox_max=np.array([0.5, 0.5, 1.0]),
    )
    registry.register(asset)
    assert registry.get(asset.object_id) is asset
    assert registry.list_all() == [asset]
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document) == {"chair_0"}
    assert document["chair_0"]["geometry_path"] == str(asset.geometry_path)
    assert registry.get(asset.object_id).transform.translation().tolist() == [
        1.0,
        2.0,
        0.0,
    ]
    restarted = AssetRegistry(auto_save_path=path)
    restarted.load_from_file(path)
    loaded = restarted.get(asset.object_id)
    assert loaded.geometry_path == asset.geometry_path
    assert loaded.sdf_path == asset.sdf_path
    assert loaded.image_path == asset.image_path
