import hashlib
import io
import json
import os
import sys
import tarfile

from dataclasses import replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import safe_extract_artiverse as extractor  # noqa: E402


def _tar_info(name: str, *, kind: bytes = tarfile.REGTYPE, payload: bytes = b""):
    info = tarfile.TarInfo(name)
    info.type = kind
    if kind in {tarfile.REGTYPE, tarfile.AREGTYPE}:
        info.size = len(payload)
    if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
        info.linkname = "../../outside"
    return info, payload


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for info, payload in members:
            archive.addfile(info, io.BytesIO(payload) if info.isreg() else None)


def _make_release(
    tmp_path: Path,
    chunks: list[tuple[list[str], list[tuple[tarfile.TarInfo, bytes]]]],
    *,
    manifest_changes: dict | None = None,
):
    repository = tmp_path / "artiverse"
    chunks_dir = repository / "dataset_chunks"
    chunks_dir.mkdir(parents=True)
    (repository / "pack_dataset_chunks.py").write_text("# publisher metadata\n")
    chunk_specs = []
    chunk_entries = []
    for index, (roots, members) in enumerate(chunks, start=1):
        name = f"test-{index:05d}-of-{len(chunks):05d}.tar.gz"
        path = chunks_dir / name
        _write_tar(path, members)
        payload = path.read_bytes()
        regular = [info for info, _data in members if info.isreg()]
        file_count = len(regular)
        input_bytes = sum(info.size for info in regular)
        spec = extractor.ChunkSpec(
            archive=name,
            sha256=hashlib.sha256(payload).hexdigest(),
            archive_bytes=len(payload),
            model_count=len(roots),
            file_count=file_count,
            input_bytes=input_bytes,
            estimated_tar_bytes=input_bytes,
        )
        chunk_specs.append(spec)
        chunk_entries.append(
            {
                "index": index,
                "archive": spec.archive,
                "sha256": spec.sha256,
                "archive_bytes": spec.archive_bytes,
                "model_count": spec.model_count,
                "file_count": spec.file_count,
                "input_bytes": spec.input_bytes,
                "estimated_tar_bytes": spec.estimated_tar_bytes,
                "roots": roots,
            }
        )
    manifest = {
        "format": extractor.MANIFEST_FORMAT,
        "created_utc": "2026-01-01T00:00:00Z",
        "data_dir": "data",
        "max_chunk_gb": 1.0,
        "chunk_count": len(chunk_specs),
        "model_count": sum(item.model_count for item in chunk_specs),
        "file_count": sum(item.file_count for item in chunk_specs),
        "input_bytes": sum(item.input_bytes for item in chunk_specs),
        "estimated_tar_bytes": sum(item.estimated_tar_bytes for item in chunk_specs),
        "chunks": chunk_entries,
    }
    if manifest_changes:
        manifest.update(manifest_changes)
    manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode()
    (chunks_dir / "manifest.json").write_bytes(manifest_payload)
    spec = extractor.ReleaseSpec(
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        chunks=tuple(chunk_specs),
        model_count=manifest["model_count"],
        file_count=manifest["file_count"],
        input_bytes=manifest["input_bytes"],
        estimated_tar_bytes=manifest["estimated_tar_bytes"],
    )
    return repository, spec


def _ordinary_chunks():
    root_one = "data/cabinet/publisher/model_one"
    root_two = "data/storage/publisher/model_two"
    return [
        (
            [root_one],
            [
                _tar_info(root_one + "/", kind=tarfile.DIRTYPE),
                _tar_info(root_one + "/mesh.bin", payload=b"mesh-one"),
            ],
        ),
        (
            [root_two],
            [
                _tar_info(root_two + "/", kind=tarfile.DIRTYPE),
                _tar_info(root_two + "/nested/asset.bin", payload=b"asset-two"),
            ],
        ),
    ]


def _chunks_with_publisher_urdf():
    root_one = "data/cabinet/publisher/model_one"
    root_two = "data/storage/publisher/model_two"
    return [
        (
            [root_one],
            [
                _tar_info(root_one + "/", kind=tarfile.DIRTYPE),
                _tar_info(
                    root_one + "/urdf_w_collider/mobility.urdf",
                    payload=b"<robot/>",
                ),
                _tar_info(root_one + "/mesh.bin", payload=b"mesh-one"),
            ],
        ),
        (
            [root_two],
            [_tar_info(root_two + "/asset.bin", payload=b"asset-two")],
        ),
    ]


def test_benign_release_is_published_with_deterministic_receipt(tmp_path: Path) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    publisher_script = (repository / "pack_dataset_chunks.py").read_bytes()
    manifest = (repository / "dataset_chunks" / "manifest.json").read_bytes()

    receipt = extractor.extract_release(repository, expected_release=spec)

    assert (repository / "data/cabinet/publisher/model_one/mesh.bin").read_bytes() == b"mesh-one"
    assert (repository / "data/storage/publisher/model_two/nested/asset.bin").read_bytes() == b"asset-two"
    assert (repository / "pack_dataset_chunks.py").read_bytes() == publisher_script
    assert (repository / "dataset_chunks" / "manifest.json").read_bytes() == manifest
    on_disk = json.loads((repository / extractor.RECEIPT_FILENAME).read_text())
    assert on_disk == receipt
    assert receipt["status"] == "pass"
    assert receipt["manifest"]["sha256"] == spec.manifest_sha256
    assert receipt["declared_roots"]["count"] == 2
    assert receipt["validated_aggregate"]["regular_file_count"] == 2
    assert receipt["validated_aggregate"]["uncompressed_bytes"] == 17
    assert receipt["data_tree_inventory"]["regular_file_count"] == 2
    assert receipt["safe_extractor"]["sha256"] == hashlib.sha256(
        Path(extractor.__file__).read_bytes()
    ).hexdigest()
    assert extractor.verify_existing_release(
        repository, expected_release=spec
    ) == receipt
    assert not list(repository.glob(".artiverse-data-staging-*"))


@pytest.mark.parametrize(
    "reserved_name",
    [extractor.DERIVED_SDF_FILENAME, extractor.DERIVED_SDF_TEMP_FILENAME],
)
def test_fresh_extraction_rejects_publisher_member_at_reserved_derived_path(
    tmp_path: Path, reserved_name: str
) -> None:
    root = "data/cabinet/publisher/model_one"
    other_root = "data/storage/publisher/model_two"
    chunks = [
        (
            [root],
            [
                _tar_info(
                    root + "/urdf_w_collider/mobility.urdf",
                    payload=b"<robot/>",
                ),
                _tar_info(
                    root + "/urdf_w_collider/" + reserved_name,
                    payload=b"publisher collision",
                ),
            ],
        ),
        ([other_root], [_tar_info(other_root + "/asset.bin", payload=b"asset")]),
    ]
    repository, spec = _make_release(tmp_path, chunks)

    with pytest.raises(
        extractor.ArtiverseExtractionError,
        match="reserved SceneSmith derived-SDF path",
    ):
        extractor.extract_release(repository, expected_release=spec)

    assert not (repository / "data").exists()


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("/data/cabinet/publisher/model_one/evil", tarfile.REGTYPE, "absolute"),
        ("data/cabinet/publisher/model_one/../../evil", tarfile.REGTYPE, "dotdot"),
        ("data\\cabinet\\publisher\\model_one\\evil", tarfile.REGTYPE, "backslash"),
        ("data/cabinet/./publisher/model_one/evil", tarfile.REGTYPE, "dot"),
        ("data/cabinet/publisher/model_one/link", tarfile.SYMTYPE, "forbidden"),
        ("data/cabinet/publisher/model_one/hard", tarfile.LNKTYPE, "forbidden"),
        ("data/cabinet/publisher/model_one/fifo", tarfile.FIFOTYPE, "forbidden"),
        ("data/cabinet/publisher/model_one/device", tarfile.CHRTYPE, "forbidden"),
        ("data/cabinet/publisher/model_one/block", tarfile.BLKTYPE, "forbidden"),
    ],
)
def test_malicious_member_types_and_paths_are_rejected(
    tmp_path: Path, name: str, kind: bytes, message: str
) -> None:
    root = "data/cabinet/publisher/model_one"
    chunks = [
        ([root], [_tar_info(name, kind=kind, payload=b"bad")]),
        (["data/storage/publisher/model_two"], [
            _tar_info("data/storage/publisher/model_two/good", payload=b"ok")
        ]),
    ]
    repository, spec = _make_release(tmp_path, chunks)

    with pytest.raises(extractor.ArtiverseExtractionError, match=message):
        extractor.extract_release(repository, expected_release=spec)

    assert not (repository / "data").exists()
    assert not (repository / extractor.RECEIPT_FILENAME).exists()
    assert not list(repository.glob(".artiverse-data-staging-*"))


def test_member_outside_its_declared_model_root_is_rejected(tmp_path: Path) -> None:
    chunks = [
        (["data/cabinet/publisher/model_one"], [
            _tar_info("data/cabinet/publisher/other/file", payload=b"bad")
        ]),
        (["data/storage/publisher/model_two"], [
            _tar_info("data/storage/publisher/model_two/file", payload=b"ok")
        ]),
    ]
    repository, spec = _make_release(tmp_path, chunks)

    with pytest.raises(extractor.ArtiverseExtractionError, match="declared roots"):
        extractor.extract_release(repository, expected_release=spec)


def test_regular_file_cannot_replace_a_declared_model_directory(tmp_path: Path) -> None:
    root = "data/cabinet/publisher/model_one"
    chunks = [
        ([root], [_tar_info(root, payload=b"not-a-model-directory")]),
        (["data/storage/publisher/model_two"], [
            _tar_info("data/storage/publisher/model_two/file", payload=b"ok")
        ]),
    ]
    repository, spec = _make_release(tmp_path, chunks)

    with pytest.raises(extractor.ArtiverseExtractionError, match="not below"):
        extractor.extract_release(repository, expected_release=spec)

    assert not (repository / "data").exists()


def test_nul_and_sparse_members_are_rejected() -> None:
    with pytest.raises(extractor.ArtiverseExtractionError, match="NUL"):
        extractor._normalize_posix_path(
            "data/cabinet/publisher/model_one/bad\x00name",
            "test member",
            directory=False,
        )
    sparse = tarfile.TarInfo("data/cabinet/publisher/model_one/sparse")
    sparse.type = tarfile.GNUTYPE_SPARSE
    with pytest.raises(extractor.ArtiverseExtractionError, match="Sparse"):
        extractor._reject_special_member(sparse)


@pytest.mark.parametrize("second_name", ["File.txt", "file.txt"])
def test_duplicate_and_casefold_colliding_members_are_rejected(
    tmp_path: Path, second_name: str
) -> None:
    root = "data/cabinet/publisher/model_one"
    first_name = root + "/File.txt"
    chunks = [
        ([root], [
            _tar_info(first_name, payload=b"one"),
            _tar_info(root + "/" + second_name, payload=b"two"),
        ]),
        (["data/storage/publisher/model_two"], [
            _tar_info("data/storage/publisher/model_two/file", payload=b"ok")
        ]),
    ]
    repository, spec = _make_release(tmp_path, chunks)

    with pytest.raises(
        extractor.ArtiverseExtractionError, match="Duplicate normalized|Case-fold"
    ):
        extractor.extract_release(repository, expected_release=spec)


def _pinned_casefold_collision(
    *,
    manifest_sha256: str = "1" * 64,
    archive: str = "pinned.tar.gz",
    archive_sha256: str = "2" * 64,
    first_payload: bytes = b"upper material",
    second_payload: bytes = b"lower material",
) -> extractor.PinnedCasefoldCollision:
    root = "data/toaster/objaverse/9662f5cc6cf6456292613d8c5075fcfb"
    parent = root + "/urdf_w_collider/objs"
    return extractor.PinnedCasefoldCollision(
        manifest_sha256=manifest_sha256,
        archive=archive,
        archive_sha256=archive_sha256,
        model_root=root,
        members=(
            extractor.PinnedCasefoldMember(
                path=parent + "/1_base__Material.mtl",
                size=len(first_payload),
                sha256=hashlib.sha256(first_payload).hexdigest(),
            ),
            extractor.PinnedCasefoldMember(
                path=parent + "/1_base__material.mtl",
                size=len(second_payload),
                sha256=hashlib.sha256(second_payload).hexdigest(),
            ),
        ),
    )


def test_official_casefold_allowlist_is_exactly_content_and_release_pinned() -> None:
    root = "data/toaster/objaverse/9662f5cc6cf6456292613d8c5075fcfb"
    parent = root + "/urdf_w_collider/objs"
    expected = {
        (
            parent + "/1_base__Material.mtl",
            176,
            "3ecce3aba2a39f02f63ba0fedde49a663464411625fd24ad0abb74d6c145ef0a",
        ),
        (
            parent + "/1_base__material.mtl",
            174,
            "cf70acff54721bb8fa667fabf88a34d0a506980a8fa1a67113dffece9ccc1fec",
        ),
        (
            parent + "/1_base__Material.obj",
            27_919,
            "643bb5505d1d281a5996220f3d0c38b782315d9020a1a89a12760e94c3940273",
        ),
        (
            parent + "/1_base__material.obj",
            21_915,
            "e6aa38146df6728147fd20c6937a0c4858e9b0e51d7cb92f16f704437aa0b866",
        ),
    }
    records = extractor.PINNED_CASEFOLD_COLLISIONS

    assert len(records) == 2
    assert {record.manifest_sha256 for record in records} == {
        extractor.PINNED_MANIFEST_SHA256
    }
    assert {record.archive for record in records} == {
        "artiverse_data-00002-of-00002.tar.gz"
    }
    assert {record.archive_sha256 for record in records} == {
        extractor.PINNED_RELEASE.chunks[1].sha256
    }
    assert {record.model_root for record in records} == {root}
    assert {
        (member.path, member.size, member.sha256)
        for record in records
        for member in record.members
    } == expected
    for record in records:
        extractor._validate_casefold_allowance_shape(record)


def _register_verified_collision_member(
    state: extractor.TreeState,
    member: extractor.PinnedCasefoldMember,
    payload: bytes,
    *,
    archive: str,
) -> None:
    parts = tuple(member.path.split("/"))
    normalized = extractor._register_member(
        state,
        parts,
        is_directory=False,
        archive=archive,
    )
    digest = hashlib.sha256(payload).digest()
    extractor._verify_casefold_member(
        state,
        normalized,
        archive=archive,
        size=len(payload),
        content_digest=digest,
    )
    state.files[normalized] = (len(payload), digest)


def test_exact_content_pinned_casefold_pair_is_narrowly_accepted() -> None:
    first_payload = b"upper material"
    second_payload = b"lower material"
    allowance = _pinned_casefold_collision(
        first_payload=first_payload,
        second_payload=second_payload,
    )
    state = extractor.TreeState(casefold_allowances=(allowance,))

    _register_verified_collision_member(
        state,
        allowance.members[0],
        first_payload,
        archive=allowance.archive,
    )
    _register_verified_collision_member(
        state,
        allowance.members[1],
        second_payload,
        archive=allowance.archive,
    )
    extractor._require_complete_casefold_allowances(state)

    assert state.verified_casefold_members == {
        member.path for member in allowance.members
    }
    assert set(state.files) == state.verified_casefold_members


@pytest.mark.parametrize(
    ("archive", "payload", "message"),
    [
        ("wrong.tar.gz", b"upper material", "wrong archive"),
        ("pinned.tar.gz", b"tampered material", "content mismatch"),
    ],
)
def test_pinned_casefold_member_rejects_wrong_archive_or_content(
    archive: str,
    payload: bytes,
    message: str,
) -> None:
    allowance = _pinned_casefold_collision()
    state = extractor.TreeState(casefold_allowances=(allowance,))
    member = allowance.members[0]
    normalized = extractor._register_member(
        state,
        tuple(member.path.split("/")),
        is_directory=False,
        archive=archive,
    )

    with pytest.raises(extractor.ArtiverseExtractionError, match=message):
        extractor._verify_casefold_member(
            state,
            normalized,
            archive=archive,
            size=len(payload),
            content_digest=hashlib.sha256(payload).digest(),
        )


def test_pinned_pair_does_not_allow_third_spelling_or_parent_ambiguity() -> None:
    allowance = _pinned_casefold_collision()
    state = extractor.TreeState(casefold_allowances=(allowance,))
    first_payload = b"upper material"
    _register_verified_collision_member(
        state,
        allowance.members[0],
        first_payload,
        archive=allowance.archive,
    )
    parent = allowance.members[0].path.rsplit("/", 1)[0]

    with pytest.raises(extractor.ArtiverseExtractionError, match="Case-fold"):
        extractor._register_member(
            state,
            tuple((parent + "/1_base__MATERIAL.mtl").split("/")),
            is_directory=False,
            archive=allowance.archive,
        )
    ambiguous_parent = parent.replace("urdf_w_collider", "URDF_w_collider")
    with pytest.raises(extractor.ArtiverseExtractionError, match="Case-fold"):
        extractor._register_member(
            state,
            tuple((ambiguous_parent + "/1_base__material.mtl").split("/")),
            is_directory=False,
            archive=allowance.archive,
        )


def test_pinned_pair_requires_both_exact_members() -> None:
    allowance = _pinned_casefold_collision()
    state = extractor.TreeState(casefold_allowances=(allowance,))
    _register_verified_collision_member(
        state,
        allowance.members[0],
        b"upper material",
        archive=allowance.archive,
    )

    with pytest.raises(
        extractor.ArtiverseExtractionError,
        match="membership mismatch",
    ):
        extractor._require_complete_casefold_allowances(state)


def test_casefold_allowance_selection_binds_release_archive_and_root(
    monkeypatch,
) -> None:
    allowance = _pinned_casefold_collision()
    chunk_spec = extractor.ChunkSpec(
        archive=allowance.archive,
        sha256=allowance.archive_sha256,
        archive_bytes=1,
        model_count=1,
        file_count=2,
        input_bytes=sum(member.size for member in allowance.members),
        estimated_tar_bytes=1,
    )
    release = extractor.ReleaseSpec(
        manifest_sha256=allowance.manifest_sha256,
        chunks=(chunk_spec,),
        model_count=1,
        file_count=2,
        input_bytes=chunk_spec.input_bytes,
        estimated_tar_bytes=1,
    )
    chunk = extractor.ValidatedChunk(
        spec=chunk_spec,
        roots=(allowance.model_root,),
    )
    monkeypatch.setattr(
        extractor,
        "PINNED_CASEFOLD_COLLISIONS",
        (allowance,),
    )

    assert extractor._casefold_allowances_for_release(release, (chunk,)) == (
        allowance,
    )
    wrong_archive_hash = replace(
        chunk_spec,
        sha256="3" * 64,
    )
    with pytest.raises(
        extractor.ArtiverseExtractionError,
        match="archive SHA-256",
    ):
        extractor._casefold_allowances_for_release(
            replace(release, chunks=(wrong_archive_hash,)),
            (replace(chunk, spec=wrong_archive_hash),),
        )


@pytest.mark.skipif(
    os.path.normcase("Material") == os.path.normcase("material"),
    reason="requires a case-sensitive extraction filesystem",
)
def test_casefold_exception_extracts_and_reverifies_only_exact_pinned_pair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_payload = b"upper material"
    second_payload = b"lower material"
    provisional = _pinned_casefold_collision(
        first_payload=first_payload,
        second_payload=second_payload,
    )
    chunks = [
        (
            [provisional.model_root],
            [
                _tar_info(provisional.members[0].path, payload=first_payload),
                _tar_info(provisional.members[1].path, payload=second_payload),
            ],
        )
    ]
    repository, release = _make_release(tmp_path, chunks)
    allowance = replace(
        provisional,
        manifest_sha256=release.manifest_sha256,
        archive=release.chunks[0].archive,
        archive_sha256=release.chunks[0].sha256,
    )
    monkeypatch.setattr(
        extractor,
        "PINNED_CASEFOLD_COLLISIONS",
        (allowance,),
    )

    receipt = extractor.extract_release(repository, expected_release=release)

    assert receipt["status"] == "pass"
    for member, payload in zip(allowance.members, (first_payload, second_payload)):
        assert (repository / member.path).read_bytes() == payload
    assert extractor.verify_existing_release(
        repository,
        expected_release=release,
    ) == receipt


@pytest.mark.parametrize("second_root", [
    "data/cabinet/publisher/model_one",
    "data/Cabinet/publisher/model_one",
])
def test_duplicate_or_casefold_declared_roots_across_chunks_are_rejected(
    tmp_path: Path, second_root: str
) -> None:
    root = "data/cabinet/publisher/model_one"
    chunks = [
        ([root], [_tar_info(root + "/one", payload=b"one")]),
        ([second_root], [_tar_info(second_root + "/two", payload=b"two")]),
    ]
    repository, spec = _make_release(tmp_path, chunks)

    with pytest.raises(
        extractor.ArtiverseExtractionError, match="root (duplicate|case-fold collision)"
    ):
        extractor.extract_release(repository, expected_release=spec)


def test_every_archive_hash_is_verified_before_staging_mutation(tmp_path: Path) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    second = repository / "dataset_chunks" / spec.chunks[1].archive
    damaged = bytearray(second.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    second.write_bytes(damaged)

    with pytest.raises(extractor.ArtiverseExtractionError, match="SHA-256 mismatch"):
        extractor.extract_release(repository, expected_release=spec)

    assert not (repository / "data").exists()
    assert not list(repository.glob(".artiverse-data-staging-*"))


def test_archive_mutation_during_stream_cannot_hide_from_authentication(
    monkeypatch, tmp_path: Path
) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    first_spec = spec.chunks[0]
    first_path = repository / "dataset_chunks" / first_spec.archive
    original_bytes = first_path.read_bytes()
    root = "data/cabinet/publisher/model_one"
    alternate_path = tmp_path / "alternate.tar.gz"
    _write_tar(
        alternate_path,
        [
            _tar_info(root + "/", kind=tarfile.DIRTYPE),
            _tar_info(root + "/mesh.bin", payload=b"evil-one"),
        ],
    )
    alternate_bytes = alternate_path.read_bytes()
    original_extract = extractor._extract_archive
    original_authenticate = extractor._AuthenticatedArchiveReader.authenticate
    mutation_started = False
    official_bytes_restored = False

    def mutate_while_extracting(archive, staging_root, state):
        nonlocal mutation_started
        if archive.chunk.spec.archive == first_spec.archive:
            first_path.write_bytes(alternate_bytes)
            mutation_started = True
        return original_extract(archive, staging_root, state)

    def restore_before_snapshot_check(reader, chunk_spec):
        nonlocal official_bytes_restored
        if chunk_spec.archive == first_spec.archive:
            # Consume the alternate compressed stream first, then restore the
            # authenticated on-disk snapshot before the legacy post-check.
            reader.drain()
            first_path.write_bytes(original_bytes)
            official_bytes_restored = True
            reader.validate(chunk_spec)
            return
        original_authenticate(reader, chunk_spec)

    monkeypatch.setattr(extractor, "_extract_archive", mutate_while_extracting)
    monkeypatch.setattr(
        extractor._AuthenticatedArchiveReader,
        "authenticate",
        restore_before_snapshot_check,
    )
    try:
        with pytest.raises(
            extractor.ArtiverseExtractionError,
            match="consumed during extraction failed authentication",
        ):
            extractor.extract_release(repository, expected_release=spec)
    finally:
        if first_path.read_bytes() != original_bytes:
            first_path.write_bytes(original_bytes)

    assert mutation_started
    assert official_bytes_restored
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first_spec.sha256
    assert not (repository / "data").exists()
    assert not (repository / extractor.RECEIPT_FILENAME).exists()


def test_exact_file_counts_and_bytes_are_enforced(tmp_path: Path) -> None:
    repository, original = _make_release(tmp_path, _ordinary_chunks())
    chunks = list(original.chunks)
    chunks[0] = replace(chunks[0], file_count=2)
    manifest_path = repository / "dataset_chunks" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["file_count"] = 2
    manifest["file_count"] = 3
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    manifest_path.write_bytes(payload)
    spec = replace(
        original,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        chunks=tuple(chunks),
        file_count=3,
    )

    with pytest.raises(extractor.ArtiverseExtractionError, match="regular-file count mismatch"):
        extractor.extract_release(repository, expected_release=spec)


def test_existing_data_requires_explicit_overwrite_and_is_replaced(tmp_path: Path) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    existing = repository / "data"
    existing.mkdir()
    (existing / "old.txt").write_text("old")

    with pytest.raises(extractor.ArtiverseExtractionError, match="already exists"):
        extractor.extract_release(repository, expected_release=spec)
    receipt = extractor.extract_release(repository, expected_release=spec, overwrite=True)

    assert receipt["status"] == "pass"
    assert not (repository / "data/old.txt").exists()
    assert (repository / "data/cabinet/publisher/model_one/mesh.bin").is_file()
    assert not list(repository.glob(".artiverse-*-backup-*"))


def test_overwrite_publish_failure_restores_data_and_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    existing = repository / "data"
    existing.mkdir()
    (existing / "old.txt").write_text("old")
    receipt_path = repository / extractor.RECEIPT_FILENAME
    receipt_path.write_text("old receipt\n")
    real_replace = extractor.os.replace

    def fail_new_receipt(source, destination):
        source_path = Path(source)
        if (
            source_path.name == extractor.RECEIPT_FILENAME
            and source_path.parent.name.startswith(".artiverse-data-staging-")
        ):
            raise OSError("simulated receipt publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(extractor.os, "replace", fail_new_receipt)
    with pytest.raises(extractor.ArtiverseExtractionError, match="Atomic publish failed"):
        extractor.extract_release(repository, expected_release=spec, overwrite=True)

    assert (repository / "data/old.txt").read_text() == "old"
    assert receipt_path.read_text() == "old receipt\n"
    assert not list(repository.glob(".artiverse-*-backup-*"))


def test_manifest_schema_and_data_dir_are_fail_closed(tmp_path: Path) -> None:
    repository, spec = _make_release(
        tmp_path, _ordinary_chunks(), manifest_changes={"data_dir": "../escape"}
    )

    with pytest.raises(extractor.ArtiverseExtractionError, match="data_dir"):
        extractor.extract_release(repository, expected_release=spec)


def test_verify_existing_rejects_content_tamper(tmp_path: Path) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    extractor.extract_release(repository, expected_release=spec)
    tampered = repository / "data/cabinet/publisher/model_one/mesh.bin"
    tampered.write_bytes(b"mesh-tw0")

    with pytest.raises(extractor.ArtiverseExtractionError, match="inventory SHA-256"):
        extractor.verify_existing_release(repository, expected_release=spec)


def test_verify_existing_rejects_symlink(tmp_path: Path) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    extractor.extract_release(repository, expected_release=spec)
    target = repository / "data/cabinet/publisher/model_one/mesh.bin"
    outside = repository / "outside.bin"
    outside.write_bytes(b"mesh-one")
    target.unlink()
    try:
        target.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable for this test user: {exc}")

    with pytest.raises(extractor.ArtiverseExtractionError, match="Link/reparse"):
        extractor.verify_existing_release(repository, expected_release=spec)


def test_verify_existing_omits_only_explicit_scenesmith_sdf_derivatives(
    tmp_path: Path,
) -> None:
    repository, spec = _make_release(tmp_path, _chunks_with_publisher_urdf())
    receipt = extractor.extract_release(repository, expected_release=spec)
    urdf_dir = (
        repository
        / "data"
        / "cabinet"
        / "publisher"
        / "model_one"
        / "urdf_w_collider"
    )
    (urdf_dir / extractor.DERIVED_SDF_FILENAME).write_text(
        "<sdf><model/></sdf>", encoding="utf-8"
    )
    (urdf_dir / extractor.DERIVED_SDF_TEMP_FILENAME).write_text(
        "incomplete transaction", encoding="utf-8"
    )

    assert extractor.verify_existing_release(
        repository, expected_release=spec
    ) == receipt


def test_verify_existing_with_derived_sdf_still_rejects_source_tamper(
    tmp_path: Path,
) -> None:
    repository, spec = _make_release(tmp_path, _chunks_with_publisher_urdf())
    extractor.extract_release(repository, expected_release=spec)
    urdf_dir = (
        repository
        / "data"
        / "cabinet"
        / "publisher"
        / "model_one"
        / "urdf_w_collider"
    )
    (urdf_dir / extractor.DERIVED_SDF_FILENAME).write_text("<sdf/>", encoding="utf-8")
    (repository / "data/cabinet/publisher/model_one/mesh.bin").write_bytes(
        b"mesh-tw0"
    )

    with pytest.raises(extractor.ArtiverseExtractionError, match="inventory SHA-256"):
        extractor.verify_existing_release(repository, expected_release=spec)


def test_verify_existing_with_derived_sdf_rejects_arbitrary_extra_file(
    tmp_path: Path,
) -> None:
    repository, spec = _make_release(tmp_path, _chunks_with_publisher_urdf())
    extractor.extract_release(repository, expected_release=spec)
    model_dir = repository / "data/cabinet/publisher/model_one"
    urdf_dir = model_dir / "urdf_w_collider"
    (urdf_dir / extractor.DERIVED_SDF_FILENAME).write_text("<sdf/>", encoding="utf-8")
    (model_dir / "unapproved-derived.bin").write_bytes(b"extra")

    with pytest.raises(
        extractor.ArtiverseExtractionError,
        match="exceeds release counts|file/byte counts mismatch",
    ):
        extractor.verify_existing_release(repository, expected_release=spec)


def test_verify_existing_rejects_derived_sdf_without_sibling_publisher_urdf(
    tmp_path: Path,
) -> None:
    repository, spec = _make_release(tmp_path, _ordinary_chunks())
    extractor.extract_release(repository, expected_release=spec)
    urdf_dir = repository / "data/cabinet/publisher/model_one/urdf_w_collider"
    urdf_dir.mkdir()
    (urdf_dir / extractor.DERIVED_SDF_FILENAME).write_text("<sdf/>", encoding="utf-8")

    with pytest.raises(
        extractor.ArtiverseExtractionError, match="not beside a publisher URDF"
    ):
        extractor.verify_existing_release(repository, expected_release=spec)


def test_verify_existing_rejects_hardlinked_derived_sdf(tmp_path: Path) -> None:
    repository, spec = _make_release(tmp_path, _chunks_with_publisher_urdf())
    extractor.extract_release(repository, expected_release=spec)
    urdf_dir = repository / "data/cabinet/publisher/model_one/urdf_w_collider"
    sdf_path = urdf_dir / extractor.DERIVED_SDF_FILENAME
    sdf_path.write_text("<sdf/>", encoding="utf-8")
    try:
        os.link(sdf_path, repository / "derived-sdf-hardlink")
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable for this test user: {exc}")

    with pytest.raises(extractor.ArtiverseExtractionError, match="Hard-linked"):
        extractor.verify_existing_release(repository, expected_release=spec)
