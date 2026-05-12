from __future__ import annotations

import gzip
import json
import math
import pickle
import struct

from pathlib import Path

import numpy as np
import pytest

import scripts.preflight_objathor_retrieval as preflight


def _write_glb(path: Path) -> None:
    body = b"test"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def _fixture(tmp_path: Path, *, dimensions: int = 768, rows: int = 6) -> dict[str, object]:
    dataset = tmp_path / "objathor-assets"
    preprocessed = dataset / "preprocessed"
    preprocessed.mkdir(parents=True)
    uids = [f"uid-{index:02d}" for index in range(rows)]
    metadata = {
        uid: {
            "name": f"asset {index}",
            "description": f"school asset {index}",
            "category": "small_objects" if index % 2 == 0 else "large_objects",
            "bounding_box": [1.0, 1.0, 1.0],
        }
        for index, uid in enumerate(uids)
    }
    categories = {
        "large_objects": [uid for index, uid in enumerate(uids) if index % 2],
        "small_objects": [uid for index, uid in enumerate(uids) if not index % 2],
    }
    (preprocessed / "metadata_index.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (preprocessed / "embedding_index.yaml").write_text(
        "uids:\n" + "".join(f"- {uid}\n" for uid in uids), encoding="utf-8"
    )
    (preprocessed / "object_categories.json").write_text(
        json.dumps(categories), encoding="utf-8"
    )
    embeddings = np.ones((rows, dimensions), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(preprocessed / "clip_embeddings.npy", embeddings)

    mesh_paths: dict[str, Path] = {}
    for uid in uids:
        asset_dir = dataset / "assets" / uid
        _write_glb(asset_dir / f"{uid}.glb")
        pickle_path = asset_dir / f"{uid}.pkl.gz"
        payload = {
            "vertices": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 1.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 1.0, "z": 0.0},
            ],
            "triangles": [0, 1, 2],
        }
        with gzip.open(pickle_path, "wb") as stream:
            pickle.dump(payload, stream, protocol=4)
        mesh_paths[uid] = pickle_path

    weight = tmp_path / "cache" / preflight.MODEL_FILENAME
    weight.parent.mkdir()
    weight.write_bytes(b"pinned-openclip-weight")
    output = tmp_path / "objathor_retrieval_preflight.json"
    return {
        "dataset": dataset,
        "preprocessed": preprocessed,
        "uids": uids,
        "mesh_paths": mesh_paths,
        "weight": weight,
        "output": output,
        "rows": rows,
    }


def _weight_evidence(weight: Path) -> dict[str, object]:
    return {
        "model_name": preflight.MODEL_NAME,
        "pretrained_tag": preflight.PRETRAINED_TAG,
        "repo_id": preflight.MODEL_REPO_ID,
        "requested_revision": preflight.MODEL_REVISION,
        "snapshot_revision": preflight.MODEL_REVISION,
        "filename": preflight.MODEL_FILENAME,
        "cache_path": str(weight),
        "blob_path": str(weight.resolve()),
        "open_clip_version": "test",
    }


def _normalized_embedding() -> np.ndarray:
    embedding = np.ones(preflight.EMBEDDING_DIMENSION, dtype=np.float32)
    return embedding / np.linalg.norm(embedding)


def _run(
    fixture: dict[str, object],
    *,
    model_loader=None,
    weight_resolver=None,
    mesh_path_resolver=None,
) -> dict:
    weight = fixture["weight"]
    mesh_paths = fixture["mesh_paths"]
    return preflight.run(
        fixture["dataset"],
        fixture["preprocessed"],
        fixture["output"],
        minimum_rows=fixture["rows"],
        maximum_rows=fixture["rows"],
        sample_count=3,
        weight_resolver=weight_resolver or (lambda: _weight_evidence(weight)),
        model_loader=model_loader
        or (
            lambda model, _query: {
                "embedding": _normalized_embedding(),
                "device": "cpu",
                "loaded_weight_blob": model["blob_path"],
            }
        ),
        mesh_path_resolver=mesh_path_resolver
        or (lambda _dataset, uid: mesh_paths[uid]),
    )


def _verify(
    fixture: dict[str, object],
    *,
    model_loader=None,
    weight_resolver=None,
    mesh_path_resolver=None,
) -> dict:
    weight = fixture["weight"]
    mesh_paths = fixture["mesh_paths"]
    return preflight.verify_output(
        fixture["output"],
        fixture["dataset"],
        fixture["preprocessed"],
        minimum_rows=fixture["rows"],
        maximum_rows=fixture["rows"],
        sample_count=3,
        weight_resolver=weight_resolver or (lambda: _weight_evidence(weight)),
        model_loader=model_loader
        or (
            lambda model, _query: {
                "embedding": _normalized_embedding(),
                "device": "cpu",
                "loaded_weight_blob": model["blob_path"],
            }
        ),
        mesh_path_resolver=mesh_path_resolver
        or (lambda _dataset, uid: mesh_paths[uid]),
    )


def test_pass_binds_indices_samples_pinned_weight_and_repeats_offline_load(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    calls = {"resolve": 0, "load": 0}

    def resolve_weight() -> dict[str, object]:
        calls["resolve"] += 1
        return _weight_evidence(fixture["weight"])

    def load_model(model: dict, query: str) -> dict[str, object]:
        calls["load"] += 1
        assert query == preflight.SMOKE_QUERY
        assert all(os_value == "1" for os_value in preflight._set_offline_environment().values())
        return {
            "embedding": _normalized_embedding(),
            "device": "cpu",
            "loaded_weight_blob": model["blob_path"],
        }

    result = _run(fixture, weight_resolver=resolve_weight, model_loader=load_model)

    assert result["status"] == "pass", result.get("error")
    assert result["dataset"]["row_count"] == fixture["rows"]
    assert result["dataset"]["embedding_dimensions"] == 768
    assert result["model"]["repo_id"] == preflight.MODEL_REPO_ID
    assert result["model"]["requested_revision"] == preflight.MODEL_REVISION
    assert result["model"]["filename"] == preflight.MODEL_FILENAME
    roles = {artifact["role"] for artifact in result["evidence"]["artifacts"]}
    assert {f"index:{name}" for name in preflight.REQUIRED_INDEX_FILES} <= roles
    assert "openclip_weight_blob" in roles
    assert len([role for role in roles if role.startswith("sample_mesh:")]) == 3
    assert calls == {"resolve": 1, "load": 1}

    verified = _verify(
        fixture, weight_resolver=resolve_weight, model_loader=load_model
    )

    assert verified["status"] == "pass", verified.get("verification_error")
    assert verified["verification"]["cache_only_model_load_repeated"] is True
    assert calls == {"resolve": 2, "load": 2}


def test_index_row_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["preprocessed"] / "embedding_index.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "- extra-uid\n", encoding="utf-8")

    result = _run(fixture)

    assert result["status"] == "fail"
    assert "do not contain the same UIDs" in result["error"]


def test_wrong_embedding_dimension_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, dimensions=512)

    result = _run(fixture)

    assert result["status"] == "fail"
    assert "512 dimensions, expected 768" in result["error"]


def test_missing_exact_production_mesh_path_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = _run(
        fixture,
        mesh_path_resolver=lambda dataset, uid: dataset / "assets" / uid / "missing.glb",
    )

    assert result["status"] == "fail"
    assert "Production ObjectThor mesh path is missing" in result["error"]


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        (np.ones(16, dtype=np.float32), "has shape"),
        (
            np.full(preflight.EMBEDDING_DIMENSION, np.nan, dtype=np.float32),
            "non-finite",
        ),
        (
            np.ones(preflight.EMBEDDING_DIMENSION, dtype=np.float32),
            "not normalized",
        ),
    ],
)
def test_invalid_cpu_text_embedding_fails_closed(
    tmp_path: Path, embedding: np.ndarray, message: str
) -> None:
    fixture = _fixture(tmp_path)

    result = _run(
        fixture,
        model_loader=lambda model, _query: {
            "embedding": embedding,
            "device": "cpu",
            "loaded_weight_blob": model["blob_path"],
        },
    )

    assert result["status"] == "fail"
    assert message in result["error"]


def test_wrong_pinned_model_identity_fails_before_load(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    calls = {"load": 0}
    identity = _weight_evidence(fixture["weight"])
    identity["requested_revision"] = "main"

    def load_model(_model, _query):
        calls["load"] += 1
        return _normalized_embedding()

    result = _run(
        fixture, weight_resolver=lambda: identity, model_loader=load_model
    )

    assert result["status"] == "fail"
    assert preflight.MODEL_REVISION in result["error"]
    assert calls["load"] == 0


def test_verify_only_rehashes_weight_and_repeats_model_load(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    assert result["status"] == "pass"
    fixture["weight"].write_bytes(b"mutated-weight")
    calls = {"load": 0}

    def load_model(model, _query):
        calls["load"] += 1
        return {
            "embedding": _normalized_embedding(),
            "device": "cpu",
            "loaded_weight_blob": model["blob_path"],
        }

    verified = _verify(fixture, model_loader=load_model)

    assert verified["status"] == "fail"
    assert "evidence differs" in verified["verification_error"]
    assert calls["load"] == 1


def test_verify_only_rejects_index_mutation_and_still_repeats_model_load(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    assert result["status"] == "pass"
    metadata_path = fixture["preprocessed"] / "metadata_index.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[fixture["uids"][0]]["description"] = "changed after pass"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls = {"load": 0}

    def load_model(model, _query):
        calls["load"] += 1
        return {
            "embedding": _normalized_embedding(),
            "device": "cpu",
            "loaded_weight_blob": model["blob_path"],
        }

    verified = _verify(fixture, model_loader=load_model)

    assert verified["status"] == "fail"
    assert "evidence differs" in verified["verification_error"]
    assert calls["load"] == 1


def test_json_tampering_breaks_attestation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    assert result["status"] == "pass"
    result["model_smoke"]["embedding_norm"] = math.pi
    fixture["output"].write_text(json.dumps(result), encoding="utf-8")

    verified = _verify(fixture)

    assert verified["status"] == "fail"
    assert "attestation does not match" in verified["verification_error"]
