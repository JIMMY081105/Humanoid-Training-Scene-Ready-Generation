from __future__ import annotations

from pathlib import Path

from codex_benchmark.cache import CacheManager, hash_files, sha256_file, sha256_text
from codex_benchmark.checkpoint import CheckpointStore
from codex_benchmark.config import load_config, quick_config


def test_config_loads_project_paths() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")

    assert config.name == "codex_cli_autonomy_replacement"
    assert Path(config.paths.database).is_absolute()
    assert config.stress.call_counts[-1] == 2000


def test_quick_config_reduces_real_call_counts() -> None:
    config = quick_config(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))

    assert config.stress.call_counts == [2]
    assert config.structured.count == 2
    assert config.resume.total_calls == 3


def test_cache_key_is_stable_and_prompt_sensitive(tmp_path: Path) -> None:
    config = quick_config(load_config(Path(__file__).resolve().parents[1] / "config.yaml"))
    config.paths.database = str(tmp_path / "benchmark.sqlite3")
    store = CheckpointStore(config.paths.database)
    try:
        cache = CacheManager(store, config)
        key1, prompt_hash1, _, _ = cache.make_key(module="stress", prompt="hello")
        key2, prompt_hash2, _, _ = cache.make_key(module="stress", prompt="hello")
        key3, prompt_hash3, _, _ = cache.make_key(module="stress", prompt="different")
    finally:
        store.close()

    assert key1 == key2
    assert prompt_hash1 == prompt_hash2
    assert key1 != key3
    assert prompt_hash1 != prompt_hash3


def test_sha256_file_matches_text_hash(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("benchmark cache input", encoding="utf-8")

    assert sha256_file(sample) == sha256_text("benchmark cache input")


def test_hash_files_records_missing_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"

    missing_hash = hash_files([missing])

    assert missing_hash is not None
    assert missing_hash == hash_files([missing])
    assert missing_hash != hash_files([])
