from __future__ import annotations

import os
import shutil
import subprocess

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "local_setup" / "compute_node_env.sh"


def _usable_bash() -> str:
    if os.name == "nt":
        pytest.skip("behavioral Bash tests require a POSIX filesystem")
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is unavailable")
    probe = subprocess.run(
        [executable, "--noprofile", "--norc", "-c", "printf ok"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode != 0 or probe.stdout != "ok":
        pytest.skip("bash is not functional")
    return executable


def _make_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "portable repo"
    local_setup = repo / "local_setup"
    bin_dir = repo / ".venv" / "bin"
    local_setup.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    shutil.copyfile(HELPER, local_setup / HELPER.name)
    (repo / ".venv" / "pyvenv.cfg").write_text(
        "home = /fixture/interpreter\n",
        encoding="utf-8",
    )
    (bin_dir / "activate").write_text(
        f"VIRTUAL_ENV={shlex_quote(os.fspath(repo / '.venv'))}\n"
        "export VIRTUAL_ENV\n"
        "PATH=\"${VIRTUAL_ENV}/bin:${PATH}\"\n"
        "export PATH\n",
        encoding="utf-8",
    )
    python = bin_dir / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "# The bootstrap's validation probe is the only supported fixture call.\n"
        "[[ \"${1:-}\" == '-c' ]]\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return repo


def shlex_quote(value: str) -> str:
    """Quote one static fixture value without importing a shell implementation."""

    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_bash(
    executable: str,
    script: str,
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        [executable, "--noprofile", "--norc", "-c", script],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_helper_is_secret_free_and_contains_no_network_or_install_actions() -> None:
    text = HELPER.read_text(encoding="utf-8")

    assert "${BASH_SOURCE[0]}" in text
    assert "${SLURM_JOB_ID:-}" in text
    assert "${REPO_ROOT:-}" in text
    assert "export PYTHON_BIN=" in text
    assert "export PYTHONPATH=" in text
    for forbidden in (
        "/root/",
        "/mnt/",
        "/data/",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "OPENAI_API_KEY",
        "HF_ENDPOINT",
        "pip install",
        "uv sync",
        "wget ",
        "curl ",
    ):
        assert forbidden not in text


def test_helper_passes_bash_syntax_check() -> None:
    bash = _usable_bash()
    result = subprocess.run(
        [bash, "-n", HELPER],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_source_activates_exact_repo_venv_without_changing_cwd_and_honors_overrides(
    tmp_path: Path,
) -> None:
    bash = _usable_bash()
    repo = _make_fixture_repo(tmp_path)
    caller_cwd = tmp_path / "caller cwd"
    caller_cwd.mkdir()
    helper = repo / "local_setup" / HELPER.name
    script = f"""
set -euo pipefail
before="$PWD"
source {shlex_quote(os.fspath(helper))}
printf 'cwd=%s\n' "$PWD"
printf 'before=%s\n' "$before"
printf 'repo=%s\n' "$REPO_ROOT"
printf 'python=%s\n' "$PYTHON_BIN"
printf 'pythonpath=%s\n' "$PYTHONPATH"
printf 'hf=%s\n' "$HF_HOME"
printf 'torch=%s\n' "$TORCH_HOME"
printf 'uv=%s\n' "$UV_CACHE_DIR"
"""
    result = _run_bash(
        bash,
        script,
        cwd=caller_cwd,
        environment={
            "SLURM_JOB_ID": "123456",
            "REPO_ROOT": os.fspath(repo),
            "PYTHONPATH": "/caller/pythonpath",
            "HF_HOME": "/caller/hf",
            "TORCH_HOME": "/caller/torch",
            "UV_CACHE_DIR": "/caller/uv",
        },
    )

    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["cwd"] == os.fspath(caller_cwd)
    assert values["before"] == os.fspath(caller_cwd)
    assert values["repo"] == os.fspath(repo)
    assert values["python"] == os.fspath(repo / ".venv" / "bin" / "python")
    assert values["pythonpath"] == f"{repo}:/caller/pythonpath"
    assert values["hf"] == "/caller/hf"
    assert values["torch"] == "/caller/torch"
    assert values["uv"] == "/caller/uv"


def test_source_derives_repo_root_from_bash_source(tmp_path: Path) -> None:
    bash = _usable_bash()
    repo = _make_fixture_repo(tmp_path)
    helper = repo / "local_setup" / HELPER.name
    result = _run_bash(
        bash,
        f"set -euo pipefail; unset REPO_ROOT; source {shlex_quote(os.fspath(helper))}; printf '%s' \"$REPO_ROOT\"",
        cwd=tmp_path,
        environment={"SLURM_JOB_ID": "123456"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith(os.fspath(repo))


def test_direct_execution_is_rejected(tmp_path: Path) -> None:
    bash = _usable_bash()
    repo = _make_fixture_repo(tmp_path)
    helper = repo / "local_setup" / HELPER.name
    result = _run_bash(
        bash,
        f"bash {shlex_quote(os.fspath(helper))}",
        cwd=tmp_path,
        environment={"SLURM_JOB_ID": "123456", "REPO_ROOT": os.fspath(repo)},
    )

    assert result.returncode == 64
    assert "must be sourced" in result.stderr


def test_source_rejects_missing_slurm_job_id(tmp_path: Path) -> None:
    bash = _usable_bash()
    repo = _make_fixture_repo(tmp_path)
    helper = repo / "local_setup" / HELPER.name
    result = _run_bash(
        bash,
        f"set -euo pipefail; if source {shlex_quote(os.fspath(helper))}; then exit 99; else printf 'status=%s' \"$?\"; fi",
        cwd=tmp_path,
        environment={"SLURM_JOB_ID": "", "REPO_ROOT": os.fspath(repo)},
    )

    assert result.returncode == 0
    assert result.stdout == "status=65"


@pytest.mark.parametrize("node_name", ["ln01", "LN-login"])
def test_source_rejects_login_node_hostname(
    tmp_path: Path, node_name: str
) -> None:
    bash = _usable_bash()
    repo = _make_fixture_repo(tmp_path)
    helper = repo / "local_setup" / HELPER.name
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    hostname = fake_bin / "hostname"
    hostname.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' {shlex_quote(node_name)}\n",
        encoding="utf-8",
    )
    hostname.chmod(0o755)
    result = _run_bash(
        bash,
        f"set -euo pipefail; if source {shlex_quote(os.fspath(helper))}; then exit 99; else printf 'status=%s' \"$?\"; fi",
        cwd=tmp_path,
        environment={
            "SLURM_JOB_ID": "123456",
            "REPO_ROOT": os.fspath(repo),
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        },
    )

    assert result.returncode == 0
    assert result.stdout == "status=65"


def test_source_rejects_missing_repo_venv(tmp_path: Path) -> None:
    bash = _usable_bash()
    repo = tmp_path / "repo"
    (repo / "local_setup").mkdir(parents=True)
    helper = repo / "local_setup" / HELPER.name
    shutil.copyfile(HELPER, helper)
    result = _run_bash(
        bash,
        f"set -euo pipefail; if source {shlex_quote(os.fspath(helper))}; then exit 99; else printf 'status=%s' \"$?\"; fi",
        cwd=tmp_path,
        environment={"SLURM_JOB_ID": "123456", "REPO_ROOT": os.fspath(repo)},
    )

    assert result.returncode == 0
    assert result.stdout == "status=67"
