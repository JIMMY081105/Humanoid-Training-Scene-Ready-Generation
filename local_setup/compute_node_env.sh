#!/usr/bin/env bash
# Portable, repository-owned environment bootstrap for SLURM compute jobs.
# This file changes environment variables only; it never changes the caller's cwd.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' \
        "ERROR: local_setup/compute_node_env.sh must be sourced, not executed." >&2
    exit 64
fi

__scenesmith_compute_node_env_main() {
    local source_path source_dir repo_root venv_dir activate_path python_bin node_name
    local cache_root current_pythonpath

    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        printf '%s\n' \
            "ERROR: SLURM_JOB_ID is required; run this only inside a SLURM compute allocation." >&2
        return 65
    fi
    if ! node_name="$(hostname 2>/dev/null)"; then
        printf '%s\n' "ERROR: cannot identify the current SLURM compute node." >&2
        return 65
    fi
    case "${node_name}" in
        ln* | Ln* | lN* | LN*)
            printf 'ERROR: refusing login node hostname=%q.\n' \
                "${node_name}" >&2
            return 65
            ;;
    esac

    source_path="${BASH_SOURCE[0]}"
    if [[ -n "${REPO_ROOT:-}" ]]; then
        if ! repo_root="$(cd -- "${REPO_ROOT}" 2>/dev/null && pwd -P)"; then
            printf 'ERROR: REPO_ROOT is not an accessible directory: %q\n' \
                "${REPO_ROOT}" >&2
            return 66
        fi
    else
        if ! source_dir="$(cd -- "$(dirname -- "${source_path}")" 2>/dev/null && pwd -P)"; then
            printf 'ERROR: cannot resolve compute environment helper directory: %q\n' \
                "${source_path}" >&2
            return 66
        fi
        if ! repo_root="$(cd -- "${source_dir}/.." 2>/dev/null && pwd -P)"; then
            printf 'ERROR: cannot derive repository root from: %q\n' \
                "${source_path}" >&2
            return 66
        fi
    fi

    venv_dir="${repo_root}/.venv"
    activate_path="${venv_dir}/bin/activate"
    python_bin="${venv_dir}/bin/python"
    if [[ ! -d "${venv_dir}" || ! -f "${venv_dir}/pyvenv.cfg" ]]; then
        printf 'ERROR: repository virtual environment is missing or invalid: %q\n' \
            "${venv_dir}" >&2
        return 67
    fi
    if [[ ! -f "${activate_path}" || ! -r "${activate_path}" ]]; then
        printf 'ERROR: virtual-environment activation script is missing: %q\n' \
            "${activate_path}" >&2
        return 67
    fi
    if [[ ! -f "${python_bin}" || ! -x "${python_bin}" ]]; then
        printf 'ERROR: virtual-environment Python is missing or not executable: %q\n' \
            "${python_bin}" >&2
        return 67
    fi

    # The activation script is sourced deliberately so its PATH and VIRTUAL_ENV
    # changes remain in the caller, just like this bootstrap's exported values.
    if ! source "${activate_path}"; then
        printf 'ERROR: failed to activate virtual environment: %q\n' \
            "${activate_path}" >&2
        return 68
    fi
    if [[ "${VIRTUAL_ENV:-}" != "${venv_dir}" ]]; then
        printf 'ERROR: activation selected unexpected VIRTUAL_ENV: %q (expected %q)\n' \
            "${VIRTUAL_ENV:-}" "${venv_dir}" >&2
        return 68
    fi
    if ! "${python_bin}" -c \
        'import os, sys; raise SystemExit(0 if os.path.realpath(sys.prefix) == os.path.realpath(sys.argv[1]) else 1)' \
        "${venv_dir}"; then
        printf 'ERROR: Python does not identify as the repository virtual environment: %q\n' \
            "${python_bin}" >&2
        return 68
    fi

    export REPO_ROOT="${repo_root}"
    export PYTHON_BIN="${python_bin}"

    current_pythonpath="${PYTHONPATH:-}"
    case ":${current_pythonpath}:" in
        *":${repo_root}:"*) ;;
        *) export PYTHONPATH="${repo_root}${current_pythonpath:+:${current_pythonpath}}" ;;
    esac

    # Cache locations are portable defaults. Every setting remains independently
    # caller-overridable, and this bootstrap does not create or populate them.
    cache_root="${SCENESMITH_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME:-${repo_root}}/.cache}}"
    export SCENESMITH_CACHE_ROOT="${cache_root}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}}"
    export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
    export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/uv}"

    printf '[compute-node] repo=%s venv=%s slurm_job_id=%s\n' \
        "${REPO_ROOT}" "${VIRTUAL_ENV}" "${SLURM_JOB_ID}"
}

if __scenesmith_compute_node_env_main; then
    unset -f __scenesmith_compute_node_env_main
else
    __scenesmith_compute_node_env_status=$?
    unset -f __scenesmith_compute_node_env_main
    return "${__scenesmith_compute_node_env_status}"
fi
unset __scenesmith_compute_node_env_status 2>/dev/null || true
return 0
