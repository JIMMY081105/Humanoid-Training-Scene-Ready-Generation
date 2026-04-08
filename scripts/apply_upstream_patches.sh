#!/usr/bin/env bash
# Apply the SceneSmith patch stack in its verified dependency order.

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 /path/to/clean/scenesmith" >&2
    exit 2
fi

REPO="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PATCH_DIR="${PROJECT_ROOT}/upstream-patches"
MANIFEST="${PATCH_DIR}/APPLY_ORDER.txt"

git -C "${REPO}" rev-parse --is-inside-work-tree >/dev/null
if [[ -n "$(git -C "${REPO}" status --porcelain)" ]]; then
    echo "ERROR: target checkout is not clean; refusing a partial/ambiguous patch stack." >&2
    exit 3
fi

base_commit="$(git -C "${REPO}" rev-parse HEAD)"
if [[ "${base_commit}" != 9df3f88* ]]; then
    echo "ERROR: expected SceneSmith base 9df3f88, found ${base_commit}." >&2
    exit 4
fi

applied=0
while IFS= read -r patch_name; do
    [[ -z "${patch_name}" || "${patch_name}" == \#* ]] && continue
    patch_path="${PATCH_DIR}/${patch_name}"
    if [[ ! -s "${patch_path}" ]]; then
        echo "ERROR: manifest patch is missing or empty: ${patch_path}" >&2
        exit 5
    fi
    echo "[patch $((applied + 1))] ${patch_name}"
    git -C "${REPO}" apply --3way "${patch_path}"
    applied=$((applied + 1))
done <"${MANIFEST}"

manifest_count="$(grep -cEv '^[[:space:]]*(#|$)' "${MANIFEST}")"
if (( applied != manifest_count )); then
    echo "ERROR: applied ${applied} patches but manifest contains ${manifest_count}." >&2
    exit 6
fi

install_overlay() {
    local source_path="$1"
    local relative_destination="$2"
    local destination="${REPO}/${relative_destination}"
    if [[ ! -s "${source_path}" ]]; then
        echo "ERROR: required overlay is missing or empty: ${source_path}" >&2
        exit 7
    fi
    if [[ -e "${destination}" ]] && ! cmp -s "${source_path}" "${destination}"; then
        echo "ERROR: refusing to overwrite a different overlay file: ${destination}" >&2
        exit 8
    fi
    mkdir -p "$(dirname "${destination}")"
    install -m 0644 "${source_path}" "${destination}"
    echo "[overlay] ${relative_destination}"
}

install_overlay \
    "${PROJECT_ROOT}/scenesmith/agent_utils/codex_vlm_backend.py" \
    "scenesmith/agent_utils/codex_vlm_backend.py"
install_overlay \
    "${PROJECT_ROOT}/_remote_patch/scenesmith/agent_utils/codex_cli.py" \
    "scenesmith/agent_utils/codex_cli.py"

echo "Applied ${applied} patches and 2 required overlays in verified order. Review and commit the changes."
