#!/usr/bin/env bash
# Publish a non-hard-linked, isolated Artiverse tree for strict preparation.
# This is a byte copy because JuiceFS has no reflink support and Artiverse
# deliberately rejects shared inodes.  It never alters the source dataset.

#SBATCH -J artiverse_source_copy
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 1-00:00:00
#SBATCH -o /data/run01/scvj260/codex_factory/logs/%x-%j.out

set -euo pipefail

readonly FACTORY=/data/run01/scvj260/codex_factory
readonly SOURCE=/data/run01/scvj260/scenesmith-factory-codex/data/artiverse
readonly RUNTIME="${FACTORY}/artiverse_runtime_v3"

case "$(hostname)" in
  ln*|Ln*|lN*|LN*)
    echo "Refusing Artiverse source copy on a login node." >&2
    exit 65
    ;;
esac

test -d "${SOURCE}/data"
test -s "${SOURCE}/artiverse_safe_extraction_receipt.json"
test -s "${SOURCE}/dataset_chunks/manifest.json"
test ! -e "${RUNTIME}"

tmp="$(mktemp -d "${FACTORY}/.artiverse_runtime.copy.XXXXXX")"
trap 'rm -rf "${tmp}"' EXIT
# The two authenticated tar archives are 65.3 GB of acquisition evidence.  The
# verified extraction receipt binds their exact hashes and is what the strict
# preparation contract authenticates; no preparation stage reads their bytes.
# Copy only the three school-required categories and the pinned proof inputs.
# ``prepare_artiverse.py`` receives the same explicit category allow-list, and
# the authenticated extraction receipt remains the authority for the complete
# publisher release without copying irrelevant categories into this workspace.
mkdir "${tmp}/data"
for category in armoire bookcase chest_of_drawers; do
  test -d "${SOURCE}/data/${category}"
  cp -a --reflink=never "${SOURCE}/data/${category}" "${tmp}/data/${category}"
done
mkdir "${tmp}/dataset_chunks"
cp -a --reflink=never "${SOURCE}/dataset_chunks/manifest.json" \
  "${tmp}/dataset_chunks/manifest.json"
cp -a --reflink=never "${SOURCE}/artiverse_safe_extraction_receipt.json" \
  "${tmp}/artiverse_safe_extraction_receipt.json"
cp -a --reflink=never "${SOURCE}/pack_dataset_chunks.py" \
  "${tmp}/pack_dataset_chunks.py"
if find "${tmp}" -type f -links +1 -print -quit | grep -q .; then
  echo "Artiverse source copy contains hard-linked regular files." >&2
  exit 66
fi
test -d "${tmp}/data"
test -s "${tmp}/artiverse_safe_extraction_receipt.json"
test -s "${tmp}/dataset_chunks/manifest.json"
touch "${tmp}/.source_copy_complete"
mv "${tmp}" "${RUNTIME}"
trap - EXIT
