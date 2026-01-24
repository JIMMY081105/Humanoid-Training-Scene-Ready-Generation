# SQZ to ParaCloud migration handoff

Status date: 2026-07-11 (+08). This document is a plan only. **Receiving or
reading this document is not authorization to execute it.** No migration,
deletion, job submission, package installation, remote authentication change,
or large transfer has been authorized or started.

Permission classes used below:

- `READ_ONLY`: inspection or a stream discarded at `/dev/null`.
- `METADATA_WRITE`: small manifests/inventories on SQZ or ParaCloud; no payload transfer.
- `REMOTE_AUTH_WRITE`: temporary public-key/known-host setup.
- `REMOTE_STAGE_WRITE`: creation of new, non-overwriting ParaCloud paths.
- `ENVIRONMENT_WRITE`: creation of the fresh ParaCloud `.venv` and isolated
  `.mujoco_venv` from frozen/hashed dependency inputs.
- `LARGE_ASSET_WRITE`: any project/asset transfer, extraction, preparation, or
  large ParaCloud-to-ParaCloud preservation copy.
- `VALIDATION_WRITE`: small receipts, manifests, links, and validation templates
  written after transferred bytes have been checked.
- `GPU_JOB`: any SLURM submission.
- `CLEANUP`: key removal or retirement work.

Claude Code must obtain an explicit user approval naming the permission class
before the first command in that class. Approval for one class does not imply
approval for another. `CLEANUP` never includes legacy data deletion; deletion
requires a separate, path-and-byte-count-specific user instruction.

Mandatory execution order (the letters are authoritative; do not simply read
the numbered sections top-to-bottom):

```text
A READ_ONLY preflight
  -> B METADATA_WRITE inventories + frozen capacity inputs
  -> C REMOTE_AUTH_WRITE temporary direct SSH setup
  -> D READ_ONLY speed test + exact quota/peak ledger
  -> E REMOTE_STAGE_WRITE fresh empty roots
  -> F LARGE_ASSET_WRITE code/outputs + exact destination comparison
  -> J ENVIRONMENT_WRITE both offline environments
  -> G LARGE_ASSET_WRITE/VALIDATION_WRITE/GPU_JOB Artiverse branch
  -> H LARGE_ASSET_WRITE/VALIDATION_WRITE ObjectThor branch
  -> I LARGE_ASSET_WRITE/VALIDATION_WRITE materials/ArtVIP/HSSD/external/SAM cache branch
  -> K VALIDATION_WRITE CPU acceptance + GPU_JOB two-GPU/SAM3D preflight
  -> acceptance report
  -> L CLEANUP temporary credentials only
```

Within G, H, and I, run source validation, the immediately preceding capacity
guard, transfer/materialization, destination validation, and link creation in
the written order. Do not start a later letter merely because bytes from an
earlier letter arrived. There is no "transfer now, validate later" exception.

## 0. Objective and non-negotiable rules

Move only the SceneSmith reference-school project and its required assets from
SQZ to ParaCloud so a larger GPU allocation can be used. Do not copy unrelated
collaborator projects and do not delete the existing ParaCloud checkout until a
separate, explicit deletion approval is given after every validation gate below
passes.

Hard rules:

1. Read `CODEX_SQZ_REMOTE_HANDOFF.md` and
   `CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md` completely before acting.
2. The Windows `SQZ` VPN must remain split-tunneled. Only `10.220.0.0/16`
   may use it. Never change the default route, Misty/v2rayN, DNS, or proxy.
3. Never use `rsync --delete`, `scp -r /root/workspace`, or a recursive copy of
   all SQZ projects.
4. Never copy or print `.env`, `.secrets`, Hugging Face tokens, OpenAI keys,
   VPN credentials, or SSH private keys. Configure secrets separately.
5. Never overwrite `/data/run01/scvj260/scenesmith`. It contains useful legacy
   assets and history. Stage the new checkout separately.
6. Never write large data under `/data/home/scvj260`; that 1 GiB filesystem is
   effectively full. Use `/data/run01/scvj260` only.
7. Never run generation on a ParaCloud login node. Use SLURM.
8. Do not start paid APIs or the full scene pipeline during migration.
9. Do not trust a copied pass receipt by itself. Rehash and rerun every offline
   verifier on ParaCloud.
10. Do not delete project, asset, output, legacy, or SQZ data at the end of this
    plan. Exact temporary credential files may be removed only under separate
    `CLEANUP` approval. The final section is a data-deletion-readiness checklist,
    not deletion authorization.

## 1. Verified current state

### Connections

- SQZ alias: `ssh sqz`, currently `root@10.220.5.94:32378` through the SQZ VPN.
- At final handoff review, `rasdial` reported **no active connections** and an
  SQZ SSH syntax-check stream timed out. Treat SQZ as disconnected until Phase A
  safely re-establishes and verifies the split tunnel per
  `CODEX_SQZ_REMOTE_HANDOFF.md`; never turn it into a full tunnel.
- ParaCloud alias: `ssh paracloud`.
- ParaCloud login was reverified on 2026-07-11:
  - login node: `ln08`
  - user: `scvj260`
  - home: `/data/home/scvj260`
  - SLURM: available
  - existing checkout: `/data/run01/scvj260/scenesmith`
  - GPU partition snapshot: 29 of 48 GPUs available; a two-GPU allocation was
    schedulable at the time of the check.

### Storage

ParaCloud `/data` is JuiceFS. The observed filesystem state was approximately
1.8 TiB total, 1.5 TiB used, and 348 GiB free. A personal quota could not be
verified because the normal `quota` utility was unavailable. Treat 348 GiB as
filesystem availability, not a guaranteed account quota.

SQZ pipeline-owned apparent sizes:

| Path | Apparent bytes | Approx. GiB | Treatment |
|---|---:|---:|---|
| `/root/workspace/scenesmith-hts` | 1,508,719,735 | 1.41 | copy exact code/inputs/outputs, excluding external asset links and secrets |
| `/root/workspace/Humanoid-Training-Scene-Ready-Generation` | 17,106,325 | 0.02 | copy handoff documents/scripts if desired |
| `/root/workspace/miniconda3` | 111,030,576,922 | 103.40 | **do not copy**; environments are path-bound and include collaborator envs |
| `/root/.objathor-assets/2023_09_23` | 58,250,947,884 | 54.25 | copy the ready payload only; omit known interrupted/staging residue |
| `/root/.cache` | 72,557,003,925 | 67.57 | do not copy blindly; copy only receipt-bound missing model/cache files |
| `/localssd/scenesmith-hts-assets/artiverse` | 152,442,283,716 | 141.97 | prefer the two pinned archives and safe re-extraction |

A literal copy of all of the above is about 368.6 GiB and does not fit safely
inside the observed 348 GiB free space. The revised plan also preserves stable
copies of legacy materials, ArtVIP, and HSSD, so its provisional final-new-data
range is closer to 280-300 GiB including a fresh `.venv`; the exact filtered
ObjectThor/cache manifests determine the real value. Its direct SQZ network
payload should be nearer 130-150 GiB because the three useful legacy trees are
copied within ParaCloud and Artiverse travels as two archives. These are planning
ranges only, not a capacity pass.

Those figures are not yet a write authorization or a peak-space proof. Before
remote staging, Claude Code must calculate all four values from the final
allowlist:

```text
network_transfer_bytes
final_new_allocated_bytes
worst_case_temporary_bytes (rsync partials + extraction/preparation staging)
safety_reserve_bytes (minimum 50 GiB)
```

Require both confirmed account quota and current filesystem availability to be
at least `final_new_allocated_bytes + worst_case_temporary_bytes + 50 GiB`.
Account for hardlink expansion if a transfer cannot preserve hardlinks. If the
exact calculation does not fit, stop; do not delete legacy data to make it fit.

### Authoritative asset facts

Artiverse on SQZ:

- root: `/localssd/scenesmith-hts-assets/artiverse`
- repository: `/localssd/scenesmith-hts-assets/artiverse/repository`
- publisher revision: `8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248`
- archive 1:
  - bytes: `38163580631`
  - SHA-256: `695d2d602faafab922ce66359ea104d81505f5b0fdee8f461d8905f0ccb4ef3b`
- archive 2:
  - bytes: `27170560473`
  - SHA-256: `56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81`
- extracted publisher tree: 3,544 roots, 531,937 regular files,
  86,992,752,890 bytes
- prepared school-role index: 500 of 560 candidates passed
- only `scripts/safe_extract_artiverse.py` may extract these archives; never
  execute the publisher unpack command.

ObjectThor on SQZ:

- root: `/root/.objathor-assets/2023_09_23`
- `data/objathor-assets` resolves to that root
- prepared index: 50,092 objects by 768 dimensions
- raw publisher files include:
  - `assets.tar` expected bytes `23177386496`
  - `annotations.json.gz` expected bytes `9740343`
  - `features.tar` expected bytes `388221440`
- the exact retrieval model is
  `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`, revision
  `1627032197142fbe2a7cfec626f4ced3ae60d07a`.

SAM3D:

- complete offline initialization passed on the SQZ A10
- evidence:
  `outputs/preflight/full_quality_school_reference_20260710/sam3d_offline_load.json`
- that receipt binds 17 checkpoint/config/cache artifacts
- do not copy all of `/root/.cache`; derive the exact bounded roots from this
  receipt and place them in the fresh migration-specific cache.

### Useful ParaCloud legacy content

The existing `/data/run01/scvj260/scenesmith` must not be deleted blindly. It
contains at least:

| Path under legacy checkout | Apparent bytes | Notes |
|---|---:|---|
| `data` | 116,600,147,761 | several datasets |
| `data/hssd-models` | 3,974,431,255 | HSSD is absent on SQZ; preserve it |
| `data/artiverse` | 52,729,828,438 | older/smaller than authoritative SQZ state; do not treat as current |
| `data/artvip_sdf` | 8,912,979,126 | likely reusable after contract/hash validation |
| `data/materials` | 47,439,068,466 | likely reusable after materials-contract validation |
| `external/checkpoints` | 16,555,926,365 | compare against the 17-artifact SAM3D receipt |
| `.venv` | 17,178,649,731 | working legacy environment; path-bound |
| `transfer_chunks` | 31,218,325,203 | historical transfer bundles; preserve until reviewed |
| `transfer_chunks/final_school_floor` | 1,464,446,892 | historical school result |

The legacy checkout also has historical outputs, `.secrets`, SAM3/SAM3D source,
and old run scripts. `.secrets` is sensitive and must be handled out of band.

### Current scene-generation status

The handoff sentence saying the classroom benchmark is active is stale. Attempt
`20260711T094756Z-844` was intentionally stopped after its first floor renders
failed because a staged GLTF texture URI contained eight `../` components and
resolved to `/root/workspace/materials` instead of the real
`/root/workspace/scenesmith-hts/materials`.

The repair changes the deterministic reference-layout materializer to use a
same-depth sibling transaction and to resolve/hash every local GLTF buffer and
image dependency before and after publication. Focused remote tests passed
10/10. The broken scene must still be non-destructively archived, regenerated,
render-smoked, and the benchmark restarted. Do not describe the school as
complete and do not transfer a final acceptance package yet.

## 2. Recommended migration shape

Use a new ParaCloud staging checkout:

```bash
/data/run01/scvj260/scenesmith-hts.stage
```

Use shared asset roots outside either checkout:

```bash
/data/run01/scvj260/assets/artiverse_sqz_20260711
/data/run01/scvj260/assets/objathor_2023_09_23
/data/run01/scvj260/assets/external_sqz_20260711
/data/run01/scvj260/assets/materials_sqz_20260711
/data/run01/scvj260/assets/artvip_sqz_20260711
/data/run01/scvj260/assets/hssd_legacy
/data/run01/scvj260/cache_sqz_20260711
```

Keep the legacy checkout at `/data/run01/scvj260/scenesmith` unchanged while
staging and validation occur.

Recommended payload profile:

1. exact secret-scanned current `scenesmith-hts` worktree payload, inputs,
   tests, specifications, patch stack, and current goal outputs, with no `.git`
   database or historical refs;
2. the two compressed Artiverse archives plus pinned manifest/pack-script, then
   safe-extract and prepare on ParaCloud;
3. the filtered full ObjectThor ready payload (`assets`, `features`,
   `preprocessed`, and the three canonical source files), plus its exact
   receipt-bound retrieval-model cache; do **not** use the raw-only rebuild route;
4. only the receipt-bound ObjectThor/SAM3D cache roots, all under a new
   migration-specific cache rather than the legacy `.cache`;
5. copy verified ParaCloud ArtVIP and materials into stable shared roots rather
   than linking the new checkout back into the legacy checkout;
6. preserve HSSD now even though it is not required by the current policy, and
   keep the legacy historical outputs/transfer chunks untouched until the user
   makes an explicit retention decision;
7. create a fresh `.venv` at the new absolute path using the lockfile and the
   existing ParaCloud UV cache in offline mode. Do not copy or symlink a
   path-bound virtualenv.

Do not migrate:

- `/root/workspace/miniconda3`
- collaborator projects under `/root/workspace`
- SQZ system Python or OS files
- transient agent databases, sockets, locks, `__pycache__`, or GPU caches not
  named by a passing receipt
- any secret-bearing file
- the anomalous top-level entries ` \\` and `%ln`, broken handoff trees,
  `hf_dl`, `.pytest_cache`, `.mujoco_venv`, or any unapproved symlink

## 3. Phase A - read-only preflight (`READ_ONLY`)

Run from the Windows repository:

```powershell
# SQZ VPN safety: inspect, do not modify unless the handoff says it is wrong.
Get-VpnConnection -Name SQZ |
  Select-Object ServerAddress,TunnelType,SplitTunneling,ConnectionStatus,RememberCredential
(Get-VpnConnection -Name SQZ).Routes
route print 0.0.0.0 | Select-String 0.0.0.0
curl.exe --noproxy "*" -s --max-time 8 -o NUL -w "%{http_code}" https://www.bing.com

ssh -o BatchMode=yes sqz "hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv"
ssh -o BatchMode=yes paracloud "hostname; id; pwd; squeue -u scvj260"
```

Expected: SQZ split tunnel remains enabled; ordinary internet works; SQZ and
ParaCloud both connect. Stop if any condition fails.

On SQZ, prove no generation or transfer is active:

```bash
pgrep -af 'run_full_quality_school_sqz|run_single_room_worker|rsync|scp|safe_extract_artiverse' || true
tmux list-sessions 2>/dev/null || true
nvidia-smi
```

Do not kill an unfamiliar process. SQZ is shared with a collaborator.

On ParaCloud, inspect storage and current content:

```bash
set -euo pipefail
df -hT /data/run01/scvj260 /data/home/scvj260
df -i /data/run01/scvj260
squeue -u scvj260
test -d /data/run01/scvj260/scenesmith
test ! -e /data/run01/scvj260/scenesmith-hts.stage
find /data/run01/scvj260/scenesmith -mindepth 1 -maxdepth 1 \
  -printf '%f %y %l\n' | sort
```

Ask the ParaCloud administrator for the `scvj260` storage quota before a large
write. Filesystem free space alone is not quota evidence.

## 4. Phase B - inventory and comparison manifests (`METADATA_WRITE`)

These commands create small manifest files only after migration execution is
explicitly approved.

On SQZ:

```bash
set -euo pipefail
umask 077
SRC=/root/workspace/scenesmith-hts
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
if [[ -e "$META" || -L "$META" ]]; then
  echo "metadata root already exists; stop and replace every 20260711 migration id consistently before retry" >&2
  exit 1
fi
install -d -m 0700 "$META"
test "$(readlink -f "$META")" = "$META"

cd "$SRC"
git rev-parse HEAD >"$META/base_commit.txt"
git status --short >"$META/scenesmith_hts_git_status.txt"

# Review every link and reject any unapproved absolute/out-of-tree target.
find . -path './.git' -prune -o -type l -printf '%P\t%l\n' | sort \
  >"$META/worktree_symlinks.tsv"

# Report suspicious names only; never print file contents.
find . -path './.git' -prune -o \
  \( -iname '.env' -o -iname '.env.*' -o -iname '.secrets' \
     -o -iname '.secrets.*' -o -iname '*.pem' -o -iname 'id_rsa*' \
     -o -iname '*id_ed25519*' \) -printf '%P\n' | sort \
  >"$META/suspicious_path_names.txt"
python - "$META/suspicious_path_names.txt" <<'PY'
from pathlib import Path
import sys
allowed_top={'.env','.mujoco_venv','.pytest_cache','outputs',
 'data.handoff-broken-20260710','external.handoff-broken-20260710'}
bad=[]
for raw in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    if raw and Path(raw).parts[0] not in allowed_top: bad.append(raw)
if bad: raise SystemExit('unreconciled secret-like selected path(s): '+repr(bad))
PY

# Do not create or transfer a Git bundle, `.git`, binary diff, stash, alternate
# ref, config, hook, or credential helper. Deleted secrets can survive in Git
# history even when the current worktree is clean. The migration copies only the
# explicitly audited current worktree payload. `base_commit.txt` is provenance,
# not a transported history database.

readlink -f data/materials >"$META/materials_target.txt"
readlink -f data/artvip_sdf >"$META/artvip_target.txt"
readlink -f data/artiverse >"$META/artiverse_target.txt"
readlink -f data/objathor-assets >"$META/objathor_target.txt"
readlink -f external >"$META/external_target.txt"
test "$(cat "$META/external_target.txt")" = /root/workspace/scenesmith/external

cd /localssd/scenesmith-hts-assets/artiverse/repository
sha256sum \
  dataset_chunks/artiverse_data-00001-of-00002.tar.gz \
  dataset_chunks/artiverse_data-00002-of-00002.tar.gz \
  dataset_chunks/manifest.json \
  pack_dataset_chunks.py \
  >"$META/artiverse_source.sha256"

cd /root/.objathor-assets/2023_09_23
sha256sum assets.tar annotations.json.gz features.tar \
  >"$META/objathor_raw_source.sha256"

cd "$SRC"
source /root/workspace/Humanoid-Training-Scene-Ready-Generation/local_setup/setup_env_sqz.sh
python scripts/pipeline_code_contract.py \
  --repo-dir "$SRC" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output "$META/pipeline_code_contract_migration_snapshot.json"

# Capture the exact working isolated simulator environment without copying its
# path-bound virtualenv. `scenesmith` is reinstalled editable from the new repo.
"$SRC/.mujoco_venv/bin/python" - "$META/mujoco_environment_requirements.txt" \
  "$META/mujoco_environment_python.json" <<'PY'
import importlib.metadata as md, json, sys
from pathlib import Path
rows=[]
for dist in md.distributions():
    name=dist.metadata.get('Name')
    if name and name.lower()!='scenesmith': rows.append(f'{name}=={dist.version}')
Path(sys.argv[1]).write_text('\n'.join(sorted(set(rows),key=str.lower))+'\n',encoding='utf-8')
Path(sys.argv[2]).write_text(json.dumps({'python':sys.version,'executable':sys.executable},sort_keys=True)+'\n',encoding='utf-8')
PY
(cd "$META" && sha256sum mujoco_environment_requirements.txt \
  mujoco_environment_python.json >mujoco_environment.sha256)
```

This intentionally does not overwrite the stopped attempt's canonical
`pipeline_code_contract.json`. Run the relevant unit/contract tests before
accepting the migration snapshot.

Copy the exact helper from Appendix A to
`$META/migration_exact_manifest.py` using Claude Code's structured file-edit
tool, then generate the canonical code and output selections. Do not improvise
with an exclude list that differs between hashing and transfer:

```bash
set -euo pipefail
SRC=/root/workspace/scenesmith-hts
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python

"$PY" "$META/migration_exact_manifest.py" \
  --root "$SRC" --profile code --secret-scan \
  --manifest "$META/code.source.jsonl" --paths0 "$META/code.paths0"

for name in \
  preflight/full_quality_school_reference_20260710 \
  2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710; do
  slug=${name//\//__}
  "$PY" "$META/migration_exact_manifest.py" \
    --root "$SRC/outputs/$name" --profile full --secret-scan \
    --manifest "$META/output_${slug}.source.jsonl" \
    --paths0 "$META/output_${slug}.paths0"
done

sha256sum \
  "$META/migration_exact_manifest.py" \
  "$META/code.source.jsonl" \
  "$META"/output_*.source.jsonl \
  >"$META/source_payload_manifests.sha256"

# Build every remaining SQZ sizing manifest before the capacity gate. Later
# phases may reuse these exact files only if a fresh regeneration is byte-equal.
declare -a specs=(
  'objathor_ready|/root/.objathor-assets/2023_09_23|objathor-ready|reject'
  'external|/root/workspace/scenesmith/external|full|reject'
  'artiverse_openclip_cache|/root/.cache/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378|full|internal'
  'objathor_model_cache|/root/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K|full|internal'
  'sam3d_moge|/root/.cache/huggingface/hub/models--Ruicheng--moge-vitl|full|internal'
  'sam3d_dinov2_source|/root/.cache/torch/hub/facebookresearch_dinov2_main|full|internal'
  'materials_sqz_authority|/root/workspace/scenesmith/data/materials|full|reject'
  'artvip_sqz_authority|/root/workspace/scenesmith/data/artvip_sdf|full|reject'
)
for spec in "${specs[@]}"; do
  IFS='|' read -r name root profile links <<<"$spec"
  audit=()
  [[ "$name" == external ]] && audit=(--secret-scan)
  "$PY" "$META/migration_exact_manifest.py" --root "$root" \
    --profile "$profile" --link-policy "$links" "${audit[@]}" \
    --manifest "$META/${name}.source.jsonl" --paths0 "$META/${name}.paths0"
done

du --apparent-size -s -B1 \
  /localssd/scenesmith-hts-assets/artiverse/repository \
  >"$META/artiverse_final_apparent_bytes.txt"
find /localssd/scenesmith-hts-assets/artiverse/repository -xdev -printf '.' \
  | wc -c >"$META/artiverse_final_inode_entries.txt"
stat -c '%n %s' \
  /root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth \
  >"$META/dinov2_checkpoint_size.txt"

# Freeze every capacity/transfer selection input. Never overwrite these files.
find "$META" -maxdepth 1 -type f \
  \( -name '*.source.jsonl' -o -name '*.paths0' \
     -o -name '*_source.sha256' -o -name '*_size.txt' \
     -o -name '*_apparent_bytes.txt' \
     -o -name '*_inode_entries.txt' \
     -o -name '*_target.txt' \
     -o -name 'mujoco_environment_*.txt' \
     -o -name 'mujoco_environment_*.json' \
     -o -name 'mujoco_environment.sha256' \
     -o -name 'pipeline_code_contract_migration_snapshot.json' \
     -o -name 'base_commit.txt' \) -print0 | sort -z | xargs -0 sha256sum \
  >"$META/capacity_inputs.sha256"
sha256sum "$META/migration_exact_manifest.py" >>"$META/capacity_inputs.sha256"
test -s "$META/capacity_inputs.sha256"
```

The helper reports only the offending path and rule, never secret content. The
source's top-level `.env` is known and deliberately outside the code allowlist;
do not open, copy, or log it. Any secret-audit hit *inside a selected profile* is
a hard stop. Do not weaken a pattern merely to make the payload pass; remove the
file from the allowlist or replace the secret with a documented placeholder,
then rebuild all manifests.

Before Phase E, sum every JSONL `type=file` size with a parser, not shell text
guessing. Add the two Artiverse archive/extracted/prepared target size from
`artiverse_final_apparent_bytes.txt`, the DINO checkpoint, the exact ParaCloud
legacy materials/ArtVIP/HSSD apparent sizes, and a conservative 25 GiB combined
`.venv`/`.mujoco_venv` bound. Record a table with `network_transfer_bytes`, `final_new_allocated_bytes`,
`worst_case_temporary_bytes`, and the 50 GiB reserve. If any later source
manifest differs, invalidate the table and repeat the quota/space gate.

On ParaCloud, record the legacy assets before considering reuse:

```bash
set -euo pipefail
LEGACY=/data/run01/scvj260/scenesmith
LEGACY_META=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/legacy_inventory
test ! -e "$LEGACY_META"
install -d -m 0750 "$LEGACY_META"

du --apparent-size -s -B1 \
  "$LEGACY/data/hssd-models" \
  "$LEGACY/data/artvip_sdf" \
  "$LEGACY/data/materials" \
  "$LEGACY/data/artiverse" \
  "$LEGACY/external/checkpoints" \
  "$LEGACY/.venv" \
  "$LEGACY/outputs" \
  "$LEGACY/transfer_chunks" \
  >"$LEGACY_META/sizes.txt"

find "$LEGACY/data/hssd-models" -type f -printf '%P\t%s\n' | sort \
  >"$LEGACY_META/hssd_files.tsv"
find "$LEGACY/outputs" -mindepth 1 -maxdepth 2 -type d -printf '%P\n' | sort \
  >"$LEGACY_META/output_runs.txt"
find "$LEGACY/transfer_chunks" -type f -printf '%P\t%s\n' | sort \
  >"$LEGACY_META/transfer_chunk_files.tsv"
```

Do not inspect or copy the contents of `$LEGACY/.secrets` in an automated log.

### Appendix A - exact payload/manifest helper

Save the following verbatim as the migration-owned
`migration_exact_manifest.py`. It is intentionally usable with Python 3.8+ and
has no third-party dependency. The same file and profile must create both sides
of every comparison. Its manifest is always outside the scanned root.

```python
#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

CODE_ENTRIES = (
    '.dockerignore', '.github', '.gitignore', '.gitmodules',
    '.pre-commit-config.yaml', '.python-version',
    'CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md',
    'CODEX_SQZ_REMOTE_HANDOFF.md', 'DEVELOPMENT.md', 'Dockerfile', 'LICENSE',
    'README.md', 'REFACTOR_PLAN.md', '_remote_patch', 'configurations', 'data',
    'docker-compose.yaml', 'docs', 'inputs', 'local_setup', 'main.py',
    'materials', 'media', 'pyproject.toml', 'remote_jobs', 'scenesmith',
    'scripts', 'tests', 'tools', 'upstream-patches', 'uv.lock',
)
CODE_SKIPS = {
    'data/artiverse', 'data/artvip_sdf', 'data/materials',
    'data/objathor-assets',
}
CODE_EXCLUDED = {
    '.env', '.git', '.mujoco_venv', '.pytest_cache', 'outputs',
    'data.handoff-broken-20260710', 'external',
    'external.handoff-broken-20260710', 'hf_dl', ' ' + chr(92), '%ln',
}
OBJATHOR_ENTRIES = (
    'annotations.json.gz', 'assets', 'assets.tar', 'features', 'features.tar',
    'preprocessed',
)
SECRET_RULES = (
    ('private-key', re.compile(br'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')),
    ('openai-token', re.compile(br'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b')),
    ('huggingface-token', re.compile(br'\bhf_[A-Za-z0-9]{20,}\b')),
    ('aws-access-key', re.compile(br'\bAKIA[0-9A-Z]{16}\b')),
    ('github-token', re.compile(br'\bgh[pousr]_[A-Za-z0-9]{30,}\b')),
    ('secret-assignment', re.compile(
        br'(?im)^\s*(?:export\s+)?(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|'
        br'HUGGING_FACE_HUB_TOKEN|AWS_SECRET_ACCESS_KEY|VPN_PASSWORD)\s*[:=]\s*'
        br'[\'\"]?(?!\$|\{|<|REDACTED|CHANGEME)([^\s#\'\"]{12,})'
    )),
)

def fail(message):
    raise SystemExit(message)

def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def hash_and_audit(path, secret_scan):
    digest = hashlib.sha256()
    tail = b''
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
            if secret_scan:
                sample = tail + chunk
                for rule, pattern in SECRET_RULES:
                    if pattern.search(sample):
                        fail('secret-content rule %s: %s' % (rule, path))
                tail = sample[-1024:]
    return digest.hexdigest()

def suspicious_name(relative):
    parts = [part.lower() for part in Path(relative).parts]
    base = parts[-1]
    return (
        base == '.env' or base.startswith('.env.') or base == '.secrets' or
        base.startswith('.secrets.') or base.endswith('.pem') or
        base.startswith('id_rsa') or base.startswith('id_ed25519') or
        any(marker in base for marker in ('secret','token','credential'))
    )

def transient_path(relative):
    parts = Path(relative).parts
    return (
        any(part in {'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'} for part in parts)
        or relative.endswith(('.pyc','.pyo'))
        or (parts and parts[-1] in {'.DS_Store','Thumbs.db'})
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--profile', choices=('code', 'code-snapshot', 'code-destination', 'full', 'objathor-ready'), required=True)
    parser.add_argument('--link-policy', choices=('reject', 'internal'), default='reject')
    parser.add_argument('--secret-scan', action='store_true')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--paths0')
    args = parser.parse_args()

    root = Path(args.root).resolve(strict=True)
    if not root.is_dir(): fail('root is not a directory: %s' % root)
    manifest = Path(args.manifest).resolve()
    paths0 = Path(args.paths0).resolve() if args.paths0 else None
    for out in (manifest, paths0):
        if out is not None:
            raw = Path(args.manifest) if out == manifest else Path(args.paths0)
            if raw.exists() or raw.is_symlink():
                fail('refusing to overwrite manifest output: %s' % raw)
            if out == root or root in out.parents:
                fail('output must be outside scanned root: %s' % out)

    if args.profile in {'code','code-snapshot','code-destination'}:
        actual={p.name for p in root.iterdir()}
        extra=(CODE_EXCLUDED if args.profile=='code' else ({'outputs'} if args.profile=='code-destination' else set()))
        classified=set(CODE_ENTRIES)|extra
        unknown=sorted(actual-classified)
        missing_exclusions=(sorted(CODE_EXCLUDED-actual) if args.profile=='code' else [])
        if unknown: fail('unclassified code-root entry: %r' % unknown)
        if args.profile=='code-snapshot' and actual!=set(CODE_ENTRIES):
            fail('code snapshot top-level set mismatch')
        if args.profile=='code-destination' and actual!=set(CODE_ENTRIES)|{'outputs'}:
            fail('code destination top-level set mismatch')
        if missing_exclusions:
            fail('expected excluded code-root entry disappeared; re-audit profile: %r' % missing_exclusions)
        selected = [root / name for name in CODE_ENTRIES]
        missing = [str(p.relative_to(root)) for p in selected if not p.exists()]
        if missing: fail('missing code allowlist entries: %r' % missing)
        skips = CODE_SKIPS
    elif args.profile == 'objathor-ready':
        selected = [root / name for name in OBJATHOR_ENTRIES]
        missing = [str(p.relative_to(root)) for p in selected if not p.exists()]
        if missing: fail('missing ObjectThor ready entries: %r' % missing)
        accepted = set(OBJATHOR_ENTRIES)
        for p in root.iterdir():
            name = p.name
            transient = (
                name == 'objects.lock' or name.endswith('.extracting') or
                '.interrupted-' in name or name.endswith(('.resume', '.part', '.tmp'))
            )
            if name not in accepted and not transient:
                fail('unrecognized ObjectThor top-level entry: %s' % name)
        skips = set()
    else:
        selected = sorted(root.iterdir(), key=lambda p: p.name)
        skips = set()

    records = []
    transfer_paths = []
    seen = set()

    def visit(path):
        relative = path.relative_to(root).as_posix()
        if relative in skips:
            return
        if transient_path(relative):
            return
        if relative in seen:
            fail('duplicate selected path: %s' % relative)
        seen.add(relative)
        if any(ch in relative for ch in ('\x00', '\n', '\r', '\t')):
            fail('unsafe path characters: %r' % relative)
        if args.secret_scan and suspicious_name(relative):
            fail('secret-like filename: %s' % relative)

        info = path.lstat()
        mode = info.st_mode
        common = {'path': relative, 'mode': stat.S_IMODE(mode)}
        if stat.S_ISLNK(mode):
            target = os.readlink(path)
            if args.link_policy != 'internal':
                fail('link rejected: %s -> %s' % (relative, target))
            if os.path.isabs(target):
                fail('absolute link rejected: %s -> %s' % (relative, target))
            try:
                path.resolve(strict=True).relative_to(root)
            except (FileNotFoundError, RuntimeError, ValueError):
                fail('non-internal or broken link: %s -> %s' % (relative, target))
            records.append(dict(common, type='link', target=target))
            transfer_paths.append(relative)
        elif stat.S_ISDIR(mode):
            records.append(dict(common, type='dir'))
            transfer_paths.append(relative)
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                visit(child)
        elif stat.S_ISREG(mode):
            digest = hash_and_audit(path, args.secret_scan)
            records.append(dict(common, type='file', size=info.st_size, sha256=digest))
            transfer_paths.append(relative)
        else:
            fail('special entry rejected: %s' % relative)

    for path in selected:
        visit(path)

    records.sort(key=lambda item: (item['path'], item['type']))
    manifest_data = b''.join(
        (json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
        for r in records
    )
    atomic_bytes(manifest, manifest_data)
    if paths0 is not None:
        atomic_bytes(paths0, b''.join(p.encode('utf-8') + b'\0' for p in sorted(set(transfer_paths))))
    print(json.dumps({
        'manifest': str(manifest), 'records': len(records),
        'sha256': hashlib.sha256(manifest_data).hexdigest(),
    }, sort_keys=True))

if __name__ == '__main__':
    main()
```

Run `python3 -m py_compile migration_exact_manifest.py` and a small temporary
fixture test covering a regular file, an internal link, an escaping link, a
special entry, a secret marker, and each selection profile before trusting it.
The source and destination manifest files must be byte-identical, not merely
equal in line count. This detects missing, changed, extra, mode-changed, linked,
and special entries inside the selected profile without `rsync --delete`.

## 5. Phase C - direct SQZ-to-ParaCloud authentication (`REMOTE_AUTH_WRITE`)

The current laptop relay is unsuitable:

- measured SQZ to laptop: about 8.3 MiB/s
- laptop to ParaCloud did not finish 32 MiB in 120 seconds, so it was below
  approximately 0.27 MiB/s
- at that rate, a 265 GiB relay would exceed 12 days

SQZ can reach `ssh.cn-zhongwei-1.paracloud.com:22` directly, but SQZ currently
has no ParaCloud private key. Never copy the laptop's private key to SQZ.

After explicit authorization, generate a dedicated temporary migration key on
SQZ:

```bash
set -euo pipefail
umask 077
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
if [[ -e "$KEY" || -L "$KEY" || -e "$KEY.pub" || -L "$KEY.pub" ]]; then
  echo "Refusing to overwrite an existing migration key" >&2
  exit 1
fi
ssh-keygen -t ed25519 -N '' -f "$KEY" \
  -C 'sqz-to-paracloud-migration-20260711'
cat "$KEY.pub"
```

From Windows, install only that public key on ParaCloud:

```powershell
$pub = (ssh -o BatchMode=yes sqz `
  "cat /root/.ssh/sqz_to_paracloud_migration_20260711.pub").Trim()
if ($pub -notmatch '^ssh-ed25519 [A-Za-z0-9+/]+={0,2} sqz-to-paracloud-migration-20260711$') {
  throw 'Unexpected migration public key format or comment'
}
$pub | ssh -o BatchMode=yes paracloud `
  'set -eu; umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; IFS= read -r key; entry="restrict $key"; if grep -qxF "$entry" ~/.ssh/authorized_keys; then :; elif grep -qF "sqz-to-paracloud-migration-20260711" ~/.ssh/authorized_keys; then echo "Conflicting migration-key entry" >&2; exit 1; else printf "%s\n" "$entry" >> ~/.ssh/authorized_keys; fi'
```

Do not establish trust by comparing two unauthenticated `ssh-keyscan` results.
Authenticate one exact ED25519 key against the laptop's already trusted
`known_hosts` entry (or an administrator-published key) and copy that exact key,
not a fresh SQZ scan, into the SQZ-only known-host file:

```powershell
$ErrorActionPreference = 'Stop'
$hostName = 'ssh.cn-zhongwei-1.paracloud.com'
$knownFile = Join-Path $HOME '.ssh\known_hosts'

# Show every algorithm currently served, but authenticate and store only one
# exact ED25519 key.
$scanAll = @(ssh-keyscan -T 10 -p 22 $hostName 2>$null |
  Where-Object { $_ -and -not $_.StartsWith('#') })
if ($scanAll.Count -eq 0) { throw 'ssh-keyscan returned no host keys' }
$scanAll | ssh-keygen -lf -

$trustedLines = @(ssh-keygen -F $hostName -f $knownFile |
  Where-Object { $_ -and -not $_.StartsWith('#') })
$trustedEd = @($trustedLines | Where-Object {
  $f = $_ -split '\s+'; $f.Count -ge 3 -and $f[1] -eq 'ssh-ed25519'
})
$scannedEd = @($scanAll | Where-Object {
  $f = $_ -split '\s+'; $f.Count -ge 3 -and $f[1] -eq 'ssh-ed25519'
})
if ($trustedEd.Count -ne 1) {
  throw 'Need exactly one already-trusted ED25519 key; ask the administrator'
}
if ($scannedEd.Count -ne 1) { throw 'Need exactly one scanned ED25519 key' }

$trustedKey = ($trustedEd[0] -split '\s+')[2]
$scannedKey = ($scannedEd[0] -split '\s+')[2]
if ($trustedKey -cne $scannedKey) { throw 'ParaCloud ED25519 host key mismatch' }
$trustedEd[0] | ssh-keygen -lf -

$knownLine = "$hostName ssh-ed25519 $trustedKey"
$knownLine | ssh -o BatchMode=yes sqz `
  'set -eu; umask 077; f=/root/.ssh/known_hosts.paracloud_migration_20260711; test ! -e "$f"; IFS= read -r line; case "$line" in "ssh.cn-zhongwei-1.paracloud.com ssh-ed25519 "*) ;; *) echo "bad host-key line" >&2; exit 1;; esac; printf "%s\n" "$line" >"$f"; chmod 600 "$f"; ssh-keygen -lf "$f"'
```

If the host is absent from trusted laptop state, stop and obtain the fingerprint
and exact public key from the ParaCloud administrator. Fingerprint text alone is
acceptable only when it comes from that authenticated channel and is checked
against `ssh-keygen -lf -`; never trust an SQZ-side scan by itself. Then test direct authentication
from SQZ using that exact known-host file:

```bash
ssh -i /root/.ssh/sqz_to_paracloud_migration_20260711 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' \
  ssh.cn-zhongwei-1.paracloud.com \
  'hostname; id; pwd'
```

Do not continue if the reported account is not `scvj260` or the host is not a
ParaCloud login node.

## 6. Phase D - direct throughput benchmark and time calculation (`READ_ONLY`)

Run a 1 GiB read-only stream from an existing SQZ archive to ParaCloud
`/dev/null`. It creates no destination file:

```bash
set -euo pipefail
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
samples=()
for run in 1 2 3; do
  start_ns=$(date +%s%N)
  dd \
    if=/localssd/scenesmith-hts-assets/artiverse/repository/dataset_chunks/artiverse_data-00001-of-00002.tar.gz \
    bs=16M count=64 status=progress |
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" 'cat >/dev/null'
  end_ns=$(date +%s%N)
  elapsed_ns=$((end_ns - start_ns))
  sample=$(awk -v ns="$elapsed_ns" 'BEGIN { printf "%.3f", 1024.0 / (ns / 1000000000.0) }')
  samples+=("$sample")
  printf 'run=%s bytes=1073741824 elapsed_ns=%s MiB_per_second=%s\n' \
    "$run" "$elapsed_ns" "$sample"
done
median=$(printf '%s\n' "${samples[@]}" | sort -n | sed -n '2p')
printf 'median_MiB_per_second=%s\n' "$median"
```

Use the reported median. Do not estimate the final duration before this direct
benchmark.

Calculation:

```text
hours = payload_GiB * 1024 / measured_MiB_per_second / 3600
```

Reference calculations only:

| Speed | 140 GiB direct SQZ payload | 290 GiB final-new-data reference |
|---:|---:|---:|
| 5 MiB/s | 8.0 h | 16.5 h |
| 10 MiB/s | 4.0 h | 8.2 h |
| 20 MiB/s | 2.0 h | 4.1 h |
| 50 MiB/s | 0.8 h | 1.6 h |

Many small files will be slower. Prefer compressed publisher archives and
server-side extraction rather than sending 531,937 Artiverse files individually.

### Executable capacity ledger and phase guard

Capacity is a hard gate, not prose. Filesystem space, byte quota, filesystem
inodes, and inode quota are four separate limits. Before Phase E, obtain from the
ParaCloud administrator one absolute, root-owned, non-group/world-writable
executable path in `PARACLOUD_QUOTA_QUERY`. It must be available with identical
identity on login and compute nodes and, on every invocation, print one JSON
object with exact fields `scope`, `remaining_bytes`, `remaining_inodes`, and
`observed_epoch`. `scope` must be `/data/run01/scvj260`; the two remaining values
must be nonnegative integers; `observed_epoch` must be no more than five minutes
old. If the cluster cannot provide this live machine-readable query (including
an inode quota or an administrator-confirmed no-extra-inode-limit value), stop.
A manually pasted number is not safe for a queued job.

After Phase C authentication and before Phase E, run on SQZ:

```bash
set -euo pipefail
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
: "${PARACLOUD_QUOTA_QUERY:?obtain the administrator-provided live quota executable}"
[[ "$PARACLOUD_QUOTA_QUERY" = /* ]]
[[ "$PARACLOUD_QUOTA_QUERY" =~ ^/[A-Za-z0-9_./-]+$ ]]
sha256sum -c "$META/capacity_inputs.sha256"
for path in "$META/paracloud_capacity_snapshot.tsv" \
  "$META/paracloud_quota_snapshot.json" \
  "$META/quota_query_identity.tsv" \
  "$META/capacity_ledger.json" "$META/capacity_ledger.json.tmp" \
  "$META/capacity_ledger.sha256"; do
  [[ ! -e "$path" && ! -L "$path" ]]
done

ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' "$HOST" bash -s -- "$PARACLOUD_QUOTA_QUERY" \
  >"$META/quota_query_identity.tsv" <<'REMOTE'
set -euo pipefail
path=$(readlink -f -- "$1")
test "$path" = "$1"
test -f "$path" && test -x "$path"
read -r owner mode < <(stat -c '%U %a' "$path")
test "$owner" = root
test $((8#$mode & 022)) -eq 0
printf 'path\t%s\nowner\t%s\nmode\t%s\nsha256\t%s\n' \
  "$path" "$owner" "$mode" "$(sha256sum "$path" | cut -d' ' -f1)"
REMOTE

ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' "$HOST" bash -s -- "$PARACLOUD_QUOTA_QUERY" \
  >"$META/paracloud_quota_snapshot.json" <<'REMOTE'
set -euo pipefail
exec "$1"
REMOTE

ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' "$HOST" '
    set -euo pipefail
    entries() { find "$1" -xdev -printf "." | wc -c; }
    printf "hostname\t%s\n" "$(hostname)"
    printf "captured_utc\t%s\n" "$(date -u +%FT%TZ)"
    printf "free_bytes\t%s\n" "$(df -B1 --output=avail /data/run01/scvj260 | tail -1 | tr -d " ")"
    printf "free_inodes\t%s\n" "$(df -Pi --output=iavail /data/run01/scvj260 | tail -1 | tr -d " ")"
    printf "materials_bytes\t%s\n" "$(du --apparent-size -s -B1 /data/run01/scvj260/scenesmith/data/materials | cut -f1)"
    printf "materials_inodes\t%s\n" "$(entries /data/run01/scvj260/scenesmith/data/materials)"
    printf "artvip_bytes\t%s\n" "$(du --apparent-size -s -B1 /data/run01/scvj260/scenesmith/data/artvip_sdf | cut -f1)"
    printf "artvip_inodes\t%s\n" "$(entries /data/run01/scvj260/scenesmith/data/artvip_sdf)"
    printf "hssd_bytes\t%s\n" "$(du --apparent-size -s -B1 /data/run01/scvj260/scenesmith/data/hssd-models | cut -f1)"
    printf "hssd_inodes\t%s\n" "$(entries /data/run01/scvj260/scenesmith/data/hssd-models)"
  ' >"$META/paracloud_capacity_snapshot.tsv"

/root/workspace/miniconda3/envs/scenesmith/bin/python - "$META" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
meta=Path(sys.argv[1])
GiB=1024**3
def manifest_stats(name):
    total=entries=0
    p=meta/name
    for line in p.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row.get('type')=='file': total+=int(row['size'])
        if row.get('type') in {'file','dir','symlink'}: entries+=1
    return {'bytes':total,'inodes':entries}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
remote={}
for line in (meta/'paracloud_capacity_snapshot.tsv').read_text(encoding='utf-8').splitlines():
    key,value=line.split('\t',1); remote[key]=value
for key in ('free_bytes','free_inodes','materials_bytes','materials_inodes',
            'artvip_bytes','artvip_inodes','hssd_bytes','hssd_inodes'):
    remote[key]=int(remote[key])

quota=json.loads((meta/'paracloud_quota_snapshot.json').read_text(encoding='utf-8'))
if set(quota)!={'scope','remaining_bytes','remaining_inodes','observed_epoch'}:
    raise SystemExit('quota query returned the wrong schema')
if quota['scope']!='/data/run01/scvj260': raise SystemExit('wrong quota scope')
for key in ('remaining_bytes','remaining_inodes','observed_epoch'):
    if type(quota[key]) is not int or quota[key] < 0: raise SystemExit('bad quota value: '+key)
age=int(time.time())-quota['observed_epoch']
if age < -30 or age > 300: raise SystemExit('quota observation is stale or future-dated')
identity=dict(line.split('\t',1) for line in (meta/'quota_query_identity.tsv').read_text().splitlines())

code=manifest_stats('code.source.jsonl')
out_preflight=manifest_stats('output_preflight__full_quality_school_reference_20260710.source.jsonl')
out_scene=manifest_stats('output_2026-07-10__full_quality_school_reference_sam3d_artvip_artiverse_20260710.source.jsonl')
obj=manifest_stats('objathor_ready.source.jsonl')
external=manifest_stats('external.source.jsonl')
art_cache=manifest_stats('artiverse_openclip_cache.source.jsonl')
obj_cache=manifest_stats('objathor_model_cache.source.jsonl')
moge=manifest_stats('sam3d_moge.source.jsonl')
dino_source=manifest_stats('sam3d_dinov2_source.source.jsonl')
dino_checkpoint=Path('/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth').stat().st_size
art_final=int((meta/'artiverse_final_apparent_bytes.txt').read_text().split()[0])
art_final_inodes=int((meta/'artiverse_final_inode_entries.txt').read_text().strip())
art_inputs=sum(Path(p).stat().st_size for p in (
 '/localssd/scenesmith-hts-assets/artiverse/repository/dataset_chunks/artiverse_data-00001-of-00002.tar.gz',
 '/localssd/scenesmith-hts-assets/artiverse/repository/dataset_chunks/artiverse_data-00002-of-00002.tar.gz',
 '/localssd/scenesmith-hts-assets/artiverse/repository/dataset_chunks/manifest.json',
 '/localssd/scenesmith-hts-assets/artiverse/repository/pack_dataset_chunks.py'))
metadata_bound=GiB
metadata_inodes_bound=10_000
main_env={'bytes':20*GiB,'inodes':200_000}
mujoco_env={'bytes':5*GiB,'inodes':100_000}
validation={'bytes':10*GiB,'inodes':100_000}
art_prepare_extra={'bytes':5*GiB,'inodes':10_000}
reserve={'bytes':50*GiB,'inodes':250_000}
art_input_inodes=6  # repository dir, dataset_chunks dir, and four selected files

def add(*rows): return {k:sum(r[k] for r in rows) for k in ('bytes','inodes')}
metadata={'bytes':metadata_bound,'inodes':metadata_inodes_bound}
art={'bytes':art_final,'inodes':art_final_inodes}
dino_ckpt={'bytes':dino_checkpoint,'inodes':1}
materials={'bytes':remote['materials_bytes'],'inodes':remote['materials_inodes']}
artvip={'bytes':remote['artvip_bytes'],'inodes':remote['artvip_inodes']}
hssd={'bytes':remote['hssd_bytes'],'inodes':remote['hssd_inodes']}
art_inputs_row={'bytes':art_inputs,'inodes':art_input_inodes}
increments={
 'F':add(code,out_preflight,out_scene,metadata),
 'J':add(main_env,mujoco_env),
 'G_FINAL':add(art,art_cache),
 'H':add(obj,obj_cache),
 'I':add(external,moge,dino_source,dino_ckpt,materials,artvip,hssd),
 'K':validation,
}
final_new={k:sum(v[k] for v in increments.values()) for k in ('bytes','inodes')}
write_bounds={
 'F_METADATA':metadata, 'F_CODE':code,
 'F_OUTPUT_PREFLIGHT':out_preflight, 'F_OUTPUT_SCENE':out_scene,
 'J_MAIN_ENV':main_env, 'J_MUJOCO_ENV':mujoco_env,
 'G_INPUTS':art_inputs_row, 'G_SOURCE_METADATA':metadata,
 'G_OPENCLIP_METADATA':metadata, 'G_OPENCLIP_PAYLOAD':art_cache,
 'G_MATERIALIZE':{
   'bytes':max(0,art_final-art_inputs)+art_prepare_extra['bytes'],
   'inodes':max(0,art_final_inodes-art_input_inodes)+art_prepare_extra['inodes']},
 'H_OBJ_METADATA':metadata, 'H_OBJ_PAYLOAD':obj,
 'H_MODEL_METADATA':metadata, 'H_MODEL_PAYLOAD':obj_cache,
 'I_AUTH_METADATA':metadata, 'I_MATERIALS':materials,
 'I_ARTVIP':artvip, 'I_HSSD':hssd,
 'I_EXTERNAL_METADATA':metadata, 'I_EXTERNAL_PAYLOAD':external,
 'I_SAM_METADATA':metadata, 'I_MOGE':moge,
 'I_DINO_SOURCE':dino_source, 'I_DINO_CHECKPOINT':dino_ckpt,
 'K_VALIDATION':validation,
}
network=add(code,out_preflight,out_scene,metadata,art_inputs_row,art_cache,obj,obj_cache,
            external,moge,dino_source,dino_ckpt)
available={'bytes':min(remote['free_bytes'],quota['remaining_bytes']),
           'inodes':min(remote['free_inodes'],quota['remaining_inodes'])}
overall_required=add(final_new,art_prepare_extra,reserve)
status='pass' if all(available[k]>=overall_required[k] for k in ('bytes','inodes')) else 'fail'
result={
 'schema_version':2,'status':status,
 'capacity_inputs_sha256':sha(meta/'capacity_inputs.sha256'),
 'remote_snapshot_sha256':sha(meta/'paracloud_capacity_snapshot.tsv'),
 'quota_snapshot_sha256':sha(meta/'paracloud_quota_snapshot.json'),
 'quota_query':identity,
 'filesystem_available':{'bytes':remote['free_bytes'],'inodes':remote['free_inodes']},
 'quota_remaining':{'bytes':quota['remaining_bytes'],'inodes':quota['remaining_inodes'],
                    'observed_epoch':quota['observed_epoch']},
 'effective_available':available, 'overall_required':overall_required,
 'network_transfer_bound':network,'final_new_allocated_bound':final_new,
 'legacy_sources':{k:{'bytes':remote[k+'_bytes'],'inodes':remote[k+'_inodes']}
                   for k in ('materials','artvip','hssd')},
 'artiverse_prepare_extra_bound':art_prepare_extra,
 'safety_reserve':reserve,'increments':increments,'write_bounds':write_bounds,
}
tmp=meta/'capacity_ledger.json.tmp'; out=meta/'capacity_ledger.json'
tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(tmp,out)
print(json.dumps(result,sort_keys=True))
if result['status']!='pass': raise SystemExit('capacity/quota gate failed')
PY
(cd "$META" && sha256sum capacity_ledger.json >capacity_ledger.sha256)
```

The initial ledger proves the entire bounded migration can fit at one instant.
The Artiverse 5 GiB extra is only for extraction/preparation transactions. The
50 GiB and 250,000-inode reserves cover metadata, filesystem accounting, and
estimation error. Never lower a bound to force a pass.

Later checks are deliberately **per next write**, not whole-phase totals. A
completed copy has already reduced current free/quota values, so reusing a phase
total would double-count it. Immediately before every non-dry-run `rsync`, `uv
sync`/`uv pip install` wrapper, extraction/preparation job, and validation job,
invoke the guard with that command's unique `WRITE_KEY`. The guard itself must
invoke the trusted live quota query; never pass or cache a quota integer:

```bash
set -euo pipefail
WRITE_KEY=${WRITE_KEY:?set one write_bounds key}
LEDGER=${LEDGER:?set absolute capacity_ledger.json path}
python3 - "$LEDGER" "$WRITE_KEY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['schema_version']==2 and d['status']=='pass'
print(d['write_bounds'][sys.argv[2]])
PY
```

The executable guard is installed in Phase E and performs byte, inode, trusted
query identity, scope, and five-minute timestamp checks. Phase F invokes it over
the authenticated direct SSH connection; all later phases invoke the same
ParaCloud copy. Any changed source manifest or quota-query identity invalidates
the ledger and forces a complete Phase D recalculation. One result authorizes
only the immediately following command; invoke it again for the next write.

## 7. Phase E - create ParaCloud staging roots (`REMOTE_STAGE_WRITE`)

Only after capacity, quota, key, and speed gates pass:

```bash
ssh -i /root/.ssh/sqz_to_paracloud_migration_20260711 \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' ssh.cn-zhongwei-1.paracloud.com '
    set -euo pipefail
    paths=(
      /data/run01/scvj260/scenesmith-hts.stage
      /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
      /data/run01/scvj260/cache_sqz_20260711
      /data/run01/scvj260/assets/artiverse_sqz_20260711
      /data/run01/scvj260/assets/objathor_2023_09_23
      /data/run01/scvj260/assets/external_sqz_20260711
      /data/run01/scvj260/assets/materials_sqz_20260711
      /data/run01/scvj260/assets/artvip_sqz_20260711
      /data/run01/scvj260/assets/hssd_legacy
      /data/run01/scvj260/migration_logs_20260711
    )
    for path in "${paths[@]}"; do
      if [[ -e "$path" || -L "$path" ]]; then
        printf "refusing non-fresh staging path: %s\n" "$path" >&2
        exit 1
      fi
    done
    install -d -m 0750 \
      /data/run01/scvj260/scenesmith-hts.stage \
      /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711 \
      /data/run01/scvj260/cache_sqz_20260711 \
      /data/run01/scvj260/cache_sqz_20260711/huggingface/hub \
      /data/run01/scvj260/cache_sqz_20260711/torch/hub/checkpoints \
      /data/run01/scvj260/assets/artiverse_sqz_20260711/repository/dataset_chunks \
      /data/run01/scvj260/assets/objathor_2023_09_23 \
      /data/run01/scvj260/assets/external_sqz_20260711 \
      /data/run01/scvj260/assets/materials_sqz_20260711 \
      /data/run01/scvj260/assets/artvip_sqz_20260711 \
      /data/run01/scvj260/assets/hssd_legacy \
      /data/run01/scvj260/migration_logs_20260711
    df -h /data/run01/scvj260
    df -i /data/run01/scvj260
  '
```

This phase is deliberately one-shot. If a later command is interrupted, do not
rerun Phase E and do not merge into newly non-empty roots. Inventory the partial
state, compare it with the source manifests, and ask the user whether to resume
the exact payload or retire the staging attempt.

Copy and verify only the two small frozen ledger files before creating the
guard (run on SQZ):

```bash
set -euo pipefail
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/capacity_ledger.json" "$META/capacity_ledger.sha256" \
  "$HOST:/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/"
ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' "$HOST" \
  'cd /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711 && sha256sum -c capacity_ledger.sha256'
```

Create `/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh`
as a small migration-owned file (not inside the worktree) with exactly:

```bash
#!/usr/bin/env bash
set -euo pipefail
(cd /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711 && sha256sum -c paracloud_migration_env.sha256)
export REPO_ROOT=/data/run01/scvj260/scenesmith-hts.stage
export SCENESMITH_CACHE_ROOT=/data/run01/scvj260/cache_sqz_20260711
export XDG_CACHE_HOME=/data/run01/scvj260/cache_sqz_20260711
export HF_HOME=/data/run01/scvj260/cache_sqz_20260711/huggingface
export TORCH_HOME=/data/run01/scvj260/cache_sqz_20260711/torch
export UV_CACHE_DIR=/data/run01/scvj260/uv-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export OMNI_KIT_ACCEPT_EULA=yes
export NO_PROXY='*'
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset OPENAI_API_KEY ANTHROPIC_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN VPN_PASSWORD
unset DASHSCOPE_API_KEY GOOGLE_API_KEY AZURE_OPENAI_API_KEY
```

Use Claude Code's structured file-edit tool, then run:

```bash
ENVFILE=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
chmod 0750 "$ENVFILE"
bash -n "$ENVFILE"
sha256sum "$ENVFILE" | tee /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sha256
```

Every later ParaCloud block that invokes the project environments, model/cache
loader, contract runtime, or SLURM job must source this file because exports do
not persist across SSH sessions. Pure `sha256sum`/`rsync`/system-`python3`
manifest blocks may omit it because they do not consult project caches.

Also save this migration-owned guard as
`/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
write_key=${1:?write key required}
root=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
(cd "$root" && sha256sum -c capacity_guard.sha256)
(cd "$root" && sha256sum -c capacity_ledger.sha256)
mapfile -t identity < <(python3 - "$root/capacity_ledger.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['schema_version']==2 and d['status']=='pass'
q=d['quota_query']
for key in ('path','owner','mode','sha256'): print(q[key])
PY
)
query=${identity[0]}
test -f "$query" && test -x "$query"
read -r owner mode < <(stat -c '%U %a' "$query")
test "$owner" = "${identity[1]}" && test "$mode" = "${identity[2]}"
test $((8#$mode & 022)) -eq 0
test "$(sha256sum "$query" | cut -d' ' -f1)" = "${identity[3]}"
quota_json=$("$query")
mapfile -t values < <(python3 - "$root/capacity_ledger.json" "$write_key" "$quota_json" <<'PY'
import json,sys,time
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['schema_version']==2 and d['status']=='pass'
b=d['write_bounds'][sys.argv[2]]; reserve=d['safety_reserve']
q=json.loads(sys.argv[3])
assert set(q)=={'scope','remaining_bytes','remaining_inodes','observed_epoch'}
assert q['scope']=='/data/run01/scvj260'
for key in ('remaining_bytes','remaining_inodes','observed_epoch'):
    assert type(q[key]) is int and q[key]>=0
age=int(time.time())-q['observed_epoch']
assert -30 <= age <= 300
print(b['bytes']+reserve['bytes'])
print(b['inodes']+reserve['inodes'])
print(q['remaining_bytes']); print(q['remaining_inodes']); print(q['observed_epoch'])
PY
)
required_bytes=${values[0]}; required_inodes=${values[1]}
quota_bytes=${values[2]}; quota_inodes=${values[3]}; observed_epoch=${values[4]}
free_bytes=$(df -B1 --output=avail /data/run01/scvj260 | tail -1 | tr -d ' ')
free_inodes=$(df -Pi --output=iavail /data/run01/scvj260 | tail -1 | tr -d ' ')
test "$free_bytes" -ge "$required_bytes"
test "$quota_bytes" -ge "$required_bytes"
test "$free_inodes" -ge "$required_inodes"
test "$quota_inodes" -ge "$required_inodes"
printf 'capacity_guard write=%s free_bytes=%s quota_bytes=%s required_bytes=%s free_inodes=%s quota_inodes=%s required_inodes=%s quota_observed_epoch=%s utc=%s\n' \
  "$write_key" "$free_bytes" "$quota_bytes" "$required_bytes" \
  "$free_inodes" "$quota_inodes" "$required_inodes" "$observed_epoch" "$(date -u +%FT%TZ)"
```

Then:

```bash
GUARD=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh
chmod 0750 "$GUARD"
bash -n "$GUARD"
sha256sum "$GUARD" | tee /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sha256
```

The invocation immediately before each later write is:

```bash
: "${WRITE_KEY:?set one unique write_bounds key}"
/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh "$WRITE_KEY"
```

## 8. Phase F - transfer exact code and outputs (`LARGE_ASSET_WRITE`)

Run the remote capacity guard with the unique keys shown below immediately
before each non-dry-run rsync. The guard queries both `df` dimensions and the
trusted live quota service on ParaCloud on every invocation.

On SQZ:

```bash
set -euo pipefail
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
TARGET=/data/run01/scvj260/scenesmith-hts.stage
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before Phase F}"
[[ "$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" =~ ^[0-9]+$ ]]
guard_f() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh F"
}

SRC=/root/workspace/scenesmith-hts

# Re-audit immediately before sending a byte; never overwrite the frozen files.
sha256sum -c "$META/capacity_inputs.sha256"
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
"$PY" "$META/migration_exact_manifest.py" --root "$SRC" --profile code \
  --secret-scan --manifest "$META/code.fresh.jsonl" \
  --paths0 "$META/code.fresh.paths0"
cmp "$META/code.source.jsonl" "$META/code.fresh.jsonl"
cmp "$META/code.paths0" "$META/code.fresh.paths0"
for name in \
  preflight/full_quality_school_reference_20260710 \
  2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710; do
  slug=${name//\//__}
  "$PY" "$META/migration_exact_manifest.py" --root "$SRC/outputs/$name" \
    --profile full --secret-scan \
    --manifest "$META/output_${slug}.fresh.jsonl" \
    --paths0 "$META/output_${slug}.fresh.paths0"
  cmp "$META/output_${slug}.source.jsonl" "$META/output_${slug}.fresh.jsonl"
  cmp "$META/output_${slug}.paths0" "$META/output_${slug}.fresh.paths0"
done

# Freeze a local, secret-scanned SQZ snapshot so a collaborator mutation cannot
# race the external transfer. Never transfer directly from the live worktree.
SNAP=/localssd/scenesmith_migration_payload_20260711
if [[ -e "$SNAP" || -L "$SNAP" ]]; then
  echo 'SQZ payload snapshot already exists; stop and audit the attempt' >&2
  exit 1
fi
install -d -m 0700 "$SNAP/repo" "$SNAP/outputs"
rsync -aH --from0 --files-from="$META/code.paths0" "$SRC/" "$SNAP/repo/"
"$PY" "$META/migration_exact_manifest.py" --root "$SNAP/repo" \
  --profile code-snapshot --secret-scan \
  --manifest "$META/code.snapshot.jsonl" --paths0 "$META/code.snapshot.paths0"
cmp "$META/code.source.jsonl" "$META/code.snapshot.jsonl"
cmp "$META/code.paths0" "$META/code.snapshot.paths0"
for name in \
  preflight/full_quality_school_reference_20260710 \
  2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710; do
  slug=${name//\//__}
  install -d -m 0700 "$SNAP/outputs/$name"
  rsync -aH --from0 --files-from="$META/output_${slug}.paths0" \
    "$SRC/outputs/$name/" "$SNAP/outputs/$name/"
  "$PY" "$META/migration_exact_manifest.py" --root "$SNAP/outputs/$name" \
    --profile full --secret-scan \
    --manifest "$META/output_${slug}.snapshot.jsonl" \
    --paths0 "$META/output_${slug}.snapshot.paths0"
  cmp "$META/output_${slug}.source.jsonl" "$META/output_${slug}.snapshot.jsonl"
  cmp "$META/output_${slug}.paths0" "$META/output_${slug}.snapshot.paths0"
done
SEND_SRC="$SNAP/repo"

# Copy the helper and source manifests outside the destination worktree.
guard_f
rsync -a --no-owner --no-group --partial --info=progress2 -e "$RSH" \
  "$META/migration_exact_manifest.py" \
  "$META/code.source.jsonl" "$META/code.paths0" \
  "$META"/output_*.source.jsonl "$META"/output_*.paths0 \
  "$META/source_payload_manifests.sha256" \
  "$META/base_commit.txt" "$META/scenesmith_hts_git_status.txt" \
  "$META/pipeline_code_contract_migration_snapshot.json" \
  "$META/mujoco_environment_requirements.txt" \
  "$META/mujoco_environment_python.json" \
  "$META/mujoco_environment.sha256" \
  "$META"/*_target.txt \
  "$META/capacity_inputs.sha256" \
  "$HOST:$MIG/"

# The NUL list and source manifest came from the same audited profile. Do not
# add -R and do not substitute a new exclude list.
rsync -anH --no-owner --no-group --from0 --files-from="$META/code.paths0" \
  --itemize-changes -e "$RSH" "$SEND_SRC/" "$HOST:$TARGET/"
guard_f
rsync -aH --no-owner --no-group --from0 --files-from="$META/code.paths0" \
  --partial --info=progress2 -e "$RSH" "$SEND_SRC/" "$HOST:$TARGET/"

# Transfer only the two goal-owned output roots.
ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
  -l 'scvj260@NC-N50R5' "$HOST" \
  'set -euo pipefail; install -d -m 0750 \
    /data/run01/scvj260/scenesmith-hts.stage/outputs/preflight/full_quality_school_reference_20260710 \
    /data/run01/scvj260/scenesmith-hts.stage/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710'
for name in \
  preflight/full_quality_school_reference_20260710 \
  2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710; do
  slug=${name//\//__}
  rsync -anH --no-owner --no-group --from0 \
    --files-from="$META/output_${slug}.paths0" --itemize-changes -e "$RSH" \
    "$SNAP/outputs/$name/" "$HOST:$TARGET/outputs/$name/"
  guard_f
  rsync -aH --no-owner --no-group --from0 \
    --files-from="$META/output_${slug}.paths0" --partial --info=progress2 -e "$RSH" \
    "$SNAP/outputs/$name/" "$HOST:$TARGET/outputs/$name/"
done
```

On ParaCloud, independently rebuild the same profiles and compare bytes:

```bash
set -euo pipefail
TARGET=/data/run01/scvj260/scenesmith-hts.stage
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
PYTHON=python3

test ! -e "$TARGET/.git"
test ! -e "$TARGET/.env"
test ! -e "$TARGET/.venv"
test ! -e "$TARGET/external"
"$PYTHON" "$MIG/migration_exact_manifest.py" \
  --root "$TARGET" --profile code-destination --secret-scan \
  --manifest "$MIG/code.destination.jsonl"
cmp "$MIG/code.source.jsonl" "$MIG/code.destination.jsonl"

for name in \
  preflight/full_quality_school_reference_20260710 \
  2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710; do
  slug=${name//\//__}
  "$PYTHON" "$MIG/migration_exact_manifest.py" \
    --root "$TARGET/outputs/$name" --profile full --secret-scan \
    --manifest "$MIG/output_${slug}.destination.jsonl"
  cmp "$MIG/output_${slug}.source.jsonl" \
      "$MIG/output_${slug}.destination.jsonl"
done
sha256sum "$MIG"/*.destination.jsonl
```

This is the exact-set proof for the code overlay and goal outputs. It detects
destination-only extras inside every approved root; `rsync -n` alone does not.
No Git history is transported. If Git history is later required, reconstruct
the public base separately and reapply the already-manifested current files;
that is a new task requiring its own secret-history audit.

If `rsync` is missing on either host, stop. Do not install it or substitute an
unsafe ad hoc copy without approval.

**Dependency stop:** after Phase F, execute Phase J and create/validate the new
offline `.venv`. Do not start Phases G, H, or I until Phase J passes.

## 9. Phase G - migrate Artiverse archive-first (`LARGE_ASSET_WRITE` / `VALIDATION_WRITE` / `GPU_JOB`)

Run the ParaCloud capacity guard as `G_TRANSFER` before every archive/cache
rsync, and as `G_MATERIALIZE` immediately before the Artiverse `sbatch`.

**Hard dependency:** jump to Section 12 and complete both offline environments before
executing any command in this section. Return here only after its import smoke
passes. This out-of-number ordering is intentional because asset subjects stay
grouped together, but it is not permission to transfer assets early.

Transfer only the two archives and pinned supply-chain files:

```bash
set -euo pipefail
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
SRC=/localssd/scenesmith-hts-assets/artiverse/repository
DST=/data/run01/scvj260/assets/artiverse_sqz_20260711/repository
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
test ! -e "$META/artiverse_final_apparent_bytes.fresh.txt"
du --apparent-size -s -B1 /localssd/scenesmith-hts-assets/artiverse/repository \
  >"$META/artiverse_final_apparent_bytes.fresh.txt"
cmp "$META/artiverse_final_apparent_bytes.txt" \
    "$META/artiverse_final_apparent_bytes.fresh.txt"
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before Artiverse transfer}"
guard_g() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh G_TRANSFER"
}

guard_g
rsync -aH --no-owner --no-group --partial --append-verify --info=progress2 -e "$RSH" \
  "$SRC/dataset_chunks/manifest.json" \
  "$SRC/dataset_chunks/artiverse_data-00001-of-00002.tar.gz" \
  "$SRC/dataset_chunks/artiverse_data-00002-of-00002.tar.gz" \
  "$HOST:$DST/dataset_chunks/"
guard_g
rsync -aH --no-owner --no-group --partial --append-verify --info=progress2 -e "$RSH" \
  "$SRC/pack_dataset_chunks.py" \
  "$HOST:$DST/"
guard_g
rsync -a --no-owner --no-group -e "$RSH" \
  /root/workspace/scenesmith-hts-migration-metadata-20260711/artiverse_source.sha256 \
  "$HOST:/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/"
```

On ParaCloud, verify exact sizes and hashes before extraction:

```bash
set -euo pipefail
ART=/data/run01/scvj260/assets/artiverse_sqz_20260711/repository
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
cd "$ART"
sha256sum -c "$MIG/artiverse_source.sha256"
test "$(stat -c %s "$ART/dataset_chunks/artiverse_data-00001-of-00002.tar.gz")" = 38163580631
test "$(stat -c %s "$ART/dataset_chunks/artiverse_data-00002-of-00002.tar.gz")" = 27170560473
echo '695d2d602faafab922ce66359ea104d81505f5b0fdee8f461d8905f0ccb4ef3b  '"$ART"'/dataset_chunks/artiverse_data-00001-of-00002.tar.gz' | sha256sum -c -
echo '56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81  '"$ART"'/dataset_chunks/artiverse_data-00002-of-00002.tar.gz' | sha256sum -c -
df -h "$ART"
df -i "$ART"
```

Preparation uses OpenCLIP `ViT-H-14-378-quickgelu`/`dfn5b`. Before extraction,
manifest and transfer its exact bounded cache root:

```bash
# Run on SQZ.
set -euo pipefail
MODEL=/root/.cache/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
DST=/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before Artiverse model cache}"
guard_g() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh G_TRANSFER"
}
"$PY" "$META/migration_exact_manifest.py" --root "$MODEL" --profile full \
  --link-policy internal --manifest "$META/artiverse_openclip_cache.fresh.jsonl" \
  --paths0 "$META/artiverse_openclip_cache.fresh.paths0"
cmp "$META/artiverse_openclip_cache.source.jsonl" \
    "$META/artiverse_openclip_cache.fresh.jsonl"
cmp "$META/artiverse_openclip_cache.paths0" \
    "$META/artiverse_openclip_cache.fresh.paths0"
guard_g
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/artiverse_openclip_cache.source.jsonl" \
  "$META/artiverse_openclip_cache.paths0" \
  "$HOST:/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/"
rsync -anH --no-owner --no-group --safe-links --from0 \
  --files-from="$META/artiverse_openclip_cache.paths0" --itemize-changes -e "$RSH" \
  "$MODEL/" "$HOST:$DST/"
guard_g
rsync -aH --no-owner --no-group --safe-links --from0 \
  --files-from="$META/artiverse_openclip_cache.paths0" --partial --info=progress2 -e "$RSH" \
  "$MODEL/" "$HOST:$DST/"
```

```bash
# Run on ParaCloud.
set -euo pipefail
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
MODEL=/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378
python3 "$MIG/migration_exact_manifest.py" --root "$MODEL" --profile full \
  --link-policy internal --manifest "$MIG/artiverse_openclip_cache.destination.jsonl"
cmp "$MIG/artiverse_openclip_cache.source.jsonl" \
    "$MIG/artiverse_openclip_cache.destination.jsonl"
```

Reject any symlink whose resolved target leaves that model root. Both manifests
stay under the migration metadata root, never inside the model root.

After the new `.venv` and cache proof are ready, do not perform the 87 GB
extraction or OpenCLIP preparation on the login node. Save this job template
outside the worktree as
`$MIG/TEMPLATE_paracloud_artiverse_prepare.sbatch` during `VALIDATION_WRITE`:

```bash
#!/usr/bin/env bash
#SBATCH -J artiverse_prepare
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /data/run01/scvj260/migration_logs_20260711/artiverse-%j.out
#SBATCH --export=NONE
set -euo pipefail
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
REPO=/data/run01/scvj260/scenesmith-hts.stage
(cd "$MIG" && sha256sum -c TEMPLATE_paracloud_artiverse_prepare.sha256)
source "$MIG/paracloud_migration_env.sh"
(cd "$MIG" && sha256sum -c capacity_ledger.sha256)
required=$(python3 - "$MIG/capacity_ledger.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8')); assert d['status']=='pass'
print(d['guards']['G_MATERIALIZE'])
PY
)
free=$(df -B1 --output=avail /data/run01/scvj260 | tail -1 | tr -d ' ')
test "$free" -ge "$required"
cd "$REPO"
source local_setup/compute_node_env.sh
mkdir -p /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
"$PYTHON_BIN" scripts/safe_extract_artiverse.py \
  --repository-root /data/run01/scvj260/assets/artiverse_sqz_20260711/repository
"$PYTHON_BIN" scripts/safe_extract_artiverse.py \
  --repository-root /data/run01/scvj260/assets/artiverse_sqz_20260711/repository \
  --verify-existing

mkdir -p data
if [[ ! -e data/artiverse && ! -L data/artiverse ]]; then
  ln -s /data/run01/scvj260/assets/artiverse_sqz_20260711/repository data/artiverse
fi
test "$(readlink -f data/artiverse)" = \
  /data/run01/scvj260/assets/artiverse_sqz_20260711/repository

"$PYTHON_BIN" scripts/prepare_artiverse.py \
  --dataset-root data/artiverse \
  --output-path data/artiverse/embeddings \
  --source-revision 8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248 \
  --max-collision-elements 32 \
  --minimum-indexed 500 \
  --minimum-indexed-per-category 3 \
  --max-failure-fraction 0.11 \
  --categories armoire bookcase chest_of_drawers

"$PYTHON_BIN" scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/artiverse_preparation_validation.json
```

Use Claude Code's structured file editor, then `chmod 0750`, run `bash -n`, and
record SHA-256. After explicit `GPU_JOB` approval, submit this one job and monitor
it with `squeue -u scvj260`; inspect the exact SLURM log and `sacct -j <jobid>`.
Do not blind-resubmit after an interruption: first inspect the extractor's
transaction state and preparation manifest.

```bash
set -euo pipefail
JOB=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_artiverse_prepare.sbatch
chmod 0750 "$JOB"
bash -n "$JOB"
sha256sum "$JOB" | tee /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_artiverse_prepare.sha256
# Run the next line only after separate GPU_JOB approval.
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before Artiverse job}"
CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
  /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh G_MATERIALIZE
sha256sum -c /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_artiverse_prepare.sha256
sbatch "$JOB"
squeue -u scvj260
```

The extractor/contract must reproduce the expected 3,544 roots, 531,937 files,
86,992,752,890 bytes, 560 considered school-role candidates, exactly 500 indexed
candidates, and the known category counts (98 armoire, 22 bookcase, 380
chest-of-drawers). Stop on any mismatch; a weaker generic pass is insufficient.

## 10. Phase H - migrate ObjectThor (`LARGE_ASSET_WRITE` / `VALIDATION_WRITE`)

Run the ParaCloud capacity guard as phase `H` before every non-dry-run rsync.

Use the filtered ready payload (approximately 47.3 GB after omitting the known
10.98 GB interrupted archive plus extraction/lock residue). The repository does not currently expose a
separately attested extraction-only ObjectThor entry point for ParaCloud, so the
smaller raw-archive route is not approved for a zero-context migration. Copying
the ready cache avoids accidentally invoking the network-capable SQZ downloader
and preserves its prepared 50,092-object index.

```bash
set -euo pipefail
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
SRC=/root/.objathor-assets/2023_09_23
DST=/data/run01/scvj260/assets/objathor_2023_09_23
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before ObjectThor transfer}"
guard_h() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh H"
}

# The profile includes only: assets/, features/, preprocessed/, assets.tar,
# annotations.json.gz, and features.tar. It rejects objects.lock,
# *.extracting, *.interrupted-*, *.resume, and every unrecognized top-level
# entry instead of silently copying interrupted downloader state.
"$PY" "$META/migration_exact_manifest.py" \
  --root "$SRC" --profile objathor-ready --link-policy reject \
  --manifest "$META/objathor_ready.fresh.jsonl" \
  --paths0 "$META/objathor_ready.fresh.paths0"
cmp "$META/objathor_ready.source.jsonl" "$META/objathor_ready.fresh.jsonl"
cmp "$META/objathor_ready.paths0" "$META/objathor_ready.fresh.paths0"
guard_h
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/objathor_ready.source.jsonl" "$META/objathor_ready.paths0" \
  "$META/objathor_raw_source.sha256" "$HOST:$MIG/"
rsync -anH --no-owner --no-group --from0 \
  --files-from="$META/objathor_ready.paths0" --itemize-changes -e "$RSH" \
  "$SRC/" "$HOST:$DST/"
guard_h
rsync -aH --no-owner --no-group --from0 \
  --files-from="$META/objathor_ready.paths0" --partial --info=progress2 -e "$RSH" \
  "$SRC/" "$HOST:$DST/"
```

The source manifests stay outside the scanned root. On ParaCloud:

```bash
set -euo pipefail
OBJ=/data/run01/scvj260/assets/objathor_2023_09_23
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
python3 "$MIG/migration_exact_manifest.py" \
  --root "$OBJ" --profile objathor-ready --link-policy reject \
  --manifest "$MIG/objathor_ready.destination.jsonl"
cmp "$MIG/objathor_ready.source.jsonl" \
    "$MIG/objathor_ready.destination.jsonl"
cd "$OBJ"
sha256sum -c "$MIG/objathor_raw_source.sha256"
test "$(stat -c %s assets.tar)" = 23177386496
test "$(stat -c %s annotations.json.gz)" = 9740343
test "$(stat -c %s features.tar)" = 388221440
test -d assets
test -d features
test -d preprocessed
test ! -e objects.lock
test ! -e assets.extracting
test ! -e features.extracting
```

Before running retrieval, parse and bind its separate model cache on SQZ:

```bash
set -euo pipefail
SRC=/root/workspace/scenesmith-hts
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
"$PY" - "$SRC" "$META/objathor_cache_binding.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
if out.exists() or out.is_symlink(): raise SystemExit('refusing ObjectThor binding overwrite')
receipt = repo / 'outputs/preflight/full_quality_school_reference_20260710/objathor_retrieval_offline.json'
doc = json.loads(receipt.read_text(encoding='utf-8'))
records = doc.get('evidence', {}).get('artifacts')
if not isinstance(records, list): raise SystemExit('bad ObjectThor receipt artifacts')
cache = [r for r in records if isinstance(r, dict) and str(r.get('path','')).startswith('/root/.cache/')]
if len(cache) != 1: raise SystemExit(f'expected one ObjectThor cache artifact, got {len(cache)}')
r = cache[0]
expected = {
 'path': '/root/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/blobs/7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019',
 'size_bytes': 1710517748,
 'sha256': '7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019',
}
if any(r.get(k) != v for k, v in expected.items()): raise SystemExit('unexpected ObjectThor cache binding')
p = Path(expected['path'])
if not p.is_file() or p.stat().st_size != expected['size_bytes']: raise SystemExit('missing ObjectThor model blob')
h=hashlib.sha256()
with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
if h.hexdigest() != expected['sha256']: raise SystemExit('ObjectThor model blob hash mismatch')
model_path = Path(doc.get('model', {}).get('cache_path',''))
root = Path('/root/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K')
if not model_path.is_file() or root not in model_path.parents: raise SystemExit('bad ObjectThor snapshot path')
out.write_text(json.dumps({'model_root':str(root), 'artifact':expected}, sort_keys=True)+'\n', encoding='utf-8')
PY

MODEL=/root/.cache/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K
"$PY" "$META/migration_exact_manifest.py" \
  --root "$MODEL" --profile full --link-policy internal \
  --manifest "$META/objathor_model_cache.fresh.jsonl" \
  --paths0 "$META/objathor_model_cache.fresh.paths0"
cmp "$META/objathor_model_cache.source.jsonl" \
    "$META/objathor_model_cache.fresh.jsonl"
cmp "$META/objathor_model_cache.paths0" \
    "$META/objathor_model_cache.fresh.paths0"

KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
DST=/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before ObjectThor model cache}"
guard_h() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh H"
}
guard_h
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/objathor_cache_binding.json" \
  "$META/objathor_model_cache.source.jsonl" \
  "$META/objathor_model_cache.paths0" \
  "$HOST:/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/"
rsync -anH --no-owner --no-group --safe-links --from0 \
  --files-from="$META/objathor_model_cache.paths0" --itemize-changes -e "$RSH" \
  "$MODEL/" "$HOST:$DST/"
guard_h
rsync -aH --no-owner --no-group --safe-links --from0 \
  --files-from="$META/objathor_model_cache.paths0" --partial --info=progress2 -e "$RSH" \
  "$MODEL/" "$HOST:$DST/"
```

On ParaCloud, generate the destination manifest with `--link-policy internal`,
require byte-identical `cmp`, and rehash the mapped blob:

```bash
set -euo pipefail
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
MODEL=/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K
python3 "$MIG/migration_exact_manifest.py" \
  --root "$MODEL" --profile full --link-policy internal \
  --manifest "$MIG/objathor_model_cache.destination.jsonl"
cmp "$MIG/objathor_model_cache.source.jsonl" \
    "$MIG/objathor_model_cache.destination.jsonl"
test "$(stat -c %s "$MODEL/blobs/7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019")" = 1710517748
echo '7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019  '"$MODEL"'/blobs/7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019' | sha256sum -c -
test -e "$MODEL/snapshots/1627032197142fbe2a7cfec626f4ced3ae60d07a/open_clip_pytorch_model.safetensors"
```

The mapped blob is:

```text
/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/blobs/7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019
```

It must still have exactly 1,710,517,748 bytes and SHA-256
`7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019`.
Do not write anything into `/data/run01/scvj260/.cache`.

Create `data/objathor-assets` as a link to the shared destination only after the
cache is extracted and prepared:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
test ! -e data/objathor-assets
ln -s /data/run01/scvj260/assets/objathor_2023_09_23 data/objathor-assets
mkdir -p /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
.venv/bin/python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/objathor_retrieval_offline.json
.venv/bin/python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/objathor_retrieval_offline.json \
  --verify-only
```

Require schema/version/status pass, exactly 50,092 objects, 768-dimensional
embeddings, a real offline CPU query, production `.pkl.gz` loads, and an
independent verify-only pass. Any attempted network access is a migration
failure.

## 11. Phase I - materials, ArtVIP, HSSD, external, and SAM3D (`LARGE_ASSET_WRITE` / `VALIDATION_WRITE`)

Run the ParaCloud capacity guard as phase `I` before every local or direct
non-dry-run rsync and before the DINO checkpoint write.

Do not link the new checkout into legacy paths. First validate, then copy the
useful trees to stable shared roots, exact-compare them, and only then create
links. The code payload must already contain
`data/materials_full_quality_contract`.

On ParaCloud, validate the legacy material source through its absolute path:

```bash
set -euo pipefail
TARGET=/data/run01/scvj260/scenesmith-hts.stage
LEGACY=/data/run01/scvj260/scenesmith
cd "$TARGET"
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
mkdir -p /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
.venv/bin/python scripts/materials_contract.py validate \
  --data-root "$LEGACY/data/materials" \
  --source-embeddings "$LEGACY/data/materials/embeddings" \
  --contract-embeddings data/materials_full_quality_contract/embeddings \
  --min-retained 1900 \
  --max-pruned 15 \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/materials_legacy_source_validation.json

test -s "$LEGACY/data/artvip_sdf/embeddings/clip_embeddings.npy"
test -s "$LEGACY/data/artvip_sdf/embeddings/embedding_index.yaml"
test -s "$LEGACY/data/artvip_sdf/embeddings/metadata_index.yaml"
```

Before copying materials or ArtVIP, run on SQZ and transfer both authoritative
manifests:

```bash
set -euo pipefail
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before authority-manifest transfer}"
guard_i_remote() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh I"
}
for row in \
  'materials_sqz_authority /root/workspace/scenesmith/data/materials' \
  'artvip_sqz_authority /root/workspace/scenesmith/data/artvip_sdf'; do
  read -r name src <<<"$row"
  "$PY" "$META/migration_exact_manifest.py" --root "$src" --profile full \
    --link-policy reject --manifest "$META/${name}.fresh.jsonl"
  cmp "$META/${name}.source.jsonl" "$META/${name}.fresh.jsonl"
done
guard_i_remote
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/materials_sqz_authority.source.jsonl" \
  "$META/artvip_sqz_authority.source.jsonl" \
  "$HOST:/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/"
```

Generate full `--link-policy reject` source manifests for the three legacy
roots below, copy into the already-created empty shared destinations, generate
destination manifests, and require byte-identical `cmp` for each pair:

```bash
set -euo pipefail
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
HELPER="$MIG/migration_exact_manifest.py"
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before Phase I copies}"
guard_i() {
  CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
    "$MIG/capacity_guard.sh" I
}
for row in \
  'materials /data/run01/scvj260/scenesmith/data/materials /data/run01/scvj260/assets/materials_sqz_20260711' \
  'artvip /data/run01/scvj260/scenesmith/data/artvip_sdf /data/run01/scvj260/assets/artvip_sqz_20260711' \
  'hssd /data/run01/scvj260/scenesmith/data/hssd-models /data/run01/scvj260/assets/hssd_legacy'; do
  read -r name src dst <<<"$row"
  test -d "$src"
  test -d "$dst"
  test -z "$(find "$dst" -mindepth 1 -print -quit)"
  expected=$(python3 - "$MIG/capacity_ledger.json" "$name" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['legacy_source_bytes'][sys.argv[2]])
PY
)
  actual=$(du --apparent-size -s -B1 "$src" | cut -f1)
  test "$actual" = "$expected"
  python3 "$HELPER" --root "$src" --profile full --link-policy reject \
    --manifest "$MIG/${name}.source.jsonl" --paths0 "$MIG/${name}.paths0"
  if [[ "$name" == materials ]]; then
    cmp "$MIG/materials_sqz_authority.source.jsonl" "$MIG/materials.source.jsonl"
  elif [[ "$name" == artvip ]]; then
    cmp "$MIG/artvip_sqz_authority.source.jsonl" "$MIG/artvip.source.jsonl"
  fi
  rsync -anH --no-owner --no-group --from0 \
    --files-from="$MIG/${name}.paths0" --itemize-changes "$src/" "$dst/"
  guard_i
  rsync -aH --no-owner --no-group --from0 \
    --files-from="$MIG/${name}.paths0" --partial --info=progress2 "$src/" "$dst/"
  python3 "$HELPER" --root "$dst" --profile full --link-policy reject \
    --manifest "$MIG/${name}.destination.jsonl"
  cmp "$MIG/${name}.source.jsonl" "$MIG/${name}.destination.jsonl"
done
```

Before accepting the ArtVIP copy, require the destination manifest to equal the
SQZ authority too, then require exactly 197 index entries:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
cmp "$MIG/artvip_sqz_authority.source.jsonl" "$MIG/artvip.source.jsonl"
cmp "$MIG/artvip_sqz_authority.source.jsonl" "$MIG/artvip.destination.jsonl"
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
p=Path('/data/run01/scvj260/assets/artvip_sqz_20260711/embeddings/embedding_index.yaml')
d=yaml.safe_load(p.read_text(encoding='utf-8'))
assert isinstance(d, list) and len(d) == 197, (type(d), len(d) if hasattr(d,'__len__') else None)
print('ArtVIP indexed records:', len(d))
PY

test ! -e data/materials
test ! -e data/artvip_sdf
ln -s /data/run01/scvj260/assets/materials_sqz_20260711 data/materials
ln -s /data/run01/scvj260/assets/artvip_sqz_20260711 data/artvip_sdf
.venv/bin/python scripts/materials_contract.py validate \
  --data-root data/materials \
  --source-embeddings data/materials/embeddings \
  --contract-embeddings data/materials_full_quality_contract/embeddings \
  --min-retained 1900 --max-pruned 15 \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/materials_shared_validation.json
```

HSSD is preserved at its shared root but is not linked because the selected
generated-SAM3D policy does not require it. Never infer that preserving HSSD
authorizes legacy deletion; the retirement audit below also covers historical
outputs, transfer chunks, and unknown files.

Populate `external` from the authoritative SQZ link target instead of assuming
the legacy tree matches:

```bash
# Run on SQZ.
set -euo pipefail
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
SRC=$(readlink -f /root/workspace/scenesmith-hts/external)
DST=/data/run01/scvj260/assets/external_sqz_20260711
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before external transfer}"
guard_i_remote() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh I"
}
test -d "$SRC"
test "$SRC" = "$(cat "$META/external_target.txt")"
"$PY" "$META/migration_exact_manifest.py" \
  --root "$SRC" --profile full --link-policy reject \
  --secret-scan --manifest "$META/external.fresh.jsonl" \
  --paths0 "$META/external.fresh.paths0"
cmp "$META/external.source.jsonl" "$META/external.fresh.jsonl"
cmp "$META/external.paths0" "$META/external.fresh.paths0"
guard_i_remote
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/external.source.jsonl" "$META/external.paths0" "$HOST:$MIG/"
rsync -anH --no-owner --no-group --from0 --files-from="$META/external.paths0" \
  --itemize-changes -e "$RSH" "$SRC/" "$HOST:$DST/"
guard_i_remote
rsync -aH --no-owner --no-group --from0 --files-from="$META/external.paths0" \
  --partial --info=progress2 -e "$RSH" "$SRC/" "$HOST:$DST/"
```

Create and compare the destination manifest, then link it:

```bash
set -euo pipefail
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
EXT=/data/run01/scvj260/assets/external_sqz_20260711
python3 "$MIG/migration_exact_manifest.py" --root "$EXT" --profile full \
  --link-policy reject --manifest "$MIG/external.destination.jsonl"
cmp "$MIG/external.source.jsonl" "$MIG/external.destination.jsonl"
cd /data/run01/scvj260/scenesmith-hts.stage
source "$MIG/paracloud_migration_env.sh"
test ! -e external
ln -s "$EXT" external
test -s external/checkpoints/sam3.pt
test -s external/checkpoints/pipeline.yaml
```

For SAM3D, bind all 17 receipt artifacts before copying cache data. Run this on
SQZ; it prints no file content and writes only paths, kinds, sizes, and hashes:

```bash
set -euo pipefail
SRC=/root/workspace/scenesmith-hts
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
"$PY" - "$META/sam3d_cache_binding.json" "$META/external_target.txt" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
receipt=Path('/root/workspace/scenesmith-hts/outputs/preflight/full_quality_school_reference_20260710/sam3d_offline_load.json')
doc=json.loads(receipt.read_text(encoding='utf-8'))
if not (
    doc.get('schema_version')==2 and doc.get('status')=='pass' and
    doc.get('offline') is True and doc.get('model_loaded') is True and
    doc.get('pipeline_loaded') is True and
    doc.get('evidence_verification',{}).get('status')=='pass' and
    doc.get('attestation',{}).get('sha256')=='0f6ac5f808d19beeec01f75054138b7671681343116ebc7b804de46ac5e0c258'
): raise SystemExit('source SAM3D receipt is not the exact passing offline receipt')
records=doc.get('evidence',{}).get('artifacts')
if not isinstance(records,list) or len(records)!=17:
    raise SystemExit('SAM3D receipt must contain exactly 17 artifacts')
external=Path(sys.argv[2]).read_text(encoding='utf-8').strip()
external=Path(external).resolve(strict=True)
if external != Path('/root/workspace/scenesmith/external'):
    raise SystemExit('unexpected authoritative external target')
cache=Path('/root/.cache').resolve(strict=True)
expected_cache={
 '/root/.cache/huggingface/hub/models--Ruicheng--moge-vitl/blobs/da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f':
   ('file',1256823446,'da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f'),
 '/root/.cache/torch/hub/facebookresearch_dinov2_main':
   ('python_source_tree',None,'3871e6c51f1a18fdbc93384ce7cc3b0f3e107c9997d678a48282c07024d2535e'),
 '/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth':
   ('file',1217607321,'36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51'),
}
seen_cache={}
external_rows=[]
def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def hash_python_tree(root):
    files=sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts)
    if not files: raise SystemExit('empty SAM3D Python source tree')
    h=hashlib.sha256(); total=0
    for p in files:
        rel=p.relative_to(root).as_posix(); size=p.stat().st_size; digest=hash_file(p)
        h.update(rel.encode()); h.update(b'\0'); h.update(str(size).encode('ascii'))
        h.update(b'\0'); h.update(digest.encode('ascii')); h.update(b'\n'); total+=size
    return h.hexdigest(),len(files),total
for r in records:
    if not isinstance(r,dict): raise SystemExit('non-object SAM3D artifact')
    raw=r.get('path'); kind=r.get('kind'); digest=r.get('sha256')
    if not all(isinstance(v,str) for v in (raw,kind,digest)):
        raise SystemExit('bad SAM3D artifact fields')
    if any(c in raw for c in '\r\n\t\0'): raise SystemExit('unsafe SAM3D artifact path')
    path=Path(raw)
    if external in path.parents:
        if kind!='file' or not path.is_file(): raise SystemExit('bad external SAM3D artifact')
        size=r.get('size_bytes')
        if not isinstance(size,int) or path.stat().st_size!=size or hash_file(path)!=digest:
            raise SystemExit('external SAM3D artifact mismatch: '+raw)
        external_rows.append({'path':raw,'size_bytes':size,'sha256':digest})
    elif cache in path.parents or path==cache:
        if raw not in expected_cache: raise SystemExit('unapproved SAM3D cache path: '+raw)
        expected=expected_cache[raw]
        if (kind,r.get('size_bytes'),digest)!=expected:
            raise SystemExit('SAM3D cache receipt mismatch: '+raw)
        if kind=='file':
            if not path.is_file() or path.stat().st_size!=expected[1] or hash_file(path)!=digest:
                raise SystemExit('SAM3D cache file mismatch: '+raw)
        elif not path.is_dir(): raise SystemExit('missing SAM3D source tree: '+raw)
        else:
            tree_hash,count,total=hash_python_tree(path)
            if not (
                tree_hash==digest and r.get('file_count')==count==157 and
                r.get('total_bytes')==total==755736 and
                r.get('selection')=='**/*.py excluding __pycache__'
            ): raise SystemExit('SAM3D Python source-tree digest mismatch')
        seen_cache[raw]={'kind':kind,'size_bytes':expected[1],'sha256':digest}
    else:
        raise SystemExit('unapproved SAM3D artifact root: '+raw)
if len(external_rows)!=14 or set(seen_cache)!=set(expected_cache):
    raise SystemExit('SAM3D artifact classification mismatch')
out=Path(sys.argv[1])
if out.exists() or out.is_symlink(): raise SystemExit('refusing SAM3D binding overwrite')
out.write_text(json.dumps({
 'receipt_attestation':doc.get('attestation',{}).get('sha256'),
 'external':sorted(external_rows,key=lambda r:r['path']),
 'cache':seen_cache,
},sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
PY
```

It must require exactly these classifications:

- 14 regular files beneath the resolved authoritative `external` root; verify
  every recorded size/SHA-256 and require that the transferred external
  manifest covers them;
- the MoGe blob beneath the single bounded root
  `/root/.cache/huggingface/hub/models--Ruicheng--moge-vitl`;
- the exact Python source tree
  `/root/.cache/torch/hub/facebookresearch_dinov2_main`;
- the exact file
  `/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`, size
  1,217,607,321 and SHA-256
  `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`.

Reject every other kind/root, CR/LF/tab-containing path, missing file, size
mismatch, or hash mismatch. Write the non-secret classification to
`$META/sam3d_cache_binding.json`. Then use Appendix A as follows:

```bash
# Run on SQZ after the parser above succeeds.
set -euo pipefail
META=/root/workspace/scenesmith-hts-migration-metadata-20260711
PY=/root/workspace/miniconda3/envs/scenesmith/bin/python
MOGE=/root/.cache/huggingface/hub/models--Ruicheng--moge-vitl
DINO=/root/.cache/torch/hub/facebookresearch_dinov2_main
"$PY" "$META/migration_exact_manifest.py" --root "$MOGE" --profile full \
  --link-policy internal --manifest "$META/sam3d_moge.fresh.jsonl" \
  --paths0 "$META/sam3d_moge.fresh.paths0"
"$PY" "$META/migration_exact_manifest.py" --root "$DINO" --profile full \
  --link-policy internal --manifest "$META/sam3d_dinov2_source.fresh.jsonl" \
  --paths0 "$META/sam3d_dinov2_source.fresh.paths0"
cmp "$META/sam3d_moge.source.jsonl" "$META/sam3d_moge.fresh.jsonl"
cmp "$META/sam3d_moge.paths0" "$META/sam3d_moge.fresh.paths0"
cmp "$META/sam3d_dinov2_source.source.jsonl" \
    "$META/sam3d_dinov2_source.fresh.jsonl"
cmp "$META/sam3d_dinov2_source.paths0" \
    "$META/sam3d_dinov2_source.fresh.paths0"

KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
HOST=ssh.cn-zhongwei-1.paracloud.com
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 -l scvj260@NC-N50R5"
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
MOGE_DST=/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--Ruicheng--moge-vitl
DINO_DST=/data/run01/scvj260/cache_sqz_20260711/torch/hub/facebookresearch_dinov2_main
CKPT=/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
CKPT_DST=/data/run01/scvj260/cache_sqz_20260711/torch/hub/checkpoints/
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before SAM3D cache transfer}"
guard_i_remote() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/root/.ssh/known_hosts.paracloud_migration_20260711 \
    -l 'scvj260@NC-N50R5' "$HOST" \
    "CONFIRMED_QUOTA_REMAINING_BYTES_NOW=$CONFIRMED_QUOTA_REMAINING_BYTES_NOW /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh I"
}
guard_i_remote
rsync -a --no-owner --no-group -e "$RSH" \
  "$META/sam3d_cache_binding.json" "$META"/sam3d_*.source.jsonl \
  "$META"/sam3d_*.paths0 "$HOST:$MIG/"
guard_i_remote
rsync -aH --no-owner --no-group --safe-links --from0 \
  --files-from="$META/sam3d_moge.paths0" --partial --info=progress2 -e "$RSH" \
  "$MOGE/" "$HOST:$MOGE_DST/"
guard_i_remote
rsync -aH --no-owner --no-group --safe-links --from0 \
  --files-from="$META/sam3d_dinov2_source.paths0" --partial --info=progress2 -e "$RSH" \
  "$DINO/" "$HOST:$DINO_DST/"
guard_i_remote
rsync -a --no-owner --no-group --partial --info=progress2 -e "$RSH" \
  "$CKPT" "$HOST:$CKPT_DST"
```

These commands map the two bounded roots, without `-R`, to:

```text
/data/run01/scvj260/cache_sqz_20260711/huggingface/hub/models--Ruicheng--moge-vitl
/data/run01/scvj260/cache_sqz_20260711/torch/hub/facebookresearch_dinov2_main
```

On ParaCloud, generate and compare the two destination manifests, then rehash
both receipt-bound cache blobs:

```bash
set -euo pipefail
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
CACHE=/data/run01/scvj260/cache_sqz_20260711
python3 "$MIG/migration_exact_manifest.py" \
  --root "$CACHE/huggingface/hub/models--Ruicheng--moge-vitl" --profile full \
  --link-policy internal --manifest "$MIG/sam3d_moge.destination.jsonl"
python3 "$MIG/migration_exact_manifest.py" \
  --root "$CACHE/torch/hub/facebookresearch_dinov2_main" --profile full \
  --link-policy internal --manifest "$MIG/sam3d_dinov2_source.destination.jsonl"
cmp "$MIG/sam3d_moge.source.jsonl" "$MIG/sam3d_moge.destination.jsonl"
cmp "$MIG/sam3d_dinov2_source.source.jsonl" \
    "$MIG/sam3d_dinov2_source.destination.jsonl"
echo 'da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f  '"$CACHE"'/huggingface/hub/models--Ruicheng--moge-vitl/blobs/da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f' | sha256sum -c -
test "$(stat -c %s "$CACHE/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")" = 1217607321
echo '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51  '"$CACHE"'/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth' | sha256sum -c -
```

Do not copy `.locks`, unrelated Hugging Face models, or anything into
`/data/run01/scvj260/.cache`.

The real SAM3D load occurs only in Phase K's GPU job and writes a new
migration-specific receipt:

```bash
/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/sam3d_offline_load.json
```

Never run the SAM3D load on the login node.

## 12. Phase J - create a path-correct ParaCloud environment (`ENVIRONMENT_WRITE`)

Execute this section immediately after F (before G). Run the ParaCloud capacity
guard as phase `J` before `uv sync`, then as `J_MUJOCO` before creation and
each install into the isolated simulator environment.

Do not copy SQZ Miniconda. Do not copy or symlink the legacy `.venv`: activation
scripts and entry-point shebangs contain absolute paths, and
`compute_node_env.sh` requires `<repo>/.venv` to identify as that repository's
environment.

Use the existing ParaCloud UV/Hugging Face caches and try an offline, lock-bound
environment creation only after approval:

```bash
cd /data/run01/scvj260/scenesmith-hts.stage
test -f pyproject.toml
test -f uv.lock
command -v uv
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
test -d "$UV_CACHE_DIR"
test ! -e .venv
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before main environment}"
CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
  /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh J
uv sync --frozen --offline
```

If this reports a missing artifact, stop and list the missing package. Do not
fall back to an online install without explicit approval.

Check imports without a GPU:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
.venv/bin/python - <<'PY'
import scenesmith
import torch
import pydrake
print('scenesmith', scenesmith.__file__)
print('torch', torch.__version__)
print('pydrake', pydrake.__file__)
PY
```

Recreate the isolated MuJoCo/OpenUSD environment from the captured exact package
list; do not byte-copy SQZ `.mujoco_venv`:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
REQ=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/mujoco_environment_requirements.txt
(cd /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711 && sha256sum -c mujoco_environment.sha256)
test -s "$REQ"
test ! -e .mujoco_venv
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before simulator environment}"
CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
  /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh J_MUJOCO
uv venv --python "$PWD/.venv/bin/python" .mujoco_venv
CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
  /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh J_MUJOCO
uv pip install --offline --python "$PWD/.mujoco_venv/bin/python" -r "$REQ"
CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
  /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh J_MUJOCO
uv pip install --offline --no-deps --python "$PWD/.mujoco_venv/bin/python" --editable "$PWD"
.mujoco_venv/bin/python - <<'PY'
import mujoco
import mujoco_usd_converter
import usdex.core
from pxr import Usd
import scenesmith
print('mujoco',mujoco.__version__)
print('scenesmith',scenesmith.__file__)
PY
```

If the ParaCloud UV cache lacks any exact package, stop and report its name and
version. Do not copy the venv, loosen versions, or install online. A separately
approved wheelhouse transfer from SQZ would be a new payload and must be
manifested/capacity-counted first.

Every SLURM script must source the migration-owned environment wrapper **before**
sourcing `local_setup/compute_node_env.sh`; otherwise the helper defaults to the
full 1 GiB home filesystem. Exports from a prior interactive shell do not count.

## 13. Phase K - integrity and runtime validation (`VALIDATION_WRITE` / `GPU_JOB`)

Run the ParaCloud capacity guard as phase `K` immediately before the final
two-GPU `sbatch`.

On ParaCloud staging:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
mkdir -p /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711

.venv/bin/python - <<'PY'
from pathlib import Path
repo=Path('/data/run01/scvj260/scenesmith-hts.stage')
expected={
 'external':'/data/run01/scvj260/assets/external_sqz_20260711',
 'data/artiverse':'/data/run01/scvj260/assets/artiverse_sqz_20260711/repository',
 'data/objathor-assets':'/data/run01/scvj260/assets/objathor_2023_09_23',
 'data/materials':'/data/run01/scvj260/assets/materials_sqz_20260711',
 'data/artvip_sdf':'/data/run01/scvj260/assets/artvip_sqz_20260711',
}
for rel,target in expected.items():
    p=repo/rel
    assert p.is_symlink(), rel
    assert p.resolve(strict=True)==Path(target).resolve(strict=True), (rel,p.resolve())
for p in repo.rglob('*'):
    if p.is_symlink() and not ({'.venv','.mujoco_venv'} & set(p.parts)):
        rel=p.relative_to(repo).as_posix()
        assert rel in expected, ('unexpected non-venv link',rel)
        assert not str(p.resolve()).startswith('/data/run01/scvj260/scenesmith/')
print('approved shared links:',len(expected))
PY
.venv/bin/python - <<'PY'
import os,sys
assert os.path.realpath(sys.prefix)==os.path.realpath('/data/run01/scvj260/scenesmith-hts.stage/.venv')
PY
.mujoco_venv/bin/python - <<'PY'
import os,sys
assert os.path.realpath(sys.prefix)==os.path.realpath('/data/run01/scvj260/scenesmith-hts.stage/.mujoco_venv')
PY
```

Use Claude Code's structured file editor to create the fresh file
`/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/mujoco_smoke/input/simple_box.sdf`
with this exact content:

```xml
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="simple_box">
    <link name="base">
      <inertial>
        <mass>1.0</mass>
        <inertia><ixx>0.1666667</ixx><iyy>0.1666667</iyy><izz>0.1666667</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="visual"><geometry><box><size>1 1 1</size></box></geometry></visual>
      <collision name="collision"><geometry><box><size>1 1 1</size></box></geometry></collision>
    </link>
  </model>
</sdf>
```

Then perform a real isolated export and independent validation:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
ROOT=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/mujoco_smoke
test -f "$ROOT/input/simple_box.sdf"
test ! -e "$ROOT/export"
.mujoco_venv/bin/python scripts/export_scene_to_mujoco.py \
  --sdf "$ROOT/input/simple_box.sdf" --output "$ROOT/export" --static --usd
.mujoco_venv/bin/python scripts/validate_simulator_exports.py \
  --output-dir "$ROOT/export" --require-usd \
  --output "$ROOT/validation.json"
.mujoco_venv/bin/python - "$ROOT/export" <<'PY'
from pathlib import Path
import sys,mujoco
from pxr import Usd
root=Path(sys.argv[1])
mujoco.MjModel.from_xml_path(str(root/'scene.xml'))
layers=sorted([*root.rglob('*.usd'),*root.rglob('*.usda'),*root.rglob('*.usdc')])
assert layers
for layer in layers:
    assert Usd.Stage.Open(str(layer)), layer
print('validated USD layers:',len(layers))
PY
```

Resume the remaining migration contracts only after this passes:

```bash
set -euo pipefail
cd /data/run01/scvj260/scenesmith-hts.stage
source /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/paracloud_migration_env.sh
.venv/bin/python scripts/pipeline_code_contract.py \
  --repo-dir "$PWD" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/pipeline_code_contract_paracloud.json
.venv/bin/python scripts/pipeline_code_contract.py \
  --repo-dir "$PWD" \
  --spec CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md \
  --runner remote_jobs/run_full_quality_school_sqz.sh \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/pipeline_code_contract_paracloud.json \
  --verify-only

.venv/bin/python scripts/validate_input_manifest.py \
  --input-dir inputs/full_quality_school_reference_20260710 \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/input_manifest_validation.json

.venv/bin/python scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/artiverse_preparation_validation.json

.venv/bin/python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/objathor_retrieval_offline.json \
  --verify-only
```

The SQZ runner is hard-coded for SQZ paths and environment setup. Do not launch
`remote_jobs/run_full_quality_school_sqz.sh` unchanged on ParaCloud. Port its
exact orchestration into a ParaCloud SLURM entry point or use the attested SLURM
templates only after path/runtime contract tests pass.

Before any generation, submit a no-paid-API GPU preflight job that:

1. sources `local_setup/compute_node_env.sh` inside SLURM;
2. proves the new repo and `.venv` paths;
3. runs the real SAM3D offline load and verify-only pass;
4. checks `torch.cuda.device_count()`;
5. performs no OpenAI/VLM call and creates no room output.

Create the following template under the migration metadata root, **outside the
worktree**, during `VALIDATION_WRITE`; review it and record its SHA-256 before
requesting `GPU_JOB` approval. Keeping it outside the transferred worktree means
it cannot invalidate the already-compared code manifest:

```bash
#!/usr/bin/env bash
#SBATCH -J scenesmith_migration_check
#SBATCH -p gpu
#SBATCH --qos=gpugpu
#SBATCH -A scvj260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scvj260/migration_logs_20260711/preflight-%j.out
#SBATCH --export=NONE

set -euo pipefail
REPO=/data/run01/scvj260/scenesmith-hts.stage
MIG=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711
(cd "$MIG" && sha256sum -c TEMPLATE_paracloud_migration_preflight.sha256)
source "$MIG/paracloud_migration_env.sh"
export REPO_ROOT="$REPO"

cd "$REPO"
source local_setup/compute_node_env.sh
"$PYTHON_BIN" - <<'PY'
import os
import torch
assert os.path.realpath(os.environ['REPO_ROOT']) == os.path.realpath('/data/run01/scvj260/scenesmith-hts.stage')
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
values=[]
for index in (0,1):
    with torch.cuda.device(index):
        x=torch.arange(4096,device=f'cuda:{index}',dtype=torch.float32)
        value=float(x.sum().cpu())
        torch.cuda.synchronize(index)
        values.append(value)
        print(index,torch.cuda.get_device_name(index),torch.cuda.get_device_properties(index).total_memory,value)
assert values[0] == values[1] == 8386560.0
PY
"$PYTHON_BIN" scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/sam3d_offline_load.json
"$PYTHON_BIN" scripts/preflight_sam3d_offline.py \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --pipeline-config external/checkpoints/pipeline.yaml \
  --output /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/sam3d_offline_load.json \
  --verify-only
```

Use Claude Code's structured file-edit tool to save that block as
`/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_migration_preflight.sbatch`,
then run under `VALIDATION_WRITE`:

```bash
set -euo pipefail
JOB=/data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_migration_preflight.sbatch
chmod 0750 "$JOB"
bash -n "$JOB"
sha256sum "$JOB" | tee /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_migration_preflight.sha256
```

After separate `GPU_JOB` approval, submit and monitor:

```bash
test -f /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_migration_preflight.sbatch
: "${CONFIRMED_QUOTA_REMAINING_BYTES_NOW:?refresh quota before two-GPU preflight}"
CONFIRMED_QUOTA_REMAINING_BYTES_NOW="$CONFIRMED_QUOTA_REMAINING_BYTES_NOW" \
  /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/capacity_guard.sh K
sha256sum -c /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_migration_preflight.sha256
sbatch /data/run01/scvj260/scenesmith-hts-migration-metadata-20260711/TEMPLATE_paracloud_migration_preflight.sbatch
squeue -u scvj260
```

Success requires job exit 0, two real CUDA devices with allocation/reduction/
readback/synchronization on each, a passing full offline SAM3D
load, and a passing independent verify-only receipt. It is not scene generation
and not final two-GPU acceptance.

Do not run the final `TEMPLATE_2gpu_drake_acceptance.sbatch` until SQZ or
ParaCloud has actually completed the full school package and produced:

- `combined_house/sqz_acceptance_record.json`
- external `scene_000.sha256`
- external `pipeline_completion.json`
- their out-of-band hashes and exact `run_attempt_id`

## 14. Phase L - temporary-key cleanup (`CLEANUP`)

After transfer and verification, remove only the exact migration public-key line
from ParaCloud by its unique comment. First preview it:

```bash
grep -n 'sqz-to-paracloud-migration-20260711' ~/.ssh/authorized_keys
```

Removing the line and deleting the temporary SQZ key require explicit approval.
Never rewrite unrelated `authorized_keys` lines and never log the private key.

After that separate approval, run on ParaCloud in one shell:

```bash
set -euo pipefail
FILE=$HOME/.ssh/authorized_keys
COMMENT=sqz-to-paracloud-migration-20260711
test "$(grep -cF "$COMMENT" "$FILE")" = 1
before=$(wc -l <"$FILE")
tmp=$(mktemp "$HOME/.ssh/authorized_keys.cleanup.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
awk -v comment="$COMMENT" 'index($0,comment)==0 {print}' "$FILE" >"$tmp"
test "$(wc -l <"$tmp")" = "$((before-1))"
chmod 0600 "$tmp"
mv -f -- "$tmp" "$FILE"
trap - EXIT
test "$(grep -cF "$COMMENT" "$FILE" || true)" = 0
```

Then run on SQZ:

```bash
set -euo pipefail
KEY=/root/.ssh/sqz_to_paracloud_migration_20260711
KNOWN=/root/.ssh/known_hosts.paracloud_migration_20260711
test -f "$KEY" && test -f "$KEY.pub" && test -f "$KNOWN"
grep -qF 'sqz-to-paracloud-migration-20260711' "$KEY.pub"
rm -f -- "$KEY" "$KEY.pub" "$KNOWN"
```

These exact-file cleanup commands do not touch any project, asset, output, or
legacy checkout.

## 15. Legacy-retirement gate - no deletion command is provided

There are two independent retirement decisions: the old ParaCloud checkout and
the SQZ pod/workspace. Approval for one never authorizes the other.

The old `/data/run01/scvj260/scenesmith` may be considered redundant only when
all of the following are true:

- new code and outputs pass exact hash/contract verification;
- Artiverse safe extraction, preparation, and independent contract pass;
- ObjectThor real offline query and verify-only pass;
- SAM3D real GPU offline load and verify-only pass;
- materials and ArtVIP validations pass;
- HSSD is copied to a stable shared root and checksum-verified;
- every historical output/transfer bundle the user wants is archived and hashed;
- any legacy Git history the user wants has either remained in place or passed a
  dedicated all-object/all-ref secret scan before a separately approved archive;
- the new `.venv` passes imports on a compute node;
- a two-GPU no-paid-API smoke test passes;
- no new checkout link points into the legacy checkout;
- `.secrets` has been handled separately by the user;
- the user explicitly authorizes deletion after reviewing a byte-counted list.

Until then, the only acceptable state is to keep the legacy checkout untouched.
There is intentionally no `rm -rf` command in this handoff.

Likewise, do **not** delete or resize away the SQZ environment merely because
the byte transfer ended. SQZ retirement additionally requires:

- every source/destination manifest comparison and cache binding above passed;
- both fresh ParaCloud environments installed fully offline, including the
  isolated MuJoCo/OpenUSD export smoke;
- Artiverse, ObjectThor, SAM3D, materials, ArtVIP, external, inputs, code, and
  selected outputs all passed their independent ParaCloud verifiers;
- any UV/wheel artifact missing on ParaCloud was resolved while SQZ still exists;
- the stopped benchmark state, repair patch, and goal receipts are preserved;
- the user reviewed exact retained/excluded byte lists and explicitly named the
  SQZ environment for deletion.

No SQZ deletion command is provided either.

## 16. Expected report back from Claude Code

Claude Code should report, with `VERIFIED`, `INFERRED`, `ESTIMATED`, or `UNKNOWN`
on every material statement:

1. exact SQZ payload chosen and byte count;
2. exact ParaCloud free bytes, free inodes, and confirmed account quota;
3. direct transfer median MiB/s and calculated duration;
4. files reused from ParaCloud with matching hashes;
5. files transferred from SQZ with source/destination hashes;
6. Artiverse root/file/byte count and preparation verdict;
7. ObjectThor object/embedding count and offline-query verdict;
8. SAM3D 17-artifact binding and peak VRAM on the compute node;
9. environment/import and GPU smoke-test results;
10. unique legacy items preserved;
11. remaining blockers;
12. an explicit statement that no legacy deletion occurred unless separately
    authorized after all retirement gates passed;
13. until a separate manifested archive task resolves historical outputs,
    transfer chunks, and optional Git history, explicitly report both ParaCloud
    legacy retirement and SQZ retirement as **blocked**, not deletion-ready.
