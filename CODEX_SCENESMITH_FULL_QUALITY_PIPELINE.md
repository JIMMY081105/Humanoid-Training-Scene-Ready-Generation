# Codex SceneSmith Full-Quality Pipeline

This file is the handoff contract for future SceneSmith runs. It exists to prevent the previous failure mode: continuing a fast HSSD-only run while the real target was a best-quality scene with SAM3D/generated assets, ArtVIP/Artiverse routes, and a room self-exam gate before final export.

## Non-Negotiable Output Rules

- Do not use an `hssd_only_fast` run as the final-quality run unless the user explicitly approves it.
- Do not set `general_asset_source=hssd` for a full-quality scene unless the user explicitly asks for HSSD-only speed.
- The current reference-school acceptance run is generated-SAM3D plus retrieval and articulated assets; HSSD is absent, is not required, and may not be introduced into this run as a fallback.
- Keep SAM3D available by using `general_asset_source=generated` and `backend=sam3d` for furniture, wall, ceiling, and manipuland agents when the target is quality.
- Use Objaverse/ObjectThor retrieval for small manipuland objects in full-quality runs to avoid primitive-shape fallbacks.
- Artiverse is compulsory for every full-quality run. The local `data/artiverse` dataset and the `artiverse_articulated` route must exist, be enabled, pass validation, and contribute at least one asset that survives into the final assembled scene. An enabled flag without proven final asset usage does not satisfy this contract.
- Every copied Artiverse SDF used by a room must normalize its material-split OBJ visuals through the matching publisher-authored GLB into deterministic room-local external `.gltf` plus `.bin` resources with explicit normals before Drake/Blender rendering; only the exact derived resource directory and grouped copied visual elements may be added or changed, while publisher sources, collisions, joints, inertials, poses, scales, prompt content, and all quality settings remain unchanged. Direct `.glb` SDF mesh URIs are forbidden because Drake ignores them.
- Keep ArtVIP articulated retrieval enabled alongside Artiverse through `asset_manager.articulated.sources.artvip.enabled=true`; ArtVIP is not a substitute for the compulsory Artiverse route.
- Do not assemble `combined_house`, final Drake exports, Isaac/USD exports, or final renders until every required room has passed a room-level quality gate.
- A render-only review folder is not a quality gate. A gate must produce pass/fail records and must block export on failure.
- Cap generated collision decomposition at 32 convex hulls per object per agent. A simulator export that cannot load is not simulation-ready.
- For the current reference-school run, SQZ `/root/workspace/scenesmith-hts` is the canonical execution checkout; local edits are overlay-uploaded there. ParaCloud is still required for the final 2-GPU Drake acceptance. Record the canonical checkout for every future run and never sync with `--delete` unless explicitly approved.

## Pipeline Overview

1. Read and hash this specification; attest the exact runner, overlay, and ordered patch stack
2. Prompt/reference-image and run setup
3. Environment, model, dataset, index, and GPU preflight
4. Compute-node API proxy verification
5. Full-quality asset policy check
6. Fail-closed ArtVIP/Artiverse router validation
7. Floor-plan-only generation and exact layout gate
8. Sequential room generation
9. Three-view room render capture
10. Deterministic, itemized VLM, and cross-classroom variation gates
11. Room repair/regeneration loop
12. Articulated-motion and Artiverse-provenance-gated final assembly
13. Whole-floor reference-image and navigation gates
14. Drake/SceneEval/collision validation, including final 2-GPU acceptance
15. Atomic Isaac/USD/MuJoCo export if requested
16. Outlook renders and local transfer
17. Hash-complete SQZ acceptance bundle and final validation report

The current SQZ reference-school implementation is `/root/workspace/scenesmith-hts/remote_jobs/run_full_quality_school_sqz.sh`. Run that exact canonical checkout; do not launch a copied runner from the handoff repository. It executes this sequence resumably and fail closed. A rerun may reuse existing artifacts, but every reused artifact is rehashed and semantically revalidated before downstream work. A room with an unchanged, passing, hash-bound visual decision is reused without rerendering or another VLM call; otherwise its final Blender scene is refreshed from the current structured state before review rendering. Successful SQZ completion is deliberately recorded as `awaiting_2gpu_acceptance`, never final `complete`.

## 1. Prompt And Run Setup

Function: define what the scene should be and create a clean run directory.

Required inputs:

- Prompt CSV, for example `inputs/my_scene.csv`.
- The complete source prompt and any supplied reference image. Preserve both under the run input directory and record their hashes in an input manifest.
- Run name that describes the real asset policy, for example `full_quality_school_sam3d_artvip_artiverse`, not `hssd_only_fast`.

Stage `prompt_original.txt`, `scene_contract_appendix.txt`, effective `prompt.txt`, one-row `prompt.csv`, the reference image, and `input_manifest.json` together. Run `scripts/validate_input_manifest.py` before any model/service preflight. It normalizes CRLF to LF for text hashes, reconstructs the effective prompt exactly, compares the CSV row, verifies the raw reference-image hash, and fails on any drift. A contract-compliant manifest must also contain `pipeline_contract.id=scenesmith_full_quality_v1`, `final_assembly_policy=external_artiverse_gated`, and `required_articulated_source=artiverse`; omitting those markers is fatal. For the current school contract the effective prompt is 17,181 characters with SHA-256 `ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f`.

Before any generation-capable code, the runner must fully read this file and create `pipeline_code_contract.json` with `scripts/pipeline_code_contract.py`. That attestation binds the exact specification, production runner, every pipeline Python helper, SAGE checker, remote job, and every patch named in `upstream-patches/APPLY_ORDER.txt`; `--verify-only` must pass again immediately before the SQZ acceptance bundle. A missing, added, removed, reordered, linked, or mutated covered file invalidates the attempt.

Guardrail:

- If resuming another Codex handoff, inspect the run name and Hydra overrides first. If it says `hssd_only`, `fast`, or sets all agents to HSSD, stop and ask before continuing.

## 2. Environment/GPU Preflight

Function: verify that the job is on a GPU node, proxy/API access works, checkpoints exist, and the selected asset sources can load.

Check these before launching a long run:

```bash
nvidia-smi
test -f external/checkpoints/sam3.pt
test -f external/checkpoints/pipeline.yaml
test -f outputs/preflight/<run_name>/sam3d_offline_load.json
test -f outputs/preflight/<run_name>/sam3d_offline_generation/receipt.json
test -d data/materials
test -s data/materials/embeddings/clip_embeddings.npy
test -s data/materials_full_quality_contract/embeddings/clip_embeddings.npy
test -s data/materials_full_quality_contract/embeddings/embedding_index.yaml
test -s data/materials_full_quality_contract/embeddings/metadata_index.yaml
python scripts/materials_contract.py validate \
  --data-root data/materials \
  --source-embeddings data/materials/embeddings \
  --contract-embeddings data/materials_full_quality_contract/embeddings \
  --min-retained 1900 \
  --max-pruned 15 \
  --output outputs/preflight/<run_name>/materials_contract_validation.json
test -d data/artvip_sdf
test -s data/artvip_sdf/embeddings/clip_embeddings.npy
test -s data/artvip_sdf/embeddings/embedding_index.yaml
test -s data/artvip_sdf/embeddings/metadata_index.yaml
test -d data/artiverse
test -s data/artiverse/embeddings/clip_embeddings.npy
test -s data/artiverse/embeddings/embedding_index.yaml
test -s data/artiverse/embeddings/metadata_index.yaml
python scripts/artiverse_contract.py \
  --dataset-root data/artiverse \
  --embeddings-path data/artiverse/embeddings \
  --output outputs/preflight/<run_name>/artiverse_preparation_validation.json
test -d data/objathor-assets
test -s data/objathor-assets/preprocessed/clip_embeddings.npy
test -s data/objathor-assets/preprocessed/embedding_index.yaml
test -s data/objathor-assets/preprocessed/metadata_index.json
python scripts/preflight_objathor_retrieval.py \
  --dataset-root data/objathor-assets \
  --preprocessed-path data/objathor-assets/preprocessed \
  --output outputs/preflight/<run_name>/objathor_retrieval_offline.json \
  --verify-only
curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models
```

Expected OpenAI probe codes: `200`, `401`, or `403`. Other codes mean proxy/network is not ready.

For production, the authenticated probe must return `200`, and the recorded SAM3D offline-load result must have `"status": "pass"`. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `DIFFUSERS_OFFLINE=1` for the run so a missing model fails during preflight instead of triggering a hidden download. Artiverse is compulsory: missing dataset files or any of its three index files is a hard stop, not a reason to fall back to ArtVIP, HSSD, generated primitives, or an unproven configuration flag.

The reproducible SAM3D proof command is `scripts/preflight_sam3d_offline.py`; it loads both the SAM3 image model and complete SAM 3D Objects pipeline under all three offline flags, then runs the production `generate_mask` path on a deterministic hash-bound blue-circle image with the exact `circle` text prompt. The schema-v3 result records and attests the CUDA bfloat16 image/text inference, binary mask shape/hash/area evidence, GPU/memory evidence, `sam3.pt`, `pipeline.yaml`, every referenced SAM 3D Objects YAML/checkpoint, the cached MoGe model, the DINOv2 weights, and the executable cached DINOv2 Python source tree. Any model-load or real image/text inference exception writes a failing result. Later `--verify-only` preflights reject a missing, malformed, failed, tampered, or stale inference receipt and rehash the bound artifacts. The ordered upstream patch stack rewrites DINOv2's otherwise network-capable `torch.hub` call to the validated local source and weight caches whenever offline mode is active. `remote_jobs/download_sam3d_missing_sqz.sh` invokes the real proof after acquiring the one previously missing MoGe cache object. A hand-written, load-only, status-only, network-assisted, or stale JSON is not acceptable preflight evidence. This lightweight proof exercises real SAM3 image/text segmentation but does not claim a full SAM-3D Objects mesh/texture decode; a direct local-image-to-loadable-GLB smoke belongs in a separate hash-bound proof so it can be run once and verified without paid image generation.

- Before any paid API probe or room generation, run the production `generate_geometry_from_image` entrypoint with `backend=sam3d` and all model hubs offline on the exact committed `tests/test_data/office_shelf.png`; require an atomically published, hash-attested, nonempty loadable GLB plus mask and masked-image evidence, then use only its model-free `--verify-only` path on later benchmark/final acceptance checks.
- Every production SAM3D room worker must enable PyTorch `expandable_segments:True` before importing CUDA and must release the full request-scoped output graph plus unused CUDA allocator cache before and after each mesh decode, including the exception path. This memory guard may not unload the cached models or reduce segmentation, inference steps, texture baking, layout postprocessing, mesh validation, or any other quality setting; a conflicting allocator setting or repeated CUDA OOM is a hard stop before further paid retries.

The mandatory full-generation command is `python scripts/preflight_sam3d_generation.py --repo-dir . --preflight-dir outputs/preflight/<run_name> --sam3-checkpoint external/checkpoints/sam3.pt --pipeline-config external/checkpoints/pipeline.yaml`. It calls no paid API, fixes foreground segmentation at threshold `0.5`, uses the same production SAM3D backend as room workers, and atomically publishes exactly `sam3d_offline_generation/office_shelf_3d.glb`, `office_shelf_mask.png`, `office_shelf_masked.png`, and `receipt.json` with no fifth entry or leaked transaction sibling. The receipt binds the exact committed input, all independently resolved model/config/cache objects, executable generator/manager/CUDA/mesh code, the production room-worker allocator setup and its memory-guard regression, the logical and resolved roots and import origins for the complete regular-file package inventories of external SAM3, external SAM-3D Objects, and installed MoGe (excluding only generated `__pycache__`/`.pyc`/`.pyo`), SAM3's required BPE vocabulary, DINO source/weights, and deterministic package/runtime identity for `torch`, `torchvision`, PyTorch3D, spconv/cumm, Kaolin, xFormers, gsplat, nvdiffrast, Warp, utils3d, trimesh, NumPy, Pillow, SciPy, Hydra, and OmegaConf. That runtime identity hashes each distribution version and installed `RECORD`, its resolved import-origin file, every directly identified core extension (including all Kaolin and Pillow imaging extensions), and all RECORD-listed utils3d package files so loaded shaders/resources are byte-bound. It deliberately does not claim to hash every shared object in large framework/CUDA installations; the named runtime boundary is paired with a real independently loaded GLB. Output evidence includes sizes and SHA-256 values, nonzero trimesh geometry/vertex/face counts, an exact `{0,255}` mask with plausible area/bounds, exact masked-image semantics, and a canonical JSON attestation. A later run must execute the same command with `--verify-only`; that path independently rediscovers model/source/runtime identities, rehashes input/code/outputs, recomputes mask semantics, and reloads the GLB without importing the production generator, allocating a GPU, or running model inference. Benchmark completion and final SQZ acceptance repeat both the schema-v3 SAM3 image/text verification and this full-generation verification. Missing, duplicate, redirected, orphaned, extra, linked, mutated, empty, unloadable, or hand-written artifacts stop the run before `probe_openai`.

The selected SceneSmith `generated_sam3d` path does not import `diffusers` or `peft`: its geometry manager instantiates SAM3, SAM 3D Objects, and MoGe directly, and the source trees contain no such imports. SQZ's complete offline SAM3D load also passes with the currently installed `diffusers==0.38.0` / `peft==0.10.0` pair even though importing the unused top-level `diffusers` package alone reports a PEFT-version error. Do not install or upgrade packages merely to repair that unused import for this contract. Re-audit dependencies if a different generation backend is selected.

ObjectThor is not ready merely because its indexes exist. `scripts/preflight_objathor_retrieval.py` must validate all four prepared files and their exact 50,092-by-768 relationship, category membership, sampled production payloads, and a real cache-only CPU text-embedding call. The retrieval model is `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` at revision `1627032197142fbe2a7cfec626f4ced3ae60d07a`, file `open_clip_pytorch_model.safetensors`, SHA-256 `7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019`. The upstream adapter must accept the publisher's canonical `<uid>.pkl.gz` geometry payload rather than assuming nonexistent GLB files. Production and final `--verify-only` runs are offline and fail if the pinned model blob or payload resolver changes.

The shared AmbientCG materials tree is read-only. Its publisher index may contain entries whose asset directories are absent; do not edit that shared index. Prepare the isolated derivative once with `python scripts/materials_contract.py prepare --data-root data/materials --source-embeddings data/materials/embeddings --contract-embeddings data/materials_full_quality_contract/embeddings --min-retained 1900 --max-pruned 15`, then use only `validate` during production. The manifest content-hashes the source and derivative indexes, records every exclusion, and binds the retained asset path/size inventory. Validation fails if a source/index/asset changes, an excluded asset later appears, more than 15 entries would be pruned, or any retained material lacks a non-empty color texture. Full-quality workers point the materials server at this derivative while continuing to read textures from the untouched `data/materials` tree.

Before a multi-day run, execute `scripts/preflight_vlm_vision.py` against the exact reference image and require its hash-bound `status=pass`. On SQZ, call `agent_proxy` and then `unset ALL_PROXY all_proxy`: SceneSmith's installed `httpx` supports the HTTP reverse proxy but does not include the optional SOCKS transport. Do not install another package merely to work around that; `HTTP_PROXY`/`HTTPS_PROXY` are sufficient. The current `gpt-5.2` image+JSON smoke must score the reference as readable and the orchestrator rejects a missing, failed, or stale result.

On SQZ, Artiverse must be acquired and prepared before generation, in a separate authenticated setup job. The official dataset is manually gated: the user must first receive access approval and then run `hf auth login` interactively on SQZ. Never request or paste the token into a command, log, or chat.

```bash
bash remote_jobs/download_prepare_artiverse_sqz.sh
# After the metadata audit and explicit approval of the large transfer:
ARTIVERSE_APPROVE_FULL_DOWNLOAD=1 bash remote_jobs/download_prepare_artiverse_sqz.sh
```

The setup is deliberately two phase and pins official revision `8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248`. Its first authorized run probes and downloads only the small publisher manifest/unpacker metadata, verifies their pinned Git object IDs, sizes, and audited SHA-256 values, and stops with exit 23 unless the explicit large-transfer switch is present. The exact audit accepted the manifest but rejected the publisher's Python unpack command: its legacy `tarfile.extractall` path does not reject links/special files and is non-atomic, so this pipeline never executes it. The approved second phase downloads only the two pinned archives (approximately 65.3 GB) through standard HTTP with Xet and `hf_transfer` disabled. It uses authenticated Range resume, a 120-second streamed-read timeout for the laptop reverse tunnel, and an outer exact-offset retry loop that never removes partials. It then independently verifies the archives' exact sizes and SHA-256 values and calls `scripts/safe_extract_artiverse.py`. That local extractor accepts only ordinary directories/files under all 3,544 declared roots, rejects traversal, links, special/sparse files and duplicates, checks the exact 531,937-file/86,992,752,890-byte inventory, stages on the same filesystem, and publishes with a hash-bound extraction receipt and rollback. Case-fold ambiguity remains fail-closed except for the official chunk-2 toaster model's two semantically required case-distinct MTL/OBJ pairs: each exception is pinned to the exact manifest, archive SHA-256, model root, two full paths, sizes, and content SHA-256 values, and the extractor requires both members exactly once with no third spelling. Existing data is reused only after a complete no-follow receipt/tree verification that enforces the same pins.

After extraction, the job links `data/artiverse` and prepares the 560 role-focused `armoire`, `bookcase`, and `chest_of_drawers` models rather than spending hours indexing unrelated categories. It converts supported URDF assets to SceneSmith-readable SDF, requires a real revolute/continuous/prismatic joint, enforces the 32-collision-element cap, minimum total/per-category indexed counts, and the failure-rate threshold, then builds `clip_embeddings.npy`, `embedding_index.yaml`, and `metadata_index.yaml`. The source collision cap and movable-joint requirement are checked before conversion, so an expected rejected candidate must leave neither a final nor transaction SDF. Path traversal, links/junctions, special files, escaped resource URIs, or source/copy paths outside their approved roots are fatal.

Artiverse link physics is also fail closed. Every publisher URDF link must contain finite, positive, physically valid inertial data. Preparation preserves the publisher mass and center of mass, rotates a nonzero URDF inertial-frame tensor into the link frame with `Rz(yaw) * Ry(pitch) * Rx(roll)`, passes the resulting exact `LinkPhysics` mapping to SceneSmith, and independently reparses the emitted SDF before atomic publication. These publisher masses are explicitly recorded as the release's unit-mass proxies, not as material-density-derived real masses; `material.json` density is never combined with an invented mesh-volume policy. SceneSmith's injected joint damping/friction (`0.05`) and collision friction (`0.5`) are named as SceneSmith defaults rather than mislabelled as publisher values. The schema-v2 preparation manifest binds the repository/revision, audited publisher metadata, safe-extraction receipt/extractor, all three indexes, every canonical four-part Artiverse ID, publisher URDF path/hash, canonical publisher/emitted physics hashes, source SDF hash, source asset-tree hash, and a sorted aggregate physics binding. `scripts/artiverse_contract.py` independently reparses those URDF/SDF pairs and rejects missing, substituted, malformed, or stale physics evidence. The pipeline-code attestation also covers the exact live `scenesmith/agent_utils/urdf_to_sdf.py` and `scenesmith/utils/inertia_utils.py` bytes. The production pipeline never invokes this downloader; it runs the authority contract and stops if any binding fails.

Artiverse visual resources are likewise fail closed. Before any paid room call, `scripts/preflight_artiverse_visual_resources.py` audits every one of the 500 prepared/indexed candidates and requires every source visual OBJ split to map unambiguously to exactly one in-tree publisher `glbs/<part>.glb`; every triangle primitive in that GLB must carry valid authored `POSITION` and `NORMAL` accessors. The audit independently parses the exact GLB JSON and BIN chunks, proves its single embedded buffer and every buffer view/accessor are in bounds, rejects external buffer/image resources, validates all material/texture/image bindings and embedded image payloads, and precomputes the exact canonical `<part>.gltf` and unpadded `<part>.bin` sizes and SHA-256 values. It revalidates the immutable authority and source-tree identity after each candidate, binds the complete visual/collision/derivation inventories plus the exact runtime normalizer and asset-manager code, writes an attested receipt outside the read-only dataset, and is rerun with `--verify-only` at benchmark completion and final acceptance. At room runtime, normalization occurs only after copying the articulated tree: each publisher GLB is losslessly externalized into the exact atomic sibling directory `scenesmith_artiverse_visuals_v2` with canonical JSON, one external `<part>.bin`, and `_derivation_manifest.json`; equivalent material-split visuals are collapsed to one `./scenesmith_artiverse_visuals_v2/<part>.gltf` visual; and their SDF material overrides are removed so the publisher glTF's complete material assignment survives. Every non-material render property, pose, and scale must match before deduplication. Collision elements, joints, inertials, all other XML, and the publisher source tree remain byte-identical. Direct `.glb` SDF mesh URIs are forbidden because Drake ignores them. The derived directory is atomically published before the copied SDF, its exact inventory and derivation plus copied hashes are stored in object metadata, and a legitimate later rescale reruns the same idempotent normalization before refreshing copied provenance. A missing normal, ambiguous part, malformed JSON/BIN/buffer/material/image binding, external resource, unequal transform/render structure, unsafe path, source mutation, stale receipt, unexpected derived file, or copied provenance mismatch is fatal rather than a reason to retrieve repeatedly or reduce the prompt, model, render, or gate requirements.

## 3. Compute-Node API Proxy Verification

Function: ensure GPU jobs can reach OpenAI for the whole run. A local laptop SSH tunnel bound only to `127.0.0.1` is not enough: compute nodes cannot reach it, and it dies when the laptop disconnects.

Required policy:

- GPU jobs must use a proxy reachable from compute nodes, for example the old cluster-facing login-node endpoint `http://ln08:18092`.
- The proxy/tunnel must be bound on an interface visible to compute nodes, not only login-node localhost.
- The proxy must be managed in a persistent login-node session or service (`tmux`, `screen`, `autossh`, or cluster-managed proxy), not a fragile foreground tunnel from the laptop.
- If the laptop disconnects, the proxy must keep running. Otherwise multi-day runs will stall at the next OpenAI call.
- Every SLURM script must probe OpenAI from inside the GPU allocation and fail fast if unreachable.
- Production room workers must use the OpenAI Responses API for multimodal agent turns. Do not force official Chat Completions: Agents SDK 0.17.4 reduces function-tool results to text on that transport and silently drops every `ToolOutputImage`, leaving the designer visually blind. The Responses request must preserve the mixed image/text `function_call_output` unchanged, and every `data:image/png;base64` payload must decode to real PNG bytes before generation starts.
- Agent request timeouts must preserve the configured provider phase budgets without entering trace serialization. Agents SDK 0.17.4 serializes the full `ModelSettings` dataclass before filtering provider-only fields—even when tracing is disabled—so the live SceneSmith adapter uses a trace-safe `ModelSettings` subclass: the actual request retains its `httpx`/OpenAI timeout object, while `to_traceable_dict()` serializes a copy with `extra_query`, `extra_body`, `extra_headers`, and `extra_args` absent. The code contract binds both `base_stateful_agent.py` and `codex_cli.py`, and the exact provider-preservation/trace regression must pass before a failed floor-plan attempt is resumed.

Preflight from a GPU allocation:

```bash
curl --max-time 60 --proxy "http://ln08:18092" \
  -s -o /dev/null -w "%{http_code}" \
  https://api.openai.com/v1/models
```

Do not launch scene generation until this works from the compute node that will run the job.

## 4. Full-Quality Asset Policy

Function: make the asset router use generated SAM3D assets where appropriate and require both ArtVIP and Artiverse articulated routes.

Full-quality overrides:

```bash
experiment.materials_retrieval_server.data_path=data/materials
experiment.materials_retrieval_server.embeddings_path=data/materials_full_quality_contract/embeddings
furniture_agent.asset_manager.general_asset_source=generated
wall_agent.asset_manager.general_asset_source=generated
ceiling_agent.asset_manager.general_asset_source=generated
manipuland_agent.asset_manager.general_asset_source=objaverse
furniture_agent.asset_manager.backend=sam3d
wall_agent.asset_manager.backend=sam3d
ceiling_agent.asset_manager.backend=sam3d
manipuland_agent.asset_manager.backend=sam3d
manipuland_agent.asset_manager.objaverse.use_top_k=10
manipuland_agent.asset_manager.objaverse.use_lenient_validation=true
furniture_agent.asset_manager.router.strategies.generated.enabled=true
wall_agent.asset_manager.router.strategies.generated.enabled=true
ceiling_agent.asset_manager.router.strategies.generated.enabled=true
manipuland_agent.asset_manager.router.strategies.generated.enabled=true
furniture_agent.asset_manager.articulated.sources.artvip.enabled=true
furniture_agent.asset_manager.articulated.sources.artvip.data_path=data/artvip_sdf
furniture_agent.asset_manager.articulated.sources.artvip.embeddings_path=data/artvip_sdf/embeddings
wall_agent.asset_manager.articulated.sources.artvip.enabled=true
ceiling_agent.asset_manager.articulated.sources.artvip.enabled=true
manipuland_agent.asset_manager.articulated.sources.artvip.enabled=true
++furniture_agent.asset_manager.articulated.sources.artiverse.enabled=true
++furniture_agent.asset_manager.articulated.sources.artiverse.data_path=data/artiverse
++furniture_agent.asset_manager.articulated.sources.artiverse.embeddings_path=data/artiverse/embeddings
++furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true
++furniture_agent.asset_manager.router.strategies.artiverse_articulated.max_retries=3
++furniture_agent.asset_manager.router.strategies.artiverse_articulated.use_lenient_validation=true
furniture_agent.collision_geometry.coacd.max_convex_hull=32
wall_agent.collision_geometry.coacd.max_convex_hull=32
ceiling_agent.collision_geometry.coacd.max_convex_hull=32
manipuland_agent.collision_geometry.coacd.max_convex_hull=32
furniture_agent.collision_geometry.vhacd.max_convex_hulls=32
wall_agent.collision_geometry.vhacd.max_convex_hulls=32
ceiling_agent.collision_geometry.vhacd.max_convex_hulls=32
manipuland_agent.collision_geometry.vhacd.max_convex_hulls=32
```

Important:

- `backend=sam3d` alone is not enough. If `general_asset_source=hssd`, SAM3D is bypassed.
- The leading `++` on the new Artiverse Hydra keys is intentional. The implemented source lives at `furniture_agent.asset_manager.articulated.sources.artiverse`, not at the obsolete `asset_manager.artiverse_articulated` path.
- Artiverse is not optional. If its dataset, converted SDF assets, embeddings/index files, source-filtered retrieval implementation, or articulated service is unavailable, stop the run and install/fix it before generation.
- `artiverse_articulated` must dispatch with `required_source=artiverse`. A result from ArtVIP or PartNet-Mobility cannot satisfy an Artiverse request.
- Set `SCENESMITH_REQUIRE_GATED_FINAL_ASSEMBLY=1` and `experiment.pipeline.final_assembly_policy=external_artiverse_gated` for contract runs. The patched direct SceneSmith entrypoint rejects every post-floor-plan stage under this policy, so the built-in ungated `HouseScene.assemble()` path cannot accidentally publish a contract result.
- Reconstructing the patched checkout must use `upstream-patches/APPLY_ORDER.txt` (or `scripts/apply_upstream_patches.sh`), not lexical filename order. Run the focused upstream `tests/unit/test_artiverse_retrieval_integration.py` before router validation so the real source-filtered dispatch is proven in the applied checkout.
- The scene prompt must include at least one appropriate articulated furniture requirement so Artiverse can be exercised legitimately; do not insert an irrelevant token asset merely to satisfy the count.
- Before launching room workers, grep `scripts/run_single_room_worker.py` for any forced `general_asset_source` override and run the worker with `--config-only`; the resolved Hydra config printed in the log must show furniture/wall/ceiling as `generated` and manipulands as `objaverse` for `--asset-pipeline generated_sam3d`.

Config-only room-worker check:

```bash
.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir <scenesmith_repo> \
  --run-dir "outputs/<date>/<run_name>" \
  --csv <prompt_csv> \
  --run-name <run_name>_room_worker_config_check \
  --room-id <room_id> \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --port-offset <unique_offset> \
  --render-gpu-id 0 \
  --config-only
```

Expected resolved policy:

```json
{
  "furniture_agent": {"general_asset_source": "generated", "backend": "sam3d"},
  "wall_agent": {"general_asset_source": "generated", "backend": "sam3d"},
  "ceiling_agent": {"general_asset_source": "generated", "backend": "sam3d"},
  "manipuland_agent": {"general_asset_source": "objaverse"},
  "articulated_contract": {
    "articulated_strategy_enabled": true,
    "artiverse_strategy_enabled": true,
    "artvip_enabled_agents": {
      "furniture_agent": true,
      "wall_agent": true,
      "ceiling_agent": true,
      "manipuland_agent": true
    },
    "artvip": {
      "enabled": true,
      "data_path_exists": true,
      "embeddings_path_exists": true,
      "missing_embedding_files": []
    },
    "artiverse": {
      "enabled": true,
      "data_path_exists": true,
      "embeddings_path_exists": true,
      "missing_embedding_files": []
    }
  }
}
```

The command exits nonzero unless both articulated sources, their dataset directories, and all required indexes are real and the Artiverse strategy is enabled. On a single 24 GB GPU, set `SCENESMITH_RETRIEVAL_DEVICE=cpu` so retrieval embeddings do not compete with the loaded SAM3D pipeline for VRAM.

## 5. Articulated Router Validation

Function: prove that both ArtVIP and Artiverse paths change actual router behavior before spending GPU/API budget on a full run.

Required command before a full-quality run:

```bash
.venv/bin/python scripts/validate_articulated_router.py \
  --output outputs/<date>/<run_name>/articulated_router_validation.json \
  --repo-dir <scenesmith_repo> \
  --artiverse-data data/artiverse \
  --artiverse-embeddings data/artiverse/embeddings \
  --artvip-data data/artvip_sdf \
  --artvip-embeddings data/artvip_sdf/embeddings \
  --vlm-backend openai \
  --top-k 3
```

On current SQZ, the orchestrator runs this on its GPU pod. On ParaCloud, run it under SLURM rather than on a login node:

```bash
sbatch remote_jobs/TEMPLATE_validate_articulated_router.sbatch
```

This script fails unless:

- Every supplied ArtVIP and Artiverse dataset/index path exists and contains the required embedding, embedding-index, and metadata-index files.
- AssetRouter selects the generic `articulated` strategy for openable-furniture prompts and retrieves an ArtVIP candidate where required.
- The articulated retrieval indexes return concrete SDF candidates with existing paths and exact source identifiers.
- The `artiverse_articulated` strategy is selected for at least one Artiverse-eligible validation prompt.
- At least one returned Artiverse candidate has an existing SDF path under `data/artiverse`, has `source=artiverse`, and records Artiverse provenance.

If this fails:

- Do not launch the full run.
- Inspect/fix the asset router analysis prompt or strategy parsing.
- Do not assume config flags are enough.

Before final assembly, additionally verify that at least one placed Artiverse asset remains in a passing room state. Final assembly must fail if the Artiverse usage count is zero, even when router validation passed earlier.

## 6. Floor-Plan-Only Generation And Layout Gate

Function: create the building layout first, without loading room-generation services, then fail closed unless the layout matches the prompt/reference contract.

Generic/non-reference SceneSmith command:

```bash
.venv/bin/python main.py \
  +name=<run_name> \
  experiment.csv_path=<prompt_csv> \
  experiment.num_workers=1 \
  experiment.pipeline.start_stage=floor_plan \
  experiment.pipeline.stop_stage=floor_plan \
  experiment.pipeline.parallel_rooms=false \
  floor_plan_agent.mode=house \
  ++codex.enabled=true \
  ++codex.cwd=<scenesmith_repo> \
  ++codex.timeout_seconds=1800 \
  hydra.run.dir="outputs/<date>/<run_name>"
```

The generic LLM placement search is not the production source for the exact
reference-school geometry. Its fixed-order, timeout-bounded solver cannot reliably
satisfy the immutable classroom rows, restroom foyer, storage slot, and centered
library simultaneously. For this one hash-bound profile, materialize the exact layout
with SceneSmith's native `RoomSpec`, `PlacedRoom`, `Wall`, `Door`, `Window`, and
`RoomGeometry` implementations:

```bash
.venv/bin/python scripts/seed_reference_school_layout.py \
  --repo-dir <scenesmith_repo> \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --config outputs/<date>/<run_name>/resolved_config.yaml \
  --prompt inputs/full_quality_school_reference_20260710/prompt.txt \
  --expected-prompt-sha256 ac8d297cc9a2d605f41b4bcd7abd52aac29bfd0f195875840342ee1e6a7da86f \
  --profile school_reference_20260710
```

This is not a preview, fake cube path, or hand-written acceptance JSON. The
materializer creates four genuine SceneSmith walls per room, applies every opening
through the real floor-plan tool methods, calls the real common-zone tool after its
backing doors exist, generates all structural SDF/GLTF assets, writes the Drake
directive, and exports the normal Blender floor plan. It builds on the same filesystem
in a unique transaction run placed beside the final run at exactly the same directory
depth, so SceneSmith's relative GLTF material URIs remain valid after publication. It
resolves and hashes every local GLTF buffer/image dependency both before and after the
atomic move, reparses every generated SDF through `HouseLayout.from_dict`, runs the
independent project layout validator, and publishes `house_layout.json` last. It
refuses an unproven existing layout. A later invocation rehashes the native geometry,
re-resolves all GLTF dependencies, and verifies a structural-layout digest that excludes
only the two prompt fields intentionally rewritten by `school_room_contract.py
bind-layout`. The floor-plan step therefore uses no paid API call; all later room
generation and VLM gates remain unchanged.

For the reference school run, enforce the exact 11-room contract immediately afterward:

```bash
.venv/bin/python scripts/validate_school_floor_layout.py \
  --layout outputs/<date>/<run_name>/scene_000/house_layout.json \
  --output outputs/<date>/<run_name>/scene_000/quality_gates/floor_plan_layout.json
```

The gate requires `classroom_01` through `classroom_06`, `library`, `boys_toilet`, `girls_toilet`, `storage_room`, and `main_corridor`; their required relative positions and area ranges; zero positive-area AABB overlaps; adjacent toilets; a horizontally centered library immediately south of the circulation spine; valid connectivity; exterior windows; and a centered south-facing exterior entrance at least 1.6 m wide with exactly two leaves. A single wide leaf does not satisfy the reference. It computes these facts itself rather than trusting the upstream booleans alone. A missing, renamed, extra, overlapping, disconnected, or incorrectly positioned required room stops the pipeline before any room generation. Repair/regenerate the floor plan and rerun this gate.

The reference restroom pair needs a real shared public foyer without adding a twelfth room. The floor-plan agent must first create a real `girls_toilet`–`main_corridor` Door/OPEN and a real `boys_toilet`–`girls_toilet` Door/OPEN inside the proposed foyer strip, then call `set_navigation_common_zones` last. Its `foyer_corridor` record must name the actual corridor door through `backing_door_id`; the tool derives width, position, and orientation from the generated wall openings. Re-placement clears this annotation. The layout gate independently requires paired finite wall openings, co-located centers, compatible widths, safe boundary-strip carves, physical connectivity across all carved pieces, and a real backed threshold to library/main corridor. Guessed metadata cannot create or authorize circulation geometry.

Output:

- `scene_000/house_layout.json`
- `scene_000/floor_plans/`
- `scene_000/room_geometry/`
- `scene_000/room_<room_id>/`
- `scene_000/quality_gates/reference_school_layout_seed.json`

## 7. Room-Level Generation

Function: generate furniture, wall objects, ceiling objects, and manipulands for each room only after the floor-plan layout gate passes.

Preferred full-quality command for a single room worker:

```bash
.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir <scenesmith_repo> \
  --run-dir "outputs/<date>/<run_name>" \
  --csv <prompt_csv> \
  --run-name <run_name>_room_worker \
  --room-id <room_id> \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --port-offset <unique_offset> \
  --render-gpu-id 0
```

Current SQZ policy:

- Run the 11 required rooms sequentially on the single A10, in a fixed list, and stop at the first failed room gate so only that room is repaired.
- Give every room a unique `--port-offset`, even though execution is sequential.
- Each worker writes only to its own `scene_000/room_<room_id>/` folder. Before continuing an interrupted room, `scripts/select_room_resume_stage.py` validates a contiguous chain of regular, unlinked, parseable checkpoints, requires every checkpoint's `text_description` to equal the immutable hash-bound room prompt, and selects only the highest committed stage. A completed room may bypass refresh/render/VLM only when `room_visual_self_exam.py --summarize-existing --rooms <id>` rehashes and revalidates its entire passing evidence graph. Otherwise `run_single_room_worker.py --refresh-final-blend` atomically reexports `scene.blend` from the current final state; a stale blend may never gate a newer JSON state.
- Do not run final assembly while any room worker is active.
- On a future multi-GPU allocation, one independent room worker per GPU is allowed, but a single serial room worker does not benefit from multiple GPUs. Preserve isolated room output folders and unique ports.

For the reference-school contract, generation must first replace every planner-shortened room prompt with a canonical room-specific prompt bound to the validated immutable effective-prompt hash:

```bash
.venv/bin/python scripts/school_room_contract.py bind-layout \
  --layout outputs/<date>/<run_name>/scene_000/house_layout.json \
  --input-manifest inputs/full_quality_school_reference_20260710/input_manifest.json \
  --output outputs/<date>/<run_name>/scene_000/quality_gates/room_prompt_binding.json
```

The canonical prompts require all six classrooms to contain exactly 12 student desks and exactly 12 corresponding student chairs plus the detailed classroom contents; they also define the required inventories for the library, toilets, storage room, and corridor. Both the unique `RoomSpec` and unique `PlacedRoom` for every one of the 11 room IDs must contain identical canonical prompt bytes bound to the immutable effective prompt; duplicate or ambiguous room records are fatal. A worker may use the planner's text only as secondary context. It may not replace or shorten these canonical requirements.

## 8. Room Render Capture

Function: render each generated room so SAGE/Codex/VLM can examine object placement.

Minimum review images:

- Top-down room image.
- At least two side/oblique views for furniture orientation and wall/door blockage.
- Optional collision/debug overlay if available.

Current helper:

```bash
.venv/bin/python scripts/render_room_review_views.py \
  --blend outputs/<date>/<run_name>/scene_000/room_<room_id>/scene_states/final_scene/scene.blend \
  --room-id <room_id> \
  --output-dir outputs/<date>/<run_name>/scene_000/review/room_review_renders
```

The standalone helper computes the actual room-mesh bounds and must write one top view plus two oblique views. It hides every classified ceiling/roof for all views, preserves the outline in the top view, and hides only geometrically proven camera-side envelope walls in each oblique. It fails on an indivisible/unclassifiable shell or a view whose cutaway cannot be established, restores the original visibility state, and writes `<room_id>_cutaway_evidence.json` with source-blend/image hashes and the exact hidden/visible geometry. The next section independently validates that proof before calling the VLM.

## 9. Room Self-Exam Gate

Function: block bad rooms before final assembly/export.

Run the deterministic gate first for the room currently being processed:

```bash
.venv/bin/python scripts/room_self_exam.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --review-dir outputs/<date>/<run_name>/scene_000/review/room_review_renders \
  --output-dir outputs/<date>/<run_name>/scene_000/quality_gates/room_self_exam_deterministic \
  --rooms <room_id> \
  --max-collision-hulls 32 \
  --contract-profile school_reference_20260710
```

Then run the compulsory image-aware self-exam using the deterministic result, generated room prompt, all three review views, and the user's reference image:

```bash
.venv/bin/python scripts/room_visual_self_exam.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --deterministic-gate-dir outputs/<date>/<run_name>/scene_000/quality_gates/room_self_exam_deterministic \
  --review-dir outputs/<date>/<run_name>/scene_000/review/room_review_renders \
  --reference-image <reference_image> \
  --output-dir outputs/<date>/<run_name>/scene_000/quality_gates/room_self_exam \
  --rooms <room_id> \
  --minimum-review-images 3 \
  --threshold 7 \
  --contract-profile school_reference_20260710 \
  --effective-prompt inputs/full_quality_school_reference_20260710/prompt.txt \
  --input-manifest inputs/full_quality_school_reference_20260710/input_manifest.json \
  --prompt-binding outputs/<date>/<run_name>/scene_000/quality_gates/room_prompt_binding.json \
  --vlm-backend openai
```

Implemented fail-closed behavior:

- The deterministic gate blocks missing final room states, missing review images, objects outside room-local bounds, suspicious object density, collision-hull risk, a missing canonical prompt marker, and every missing/over-counted school inventory requirement. In particular, each classroom must resolve to exactly 12 student desks and exactly 12 student chairs, with 12 finite one-to-one spatial desk/chair pairings at plausible distances. Each explicitly requested small-item category is independently required. A unique bipartite assignment prevents one generically named object from satisfying several checklist entries.
- The visual gate cannot override a missing, malformed, or failed deterministic gate and does not call the VLM until deterministic evidence passes.
- The visual gate requires the canonical room-specific checklist, the complete immutable effective prompt, the user's supplied appearance reference image, and at least three distinct room views. Its response must include one explicit pass/fail evidence record for every inventory item, reference-style/material/daylight/circulation/collision requirement, room identity, and applicable articulated role, with cited generated view indices and specific observations. Aggregate prose or a generated shortened room prompt is never authoritative.
- Every passing visual gate binds schema-versioned SHA-256 evidence for the canonical final room state, source `scene.blend`, exact three cutaway review images, cutaway proof JSON, `house_layout.json`, reference image, immutable prompt, validated input manifest, and room-prompt binding; it rehashes them after the VLM call to catch mid-review changes. Assembly independently reparses the cutaway proof and all of its internal source/image bindings rather than trusting its status field.
- VLM transport failures, refusals, malformed JSON, missing scores, or scores outside the accepted schema fail the room.
- Final scores are the minimum of deterministic and visual scores, so visual judgment cannot raise a deterministic score.

Required behavior:

- Input: room review images plus structured room state.
- Output: one JSON result per room.
- Pass/fail must be explicit.
- Failed rooms must not be assembled into final output.

Required JSON shape:

```json
{
  "room_id": "classroom_01",
  "status": "pass",
  "scores": {
    "object_relevance": 8,
    "placement_realism": 8,
    "clearance_and_access": 8,
    "collision_risk": 9,
    "prompt_alignment": 8
  },
  "critical_issues": [],
  "repair_instructions": []
}
```

Failure examples:

- Desks/chairs outside room bounds.
- Furniture floating, intersecting walls, or stacked unnaturally.
- Doorways or corridors blocked.
- Objects bunched in one strip while most room is empty.
- Classroom rows facing the wrong direction or unusable.
- Restroom, closet, office, library, or corridor objects nonsensical for the room type.

Pass threshold:

- No critical issues.
- Every combined score is at least 7: `object_relevance`, `placement_realism`, `clearance_and_access`, `collision_risk`, and `prompt_alignment`.
- The deterministic gate status remains `pass`.

Run the library, storage room, and `classroom_01` first so all three compulsory articulated roles fail early instead of after days of unrelated room work; then continue the remaining rooms sequentially. Run render -> deterministic gate -> visual gate immediately after each room. If any required room fails, stop at that room and enter the repair loop. After all 11 rooms pass individually, rerun the deterministic gate over the complete room set and call the visual gate with `--summarize-existing`. That mode revalidates the exact 11 regular per-room JSON files, itemized checklists, canonical paths, thresholds, and all bound hashes, then writes the required all-room summary without repeating 11 VLM evaluations or rewriting a room decision. Then run `scripts/classroom_variation_gate.py` over the six final classroom states and their top views. It hash-binds all inputs, rejects identical semantic/seating/decoration fingerprints deterministically, and requires a schema-valid VLM comparison score of at least 7 with six specific classroom identities/features and pairwise distinctions. On a rerun, its existing decision may bypass the VLM only after `--verify-only` revalidates the complete attestation and every bound hash. Do not combine/export on missing, failed, or stale evidence.

## 10. Room Repair/Regeneration Loop

Function: fix failed rooms only, preserving good rooms.

Preferred order:

1. If object placement is bad but asset choices are okay, rerun from the relevant placement stage.
2. If asset choices are wrong, rerun from furniture generation for that room.
3. If geometry/floor plan is wrong, stop and reconsider the floor plan rather than patching final export.

Use the same full-quality asset policy:

```bash
--asset-pipeline generated_sam3d
```

After each repair:

```text
rerender top + two oblique views -> rerun deterministic gate -> rerun reference-aware visual gate -> only mark pass if both JSON results pass
```

After all room and classroom-variation gates pass, exercise the three required articulated assets before assembly:

```bash
.venv/bin/python scripts/validate_articulated_motion.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --output outputs/<date>/<run_name>/scene_000/quality_gates/articulated_motion.json
```

This gate requires exactly one library glass-door bookcase role, one two-door storage utility-cabinet role, and one teacher filing-drawer role, with at least one role carrying validated Artiverse authority. It safely exercises every relevant revolute or prismatic joint at interior positions through isolated `pydrake` plants and requires a measured child-body transform change, not merely a joint declaration. It hash-binds each room state and complete SDF tree and rejects unsafe resource paths and mutations during loading. Before acceptance, repeat the real exercises with `--verify-only --verification-output <scene>/quality_gates/articulated_motion_verification.json`; that immutable receipt must bind the original gate and the freshly recomputed Drake-motion result.

## 11. Artiverse-Provenance-Gated Final Assembly

Function: merge all passed room outputs into one `combined_house`.

Command:

```bash
.venv/bin/python scripts/assemble_final_house_and_render.py \
  --repo-dir <scenesmith_repo> \
  --run-dir "outputs/<date>/<run_name>" \
  --csv <prompt_csv> \
  --run-name <run_name>_final_assemble \
  --gate-dir "outputs/<date>/<run_name>/scene_000/quality_gates/room_self_exam" \
  --artiverse-data data/artiverse \
  --contract-profile school_reference_20260710 \
  --input-dir inputs/full_quality_school_reference_20260710 \
  --render
```

Guardrail:

- Before this command, verify that every expected `room_<room_id>` has a passing gate JSON.
- Assemble into a unique hidden staging directory. Keep the existing `combined_house` untouched until the candidate passes export, render, room-gate, Artiverse-lineage, and post-merge survival validation; publish only the passing candidate and restore the previous directory if post-promotion verification fails.
- `scripts/assemble_final_house_and_render.py` refuses to run if any room gate JSON is missing or non-passing, unless `--allow-ungated` is explicitly passed for debugging.
- Before staging/assembly, the script scans all passing final room states and requires at least one exact `asset_source=articulated`, `articulated_source=artiverse`, `is_articulated=true` record. Its canonical ID must exist in the hash-bound prepared metadata/embedding indexes; its source path/hash/tree must match that indexed asset; and its copied room SDF/tree hashes must match the provenance written by the patched asset manager. The patched rescale helper refreshes only the copied-content bindings after a legitimate in-room scale operation while preserving the immutable source binding. A hand-written `artiverse` label, unknown ID, cross-source path, stale rescale evidence, or swapped file is rejected.
- Before trusting a passing gate, assembly recomputes every bound SHA-256 and rejects legacy/malformed manifests, substituted paths, missing or duplicate review views, mixed references, or any state/layout/image changed since review. Stale pass JSON can never authorize assembly.
- The reference-school assembly contract independently requires all three functional articulated roles: a library bookcase cabinet with hinged glass doors, a school-supply utility cabinet with two hinged doors, and a teacher filing cabinet with operable drawers. Each role must have real runtime articulated provenance, a nonempty canonical ID and SDF, and ArtVIP or Artiverse authority; semantic labels alone are rejected. At least one of these three role records must be the separately authority-validated compulsory Artiverse asset, and all three identities/sources must survive final assembly.
- After assembly, it scans the candidate `house_state.json`, requires that the same compulsory Artiverse usage survives the merge, renders and verifies all three nonempty overviews, and writes schema-v2 `artiverse_usage.json`. That manifest binds the official authority, every final room state, combined house state, source SDF/tree, and copied SDF/tree by SHA-256. It is revalidated after publication and again immediately before SQZ may write `awaiting_2gpu_acceptance`. Zero usage, invalid/forged provenance, mutation, a missing source file, or loss during assembly is fatal.
- `--allow-ungated` is for debugging only and does not waive the compulsory Artiverse provenance/survival requirement for a contract-compliant final run.

## 12. Whole-Floor Reference Gate

Function: compare the assembled floor, exact room arrangement, visual finish, access, and simulation readiness against the supplied reference before simulator export is accepted.

Required command after gated assembly has rendered the overviews:

```bash
.venv/bin/python scripts/whole_floor_reference_gate.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --reference-image <reference_image> \
  --output outputs/<date>/<run_name>/scene_000/quality_gates/whole_floor_reference.json \
  --threshold 7 \
  --vlm-backend openai
```

This gate fails closed unless:

- The layout contains exactly the 11 required IDs in the exact reference-relative wing order.
- `combined_house/outlook_renders/overview_top.png`, `overview_isometric.png`, and `overview_front.png` all exist.
- The gate SHA-256 binds `house_layout.json`, the exact supplied reference image, all three overview renders, `combined_house/house_state.json`, and `combined_house/artiverse_usage.json`; it rehashes them after the VLM returns. Missing, substituted, or concurrently changed evidence fails closed.
- There are no critical issues and all seven scores are at least 7: `room_count_and_identity`, `room_arrangement`, `warm_visual_style`, `circulation_and_access`, `furnishing_completeness`, `simulation_readiness`, and `reference_similarity`.
- The VLM call and response schema succeed. Transport, refusal, parsing, missing-evidence, or malformed-score errors fail the gate.

A failure returns repair instructions. Repair the affected floor plan, room, furnishing, material, render, or assembly stage and repeat all downstream gates; do not declare or transfer a final result from a failed whole-floor gate. Immediately before a completion receipt, rerun the same command with `--verify-only` to rehash the saved decision against the canonical published files without issuing another VLM call.

After the visual reference gate, prove usable humanoid circulation from the exterior entrance to every required room:

```bash
.venv/bin/python scripts/validate_school_navigation.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --output outputs/<date>/<run_name>/scene_000/quality_gates/school_navigation.json
```

This deterministic gate independently reconstructs matching door portals from layout geometry, requires the >=1.6 m two-leaf library-south entrance and >=0.9 m interior thresholds, transforms and rotates final object AABBs into house coordinates, inflates them by a 0.35 m humanoid radius, and searches a 0.15 m grid. Every target route must use only the entrance/library/corridor/common circulation plus the target room; it may not cheat through another classroom, restroom, or storage room. The library, corridor, and six classrooms also need 1.5 m turning witnesses. If the restrooms are reached through a shared foyer, that foyer must be physically explicit in `navigation_common_zones` with bounds and threshold geometry; a text label or a route through the other restroom is rejected. The gate binds the layout, combined state, all 11 final room states, portals, obstacles, route cells, and waypoints. Before acceptance, run `--verify-only --verification-output <scene>/quality_gates/school_navigation_verification.json`; the saved repeat receipt must contain the freshly recomputed full navigation result and bind the original gate hash.

## 13. Drake/SceneEval/Collision Validation

Function: create simulator-ready scene files.

Expected outputs:

- `combined_house/house_state.json`
- `combined_house/sceneeval_state.json`
- `combined_house/house.dmd.yaml`
- `combined_house/house.blend`
- `room_geometry/*.sdf`
- collision meshes under generated asset folders

Validation:

```bash
.venv/bin/python scripts/validate_drake_scene.py \
  --dmd outputs/<date>/<run_name>/scene_000/combined_house/house.dmd.yaml \
  --package-root outputs/<date>/<run_name>/scene_000 \
  --require-gpus 0 \
  --max-collision-elements 32 \
  --minimum-models 12 \
  --expected-rooms 11 \
  --output outputs/<date>/<run_name>/scene_000/quality_gates/drake_load.json
```

The validator independently parses and hashes a nonempty, duplicate-free DMD `add_model` inventory and the combined house state, requires exactly 11 room identities and at least 12 models, then loads model directives through `pydrake` with `package://scene` mapped to the supplied package root. It finalizes the multibody plant, builds the diagram, requires the Drake model count to match the directives and more than one body, records model/body/joint/position/velocity counts, reports visible GPUs, and reports generated-SDF collision complexity including assets over 32 collision elements.

Full-quality acceptance:

- A local or SQZ load is a useful structural precheck, but does not satisfy the final hardware acceptance when fewer than two GPUs are visible.
- Copy or mount the exact hash-verified final package on a 2-GPU ParaCloud allocation and rerun `scripts/validate_drake_scene.py` there with `--require-gpus 2 --max-collision-elements 32`. The report must show `status=pass`, `visible_gpu_count >= 2`, `two_gpu_acceptance_environment=true`, no malformed/over-cap SDFs, and the full-house load must complete without OOM.
- If full-house Drake load OOMs, the export is not simulation-ready even if files exist.
- Collision mesh counts must be reported, including max collision files for a single object.

If Drake package assets are exported to a standalone folder, include `floor_plans/`, `room_geometry/`, and all `room_*/generated_assets/` dependencies.

## 14. Isaac/USD/MuJoCo Export

Function: export to additional simulation formats when requested.

MuJoCo/USD uses the isolated `.mujoco_venv` because Blender's bundled `pxr` must not be mixed with the OpenUSD libraries used by `mujoco-usd-converter`. The SQZ preflight must import `mujoco`, `mujoco_usd_converter`, `usdex.core`, and `pxr.Usd` from this environment before generation. Export and validate both formats through the atomic wrapper:

```bash
.mujoco_venv/bin/python scripts/export_simulator_artifacts_atomic.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --published-dir outputs/<date>/<run_name>/scene_000/mujoco_export \
  --validation-output outputs/<date>/<run_name>/scene_000/quality_gates/simulator_exports.json \
  --exporter scripts/export_scene_to_mujoco.py \
  --run-attempt-id "$RUN_ATTEMPT_ID" \
  --require-usd
```

The wrapper exports to a unique attempt-bound hidden staging directory, requires the exporter to return nonzero on any USD-layer failure, and validates before atomic publication with rollback. The schema-v2 validator loads and steps the exact MJCF with MuJoCo, resolves every reference, opens every expected USD layer with OpenUSD, requires zero exporter failures, rejects symlinks/path escapes/external references/empty outputs, proves the exporter behavior from its exact source, and records a SHA-256 inventory. At receipt time, `final_acceptance_bundle.py` independently rewalks the live `mujoco_export` tree, matches the exact file set/sizes/hashes and current attempt marker, reparses MJCF references, and reopens every report-bound USD layer. Existing, added, removed, mutated, partial, linked, or status-only exports do not count.

Isaac Sim note:

- Isaac Sim consumes USD best. Drake DMD/SDF is not automatically an Isaac-native export.
- If Isaac is required, run the USD path and then run the Isaac compatibility fixer if needed.

## 15. Outlook Renders And Local Transfer

Function: generate user-facing preview images and copy the final package locally.

Expected renders:

- `overview_top.png`
- `overview_isometric.png`
- `overview_front.png`
- per-room review images
- contact sheet

Transfer rule:

- Use hash-verified archives/chunks for large folders.
- Do not report final success until local files are present and at least one render is visually inspected.

## 16. SQZ Acceptance Bundle And 2-GPU Handoff

Function: make the exact single-GPU-approved scene immutable and carry its Artiverse and quality evidence into the required two-GPU acceptance without a self-hash loophole.

After every final `--verify-only` gate passes, create and immediately verify the internal record:

```bash
.venv/bin/python scripts/final_acceptance_bundle.py \
  --repo-dir <scenesmith_repo> \
  --run-dir outputs/<date>/<run_name> \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --input-dir inputs/full_quality_school_reference_20260710 \
  --run-attempt-id "$RUN_ATTEMPT_ID" \
  --output outputs/<date>/<run_name>/scene_000/combined_house/sqz_acceptance_record.json

.venv/bin/python scripts/final_acceptance_bundle.py \
  --repo-dir <scenesmith_repo> \
  --run-dir outputs/<date>/<run_name> \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --input-dir inputs/full_quality_school_reference_20260710 \
  --run-attempt-id "$RUN_ATTEMPT_ID" \
  --output outputs/<date>/<run_name>/scene_000/combined_house/sqz_acceptance_record.json \
  --verify-only
```

The bundle copies the exact immutable inputs and outside-scene preflight evidence into `quality_gates/final_acceptance_evidence`, semantically revalidates every required verdict, and records scene-relative hashes. It requires all 11 deterministic/visual/cutaway sets, classroom variation, articulated motion plus its saved repeat-execution proof, navigation plus its saved recomputation proof, whole-floor reference, Artiverse preparation/router/usage/final survival, the SAM3D schema-v3 load/inference proof, the separate full-generation GLB/mask/masked-image proof, ObjectThor, materials, code attestation, Drake, SAGE, and the independently rehashed live simulator export tree. It rejects added/removed/mutated evidence, symlinks/special files, and leaked assembly/simulator/acceptance staging or backup directories. Its status is deliberately `awaiting_2gpu_acceptance`.

Next create an external exact-file-set manifest and the outside-scene SQZ completion receipt:

```bash
.venv/bin/python scripts/two_gpu_drake_acceptance_contract.py create-manifest \
  --package-root outputs/<date>/<run_name>/scene_000 \
  --manifest outputs/<date>/<run_name>/scene_000.sha256 \
  --expected-run-attempt-id "$RUN_ATTEMPT_ID" \
  --output outputs/<date>/<run_name>/package_manifest_validation.json

.venv/bin/python scripts/two_gpu_drake_acceptance_contract.py create-sqz-completion \
  --package-validation outputs/<date>/<run_name>/package_manifest_validation.json \
  --expected-run-attempt-id "$RUN_ATTEMPT_ID" \
  --output outputs/<date>/<run_name>/pipeline_completion.json
```

The manifest must remain outside `scene_000`; it covers the internal acceptance record and every other regular package file exactly once. Preserve the manifest SHA-256, completion-receipt SHA-256, and `run_attempt_id` out of band when transferring the scene. `remote_jobs/TEMPLATE_2gpu_drake_acceptance.sbatch` requires all three values, rehashes the package before and after the full Drake load, semantically binds the internal SQZ record and external pending receipt, and produces the terminal `two_gpu_drake_acceptance_receipt.json`. Before sourcing the compute-node environment it must also compare the executing template, environment helper, acceptance contract, Drake validator, and pipeline-contract helper against the exact artifact hashes in the packaged `pipeline_code_contract.json`. A two-GPU pass requires a real independent CUDA allocation, reduction, result readback, and synchronization on devices 0 and 1; merely reporting two visible device names is not acceptance evidence. The terminal receipt binds the runtime validation and both per-device exercises. SQZ's `pipeline_completion.json` remains pending; it is never rewritten to pretend a one-GPU run was final.

## 17. Final Validation Report

Function: make it clear what was actually run.

The final report must include:

- Run path.
- Exact asset policy used.
- Whether SAM3D was enabled and actually reachable.
- Whether HSSD-only was used anywhere.
- Whether ArtVIP and Artiverse routes were enabled.
- Artiverse dataset path, router validation evidence, selected asset identifiers, placed-asset count, and final surviving asset count. The final surviving count must be at least one.
- Number of rooms generated.
- Immutable prompt/input-manifest/room-prompt-binding hashes and the exact semantic inventory result for every room.
- Floor-plan layout-gate output path and pass/fail status.
- ParaCloud runtime-code validation hash and the passing synchronized CUDA exercise record for GPU 0 and GPU 1.
- Number of rooms passed both the deterministic and itemized reference-aware visual gates, with both gate directories recorded, plus the six-classroom variation verdict.
- Failed rooms and repair history.
- Whole-floor reference-gate output path, seven scores, and pass/fail status.
- Final export files.
- SQZ/local Drake structural-load result and collision-complexity report.
- MuJoCo/USD validation status and hash-bound export inventory; raw export files alone are insufficient.
- Articulated-motion verdict for the library bookcase doors, storage cabinet doors, and teacher filing drawers, including before/after joint positions and kinematic pose changes.
- Whole-house navigation/free-space verdict proving that required interior room transitions and the exterior entrance are traversable without using visual-only labels as evidence.
- Collision hull caps used.
- Full-house Drake load result on a 2-GPU ParaCloud allocation, including `visible_gpu_count` and `two_gpu_acceptance_environment`; a single-GPU result must be labeled precheck-only.
- Articulated router validation output path and pass/fail status.
- `combined_house/artiverse_usage.json`, including the pinned official dataset authority, index and source hashes, pre-assembly placed count, final surviving count, identifiers, room/house-state hashes, and source/copied SDF/tree hashes. Both counts must be at least one and survival must match. Also include `quality_gates/artiverse_final_validation.json`, the internal `combined_house/sqz_acceptance_record.json`, the external exact-file package manifest, and the outside-scene `pipeline_completion.json` that binds both for the current attempt.
- Worker `--config-only` resolved asset policy output proving no forced HSSD path.
- Compute-node OpenAI proxy endpoint and probe result.
- Whether manipulands used ObjectThor instead of primitive fallback, including the exact pinned OpenCLIP model/revision/blob hash, 50,092-row index result, sampled production-payload evidence, and final cache-only verification.

## Cost And Schedule Estimate

These are planning estimates, not guarantees. Actual cost depends on object count, SAM3D checkpoint load time, OpenAI usage limits, image generation retries, and repair loops.

Every option described as full-quality, including any generated/retrieval/articulated asset mix, still requires the compulsory Artiverse dataset, source-filtered route, legitimate placement, final surviving provenance, and all gates above. The fast/HSSD option is explicitly outside this full-quality acceptance contract.

Historical Run 3 scale reference:

- Around 18 rooms.
- Around 770 placed objects.
- HSSD-heavy generation still hit OpenAI usage pressure.

Current 11-room reference-school generated-SAM3D plan on SQZ's single A10:

- **Do not use the former 120-300 GPU-hour, 11-27-hour-per-room, or 7-19-day figures.** They were arithmetic planning guesses, not measurements from this school pipeline, and are withdrawn.
- No room from the current reference-school run has completed the real generated-SAM3D -> three-view render -> deterministic gate -> itemized visual gate path yet, so an evidence-backed total runtime is currently unknown.
- The available older evidence is not a school-room benchmark: one generated-SAM3D living-room chain took 4 h 13 min 27 s, while an older kitchen reached floor/furniture work after 4 h 55 min and remained incomplete after more than 9 h. Different prompts, checkpoints, stage coverage, retries, and gates make those observations unsuitable for multiplying by 11.
- Before projecting the full run, execute one representative dense room (`classroom_01`) through the exact production path, record per-stage start/end timestamps, OpenAI request/usage evidence, GPU utilization, and peak VRAM, and inspect the resulting gate evidence. The benchmark may populate the canonical run so passing artifacts are reused rather than discarded.
- The exact SQZ benchmark command is:

  ```bash
  SCENESMITH_EXECUTION_MODE=benchmark_classroom_01 \
    bash remote_jobs/run_full_quality_school_sqz.sh
  ```

  This is an explicit mode of the production runner, not a preview wrapper. It performs the same code/input/model/asset/router preflights, deterministic native SceneSmith reference-layout materialization/reverification and layout gate, generated-SAM3D `classroom_01` worker, atomic blend refresh, three cutaway renders, deterministic gate, paid itemized reference-aware visual gate, and saved-gate revalidation. It then writes `classroom_01_full_quality_benchmark.json` plus timestamp and `nvidia-smi` sample logs directly under the canonical run directory and exits with status `benchmark_complete` before all-room summaries, variation, articulation, assembly, export, or acceptance. The receipt is hash-bound timing evidence and explicitly is not whole-school acceptance. A later normal runner invocation reuses the room only if its complete evidence graph still verifies byte-for-byte.
- Derive the post-benchmark estimate explicitly: measured serial preflight/floor-plan time + the measured or separately benchmarked room-class sums + serialized variation/assembly/final-gate/export time + a stated repair contingency. Label any unmeasured term unknown instead of substituting an average.
- API cost is likewise unknown until the benchmark records actual model calls and usage. A request count without token/image usage and retry evidence is not a cost measurement.

Parallel generated-SAM3D plan on a future multi-GPU allocation:

- Assign independent rooms to independent GPUs, retain unique output folders/ports, and keep assembly serialized after all room gates.
- Do not claim near-3x or a fixed 2-7-day wall clock before the representative benchmark identifies the GPU-bound fraction and the API/CPU/serial tail. Three GPUs can overlap independent room workers, but cannot parallelize floor-plan generation, global variation, final assembly, navigation, exports, acceptance, API throttling, or repair of the slowest room.
- After measurement, estimate the three-GPU schedule by packing measured room durations onto three workers, then add the measured serial stages and repair tail. Report both ideal packing speedup and the lower realistic speedup.
- This remains the same full-quality policy: compulsory Artiverse, generated SAM3D, ObjectThor retrieval, and every gate still apply.

Fast/HSSD plan:

- Use only when the user explicitly chooses speed over quality.
- GPU: roughly 10-40 GPU-hours.
- Wall clock on 3 GPUs: same day is plausible.
- Risk: asset variety and placement realism can be poor; not acceptable as the default full-quality contract.

## Quick Checklist Before Launch

Do not launch until these are answered:

- Output target: full-quality or fast/HSSD-only?
- Does the resolved full-quality policy say `generated_sam3d`? An `hssd` answer is a different, non-compliant fast/debug run.
- Are SAM3D and every required retrieval model already present and proven to load offline?
- Are both ArtVIP and compulsory Artiverse installed, converted/indexed, enabled, source-filtered, and validated?
- Does the prompt contain a legitimate Artiverse-eligible articulated furniture requirement?
- What gate proves that at least one Artiverse asset survives into the final assembled scene?
- Are manipulands using Objaverse/ObjectThor rather than primitive fallback?
- Has `run_single_room_worker.py --config-only` proved no forced HSSD path?
- Is the OpenAI proxy reachable from the actual compute node and persistent after laptop disconnect?
- How many GPUs, and which room per GPU?
- What exact files will each job write?
- Has floor-plan-only output passed `validate_school_floor_layout.py` before any room work?
- Does each room have top plus two oblique review renders and passing deterministic and VLM gate JSON?
- Are those review renders cutaway views that expose the room contents instead of an opaque ceiling/roof, and is that cutaway state recorded in evidence?
- Do all six classroom inventories prove exactly 12 student desks and 12 student chairs, and do the other five rooms satisfy their canonical inventory checklists?
- Do all 12 classroom desk/chair pairs have unique, finite, plausible spatial matches, and does every explicit small-item category have a distinct semantic match?
- Do the six classrooms pass the hash-bound deterministic and VLM variation gate?
- Do the library, storage room, and classroom_01 provide all three required articulated furniture roles, with real runtime provenance and at least one role validated as Artiverse?
- Do all three roles pass a real Drake articulation-motion test rather than only containing joint metadata?
- What condition allows final assembly?
- Does the assembled floor pass the three-view whole-floor reference gate?
- Does the assembled house pass the navigation/free-space gate through every required room and the two-leaf exterior entrance?
- Does the internal SQZ acceptance receipt cover the exact scene file set, and does an external package manifest bind it without a self-hash loophole?
- Where will the exact package receive its compulsory 2-GPU ParaCloud Drake acceptance?
- Which checkout is canonical for this run: SQZ, ParaCloud, or local development?
- Are uploads overlay-only with no `--delete`?

## Failure Lesson From Run 3

Run 3 completed a scene, but it was not a valid best-quality result because:

- It continued an existing `furniture_hssd_only_fast` run.
- SAM3D was bypassed by `general_asset_source=hssd`.
- Review images were generated after the fact, but no pass/fail self-exam gate blocked final export.
- Bad placement survived into the final outlook render.

For future scenes, treat this file as the pipeline contract.
