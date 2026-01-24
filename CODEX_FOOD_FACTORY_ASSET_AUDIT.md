# Food factory asset-source audit

Audit time: 2026-07-13, Asia/Kuala_Lumpur (`+08:00`)

Scope: the local repository and the live SQZ execution checkout at
`/root/workspace/scenesmith-hts`. This was a read-only audit. No full factory was
started, no dataset was downloaded, no retrieval server was launched, and no SAM 3D
asset was generated.

The complete 229-row object-by-object classification is in
`CODEX_FOOD_FACTORY_ASSET_INVENTORY.csv`. The CSV retains the requested columns,
adds the room/area section, records exact indexed IDs where a reliable match exists,
and assigns exactly one A-J status to every requested row.

## Executive result

The factory should **not** start as a full generation yet. The active SceneSmith
policy can already cover ordinary furniture, storage, staff facilities, laboratory
props, small appliances, and safety fixtures. Structural industrial forms should be
procedural. The current connected indexes do not contain credible factory machinery.

For a consolidated minimum plan of 70 unique model families (rather than 229 raw
rows containing synonyms and duplicates):

- 40 families (57.1%) can reuse a connected dataset asset or a modular assembly
  anchored by one.
- 23 families (32.9%) should be procedural primitives.
- 7 families (10.0%) merit one SAM 3D generation each.

The 7 recommended SAM models are: a forklift; a pallet jack; a hygienic
mixer/hopper; a food washing/sorting/inspection unit; a filling/dosing unit; a
combined sealing/carton/label module; and a box/delivery truck. Requested aliases
such as `packaging machine`, `packing machine`, `heat sealer`, and `carton sealer`
must reuse the same modular output rather than trigger separate generations.

## A. Real asset sources

### Active policy and exact configuration

The observed live worker command uses `--asset-pipeline generated_sam3d` and
`--port-offset 100`. Its configuration is composed from
`configurations/config.yaml` by `scripts/run_single_room_worker.py::_load_cfg`.
The decisive function is `_asset_pipeline_overrides` in that script:

- `furniture_agent.asset_manager.general_asset_source=generated`
- `wall_agent.asset_manager.general_asset_source=generated`
- `ceiling_agent.asset_manager.general_asset_source=generated`
- `manipuland_agent.asset_manager.general_asset_source=objaverse`
- all four agent backends are `sam3d`
- `manipuland_agent.asset_manager.objaverse.use_top_k=10`
- `*.asset_manager.articulated.sources.artvip.enabled=true` for all four agents
- `furniture_agent.asset_manager.articulated.sources.artiverse.enabled=true`
- `furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true`
- materials use `data/materials` with the validated derivative index at
  `data/materials_full_quality_contract/embeddings`

`scripts/run_single_room_worker.py::_service_names` selects `geometry`,
`objaverse`, `articulated`, and `materials` for this policy. HSSD is selected only by
the alternate `hssd` policy.

### Source state

| Source | Enabled now | Installed state | Search/index evidence | Static/articulated and collision | Conversion requirement |
|---|---:|---|---|---|---|
| HSSD | No | Retrieval code is installed; data directories `data/hssd` and `data/hssd-models` are absent | Defaults are `HSSD_DATA_PATH=data/hssd-models`, `HSSD_PREPROCESSED_PATH=data/preprocessed`; `HssdConfig` and `HssdRetriever`; no live port 7106 | Would be static; no local assets to inspect | Not usable in current environment. `_download_hssd_mesh_if_missing` supports per-mesh retrieval only after a valid HSSD installation/configuration |
| ArtVIP | Yes | Complete, 8.7 GB, read-only symlink `data/artvip_sdf -> /root/workspace/scenesmith/data/artvip_sdf` | 197/197 IDs in `data/artvip_sdf/embeddings/{clip_embeddings.npy,embedding_index.yaml,metadata_index.yaml}` | Articulated SDF. All 197 SDFs exist; every SDF has joints and collision tags. Collision-tag counts are high (129-1,922), so selected assets need simulation-cost review | Direct SDF copy; SceneSmith combines link meshes for placement and retains articulation |
| Artiverse connected index | Yes, furniture router | Full release is installed, but the connected index is deliberately narrow | 500 indexed assets in `data/artiverse/embeddings`; manifest status `pass`; categories: `armoire` 98, `bookcase` 22, `chest_of_drawers` 380 | Articulated, publisher physics bound. Each indexed item has 1-18 movable joints and 3-32 collision elements | Direct SDF use after the existing provenance/visual-normalization checks |
| Artiverse raw release | Installed but mostly disconnected | Complete extracted tree, 144 GB; 84 top-level categories | Raw root `data/artiverse/data`; examples: `locker/3dw/be128...`, `door/objaverse/e64d...`, `window/3dw/2b12...`, `refrigerator/objaverse/9052...`, `shopping_cart/objaverse_xl/f8d5...`, `tool_cabinet/3dw/d33c...` | Mixed raw publisher assets. Only the 500 indexed records have current SceneSmith physics/provenance authority | A new, category-scoped preparation and validation pass is required before these raw categories are routable. Do not treat their mere directory presence as availability |
| PartNet-Mobility | No | No `data/partnet_processed` or `data/partnet_embeddings` | Fallback keys in `ArticulatedRetrievalApp`: `PARTNET_DATA_PATH` and `PARTNET_EMBEDDINGS_PATH` | No connected records | Missing; raw Artiverse `partnext` subtrees do not make a PartNet-Mobility retrieval source |
| ObjectThor through the `objaverse` adapter | Yes, manipulands | Complete, 56 GB, symlink `data/objathor-assets -> /root/.objathor-assets/2023_09_23` | 50,092 records, 768-dimensional index, no missing payloads; `data/objathor-assets/preprocessed/{clip_embeddings.npy,embedding_index.yaml,metadata_index.json,object_categories.json}` | Static visual payloads (`50,092 .pkl.gz`, zero GLB). Categories: 16,826 large, 29,957 small, 2,915 wall, 394 ceiling. No authored SceneSmith collision SDF | `construct_objaverse_mesh_path` loads the publisher payload; `AssetManager._convert_mesh_to_simulation_asset` must canonicalize, scale, run CoACD/V-HACD, and emit Drake SDF |
| General Objaverse | No separate source | Adapter code exists, but `data/objaverse` is absent. The configured adapter points to ObjectThor | `ObjaverseConfig` and `ObjaverseRetriever`; current `OBJAVERSE_DATA_PATH=data/objathor-assets` | No separately configured corpus | Not separately usable |
| Materials | Yes | Source 45 GB; validated derivative index 7.9 MB | Source index 1,949 rows. `materials_contract_manifest.json` retains 1,934 and names 15 missing material directories | Textures/materials only, not geometry or collision | Applied to generated/existing meshes |
| Primitive/procedural | Partially implemented | Code present | `AssetRouter._create_primitive_ceiling_panel_geometry`, `_create_primitive_projector_geometry`, `_create_primitive_manipuland_fallback_geometry`; `thin_covering_generator.py` | Static generated GLB; thin coverings can use analytic collision, other generated meshes pass through CoACD | The current router is not a turnkey factory-part library. Factory tanks, pipes, tables, racks, barriers, and conveyors require a later small composition layer; none was added during this audit |
| SAM 3D | Yes | Code, weights, DINO cache, and MoGe cache are complete | Entry point `generate_geometry_from_image(..., backend="sam3d")`, then `generate_with_sam3d`; preflight evidence listed below | Produces textured static GLB; collision is not produced by SAM 3D itself | `AssetManager._convert_mesh_to_simulation_asset` creates collision and SDF after generation |

### Download/completeness conclusions

- Fully downloaded: ArtVIP, the full Artiverse release, ObjectThor, materials, and
  all 17 attested SAM 3D configuration/checkpoint/cache artifacts.
- On-demand capable: HSSD contains `_download_hssd_mesh_if_missing`, but is unusable
  because its base data/index is absent. ObjectThor retrieval is local and complete;
  it is not downloading query results on demand. Artiverse is local and complete but
  unindexed categories require deliberate preprocessing, not opportunistic downloads.
- Missing files: materials source names 15 absent directories (the 1,934-row contract
  removes them); Artiverse preparation discovered 560 targeted candidates and indexed
  500, leaving 60 cleanly unresolved. ObjectThor has 0/50,092 missing payloads and
  ArtVIP has 0/197 missing SDFs.

### Live servers and endpoints

At audit time, process 11860/12174 owned the active room worker and these localhost
listeners:

| Port | Service | Endpoint | Live observation |
|---:|---|---|---|
| 7105 | SAM 3D geometry generation | `POST /generate_geometries`, `GET /health` | Healthy; one live GPU worker, zero audit-time requests |
| 7106 | HSSD retrieval | `POST /retrieve_objects`, `GET /health` | Not listening |
| 7107 | articulated retrieval (ArtVIP + Artiverse) | `POST /retrieve_objects`, `GET /health` | Healthy; retriever loaded, zero audit-time requests |
| 7108 | materials retrieval | `POST /retrieve_materials`, `GET /health` | Healthy; retriever loaded, zero audit-time requests |
| 7109 | ObjectThor/Objaverse adapter | `POST /retrieve_objects`, `GET /health` | Healthy; retriever loaded, 15 completed requests, 0 failures, 10.50 s average processing time |
| 8187 | Blender | service-specific API | Listening |
| 7354 | CoACD/V-HACD collision service | `GET /health` | Healthy |

The port mapping is exact from `scripts/run_single_room_worker.py::_configure_ports`
(`7005..7009 + offset 100`). Route definitions are in the corresponding
`server_app.py` classes under `scenesmith/agent_utils/*_server/`.

## B-D. Factory inventory search and classification

The audit searched all 229 requested rows (213 unique labels) using exact names,
normalized hyphen/spacing variants, singular/plural forms, and common aliases. It
searched the complete connected metadata for 197 ArtVIP, 500 Artiverse, and 50,092
ObjectThor records. It also inspected all 84 raw Artiverse category names.

Raw row status totals in the CSV are:

| Status | Rows | Meaning in this audit |
|---|---:|---|
| C | 55 | Strong ObjectThor visual candidate, but collision/SDF conversion is mandatory |
| D | 37 | Strong ArtVIP/Artiverse articulated candidate |
| F | 92 | Safer and more accurate as dimensional primitives |
| G | 11 | Compose from multiple existing/primitives |
| H | 24 | Missing distinctive equipment; aliases collapse to 7 unique SAM outputs |
| I | 6 | Omit from the minimum build |
| J | 4 | No reliable result; manual verification required |

No row is classified A or B because the audit does not claim a static source mesh is
simulation-ready without loading and validating its converted collision, and no
factory-specific resizing/material edit was performed. This is intentionally
fail-closed.

Important rejected lexical false positives include a toy/boxed truck, a trolley
figurine, a refrigerator magnet, a door tag, a painting containing lockers, a mixer
image printed on a box, and unrelated objects whose descriptions merely contain
`tank`, `pump`, `motor`, `scale`, `barrier`, or `grass`. They are not counted as
available.

Useful exact examples include:

- ArtVIP cartons: `household_items/model_carton_10`
- ArtVIP workshop toolbox: `household_items/model_clamshell _tool_box_2`
- ArtVIP refrigerator: `major_appliances/model_refrigerator_6`
- ArtVIP water dispenser: `major_appliances/model_water_dispenser_5`
- ArtVIP toilet: `major_appliances/model_closestool_4`
- ArtVIP office chair: `small_furniture/model_office_chair_3`
- Artiverse cabinet: `artiverse/armoire/3dfModel/ab16f4ce-dc13-4694-bb5b-288c850a12aa`
- Artiverse bookcase: `artiverse/bookcase/3dfModel/4c57d811-0177-4d2b-9a18-369efb8ae234`
- ObjectThor metal shelf: `f090d4b341d749af8d74ac317e9d4b66`
- ObjectThor microscope: `f195dea3125d4b188ad6134121ce60d3`
- ObjectThor workbench: `9903d3f3a9c94c478cca7eee8ba9e6ab`
- ObjectThor fire extinguisher: `3b38813c391d4d3f948076123eb97eca`

## E. Minimum factory asset plan

### Required for functional layout (43 model families)

Procedural shell/windows/doors/partitions; loading platform; floor drains; pallets
and rack bays; shelves/cabinets; cartons/crates/bins/drums; receiving desk and scales;
wash sinks and stainless tables; one conveyor family; tanks/vats/hoppers; pipes,
valves, motors and control panels; guards and emergency stops; cold-room shell and
refrigeration; QC bench/microscope/workstation; maintenance workbench/tool storage;
office workstation; changing lockers; break-room table/appliances; toilets; fire
extinguisher, rails, bollards, exit/hygiene signs; forklift and pallet jack.

### Required for high visual quality (20 model families)

Hygienic mixer/hopper; food washing/sorting/inspection unit; filling/dosing unit;
sealing/carton/label module; box truck; articulated cabinets/lockers; realistic
commercial refrigeration; tool chest; office chair; microscope; soap/hand-dryer
fixtures; exterior light; material variants for stainless steel, safety yellow,
food-safe plastic, rubber, cardboard, and galvanized steel.

### Optional decoration (7 model families)

Protective clothing, safety vest, helmet, shrubs, small trees, decorative grass, and
nonessential flower-bed variants. These do not justify SAM generation.

### Counts

- Raw required labels: 213 unique (229 rows including duplicates across rooms).
- Consolidated minimum model families: 70.
- Selected connected dataset/model families: 40.
- Model families expected to repeat instances: 31 (pallets, cartons, crates, bins,
  rack bays, shelves, tables, chairs, signs, lights, barriers, pipes, valves, etc.).
  The exact placed-instance count cannot be known before layout design.
- Procedural unique model families: 23.
- Genuinely missing distinctive model families: 7.
- Recommended SAM 3D generations: 7, one per family, then instance reuse.

### Exact source recommendations

- HSSD: **none**; it is absent and disabled.
- Artiverse connected index: storage/control/lab/PPE cabinets via the indexed
  `armoire`, `bookcase`, and `chest_of_drawers` families. Do not use raw categories
  until a scoped preparation contract exists.
- ArtVIP: cartons, toolboxes/tool chests, refrigerator/freezer, water dispenser,
  microwave, toilet, office chair, filing pedestal, shoe cabinet, kitchen unit, and
  storage cabinet. Articulation is useful for appliance/cabinet doors and drawers;
  freeze nonessential joints when simulation cost outweighs value.
- ObjectThor: metal shelf, crates, food container/tray, drum/barrel, terminal,
  workstation, monitor/printer, sink, soap dispenser, control panel, cooling unit,
  compressor, microscope, task light, workbench, notice board, clock/plant, hand
  dryer, extinguisher, and exterior light. Every selected payload needs visual review
  plus SceneSmith collision/SDF conversion.
- Procedural: industrial windows and doors; loading dock/platform/leveller; drains,
  ducts and grilles; pallets/racks/wrapped pallet; scales; stainless tables; conveyor
  frame/belt/rollers; rectangular tanks/vats/hoppers; pipes; guards/rails/barriers;
  bollards; floor markings; signs; cold-room shell/door/shelves; lab benches/racks;
  toilet partitions; ramps/canopies/planters.

## F-G. SAM 3D runtime audit

### Implementation

- Model stack: SAM 3 image segmentation + SAM 3D Objects
  `InferencePipelinePointMap`, version `3dfy_v9`, with MoGe ViT-L depth and cached
  DINOv2 ViT-L/14-reg features.
- Entry point: `scenesmith/agent_utils/geometry_generation_server/geometry_generation.py::generate_geometry_from_image`.
- SAM implementation: `sam3d_pipeline_manager.py::generate_with_sam3d`, calling
  `generate_mask` then `generate_3d_from_mask`.
- Input: one RGB image plus either foreground mode or an object-description prompt.
- Internal preprocessor resolution: 518x518. Structured latent resolution is 64;
  sparse-structure resolution is 16.
- Output: textured, UV-mapped GLB. The pipeline enables mesh postprocessing, texture
  baking, and layout postprocessing.
- Stages inside the selected implementation: cached model startup; SAM mask;
  MoGe/point-map and sparse-structure inference; structured-latent/mesh decode; mesh
  postprocess; texture bake; GLB export. SceneSmith then performs VLM physical
  analysis, canonicalization/resizing, CoACD/V-HACD, and Drake SDF export.
- Collision is **not** a SAM 3D output. It is generated later by
  `AssetManager._generate_collision_geometry` and
  `_convert_mesh_to_simulation_asset`.
- Reuse: pipeline objects are singleton-cached, and one generated GLB/SDF can be
  instanced repeatedly.
- Parallelism: independent assets can be split across GPUs. The mask, reconstruction,
  mesh decode/postprocess, texture bake, and export for one asset are serial. Each GPU
  needs its own approximately 19 GB model residency.

### Real cache and GPU evidence

Preflight file:
`outputs/preflight/full_quality_school_reference_20260710/sam3d_offline_load.json`.
It passed with 17 hash-bound artifacts, including:

- SAM checkpoint `/root/workspace/scenesmith/external/checkpoints/sam3.pt`
  (3,450,062,241 bytes)
- pipeline config `/root/workspace/scenesmith/external/checkpoints/pipeline.yaml`
- SAM 3D sparse/structured generators and decoders
- MoGe cache `/root/.cache/huggingface/hub/models--Ruicheng--moge-vitl/.../model.pt`
- DINO source `/root/.cache/torch/hub/facebookresearch_dinov2_main`
- DINO weights `/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

Offline load elapsed 88.318 s and passed a 512x512 segmentation smoke test. Model
residency after load was 17,287,064,576 allocated bytes; peak allocated was
19,037,137,408 bytes and peak reserved 19,306,381,312 bytes.

The machine has one NVIDIA A10 (24,564 MiB). At the end of the audit it was already
occupied by the active room worker: 16,927 MiB used, 7,188 MiB free. Historical GPU
CSV evidence contains 7,346 samples and reached 23,941 MiB and 100% utilization.
System RAM is 540,663,087,104 bytes total; current used RAM was 47,030,210,560 bytes.
No trustworthy peak-system-RAM metric was logged, so none is invented.

### Real timings

Primary evidence:
`outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710/pipeline.log`.

| Stage | Samples | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| SAM stack startup | 15 | 40.51 s | 44.15 s | 50.27 s | 63.22 s |
| SAM 3 image-model build | 18 | 10.71 s | 11.23 s | 13.32 s | 14.28 s |
| SAM 3D pipeline instantiation | 15 | 24.24 s | 27.71 s | 32.98 s | 42.95 s |
| Source image generation | 45 | 7.58 s | 11.51 s | 14.49 s | 20.84 s |
| SAM 3D geometry/texture/GLB | 39 | 29.78 s | 36.61 s | 53.80 s | 92.65 s |
| Mesh VLM physical analysis | 132 | 5.97 s | 11.22 s | 16.69 s | 40.64 s |
| Scene collision validation (not collision creation) | 163 | 6.04 s | 6.82 s | 11.00 s | 34.72 s |

A measured 10-asset furniture batch completed in 649.21 s (64.92 s/asset average,
with startup amortized). Existing generated GLBs number 320 in the audited room,
347,628,292 bytes total; median 407,794 bytes, P90 1,769,196 bytes, maximum
19,194,204 bytes. That file-size corpus includes all generated room assets, not only
the 39 timing-matched SAM entries.

### Smoke-benchmark decision

No new smoke benchmark was run. Existing evidence is substantially stronger than a
single new sample, and the only A10 was already allocated to the live generation run.
Starting another model would risk out-of-memory and violate the audit's low-risk
condition.

## H. Runtime estimates

These estimates cover the 7 recommended unique SAM models and reuse their outputs.
They do not include factory layout generation, agent critique loops, final rendering,
or simulator acceptance.

| Scenario | Warm time/model | One cold model | Five models | Ten models | Seven recommended |
|---|---:|---:|---:|---:|---:|
| Optimistic | 50 s | 1m30s | 4m50s | 9m00s | 6m30s |
| Most likely | 65 s | 1m49s | 6m09s | 11m34s | 8m19s |
| Conservative | 190 s | 4m13s | 16m53s | 32m43s | 23m13s |

Assumptions: one startup is paid per worker; source images are available or generated
once; no retries; the most-likely warm rate follows the measured 10-asset batch; the
conservative rate sums observed high-tail stages rather than pretending collision
creation was separately measured.

On one A10, the most-likely SAM-only time is about 8m19s. On two equivalent GPUs,
four serial waves plus parallel startup are about 5m04s. On three, three waves plus
startup are about 3m59s. These are potential throughput estimates, not SQZ promises:
SQZ currently exposes only one A10. Image calls and CPU collision work may overlap,
but each asset's SAM pipeline is serial and model loading occurs once per GPU worker.

## I. Recommendation

Do not start the full factory generation now. First approve a 70-family asset
manifest, manually review the exact 40 selected dataset meshes, validate converted
collision for ObjectThor assets, and decide whether to add a scoped Artiverse index
for `locker`, `door`, `window`, `refrigerator`, `shopping_cart`, and `tool_cabinet`.
That index expansion could reduce substitutions, but it is not required to eliminate
the 7 genuinely distinctive missing models.

Fastest low-risk implementation sequence:

1. Freeze the minimum inventory and map all repeated instances to one selected ID.
2. Build the 23 dimensional primitive families and validate navigation/collision.
3. Convert and visually review the 40 selected ArtVIP/Artiverse/ObjectThor families.
4. Generate only the 7 missing SAM models, one per family, with the existing cached
   worker and strict per-asset timeout/receipts.
5. Assemble a small receiving/processing/packing proof room before any full factory.
6. Start the full factory only after that proof passes visual, collision, and simulator
   checks.
