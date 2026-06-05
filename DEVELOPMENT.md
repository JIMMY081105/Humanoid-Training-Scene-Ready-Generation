# Development Notes

These commands cover the local checks used for the repository-owned tooling.
They do not require the full upstream SceneSmith checkout unless a script says
so explicitly.

## Python Unit Tests

Run the repository test suite from the project root:

```powershell
python -m pytest tests/unit tools/codex_benchmark/tests
```

The unit tests cover the SAGE adapter, the no-paid-API guard, benchmark cache
behavior, checkpoint handling, statistics generation, and report rendering.

## Scene Checker

The scene checker expects a SceneSmith output directory containing
`combined_house/house_state.json`.

```powershell
python tools/sage_scene_checker/check_scenesmith_output.py path\to\scene_dir --no-paid-api
```

Use `--no-paid-api` for local validation runs so hidden fallback API paths are
caught before they reach HPC jobs.

## Benchmark Suite

The benchmark package has its own configuration and guarded run scripts:

```powershell
python -m tools.codex_benchmark.benchmark --config tools/codex_benchmark/config.yaml
```

Use the PowerShell scripts under `tools/codex_benchmark/scripts/` for curated
smoke, small, and full benchmark profiles.

## Rendering

Local rendering scripts target exported scene artifacts and are designed to
stay within consumer GPU memory limits:

```powershell
python tools/local_render/render_rooms.py --help
```

Keep render outputs outside version control unless they are intentional
documentation images under `docs/renders/`.

## Repository Hygiene

- Keep generated outputs out of Git.
- Keep credentials and proxy settings in local environment configuration only.
- Prefer focused commits that pair code changes with relevant tests.
- Preserve upstream SceneSmith changes as patch files under
  `upstream-patches/`.
