# SceneSmith SAGE-Style Checker

This tool validates SceneSmith outputs with cheap local checks. It does not
merge SAGE and SceneSmith, does not run IsaacSim, and does not call paid model
APIs.

## Money Rules

Use `--no-paid-api` for dataset runs. The checker fails if these environment
variables are present:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DASHSCOPE_API_KEY`
- `NVIDIA_API_KEY`
- `ARK_API_KEY`

The report always includes API counters:

```json
{
  "openai_api_calls": 0,
  "gemini_api_calls": 0,
  "anthropic_api_calls": 0,
  "external_paid_api_calls": 0,
  "codex_cli_calls": 0
}
```

## Usage

```powershell
python tools/sage_scene_checker/check_scenesmith_output.py `
  --scene-dir outputs/latest-run/scene_000 `
  --sage-root "E:/Researches/Tsinghua papers/SAGE/sage-main" `
  --out outputs/latest-run/scene_000/validation_report.json `
  --fail-on-warnings `
  --no-paid-api
```

`--sage-root` is optional. When present, the checker tries to load only
`server/validation.py` and use `validate_room_only_layout()`. If that import
fails or tries to pull unavailable dependencies, the checker keeps using its
built-in checks.

By default, failed warning checks are reported but do not change the top-level
`pass` result. Production acceptance should add `--fail-on-warnings`; this makes
coarse collisions, blocked doors, missing assets, and every other failed warning
produce `pass: false` and a nonzero CLI exit. The report records the selected
policy and its fatal check IDs under `acceptance_policy`.

## Inputs

Expected SceneSmith files:

- `combined_house/house_state.json`
- `combined_house/sceneeval_state.json`
- `combined_house/house.dmd.yaml` if available

## Checks

- required file existence
- JSON parse validity
- room/object count sanity
- finite transform values
- valid bbox dimensions
- object inside assigned room
- coarse AABB collision
- door clearance blockage
- missing local asset path
- missing semantic category
- missing support surface parent

## Paracloud

The checker is pure Python and does not import SceneSmith runtime modules by
default. Copy the `tools/sage_scene_checker` directory with the SceneSmith
checkout on paracloud and run the same command there. Keep paid API environment
variables unset when using `--no-paid-api`.
