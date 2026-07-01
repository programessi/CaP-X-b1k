# Script Entry Points

This directory contains both stable smoke-test entry points and historical X2
experiments. For X2 CaP-X integration, use the scripts below first.

## Recommended X2 Entry Points

Single red-cube non-oracle smoke:

```bash
scripts/run_x2_pick_place_red_cube_non_oracle_smoke.sh
```

Two-target codex-a non-oracle smoke, using the user's local Codex CLI config:

```bash
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

Two-object blue-cube-to-right codex-a non-oracle smoke:

```bash
scripts/run_x2_two_object_blue_right_codex_a_non_oracle_smoke.sh
```

Two-target codex-a stability loop plus local summary and acceptance check:

```bash
REPEATS=1 \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

Two-target generic OpenAI-compatible direct API non-oracle smoke:

```bash
scripts/run_x2_two_target_api_non_oracle_smoke.sh
```

Two-target generic OpenAI-compatible direct API stability loop:

```bash
REPEATS=1 \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_smoke.sh
```

Two-target generic OpenAI-compatible direct API stability loop plus local
summary and acceptance check:

```bash
REPEATS=1 \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_and_check.sh
```

Two-target oracle stability loop:

```bash
scripts/run_x2_two_target_oracle_stability_smoke.sh
```

Two-object blue-cube-to-right oracle smoke:

```bash
scripts/run_x2_two_object_blue_right_oracle_smoke.sh
```

Two-object experimental RGB-D visual obstacle oracle smoke:

```bash
scripts/run_x2_two_object_blue_right_rgbd_visual_oracle_smoke.sh
```

This route uses RGB-D object/table obstacle boxes, visual grasp-pose place
offset, precontact one-shot reobserve with quality-gated fallback, and slow
vertical place descent. Summaries should show `reobserve_adopted`,
`reobserve_reason`, `final_close_axis_offset_m`, and
`place_descent_waypoints`.

Summarize saved X2 runs:

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
python scripts/summarize_x2_runs.py outputs/codex-a/x2_pick_place_two_objects_blue_right_codex_a_non_oracle_<STAMP>
```

Check saved X2 runs against the current acceptance gate:

```bash
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

Audit the full local X2 CaP-X integration state:

```bash
python scripts/audit_x2_capx_integration.py
```

Use strict mode after collecting direct API evidence:

```bash
python scripts/audit_x2_capx_integration.py --strict
```

## Optional Historical Path

The Codex CLI proxy scripts are the preferred route when the GPT API key is
already configured in the local `codex-a` command:

```text
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
scripts/run_x2_two_target_codex_a_stability_smoke.sh
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

Set `CODEX_BIN=codex` only if your local command is named `codex`; otherwise
the script will prefer `codex-a` when it is available.

If `codex-a` is a shell alias for `codex -c 'model_provider="axonhub"'`, the
script's fallback path is already equivalent: `CODEX_BIN=codex` and
`CODEX_MODEL_PROVIDER=axonhub`.

## Debug Helpers

These scripts are not the formal LLM-facing API, but they are useful when
debugging X2 visual/action contracts:

```text
scripts/archive/x2_experiments/x2_chest_camera_visual_chain_smoke.py
scripts/archive/x2_experiments/x2_chest_visual_grasp_to_joint_ik_demo.py
scripts/archive/x2_experiments/x2_code_exec_grasp_only_demo.py
scripts/archive/x2_experiments/x2_injected_visual_grasp_api_example.py
scripts/archive/x2_experiments/x2_pyroki_precontact_insert_grasp_demo.py
scripts/archive/x2_experiments/x2_replay_visual_target_joint_tracking.py
scripts/archive/x2_experiments/x2_replay_visual_target_orientation_sweep.py
scripts/archive/x2_experiments/x2_visual_grasp_api_closed_loop_smoke.py
scripts/archive/x2_experiments/x2_visual_pyroki_precontact_insert_grasp_demo.py
```

## Archived Experiments

Older X2 tuning and diagnostic scripts live under:

```text
scripts/archive/x2_experiments/
```

They are retained as evidence and parameter history. They are not the
recommended entry points for new CaP-X task validation.
