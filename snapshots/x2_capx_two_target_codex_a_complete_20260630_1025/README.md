# X2 CaP-X Two-Target Codex-A Complete Baseline

Date: 2026-06-30 Asia/Shanghai

This snapshot preserves the accepted X2 CaP-X two-target visual pick-place
baseline after right and left non-oracle `codex-a` runs passed acceptance.

The machine-readable manifest is preserved at:

```text
docs/x2-accepted-baseline-20260630.manifest.json
```

## What Is Preserved

This snapshot contains the current source/config/test files needed for the
accepted path:

- X2 BEHAVIOR simulator adapter and task wrappers
- X2 visual/action primitive implementation
- X2 task configs
- Codex CLI proxy
- oracle, direct API, and codex-a smoke/stability scripts
- acceptance, summary, and audit scripts
- X2 docs and tests

Videos and visual artifacts are not copied here. README demo videos live in
the repository-level `docs/media/x2/` directory; full run artifacts remain
under `outputs/`.

## Accepted Evidence

Accepted stamp:

```text
manual_codex_a_20260630_101117
```

Right:

```text
outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
before_close_tcp_error_m=0.015655039528970842
before_close_ori_error_rad=0.0547740942117125
place_error_m=0.014131927921253392
reward=1.0
task_completed=1
```

Left:

```text
outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
before_close_tcp_error_m=0.005309625868222024
before_close_ori_error_rad=0.013709298176369174
place_error_m=0.003187055511323606
reward=1.0
task_completed=1
```

The generated code files in both runs call exactly one high-level X2 primitive:

```python
pick_and_place_red_cube_to_right_target()
pick_and_place_red_cube_to_left_target()
```

## Validate From Repository Root

```bash
python scripts/check_x2_acceptance.py \
  outputs/stability/codex-a/two_targets_*_codex_a_stability_manual_codex_a_20260630_101117_run*

python scripts/audit_x2_capx_integration.py --strict

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xingshu/miniforge3/bin/conda run --no-capture-output -n behavior \
  python -m pytest tests/test_x2_llm_api.py tests/test_x2_run_summary.py -q
```

Expected:

```text
X2_ACCEPTANCE PASS
X2_INTEGRATION_AUDIT COMPLETE
10 passed
```

## Re-run Non-Oracle Stability

```bash
REPEATS=1 scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

The script uses the local Codex configuration through `codex-a` when available,
or `codex -c 'model_provider="axonhub"'` via `CODEX_BIN=codex` and
`CODEX_MODEL_PROVIDER=axonhub`.

## Contract

Visual grasp output and action input meet at `T_world_tcp`. The action API
converts TCP to EEF internally before IK/execution.

Normal LLM-generated code should use only the reduced high-level API:

```python
pick_and_place_red_cube()
pick_and_place_red_cube_to_left_target()
pick_and_place_red_cube_to_right_target()
pick_and_place_visual_object()
```

Lower-level IK, visual, planning, and gripper functions are retained for
debugging, not as the default LLM-facing task surface.
