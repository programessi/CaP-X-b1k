# X2 CaP-X Accepted Baseline - 2026-06-30

This document pins the current accepted X2 CaP-X simulation baseline.

## Scope

The accepted baseline is a simple tabletop two-target pick-place task for X2:

- Perception detects and segments the red cube.
- GraspNet-style visual grasp generation returns candidate TCP poses.
- The X2 task API selects a reachable candidate.
- PyRoKi/trajopt plus simplified table/object guards generate an approach route.
- Joint IK execution reaches the grasp pose, closes the gripper, lifts, moves
  above the selected target, descends, opens the gripper, and verifies placement.
- CaP-X non-oracle code generation calls the reduced LLM-facing X2 primitive API.

This baseline is for CaP-X/BEHAVIOR simulation integration. It still uses
simulation-known table/object boxes for the short-term guarded approach.

## Stable LLM-Facing API

Normal generated task code should use only these high-level functions:

```python
pick_and_place_red_cube()
pick_and_place_red_cube_to_left_target()
pick_and_place_red_cube_to_right_target()
pick_and_place_visual_object()
```

The lower-level APIs remain available for debugging, but should not be exposed
as the default LLM task surface:

- `plan_visual_grasp_tcp_pose`
- `execute_tcp_grasp_plan`
- `move_tcp_joint_ik`
- IK, PyRoKi, gripper, and environment setup helpers

## Coordinate Contract

The visual/action handoff uses:

```text
T_world_tcp
```

Visual grasp output is interpreted as TCP relative to world. X2 action code
internally converts TCP to EEF before solving/executing arm motion.

## Core Files

Task/config layer:

- `env_configs/x2/x2_pick_place_red_cube.yaml`
- `env_configs/x2/x2_pick_place_red_cube_two_targets.yaml`
- `env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml`
- `capx/envs/tasks/x2/x2_pick_place_red_cube.py`
- `capx/envs/tasks/x2/x2_pick_place_red_cube_two_targets.py`
- `capx/envs/simulators/x2_b1k.py`

Primitive/API layer:

- `capx/integrations/x2/control.py`
- `capx/integrations/x2/vision.py`
- `capx/integrations/motion/pyroki_snippets/_trajopt.py`

LLM/proxy and validation:

- `capx/serving/codex_cli_server.py`
- `scripts/run_x2_two_target_codex_a_stability_and_check.sh`
- `scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh`
- `scripts/check_x2_acceptance.py`
- `scripts/summarize_x2_runs.py`
- `scripts/audit_x2_capx_integration.py`
- `tests/test_x2_llm_api.py`
- `tests/test_x2_run_summary.py`

## Accepted Evidence

Accepted non-oracle `codex-a` run stamp:

```text
manual_codex_a_20260630_101117
```

Right target:

```text
outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
before_close_tcp_error_m=0.015655039528970842
before_close_ori_error_rad=0.0547740942117125
object_in_hand_after_close=True
place_error_m=0.014131927921253392
reward=1.0
task_completed=1
```

Left target:

```text
outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
before_close_tcp_error_m=0.005309625868222024
before_close_ori_error_rad=0.013709298176369174
object_in_hand_after_close=True
place_error_m=0.003187055511323606
reward=1.0
task_completed=1
```

Generated code:

- `outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py`
- `outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py`

Videos:

- `video_combined_global.mp4`
- `video_combined_robot.mp4`

GitHub README demo copies:

- `docs/media/x2/x2_right_target_global.mp4`
- `docs/media/x2/x2_right_target_robot.mp4`
- `docs/media/x2/x2_left_target_global.mp4`
- `docs/media/x2/x2_left_target_robot.mp4`

Visual artifacts:

- `outputs/x2_visual_artifacts/stability/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/x2_pick_place_red_cube_20260630_101419_805/`
- `outputs/x2_visual_artifacts/stability/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/x2_pick_place_red_cube_20260630_101910_210/`

## Validation Commands

Acceptance for the accepted run:

```bash
python scripts/check_x2_acceptance.py \
  outputs/stability/codex-a/two_targets_*_codex_a_stability_manual_codex_a_20260630_101117_run*
```

Full integration audit:

```bash
python scripts/audit_x2_capx_integration.py --strict
```

Unit tests:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xingshu/miniforge3/bin/conda run --no-capture-output -n behavior \
  python -m pytest tests/test_x2_llm_api.py tests/test_x2_run_summary.py -q
```

Latest validation result:

```text
X2_ACCEPTANCE PASS
X2_INTEGRATION_AUDIT COMPLETE
10 passed
```

## Snapshot

The source/config/test snapshot for this baseline is:

```text
snapshots/x2_capx_two_target_codex_a_complete_20260630_1025/
```

The machine-readable baseline manifest is:

```text
docs/x2-accepted-baseline-20260630.manifest.json
```

Large videos and visual artifacts are not copied into the snapshot. They remain
in `outputs/` and are referenced above.

## Change Discipline

Treat this as the restore point before extending X2 tasks. Future changes should
preserve:

- the `T_world_tcp` visual/action contract,
- reduced LLM-facing API exposure,
- acceptance thresholds in `scripts/check_x2_acceptance.py`,
- right/left two-target non-oracle evidence,
- oracle stability evidence.

Exploratory X2 scripts should stay under `scripts/archive/x2_experiments/` or
use clearly named smoke-test scripts. The core accepted path should remain
small and reproducible.
