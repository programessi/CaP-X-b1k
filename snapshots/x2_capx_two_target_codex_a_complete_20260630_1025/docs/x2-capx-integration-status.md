# X2 CaP-X Integration Status

Date: 2026-06-30

This page is the current acceptance checklist for bringing X2 into the CaP-X
BEHAVIOR tabletop workflow. It records the accepted stable path and the
remaining work that is outside the current simulation scope.

## Current Status

X2 is integrated far enough to run simple CaP-X tabletop pick-place tasks in
simulation:

- The BEHAVIOR-backed X2 low-level environment is registered through CaP-X
  configs.
- The task setup is owned by the CaP-X/BEHAVIOR config layer, not by generated
  code.
- The LLM-facing API is reduced to task-level primitives through
  `X2PickPlaceApi`.
- The visual chain can produce a selected X2 grasp target as `T_world_tcp`.
- The action chain consumes `T_world_tcp`, converts to internal EEF/joint
  targets as needed, and executes approach, close, lift, transfer, place, and
  release.
- Right-target and left-target oracle stability runs passed after insertion and
  fine-align hold-step hardening.
- Right-target and left-target `codex-a` non-oracle stability runs passed with
  generated code that calls only the reduced X2 task-level primitive API.
- `scripts/check_x2_acceptance.py` and
  `scripts/audit_x2_capx_integration.py --strict` both pass.

The current accepted baseline is documented here:

```text
docs/x2-accepted-baseline-20260630.md
snapshots/x2_capx_two_target_codex_a_complete_20260630_1025/
```

## Stable User-Facing Tasks

Single target:

```text
env_configs/x2/x2_pick_place_red_cube.yaml
```

Two targets, right marker:

```text
env_configs/x2/x2_pick_place_red_cube_two_targets.yaml
```

Two targets, left marker:

```text
env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml
```

All three tasks use `X2PickPlaceApi`; ordinary generated code should not create
the scene, reset the simulator, instantiate robots, or call low-level camera,
IK, PyRoKi, GraspNet, or gripper helpers directly.

## LLM-Facing Primitive Surface

The formal injected primitive set is:

```python
pick_and_place_red_cube(...)
pick_and_place_red_cube_to_left_target(...)
pick_and_place_red_cube_to_right_target(...)
pick_and_place_visual_object(...)
```

The intended generated code for the two-target right task is:

```python
RESULT = pick_and_place_red_cube_to_right_target()
```

The intended generated code for the left variant is:

```python
RESULT = pick_and_place_red_cube_to_left_target()
```

`plan_visual_grasp_tcp_pose()` and `execute_tcp_grasp_plan()` remain important
internal/lower-level building blocks. They are deliberately not exposed by
`X2PickPlaceApi.functions()` for ordinary pick-place tasks.

## Coordinate And Data Contracts

The working visual/action contract is:

```text
visual selected grasp target: T_world_tcp
action target input:          T_world_tcp
place target:                 world-frame object center [x, y, z]
```

The raw Contact-GraspNet pose is adapted before execution. Generated task code
does not convert TCP to EEF; the X2 action API handles that internally.

For the current short-term simulation baseline, table and cube obstacle boxes
come from sim-known OmniGibson scene AABBs with engineering inflation margins.
The elevated object-center correction before release is also sim-only. This is
acceptable for the current CaP-X simulation integration goal, but it is not a
2real obstacle-perception solution.

## Validated Evidence

Current post-hardening oracle stability evidence:

```text
Right:
outputs/oracle/stability/oracle/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.0022920335900174634
before_close_ori_error_rad=0.007841036302440513
object_in_hand_after_close=True
place_error_m=0.007655968983878265
Reward=1.0
Task Completed=True

Left:
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.011122489380341058
before_close_ori_error_rad=0.03634797048888362
object_in_hand_after_close=True
place_error_m=0.013434781081554163
Reward=1.0
Task Completed=True
```

Visual artifacts for those runs are under:

```text
outputs/x2_visual_artifacts/stability/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/
outputs/x2_visual_artifacts/stability/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/
```

The frozen source snapshot for this baseline is:

```text
snapshots/x2_two_target_stability_hold_baseline_20260629_1330/
```

Accepted non-oracle `codex-a` stability evidence:

```text
Stamp:
manual_codex_a_20260630_101117

Right:
outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.015655039528970842
before_close_ori_error_rad=0.0547740942117125
object_in_hand_after_close=True
place_error_m=0.014131927921253392
Reward=1.0
Task Completed=True

Left:
outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.005309625868222024
before_close_ori_error_rad=0.013709298176369174
object_in_hand_after_close=True
place_error_m=0.003187055511323606
Reward=1.0
Task Completed=True
```

Generated code for the accepted non-oracle runs:

```text
outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py
outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py
```

Visual artifacts:

```text
outputs/x2_visual_artifacts/stability/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/
outputs/x2_visual_artifacts/stability/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/
```

The complete accepted source/config/test snapshot is:

```text
snapshots/x2_capx_two_target_codex_a_complete_20260630_1025/
```

## Re-run Non-Oracle Evidence

If the GPT API key is already configured in the local `codex-a` command, use
the codex-a path. It starts `capx/serving/codex_cli_server.py` and forwards the
CaP-X OpenAI-compatible request to the local Codex CLI configuration.

```bash
REPEATS=1 \
STAMP=manual_codex_a_$(date +%Y%m%d_%H%M%S) \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

If your local executable is named `codex` instead of `codex-a`, override it:

```bash
CODEX_BIN=codex REPEATS=1 scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

The generic direct API path remains available when you explicitly start an
OpenAI-compatible proxy:

```bash
REPEATS=1 \
STAMP=manual_direct_api_$(date +%Y%m%d_%H%M%S) \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_and_check.sh
```

Summarize the saved outputs:

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_codex_a_stability_<STAMP>_run*
python scripts/summarize_x2_runs.py outputs/stability/*/two_targets_*_codex_a_stability_<STAMP>_run*
```

Run the acceptance checker:

```bash
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_codex_a_stability_<STAMP>_run*
python scripts/check_x2_acceptance.py outputs/stability/*/two_targets_*_codex_a_stability_<STAMP>_run*
```

Run the full local integration audit:

```bash
python scripts/audit_x2_capx_integration.py
```

Strict audit for the accepted baseline:

```bash
python scripts/audit_x2_capx_integration.py --strict
```

Acceptance for this last gap:

- Generated code calls exactly one target primitive for each task.
- Both right and left runs save videos.
- Both runs save visual artifacts with `grasp_summary.json`.
- `ok=True`, `object_in_hand_after_close=True`, and `Task Completed=True`.
- Before-close TCP error remains within the task tolerance; current oracle
  target is below 2 cm.

The checker encodes the current thresholds:

```text
required targets: right,left
before_close_tcp_error_m <= 0.02
before_close_ori_error_rad <= 0.10
place_error_m <= 0.10
video_*.mp4 required
matching grasp_summary.json visual artifact required
code.py must call the target primitive instead of lower-level debug APIs
```

The latest accepted run passed:

```text
X2_ACCEPTANCE PASS
X2_INTEGRATION_AUDIT COMPLETE
```

## Remaining Work Outside This Baseline

The X2 CaP-X tabletop integration can be treated as complete for the current
simulation scope. The next work is new scope:

- Replace sim-known obstacle boxes with perception-derived table/object
  geometry.
- Add more task objects and target layouts.
- Decide whether to expose lower-level `plan_visual_grasp_tcp_pose()` /
  `execute_tcp_grasp_plan()` for tasks that need custom post-grasp behavior.
- Improve trajectory quality beyond the current guarded tabletop route.
