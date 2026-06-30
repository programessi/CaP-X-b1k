# X2 Pick-Place Current Baseline

Date: 2026-06-29

This document records the current X2 CaP-X tabletop pick-place baseline. The
purpose is version management: keep the working path stable while future changes
improve trajectory quality, perception-based obstacles, and broader tasks.

For the current integration acceptance checklist, see
`docs/x2-capx-integration-status.md`. The short version is:

```text
Stable in oracle simulation: single red-cube pick-place, two-target right, two-target left.
LLM-facing API: X2PickPlaceApi with four task/composable pick-place primitives.
Remaining evidence gap: manual direct-API non-oracle two-target stability run.
```

## Working Entry Points

Task config:

```text
env_configs/x2/x2_pick_place_red_cube.yaml
```

Task implementation:

```text
capx/envs/tasks/x2/x2_pick_place_red_cube.py
```

X2 primitive implementation:

```text
capx/integrations/x2/control.py
```

LLM-facing task primitive:

```python
RESULT = pick_and_place_red_cube()
```

Reusable visual/action primitive:

```python
pick_and_place_visual_object(...)
```

Reusable action primitive:

```python
execute_tcp_grasp_plan(...)
```

`execute_tcp_grasp_plan()` remains useful as the execution core. For the current
task it performs approach, grasp, lift, transfer, placement, release, and
retreat. For future tasks it can also be used as the lower-level "reach a visual
TCP grasp and close" building block if the rest of the task needs custom code.

## Current Call Chain

The non-oracle CaP-X path is:

```text
LLM generated code
-> pick_and_place_red_cube()
-> pick_and_place_visual_object()
-> plan_visual_grasp_tcp_pose()
-> OWL-ViT detection
-> SAM2 segmentation
-> Contact-GraspNet / GraspNet grasp generation
-> X2 TCP adapter
-> candidate selection and retry
-> execute_tcp_grasp_plan()
-> PyRoKi collision-aware precontact approach
-> slow TCP insertion
-> gripper close and in-hand check
-> lift
-> explicit high TCP waypoint transfer
-> elevated sim-known place correction
-> mostly vertical descent
-> release
```

The X2 red-cube task uses `X2PickPlaceApi`, which intentionally exposes a small
API surface to generated code. This reduces accidental calls to low-level debug
helpers and makes the generated program close to the intended CaP-X style.

## Coordinate Contract

The visual/action contract that currently works is:

```text
visual final grasp target: T_world_tcp
action primitive target:   T_world_tcp
```

The raw GraspNet output is not directly sent to the controller. It is adapted to
the X2 TCP frame first:

```text
T_world_x2_tcp = T_world_graspnet_raw @ T_graspnet_raw_to_x2_tcp
```

If the controller or IK needs an EEF target, the X2 API performs TCP-to-EEF
conversion internally. The generated CaP-X task code should not need to reason
about EEF/TCP offsets for this task.

The place target is a world-frame object-center position:

```text
[x, y, z]
```

## Planning And Collision Handling

Current short-term simulation behavior:

- The grasp pose comes from the vision chain.
- The obstacle boxes for the cube and table come from sim-known OmniGibson scene
  AABBs plus engineering inflation margins.
- PyRoKi is used for the precontact approach to reduce early collision with the
  cube and table.
- TCP insertion into the grasp is slow and local.
- The final insertion/fine-align commands are intentionally held for multiple
  action steps. A left-target stability rerun showed that the visual grasp
  target can be valid while the joint-position drive has not fully converged;
  longer insertion hold/settle time reduced the before-close TCP error back to
  the accepted 2 cm range.
- Transfer after grasp no longer uses one free PyRoKi target from grasp to place.
  It uses explicit high TCP waypoints, then descends mostly vertically.
- A sim-known object-center correction is applied above the placement target
  before the final descent.

This is acceptable for short-term CaP-X simulation integration. It is not a
2real obstacle-perception solution because the table/cube obstacle boxes and the
place correction read simulator state.

## Accepted Non-Oracle Run

Output:

```text
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_waypoint_transfer_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Metrics:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.016780745142452466
before_close_ori_error_rad=0.04858480025799528
object_in_hand_after_close=True
place_error_m=0.01853152439298318
Reward=1.0
Task Completed=True
```

Videos:

```text
video_combined_global.mp4
video_combined_robot.mp4
```

This is the current baseline to preserve before further trajectory/planning
experiments.

## Reproduce

Run the non-oracle CaP-X smoke:

```bash
scripts/run_x2_pick_place_red_cube_non_oracle_smoke.sh
```

Useful overrides:

```bash
OUTPUT_DIR=./outputs/x2_pick_place_red_cube_my_run \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_pick_place_red_cube_non_oracle_smoke.sh
```

`capx/envs/launch.py` writes non-oracle outputs under a model-name subdirectory,
so `OUTPUT_DIR` should normally not include `MiniMax-M2.7` itself.

The script uses:

```text
BEHAVIOR_ROOT=/home/xingshu/workspaces/fys/BEHAVIOR-1K
CONDA_ENV=behavior
```

by default.

## Hardened Code Extraction Smoke

After hardening `_clean_model_code()` to strip `<think>...</think>` output, the
same non-oracle path was rerun successfully.

Output:

```text
outputs/MiniMax-M2.7/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_baseline_hardened_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

This path has a duplicated `MiniMax-M2.7` directory because the manual
`OUTPUT_DIR` override already included the model name. The smoke result itself
is valid.

Metrics:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.007781398923911473
before_close_ori_error_rad=0.02522619288209918
object_in_hand_after_close=True
place_error_m=0.014900085488091878
Reward=1.0
Task Completed=True
```

Generated code saved by CaP-X:

```python
RESULT = pick_and_place_red_cube()
execution = RESULT.get("execution", {})
before_close_error = execution.get("before_close_error", {})
place = execution.get("place", {})
print("X2_PICK_PLACE_RESULT "
      + "ok=" + str(bool(RESULT.get("ok"))) + " "
      + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
      + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
      + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
      + "place_error_m=" + str(place.get("place_error_m")))
```

The raw model response contained a `<think>...</think>` block. The extractor
removed it and executed only the Python code above.

## Reduced API Oracle Smoke

After narrowing `X2PickPlaceApi.functions()` to the formal LLM-facing
primitive set, the oracle path was rerun with the same one-call program shape:

```python
RESULT = pick_and_place_red_cube()
```

Output:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_reduced_api_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Metrics:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.009280033147572208
before_close_ori_error_rad=0.027969652238196776
object_in_hand_after_close=True
place_error_m=0.01605146254181371
Reward=1.0
Task Completed=True
```

## Two-Target Reduced API Oracle Smoke

The second X2 tabletop task validates that the reduced API can expose a simple
target-selection primitive to CaP-X code:

```python
RESULT = pick_and_place_red_cube_to_right_target()
```

The stable two-target layout keeps the cube at `[0.32, -0.08, 0.921]` and moves
the visual target markers away from the cube into the forward placement region:

```text
left target:  [0.27, 0.055, 0.921]
right target: [0.37, 0.055, 0.921]
```

Output:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_marker_forward_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Metrics:

```text
X2_TWO_TARGET_RESULT ok=True target=right
before_close_tcp_error_m=0.009925023974328685
before_close_ori_error_rad=0.029553686067071306
object_in_hand_after_close=True
place_error_m=0.018812150410754087
Reward=1.0
Task Completed=True
```

Videos:

```text
video_combined_global.mp4
video_combined_robot.mp4
```

The failed intermediate two-target layout placed the markers beside the cube on
the same `y=-0.08` line. That failure was concentrated before gripper close
with `before_close_tcp_error_m=0.09472798472790968`, so the accepted layout
keeps markers out of the grasp perception region.

## Two-Target Codex-A Non-Oracle Smoke

The two-target task was also run through a local Codex CLI proxy instead of
MiniMax. The proxy is:

```text
capx/serving/codex_cli_server.py
```

It exposes `/chat/completions` and internally calls:

```bash
codex -c 'model_provider="axonhub"' exec ...
```

Output:

```text
outputs/codex-a/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Generated code:

```python
RESULT = pick_and_place_red_cube_to_right_target()
execution = RESULT.get("execution", {})
before_close_error = execution.get("before_close_error", {})
place = execution.get("place", {})
print("X2_TWO_TARGET_RESULT "
      + "ok=" + str(bool(RESULT.get("ok"))) + " "
      + "target=" + str(RESULT.get("target_name")) + " "
      + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
      + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
      + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
      + "place_error_m=" + str(place.get("place_error_m")))
```

Metrics:

```text
X2_TWO_TARGET_RESULT ok=True target=right
before_close_tcp_error_m=0.018581159238777945
before_close_ori_error_rad=0.06518267996169509
object_in_hand_after_close=True
place_error_m=0.022110423495201145
Reward=1.0
Task Completed=True
```

Videos:

```text
video_combined_global.mp4
video_combined_robot.mp4
```

Extra visual-chain artifacts for this run:

```text
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/
```

This directory contains the original RGB frame, OWL-ViT detection overlay,
SAM2 mask, SAM2 mask overlay, and `grasp_summary.json` with the selected
`T_world_tcp` grasp target.

## Two-Target Left Variant

The left-target variant reuses the same tabletop scene, object, chest camera,
visual chain, and reduced X2 API. The only task-level difference is the target
marker and primitive:

```text
env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml
pick_and_place_red_cube_to_left_target()
```

The left task was validated in both oracle and codex-a non-oracle modes.

Oracle output:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Oracle metrics:

```text
X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.011372143517779223
before_close_ori_error_rad=0.04381623598375953
object_in_hand_after_close=True
place_error_m=0.010966659369626599
Reward=1.0
Task Completed=True
```

Codex-a non-oracle output:

```text
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Generated code:

```python
RESULT = pick_and_place_red_cube_to_left_target()
execution = RESULT.get("execution", {})
before_close_error = execution.get("before_close_error", {})
place = execution.get("place", {})
attempts = RESULT.get("attempts", [])
print("X2_TWO_TARGET_RESULT "
      + "ok=" + str(bool(RESULT.get("ok"))) + " "
      + "target=" + str(RESULT.get("target_name")) + " "
      + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
      + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
      + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
      + "place_error_m=" + str(place.get("place_error_m")))
```

Codex-a metrics:

```text
X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.013898993392196681
before_close_ori_error_rad=0.040283250155416964
object_in_hand_after_close=True
place_error_m=0.020876950973317487
Reward=1.0
Task Completed=True
```

Videos:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Visual artifacts:

```text
outputs/x2_visual_artifacts/two_targets_left_oracle_smoke_v3/x2_pick_place_red_cube_20260629_100323_273/
outputs/x2_visual_artifacts/two_targets_left_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_100830_465/
```

## Two-Target Oracle Stability Check

The two-target oracle stability script is:

```bash
scripts/run_x2_two_target_oracle_stability_smoke.sh
```

`REPEATS=1 RUN_RIGHT=1 RUN_LEFT=1` found that the right task remained stable,
but one left run failed before gripper close even though the visual chain had
produced a plausible `T_world_tcp` grasp. The failure concentrated in final TCP
tracking, not in detection or segmentation:

```text
Failed left output:
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1022b_run01/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/

Best attempt:
before_close_tcp_error_m=0.043477231616007576
before_close_ori_error_rad=0.1602706117350232
object_in_hand_after_close=False
```

The current baseline therefore includes slower insertion/fine-align execution:

```text
insertion max_steps=420
fine-align max_steps=560
fine-align hold_steps_per_waypoint>=8
fine-align settle_steps>=20
task insert_hold_steps_per_waypoint=8
```

After this change, the left oracle stability run passed:

```text
Output:
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1330_left_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.019172631962380303
before_close_ori_error_rad=0.05583956751857787
object_in_hand_after_close=True
place_error_m=0.016582604022264174
Reward=1.0
Task Completed=True
```

Videos:

```text
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1330_left_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1330_left_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

The current source/config/script/doc set was snapshotted at:

```text
snapshots/x2_two_target_stability_hold_baseline_20260629_1330/
```

Follow-up right/left oracle stability after this snapshot:

```text
Right output:
outputs/oracle/stability/oracle/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=right
before_close_tcp_error_m=0.0022920335900174634
before_close_ori_error_rad=0.007841036302440513
object_in_hand_after_close=True
place_error_m=0.007655968983878265

Left output:
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.011122489380341058
before_close_ori_error_rad=0.03634797048888362
object_in_hand_after_close=True
place_error_m=0.013434781081554163
```

The codex-a stability loop should be run from a normal terminal when the GPT
API key is configured in the local `codex-a` command. It is not run from the
managed assistant tool environment because it sends workspace/task prompt data
to an external provider.

## Non-Oracle Paths

The recommended non-oracle path for this workstation is now the local codex-a
script:

```bash
REPEATS=1 \
STAMP=<STAMP> \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

If your local executable is named `codex` instead of `codex-a`, use:

```bash
CODEX_BIN=codex REPEATS=1 STAMP=<STAMP> scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

The generic OpenAI-compatible API scripts remain available for an explicit
local proxy:

```bash
scripts/run_x2_two_target_api_non_oracle_smoke.sh
scripts/run_x2_two_target_api_stability_smoke.sh
scripts/run_x2_two_target_api_stability_and_check.sh
```

These scripts do not start `capx/serving/codex_cli_server.py`; they call the
configured `SERVER_URL` directly:

```text
MODEL=gpt-5
SERVER_URL=http://127.0.0.1:8110/chat/completions
```

If using a local OpenAI-compatible proxy, start it with:

```bash
env ALL_PROXY=http://127.0.0.1:7897/ all_proxy=http://127.0.0.1:7897/ \
/home/xingshu/miniforge3/bin/conda run --no-capture-output -n behavior \
python capx/serving/openrouter_server.py \
  --key-file .openai_key \
  --base-url https://api.openai.com/v1/ \
  --host 127.0.0.1 \
  --port 8110
```

The `ALL_PROXY` override is needed on this workstation when the shell has
`ALL_PROXY=socks://127.0.0.1:7897/`; the current `httpx` dependency used by the
proxy rejects that scheme at startup.

Run right/left once each:

```bash
REPEATS=1 \
STAMP=<STAMP> \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_smoke.sh
```

Or run right/left, summarize, and check acceptance in one command:

```bash
REPEATS=1 \
STAMP=<STAMP> \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_and_check.sh
```

Then summarize the saved outputs:

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

Check the saved outputs against the current acceptance gate:

```bash
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

Audit the full local integration state:

```bash
python scripts/audit_x2_capx_integration.py
```

This summary script reads local files only. It reports generated code,
task/reward status, before-close TCP/orientation errors, object-in-hand state,
place error, videos, and visual artifact directories.

In the managed Codex tool environment, this direct API run cannot be executed
by the assistant because it sends CaP-X task/API prompt text to an external LLM
provider. That is a tooling policy limitation, not an X2/CaP-X code-path
failure. Run the command from a normal terminal, then use the summary script and
saved artifacts for acceptance.

## Current Limits

- Only right arm pick-place is validated.
- The task is a simple red cube on a low tabletop.
- Obstacle boxes are sim-known, not perceived.
- The table/cube geometry in PyRoKi is simplified as boxes.
- The visual chain is used for grasp pose, but not yet for obstacle geometry.
- The accepted success threshold is task-level simulation success, not a
  precision manipulation benchmark.

## Next Work

Reasonable next steps are:

1. Keep the current baseline unchanged unless a regression is intentional.
2. Improve transfer trajectory quality while preserving the accepted baseline in
   `snapshots/`.
3. Replace sim-known table/object obstacle boxes with depth/vision-derived
   obstacle geometry.
4. Split `execute_tcp_grasp_plan()` into smaller reusable actions if future
   tasks need custom post-grasp behavior.
5. Add more X2 task configs after the red-cube task remains repeatable.
