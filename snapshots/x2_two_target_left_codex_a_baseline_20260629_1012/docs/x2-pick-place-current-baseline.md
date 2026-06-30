# X2 Pick-Place Current Baseline

Date: 2026-06-26

This document records the current X2 CaP-X tabletop pick-place baseline. The
purpose is version management: keep the working path stable while future changes
improve trajectory quality, perception-based obstacles, and broader tasks.

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
