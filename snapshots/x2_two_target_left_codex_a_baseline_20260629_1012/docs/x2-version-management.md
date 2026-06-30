# X2 Version Management Notes

## Frozen Working Path

As of the accepted CaP-X pick-place smoke test on 2026-06-26 13:37 Asia/Shanghai,
treat this path as the current X2 tabletop pick-place baseline:

```text
capx/integrations/x2/control.py
capx/envs/tasks/x2/x2_pick_place_red_cube.py
capx/envs/tasks/x2/x2_pick_place_red_cube_two_targets.py
capx/serving/codex_cli_server.py
env_configs/x2/x2_pick_place_red_cube.yaml
env_configs/x2/x2_pick_place_red_cube_two_targets.yaml
scripts/x2_visual_pyroki_precontact_insert_grasp_demo.py
scripts/x2_visual_grasp_api_closed_loop_smoke.py
scripts/x2_injected_visual_grasp_api_example.py
docs/x2-visual-grasp-to-ik-v11-report.md
docs/x2-capx-pick-place-sim-task.md
docs/x2-llm-facing-primitives.md
docs/x2-pick-place-current-baseline.md
scripts/run_x2_pick_place_red_cube_non_oracle_smoke.sh
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

Do not casually refactor these files. Small fixes are acceptable, but changes should be followed by running:

```text
python capx/envs/launch.py --config-path env_configs/x2/x2_pick_place_red_cube.yaml --use-oracle-code True --record-video True --total-trials 1 --num-workers 1
```

## Active Helper Scripts

These scripts remain in `scripts/` because the working path imports them:

```text
scripts/x2_chest_camera_visual_chain_smoke.py
scripts/x2_chest_visual_grasp_to_joint_ik_demo.py
scripts/x2_code_exec_grasp_only_demo.py
scripts/x2_pyroki_precontact_insert_grasp_demo.py
scripts/x2_replay_visual_target_joint_tracking.py
scripts/x2_replay_visual_target_orientation_sweep.py
```

They are not the final API shape, but they contain shared test helpers and should not be moved until those helpers are migrated into proper package modules.

## Archived Scripts

Historical X2 experiments were moved to:

```text
scripts/archive/x2_experiments
```

They are kept for evidence and parameter history. They are not deleted, and they are not the recommended entry points.

## Output Folders

Current validated outputs:

```text
outputs/x2_visual_pyroki_precontact_insert_grasp_v11_adapted_proxy_guarded_selection
outputs/x2_visual_grasp_api_closed_loop_smoke_v1
outputs/oracle/x2_pick_place_red_cube/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_task_api_retry_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_waypoint_transfer_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
outputs/MiniMax-M2.7/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_baseline_hardened_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_marker_forward_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
outputs/codex-a/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950
```

Historical outputs were moved to:

```text
outputs/archive/x2_experiments
```

## Current Technical Contract

The current visual-to-action contract is:

```text
T_world_graspnet_raw @ T_graspnet_raw_x2_tcp = T_world_x2_tcp
```

The action layer consumes `T_world_x2_tcp`. It converts TCP to EEF internally through `tcp_pose_to_eef_pose()`.

The working API path is:

```text
X2PickPlaceApi
pick_and_place_red_cube()
pick_and_place_visual_object()
plan_visual_grasp_tcp_pose()
execute_tcp_grasp_plan()
```

`X2PickPlaceApi` is the reduced LLM-facing API used by the accepted task
config. It exposes the task-level primitive and a few safe helpers. The full
`X2VisualGraspApi` / `X2ControlApi` remains available for lower-level debugging
and experiments, but it should not be the default API for this simple task.

For `x2_pick_place_red_cube.yaml`, the LLM-facing entry point is
`pick_and_place_red_cube()`. It fixes the task-specific visual/action
parameters and internally calls the reusable visual pick-place primitive. It
also caches the first result in a generated program to protect against
accidental duplicate LLM calls.

The reusable visual pick-place path now has two task-stability guards:

```text
candidate_indices: ranked adapted GraspNet candidates can be tried in order
skip_place_if_no_object_in_hand: empty gripper aborts the place leg and retreats
```

The held-object transfer leg uses explicit high TCP waypoints instead of a
single unconstrained PyRoKi target from grasp to place prepose. This preserves
the successful pick-place behavior while avoiding the large body-side transfer
arc observed in the earlier non-oracle run.

The pre-waypoint working version was saved here before modifying the transfer
leg:

```text
snapshots/x2_pick_place_red_cube_working_20260626_1439
```

The current hardened non-oracle CaP-X baseline was saved here after validating
LLM code extraction, visual grasp, motion execution, videos, and metrics:

```text
snapshots/x2_pick_place_red_cube_capx_baseline_20260626_1530
```

The current two-target codex-a non-oracle baseline was saved here after
validating the local Codex CLI proxy, generated task code, visual grasp,
motion execution, videos, and visual artifacts:

```text
snapshots/x2_two_target_codex_a_baseline_20260629_0935
```

The current accepted task remains simple and fixed-base: red cube on the
chest-camera table, right arm, visual grasp, lift, transfer, and place. The
task-level success condition requires:

```text
before-close TCP pose reached
object_in_hand_after_close=True
released cube center within 0.10 m of the target
```

Accepted CaP-X smoke result:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.012187954567831028
before_close_ori_error_rad=0.0333455628915337
object_in_hand_after_close=True
place_error_m=0.04355407559125816
```

The accepted video is the 2026-06-26 13:37 run in:

```text
outputs/oracle/x2_pick_place_red_cube/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
```

Accepted non-oracle task API smoke result:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.016780745142452466
before_close_ori_error_rad=0.04858480025799528
object_in_hand_after_close=True
place_error_m=0.01853152439298318
Reward=1.0
Task Completed=True
```

The accepted non-oracle video is the 2026-06-26 15:09 run in:

```text
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_waypoint_transfer_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
```

Hardened non-oracle code-extraction smoke result:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.007781398923911473
before_close_ori_error_rad=0.02522619288209918
object_in_hand_after_close=True
place_error_m=0.014900085488091878
Reward=1.0
Task Completed=True
```

This run verified that raw MiniMax output containing `<think>...</think>` was
cleaned before execution. The saved CaP-X generated program contains only the
Python task call and metric printout.

Reduced `X2PickPlaceApi` oracle smoke result after narrowing the LLM-facing API
to task/composable pick-place primitives:

```text
Output:
outputs/oracle/oracle/x2_pick_place_red_cube_reduced_api_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

Executed code:
RESULT = pick_and_place_red_cube()

X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.009280033147572208
before_close_ori_error_rad=0.027969652238196776
object_in_hand_after_close=True
place_error_m=0.01605146254181371
Reward=1.0
Task Completed=True
```

This run verifies that the oracle path now uses the same reduced task primitive
shape expected from generated code. The videos are:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_reduced_api_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_red_cube_reduced_api_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

The known OmniGibson / Isaac teardown segfault after videos and summary are
saved is not treated as a task failure.

## Two-Target Task Smoke

The two-target task uses the same reduced API surface but asks generated code
to choose the right marker:

```python
RESULT = pick_and_place_red_cube_to_right_target()
```

The first two-target scene placed the left/right target markers at the same
`y=-0.08` line as the initial red cube. That made the visual target markers too
close to the object being grasped and produced unstable grasp execution:

```text
Output:
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_stable_y_oracle_smoke/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0

X2_TWO_TARGET_RESULT ok=False target=right
before_close_tcp_error_m=0.09472798472790968
before_close_ori_error_rad=0.4755532897831224
object_in_hand_after_close=False
place_error_m=None
Reward=0.0
Task Completed=False
```

The accepted two-target oracle smoke keeps the red cube at the validated
single-target initial position `[0.32, -0.08, 0.921]`, but moves the markers
forward to the placement region:

```text
left marker:  [0.27, 0.055, 0.921]
right marker: [0.37, 0.055, 0.921]
```

Accepted two-target oracle result:

```text
Output:
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_marker_forward_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

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
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_marker_forward_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_marker_forward_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Accepted two-target codex-a non-oracle result:

```text
Output:
outputs/codex-a/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

Generated code:
RESULT = pick_and_place_red_cube_to_right_target()

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
outputs/codex-a/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/codex-a/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Visual-chain artifacts:

```text
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/rgb.png
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/detection_overlay.png
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/sam2_mask.png
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/sam2_mask_overlay.png
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/grasp_summary.json
```

The visual summary recorded:

```text
mask_pixels=8241
graspnet_candidate_count=48
object_position_world=[0.3197035017668407, -0.07959770678727666, 0.9212843309825731]
selected_rank=0
selected_raw_index=12
grasp_tcp_pose.position=[0.3260178858659517, -0.08302930016842375, 0.9249030408645927]
grasp_tcp_pose.quat_xyzw=[0.5411638007994698, 0.6343949672102707, 0.4155622132503154, -0.3633081518505343]
```

### 2026-06-29 10:03/10:08 - Two-Target LEFT Variant Accepted

Purpose:

- Add a minimal third X2 CaP-X tabletop task that uses the same scene and
  reduced primitive API, but asks the generated program to place the red cube
  at the left target marker.
- Verify both oracle and codex-a non-oracle paths before treating the new
  config as part of the usable X2 task set.

Code/config changes:

```text
capx/envs/tasks/x2/x2_pick_place_red_cube_two_targets.py
env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml
capx/envs/tasks/__init__.py
tests/test_x2_llm_api.py
```

The right task remains registered as:

```text
x2_pick_place_red_cube_two_targets_code_env
```

The left task is registered as:

```text
x2_pick_place_red_cube_two_targets_left_code_env
```

The reduced LLM-facing API remains task-level only:

```text
pick_and_place_red_cube()
pick_and_place_red_cube_to_left_target()
pick_and_place_red_cube_to_right_target()
pick_and_place_visual_object()
```

Oracle LEFT result:

```text
Output:
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

Generated code:
RESULT = pick_and_place_red_cube_to_left_target()

X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.011372143517779223
before_close_ori_error_rad=0.04381623598375953
object_in_hand_after_close=True
place_error_m=0.010966659369626599
X2_TWO_TARGET_ATTEMPT candidate_index=0 ok=True
Reward=1.0
Task Completed=True
```

Oracle videos:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Oracle visual artifacts:

```text
outputs/x2_visual_artifacts/two_targets_left_oracle_smoke_v3/x2_pick_place_red_cube_20260629_100323_273/
```

Codex-a LEFT non-oracle result:

```text
Output:
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

Generated code:
RESULT = pick_and_place_red_cube_to_left_target()

X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.013898993392196681
before_close_ori_error_rad=0.040283250155416964
object_in_hand_after_close=True
place_error_m=0.020876950973317487
X2_TWO_TARGET_ATTEMPT candidate_index=0 ok=True
Reward=1.0
Task Completed=True
```

Codex-a videos:

```text
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Codex-a visual artifacts:

```text
outputs/x2_visual_artifacts/two_targets_left_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_100830_465/
```

Notes:

- An earlier left oracle attempt failed at grasp time with
  `before_close_tcp_error_m=0.032965690530027986`,
  `before_close_ori_error_rad=0.25520838300487997`, and
  `object_in_hand_after_close=False`.
- The task logging now prints one `X2_TWO_TARGET_ATTEMPT` line per grasp
  candidate, so future failures expose whether the issue is candidate
  selection, final TCP tracking, or post-close object-in-hand state.
- The default red-cube candidate retry set was widened from `(0, 1, 2)` to
  `(0, 1, 2, 3, 4, 5)`. The accepted oracle and codex-a left runs both
  succeeded on candidate `0`; the wider set is a robustness guard for
  stochastic visual grasp output.
