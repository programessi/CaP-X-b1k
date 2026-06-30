# X2 Version Management Notes

## Frozen Working Path

As of the accepted CaP-X pick-place smoke test on 2026-06-26 13:37 Asia/Shanghai,
treat this path as the current X2 tabletop pick-place baseline:

```text
capx/integrations/x2/control.py
capx/envs/tasks/x2/x2_pick_place_red_cube.py
env_configs/x2/x2_pick_place_red_cube.yaml
scripts/x2_visual_pyroki_precontact_insert_grasp_demo.py
scripts/x2_visual_grasp_api_closed_loop_smoke.py
scripts/x2_injected_visual_grasp_api_example.py
docs/x2-visual-grasp-to-ik-v11-report.md
docs/x2-capx-pick-place-sim-task.md
docs/x2-llm-facing-primitives.md
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
before_close_tcp_error_m=0.02617899145823312
before_close_ori_error_rad=0.07640485198452023
object_in_hand_after_close=True
place_error_m=0.03806729894863623
Reward=1.0
Task Completed=True
```

The accepted non-oracle video is the 2026-06-26 14:39 run in:

```text
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_task_api_retry_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
```

The known OmniGibson / Isaac teardown segfault after videos and summary are
saved is not treated as a task failure.
