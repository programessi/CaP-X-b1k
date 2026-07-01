# X2 RGB-D Visual Obstacle Upgrade

Date: 2026-06-30

This note records the experimental upgrade from the accepted X2 simulation
baseline toward a later real-world visual pick-place route. The accepted
sim-known route is preserved as the default.

## What Changed

`pick_and_place_visual_object()` now has explicit route switches:

```python
pick_and_place_visual_object(
    object_name,
    place_position,
    ...,
    obstacle_source=None,
    use_sim_known_obstacles=True,
    place_offset_source="after_close_sim_known",
    sim_place_correction_steps=None,
    grasp_tcp_axis_offsets_m=None,
    reobserve_at_precontact=False,
    place_descent_waypoints=1,
)
```

Default behavior remains the accepted simulation baseline:

```text
obstacle_source=None -> "sim_known" when use_sim_known_obstacles=True
place_offset_source="after_close_sim_known"
sim_place_correction_steps=None -> 2
reobserve_at_precontact=False
place_descent_waypoints=1
```

The new experimental route is:

```python
RESULT = pick_and_place_visual_object(
    "x2_pick_place_blue_cube",
    [0.37, 0.055, 0.921],
    prompts=["blue cube", "blue block", "blue box"],
    table_name="x2_pick_place_table",
    obstacle_source="rgbd_visual",
    place_offset_source="visual_grasp_pose",
    sim_place_correction_steps=0,
    candidate_indices=(1, 2),
    grasp_tcp_axis_offsets_m=(0.0, 0.004, 0.008, 0.012),
    reobserve_at_precontact=True,
    reobserve_distance_m=0.08,
    reobserve_max_object_shift_m=0.025,
    reobserve_max_grasp_shift_m=0.035,
    place_descent_waypoints=4,
    place_descent_max_joint_step=0.006,
    place_descent_hold_steps_per_waypoint=8,
    place_pre_release_settle_steps=16,
)
```

## Coordinate Contract

The contract is unchanged:

```text
visual selected grasp target: T_world_tcp
action target input:          T_world_tcp
place target:                 world-frame object center [x, y, z]
obstacle boxes:               world-frame box center/extent
```

The visual plan stores camera intrinsics, camera pose, SAM2 mask, depth image,
visual object pose estimate, selected `grasp_tcp_pose`, `precontact_tcp_pose`,
and insertion waypoints. The action primitive consumes the selected TCP poses
directly.

## Pick Precontact Reobserve

The upgraded experimental route is not continuous visual servoing. It performs
one half-closed-loop correction at the precontact pose:

```text
initial RGB-D visual grasp plan
-> move to precontact TCP pose
-> settle
-> run OWL-ViT / SAM2 / GraspNet once more from the same task API
-> quality-gate the new plan
-> adopt the new final grasp/insertion plan only if all gates pass
-> otherwise keep the initial plan
-> slow short-range insertion
-> close gripper
```

The quality gates currently recorded under
`execution["precontact_reobserve"]` are:

```text
mask_pixels >= reobserve_min_mask_pixels
depth_points >= reobserve_min_depth_points
object center shift <= reobserve_max_object_shift_m
grasp TCP shift <= reobserve_max_grasp_shift_m
precontact TCP shift <= reobserve_max_precontact_shift_m
one-shot IK FK position/orientation errors within thresholds
```

This protects against the main failure mode near precontact: the wrist/hand can
partially occlude the target. If the second observation is weak or inconsistent,
the executor sets `adopted=False`, records `reason="quality_gate_failed"` or a
more specific failure reason, and falls back to the initial visual grasp plan.

The run artifact `pick_place_result_summary.json` stores both the initial plan
summary and the active plan summary used for insertion/close.

## RGB-D Obstacle Estimation

`get_rgbd_visual_tabletop_obstacles(visual_grasp_plan, ...)` converts the
visual plan into PyRoKi-style world-frame box obstacles.

Object obstacle:

```text
SAM2 target mask + depth
-> backproject masked pixels to world points
-> filter by optional workspace bounds
-> robust AABB from percentiles
-> center at visual pose_estimate.position_world
-> source="rgbd_object_mask_aabb"
```

Table obstacle:

```text
valid depth pixels outside target mask
-> backproject to world points
-> filter by workspace bounds
-> keep high support points below the object
-> estimate tabletop z and XY AABB
-> source="rgbd_table_plane_aabb"
```

This function is intentionally not allowed to read `env.scene`,
`object_registry`, object AABBs, or table AABBs. Unit tests use a dummy env
that raises if `env.scene` is accessed.

Saved artifacts under the visual artifact directory:

```text
rgb.png
sam2_mask.png
sam2_mask_overlay.png
detection_overlay.png
grasp_summary.json
visual_obstacles.json
pick_place_result_summary.json
object_obstacle_points_world.npy
table_obstacle_points_world.npy
```

## Placement Offset

The accepted baseline computes the release TCP offset from the sim-known object
pose after gripper close:

```text
tcp_from_object = current_tcp_after_close - sim_object_position_after_close
```

The experimental route computes it from the grasp-time visual estimate:

```text
tcp_from_object = tcp_after_close_position - visual_pose_estimate.position_world
release_tcp_position = target_object_position_world + tcp_from_object
```

When `place_offset_source="visual_grasp_pose"`,
`read_after_close_sim_object_pose` is passed as `False` to
`execute_tcp_grasp_plan()`.

## Close-Time TCP Axis Sweep

The executor now accepts:

```python
grasp_tcp_axis_offsets_m=(0.0, 0.004, 0.008, 0.012)
```

The default is `None`, which preserves the accepted baseline behavior: one
close at the selected `grasp_tcp_pose`. The RGB-D experimental task passes the
explicit sweep above. Each offset means:

```text
close_target_tcp = grasp_tcp_pose.position + tcp_axis_world * offset
```

Positive offsets move a few millimeters deeper along the visual approach axis
before closing. This is still perception-frame control; it does not read
sim-known object or table geometry. The execution result records
`preclose_axis_offsets_m`, `preclose_attempts`, and `final_close_attempt`.

## Slow Place Descent

The place leg now has an explicit high-transfer and vertical-descent structure:

```text
lift from grasp
-> move up to a safe high transfer z
-> lateral high waypoints above the target
-> settle at place pre-descent pose
-> descend through place_descent_waypoints vertical TCP waypoints
-> optional pre-release settle
-> open gripper
-> retreat upward
```

`place_descent_waypoints=1` keeps the previous single descent behavior. The
experimental RGB-D route uses four descent waypoints, a smaller joint step cap,
extra hold steps per waypoint, and a pre-release settle. This is intended to
reduce tabletop dragging when the held cube contacts the support surface.

## Remaining Sim-Only Parts

The experimental route removes sim-known obstacle boxes and the after-close
sim-known placement offset from the control input path. These parts are still
simulation-only:

```text
check_object_in_hand()
reward / task_completed
place_error_m metric after release
object_after_release metric
```

Those remaining reads are used as evaluation evidence in simulation, not as
the experimental obstacle or placement-offset control input.

## New Smoke Entry Point

Oracle smoke for the experimental route:

```bash
scripts/run_x2_two_object_blue_right_rgbd_visual_oracle_smoke.sh
```

Codex-a non-oracle smoke for the experimental route:

```bash
scripts/run_x2_two_object_blue_right_rgbd_visual_codex_a_non_oracle_smoke.sh
```

Config:

```text
env_configs/x2/x2_pick_place_two_objects_blue_right_rgbd_visual.yaml
```

The original accepted two-object config remains:

```text
env_configs/x2/x2_pick_place_two_objects_blue_right.yaml
```

## Verification

Lightweight checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xingshu/miniforge3/envs/behavior/bin/python -m pytest \
  tests/test_x2_rgbd_visual_obstacles.py \
  tests/test_x2_llm_api.py \
  -q

/home/xingshu/miniforge3/envs/behavior/bin/python -m py_compile \
  capx/integrations/x2/control.py \
  tests/test_x2_rgbd_visual_obstacles.py \
  tests/test_x2_llm_api.py
```

Full smoke should be summarized with:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/summarize_x2_runs.py \
  outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023 \
  --repo-root .
```

Latest successful smoke:

```text
trial:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

videos:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4

visual artifacts:
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023/x2_pick_place_blue_cube_20260630_173244_130/

metrics:
ok=True
reward=1.000
task_completed=1
candidate 0: before_close_reached=False, tcp_error=0.043857 m
candidate 1: before_close_reached=True, tcp_error=0.007958 m, ori_error=0.022499 rad
object_in_hand_after_close=True
final_close_axis_offset_m=0.0
place_error_m=0.025959 m
```

RGB-D obstacle evidence from this run:

```text
sim_truth=False
object obstacle source=rgbd_object_mask_aabb, points=8206
table obstacle source=rgbd_table_plane_aabb, table plane points=44596
```

This run predates the precontact reobserve / slow descent upgrade. The first
attempt at the upgraded route tried candidates `(0, 1, 2, 3, 4, 5)` and timed
out at the runner's 1000 s per-trial limit. The accepted upgraded route now
uses the previously validated candidates `(1, 2)` and four slow-descent
waypoints.

Latest upgraded successful smoke:

```text
trial:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

videos:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4

visual artifacts:
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/x2_pick_place_blue_cube_20260630_183817_658/
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/x2_pick_place_blue_cube_20260630_183928_672/

metrics:
ok=True
reward=1.000
task_completed=1
candidate_index=1
reobserve_adopted=True
reobserve_reason=quality_gates_passed
reobserve mask_pixels=8215
reobserve depth_points=8210
reobserve object_shift_m=0.000014924572561476803
reobserve grasp_shift_m=0.004602285628847467
before_close_tcp_error_m=0.011525492473389909
before_close_ori_error_rad=0.03224944203486641
final_close_axis_offset_m=0.0
object_in_hand_after_close=True
place_error_m=0.02411113666043054
place_descent_waypoints=4
trial_time_s=245.88
rgbd_obstacles_sim_truth=False
object obstacle source=rgbd_object_mask_aabb, points=8208
table obstacle source=rgbd_table_plane_aabb, table plane points=44592
```

Latest codex-a non-oracle smoke:

```text
trial:
outputs/codex-a/codex-a/x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1

videos:
outputs/codex-a/codex-a/x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/codex-a/codex-a/x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4

visual artifacts:
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/x2_pick_place_blue_cube_20260701_115434_152/
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/x2_pick_place_blue_cube_20260701_115611_311/

generated code:
RESULT = pick_and_place_visual_object(...)

metrics:
ok=True
reward=1.000
task_completed=1
candidate_index=1
obstacle_source=rgbd_visual
place_offset_source=visual_grasp_pose
reobserve_adopted=True
reobserve_reason=quality_gates_passed
before_close_tcp_error_m=0.010910568687270814
before_close_ori_error_rad=0.026704567279301608
final_close_axis_offset_m=0.004
object_in_hand_after_close=True
place_error_m=0.019196923124027467
place_descent_waypoints=4
rgbd_obstacles_sim_truth=False
object obstacle source=rgbd_object_mask_aabb, points=8207
table obstacle source=rgbd_table_plane_aabb, table plane points=44596
```
