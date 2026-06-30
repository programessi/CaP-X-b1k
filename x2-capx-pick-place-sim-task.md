# X2 CaP-X Pick-Place Simulation Task

This is the short-term X2 CaP-X integration path. It validates that the X2
visual grasp primitives and motion primitives can be used from injected CaP-X
code in an existing BEHAVIOR environment.

## Task

Config:

```text
env_configs/x2/x2_pick_place_red_cube.yaml
```

The config exposes `X2PickPlaceApi` to generated code. This is the reduced
LLM-facing X2 API for this task; it wraps the visual chain, PyRoKi precontact
motion, TCP insertion, grasp check, lift, transfer, elevated placement
correction, vertical descent, and release.

The environment creates:

- fixed-base X2 with right arm enabled for the task
- chest RGB-D camera
- low tabletop
- movable red cube
- visual target marker for placement
- video recording

Injected code should not create or reset the simulator. It should call the
injected APIs only.

## Primitive Flow

The default task-level sequence is:

```python
RESULT = pick_and_place_red_cube()
```

`pick_and_place_red_cube()` is the narrow LLM-facing primitive for this task.
It wraps the reusable `pick_and_place_visual_object()` with the validated red
cube task parameters: right arm, chest camera, object/table names, prompts,
TCP orientation prior, workspace bounds, sim-known obstacles, four elevated
placement correction steps, and ranked candidate indices `(0, 1, 2)`.

Internally, it runs the visual chain:

```text
chest RGB-D -> OWL-ViT -> SAM2 -> GraspNet raw 6D grasp
-> X2 TCP adapter -> selected T_world_tcp grasp plan
```

and then calls `execute_tcp_grasp_plan()` to execute:

```text
open gripper
-> PyRoKi move to precontact
-> slow TCP insertion to grasp
-> before-close TCP error check
-> close gripper
-> if empty after close: open, retreat to precontact, optionally try next candidate
-> lift
-> move through explicit high TCP waypoints to the place prepose
-> descend to place pose
-> optional sim-known object-center correction before release
-> open gripper
-> retreat
```

## Coordinate Contract

The visual/action contract is:

```text
visual final grasp target: T_world_tcp
action primitive target:   T_world_tcp
```

Raw GraspNet poses are not action targets. They must first be adapted:

```text
T_world_x2_tcp = T_world_graspnet_raw @ T_graspnet_raw_to_x2_tcp
```

If the controller needs an EEF target, X2 APIs convert TCP to EEF internally.

## Obstacle Source

This task intentionally uses a short-term simulation obstacle source:

```text
get_sim_known_tabletop_obstacles(...)
```

It reads the current OmniGibson scene AABB for the cube and table, inflates
those boxes by engineering margins, and passes them to PyRoKi. This is useful
for validating CaP-X task/API integration, but it is not a 2real perception
primitive.

The visual grasp pose is produced by the visual chain. The collision boxes are
currently sim-known.

## Success Criteria

The smoke run should record:

- selected visual `grasp_tcp_pose`
- before-close TCP position/orientation error
- object state after close
- release/place target
- final object pose
- place error
- video

Expected short-term target:

- before-close TCP position error around 2 cm or lower
- cube is movable and follows the gripper after close
- cube is released within the task success radius near the target marker
- no obvious early table/cube hard collision before grasp

For `execute_tcp_grasp_plan()`, `ok=True` is task-level success:

```text
before-close TCP pose reached within thresholds
and, when place_position is requested, the object is detected in hand after close
and the released object is near that target
```

The success radius is explicit: `place_position_threshold` in
`execute_tcp_grasp_plan()` and `SUCCESS_THRESHOLD_M` in the task. The current
short-term CaP-X integration task uses `0.10 m`; this is intentionally a
coarse simulation success radius for validating API/task integration, not a
precision placement claim.

The oracle enables two sim-only pre-release correction steps:
`place_object_correction_steps=2`. The API now applies this correction at the
elevated pre-release pose above the target, then descends mostly vertically to
release. This avoids doing low-height horizontal corrections that can drag the
cube across the table. The correction reads the sim-known object center, so it
is not a 2real primitive; it is a short-term simulation stabilizer for the
pick-place integration task.

The non-oracle task API wrapper uses four elevated correction steps. It also
enables empty-gripper place skipping, so a failed grasp does not continue as an
empty-handed place/push, and can try the next ranked GraspNet candidate.

The transfer from grasp to place prepose uses explicit high TCP waypoints
instead of a single free PyRoKi transfer target. This keeps the held cube above
the tabletop and close to the direct tabletop route, avoiding the previous
large arc around the robot body.

The per-segment controller return flags are still reported for diagnosis:
`lift_ok`, `place_pre_ok`, `place_insert_ok`, and
`place_motion_steps_ok`. These are useful for tuning tracking, but a segment
flag can be false even when the final object placement succeeds.

Accepted CaP-X smoke result:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.012187954567831028
before_close_ori_error_rad=0.0333455628915337
object_in_hand_after_close=True
place_error_m=0.04355407559125816
Reward=1.0
Task Completed=True
```

This is the accepted 2026-06-26 13:37 Asia/Shanghai run. The video showed an
acceptable pick-place trajectory after moving the sim-only object-center
correction to the elevated pre-release pose.

Known false-positive historical CaP-X smoke result:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.0349
before_close_ori_error_rad=0.1025
object_in_hand_after_close=False
place_error_m=0.0920
Reward=1.0
Task Completed=True
```

This historical result used the task-level `pick_and_place_visual_object()`
wrapper from the CaP-X oracle code. The video showed this was not a real
pick-place success. The old task-level reward only checked whether the final
cube center was within `0.10 m` of the target, so it could mark a push or near
miss as successful. The task and primitive success criteria now require
`object_in_hand_after_close=True` for pick-place success.

The simulator still reports a known Isaac/OmniGibson teardown segfault after
summary and videos are saved. Treat the smoke result as valid when the summary
and video files have already been written.

Videos:

```text
outputs/oracle/x2_pick_place_red_cube/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/x2_pick_place_red_cube/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
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

This is the 2026-06-26 15:09 Asia/Shanghai run using
`pick_and_place_red_cube()` from LLM-generated code with explicit high TCP
waypoint transfer. Videos:

```text
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_waypoint_transfer_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_waypoint_transfer_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

## Current Limits

- Only right arm `arm=1` is targeted.
- Only a simple tabletop red-cube pick-place is covered.
- PyRoKi uses simplified box obstacles, not a full scene collision model.
- Sim-known obstacles are acceptable for this integration step and must be
  replaced by depth/vision-derived obstacles before claiming 2real readiness.
