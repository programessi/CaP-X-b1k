# X2 LLM-Facing Primitives

These are the primitives that should be shown to an LLM for the current
short-term X2 CaP-X tabletop simulation phase.

The accepted pick-place task configs use `X2PickPlaceApi`, a reduced wrapper
around `X2ControlApi`. It intentionally exposes only task-level pick-place
primitives plus one reusable visual pick-place primitive. Use the full
`X2VisualGraspApi` / `X2ControlApi` only for debugging lower-level visual or
motion contracts.

## Coordinate Contract

```text
Visual grasp output: T_world_tcp
Action target input: T_world_tcp
Place target:        world-frame object center position [x, y, z]
```

Do not execute raw GraspNet poses directly. X2 APIs adapt raw GraspNet poses
into X2 TCP targets before motion. If EEF targets are needed internally, the X2
API converts TCP to EEF.

## Task-Level Primitive

For the current single-target red-cube task, expose this narrow primitive to
the LLM:

```python
pick_and_place_red_cube(place_position=None, place_position_threshold=0.10)
```

Use it for `env_configs/x2/x2_pick_place_red_cube.yaml`. It fixes the task
object, table, chest camera, prompts, validated TCP orientation prior,
workspace bounds, sim-known table/cube obstacles, elevated placement
correction, and a short ranked GraspNet candidate retry list. It also caches
the first result, so accidental duplicate LLM calls in one generated program do
not command a second robot execution.

The reusable visual pick-place primitive remains:

```python
pick_and_place_visual_object(
    object_name,
    place_position,
    prompts=None,
    camera_name=None,
    arm=1,
    table_name=None,
    orientation_quat_xyzw=None,
    workspace_bounds=None,
    candidate_indices=None,
    place_position_threshold=0.10,
    use_sim_known_obstacles=True,
    sim_place_correction_steps=2,
)
```

Use this for simple tabletop pick-place tasks. It runs:

```text
settle/open gripper
-> OWL-ViT detection
-> SAM2 segmentation
-> GraspNet grasp generation
-> X2 TCP grasp selection
-> optional ranked candidate retry
-> optional sim-known table/object obstacle boxes
-> PyRoKi precontact motion
-> TCP insertion
-> close gripper
-> abort place leg if the gripper is empty after close
-> lift and transfer through explicit high TCP waypoints
-> optional sim-known object-center correction above the place target
-> mostly vertical descent to the release pose
-> release
```

Returns a dict with `ok`, `plan`, `obstacles_world`, and `execution`.

For the two-target task, expose these narrow target-selection primitives:

```python
pick_and_place_red_cube_to_left_target(place_position_threshold=0.10, force=False)
pick_and_place_red_cube_to_right_target(place_position_threshold=0.10, force=False)
```

Use them for `env_configs/x2/x2_pick_place_red_cube_two_targets.yaml`. The
current prompt asks for the right target, so generated code should call exactly
one `pick_and_place_red_cube_to_right_target()`.

The validated two-target scene keeps the red cube at the single-target initial
position and places the target markers in the forward placement region:

```text
red cube:      [0.32, -0.08, 0.921]
left target:   [0.27,  0.055, 0.921]
right target:  [0.37,  0.055, 0.921]
```

Do not place the target markers immediately beside the cube for this smoke
task. A previous layout with markers on the cube's `y=-0.08` line destabilized
the visual grasp and failed before gripper close.

## Reduced API Surface

`X2PickPlaceApi.functions()` is the formal function set injected into ordinary
X2 pick-place task code:

```python
pick_and_place_red_cube(...)
pick_and_place_red_cube_to_left_target(...)
pick_and_place_red_cube_to_right_target(...)
pick_and_place_visual_object(...)
```

The task prompts normally instruct the LLM to use the narrow task primitive.
`pick_and_place_visual_object()` is kept available for the next simple tabletop
tasks where the object/target names differ but the same visual-action contract
is valid.

## Prompt Contract For Generated Code

Show the LLM a short API list and task instruction. Do not ask it to build the
scene, reset the simulator, create cameras, import OmniGibson, or instantiate
robots. CaP-X / BEHAVIOR owns task setup; generated code should only call the
injected primitives.

For the right-target two-target task, the intended generated program shape is:

```python
RESULT = pick_and_place_red_cube_to_right_target()
```

For the left-target variant:

```python
RESULT = pick_and_place_red_cube_to_left_target()
```

For debugging and evaluation, the prompt/oracle code may print metrics from the
returned dict:

```python
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

Important return fields:

```text
RESULT["ok"]                       task-level success
RESULT["target_name"]              "left" or "right" for two-target tasks
RESULT["plan"]["grasp_tcp_pose"]   selected T_world_tcp grasp target
RESULT["execution"]                motion/grasp/place diagnostics
RESULT["attempts"]                 ranked GraspNet candidate attempts

execution["before_close_error"]["tcp_error_m"]
execution["before_close_error"]["ori_error_rad"]
execution["before_close_reached"]
execution["object_in_hand_after_close"]
execution["place"]["place_error_m"]
```

Forbidden in ordinary generated X2 task code:

```text
Do not call env.reset().
Do not create or move cameras.
Do not instantiate X2ControlApi manually.
Do not import OmniGibson, PyRoKi, SAM2, OWL-ViT, or Contact-GraspNet.
Do not directly command raw joint positions unless the task explicitly exposes a lower-level debug API.
Do not convert TCP to EEF manually; the X2 API handles this internally.
```

## Lower-Level Visual Primitive

```python
plan_visual_grasp_tcp_pose(object_name, prompts=None, camera_name=None, arm=1, ...)
```

Runs the visual chain only. It does not move the robot. On success it returns
`grasp_tcp_pose`, `precontact_tcp_pose`, and `insertion_waypoints`, all as
`T_world_tcp`.

## Lower-Level Action Primitive

```python
execute_tcp_grasp_plan(plan, arm=1, place_position=None, obstacles_world=None, ...)
```

Consumes the plan from `plan_visual_grasp_tcp_pose()`. It opens the gripper,
moves to precontact, inserts to the grasp TCP pose, closes the gripper, and can
optionally lift/place/release the object.

`ok=True` means the before-close TCP target was reached and, if
`place_position` is provided, the object was detected in hand after closing and
the released object center is within `place_position_threshold`.

When `skip_place_if_no_object_in_hand=True`, the action primitive opens the
gripper and retreats to precontact instead of continuing an empty-handed place
leg. The task-level visual wrapper uses this behavior before trying another
ranked GraspNet candidate.

For the held-object transfer leg, the current task primitive does not give
PyRoKi one unconstrained goal from grasp to place prepose. It inserts explicit
high TCP waypoints and tracks them with joint-IK moves, then descends mostly
vertically for release. This keeps the path close to the tabletop work area and
avoids the previously observed large body-side arc.

## Sim-Only Helper

```python
get_sim_known_tabletop_obstacles(object_name, table_name)
```

Builds inflated PyRoKi box obstacles from current OmniGibson scene AABBs for
the object and table. This is for short-term simulation integration only. It is
not a 2real perception primitive.

`sim_place_correction_steps` in `pick_and_place_visual_object()` /
`execute_tcp_grasp_plan()` is also sim-only because it reads the sim-known
object center before release. The current pick-place baseline applies this
correction at the elevated pre-release pose above the target, then descends
mostly vertically to avoid dragging the cube across the tabletop.

## Debug Motion Helpers

```python
get_chest_camera_name()
open_gripper(arm=1)
close_gripper(arm=1)
settle_robot(steps=...)
get_current_tcp_pose(arm=1)
move_tcp_joint_ik(tcp_pose, arm=1, ...)
move_tcp_pyroki_trajopt(tcp_pose, arm=1, obstacles_world=None, ...)
```

These are not exposed by `X2PickPlaceApi`. They remain available through the
full `X2ControlApi` / `X2VisualGraspApi` for lower-level debugging or custom
motion experiments. For ordinary pick-place tasks, prefer the reduced API
above.
