# X2 Visual Grasp to IK v11 Report

## Source Script

The validated demo script is:

```text
scripts/archive/x2_experiments/x2_visual_pyroki_precontact_insert_grasp_demo.py
```

Absolute path:

```text
/home/xingshu/workspaces/fys/CaP-X-b1k/scripts/archive/x2_experiments/x2_visual_pyroki_precontact_insert_grasp_demo.py
```

The successful run output is:

```text
outputs/x2_visual_pyroki_precontact_insert_grasp_v11_adapted_proxy_guarded_selection
```

Important artifacts:

```text
summary.json
video_combined.mp4
robot.mp4
global.mp4
camera/chest_rgb.png
camera/chest_sam2_mask_overlay.png
```

## What v11 Validates

v11 validates the full visual-grasp-to-motion path in the current X2 BEHAVIOR environment:

1. Use the X2 chest camera RGB-D observation.
2. Detect the red cube with OWL-ViT.
3. Segment the object with SAM2.
4. Estimate the object pose from the SAM2 mask and RGB-D depth.
5. Run Contact-GraspNet on the masked depth.
6. Take raw GraspNet 6D grasp candidates.
7. Convert each raw GraspNet pose into an X2 executable TCP pose.
8. Select a candidate using X2-specific geometry and gripper proxy guard checks.
9. Use PyRoKi trajectory optimization to move to precontact.
10. Use joint IK waypoints to insert along the TCP approach axis.
11. Close the gripper after reaching the selected TCP grasp pose.

This is not the old shortcut that used only the visual object center plus a fixed hand orientation. In v11, both the target grasp position and target grasp orientation come from GraspNet raw 6D output after applying the X2 TCP adapter.

## Coordinate Contract

The visual and action contract is:

```text
raw_graspnet_pose:       T_world_graspnet_raw
adapter:                 T_graspnet_raw_x2_tcp
adapted grasp target:    T_world_x2_tcp = T_world_graspnet_raw @ T_graspnet_raw_x2_tcp
action primitive input:  T_world_x2_tcp
internal action target:  T_world_eef = T_world_x2_tcp converted by tcp_pose_to_eef_pose()
```

The key point is that action primitives consume an X2 TCP pose in world frame. They do not directly consume the raw GraspNet frame.

The current adapter used by v11 is:

```text
position:  [0.003604, 0.002704, -0.059466]
quat_xyzw: [0.488425, 0.670197, 0.342387, -0.441642]
```

This adapter is an empirical frame adapter derived from previous X2 validation. It is good enough for the current cube task, but should eventually be calibrated from multiple grasp samples.

## v11 Result

v11 selected this adapted GraspNet TCP target:

```text
position:  [0.321686, -0.043247, 0.923089]
quat_xyzw: [0.487365, 0.67235, 0.347501, -0.435505]
```

Measured results:

```text
visual object position error: 0.001869 m
precontact TCP error:        0.078021 m
before-close TCP error:      0.014004 m
before-close orientation:    0.0455 rad
precontact contact count:    0
insertion contact events:    0
before-close contact count:  0
after-close contact count:   1
overall checks:              pass
```

The precontact error is not the final grasp accuracy. The arm recovered during the slow insertion waypoints. The important result is that before closing the gripper, the TCP was within about 1.4 cm of the selected visual grasp target and had no object contact before close.

## Why v10 Failed and v11 Passed

v10 selected a very centered and low adapted GraspNet candidate:

```text
position: [0.322006, -0.039381, 0.918465]
```

That produced one early contact at `insert_00`, specifically with the right inner finger. The selected target was close to the visual object center, but the open gripper geometry was not safe during the first insertion waypoint.

v11 added a lightweight X2 gripper proxy guard into candidate selection. This changed the chosen target to a slightly higher and safer candidate:

```text
position: [0.321686, -0.043247, 0.923089]
```

That candidate kept all insertion waypoints contact-free before closing.

## Demo Script vs CaP-X Injected Code

The demo script is still a self-contained validation script. It creates the env, creates the API, starts from a configured scene, records videos, writes summary files, and performs diagnostics.

CaP-X injected code should not do those things. Injected code should assume that the task env and API already exist, then call high-level primitives.

The reusable logic has now been moved into:

```text
capx/integrations/x2/control.py
```

New exposed API functions:

```text
get_current_tcp_pose()
adapt_graspnet_raw_pose_to_x2_tcp()
select_adapted_graspnet_tcp_candidate()
plan_visual_grasp_tcp_pose()
execute_tcp_grasp_plan()
```

The intended injected-code shape is:

```python
plan = plan_visual_grasp_tcp_pose(
    "red cube",
    camera_name=get_chest_camera_name(),
    external=False,
    orientation_quat_xyzw=validated_x2_grasp_quat,
)

execute_tcp_grasp_plan(plan, obstacles_world=task_obstacles)
```

In other words, the generated code no longer needs to create the env, read camera intrinsics manually, call OWL-ViT/SAM2/GraspNet directly, implement the GraspNet-to-X2 TCP adapter, or hand-rank raw candidates.

## Remaining Limits

The current grasp object was fixed, so `after-close contact_count=1` proves contact after close, not successful object lifting.

The adapter is empirical and should later be re-derived from multiple GraspNet/X2 pairs.

The proxy guard is a lightweight task-space guard, not full robot-world collision planning. It is useful for avoiding early finger/object contact, but it is not a replacement for full trajectory optimization with a complete collision model.
