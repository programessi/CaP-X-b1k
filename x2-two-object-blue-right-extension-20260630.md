# X2 Two-Object Blue-Right Extension - 2026-06-30

This document records the first X2 object-selection extension after the
accepted red-cube two-target baseline.

## Goal

The scene contains two movable tabletop cubes:

- `x2_pick_place_blue_cube`: target object.
- `x2_pick_place_red_cube`: distractor object.

The task code must choose the blue cube through the generic visual primitive
and place it at the right target marker. The point of this task is to verify
that CaP-X code can use the generic X2 visual pick-place API instead of only
calling fixed red-cube helpers.

## Task And Config

```text
capx/envs/tasks/x2/x2_pick_place_two_objects.py
env_configs/x2/x2_pick_place_two_objects_blue_right.yaml
scripts/run_x2_two_object_blue_right_oracle_smoke.sh
scripts/run_x2_two_object_blue_right_codex_a_non_oracle_smoke.sh
```

The LLM-facing call is still a single high-level primitive:

```python
RESULT = pick_and_place_visual_object(
    "x2_pick_place_blue_cube",
    [0.370, 0.055, 0.921],
    prompts=["blue cube", "blue block", "blue box"],
    table_name="x2_pick_place_table",
    orientation_quat_xyzw=[
        0.53604597,
        0.63824742,
        0.39958203,
        -0.38161388,
    ],
    workspace_bounds={"x": (0.16, 0.50), "y": (-0.38, 0.12), "z": (0.78, 1.12)},
    candidate_indices=(0, 1, 2, 3, 4, 5),
    sim_place_correction_steps=4,
)
```

Generated code must not call lower-level camera, IK, PyRoKi, GraspNet,
gripper, or environment setup helpers directly.

## Pose Contract

The extension uses the same contract as the accepted X2 baseline:

```text
visual selected grasp target: T_world_tcp
action target input:          T_world_tcp
place target:                 world-frame object center [x, y, z]
```

The task-level API handles TCP-to-EEF conversion and joint trajectory
execution internally.

## Scene Notes

The successful two-object configuration keeps the blue target cube in the same
reachable grasp region validated by the accepted red-cube baseline:

```text
blue target object: [0.32, -0.08, 0.921]
red distractor:     [0.24, -0.09, 0.921]
right target:       [0.37,  0.055, 0.921]
```

An earlier run with the blue cube at `[0.34, -0.08, 0.921]` produced accurate
TCP tracking but did not create an in-hand grasp. That failure is useful
evidence that the issue was grasp/contact geometry at that object placement,
not visual pose estimation or IK tracking.

## Passing Oracle Evidence

```text
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_run03/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Metrics:

```text
X2_TWO_OBJECT_RESULT ok=True
object=x2_pick_place_blue_cube
target=right
before_close_tcp_error_m=0.008206982467097177
before_close_ori_error_rad=0.021700898689255475
object_in_hand_after_close=True
place_error_m=0.009616035120709184
Reward=1.0
Task Completed=True
```

Videos:

```text
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_run03/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_run03/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Visual artifacts:

```text
outputs/x2_visual_artifacts/two_objects_blue_right_oracle_run03/x2_pick_place_blue_cube_20260630_154432_908/
```

The visual estimate for the successful run selected the blue cube and produced
a world-frame object pose close to the configured target-object pose.

## Passing Non-Oracle Evidence

This run used the local Codex/GPT route through the OpenAI-compatible proxy,
not MiniMax. The model generated only one high-level API call and did not call
lower-level camera, IK, PyRoKi, GraspNet, gripper, or environment setup helpers.

```text
outputs/codex-a/x2_pick_place_two_objects_blue_right_codex_a_non_oracle_manual_codex_a_two_object_retry02_20260630_160157/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Generated code:

```python
RESULT = pick_and_place_visual_object(
    "x2_pick_place_blue_cube",
    [0.370, 0.055, 0.921],
    prompts=["blue cube", "blue block", "blue box"],
    arm=1,
    table_name="x2_pick_place_table",
    orientation_quat_xyzw=[
        0.53604597,
        0.63824742,
        0.39958203,
        -0.38161388,
    ],
    workspace_bounds={"x": (0.16, 0.50), "y": (-0.38, 0.12), "z": (0.78, 1.12)},
    candidate_indices=(0, 1, 2, 3, 4, 5),
    sim_place_correction_steps=4,
)
```

Metrics:

```text
X2_TWO_OBJECT_RESULT ok=True
object=x2_pick_place_blue_cube
target=right
before_close_tcp_error_m=0.014342061390082383
before_close_ori_error_rad=0.0467510987482574
object_in_hand_after_close=True
place_error_m=0.01886219526172648
Reward=1.0
Task Completed=True
```

Videos:

```text
outputs/codex-a/x2_pick_place_two_objects_blue_right_codex_a_non_oracle_manual_codex_a_two_object_retry02_20260630_160157/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/codex-a/x2_pick_place_two_objects_blue_right_codex_a_non_oracle_manual_codex_a_two_object_retry02_20260630_160157/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Visual pipeline evidence from stdout:

```text
OWL-ViT detect -> SAM2 segment -> ContactGraspNet plan
Generated 122 grasps for object 1
PyRoKi trajectory optimization with World Collision (sweep)
```

## Validation

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xingshu/miniforge3/envs/behavior/bin/python \
  -m pytest tests/test_code_extraction.py tests/test_x2_llm_api.py tests/test_x2_run_summary.py -q
```

Latest result:

```text
14 passed
```

The extra code-extraction test protects the non-oracle case where a model
returns an unfenced multi-line function call. Without it, CaP-X could trim the
bare closing parenthesis and execute only a malformed tail fragment.
