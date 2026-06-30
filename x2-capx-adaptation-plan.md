# X2 CAP-X Low-Level API Adaptation Plan

## Current Accepted Baseline

As of 2026-06-26 13:37 Asia/Shanghai, the short-term X2 CaP-X simulation
baseline is no longer just low-level API adaptation. The accepted path is a
complete simple tabletop pick-place task:

```text
env_configs/x2/x2_pick_place_red_cube.yaml
  -> X2PickPlaceApi
  -> pick_and_place_visual_object()
  -> visual grasp chain
  -> PyRoKi precontact / transfer
  -> joint-IK insertion and vertical placement descent
```

The accepted smoke output is:

```text
outputs/oracle/x2_pick_place_red_cube/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
```

Accepted metrics:

```text
before_close_tcp_error_m=0.012187954567831028
before_close_ori_error_rad=0.0333455628915337
object_in_hand_after_close=True
place_error_m=0.04355407559125816
```

The LLM-facing task config now uses `X2PickPlaceApi`, a reduced wrapper that
exposes only the task-level pick-place primitive and a few safe helpers. The
full `X2ControlApi` / `X2VisualGraspApi` remain available for debugging but are
not the default prompt surface for this task.

The older sections below are retained as historical adaptation notes.

## Goal

Adapt X2 to CAP-X by following the existing R1Pro layering:

```text
BEHAVIOR low-level helper
  -> CAP-X low-level environment wrapper
  -> CAP-X ControlApi functions visible to generated code / LLM
  -> CAP-X higher-level skills
```

For X2, the BEHAVIOR-side low-level helpers have already been smoke tested in a fixed-base, no-CuRobo setup:

- `_empty_action`
- `_settle_robot`
- `_sample_grasp_pose`
- `_move_hand_direct_ik`
- `_move_hand`
- `_move_fingers_to_limit`
- `_reset_hand`

Navigation and CuRobo-related planner paths are intentionally out of scope for the current X2 phase.

## Layer 1: API Shape Alignment

Objective: make CAP-X able to instantiate an X2 control API whose public function names and basic signatures are close to R1Pro.

Initial public functions:

- `get_env_observation`
- `get_object_pose`
- `get_current_joint_positions`
- `get_current_eef_pose`
- `get_robot_relative_eef_pose`
- `move_hand`
- `open_gripper`
- `close_gripper`
- `sample_grasp_pose`
- `grasp_object`
- `check_object_in_hand`
- `move_to_joint_positions`
- `solve_ik`
- `lift_arm`

Not exposed in the current fixed-base phase:

- `navigate_to_pose`
- `get_navigation_pose`
- `find_object_base_rotate`
- any other base locomotion / base rotation API

Expected result:

- `X2ControlApi(env).functions()` returns these names.
- The functions can be imported and inspected by CAP-X prompt generation.
- `solve_ik` and `move_to_joint_positions` are present for R1Pro API shape compatibility, but unsupported or intentionally deferred functions fail clearly with `NotImplementedError`, instead of silently doing the wrong thing.

Layer 1 does not require full task success and does not require LLM execution.

Current Layer 1 status:

- Added `capx.envs.simulators.x2_b1k.X2BehaviourLowLevel`.
- Added `capx.integrations.x2.control.X2ControlApi`.
- Registered `x2_b1k_low_level`.
- Registered `X2ControlApi`.
- Verified `X2ControlApi.functions()` returns the expected fixed-base manipulation API names.
- Verified the API does not expose `navigate_to_pose`, `get_navigation_pose`, or `find_object_base_rotate`.
- Verified CAP-X registries list both `x2_b1k_low_level` and `X2ControlApi`.

## Layer 2: Single-Function Behavior Smoke Test

Objective: call each X2 API directly from Python, without LLM code generation.

Recommended script sequence:

```text
1. Create X2 BEHAVIOR low-level env.
2. Instantiate X2ControlApi.
3. Call get_env_observation().
4. Call get_current_eef_pose(arm=1).
5. Call open_gripper(arm=1).
6. Call close_gripper(arm=1).
7. Move right EEF +5 cm with move_hand(..., arm=1).
8. Place a target object near the current EEF.
9. Call sample_grasp_pose(object_name).
10. Call grasp_object(pregrasp_pose, grasp_pose, object_name, arm=1).
11. Call check_object_in_hand(arm=1).
```

Passing criteria:

- Observation returns nested gripper-camera RGB tensors without shape or key errors; proprio/state is checked through the explicit state APIs.
- EEF pose returns `(position, quaternion_xyzw)`.
- Gripper open reaches the Robotiq open region, around `0.785 rad` for controlled knuckles.
- Gripper close reaches the closed region, around `0.0 rad`.
- A +5 cm hand move goes in the correct direction, with roughly centimeter-level error.
- `sample_grasp_pose` returns pregrasp and grasp poses without relying on missing R1Pro fingertip metadata.
- `grasp_object` executes a complete attempt and reports success/failure cleanly.

Current Layer 2 status:

- Added `scripts/archive/x2_experiments/x2_capx_api_smoke.py`.
- Updated `X2BehaviourLowLevel._sample_grasp_pose()` to use an X2 fixed-base near-field sampler instead of the generic BEHAVIOR sampler. The sampler uses the current object pose, AABB height, current EEF orientation, and an approach offset from the selected arm.
- The smoke test uses a `fixed_base` / `kinematic_only` target object so that Layer 2 validates API behavior without the object falling or being pushed away in the empty scene. Real grasp success should be re-tested later with a physically supported, non-kinematic object.
- Full direct API smoke passed with `outputs/x2_capx_api_smoke/summary.json` reporting `ok: true`.
- Verified passing API steps:
  - `get_env_observation`
  - `get_current_joint_positions`
  - `get_current_eef_pose`
  - `get_robot_relative_eef_pose`
  - `get_object_pose`
  - `open_gripper`
  - `close_gripper`
  - `move_hand`
  - `sample_grasp_pose`
  - `lift_arm`
  - `grasp_object`
  - `check_object_in_hand`
  - `solve_ik` expected `NotImplementedError`
  - `move_to_joint_positions` expected `NotImplementedError`
- Latest measured `move_hand(+x=5cm)` result: commanded target moved in the correct +x dominant direction, actual delta was about `[0.0375, 0.0005, 0.0055]` m, target error about `1.36 cm`.
- Latest measured gripper result: open controlled joints about `0.785 rad`, closed about `0.00024 rad`.
- Latest measured `sample_grasp_pose` result: grasp pose was about `6.04 cm` from object center, pregrasp-grasp separation about `6.71 cm`, and grasp z was above the object pose.
- `grasp_object` completed the open -> pregrasp -> grasp -> close -> lift -> check chain and returned a bool. In the kinematic-object smoke it returned `False`, which is acceptable for Layer 2 because grasp success is not the criterion for this fixed-object API test.
- Isaac / Kit still exits with a shutdown-time segmentation fault (`139`) after the script writes the summary. The business result should be judged from `summary.json`; the crash stack is in Kit teardown / `atexit`, not in the CAP-X API calls.

## Layer 3: CAP-X Code-Execution Integration

Objective: verify that CAP-X can inject X2 APIs into generated-code execution.

Use fixed code first, not an LLM:

```python
import numpy as np

obs = get_env_observation()
pos, quat = get_current_eef_pose(arm=1)
open_gripper(arm=1)
move_hand((pos + np.array([0.05, 0.0, 0.0]), quat), arm=1)
close_gripper(arm=1)
```

Passing criteria:

- The CAP-X runner can create the X2 task environment.
- The API functions are injected into the code execution namespace.
- The fixed code runs without signature, serialization, or numpy/torch conversion errors.
- Robot behavior matches the direct Layer 2 smoke test.

## Simulation Environment Notes

The current plan should not require a new Gym-style wrapper from scratch.

CAP-X already provides the generic structure:

```text
CodeExecutionEnvBase
  -> low_level env object
  -> ControlApi.functions()
```

R1Pro uses `R1ProBehaviourLowLevel`, which wraps OmniGibson / BEHAVIOR and then exposes control through `R1ProControlApi`.

For X2, the expected work is a sibling wrapper, not a rewrite of the simulation stack:

```text
X2BehaviourLowLevel
  -> OmniGibson X2 config
  -> StarterSemanticActionPrimitives with skip_curobo_initilization=True
  -> X2ControlApi
```

The environment layer is still BEHAVIOR / OmniGibson underneath. The CAP-X side mainly needs robot-specific configuration, observation key handling, and API registration.

The biggest R1Pro-specific parts to avoid copying directly are:

- R1Pro joint indices.
- R1Pro torso and base motion assumptions.
- Pyroki IK mapping for R1Pro.
- R1Pro gripper joint controller assumptions.
- R1Pro camera link names.

X2 should instead rely on the BEHAVIOR X2 helpers and controller semantics already validated in the X2 smoke tests.
