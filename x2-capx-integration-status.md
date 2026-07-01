# X2 CaP-X Integration Status

Date: 2026-07-01

This page is the current acceptance checklist for bringing X2 into the CaP-X
BEHAVIOR tabletop workflow. It records the accepted stable path and the
remaining work that is outside the current simulation scope.

## Current Status

X2 is integrated far enough to run simple CaP-X tabletop pick-place tasks in
simulation:

- The BEHAVIOR-backed X2 low-level environment is registered through CaP-X
  configs.
- The task setup is owned by the CaP-X/BEHAVIOR config layer, not by generated
  code.
- The LLM-facing API is reduced to task-level primitives through
  `X2PickPlaceApi`.
- The visual chain can produce a selected X2 grasp target as `T_world_tcp`.
- The action chain consumes `T_world_tcp`, converts to internal EEF/joint
  targets as needed, and executes approach, close, lift, transfer, place, and
  release.
- Right-target and left-target oracle stability runs passed after insertion and
  fine-align hold-step hardening.
- Right-target and left-target `codex-a` non-oracle stability runs passed with
  generated code that calls only the reduced X2 task-level primitive API.
- The two-object extension passed oracle and `codex-a` non-oracle runs where
  generated code selected the blue cube through `pick_and_place_visual_object`
  and placed it at the right marker.
- The experimental RGB-D obstacle route now has oracle and `codex-a`
  non-oracle successes using perception-derived object/table boxes,
  visual grasp-pose placement offset, one precontact reobserve with
  quality-gated fallback, and a slower vertical place descent. It is evidence
  for the next 2real-oriented path, not a replacement for the accepted
  sim-known baseline yet.
- `scripts/check_x2_acceptance.py` and
  `scripts/audit_x2_capx_integration.py --strict` both pass.

The current accepted baseline is documented here:

```text
docs/x2-accepted-baseline-20260630.md
snapshots/x2_capx_two_target_codex_a_complete_20260630_1025/
```

## Stable User-Facing Tasks

Single target:

```text
env_configs/x2/x2_pick_place_red_cube.yaml
```

Two targets, right marker:

```text
env_configs/x2/x2_pick_place_red_cube_two_targets.yaml
```

Two targets, left marker:

```text
env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml
```

Two objects, choose blue cube and place at right marker:

```text
env_configs/x2/x2_pick_place_two_objects_blue_right.yaml
docs/x2-two-object-blue-right-extension-20260630.md
```

Experimental two-object route with RGB-D visual obstacle boxes and visual
grasp-pose placement offset:

```text
env_configs/x2/x2_pick_place_two_objects_blue_right_rgbd_visual.yaml
docs/x2-rgbd-visual-obstacle-upgrade-20260630.md
scripts/run_x2_two_object_blue_right_rgbd_visual_oracle_smoke.sh
scripts/run_x2_two_object_blue_right_rgbd_visual_codex_a_non_oracle_smoke.sh
```

These tasks use `X2PickPlaceApi`; ordinary generated code should not create
the scene, reset the simulator, instantiate robots, or call low-level camera,
IK, PyRoKi, GraspNet, or gripper helpers directly.

## LLM-Facing Primitive Surface

The formal injected primitive set is:

```python
pick_and_place_red_cube(...)
pick_and_place_red_cube_to_left_target(...)
pick_and_place_red_cube_to_right_target(...)
pick_and_place_visual_object(...)
```

The intended generated code for the two-target right task is:

```python
RESULT = pick_and_place_red_cube_to_right_target()
```

The intended generated code for the left variant is:

```python
RESULT = pick_and_place_red_cube_to_left_target()
```

`plan_visual_grasp_tcp_pose()` and `execute_tcp_grasp_plan()` remain important
internal/lower-level building blocks. They are deliberately not exposed by
`X2PickPlaceApi.functions()` for ordinary pick-place tasks.

## Coordinate And Data Contracts

The working visual/action contract is:

```text
visual selected grasp target: T_world_tcp
action target input:          T_world_tcp
place target:                 world-frame object center [x, y, z]
```

The raw Contact-GraspNet pose is adapted before execution. Generated task code
does not convert TCP to EEF; the X2 action API handles that internally.

For the current short-term simulation baseline, table and cube obstacle boxes
come from sim-known OmniGibson scene AABBs with engineering inflation margins.
The elevated object-center correction before release is also sim-only. This is
acceptable for the current CaP-X simulation integration goal, but it is not a
2real obstacle-perception solution.

The experimental route can be selected explicitly with:

```python
pick_and_place_visual_object(
    ...,
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
)
```

In that route, the object obstacle box is estimated from the target SAM2 mask
and depth points; the table box is estimated from RGB-D points outside the
target mask; the executor can reobserve once at precontact and adopt the new
grasp only if mask/depth/shift/IK quality gates pass; the release TCP offset is
computed from the grasp-time visual object pose and the actual TCP pose after
close; and place descends through multiple vertical waypoints before opening
the gripper. Remaining sim-only reads are evaluation signals such as
`check_object_in_hand`, reward, and `place_error_m`.

## Validated Evidence

Current post-hardening oracle stability evidence:

```text
Right:
outputs/oracle/stability/oracle/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.0022920335900174634
before_close_ori_error_rad=0.007841036302440513
object_in_hand_after_close=True
place_error_m=0.007655968983878265
Reward=1.0
Task Completed=True

Left:
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.011122489380341058
before_close_ori_error_rad=0.03634797048888362
object_in_hand_after_close=True
place_error_m=0.013434781081554163
Reward=1.0
Task Completed=True
```

Latest `codex-a` non-oracle RGB-D route evidence:

```text
Codex-a non-oracle success:
outputs/codex-a/codex-a/x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

Videos:
video_combined_global.mp4
video_combined_robot.mp4

Generated code:
RESULT = pick_and_place_visual_object(...)

Visual artifacts:
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/x2_pick_place_blue_cube_20260701_115434_152/
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145/x2_pick_place_blue_cube_20260701_115611_311/

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
Reward=1.0
Task Completed=True
```

Visual artifacts for those runs are under:

```text
outputs/x2_visual_artifacts/stability/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/
outputs/x2_visual_artifacts/stability/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/
```

The frozen source snapshot for this baseline is:

```text
snapshots/x2_two_target_stability_hold_baseline_20260629_1330/
```

Accepted non-oracle `codex-a` stability evidence:

```text
Stamp:
manual_codex_a_20260630_101117

Right:
outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.015655039528970842
before_close_ori_error_rad=0.0547740942117125
object_in_hand_after_close=True
place_error_m=0.014131927921253392
Reward=1.0
Task Completed=True

Left:
outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.005309625868222024
before_close_ori_error_rad=0.013709298176369174
object_in_hand_after_close=True
place_error_m=0.003187055511323606
Reward=1.0
Task Completed=True
```

Generated code for the accepted non-oracle runs:

```text
outputs/stability/codex-a/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py
outputs/stability/codex-a/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/code.py
```

Visual artifacts:

```text
outputs/x2_visual_artifacts/stability/two_targets_right_codex_a_stability_manual_codex_a_20260630_101117_run01/
outputs/x2_visual_artifacts/stability/two_targets_left_codex_a_stability_manual_codex_a_20260630_101117_run01/
```

The complete accepted source/config/test snapshot is:

```text
snapshots/x2_capx_two_target_codex_a_complete_20260630_1025/
```

Two-object blue-cube extension evidence:

```text
Oracle:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_run03/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.008206982467097177
before_close_ori_error_rad=0.021700898689255475
object_in_hand_after_close=True
place_error_m=0.009616035120709184
Reward=1.0
Task Completed=True

Codex/GPT non-oracle:
outputs/codex-a/x2_pick_place_two_objects_blue_right_codex_a_non_oracle_manual_codex_a_two_object_retry02_20260630_160157/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

before_close_tcp_error_m=0.014342061390082383
before_close_ori_error_rad=0.0467510987482574
object_in_hand_after_close=True
place_error_m=0.01886219526172648
Reward=1.0
Task Completed=True
```

The non-oracle generated program called only:

```python
RESULT = pick_and_place_visual_object(...)
```

The full extension record is:

```text
docs/x2-two-object-blue-right-extension-20260630.md
```

The experimental RGB-D obstacle route record is:

```text
docs/x2-rgbd-visual-obstacle-upgrade-20260630.md

Oracle success:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

Visual artifacts:
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_manual_rgbd_visual_overinsert_20260630_173023/x2_pick_place_blue_cube_20260630_173244_130/

before_close_tcp_error_m=0.007957604188105115
before_close_ori_error_rad=0.022499107681727935
object_in_hand_after_close=True
final_close_axis_offset_m=0.0
place_error_m=0.025959445657849266
Reward=1.0
Task Completed=True
```

That success predates the precontact reobserve / slow descent upgrade. New
evidence for the upgraded route:

```text
Oracle success:
outputs/oracle/oracle/x2_pick_place_two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

Videos:
video_combined_global.mp4
video_combined_robot.mp4

Visual artifacts:
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/x2_pick_place_blue_cube_20260630_183817_658/
outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_manual_rgbd_visual_reobserve_fast_20260630_183604/x2_pick_place_blue_cube_20260630_183928_672/

reobserve_adopted=True
reobserve_reason=quality_gates_passed
reobserve mask_pixels=8215
reobserve depth_points=8210
reobserve object_shift_m=0.000014924572561476803
reobserve grasp_shift_m=0.004602285628847467
before_close_tcp_error_m=0.011525492473389909
before_close_ori_error_rad=0.03224944203486641
object_in_hand_after_close=True
place_error_m=0.02411113666043054
place_descent_waypoints=4
Reward=1.0
Task Completed=True
```

## Re-run Non-Oracle Evidence

If the GPT API key is already configured in the local `codex-a` command, use
the codex-a path. It starts `capx/serving/codex_cli_server.py` and forwards the
CaP-X OpenAI-compatible request to the local Codex CLI configuration.

```bash
REPEATS=1 \
STAMP=manual_codex_a_$(date +%Y%m%d_%H%M%S) \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

If your local executable is named `codex` instead of `codex-a`, override it:

```bash
CODEX_BIN=codex REPEATS=1 scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

The generic direct API path remains available when you explicitly start an
OpenAI-compatible proxy:

```bash
REPEATS=1 \
STAMP=manual_direct_api_$(date +%Y%m%d_%H%M%S) \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_and_check.sh
```

Summarize the saved outputs:

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_codex_a_stability_<STAMP>_run*
python scripts/summarize_x2_runs.py outputs/stability/*/two_targets_*_codex_a_stability_<STAMP>_run*
python scripts/summarize_x2_runs.py outputs/codex-a/x2_pick_place_two_objects_blue_right_codex_a_non_oracle_<STAMP>
```

Run the acceptance checker:

```bash
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_codex_a_stability_<STAMP>_run*
python scripts/check_x2_acceptance.py outputs/stability/*/two_targets_*_codex_a_stability_<STAMP>_run*
```

Run the full local integration audit:

```bash
python scripts/audit_x2_capx_integration.py
```

Strict audit for the accepted baseline:

```bash
python scripts/audit_x2_capx_integration.py --strict
```

Acceptance for this last gap:

- Generated code calls exactly one target primitive for each task.
- Both right and left runs save videos.
- Both runs save visual artifacts with `grasp_summary.json`.
- `ok=True`, `object_in_hand_after_close=True`, and `Task Completed=True`.
- Before-close TCP error remains within the task tolerance; current oracle
  target is below 2 cm.

The checker encodes the current thresholds:

```text
required targets: right,left
before_close_tcp_error_m <= 0.02
before_close_ori_error_rad <= 0.10
place_error_m <= 0.10
video_*.mp4 required
matching grasp_summary.json visual artifact required
code.py must call the target primitive instead of lower-level debug APIs
```

The latest accepted run passed:

```text
X2_ACCEPTANCE PASS
X2_INTEGRATION_AUDIT COMPLETE
```

## Remaining Work Outside This Baseline

The X2 CaP-X tabletop integration can be treated as complete for the current
accepted simulation scope. The next work is new scope:

- Promote the experimental RGB-D route only after repeated oracle/non-oracle
  evidence; the current accepted record is still the sim-known baseline.
- Add more task objects and target layouts.
- Decide whether to expose lower-level `plan_visual_grasp_tcp_pose()` /
  `execute_tcp_grasp_plan()` for tasks that need custom post-grasp behavior.
- Improve trajectory quality beyond the current guarded tabletop route.
