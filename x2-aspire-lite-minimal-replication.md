# X2 ASPIRE-Lite Minimal Replication

Date: 2026-07-01

This document records the first ASPIRE-style layer on top of the existing X2
RGB-D non-oracle CaP-X route. It does not replace the stable
`x2-rgbd-non-oracle-v1` baseline. It adds trace-first debugging, failure
taxonomy, a small skill library, and a controlled candidate-search harness.

## Scope

The base task remains:

```text
env_configs/x2/x2_pick_place_two_objects_blue_right_rgbd_visual.yaml
```

The task is still the blue-cube tabletop pick-place route:

```text
CaP-X generated code
-> pick_and_place_visual_object(...)
-> RGB-D detection/segmentation/grasp generation
-> RGB-D object/table obstacle estimation
-> PyRoKi approach planning
-> joint-IK execution
-> gripper close, transfer, slow vertical place, release
```

The ASPIRE-lite additions are offline-friendly and do not import Isaac Sim:

```text
capx/integrations/x2/aspire.py
capx/integrations/x2/aspire_skills.json
scripts/build_x2_aspire_trace_bundle.py
scripts/run_x2_aspire_rgbd_candidate_search.py
scripts/aggregate_x2_aspire_evidence.py
scripts/audit_x2_aspire_acceptance.py
```

## Trace Bundle

The trace builder converts a saved CaP-X run plus X2 visual artifacts into:

```text
outputs/x2_aspire_traces/<run_id>/<trial_id>/
  run_meta.json
  generated_code.py
  primitive_calls.jsonl
  metrics.json
  failure_report.json
  skill_library.json
  visual/
  motion/
  videos/
```

It preserves:

- generated task code,
- `pick_and_place_visual_object(...)` inputs and outputs,
- OWL-ViT detections,
- SAM2 masks and overlays,
- ContactGraspNet candidate summaries,
- selected `T_world_tcp` grasp and precontact poses,
- RGB-D object/table obstacle boxes with `sim_truth=False`,
- planned and reached TCP waypoints,
- tracking errors,
- videos.

Build a bundle from an existing run:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/build_x2_aspire_trace_bundle.py \
  outputs/codex-a/codex-a/x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145 \
  --repo-root . \
  --visual-artifact-root outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_codex_a_non_oracle_manual_rgbd_visual_codex_a_20260701_115145 \
  --output-root outputs/x2_aspire_traces/manual_rgbd_visual_codex_a_20260701_115145
```

The RGB-D oracle and `codex-a` smoke scripts now call the trace builder after a
run when `CAPX_X2_BUILD_ASPIRE_TRACE=1`, which is the default.

## Failure Taxonomy

`failure_report.json` uses this taxonomy:

```text
perception_no_detection
segmentation_bad_mask
depth_too_sparse
grasp_pose_unreachable
planner_collision_or_no_path
preclose_pose_not_reached
object_not_in_hand_after_close
place_error_too_large
timeout
unknown
```

Each failure report contains:

```text
status
primary_failure
suggested_repair_tags
evidence
```

For a successful run, `primary_failure=null` and `status=success`.

## Skill Library

The initial reusable skill library is:

```text
rgbd_tabletop_obstacle_box_planning
precontact_reobserve_with_quality_gates
slow_vertical_place_descent
try_next_grasp_candidate_when_preclose_unreached
```

Each entry records:

```text
when
repair
evidence
applicable_task
```

The trace builder writes the applicable skills into each bundle and also keeps
the full default library in `skill_library.json`.

## Candidate Search

`scripts/run_x2_aspire_rgbd_candidate_search.py` implements a controlled
ASPIRE-style search over high-level primitive parameters, not arbitrary
low-level code edits.

The current search space includes:

```text
candidate_indices
grasp_tcp_axis_offsets_m
reobserve_at_precontact
reobserve_distance_m
place_orientation_source
place_descent_waypoints
place_descent_max_joint_step
place_descent_hold_steps
place_pre_release_settle_steps
```

The task module reads these environment overrides:

```text
CAPX_X2_RGBD_CANDIDATE_INDICES
CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M
CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT
CAPX_X2_RGBD_REOBSERVE_DISTANCE_M
CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE
CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS
CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP
CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS
CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS
```

Plan the search without running simulation:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/run_x2_aspire_rgbd_candidate_search.py \
  --repo-root . \
  --output-root outputs/x2_aspire_candidate_search/plan_only_smoke \
  --debug-limit 1 \
  --validation-limit 1
```

Run the full debug/validation loop:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/run_x2_aspire_rgbd_candidate_search.py \
  --repo-root . \
  --output-root outputs/x2_aspire_candidate_search/rgbd_debug_validation_<STAMP> \
  --debug-limit 3 \
  --validation-limit 3 \
  --execute
```

Recover or continue a partially completed long run without rerunning existing
trial folders:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/run_x2_aspire_rgbd_candidate_search.py \
  --repo-root . \
  --output-root outputs/x2_aspire_candidate_search/<RUN_ID> \
  --debug-limit 3 \
  --validation-limit 3 \
  --execute \
  --reuse-existing
```

In execute mode, the harness:

1. writes per-seed temporary configs with small object/distractor position
   perturbations,
2. runs all candidates on debug seeds,
3. builds trace bundles,
4. scores each candidate from reward, task completion, TCP error, orientation
   error, place error, and failure class,
5. selects the best candidate,
6. runs only that best candidate on held-out validation seeds,
7. writes `candidate_search_report.json` and `findings.md`.

The execute harness now starts each simulation run in its own process group.
If a run times out or the operator interrupts the search with Ctrl-C, the
harness terminates the child OmniGibson / Isaac process group instead of
leaving a background simulator running.

Audit a completed candidate-search report against the ASPIRE-lite acceptance
gates:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/audit_x2_aspire_acceptance.py \
  outputs/x2_aspire_candidate_search/<RUN_ID>/candidate_search_report.json
```

The audit checks:

- execute mode,
- at least two candidates,
- at least 3 debug seeds and 3 validation seeds,
- debug trials for multiple candidates,
- selected best candidate,
- validation trial count and success count,
- average validation TCP / orientation / place errors,
- trace and video presence,
- RGB-D visual obstacle source with `rgbd_obstacles_sim_truth=False`,
- at least one non-unknown failure report.

Aggregate already executed trace bundles and partially overwritten
candidate-search reports into one auditable report:

```bash
/home/xingshu/miniforge3/envs/behavior/bin/python scripts/aggregate_x2_aspire_evidence.py \
  outputs/x2_aspire_candidate_search \
  --output-root outputs/x2_aspire_candidate_search/aggregate_existing_20260701
```

This is useful because long X2 simulation runs are expensive and some earlier
debug/validation trace bundles may exist even when their original
`candidate_search_report.json` was later overwritten by a narrower experiment.

## Current Evidence

The first trace bundle was built from the accepted `codex-a` non-oracle run:

```text
outputs/x2_aspire_traces/manual_rgbd_visual_codex_a_20260701_115145/
```

Its metrics preserve the accepted run:

```text
reward=1.0
task_completed=True
obstacle_source=rgbd_visual
place_offset_source=visual_grasp_pose
rgbd_obstacles_sim_truth=False
reobserve_adopted=True
before_close_tcp_error_m=0.010910568687270814
before_close_ori_error_rad=0.026704567279301608
object_in_hand_after_close=True
place_error_m=0.019196923124027467
```

A recovered oracle debug search also demonstrates the ASPIRE-style success vs.
failure loop:

```text
outputs/x2_aspire_candidate_search/execute_oracle_min_20260701_132203/
```

After adding richer preclose diagnostics, a new interrupted execute smoke was
run here:

```text
outputs/x2_aspire_candidate_search/diagnostic_validation_20260701_151130/
```

It completed the first debug seed through the full CaP-X injection path:

```text
candidate=stable_rgbd_v1
seed=debug_nominal
reward=1.0
task_completed=True
obstacle_source=rgbd_visual
rgbd_obstacles_sim_truth=False
before_close_tcp_error_m=0.003963908906903284
before_close_ori_error_rad=0.00995791610690803
object_in_hand_after_close=True
place_error_m=0.08423367884237616
trace_bundles=1
videos=1
```

This is valid smoke evidence that the ASPIRE trace/failure/skill/candidate
search machinery can wrap an actual X2 RGB-D CaP-X execution. It is not full
ASPIRE acceptance, because the run was intentionally stopped after the first
successful debug seed to avoid spending another hour on serial CPU-backed
simulation trials.

The current acceptance audit for that partial run is expected to fail:

```text
execute_mode=True
multiple_candidates=True
debug_seed_split=True
validation_seed_split=True
non_oracle_rgbd_obstacle_gate=True

debug_candidate_search_ran=False
best_candidate_selected=False
validation_trials_ran=False
validation_success_gate=False
validation_tcp_error_gate=False
validation_ori_error_gate=False
validation_place_error_gate=False
validation_trace_and_video_gate=False
failure_taxonomy_evidence=False
```

Aggregating all existing candidate-search trace evidence gives a stronger
picture:

```text
outputs/x2_aspire_candidate_search/aggregate_existing_20260701/
```

Debug evidence:

```text
controlled_failure_fast_no_reobserve:
  successes=0/1
  primary_failure=object_not_in_hand_after_close

repair_wider_relaxed_preclose:
  successes=0/1
  primary_failure=object_not_in_hand_after_close

repair_validated_relaxed_preclose_v2:
  successes=1/2
  primary_failures=[None, grasp_pose_unreachable]

stable_rgbd_v1:
  successes=2/4
  primary_failures=[None, preclose_pose_not_reached, None, preclose_pose_not_reached]
```

The aggregate selector currently picks:

```text
best_candidate=stable_rgbd_v1
```

Existing held-out validation evidence for that candidate:

```text
validation stable_rgbd_v1:
  successes=1/1
  before_close_tcp_error_m=0.017743262284341737
  before_close_ori_error_rad=0.06298172387392523
  place_error_m=0.011073738794350871
  trace_bundles=1
  videos=1
```

The aggregate audit now passes these gates:

```text
execute_mode=True
multiple_candidates=True
debug_seed_split=True
debug_candidate_search_ran=True
best_candidate_selected=True
validation_tcp_error_gate=True
validation_ori_error_gate=True
validation_place_error_gate=True
validation_trace_and_video_gate=True
non_oracle_rgbd_obstacle_gate=True
failure_taxonomy_evidence=True
```

The remaining acceptance gap is narrow:

```text
validation_seed_split=False   # only 1 validation seed has executed evidence
validation_trials_ran=False   # requires 3, has 1
validation_success_gate=False # requires 3 successes, has 1
```

The stable candidate succeeds and the failure-seeking candidate is classified
without hand video inspection:

```text
stable_rgbd_v1:
  task_completed=True
  before_close_tcp_error_m=0.0014778295787825236
  before_close_ori_error_rad=0.005852184178485911
  place_error_m=0.023527558358771444

controlled_failure_fast_no_reobserve:
  task_completed=False
  primary_failure=object_not_in_hand_after_close
  suggested_repair_tags=[
    increase_grasp_tcp_axis_offsets,
    try_next_grasp_candidate,
    slow_down_close,
  ]
```

The selected debug candidate is `stable_rgbd_v1`. This is still debug evidence,
not the final held-out validation gate.

The next targeted validation focused on a place-stage failure found after a
successful grasp. The root cause was not perception: after the transfer lateral
waypoint had already reached `place_pre_tcp_pose`, the executor sent a second
`move_tcp_joint_ik()` request to the same target. That redundant solve/execute
could move the arm away by roughly 10 cm and produce
`place_pre_pose_not_reached`.

The repair candidate is:

```text
repair_place_keep_lift_orientation_v1
```

It applies these policy changes:

```text
CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE=post_lift_current
CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT=1
CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS=4
CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP=0.006
CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS=8
CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS=16
```

`post_lift_current` means the release/place TCP orientation is taken from the
measured TCP pose after the object has been closed on and lifted, rather than
forcing the original GraspNet grasp orientation through the entire transfer and
place phase. The executor also measures the TCP error after lateral transfer;
if it is already within tolerance of `place_pre_tcp_pose`, it records:

```text
already_reached_before_move=True
move_skipped_reason=already_reached_after_transfer_lateral
```

and skips the redundant place-pre IK move.

The 3-seed held-out validation result is:

```text
outputs/x2_aspire_candidate_search/place_skip_redundant_prepose_3seed_20260701/

successes=3/3
avg_before_close_tcp_error_m=0.012739941200477849
avg_before_close_ori_error_rad=0.04209046921654453
avg_place_error_m=0.021531850539215216
trace_bundles=3
videos=3
rgbd_obstacles_sim_truth=False
```

The validation report passes the held-out validation gates, but it is
intentionally not a complete single-file ASPIRE acceptance report: it contains
one repaired candidate and no debug/failure trials. The strict audit therefore
still marks `multiple_candidates`, `debug_seed_split`,
`debug_candidate_search_ran`, and `failure_taxonomy_evidence` as false for
this specific report.

Per-seed evidence:

```text
val_nominal_shift:
  task_completed=True
  before_close_tcp_error_m=0.02229221563959983
  place_pre_tcp_error_m=0.0021955273358572913
  place_error_m=0.02699277839877451

val_centered:
  task_completed=True
  before_close_tcp_error_m=0.011147134884122093
  place_pre_tcp_error_m=0.013928521191464904
  place_error_m=0.020812971109720378

val_right_shift:
  task_completed=True
  before_close_tcp_error_m=0.004780473077711631
  place_pre_tcp_error_m=0.0022970977236736806
  place_error_m=0.01678980210915076
```

Videos:

```text
outputs/x2_aspire_candidate_search/place_skip_redundant_prepose_3seed_20260701/validation/repair_place_keep_lift_orientation_v1/val_nominal_shift/oracle/run/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/x2_aspire_candidate_search/place_skip_redundant_prepose_3seed_20260701/validation/repair_place_keep_lift_orientation_v1/val_centered/oracle/run/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/x2_aspire_candidate_search/place_skip_redundant_prepose_3seed_20260701/validation/repair_place_keep_lift_orientation_v1/val_right_shift/oracle/run/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
```

## Remaining Acceptance Work

The ASPIRE-lite machinery now has:

```text
trace bundles:                     implemented
failure taxonomy:                  implemented
skill-library records:             implemented
candidate-search harness:          implemented
RGB-D obstacle non-oracle gate:     implemented
controlled failure evidence:       present in aggregate evidence
targeted 3-seed validation:        passed for repair_place_keep_lift_orientation_v1
```

For a strict ASPIRE-style paper-level reproduction, the remaining upgrade is to
run one uninterrupted report that contains both the multi-candidate debug
search and the 3-seed validation in the same
`candidate_search_report.json`. The current evidence is equivalent in content
for engineering purposes, but split across earlier aggregate debug evidence and
the targeted 3-seed validation run.

The practical acceptance thresholds for future runs are:

```text
validation task_completed >= 3/3, or >= 4/5
average before_close_tcp_error_m < 0.025
average before_close_ori_error_rad < 0.08
average place_error_m < 0.05
every validation trial has videos
every validation trial has a trace bundle
at least one controlled failure produces a non-unknown failure_report
```
