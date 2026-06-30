# X2 Two-Target Codex-A Baseline Snapshot

Date: 2026-06-29 Asia/Shanghai

This snapshot preserves the current X2 tabletop baseline after validating:

- single-target reduced API pick-place,
- two-target right-marker pick-place,
- codex-a non-oracle code generation via `capx/serving/codex_cli_server.py`,
- visual artifact export for RGB, detection overlay, SAM2 mask, mask overlay,
  and selected grasp summary.

## Reproduce

From the repository root:

```bash
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

The script starts the local Codex CLI proxy, runs:

```text
env_configs/x2/x2_pick_place_red_cube_two_targets.yaml
```

then stops the proxy on exit.

Useful overrides:

```bash
OUTPUT_DIR=./outputs/my_two_target_codex_a_run \
VISUAL_ARTIFACT_DIR=./outputs/x2_visual_artifacts/my_two_target_codex_a_run \
CODEX_MODEL_PROVIDER=axonhub \
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

## Accepted Run

Output:

```text
outputs/codex-a/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Generated code:

```python
RESULT = pick_and_place_red_cube_to_right_target()
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

Metrics:

```text
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
video_combined_global.mp4
video_combined_robot.mp4
```

Visual artifacts:

```text
outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_092853_950/
```

Key visual summary:

```text
mask_pixels=8241
graspnet_candidate_count=48
object_position_world=[0.3197035017668407, -0.07959770678727666, 0.9212843309825731]
selected_rank=0
selected_raw_index=12
grasp_tcp_pose.position=[0.3260178858659517, -0.08302930016842375, 0.9249030408645927]
grasp_tcp_pose.quat_xyzw=[0.5411638007994698, 0.6343949672102707, 0.4155622132503154, -0.3633081518505343]
```

## Preserved Files

This snapshot includes copies of the X2 primitive implementation, task wrappers,
task configs, Codex CLI proxy, smoke scripts, and current X2 docs under the same
relative paths used in the main repository.
