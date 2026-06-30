# X2 Two-Target Left Codex-A Baseline Snapshot

Date: 2026-06-29 Asia/Shanghai

This snapshot preserves the X2 tabletop baseline after validating both
two-target directions:

- right target: `pick_and_place_red_cube_to_right_target()`
- left target: `pick_and_place_red_cube_to_left_target()`
- codex-a non-oracle generation through `capx/serving/codex_cli_server.py`
- visual artifact export for RGB, detection overlay, SAM2 mask, mask overlay,
  and selected grasp summary

## Reproduce

Right target:

```bash
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

Left target:

```bash
CONFIG_PATH=env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml \
OUTPUT_DIR=./outputs/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke \
VISUAL_ARTIFACT_DIR=./outputs/x2_visual_artifacts/two_targets_left_codex_a_non_oracle_smoke \
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
```

The script starts the local Codex CLI proxy, runs the configured CaP-X task,
then stops the proxy on exit.

## Accepted LEFT Runs

Oracle output:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Oracle metrics:

```text
X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.011372143517779223
before_close_ori_error_rad=0.04381623598375953
object_in_hand_after_close=True
place_error_m=0.010966659369626599
Reward=1.0
Task Completed=True
```

Codex-a non-oracle output:

```text
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

Generated code:

```python
RESULT = pick_and_place_red_cube_to_left_target()
```

Codex-a metrics:

```text
X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.013898993392196681
before_close_ori_error_rad=0.040283250155416964
object_in_hand_after_close=True
place_error_m=0.020876950973317487
Reward=1.0
Task Completed=True
```

Videos:

```text
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/oracle/oracle/x2_pick_place_red_cube_two_targets_left_oracle_smoke_v3/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_global.mp4
outputs/codex-a/x2_pick_place_red_cube_two_targets_left_codex_a_non_oracle_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/video_combined_robot.mp4
```

Visual artifacts:

```text
outputs/x2_visual_artifacts/two_targets_left_oracle_smoke_v3/x2_pick_place_red_cube_20260629_100323_273/
outputs/x2_visual_artifacts/two_targets_left_codex_a_non_oracle_smoke/x2_pick_place_red_cube_20260629_100830_465/
```

## Preserved Files

This snapshot includes copies of the X2 primitive implementation, two-target
task wrappers, task configs, Codex CLI proxy, smoke scripts, tests, and current
X2 docs under the same relative paths used in the main repository.
