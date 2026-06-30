# X2 Two-Target Stability Hold Baseline Snapshot

Date: 2026-06-29 Asia/Shanghai

This snapshot preserves the X2 CaP-X reduced-API tabletop pick-place baseline
after the two-target left/right task was validated and the final grasp
insertion/fine-align execution was hardened with longer command hold time.

## Why This Snapshot Exists

The right two-target oracle stability run passed, but an initial left stability
run failed before gripper close even though the visual chain produced a
plausible grasp TCP target. The failure was concentrated in final joint-drive
tracking rather than OWL-ViT/SAM2/Contact-GraspNet output.

The current baseline therefore keeps the same visual/API contract and changes
only the action timing:

```text
execute_tcp_grasp_plan insertion max_steps: 420
execute_tcp_grasp_plan fine-align max_steps: 560
execute_tcp_grasp_plan fine-align hold_steps_per_waypoint: at least 8
execute_tcp_grasp_plan fine-align settle_steps: at least 20
pick_and_place_visual_object insert_hold_steps_per_waypoint: 8
```

## Validated Results

Right oracle stability:

```text
outputs/oracle/stability/oracle/two_targets_right_oracle_stability_20260629_1022b_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=right
before_close_tcp_error_m=0.019476928352765467
before_close_ori_error_rad=0.06209430213803601
object_in_hand_after_close=True
place_error_m=0.022797562558285914
Reward=1.0
Task Completed=True
```

Left oracle stability after hold-step hardening:

```text
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1330_left_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.019172631962380303
before_close_ori_error_rad=0.05583956751857787
object_in_hand_after_close=True
place_error_m=0.016582604022264174
Reward=1.0
Task Completed=True
```

Follow-up right/left oracle stability after this snapshot also passed:

```text
Right output:
outputs/oracle/stability/oracle/two_targets_right_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=right
before_close_tcp_error_m=0.0022920335900174634
before_close_ori_error_rad=0.007841036302440513
object_in_hand_after_close=True
place_error_m=0.007655968983878265

Left output:
outputs/oracle/stability/oracle/two_targets_left_oracle_stability_20260629_1345_oracle_stability_after_hold_run01/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/

X2_TWO_TARGET_RESULT ok=True target=left
before_close_tcp_error_m=0.011122489380341058
before_close_ori_error_rad=0.03634797048888362
object_in_hand_after_close=True
place_error_m=0.013434781081554163
```

Accepted videos and visual artifacts are stored in the corresponding `outputs/`
directories, not copied into this snapshot.

## LLM-Facing Reduced API

Generated CaP-X code should normally call one of:

```text
pick_and_place_red_cube()
pick_and_place_red_cube_to_left_target()
pick_and_place_red_cube_to_right_target()
pick_and_place_visual_object()
```

The full X2 control/debug API remains in the main repository for primitive
debugging, but it is not the default LLM-facing surface.

## Reproduce

Oracle stability:

```bash
scripts/run_x2_two_target_oracle_stability_smoke.sh
```

Recommended codex-a path with local summary and acceptance check:

```bash
REPEATS=1 \
scripts/run_x2_two_target_codex_a_stability_and_check.sh
```

If `codex-a` is a shell alias for `codex -c 'model_provider="axonhub"'`, this
script's default `CODEX_BIN=codex` and `CODEX_MODEL_PROVIDER=axonhub` are the
equivalent non-interactive form.

Generic direct OpenAI-compatible API path:

```bash
REPEATS=1 \
MODEL=gpt-5 \
SERVER_URL=http://127.0.0.1:8110/chat/completions \
scripts/run_x2_two_target_api_stability_and_check.sh
```

Summarize local outputs after a run:

```bash
python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

Check local outputs against the current gate:

```bash
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_<STAMP>_run*
```

Audit the local integration state:

```bash
python scripts/audit_x2_capx_integration.py
python scripts/audit_x2_capx_integration.py --strict
```

Codex CLI proxy path, kept for historical comparison:

```bash
scripts/run_x2_two_target_codex_a_non_oracle_smoke.sh
scripts/run_x2_two_target_codex_a_stability_smoke.sh
```

The codex-a path sends task/API prompts to the configured external Codex CLI
provider. The direct API path sends the same CaP-X task/API prompt to the
configured `SERVER_URL`. Use oracle stability when external LLM calls are not
desired.

## Preserved Files

This snapshot includes the current X2 primitive implementation, task wrappers,
task configs, Codex CLI proxy, direct API scripts, smoke/stability scripts,
acceptance checker, tests, and X2 docs under the same relative paths used in
the main repository.

The codex-a stability loop was attempted after creating this snapshot, but it
was rejected by the current managed-tool data-exfiltration policy because it
sends workspace/task prompt data to an external provider. No workaround was
used.
