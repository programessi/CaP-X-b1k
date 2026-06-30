# X2 Red Cube Pick-Place Working Snapshot

This snapshot preserves the working non-oracle CaP-X task API version before
the transfer-path refinement.

Validated run:

```text
outputs/MiniMax-M2.7/x2_pick_place_red_cube_non_oracle_task_api_retry_smoke/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1
```

Metrics:

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.02617899145823312
before_close_ori_error_rad=0.07640485198452023
object_in_hand_after_close=True
place_error_m=0.03806729894863623
Reward=1.0
Task Completed=True
```

Known issue:

```text
The task succeeds, but after grasping the cube the transfer trajectory takes a
large arc around the robot body before returning to the place target.
```

Snapshot files:

```text
control.py
x2_pick_place_red_cube.py
x2_pick_place_red_cube.yaml
x2-llm-facing-primitives.md
x2-capx-pick-place-sim-task.md
x2-version-management.md
```
