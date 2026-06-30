# X2 Pick-Place CaP-X Baseline Snapshot

Date: 2026-06-26 15:30 Asia/Shanghai

This snapshot preserves the accepted X2 CaP-X non-oracle red-cube pick-place
baseline after hardening code extraction for `<think>...</think>` model output.

## What This Proves

The LLM generated normal CaP-X task code:

```python
RESULT = pick_and_place_red_cube()
```

That call invoked the X2 LLM-facing primitive, which internally connected:

```text
visual detection/segmentation/grasp generation
-> X2 TCP grasp pose adaptation
-> PyRoKi precontact approach
-> TCP insertion
-> gripper close and in-hand check
-> lift
-> high-waypoint transfer
-> placement
-> release
```

The run did not use oracle code. It used CaP-X code injection with
`X2PickPlaceApi`.

## Result

```text
X2_PICK_PLACE_RESULT ok=True
before_close_tcp_error_m=0.007781398923911473
before_close_ori_error_rad=0.02522619288209918
object_in_hand_after_close=True
place_error_m=0.014900085488091878
Reward=1.0
Task Completed=True
```

## Files

`files/` contains copies of the implementation and documentation that define the
baseline:

```text
control.py
x2_pick_place_red_cube.py
x2_pick_place_red_cube.yaml
run_x2_pick_place_red_cube_non_oracle_smoke.sh
x2-pick-place-current-baseline.md
x2-llm-facing-primitives.md
```

`output/` contains evidence from the accepted run:

```text
code.py
raw_response.sh
all_responses.json
summary.txt
video_combined_global.mp4
video_combined_robot.mp4
global_contact_sheet.jpg
robot_contact_sheet.jpg
```

## Video Review

The videos are 45.6 seconds at 30 FPS. Contact sheets were extracted at frames:

```text
0, 180, 360, 540, 720, 900, 1080, 1320
```

The sampled frames show the cube being grasped and transferred near the green
placement target. They do not show the earlier long tabletop dragging failure.
The robot camera view is partially occluded by the arm/body, so final acceptance
should still rely on the full videos plus the metric summary.

## Notes

- The visual grasp pose is produced by the visual chain.
- Table/cube collision boxes and elevated place correction are sim-known and are
  not 2real-ready perception primitives.
- This snapshot is the rollback point before future trajectory-quality,
  obstacle-perception, or multi-task changes.
