"""CAP-X X2 motion-primitives demo without scene objects."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np

import capx.envs.simulators  # noqa: F401 - registers low-level envs
import capx.envs.tasks  # noqa: F401 - registers code-execution envs/configs
import capx.integrations  # noqa: F401 - registers APIs
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.envs.tasks import CodeExecEnvConfig, get_exec_env

MOVE_DELTA = [0.08, 0.0, -0.04]
STUCK_PATIENCE_STEPS = 45
POST_HOLD_STEPS_DEFAULT = 0

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb"],
    "sensor_kwargs": {
        "image_height": 768,
        "image_width": 768,
    },
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


MOTION_PRIMITIVES_CODE = r'''
import numpy as np

ARM = 1
MOVE_DELTA = np.array([0.08, 0.0, -0.04], dtype=np.float32)
STUCK_PATIENCE_STEPS = 45

trace = []

def pose_to_list(pose):
    pos, quat = pose
    return {
        "pos": np.asarray(pos, dtype=np.float32).round(6).tolist(),
        "quat": np.asarray(quat, dtype=np.float32).round(6).tolist(),
    }

def record(label, **kwargs):
    entry = {"label": label}
    entry.update(kwargs)
    trace.append(entry)

def pause(steps=12):
    settle_robot(steps=steps)

initial_eef = get_current_eef_pose(arm=ARM)
record("initial", eef=pose_to_list(initial_eef), gripper=get_gripper_state(arm=ARM))
pause(steps=12)

open_gripper(arm=ARM)
open_gripper_state = get_gripper_state(arm=ARM)
record("open_gripper", eef=pose_to_list(get_current_eef_pose(arm=ARM)), gripper=open_gripper_state)
pause(steps=12)

target_pose = (
    np.asarray(initial_eef[0], dtype=np.float32) + MOVE_DELTA,
    np.asarray(initial_eef[1], dtype=np.float32),
)
record("target_pose", target=pose_to_list(target_pose), gripper=get_gripper_state(arm=ARM))
pause(steps=8)

move_ok = move_hand(
    target_pose,
    arm=ARM,
    pos_thresh=0.005,
    ori_thresh=0.1,
    stop_if_stuck=True,
    stuck_patience_steps=STUCK_PATIENCE_STEPS,
)
reached_eef = get_current_eef_pose(arm=ARM)
record("move_to_target", ok=bool(move_ok), eef=pose_to_list(reached_eef), gripper=get_gripper_state(arm=ARM))
pause(steps=12)

close_gripper(arm=ARM)
closed_gripper_state = get_gripper_state(arm=ARM)
after_close_eef = get_current_eef_pose(arm=ARM)
record("close_gripper", eef=pose_to_list(after_close_eef), gripper=closed_gripper_state)
pause(steps=18)

return_ok = move_hand(
    initial_eef,
    arm=ARM,
    pos_thresh=0.005,
    ori_thresh=0.1,
    stop_if_stuck=True,
    stuck_patience_steps=STUCK_PATIENCE_STEPS,
)
home_eef = get_current_eef_pose(arm=ARM)
record("return_home", ok=bool(return_ok), eef=pose_to_list(home_eef), gripper=get_gripper_state(arm=ARM))

final_eef = get_current_eef_pose(arm=ARM)
final_gripper_state = get_gripper_state(arm=ARM)
record("final", eef=pose_to_list(final_eef), gripper=final_gripper_state)

RESULT = {
    "trace": trace,
    "move_ok": bool(move_ok),
    "return_ok": bool(return_ok),
    "initial_eef_pos": np.asarray(initial_eef[0], dtype=np.float32).round(6).tolist(),
    "target_eef_pos": np.asarray(target_pose[0], dtype=np.float32).round(6).tolist(),
    "reached_eef_pos": np.asarray(reached_eef[0], dtype=np.float32).round(6).tolist(),
    "home_eef_pos": np.asarray(home_eef[0], dtype=np.float32).round(6).tolist(),
    "final_eef_pos": np.asarray(final_eef[0], dtype=np.float32).round(6).tolist(),
    "open_finger_span_y_eef": open_gripper_state.get("finger_span_y_eef"),
    "closed_finger_span_y_eef": closed_gripper_state.get("finger_span_y_eef"),
    "final_finger_span_y_eef": final_gripper_state.get("finger_span_y_eef"),
}
'''


def _check(condition: bool, message: str) -> str:
    return "pass" if condition else f"fail: {message}"


def _validate_result(result: dict[str, Any] | None, info: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    checks: list[str] = []
    detail: dict[str, Any] = {
        "result": result,
        "sandbox_rc": info.get("sandbox_rc"),
        "stdout_tail": info.get("stdout", "")[-2000:],
        "stderr_tail": info.get("stderr", "")[-4000:],
    }
    checks.append(_check(info.get("sandbox_rc") == 0, f"sandbox_rc={info.get('sandbox_rc')}"))
    checks.append(_check(isinstance(result, dict), f"RESULT must be dict, got {type(result).__name__}"))
    if not isinstance(result, dict):
        return False, detail, checks

    expected_labels = [
        "initial",
        "open_gripper",
        "target_pose",
        "move_to_target",
        "close_gripper",
        "return_home",
        "final",
    ]
    trace = result.get("trace")
    labels = [entry.get("label") for entry in trace] if isinstance(trace, list) else []
    checks.append(_check(labels == expected_labels, f"trace labels {labels} != {expected_labels}"))
    checks.append(_check(result.get("move_ok") is True, "move_hand(target_pose) returned False"))
    checks.append(_check(result.get("return_ok") is True, "move_hand(initial_eef) returned False"))

    initial = np.asarray(result.get("initial_eef_pos", []), dtype=np.float32)
    target = np.asarray(result.get("target_eef_pos", []), dtype=np.float32)
    reached = np.asarray(result.get("reached_eef_pos", []), dtype=np.float32)
    home = np.asarray(result.get("home_eef_pos", []), dtype=np.float32)
    final = np.asarray(result.get("final_eef_pos", []), dtype=np.float32)
    shapes_ok = all(arr.shape == (3,) for arr in [initial, target, reached, home, final])
    checks.append(_check(shapes_ok, "all reported EEF positions must be 3D"))
    if shapes_ok:
        target_error = float(np.linalg.norm(reached - target))
        home_error = float(np.linalg.norm(home - initial))
        final_drift = float(np.linalg.norm(final - home))
        requested_delta = float(np.linalg.norm(target - initial))
        detail.update(
            {
                "requested_delta": round(requested_delta, 6),
                "target_eef_error": round(target_error, 6),
                "home_eef_error": round(home_error, 6),
                "final_drift": round(final_drift, 6),
            }
        )
        checks.append(_check(target_error < 0.015, f"target EEF error {target_error:.4f}m"))
        checks.append(_check(home_error < 0.02, f"home EEF error {home_error:.4f}m"))
        checks.append(_check(final_drift < 0.02, f"final drift {final_drift:.4f}m"))

    open_span = result.get("open_finger_span_y_eef")
    closed_span = result.get("closed_finger_span_y_eef")
    detail["gripper_span"] = {"open": open_span, "closed": closed_span}
    if isinstance(open_span, (int, float)) and isinstance(closed_span, (int, float)):
        checks.append(_check(closed_span < open_span, f"closed span {closed_span} is not below open span {open_span}"))

    return all(v.startswith("pass") for v in checks), detail, checks


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> dict[str, Any]:
    if not frames:
        return {"path": None, "frame_count": 0, "written": False}
    arr = np.asarray(frames)
    media.write_video(path, arr, fps=fps)
    return {"path": str(path), "frame_count": int(len(frames)), "shape": list(arr.shape), "fps": fps, "written": True}


def _resize_nearest(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    y_idx = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
    return frame[y_idx][:, x_idx]


def _write_videos(output_dir: Path, frames_by_view: dict[str, list[np.ndarray]], fps: int) -> dict[str, Any]:
    result: dict[str, Any] = {"views": {}, "fps": fps}
    for view, frames in frames_by_view.items():
        result["views"][view] = _write_video(output_dir / f"{view}.mp4", frames, fps=fps)
    views = [view for view in ["global", "robot", "rgb"] if frames_by_view.get(view)]
    if len(views) >= 2:
        frame_count = min(len(frames_by_view[view]) for view in views)
        height = max(frames_by_view[view][0].shape[0] for view in views)
        combined = [
            np.concatenate([_resize_nearest(frames_by_view[view][i], height) for view in views], axis=1)
            for i in range(frame_count)
        ]
    else:
        combined = []
    result["combined"] = _write_video(output_dir / "video_combined.mp4", combined, fps=fps)
    return result


def _hold_current_target(env: X2BehaviourLowLevel, steps: int, arm: int = 1) -> dict[str, int]:
    env._hold_current_hand_target(arm=arm)
    before_pos, before_quat = env.get_robot_eef_pose(arm=arm)
    start_frame = env.get_video_frame_count()
    for _ in range(max(0, int(steps))):
        action = env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=True))
        env.step(action)
    after_pos, after_quat = env.get_robot_eef_pose(arm=arm)
    before_np = env._debug_list(before_pos)
    after_np = env._debug_list(after_pos)
    return {
        "steps": max(0, int(steps)),
        "video_frames_start": int(start_frame),
        "video_frames_end": int(env.get_video_frame_count()),
        "eef_pos_before": before_np,
        "eef_quat_before": env._debug_list(before_quat),
        "eef_pos_after": after_np,
        "eef_quat_after": env._debug_list(after_quat),
        "eef_pos_drift": round(float(np.linalg.norm(np.asarray(after_np) - np.asarray(before_np))), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-X X2 pure motion-primitives demo")
    parser.add_argument("--output-dir", default="outputs/x2_code_exec_motion_primitives_demo_v1")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--post-hold-steps", type=int, default=POST_HOLD_STEPS_DEFAULT)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "ok": False,
        "move_delta": MOVE_DELTA,
        "global_camera": GLOBAL_CAMERA,
        "fixed_code": MOTION_PRIMITIVES_CODE,
        "steps": {},
        "video": {},
        "errors": [],
    }

    try:
        low_level = X2BehaviourLowLevel(
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=1,
            robot_camera_resolution=256,
        )
        exec_env_cls = get_exec_env("x2_behavior_code_env")
        exec_env = exec_env_cls(CodeExecEnvConfig(low_level=low_level, apis=["X2ControlApi"]))
        exec_env.reset()
        exec_env.enable_video_capture(True, clear=True)

        _obs, reward, terminated, truncated, info = exec_env.step(MOTION_PRIMITIVES_CODE)
        result = exec_env._exec_globals.get("RESULT")
        passed, detail, checks = _validate_result(result, info)
        detail.update(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        summary["steps"]["motion_primitives"] = detail
        summary["checks"] = checks

        post_hold_steps = max(0, int(args.post_hold_steps))
        if post_hold_steps:
            summary["steps"]["post_hold"] = _hold_current_target(low_level, post_hold_steps, arm=1)

        frames = exec_env.get_video_frames()
        summary["video"] = _write_videos(output_dir, frames, fps=args.video_fps)
        summary["video_sources"] = getattr(low_level, "_last_video_sources", {})
        summary["ok"] = bool(passed)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    for view, meta in summary["video"].get("views", {}).items():
        if meta.get("written"):
            print(f"Wrote {view} video {meta['path']} ({meta['frame_count']} frames)")
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"Wrote combined video {meta['path']} ({meta['frame_count']} frames)")
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
