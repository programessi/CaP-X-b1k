"""CAP-X X2 grasp-only demo based on the stable objectless task-template motion."""

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
import torch

import capx.envs.simulators  # noqa: F401 - registers low-level envs
import capx.envs.tasks  # noqa: F401 - registers code-execution envs/configs
import capx.integrations  # noqa: F401 - registers APIs
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.envs.tasks import CodeExecEnvConfig, get_exec_env

OBJECT_NAME = "demo_grasp_cube"
OBJECT_SIZE = 0.025
MOVE_DELTA = [0.08, 0.0, -0.04]
STUCK_PATIENCE_STEPS = 45

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


MOVE_TO_STABLE_POSE = r'''
import numpy as np

ARM = 1
MOVE_DELTA = np.array([0.08, 0.0, -0.04], dtype=np.float32)
STUCK_PATIENCE_STEPS = 45

def pose_to_list(pose):
    pos, quat = pose
    return {
        "pos": np.asarray(pos, dtype=np.float32).round(6).tolist(),
        "quat": np.asarray(quat, dtype=np.float32).round(6).tolist(),
    }

initial_eef = get_current_eef_pose(arm=ARM)
open_gripper(arm=ARM)
settle_robot(steps=12)
target_pose = (np.asarray(initial_eef[0], dtype=np.float32) + MOVE_DELTA, np.asarray(initial_eef[1], dtype=np.float32))
move_ok = move_hand(
    target_pose,
    arm=ARM,
    pos_thresh=0.005,
    ori_thresh=0.1,
    stop_if_stuck=True,
    stuck_patience_steps=STUCK_PATIENCE_STEPS,
)
reached_eef = get_current_eef_pose(arm=ARM)
gripper = get_gripper_state(arm=ARM)

RESULT = {
    "initial_eef": pose_to_list(initial_eef),
    "target_pose": pose_to_list(target_pose),
    "reached_eef": pose_to_list(reached_eef),
    "gripper": gripper,
    "move_ok": bool(move_ok),
    "target_eef_error": round(float(np.linalg.norm(np.asarray(reached_eef[0]) - np.asarray(target_pose[0]))), 6),
}
'''


CLOSE_ON_PLACED_OBJECT = r'''
import numpy as np

ARM = 1
OBJECT_NAME = "demo_grasp_cube"

def pose_to_list(pose):
    pos, quat = pose
    return {
        "pos": np.asarray(pos, dtype=np.float32).round(6).tolist(),
        "quat": np.asarray(quat, dtype=np.float32).round(6).tolist(),
    }

before_close_eef = get_current_eef_pose(arm=ARM)
before_object = get_object_pose(OBJECT_NAME)
before_gripper = get_gripper_state(arm=ARM)
close_gripper(arm=ARM)
settle_robot(steps=24)
after_close_eef = get_current_eef_pose(arm=ARM)
after_object = get_object_pose(OBJECT_NAME)
after_gripper = get_gripper_state(arm=ARM)
try:
    grasp_success = bool(check_object_in_hand(arm=ARM))
except Exception:
    grasp_success = False

RESULT = {
    "before_close_eef": pose_to_list(before_close_eef),
    "before_object": pose_to_list(before_object),
    "before_gripper": before_gripper,
    "after_close_eef": pose_to_list(after_close_eef),
    "after_object": pose_to_list(after_object),
    "after_gripper": after_gripper,
    "grasp_success": bool(grasp_success),
    "eef_drift_after_close": round(float(np.linalg.norm(np.asarray(after_close_eef[0]) - np.asarray(before_close_eef[0]))), 6),
    "object_motion_after_close": round(float(np.linalg.norm(np.asarray(after_object[0]) - np.asarray(before_object[0]))), 6),
}
'''


def _as_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value) -> list[float]:
    return [round(float(v), 6) for v in _as_np(value).reshape(-1)]


def _quat_xyzw_apply(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat_xyzw[:3]
    q_w = quat_xyzw[3]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + q_w * t + np.cross(q_xyz, t)


def _place_object_in_right_gripper(env: X2BehaviourLowLevel) -> dict[str, Any]:
    eef_pos, eef_quat = env.get_robot_eef_pose(arm=1)
    gripper = env.get_gripper_state(arm=1)
    finger_center_eef = np.asarray(gripper.get("finger_center_eef") or [0.0, 0.0, 0.1046], dtype=np.float64)
    object_pos = _as_np(eef_pos).astype(np.float64) + _quat_xyzw_apply(_as_np(eef_quat).astype(np.float64), finger_center_eef)

    obj = env.env.scene.object_registry("name", OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Object {OBJECT_NAME!r} not found")
    obj.set_position_orientation(
        position=torch.tensor(object_pos, dtype=torch.float32),
        orientation=torch.tensor(_as_np(eef_quat), dtype=torch.float32),
    )
    obj.keep_still()
    placed_pos, placed_quat = obj.get_position_orientation()

    env.step(env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=False)))
    after_step_pos, after_step_quat = obj.get_position_orientation()
    return {
        "eef_pos": _as_list(eef_pos),
        "eef_quat_xyzw": _as_list(eef_quat),
        "finger_center_eef": _as_list(finger_center_eef),
        "requested_object_pos": _as_list(object_pos),
        "placed_object_pos": _as_list(placed_pos),
        "placed_object_quat_xyzw": _as_list(placed_quat),
        "after_one_step_object_pos": _as_list(after_step_pos),
        "after_one_step_object_quat_xyzw": _as_list(after_step_quat),
        "object_size": OBJECT_SIZE,
        "fixed_base": False,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-X X2 grasp-only demo")
    parser.add_argument("--output-dir", default="outputs/x2_code_exec_grasp_only_demo_v1")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "ok": False,
        "object_name": OBJECT_NAME,
        "object_size": OBJECT_SIZE,
        "move_delta": MOVE_DELTA,
        "global_camera": GLOBAL_CAMERA,
        "steps": {},
        "video": {},
        "errors": [],
    }

    try:
        objects = [
            {
                "type": "PrimitiveObject",
                "name": OBJECT_NAME,
                "primitive_type": "Cube",
                "size": OBJECT_SIZE,
                "position": [1.2, 0.5, 0.5],
                "orientation": [0, 0, 0, 1],
                "fixed_base": False,
                "rgba": [1.0, 0.05, 0.05, 1.0],
            }
        ]
        low_level = X2BehaviourLowLevel(
            objects=objects,
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=1,
            robot_camera_resolution=256,
        )
        exec_env_cls = get_exec_env("x2_behavior_code_env")
        exec_env = exec_env_cls(CodeExecEnvConfig(low_level=low_level, apis=["X2ControlApi"]))
        exec_env.reset()
        exec_env.enable_video_capture(True, clear=True)

        _obs, move_reward, move_terminated, move_truncated, move_info = exec_env.step(MOVE_TO_STABLE_POSE)
        move_result = exec_env._exec_globals.get("RESULT")
        summary["steps"]["move_to_stable_pose"] = {
            "result": move_result,
            "sandbox_rc": move_info.get("sandbox_rc"),
            "reward": float(move_reward),
            "terminated": bool(move_terminated),
            "truncated": bool(move_truncated),
            "stderr_tail": move_info.get("stderr", "")[-4000:],
        }

        summary["steps"]["place_object_in_open_gripper"] = _place_object_in_right_gripper(low_level)

        _obs, close_reward, close_terminated, close_truncated, close_info = exec_env.step(CLOSE_ON_PLACED_OBJECT)
        close_result = exec_env._exec_globals.get("RESULT")
        summary["steps"]["close_on_placed_object"] = {
            "result": close_result,
            "sandbox_rc": close_info.get("sandbox_rc"),
            "reward": float(close_reward),
            "terminated": bool(close_terminated),
            "truncated": bool(close_truncated),
            "stderr_tail": close_info.get("stderr", "")[-4000:],
        }

        frames = exec_env.get_video_frames()
        summary["video"] = _write_videos(output_dir, frames, fps=args.video_fps)
        summary["video_sources"] = getattr(low_level, "_last_video_sources", {})

        move_ok = bool(move_result and move_result.get("move_ok") and move_result.get("target_eef_error", 999.0) < 0.01)
        close_ok = bool(close_result and close_result.get("grasp_success"))
        summary["ok"] = move_ok and close_ok
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    summary_path = output_dir / "summary.json"
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
