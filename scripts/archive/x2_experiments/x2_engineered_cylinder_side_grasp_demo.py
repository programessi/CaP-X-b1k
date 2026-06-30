"""Short engineered X2 side-grasp demo for a vertical cylinder.

No vision model and no GraspNet are used here.  The cylinder is placed in the
open right gripper's finger center after reset, then the gripper closes and the
hand performs one short lift/translate motion.
"""

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

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi


ARM = 1
OBJECT_NAME = "engineered_vertical_cylinder"
SUPPORT_NAME = "engineered_cylinder_support"
CYLINDER_RADIUS = 0.012
CYLINDER_HEIGHT = 0.055
SUPPORT_THICKNESS = 0.012
MAX_MOVE_STEPS = 240
STUCK_PATIENCE_STEPS = 14

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb"],
    "sensor_kwargs": {"image_height": 512, "image_width": 512},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value: Any) -> list[float]:
    return [round(float(v), 6) for v in _as_np(value).reshape(-1)]


def _quat_xyzw_apply(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _object_state(env: X2BehaviourLowLevel) -> dict[str, Any]:
    obj = env.env.scene.object_registry("name", OBJECT_NAME)
    if obj is None:
        return {"found": False}
    pos, quat = obj.get_position_orientation()
    result = {"found": True, "position": _as_list(pos), "quat_xyzw": _as_list(quat)}
    for attr in ("aabb_center", "aabb_extent"):
        try:
            result[attr] = _as_list(getattr(obj, attr))
        except Exception:
            pass
    return result


def _place_cylinder_in_open_gripper(env: X2BehaviourLowLevel, api: X2ControlApi) -> dict[str, Any]:
    eef_pos, eef_quat = api.get_current_eef_pose(arm=ARM)
    gripper = api.get_gripper_state(arm=ARM)
    finger_center_eef = np.asarray(gripper.get("finger_center_eef") or [0.0, 0.0, 0.1046], dtype=np.float64)
    object_pos = np.asarray(eef_pos, dtype=np.float64) + _quat_xyzw_apply(np.asarray(eef_quat), finger_center_eef)

    obj = env.env.scene.object_registry("name", OBJECT_NAME)
    if obj is None:
        raise RuntimeError(f"Object {OBJECT_NAME!r} not found")
    obj.set_position_orientation(
        position=torch.tensor(object_pos, dtype=torch.float32),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    )
    obj.keep_still()
    support = env.env.scene.object_registry("name", SUPPORT_NAME)
    support_state = None
    if support is not None:
        support_pos = object_pos - np.array([0.0, 0.0, CYLINDER_HEIGHT / 2.0 + SUPPORT_THICKNESS / 2.0])
        support.set_position_orientation(
            position=torch.tensor(support_pos, dtype=torch.float32),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        )
        support.keep_still()
        support_state = {
            "position": _as_list(support_pos),
            "top_z": round(float(support_pos[2] + SUPPORT_THICKNESS / 2.0), 6),
        }
    return {
        "eef": {"position": _as_list(eef_pos), "quat_xyzw": _as_list(eef_quat)},
        "open_gripper": gripper,
        "finger_center_eef": _as_list(finger_center_eef),
        "requested_object_position": _as_list(object_pos),
        "support": support_state,
        "object_after_place": _object_state(env),
    }


def _move_eef_short(api: X2ControlApi, target_pos: np.ndarray, target_quat: np.ndarray, label: str) -> dict[str, Any]:
    before = api.get_current_eef_pose(arm=ARM)
    ok = api.move_hand(
        (np.asarray(target_pos, dtype=np.float64), np.asarray(target_quat, dtype=np.float64)),
        arm=ARM,
        pos_thresh=0.008,
        ori_thresh=0.35,
        stop_if_stuck=True,
        stuck_patience_steps=STUCK_PATIENCE_STEPS,
        max_steps=MAX_MOVE_STEPS,
    )
    after = api.get_current_eef_pose(arm=ARM)
    return {
        "label": label,
        "ok": bool(ok),
        "before_eef": {"position": _as_list(before[0]), "quat_xyzw": _as_list(before[1])},
        "target_eef": {"position": _as_list(target_pos), "quat_xyzw": _as_list(target_quat)},
        "after_eef": {"position": _as_list(after[0]), "quat_xyzw": _as_list(after[1])},
        "eef_target_error_m": round(float(np.linalg.norm(np.asarray(after[0]) - np.asarray(target_pos))), 6),
        "max_steps": MAX_MOVE_STEPS,
        "stuck_patience_steps": STUCK_PATIENCE_STEPS,
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
    views = [view for view in ("global", "robot", "rgb") if frames_by_view.get(view)]
    combined = []
    if len(views) >= 2:
        frame_count = min(len(frames_by_view[view]) for view in views)
        height = max(frames_by_view[view][0].shape[0] for view in views)
        combined = [
            np.concatenate([_resize_nearest(frames_by_view[view][i], height) for view in views], axis=1)
            for i in range(frame_count)
        ]
    result["combined"] = _write_video(output_dir / "video_combined.mp4", combined, fps=fps)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Engineered X2 vertical-cylinder side grasp demo")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_engineered_cylinder_side_grasp_demo")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "ok": False,
        "mode": "engineered_no_vision_no_graspnet",
        "object": {"name": OBJECT_NAME, "radius": CYLINDER_RADIUS, "height": CYLINDER_HEIGHT},
        "max_move_steps": MAX_MOVE_STEPS,
        "stuck_patience_steps": STUCK_PATIENCE_STEPS,
        "steps": {},
        "video": {},
        "errors": [],
    }

    try:
        objects = [
            {
                "type": "PrimitiveObject",
                "name": SUPPORT_NAME,
                "primitive_type": "Cube",
                "size": 1.0,
                "scale": [0.042, 0.042, SUPPORT_THICKNESS],
                "position": [1.4, 0.4, 0.7],
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "rgba": [0.36, 0.36, 0.36, 1.0],
            },
            {
                "type": "PrimitiveObject",
                "name": OBJECT_NAME,
                "primitive_type": "Cylinder",
                "radius": CYLINDER_RADIUS,
                "height": CYLINDER_HEIGHT,
                "position": [1.4, 0.4, 0.8],
                "orientation": [0, 0, 0, 1],
                "fixed_base": False,
                "rgba": [1.0, 0.04, 0.03, 1.0],
            }
        ]
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=objects,
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=256,
            robot_obs_modalities=["rgb"],
        )
        api = X2ControlApi(env)
        env.reset()
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)

        summary["steps"]["place_cylinder_in_open_gripper"] = _place_cylinder_in_open_gripper(env, api)

        env.enable_video_capture(True, clear=True)
        env._record_frame()
        summary["steps"]["before_close_object"] = _object_state(env)
        summary["steps"]["before_close_gripper"] = api.get_gripper_state(arm=ARM)

        api.close_gripper(arm=ARM)
        api.settle_robot(steps=18)
        summary["steps"]["after_close_object"] = _object_state(env)
        summary["steps"]["after_close_gripper"] = api.get_gripper_state(arm=ARM)
        try:
            summary["steps"]["in_hand_after_close"] = bool(api.check_object_in_hand(arm=ARM))
        except Exception as exc:
            summary["steps"]["in_hand_after_close"] = False
            summary["steps"]["check_object_in_hand_error"] = str(exc)

        current_eef = api.get_current_eef_pose(arm=ARM)
        current_pos = np.asarray(current_eef[0], dtype=np.float64)
        current_quat = np.asarray(current_eef[1], dtype=np.float64)
        lift_target = current_pos + np.array([0.0, 0.0, 0.045], dtype=np.float64)
        summary["steps"]["lift_move"] = _move_eef_short(api, lift_target, current_quat, "short_lift_after_close")
        api.settle_robot(steps=12)
        summary["steps"]["after_lift_object"] = _object_state(env)

        side_target = np.asarray(api.get_current_eef_pose(arm=ARM)[0], dtype=np.float64) + np.array([0.035, 0.0, 0.0])
        summary["steps"]["short_translate_move"] = _move_eef_short(api, side_target, current_quat, "short_translate_after_lift")
        api.settle_robot(steps=12)
        summary["steps"]["final_object"] = _object_state(env)
        summary["steps"]["final_gripper"] = api.get_gripper_state(arm=ARM)

        summary["video"] = _write_videos(output_dir, env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})

        before = np.asarray(summary["steps"]["before_close_object"].get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)
        after_lift = np.asarray(summary["steps"]["after_lift_object"].get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)
        lifted = bool(np.all(np.isfinite(before)) and np.all(np.isfinite(after_lift)) and after_lift[2] - before[2] > 0.015)
        video_ok = bool(summary["video"].get("combined", {}).get("written", False))
        summary["lifted_delta_m"] = None if not np.all(np.isfinite(before)) else round(float(after_lift[2] - before[2]), 6)
        summary["ok"] = bool(video_ok and (summary["steps"]["in_hand_after_close"] or lifted))
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"Wrote combined video {meta['path']} ({meta['frame_count']} frames)")
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
