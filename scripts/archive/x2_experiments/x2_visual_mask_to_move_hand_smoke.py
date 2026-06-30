"""Closed-loop X2 smoke: oracle visual mask -> RGB-D pose -> move_hand.

This validates the interface between the visual geometry primitive and the
motion primitive before introducing real detector / segmenter masks.
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
from PIL import Image
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.control import X2ControlApi


OBJECT_NAME = "visual_target_marker"
OBJECT_SIZE = 0.02
ARM = 1
FIXED_REACHABLE_TARGET_POS = [0.275818, -0.227553, 0.938968]
STUCK_PATIENCE_STEPS = 45
MOVE_POS_THRESHOLD_M = 0.015
FINGER_CENTER_POS_THRESHOLD_M = 0.02
HOME_POS_THRESHOLD_M = 0.02
VISION_POS_WARN_THRESHOLD_M = 0.06
POST_HOLD_STEPS_DEFAULT = 0
GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb", "depth_linear"],
    "sensor_kwargs": {
        "image_height": 256,
        "image_width": 256,
    },
    # Fixed torso/ego-style view for the current fixed-base X2 phase.
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value: Any) -> list[float]:
    return [round(float(v), 6) for v in _as_numpy(value).reshape(-1)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _check(condition: bool, message: str) -> str:
    return "pass" if condition else f"fail: {message}"


def _global_camera_config(image_size: int) -> dict[str, Any]:
    camera = dict(GLOBAL_CAMERA)
    sensor_kwargs = dict(camera["sensor_kwargs"])
    sensor_kwargs["image_height"] = int(image_size)
    sensor_kwargs["image_width"] = int(image_size)
    camera["sensor_kwargs"] = sensor_kwargs
    return camera


def _save_rgb(path: Path, value: Any) -> None:
    arr = _as_numpy(value)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim < 3:
        return
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(arr)).save(path)


def _save_depth_preview(path: Path, value: Any) -> None:
    arr = np.squeeze(_as_numpy(value).astype(np.float32))
    finite = np.isfinite(arr)
    if arr.ndim != 2 or not finite.any():
        return
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi <= lo:
        return
    vis = np.zeros_like(arr, dtype=np.float32)
    vis[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(path)


def _save_mask_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    rgb = _as_numpy(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def _resize_nearest(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    y_idx = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
    return frame[y_idx][:, x_idx]


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> dict[str, Any]:
    if not frames:
        return {"path": None, "frame_count": 0, "written": False}
    arr = np.asarray(frames)
    media.write_video(path, arr, fps=fps)
    return {"path": str(path), "frame_count": int(len(frames)), "shape": list(arr.shape), "fps": fps, "written": True}


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


def _finger_center_offset_eef(gripper_state: dict[str, Any]) -> np.ndarray:
    finger_center = gripper_state.get("finger_center_eef")
    if finger_center is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(finger_center, dtype=np.float64).reshape(3)


def _world_from_eef_offset(eef_quat_xyzw: Any, offset_eef: np.ndarray) -> np.ndarray:
    return x2_vision.quat_xyzw_to_matrix(_as_numpy(eef_quat_xyzw)) @ np.asarray(offset_eef, dtype=np.float64).reshape(3)


def _hold_current_target(env: X2BehaviourLowLevel, steps: int, arm: int = ARM) -> dict[str, Any]:
    env._hold_current_hand_target(arm=arm)
    before_pos, before_quat = env.get_robot_eef_pose(arm=arm)
    start_frame = env.get_video_frame_count()
    for _ in range(max(0, int(steps))):
        action = env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=True))
        env.step(action)
    after_pos, after_quat = env.get_robot_eef_pose(arm=arm)
    return {
        "steps": max(0, int(steps)),
        "video_frames_start": int(start_frame),
        "video_frames_end": int(env.get_video_frame_count()),
        "eef_pos_before": _as_list(before_pos),
        "eef_quat_before": _as_list(before_quat),
        "eef_pos_after": _as_list(after_pos),
        "eef_quat_after": _as_list(after_quat),
        "eef_pos_drift": round(float(np.linalg.norm(_as_numpy(after_pos) - _as_numpy(before_pos))), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 closed-loop visual mask -> move_hand smoke test")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_visual_mask_to_move_hand_smoke")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bbox-margin-px", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--post-hold-steps", type=int, default=POST_HOLD_STEPS_DEFAULT)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    camera_dir = output_dir / "cameras"
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    global_camera = _global_camera_config(args.image_size)
    summary: dict[str, Any] = {
        "ok": False,
        "object_name": OBJECT_NAME,
        "object_size": OBJECT_SIZE,
        "target_marker_position": FIXED_REACHABLE_TARGET_POS,
        "global_camera": global_camera,
        "steps": {},
        "verdicts": {},
        "video": {},
        "errors": [],
    }

    try:
        objects = [
            {
                "type": "PrimitiveObject",
                "name": OBJECT_NAME,
                "primitive_type": "Sphere",
                "radius": OBJECT_SIZE / 2.0,
                "position": FIXED_REACHABLE_TARGET_POS,
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "kinematic_only": True,
                "visual_only": True,
                "rgba": [1.0, 0.05, 0.05, 1.0],
            }
        ]
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=objects,
            external_sensors=[global_camera],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=args.image_size,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
        )
        api = X2ControlApi(env)
        env.reset()
        env.enable_video_capture(True, clear=True)
        for _ in range(max(0, int(args.settle_steps))):
            env.step(env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=False)))

        initial_eef = api.get_current_eef_pose(arm=ARM)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        open_gripper_state = api.get_gripper_state(arm=ARM)

        camera_obs = api.get_external_camera_observation("global_camera")
        rgb = _as_numpy(camera_obs["rgb"])
        depth = np.squeeze(_as_numpy(camera_obs["depth_linear"]))
        K = api.get_external_camera_intrinsics("global_camera")
        cam_pos, cam_quat = api.get_external_camera_pose("global_camera")
        T_world_cam = x2_vision.pose_to_matrix(cam_pos, cam_quat)

        obj = env.env.scene.object_registry("name", OBJECT_NAME)
        if obj is None:
            raise RuntimeError(f"Object {OBJECT_NAME!r} not found")
        obj_pos, obj_quat = obj.get_position_orientation()
        obj_pos = _as_numpy(obj_pos).astype(np.float64)
        obj_quat = _as_numpy(obj_quat).astype(np.float64)
        aabb_center = _as_numpy(getattr(obj, "aabb_center", obj_pos)).astype(np.float64)
        aabb_extent = _as_numpy(getattr(obj, "aabb_extent", np.array([OBJECT_SIZE] * 3))).astype(np.float64)

        mask, mask_detail = x2_vision.make_projected_aabb_mask(
            aabb_center,
            aabb_extent,
            K,
            T_world_cam,
            depth.shape[:2],
            margin_px=args.bbox_margin_px,
            min_half_extent=0.0,
        )
        projected_center, center_valid = x2_vision.project_world_points(aabb_center[None, :], K, T_world_cam)
        expected_depth = x2_vision.expected_depth_for_world_point(aabb_center, T_world_cam)
        depth_window = max(0.025, float(np.linalg.norm(aabb_extent)) * 1.5)
        pose_estimate = api.estimate_position_from_mask(
            mask,
            camera_name="global_camera",
            external=True,
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        estimated_position = pose_estimate["position"]
        if estimated_position is None:
            raise RuntimeError("estimate_position_from_mask returned None")
        estimated_position = np.asarray(estimated_position, dtype=np.float64)

        _save_rgb(camera_dir / "global_camera_rgb.png", rgb)
        np.save(camera_dir / "global_camera_depth_linear.npy", depth)
        _save_depth_preview(camera_dir / "global_camera_depth_linear.preview.png", depth)
        np.save(camera_dir / "oracle_mask.npy", mask)
        _save_mask_overlay(camera_dir / "oracle_mask_overlay.png", rgb, mask)

        finger_center_offset_eef = _finger_center_offset_eef(open_gripper_state)
        target_eef_position = estimated_position - _world_from_eef_offset(initial_eef[1], finger_center_offset_eef)
        target_pose = (target_eef_position.astype(np.float32), np.asarray(initial_eef[1], dtype=np.float32))
        move_ok = api.move_hand(
            target_pose,
            arm=ARM,
            pos_thresh=0.005,
            ori_thresh=0.1,
            stop_if_stuck=True,
            stuck_patience_steps=STUCK_PATIENCE_STEPS,
            max_steps=1200,
        )
        reached_eef = api.get_current_eef_pose(arm=ARM)
        api.close_gripper(arm=ARM)
        api.settle_robot(steps=12)
        return_ok = api.move_hand(
            initial_eef,
            arm=ARM,
            pos_thresh=0.005,
            ori_thresh=0.1,
            stop_if_stuck=True,
            stuck_patience_steps=STUCK_PATIENCE_STEPS,
            max_steps=1200,
        )
        home_eef = api.get_current_eef_pose(arm=ARM)
        post_hold = _hold_current_target(env, args.post_hold_steps, arm=ARM) if args.post_hold_steps else None

        vision_error = float(np.linalg.norm(estimated_position - aabb_center))
        move_error = float(np.linalg.norm(_as_numpy(reached_eef[0]) - target_eef_position))
        reached_finger_center = _as_numpy(reached_eef[0]) + _world_from_eef_offset(reached_eef[1], finger_center_offset_eef)
        finger_center_error = float(np.linalg.norm(reached_finger_center - estimated_position))
        home_error = float(np.linalg.norm(_as_numpy(home_eef[0]) - _as_numpy(initial_eef[0])))

        summary["steps"]["visual_geometry"] = {
            "camera_obs_keys": sorted(camera_obs.keys()),
            "rgb_shape": list(np.asarray(rgb).shape),
            "depth_shape": list(np.asarray(depth).shape),
            "intrinsic_matrix": _jsonable(K),
            "camera_pose": {"position": _jsonable(cam_pos), "quat_xyzw": _jsonable(cam_quat)},
            "object_pose": {"position": _jsonable(obj_pos), "quat_xyzw": _jsonable(obj_quat)},
            "aabb_center": _jsonable(aabb_center),
            "aabb_extent": _jsonable(aabb_extent),
            "projected_center_uv": np.round(projected_center[0], 3).tolist(),
            "projected_center_valid": bool(center_valid[0]),
            "mask": mask_detail,
            "estimate": {
                "camera_name": pose_estimate["camera_name"],
                "depth_key": pose_estimate["depth_key"],
                "raw_point_count": pose_estimate["raw_point_count"],
                "filtered_point_count": pose_estimate["filtered_point_count"],
                "expected_depth": pose_estimate["expected_depth"],
                "depth_window": pose_estimate["depth_window"],
                "fallback_used": pose_estimate["fallback_used"],
                "position": np.round(estimated_position, 6).tolist(),
                "vision_error_m": round(vision_error, 6),
            },
            "saved_files": sorted(str(path) for path in camera_dir.iterdir()),
        }
        summary["steps"]["motion"] = {
            "initial_eef": {"position": _as_list(initial_eef[0]), "quat_xyzw": _as_list(initial_eef[1])},
            "open_gripper_state": open_gripper_state,
            "finger_center_offset_eef": np.round(finger_center_offset_eef, 6).tolist(),
            "target_pose_from_visual": {
                "visual_position": np.round(estimated_position, 6).tolist(),
                "eef_position": np.round(target_eef_position, 6).tolist(),
                "quat_xyzw": _as_list(initial_eef[1]),
            },
            "move_ok": bool(move_ok),
            "reached_eef": {"position": _as_list(reached_eef[0]), "quat_xyzw": _as_list(reached_eef[1])},
            "reached_finger_center": np.round(reached_finger_center, 6).tolist(),
            "close_gripper_state": api.get_gripper_state(arm=ARM),
            "return_ok": bool(return_ok),
            "home_eef": {"position": _as_list(home_eef[0]), "quat_xyzw": _as_list(home_eef[1])},
            "eef_target_error_m": round(move_error, 6),
            "finger_center_to_visual_target_error_m": round(finger_center_error, 6),
            "home_error_m": round(home_error, 6),
            "post_hold": post_hold,
        }

        frames = env.get_video_frames()
        summary["video"] = _write_videos(output_dir, frames, fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})

        checks = [
            _check(mask.sum() > 0, "oracle mask has no pixels"),
            _check(estimated_position.shape == (3,), f"estimated position shape {estimated_position.shape} != (3,)"),
            _check(vision_error < VISION_POS_WARN_THRESHOLD_M, f"vision diagnostic error {vision_error:.4f}m"),
            _check(bool(move_ok), "move_hand(target_pose_from_visual) returned False"),
            _check(move_error < MOVE_POS_THRESHOLD_M, f"EEF target error {move_error:.4f}m"),
            _check(
                finger_center_error < FINGER_CENTER_POS_THRESHOLD_M,
                f"finger center target error {finger_center_error:.4f}m",
            ),
            _check(bool(return_ok), "move_hand(initial_eef) returned False"),
            _check(home_error < HOME_POS_THRESHOLD_M, f"home error {home_error:.4f}m"),
            _check(summary["video"].get("combined", {}).get("written", False), "combined video was not written"),
        ]
        summary["verdicts"]["visual_mask_to_move_hand"] = {
            "passed": all(item.startswith("pass") for item in checks),
            "checks": checks,
        }
        summary["ok"] = bool(summary["verdicts"]["visual_mask_to_move_hand"]["passed"])
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])
    finally:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {summary_path}")
        print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
