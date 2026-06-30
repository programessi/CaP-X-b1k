"""X2 smoke: deterministic visual output -> TCP pose -> move_hand target.

This test keeps the visual model mocked.  It validates that the geometry
primitive produces a world-frame object pose that can be converted into the
EEF pose consumed by the X2 motion primitive.
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
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.control import X2ControlApi


OBJECT_NAME = "visual_tcp_target"
OBJECT_DIAMETER = 0.02
OBJECT_POS = [0.34, -0.17, 0.95]
ARM = 1
GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb", "depth_linear"],
    "sensor_kwargs": {"image_height": 256, "image_width": 256},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value: Any) -> list[float]:
    return [round(float(v), 6) for v in _to_numpy(value).reshape(-1)]


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


def _camera_config(image_size: int) -> dict[str, Any]:
    camera = dict(GLOBAL_CAMERA)
    sensor_kwargs = dict(camera["sensor_kwargs"])
    sensor_kwargs["image_height"] = int(image_size)
    sensor_kwargs["image_width"] = int(image_size)
    camera["sensor_kwargs"] = sensor_kwargs
    return camera


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


def _world_from_eef_offset(quat_xyzw: np.ndarray, offset_eef: np.ndarray) -> np.ndarray:
    return x2_vision.quat_xyzw_to_matrix(quat_xyzw) @ offset_eef


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 deterministic visual output -> TCP target smoke")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_visual_to_tcp_pose_smoke")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bbox-margin-px", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--pre-hold-steps", type=int, default=30)
    parser.add_argument("--post-hold-steps", type=int, default=30)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "object_name": OBJECT_NAME,
        "object_diameter": OBJECT_DIAMETER,
        "model_output_source": "deterministic_oracle_projected_aabb_mask",
        "steps": {},
        "checks": [],
        "video": {},
        "errors": [],
    }

    try:
        objects = [
            {
                "type": "PrimitiveObject",
                "name": OBJECT_NAME,
                "primitive_type": "Sphere",
                "radius": OBJECT_DIAMETER / 2.0,
                "position": OBJECT_POS,
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
            external_sensors=[_camera_config(args.image_size)],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=args.image_size,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
        )
        api = X2ControlApi(env)
        env.reset()
        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        api.settle_robot(steps=args.pre_hold_steps)

        initial_eef = api.get_current_eef_pose(arm=ARM)
        bundle = api._camera_bundle(camera_name="global_camera", external=True)
        obj = env.env.scene.object_registry("name", OBJECT_NAME)
        aabb_center = _to_numpy(obj.aabb_center).astype(np.float64)
        aabb_extent = _to_numpy(obj.aabb_extent).astype(np.float64)
        T_world_cam = x2_vision.pose_to_matrix(bundle["camera_position"], bundle["camera_quat_xyzw"])
        mask, mask_detail = x2_vision.make_projected_aabb_mask(
            aabb_center,
            aabb_extent,
            bundle["intrinsic_matrix"],
            T_world_cam,
            bundle["depth"].shape[:2],
            margin_px=args.bbox_margin_px,
            min_half_extent=0.0,
        )
        expected_depth = x2_vision.expected_depth_for_world_point(aabb_center, T_world_cam)
        depth_window = max(0.025, float(np.linalg.norm(aabb_extent)) * 1.5)

        object_pos, object_quat, bbox_extent = api.get_object_pose(
            OBJECT_NAME,
            return_bbox_extent=True,
            mask=mask,
            camera_name="global_camera",
            external=True,
            method="obb_center",
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        object_pos = np.asarray(object_pos, dtype=np.float64)
        object_quat = np.asarray(object_quat, dtype=np.float64)

        # This is the intended visual primitive output consumed by grasp logic:
        # a desired TCP pose in world frame.  For this smoke, keep TCP
        # orientation equal to the current EEF orientation and validate the
        # position/frame conversion independently.
        tcp_target_pose = (object_pos, np.asarray(initial_eef[1], dtype=np.float64))
        tcp_offset_eef = api.get_tcp_offset_eef(arm=ARM)
        eef_target_pose = api.tcp_pose_to_eef_pose(tcp_target_pose, arm=ARM, tcp_offset_eef=tcp_offset_eef)

        move_ok = api.move_hand(
            eef_target_pose,
            arm=ARM,
            pos_thresh=0.005,
            ori_thresh=0.1,
            stop_if_stuck=True,
            stuck_patience_steps=45,
            max_steps=1200,
        )
        api.settle_robot(steps=args.post_hold_steps)
        reached_eef = api.get_current_eef_pose(arm=ARM)
        reached_tcp = _to_numpy(reached_eef[0]) + _world_from_eef_offset(_to_numpy(reached_eef[1]), tcp_offset_eef)

        object_pose_error = float(np.linalg.norm(object_pos - aabb_center))
        eef_target_error = float(np.linalg.norm(_to_numpy(reached_eef[0]) - np.asarray(eef_target_pose[0])))
        tcp_target_error = float(np.linalg.norm(reached_tcp - object_pos))
        tcp_to_true_object_error = float(np.linalg.norm(reached_tcp - aabb_center))

        projected_object_uv, projected_valid = x2_vision.project_world_points(
            object_pos[None, :],
            bundle["intrinsic_matrix"],
            T_world_cam,
        )
        x0, y0, x1, y1 = mask_detail["bbox_xyxy"]
        u, v = projected_object_uv[0]
        projection_in_mask_bbox = bool(projected_valid[0] and x0 <= u <= x1 and y0 <= v <= y1)

        summary["steps"]["visual_model_output"] = {
            "source": summary["model_output_source"],
            "mask": mask_detail,
            "mask_pixels": int(mask.sum()),
            "expected_depth": expected_depth,
            "depth_window": depth_window,
        }
        summary["steps"]["visual_object_pose"] = {
            "position": np.round(object_pos, 6).tolist(),
            "quat_xyzw": np.round(object_quat, 6).tolist(),
            "bbox_extent": None if bbox_extent is None else np.round(bbox_extent, 6).tolist(),
            "true_aabb_center": np.round(aabb_center, 6).tolist(),
            "true_aabb_extent": np.round(aabb_extent, 6).tolist(),
            "object_pose_error_m": round(object_pose_error, 6),
            "projected_object_uv": np.round(projected_object_uv[0], 3).tolist(),
            "projection_in_mask_bbox": projection_in_mask_bbox,
            "last_pose_estimate": _jsonable(getattr(api, "_last_object_pose_estimate", {})),
        }
        summary["steps"]["tcp_to_action_target"] = {
            "tcp_target_pose": {
                "position": np.round(tcp_target_pose[0], 6).tolist(),
                "quat_xyzw": np.round(tcp_target_pose[1], 6).tolist(),
            },
            "tcp_offset_eef": np.round(tcp_offset_eef, 6).tolist(),
            "eef_target_pose_for_move_hand": {
                "position": np.round(eef_target_pose[0], 6).tolist(),
                "quat_xyzw": np.round(eef_target_pose[1], 6).tolist(),
            },
            "move_hand_contract": {
                "position_shape": list(np.asarray(eef_target_pose[0]).shape),
                "quat_shape": list(np.asarray(eef_target_pose[1]).shape),
                "frame": "world",
                "quat_format": "xyzw",
            },
        }
        summary["steps"]["motion_result"] = {
            "move_ok": bool(move_ok),
            "initial_eef": {"position": _as_list(initial_eef[0]), "quat_xyzw": _as_list(initial_eef[1])},
            "reached_eef": {"position": _as_list(reached_eef[0]), "quat_xyzw": _as_list(reached_eef[1])},
            "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
            "eef_target_error_m": round(eef_target_error, 6),
            "tcp_target_error_m": round(tcp_target_error, 6),
            "tcp_to_true_object_error_m": round(tcp_to_true_object_error, 6),
        }
        summary["video"] = _write_videos(output_dir, env.get_video_frames(), fps=args.video_fps)

        checks = [
            _check(mask.sum() > 0, "deterministic mask is empty"),
            _check(np.asarray(object_pos).shape == (3,), "visual object position is not shape (3,)"),
            _check(np.asarray(object_quat).shape == (4,), "visual object quat is not shape (4,)"),
            _check(projection_in_mask_bbox, "visual object position does not project back into model mask bbox"),
            _check(np.asarray(eef_target_pose[0]).shape == (3,), "move_hand target position is not shape (3,)"),
            _check(np.asarray(eef_target_pose[1]).shape == (4,), "move_hand target quat is not shape (4,)"),
            _check(bool(move_ok), "move_hand returned False for visual-derived EEF target"),
            _check(eef_target_error < 0.015, f"EEF did not reach action target: {eef_target_error:.4f}m"),
            _check(tcp_target_error < 0.02, f"TCP did not reach visual TCP target: {tcp_target_error:.4f}m"),
            _check(summary["video"].get("combined", {}).get("written", False), "combined video was not written"),
        ]
        summary["checks"] = checks
        summary["ok"] = all(check == "pass" for check in checks)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
