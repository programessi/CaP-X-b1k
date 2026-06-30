"""X2 RGB-D pose contract smoke without IK.

This isolates the visual geometry contract before action execution:

    truth object world pose/AABB -> projected mask -> depth samples
    -> RGB-D backprojection -> estimated world position

No vision model and no arm motion are used.  The object is a non-visual-only
primitive cube so it should produce stable RGB and depth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.control import X2ControlApi


OBJECT_NAME = "x2_rgbd_contract_cube"
TABLE_NAME = "x2_rgbd_contract_support"
OBJECT_SIZE = 0.04
OBJECT_CENTER = np.array([0.33889, -0.188279, 0.945], dtype=np.float64)
TABLE_SCALE = np.array([0.10, 0.10, 0.012], dtype=np.float64)
TABLE_CENTER = np.array([0.33889, -0.188279, 0.925], dtype=np.float64)
POSITION_ERROR_THRESHOLD_M = 0.025

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb", "depth_linear"],
    "sensor_kwargs": {"image_height": 384, "image_width": 384},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def _rgb_u8(value: Any) -> np.ndarray:
    rgb = _as_numpy(value)
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _save_depth_preview(path: Path, depth: np.ndarray) -> None:
    arr = np.squeeze(np.asarray(depth, dtype=np.float32))
    finite = np.isfinite(arr) & (arr > 0)
    if arr.ndim != 2 or not finite.any():
        return
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi <= lo:
        return
    vis = np.zeros_like(arr, dtype=np.float32)
    vis[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(path)


def _save_mask_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: list[int] | None = None) -> None:
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=bool)
    overlay[mask] = (0.50 * overlay[mask] + 0.50 * np.array([255, 0, 0])).astype(np.uint8)
    image = Image.fromarray(overlay)
    if bbox_xyxy is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle([int(v) for v in bbox_xyxy], outline=(255, 255, 0), width=2)
    image.save(path)


def _camera_config(image_size: int) -> dict[str, Any]:
    camera = dict(GLOBAL_CAMERA)
    sensor_kwargs = dict(camera["sensor_kwargs"])
    sensor_kwargs["image_height"] = int(image_size)
    sensor_kwargs["image_width"] = int(image_size)
    camera["sensor_kwargs"] = sensor_kwargs
    return camera


def _depth_stats(depth: np.ndarray, mask: np.ndarray, expected_depth: float) -> dict[str, Any]:
    values = np.squeeze(np.asarray(depth, dtype=np.float64))[np.asarray(mask, dtype=bool)]
    valid = values[np.isfinite(values) & (values > 0)]
    if len(valid) == 0:
        return {"count": int(len(values)), "valid_count": 0}
    quantiles = np.percentile(valid, [0, 5, 25, 50, 75, 95, 100])
    close_2cm = int(np.count_nonzero(np.abs(valid - expected_depth) <= 0.02))
    close_5cm = int(np.count_nonzero(np.abs(valid - expected_depth) <= 0.05))
    return {
        "count": int(len(values)),
        "valid_count": int(len(valid)),
        "min_p05_p25_p50_p75_p95_max": np.round(quantiles, 6).tolist(),
        "expected_depth": round(float(expected_depth), 6),
        "median_minus_expected_m": round(float(np.median(valid) - expected_depth), 6),
        "fraction_within_2cm": round(float(close_2cm / len(valid)), 6),
        "fraction_within_5cm": round(float(close_5cm / len(valid)), 6),
    }


def _estimate_summary(result: dict[str, Any], truth: np.ndarray) -> dict[str, Any]:
    position = result.get("position")
    if position is None:
        error = None
    else:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        error = float(np.linalg.norm(position - truth))
    return {
        "position_world": None if position is None else np.round(position, 6).tolist(),
        "position_error_m": None if error is None else round(error, 6),
        "raw_point_count": int(result.get("raw_point_count", 0)),
        "filtered_point_count": int(result.get("filtered_point_count", 0)),
        "fallback_used": result.get("fallback_used"),
        "camera_name": result.get("camera_name"),
        "depth_key": result.get("depth_key"),
    }


def _check(condition: bool, message: str) -> str:
    return "pass" if condition else f"fail: {message}"


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 RGB-D pose contract smoke without IK")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_rgbd_pose_contract_smoke")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--bbox-margin-px", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=24)
    parser.add_argument("--position-error-threshold", type=float, default=POSITION_ERROR_THRESHOLD_M)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    camera_dir = output_dir / "cameras"
    camera_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "ok": False,
        "purpose": "verify RGB-D world pose contract before connecting IK",
        "object_name": OBJECT_NAME,
        "target_contract": {
            "reference_frame": "world",
            "estimated_quantity": "object/aabb center position, not TCP and not EEF",
            "action_connection": "only connect IK after this estimate can be converted into a valid T_world_tcp target",
        },
        "steps": {},
        "checks": [],
        "artifacts": {},
        "errors": [],
    }

    try:
        objects = [
            {
                "type": "PrimitiveObject",
                "name": TABLE_NAME,
                "primitive_type": "Cube",
                "size": 1.0,
                "scale": TABLE_SCALE.tolist(),
                "position": TABLE_CENTER.tolist(),
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "rgba": [0.35, 0.35, 0.35, 1.0],
            },
            {
                "type": "PrimitiveObject",
                "name": OBJECT_NAME,
                "primitive_type": "Cube",
                "size": OBJECT_SIZE,
                "position": OBJECT_CENTER.tolist(),
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "kinematic_only": True,
                "rgba": [1.0, 0.04, 0.03, 1.0],
            },
        ]
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=objects,
            external_sensors=[_camera_config(args.image_size)],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=1,
            robot_camera_resolution=args.image_size,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
        )
        api = X2ControlApi(env)
        env.reset()
        api.settle_robot(steps=max(0, int(args.settle_steps)))

        camera_obs = api.get_external_camera_observation("global_camera")
        rgb = _rgb_u8(camera_obs["rgb"])
        depth = np.squeeze(_as_numpy(camera_obs["depth_linear"]))
        K = api.get_external_camera_intrinsics("global_camera")
        cam_pos, cam_quat = api.get_external_camera_pose("global_camera")
        T_world_cam = x2_vision.pose_to_matrix(cam_pos, cam_quat)

        obj = env.env.scene.object_registry("name", OBJECT_NAME)
        if obj is None:
            raise RuntimeError(f"Object {OBJECT_NAME!r} not found")
        obj_pos, obj_quat = obj.get_position_orientation()
        obj_pos = _as_numpy(obj_pos).astype(np.float64).reshape(3)
        obj_quat = _as_numpy(obj_quat).astype(np.float64).reshape(4)
        aabb_center = _as_numpy(getattr(obj, "aabb_center", obj_pos)).astype(np.float64).reshape(3)
        aabb_extent = _as_numpy(getattr(obj, "aabb_extent", np.array([OBJECT_SIZE] * 3))).astype(np.float64).reshape(3)

        mask, mask_detail = x2_vision.make_projected_aabb_mask(
            aabb_center,
            aabb_extent,
            K,
            T_world_cam,
            depth.shape[:2],
            margin_px=args.bbox_margin_px,
            min_half_extent=0.0,
        )
        expected_depth = x2_vision.expected_depth_for_world_point(aabb_center, T_world_cam)
        depth_window = max(0.025, float(np.linalg.norm(aabb_extent)) * 1.25)
        projected_center_uv, projected_valid = x2_vision.project_world_points(aabb_center[None, :], K, T_world_cam)

        explicit_estimate = api.estimate_position_from_mask(
            mask,
            camera_name="global_camera",
            external=True,
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        default_pose_pos, default_pose_quat, default_extent = api.get_object_pose(
            OBJECT_NAME,
            return_bbox_extent=True,
            camera_name="global_camera",
            external=True,
            expected_depth=expected_depth,
            depth_window=depth_window,
            method="aabb_center",
        )
        default_pose_pos = np.asarray(default_pose_pos, dtype=np.float64).reshape(3)
        default_pose_quat = np.asarray(default_pose_quat, dtype=np.float64).reshape(4)
        default_pose_error = float(np.linalg.norm(default_pose_pos - aabb_center))

        rgb_path = camera_dir / "global_camera_rgb.png"
        depth_path = camera_dir / "global_camera_depth_linear.npy"
        depth_preview_path = camera_dir / "global_camera_depth_linear_preview.png"
        mask_path = camera_dir / "projected_aabb_mask.npy"
        overlay_path = camera_dir / "projected_aabb_mask_overlay.png"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)
        _save_depth_preview(depth_preview_path, depth)
        np.save(mask_path, mask)
        _save_mask_overlay(overlay_path, rgb, mask, mask_detail.get("bbox_xyxy"))

        explicit_summary = _estimate_summary(explicit_estimate, aabb_center)
        default_summary = {
            "position_world": np.round(default_pose_pos, 6).tolist(),
            "quat_xyzw": np.round(default_pose_quat, 6).tolist(),
            "bbox_extent": None if default_extent is None else np.round(default_extent, 6).tolist(),
            "position_error_m": round(default_pose_error, 6),
        }
        depth_summary = _depth_stats(depth, mask, expected_depth)

        bbox_xyxy = mask_detail.get("bbox_xyxy")
        u, v = projected_center_uv[0]
        center_in_bbox = bool(
            projected_valid[0]
            and bbox_xyxy is not None
            and bbox_xyxy[0] <= u <= bbox_xyxy[2]
            and bbox_xyxy[1] <= v <= bbox_xyxy[3]
        )
        explicit_error = explicit_summary["position_error_m"]

        checks = [
            _check(rgb.shape[:2] == depth.shape[:2], f"rgb/depth shape mismatch: {rgb.shape} vs {depth.shape}"),
            _check(bool(projected_valid[0]), "truth aabb center does not project in front of camera"),
            _check(center_in_bbox, "projected truth center is outside projected mask bbox"),
            _check(int(mask.sum()) > 0, "projected mask is empty"),
            _check(depth_summary.get("valid_count", 0) > 0, "mask has no valid positive depth"),
            _check(
                explicit_error is not None and explicit_error <= float(args.position_error_threshold),
                f"explicit backproject estimate error {explicit_error} exceeds {args.position_error_threshold}m",
            ),
            _check(
                default_pose_error <= float(args.position_error_threshold),
                f"default get_object_pose error {default_pose_error} exceeds {args.position_error_threshold}m",
            ),
        ]

        summary["steps"] = {
            "scene_truth": {
                "object_position_world": np.round(obj_pos, 6).tolist(),
                "object_quat_xyzw_world": np.round(obj_quat, 6).tolist(),
                "aabb_center_world": np.round(aabb_center, 6).tolist(),
                "aabb_extent_m": np.round(aabb_extent, 6).tolist(),
                "configured_object_center_world": np.round(OBJECT_CENTER, 6).tolist(),
            },
            "camera": {
                "name": "global_camera",
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth.shape),
                "intrinsic_matrix": _jsonable(K),
                "position_world": _jsonable(cam_pos),
                "quat_xyzw_world": _jsonable(cam_quat),
                "projected_aabb_center_uv": np.round(projected_center_uv[0], 3).tolist(),
                "projected_aabb_center_valid": bool(projected_valid[0]),
            },
            "projected_mask": {
                **_jsonable(mask_detail),
                "mask_pixels": int(mask.sum()),
                "center_in_mask_bbox": center_in_bbox,
            },
            "masked_depth": depth_summary,
            "explicit_estimate_from_same_mask": explicit_summary,
            "default_get_object_pose": default_summary,
            "thresholds": {
                "position_error_threshold_m": float(args.position_error_threshold),
                "depth_window_m": round(float(depth_window), 6),
            },
        }
        summary["checks"] = checks
        summary["ok"] = all(item == "pass" for item in checks)
        summary["artifacts"] = {
            "rgb": str(rgb_path),
            "depth_linear_npy": str(depth_path),
            "depth_preview": str(depth_preview_path),
            "mask_npy": str(mask_path),
            "mask_overlay": str(overlay_path),
        }
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)
    finally:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[x2-rgbd-contract] wrote {summary_path}", flush=True)
        print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
