"""Validate X2 chest-camera visual primitives on a simple red cube.

This smoke test uses the torso-mounted robot camera, not the external
``global_camera``.  It checks the real visual chain:

1. OWL-ViT detection.
2. SAM2 segmentation.
3. RGB-D mask backprojection to a world-frame object pose.
4. Optional Contact-GraspNet sampling and X2 TCP grasp-plan conversion.

No robot motion is executed here; the goal is to verify the visual output
contract before connecting it to the joint-IK action path.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi


ARM = 1
OBJECT_NAME = "x2_chest_red_cube"
TABLE_NAME = "x2_chest_table"
OBJECT_SIZE = 0.04
OBJECT_CENTER = np.array([0.34, -0.14, 0.921], dtype=np.float64)
TABLE_CENTER = np.array([0.34, -0.14, 0.895], dtype=np.float64)
TABLE_SCALE = np.array([0.20, 0.18, 0.012], dtype=np.float64)
PROMPTS = ["red cube", "red block", "red box", "cube"]
WORKSPACE_BOUNDS = {"x": (0.16, 0.50), "y": (-0.38, 0.06), "z": (0.78, 1.08)}


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


def _as_rgb_u8(value: Any) -> np.ndarray:
    arr = _as_numpy(value)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _save_rgb(path: Path, value: Any) -> None:
    Image.fromarray(_as_rgb_u8(value)).save(path)


def _save_depth_preview(path: Path, value: Any) -> None:
    arr = np.squeeze(_as_numpy(value)).astype(np.float32)
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


def _save_mask_overlay(path: Path, rgb: np.ndarray, box: list[float], mask: np.ndarray, label: str) -> None:
    base = _as_rgb_u8(rgb)
    overlay = base.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=3)
    draw.text((max(0, x1), max(0, y1 - 14)), label, fill=(0, 255, 255))
    image.save(path)


def _best_by_score(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return max(items, key=lambda item: float(item.get("score", 0.0)))


def _mask_depth_hint(mask: np.ndarray, depth: np.ndarray) -> tuple[float | None, float | None]:
    values = np.squeeze(depth)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None, None
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    depth_window = max(0.025, float(q75 - q25) * 1.5 + 0.01)
    return float(q50), float(depth_window)


def _scene_objects() -> list[dict[str, Any]]:
    return [
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
            "rgba": [1.0, 0.02, 0.02, 1.0],
        },
    ]


def _detect_and_segment(api: X2ControlApi, rgb: np.ndarray, prompts: list[str]) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for prompt in prompts:
        for det in api.detect_object_owlvit(rgb, prompt):
            det = dict(det)
            det.setdefault("prompt", prompt)
            detections.append(det)
    detection = _best_by_score(detections)
    if detection is None:
        raise RuntimeError(f"OWL-ViT found no detection for prompts={prompts}")
    masks = api.segment_sam2(rgb, box=detection["box"], max_masks=3)
    mask_result = _best_by_score(masks)
    if mask_result is None:
        raise RuntimeError(f"SAM2 returned no masks for detection={detection}")
    mask = np.asarray(mask_result["mask"], dtype=bool)
    return {
        "prompts": prompts,
        "detections": detections,
        "detection": detection,
        "masks": masks,
        "mask_result": mask_result,
        "mask": mask,
        "mask_pixels": int(mask.sum()),
    }


def _pose_summary(pose: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    pos, quat = pose
    return {
        "position": np.round(np.asarray(pos, dtype=np.float64), 6).tolist(),
        "quat_xyzw": np.round(np.asarray(quat, dtype=np.float64), 6).tolist(),
    }


def _summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": candidate.get("name"),
        "source": candidate.get("source"),
        "variant": candidate.get("variant"),
        "rank": candidate.get("rank"),
        "score": None if candidate.get("score") is None else round(float(candidate.get("score", 0.0)), 6),
        "pregrasp_pose": _pose_summary(candidate.get("pregrasp_pose")),
        "grasp_pose": _pose_summary(candidate.get("grasp_pose")),
        "raw_graspnet_pose": _pose_summary(candidate.get("raw_graspnet_pose")),
        "contact_point_world": None
        if candidate.get("contact_point_world") is None
        else np.round(np.asarray(candidate["contact_point_world"], dtype=np.float64), 6).tolist(),
    }


def _summarize_plan(plan: dict[str, Any], max_candidates: int = 5) -> dict[str, Any]:
    return {
        "ok": bool(plan.get("ok", False)),
        "source": plan.get("source"),
        "error": plan.get("error"),
        "camera_name": plan.get("camera_name"),
        "external": plan.get("external"),
        "depth_key": plan.get("depth_key"),
        "mask_source": _jsonable(plan.get("mask_source")),
        "raw_grasp_count": plan.get("raw_grasp_count"),
        "filtered_grasp_count": plan.get("filtered_grasp_count"),
        "candidate_count": plan.get("candidate_count"),
        "candidates": [_summarize_candidate(candidate) for candidate in plan.get("candidates", [])[:max_candidates]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 chest-camera visual chain smoke")
    parser.add_argument("--config", default="x2_robotiq85_joint_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_chest_camera_visual_chain_smoke")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--skip-graspnet", action="store_true")
    parser.add_argument("--owlvit-device", default="cpu")
    parser.add_argument("--owlvit-threshold", type=float, default=0.03)
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--graspnet-device", default="cuda")
    parser.add_argument("--graspnet-forward-passes", type=int, default=4)
    parser.add_argument("--graspnet-max-retries", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = output_dir / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "config": args.config,
        "object_name": OBJECT_NAME,
        "configured_object_center_world": np.round(OBJECT_CENTER, 6).tolist(),
        "configured_table_center_world": np.round(TABLE_CENTER, 6).tolist(),
        "camera_contract": "robot chest camera RGB-D, pose T_world_camera; visual pose output is T_world_object; X2 grasp execution plan outputs T_world_tcp",
        "checks": [],
    }
    env = None
    try:
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=_scene_objects(),
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
            robot_camera_resolution=args.image_size,
            chest_camera=True,
            chest_camera_resolution=args.image_size,
            save_video=False,
        )
        api = X2ControlApi(
            env,
            use_vision_models=True,
            owlvit_device=args.owlvit_device,
            owlvit_threshold=args.owlvit_threshold,
            sam2_device=args.sam2_device,
            use_graspnet=not args.skip_graspnet,
            graspnet_device=args.graspnet_device,
        )
        api.settle_robot(steps=8)
        chest_name = api.get_chest_camera_name()
        obs = api.get_chest_camera_observation()
        rgb = _as_rgb_u8(obs["rgb"])
        depth = obs.get("depth_linear", obs.get("depth"))
        if depth is None:
            raise RuntimeError("Chest camera observation has no depth/depth_linear")
        cam_pos, cam_quat = api.get_chest_camera_pose()
        K = api.get_chest_camera_intrinsics()
        _save_rgb(camera_dir / "chest_rgb.png", rgb)
        np.save(camera_dir / "chest_depth_linear.npy", _as_numpy(depth))
        _save_depth_preview(camera_dir / "chest_depth_linear.preview.png", depth)

        visual = _detect_and_segment(api, rgb, PROMPTS)
        _save_mask_overlay(
            camera_dir / "chest_sam2_mask_overlay.png",
            rgb,
            visual["detection"]["box"],
            visual["mask"],
            f"{visual['detection'].get('prompt', '')} {float(visual['detection'].get('score', 0.0)):.2f}",
        )
        expected_depth, depth_window = _mask_depth_hint(visual["mask"], depth)
        object_pos, object_quat, bbox_extent = api.get_object_pose(
            visual["detection"].get("prompt", OBJECT_NAME),
            return_bbox_extent=True,
            mask=visual["mask"],
            camera_name=chest_name,
            external=False,
            method="aabb_center",
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        object_pos = np.asarray(object_pos, dtype=np.float64)
        object_quat = np.asarray(object_quat, dtype=np.float64)
        bbox_extent = None if bbox_extent is None else np.asarray(bbox_extent, dtype=np.float64)
        pos_error = float(np.linalg.norm(object_pos - OBJECT_CENTER))

        summary.update(
            {
                "camera": {
                    "name": chest_name,
                    "position_world": np.round(np.asarray(cam_pos, dtype=np.float64), 6).tolist(),
                    "quat_xyzw_world": np.round(np.asarray(cam_quat, dtype=np.float64), 6).tolist(),
                    "intrinsic_matrix": np.round(K, 6).tolist(),
                },
                "artifacts": {
                    "rgb": str(camera_dir / "chest_rgb.png"),
                    "depth": str(camera_dir / "chest_depth_linear.npy"),
                    "depth_preview": str(camera_dir / "chest_depth_linear.preview.png"),
                    "mask_overlay": str(camera_dir / "chest_sam2_mask_overlay.png"),
                },
                "visual": {
                    "prompts": PROMPTS,
                    "detection": _jsonable(visual["detection"]),
                    "detection_count": len(visual["detections"]),
                    "mask_pixels": visual["mask_pixels"],
                    "expected_depth": expected_depth,
                    "depth_window": depth_window,
                },
                "pose_estimate": {
                    "meaning": "T_world_object estimated from SAM2 mask and chest RGB-D depth",
                    "position_world": np.round(object_pos, 6).tolist(),
                    "quat_xyzw_world": np.round(object_quat, 6).tolist(),
                    "bbox_extent": None if bbox_extent is None else np.round(bbox_extent, 6).tolist(),
                    "position_error_to_configured_center_m": round(pos_error, 6),
                    "last_pose_estimate": _jsonable(getattr(api, "_last_object_pose_estimate", {})),
                },
            }
        )
        summary["checks"].extend(
            [
                "pass" if chest_name and "lidar_chest_front" in chest_name else f"fail: unexpected chest camera {chest_name}",
                "pass" if visual["mask_pixels"] >= 50 else f"fail: mask too small {visual['mask_pixels']}",
                "pass" if pos_error <= 0.08 else f"fail: pose error too large {pos_error:.4f}m",
            ]
        )

        if not args.skip_graspnet:
            initial_quat = api.get_current_eef_pose(arm=ARM)[1]
            grasp_plan = api.sample_grasp_pose_graspnet(
                OBJECT_NAME,
                mask=visual["mask"],
                camera_name=chest_name,
                arm=ARM,
                external=False,
                orientation_quat_xyzw=initial_quat,
                include_simple_fallback=False,
                max_candidates=24,
                min_mask_pixels=12,
                expected_depth=expected_depth,
                depth_window=depth_window,
                workspace_bounds=WORKSPACE_BOUNDS,
                forward_passes=args.graspnet_forward_passes,
                max_retries=args.graspnet_max_retries,
            )
            x2_plan = api.plan_x2_grasp_execution(
                grasp_plan,
                object_pos,
                bbox_extent=bbox_extent,
                arm=ARM,
                orientation_quat_xyzw=initial_quat,
                workspace_bounds=WORKSPACE_BOUNDS,
                max_candidates=12,
            )
            summary["graspnet"] = _summarize_plan(grasp_plan)
            summary["x2_execution_plan"] = {
                "meaning": "candidate poses are T_world_tcp; pass pregrasp_tcp_pose/grasp_tcp_pose/lift_tcp_pose directly to move_tcp_joint_ik",
                "ok": bool(x2_plan.get("ok", False)),
                "strategy": x2_plan.get("strategy"),
                "candidate_count": x2_plan.get("candidate_count"),
                "error": x2_plan.get("error"),
                "candidates": _jsonable(x2_plan.get("candidates", [])[:5]),
            }
            summary["checks"].append("pass" if grasp_plan.get("ok") else f"fail: graspnet {grasp_plan.get('error')}")
            summary["checks"].append("pass" if x2_plan.get("ok") else f"fail: x2 plan {x2_plan.get('error')}")

        summary["ok"] = all(str(item) == "pass" for item in summary["checks"])
    except Exception as exc:
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        print(summary["traceback"], file=sys.stderr)
    finally:
        (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True))

    print(json.dumps({"ok": summary["ok"], "checks": summary["checks"], "output_dir": str(output_dir)}, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
