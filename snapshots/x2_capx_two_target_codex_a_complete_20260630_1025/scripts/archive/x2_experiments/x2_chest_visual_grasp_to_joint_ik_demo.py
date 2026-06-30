"""Execute X2 joint-IK motions from the chest-camera visual grasp chain.

This is a controlled bridge test:

1. Detect and segment the red cube from the robot chest camera.
2. Estimate ``T_world_object`` from the SAM2 mask and chest RGB-D.
3. Generate Contact-GraspNet candidates and convert them to X2 ``T_world_tcp``.
4. Execute pregrasp, grasp, close, lift, and open using ``move_tcp_joint_ik``.

The red cube is kept fixed/kinematic in this demo.  The pass criterion is TCP
tracking of visual-chain targets, not physical object pickup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi
from capx.integrations.x2 import vision as x2_vision
from x2_chest_camera_visual_chain_smoke import (
    ARM,
    OBJECT_CENTER,
    OBJECT_NAME,
    PROMPTS,
    WORKSPACE_BOUNDS,
    _as_rgb_u8,
    _detect_and_segment,
    _jsonable,
    _mask_depth_hint,
    _save_depth_preview,
    _save_mask_overlay,
    _save_rgb,
    _scene_objects,
    _summarize_plan,
)
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_summary(pose: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    pos, quat = pose
    return {
        "position": np.round(np.asarray(pos, dtype=np.float64), 6).tolist(),
        "quat_xyzw": np.round(np.asarray(quat, dtype=np.float64), 6).tolist(),
    }


def _quat_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
    qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))


def _current_tcp(api: X2ControlApi, arm: int = ARM) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    eef_pos, eef_quat = api.get_current_eef_pose(arm=arm)
    offset = np.asarray(api.get_tcp_offset_eef(arm=arm), dtype=np.float64).reshape(3)
    tcp_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3) + x2_vision.quat_xyzw_to_matrix(eef_quat) @ offset
    return tcp_pos, (np.asarray(eef_pos, dtype=np.float64), np.asarray(eef_quat, dtype=np.float64))


def _execute_tcp_target(
    api: X2ControlApi,
    label: str,
    target_tcp_pose: tuple[np.ndarray, np.ndarray],
    *,
    arm: int = ARM,
    pos_thresh: float = 0.02,
    ori_thresh: float = 0.26,
    max_joint_step: float = 0.022,
    max_steps: int = 240,
    settle_steps: int = 20,
) -> dict[str, Any]:
    target_pos = np.asarray(target_tcp_pose[0], dtype=np.float64).reshape(3)
    target_quat = np.asarray(target_tcp_pose[1], dtype=np.float64).reshape(4)
    target_eef_pose = api.tcp_pose_to_eef_pose((target_pos, target_quat), arm=arm)
    start_tcp, start_eef = _current_tcp(api, arm=arm)
    q_before = np.asarray(api.get_current_joint_positions(), dtype=np.float64)
    t0 = time.time()
    ok = bool(
        api.move_tcp_joint_ik(
            (target_pos, target_quat),
            arm=arm,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            max_joint_step=max_joint_step,
            max_steps=max_steps,
            settle_steps=settle_steps,
        )
    )
    api.settle_robot(steps=12)
    reached_tcp, reached_eef = _current_tcp(api, arm=arm)
    q_after = np.asarray(api.get_current_joint_positions(), dtype=np.float64)
    tcp_error = float(np.linalg.norm(reached_tcp - target_pos))
    eef_pos_error = float(np.linalg.norm(np.asarray(reached_eef[0]) - np.asarray(target_eef_pose[0])))
    eef_ori_error = _quat_error_rad(np.asarray(reached_eef[1]), np.asarray(target_eef_pose[1]))
    return {
        "label": label,
        "ok": ok,
        "elapsed_s": round(time.time() - t0, 3),
        "start_tcp_position": np.round(start_tcp, 6).tolist(),
        "start_eef": _pose_summary(start_eef),
        "target_tcp_pose": _pose_summary((target_pos, target_quat)),
        "target_eef_pose": _pose_summary(target_eef_pose),
        "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
        "reached_eef": _pose_summary(reached_eef),
        "tcp_error_m": round(tcp_error, 6),
        "eef_pos_error_m": round(eef_pos_error, 6),
        "eef_ori_error_rad": round(eef_ori_error, 6),
        "max_abs_joint_delta_rad": round(float(np.max(np.abs(q_after - q_before))), 6),
        "last_motion_debug": _jsonable(api.get_last_motion_debug()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 chest visual grasp to joint IK demo")
    parser.add_argument("--config", default="x2_robotiq85_joint_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_chest_visual_grasp_to_joint_ik_demo")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--owlvit-device", default="cpu")
    parser.add_argument("--owlvit-threshold", type=float, default=0.03)
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--graspnet-device", default="cuda")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--tcp-error-threshold", type=float, default=0.035)
    parser.add_argument("--graspnet-forward-passes", type=int, default=4)
    parser.add_argument("--graspnet-max-retries", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = output_dir / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "object_name": OBJECT_NAME,
        "configured_object_center_world": np.round(OBJECT_CENTER, 6).tolist(),
        "camera_contract": "chest camera RGB-D pose is T_world_camera; object pose is T_world_object; execution targets are T_world_tcp",
        "execution_contract": "move_tcp_joint_ik receives T_world_tcp and internally converts to T_world_eef",
        "checks": [],
        "motions": [],
        "video": {},
        "errors": [],
    }
    start = time.time()
    env = None
    try:
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=_scene_objects(),
            external_sensors=[GLOBAL_CAMERA],
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
            use_graspnet=True,
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
                "graspnet": _summarize_plan(grasp_plan),
                "x2_execution_plan": {
                    "meaning": "candidate poses are T_world_tcp; pass pregrasp_tcp_pose/grasp_tcp_pose/lift_tcp_pose directly to move_tcp_joint_ik",
                    "ok": bool(x2_plan.get("ok", False)),
                    "strategy": x2_plan.get("strategy"),
                    "candidate_count": x2_plan.get("candidate_count"),
                    "error": x2_plan.get("error"),
                    "candidates": _jsonable(x2_plan.get("candidates", [])[:5]),
                },
            }
        )
        summary["checks"].extend(
            [
                "pass" if visual["mask_pixels"] >= 50 else f"fail: mask too small {visual['mask_pixels']}",
                "pass" if pos_error <= 0.08 else f"fail: pose error too large {pos_error:.4f}m",
                "pass" if grasp_plan.get("ok") else f"fail: graspnet {grasp_plan.get('error')}",
                "pass" if x2_plan.get("ok") else f"fail: x2 plan {x2_plan.get('error')}",
            ]
        )
        candidates = list(x2_plan.get("candidates", []) or [])
        if not candidates:
            raise RuntimeError(f"No executable X2 candidates: {x2_plan.get('error')}")
        candidate_index = int(np.clip(args.candidate_index, 0, len(candidates) - 1))
        candidate = candidates[candidate_index]
        summary["selected_candidate"] = _jsonable(candidate)

        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        for label, key in (
            ("pregrasp", "pregrasp_tcp_pose"),
            ("grasp", "grasp_tcp_pose"),
            ("lift", "lift_tcp_pose"),
        ):
            if label == "lift":
                api.close_gripper(arm=ARM)
                api.settle_robot(steps=16)
            move_result = _execute_tcp_target(api, label, candidate[key], arm=ARM)
            summary["motions"].append(move_result)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        env._record_frame()
        summary["video"] = _write_videos(output_dir, env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})

        motion_ok = all(
            bool(item["ok"]) and float(item["tcp_error_m"]) <= float(args.tcp_error_threshold)
            for item in summary["motions"]
        )
        summary["checks"].append("pass" if motion_ok else "fail: one or more TCP motions exceeded threshold")
        summary["ok"] = all(str(item) == "pass" for item in summary["checks"])
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], file=sys.stderr, flush=True)
    finally:
        summary["elapsed_s"] = round(time.time() - start, 3)
        (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"ok": summary["ok"], "checks": summary["checks"], "output_dir": str(output_dir)}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
