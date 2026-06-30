"""Sweep TCP orientations at one visual target position for X2.

The visual object pose is estimated once from RGB-D.  The TCP target position
is fixed relative to that visual object center.  Each candidate TCP
orientation starts from the same restored robot state, runs a pre-approach and
final hover, then records TCP position error and EEF/TCP orientation error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.control import X2ControlApi


ARM = 1
OBJECT_NAME = "x2_visual_orientation_cube"
TABLE_NAME = "x2_visual_orientation_support"
OBJECT_SIZE = 0.04
OBJECT_CENTER = np.array([0.37389, -0.163279, 0.945], dtype=np.float64)
TABLE_SCALE = np.array([0.10, 0.10, 0.012], dtype=np.float64)
TABLE_CENTER = np.array([0.37389, -0.163279, 0.925], dtype=np.float64)
EXPECTED_OBJECT_DEPTH_M = 1.113063
DEPTH_WINDOW_M = 0.086603
TCP_CLEARANCE_M = 0.055
PRE_APPROACH_LIFT_M = 0.035
POSITION_OK_M = 0.025
ORIENTATION_OK_RAD = 0.20
SAFE_TCP_QUAT_XYZW = np.array([0.648902, 0.169021, 0.73383, 0.108867], dtype=np.float64)
FIXED_TARGET_REACHED_QUAT_XYZW = np.array([0.623938, 0.172114, 0.755461, 0.101771], dtype=np.float64)

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb", "depth_linear"],
    "sensor_kwargs": {"image_height": 384, "image_width": 384},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value: Any, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in _as_np(value).reshape(-1)]


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


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = _quat_normalize(a)
    bx, by, bz, bw = _quat_normalize(b)
    return _quat_normalize(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )
    )


def _axis_angle_quat_xyzw(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * float(angle_rad)
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


def _quat_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(_quat_normalize(a), _quat_normalize(b))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _tcp_from_eef(api: X2ControlApi, eef_pose: tuple[np.ndarray, np.ndarray], arm: int = ARM) -> np.ndarray:
    eef_pos, eef_quat = eef_pose
    offset = api.get_tcp_offset_eef(arm=arm)
    return np.asarray(eef_pos, dtype=np.float64).reshape(3) + x2_vision.quat_xyzw_to_matrix(eef_quat) @ offset


def _restore_robot(env: X2BehaviourLowLevel, root_pose: tuple[Any, Any], q: Any, qd: Any, arm: int = ARM) -> None:
    env.robot.set_position_orientation(position=root_pose[0], orientation=root_pose[1])
    env.robot.set_joint_positions(q, drive=False)
    env.robot.set_joint_velocities(qd, drive=False)
    env._set_arm(arm)
    env._hold_current_hand_target(env._arm_name(arm))


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
            "rgba": [1.0, 0.04, 0.03, 1.0],
        },
    ]


def _candidate_quats(initial_quat: np.ndarray) -> list[dict[str, Any]]:
    initial_quat = _quat_normalize(initial_quat)
    candidates: list[dict[str, Any]] = [
        {"label": "initial_eef_quat", "quat_xyzw": initial_quat, "source": "current eef orientation after reset"},
        {"label": "safe_fixed_target_quat", "quat_xyzw": SAFE_TCP_QUAT_XYZW, "source": "previous fixed TCP target command"},
        {
            "label": "fixed_target_reached_quat",
            "quat_xyzw": FIXED_TARGET_REACHED_QUAT_XYZW,
            "source": "previous fixed TCP target reached EEF orientation",
        },
    ]
    axes = {
        "local_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "local_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "local_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    for base_label, base_quat in (
        ("safe", SAFE_TCP_QUAT_XYZW),
        ("reached", FIXED_TARGET_REACHED_QUAT_XYZW),
    ):
        for axis_name, axis in axes.items():
            for angle_deg in (-15.0, 15.0):
                dq = _axis_angle_quat_xyzw(axis, math.radians(angle_deg))
                candidates.append(
                    {
                        "label": f"{base_label}_{axis_name}_{angle_deg:+.0f}deg",
                        "quat_xyzw": _quat_multiply_xyzw(base_quat, dq),
                        "source": f"{base_label} quat post-multiplied by {angle_deg:+.0f}deg about {axis_name}",
                    }
                )
    return candidates


def _move_tcp_and_measure(
    api: X2ControlApi,
    target_tcp_pos: np.ndarray,
    target_tcp_quat: np.ndarray,
    *,
    label: str,
    max_steps: int,
    pos_thresh: float,
    ori_thresh: float,
) -> dict[str, Any]:
    env = api._env
    target_tcp_pos = np.asarray(target_tcp_pos, dtype=np.float64).reshape(3)
    target_tcp_quat = _quat_normalize(target_tcp_quat)
    target_eef = api.tcp_pose_to_eef_pose((target_tcp_pos, target_tcp_quat), arm=ARM)
    target_eef_pos = np.asarray(target_eef[0], dtype=np.float64).reshape(3)
    target_eef_quat = _quat_normalize(np.asarray(target_eef[1], dtype=np.float64).reshape(4))
    before_eef = api.get_current_eef_pose(arm=ARM)
    before_tcp = _tcp_from_eef(api, before_eef)
    start_frame = env.get_video_frame_count() if hasattr(env, "get_video_frame_count") else None
    start_time = time.time()
    ok = api.move_tcp(
        (target_tcp_pos, target_tcp_quat),
        arm=ARM,
        pos_thresh=pos_thresh,
        ori_thresh=ori_thresh,
        stop_if_stuck=True,
        stuck_patience_steps=40,
        stuck_pos_thresh=0.0001,
        stuck_ori_thresh=0.001,
        max_steps=max_steps,
    )
    api.settle_robot(steps=8)
    after_eef = api.get_current_eef_pose(arm=ARM)
    after_tcp = _tcp_from_eef(api, after_eef)
    after_eef_pos = np.asarray(after_eef[0], dtype=np.float64).reshape(3)
    after_eef_quat = _quat_normalize(np.asarray(after_eef[1], dtype=np.float64).reshape(4))
    end_frame = env.get_video_frame_count() if hasattr(env, "get_video_frame_count") else None
    tcp_pos_error = float(np.linalg.norm(after_tcp - target_tcp_pos))
    eef_pos_error = float(np.linalg.norm(after_eef_pos - target_eef_pos))
    ori_error = float(_quat_angle_error(after_eef_quat, target_eef_quat))
    return {
        "label": label,
        "ok": bool(ok),
        "elapsed_s": round(time.time() - start_time, 3),
        "target_tcp": {"position": _as_list(target_tcp_pos), "quat_xyzw": _as_list(target_tcp_quat)},
        "target_eef": {"position": _as_list(target_eef_pos), "quat_xyzw": _as_list(target_eef_quat)},
        "before_tcp_position": _as_list(before_tcp),
        "after_tcp_position": _as_list(after_tcp),
        "before_eef": {"position": _as_list(before_eef[0]), "quat_xyzw": _as_list(before_eef[1])},
        "after_eef": {"position": _as_list(after_eef[0]), "quat_xyzw": _as_list(after_eef[1])},
        "tcp_pos_error_m": round(tcp_pos_error, 6),
        "eef_pos_error_m": round(eef_pos_error, 6),
        "eef_ori_error_rad": round(ori_error, 6),
        "eef_ori_error_deg": round(math.degrees(ori_error), 3),
        "pos_thresh": float(pos_thresh),
        "ori_thresh": float(ori_thresh),
        "max_steps": int(max_steps),
        "video_frames_start": start_frame,
        "video_frames_end": end_frame,
    }


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
    return {"path": str(path), "frame_count": int(len(frames)), "shape": list(arr.shape), "fps": int(fps), "written": True}


def _write_videos(output_dir: Path, frames: Any, fps: int) -> dict[str, Any]:
    frames_by_view = frames if isinstance(frames, dict) else {"rgb": frames}
    result: dict[str, Any] = {"views": {}, "fps": int(fps)}
    for view, view_frames in frames_by_view.items():
        result["views"][view] = _write_video(output_dir / f"{view}.mp4", view_frames, fps=fps)
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
    parser = argparse.ArgumentParser(description="Sweep X2 TCP orientations at a fixed visual target")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/x2_visual_tcp_orientation_sweep"))
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=15)
    parser.add_argument("--pos-threshold", type=float, default=0.008)
    parser.add_argument("--ori-threshold", type=float, default=ORIENTATION_OK_RAD)
    parser.add_argument("--position-ok-threshold", type=float, default=POSITION_OK_M)
    parser.add_argument("--orientation-ok-threshold", type=float, default=ORIENTATION_OK_RAD)
    parser.add_argument("--ik-pos-gain", type=float, default=1.0)
    parser.add_argument("--ik-ori-gain", type=float, default=0.2)
    parser.add_argument("--isaac-kp", type=float, default=None)
    parser.add_argument("--isaac-kd", type=float, default=None)
    parser.add_argument("--smoothing-filter-size", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "purpose": "fixed visual target position, TCP orientation sweep",
        "target_contract": {
            "visual_pose": "T_world_object_center from RGB-D get_object_pose",
            "tcp_target": "T_world_tcp/finger_center = visual object center + z clearance, with swept quaternion",
            "eef_target": "T_world_eef converted from T_world_tcp by tcp_pose_to_eef_pose",
        },
        "scene": {
            "object_name": OBJECT_NAME,
            "object_center_configured": _as_list(OBJECT_CENTER),
            "object_is_visual_only": False,
            "tcp_clearance_m": TCP_CLEARANCE_M,
            "pre_approach_lift_m": PRE_APPROACH_LIFT_M,
        },
        "controller_tuning": {
            "ik_pose_delta_pos_gain": float(args.ik_pos_gain),
            "ik_pose_delta_ori_gain": float(args.ik_ori_gain),
            "isaac_kp_override": args.isaac_kp,
            "isaac_kd_override": args.isaac_kd,
            "smoothing_filter_size_override": args.smoothing_filter_size,
        },
        "visual": {},
        "initial_state": {},
        "candidates": [],
        "stable_candidates": [],
        "ranked_candidates": [],
        "video": {},
        "errors": [],
    }

    start = time.time()
    try:
        arm_controller_override: dict[str, Any] = {}
        if args.isaac_kp is not None:
            arm_controller_override["isaac_kp"] = float(args.isaac_kp)
        if args.isaac_kd is not None:
            arm_controller_override["isaac_kd"] = float(args.isaac_kd)
        if args.smoothing_filter_size is not None:
            arm_controller_override["smoothing_filter_size"] = int(args.smoothing_filter_size)
        env = X2BehaviourLowLevel(
            objects=_scene_objects(),
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=384,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
            ik_pose_delta_pos_gain=float(args.ik_pos_gain),
            ik_pose_delta_ori_gain=float(args.ik_ori_gain),
            arm_controller_override=arm_controller_override or None,
        )
        api = X2ControlApi(env)
        env.reset()
        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=24)

        initial_root_pose = env.robot.get_position_orientation()
        initial_q = env.robot.get_joint_positions().clone()
        initial_qd = env.robot.get_joint_velocities().clone()
        initial_eef = api.get_current_eef_pose(arm=ARM)
        initial_tcp = _tcp_from_eef(api, initial_eef)

        obj = env.env.scene.object_registry("name", OBJECT_NAME)
        truth_center = _as_np(obj.aabb_center).astype(np.float64).reshape(3)
        object_pos, object_quat, object_extent = api.get_object_pose(
            OBJECT_NAME,
            return_bbox_extent=True,
            camera_name="global_camera",
            external=True,
            expected_depth=EXPECTED_OBJECT_DEPTH_M,
            depth_window=DEPTH_WINDOW_M,
            method="aabb_center",
        )
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4)
        object_extent = np.asarray(object_extent, dtype=np.float64).reshape(3)
        target_tcp_pos = object_pos + np.array([0.0, 0.0, TCP_CLEARANCE_M], dtype=np.float64)
        pre_tcp_pos = target_tcp_pos + np.array([0.0, 0.0, PRE_APPROACH_LIFT_M], dtype=np.float64)

        summary["visual"] = {
            "object_position_world": _as_list(object_pos),
            "object_quat_xyzw": _as_list(object_quat),
            "object_extent": _as_list(object_extent),
            "truth_object_center_world": _as_list(truth_center),
            "visual_object_error_to_truth_m": round(float(np.linalg.norm(object_pos - truth_center)), 6),
            "expected_depth_m": EXPECTED_OBJECT_DEPTH_M,
            "depth_window_m": DEPTH_WINDOW_M,
            "fixed_target_tcp_position_world": _as_list(target_tcp_pos),
            "pre_approach_tcp_position_world": _as_list(pre_tcp_pos),
            "last_object_pose_estimate": _jsonable(getattr(api, "_last_object_pose_estimate", {})),
        }
        summary["initial_state"] = {
            "eef": {"position": _as_list(initial_eef[0]), "quat_xyzw": _as_list(initial_eef[1])},
            "tcp_position": _as_list(initial_tcp),
            "tcp_offset_eef": _as_list(api.get_tcp_offset_eef(arm=ARM)),
            "gripper": api.get_gripper_state(arm=ARM),
        }
        summary["controller_tuning"]["loaded_robot_controller_config"] = _jsonable(
            getattr(env.robot, "_controller_config", {})
        )

        candidates = _candidate_quats(np.asarray(initial_eef[1], dtype=np.float64))
        candidates = candidates[: max(1, int(args.max_candidates))]
        for idx, candidate in enumerate(candidates):
            _restore_robot(env, initial_root_pose, initial_q, initial_qd, arm=ARM)
            api.settle_robot(steps=8)
            quat = _quat_normalize(np.asarray(candidate["quat_xyzw"], dtype=np.float64))
            print(f"[x2-orientation-sweep] candidate {idx + 1}/{len(candidates)} {candidate['label']}", flush=True)
            pre = _move_tcp_and_measure(
                api,
                pre_tcp_pos,
                quat,
                label=f"{candidate['label']}::pre_approach",
                max_steps=650,
                pos_thresh=0.010,
                ori_thresh=float(args.ori_threshold),
            )
            if pre["tcp_pos_error_m"] <= float(args.position_ok_threshold):
                final = _move_tcp_and_measure(
                    api,
                    target_tcp_pos,
                    quat,
                    label=f"{candidate['label']}::target",
                    max_steps=700,
                    pos_thresh=float(args.pos_threshold),
                    ori_thresh=float(args.ori_threshold),
                )
            else:
                final = {"label": f"{candidate['label']}::target", "skipped": True, "reason": "pre_approach_position_failed"}

            target_pos_ok = (not final.get("skipped")) and float(final["tcp_pos_error_m"]) <= float(args.position_ok_threshold)
            target_ori_ok = (not final.get("skipped")) and float(final["eef_ori_error_rad"]) <= float(args.orientation_ok_threshold)
            stable = bool(target_pos_ok and target_ori_ok and final.get("ok"))
            item = {
                "index": idx,
                "label": candidate["label"],
                "source": candidate["source"],
                "target_tcp_quat_xyzw": _as_list(quat),
                "pre_approach": pre,
                "target": final,
                "position_ok": bool(target_pos_ok),
                "orientation_ok": bool(target_ori_ok),
                "stable": stable,
            }
            summary["candidates"].append(item)

        ranked = sorted(
            summary["candidates"],
            key=lambda item: (
                not bool(item.get("stable")),
                float(item.get("target", {}).get("tcp_pos_error_m", 999.0)),
                float(item.get("target", {}).get("eef_ori_error_rad", 999.0)),
            ),
        )
        summary["ranked_candidates"] = [
            {
                "rank": i + 1,
                "label": item["label"],
                "stable": bool(item["stable"]),
                "target_tcp_quat_xyzw": item["target_tcp_quat_xyzw"],
                "tcp_pos_error_m": item.get("target", {}).get("tcp_pos_error_m"),
                "eef_ori_error_rad": item.get("target", {}).get("eef_ori_error_rad"),
                "eef_ori_error_deg": item.get("target", {}).get("eef_ori_error_deg"),
                "move_ok": item.get("target", {}).get("ok"),
            }
            for i, item in enumerate(ranked)
        ]
        summary["stable_candidates"] = [item for item in summary["ranked_candidates"] if item["stable"]]
        summary["video"] = _write_videos(args.output_dir, env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})
        summary["checks"] = {
            "visual_object_pose_ok": summary["visual"]["visual_object_error_to_truth_m"] <= POSITION_OK_M,
            "stable_candidate_count": len(summary["stable_candidates"]),
            "position_ok_threshold_m": float(args.position_ok_threshold),
            "orientation_ok_threshold_rad": float(args.orientation_ok_threshold),
            "orientation_ok_threshold_deg": round(math.degrees(float(args.orientation_ok_threshold)), 3),
        }
        summary["ok"] = bool(summary["checks"]["visual_object_pose_ok"] and len(summary["stable_candidates"]) > 0)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)

    summary["elapsed_s"] = round(time.time() - start, 3)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-orientation-sweep] wrote {output_path}", flush=True)
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "stable_candidate_count": len(summary.get("stable_candidates", [])),
                "best": (summary.get("ranked_candidates") or [None])[0],
                "errors": len(summary["errors"]),
                "elapsed_s": summary["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
