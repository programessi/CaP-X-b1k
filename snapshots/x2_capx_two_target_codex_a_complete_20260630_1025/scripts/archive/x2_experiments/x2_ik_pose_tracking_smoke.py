"""Check X2 IK pose tracking around the current CAP-X grasp targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.control import X2ControlApi


ARM = 1
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


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value: Any) -> list[float]:
    return [round(float(v), 6) for v in _as_np(value).reshape(-1)]


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
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / axis_norm
    half = float(angle_rad) * 0.5
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


def _quat_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    qa = _quat_normalize(a)
    qb = _quat_normalize(b)
    dot = abs(float(np.dot(qa, qb)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * math.acos(dot)


def _tcp_from_eef(api: X2ControlApi, eef_pose: tuple[np.ndarray, np.ndarray], arm: int) -> np.ndarray:
    eef_pos, eef_quat = eef_pose
    offset = api.get_tcp_offset_eef(arm=arm)
    return np.asarray(eef_pos, dtype=np.float64).reshape(3) + x2_vision.quat_xyzw_to_matrix(eef_quat) @ offset


def _move_and_measure(
    api: X2ControlApi,
    target_pose: tuple[np.ndarray, np.ndarray],
    *,
    label: str,
    arm: int,
    max_steps: int,
    pos_thresh: float,
    ori_thresh: float,
) -> dict[str, Any]:
    target_pos = np.asarray(target_pose[0], dtype=np.float64).reshape(3)
    target_quat = _quat_normalize(np.asarray(target_pose[1], dtype=np.float64).reshape(4))
    env = api._env
    before = api.get_current_eef_pose(arm=arm)
    before_tcp = _tcp_from_eef(api, before, arm)
    start_frame = env.get_video_frame_count() if hasattr(env, "get_video_frame_count") else None
    trace: list[dict[str, Any]] = []
    steps = 0
    ok = True
    error_message = None
    target_t = (
        torch.tensor(target_pos, dtype=torch.float32),
        torch.tensor(target_quat, dtype=torch.float32),
    )

    try:
        env._set_arm(arm)
        gen = env._move_hand_direct_ik_with_stuck_patience(
            target_t,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            stop_if_stuck=True,
            stuck_patience_steps=45,
            stuck_pos_thresh=0.0001,
            stuck_ori_thresh=0.001,
            max_steps=max_steps,
        )
        for steps, action in enumerate(gen, start=1):
            if action is not None:
                env.step(action)
            current = api.get_current_eef_pose(arm=arm)
            current_pos = np.asarray(current[0], dtype=np.float64).reshape(3)
            current_quat = _quat_normalize(np.asarray(current[1], dtype=np.float64).reshape(4))
            trace.append(
                {
                    "step": int(steps),
                    "eef_pos_error_m": round(float(np.linalg.norm(current_pos - target_pos)), 6),
                    "eef_ori_error_rad": round(float(_quat_angle_error(current_quat, target_quat)), 6),
                    "eef_position": _as_list(current_pos),
                    "eef_quat_xyzw": _as_list(current_quat),
                }
            )
    except Exception as exc:  # noqa: BLE001 - diagnostic script records failures
        ok = False
        error_message = f"{type(exc).__name__}: {exc}"

    end_frame = env.get_video_frame_count() if hasattr(env, "get_video_frame_count") else None
    after = api.get_current_eef_pose(arm=arm)
    after_pos = np.asarray(after[0], dtype=np.float64).reshape(3)
    after_quat = _quat_normalize(np.asarray(after[1], dtype=np.float64).reshape(4))
    after_tcp = _tcp_from_eef(api, after, arm)
    before_quat = _quat_normalize(np.asarray(before[1], dtype=np.float64).reshape(4))
    actual_orientation_delta_rad = _quat_angle_error(after_quat, before_quat)
    target_orientation_delta_rad = _quat_angle_error(target_quat, before_quat)
    return {
        "label": label,
        "ok": bool(ok),
        "error_message": error_message,
        "target_eef": {"position": _as_list(target_pos), "quat_xyzw": _as_list(target_quat)},
        "before_eef": {"position": _as_list(before[0]), "quat_xyzw": _as_list(before[1])},
        "after_eef": {"position": _as_list(after_pos), "quat_xyzw": _as_list(after_quat)},
        "before_tcp_position": _as_list(before_tcp),
        "after_tcp_position": _as_list(after_tcp),
        "eef_pos_error_m": round(float(np.linalg.norm(after_pos - target_pos)), 6),
        "eef_ori_error_rad": round(float(_quat_angle_error(after_quat, target_quat)), 6),
        "target_orientation_delta_rad": round(float(target_orientation_delta_rad), 6),
        "actual_orientation_delta_rad": round(float(actual_orientation_delta_rad), 6),
        "steps": int(steps),
        "max_steps": int(max_steps),
        "pos_thresh": float(pos_thresh),
        "ori_thresh": float(ori_thresh),
        "video_frames_start": start_frame,
        "video_frames_end": end_frame,
        "trace": trace,
    }


def _tcp_move_and_measure(
    api: X2ControlApi,
    target_tcp_pos: np.ndarray,
    target_tcp_quat: np.ndarray,
    *,
    label: str,
    arm: int,
) -> dict[str, Any]:
    target_tcp_pos = np.asarray(target_tcp_pos, dtype=np.float64).reshape(3)
    target_tcp_quat = _quat_normalize(np.asarray(target_tcp_quat, dtype=np.float64).reshape(4))
    offset_at_plan = api.get_tcp_offset_eef(arm=arm)
    target_eef_pose = api.tcp_pose_to_eef_pose(
        (target_tcp_pos, target_tcp_quat),
        arm=arm,
        tcp_offset_eef=offset_at_plan,
    )
    result = _move_and_measure(
        api,
        target_eef_pose,
        label=label,
        arm=arm,
        max_steps=450,
        pos_thresh=0.006,
        ori_thresh=0.3,
    )
    reached_tcp = np.asarray(result["after_tcp_position"], dtype=np.float64)
    result.update(
        {
            "target_tcp": {"position": _as_list(target_tcp_pos), "quat_xyzw": _as_list(target_tcp_quat)},
            "planned_tcp_offset_eef": _as_list(offset_at_plan),
            "tcp_pos_error_m": round(float(np.linalg.norm(reached_tcp - target_tcp_pos)), 6),
        }
    )
    return result


def _load_previous_targets(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    attempts = summary.get("steps", {}).get("action_chain", {}).get("attempts", [])
    if not attempts:
        return []
    first = attempts[0]
    targets = []
    for key in ("pregrasp_pose", "grasp_pose"):
        pose = first.get("execution_candidate", {}).get(key)
        if pose is not None:
            targets.append(
                {
                    "label": f"previous_attempt0_{key}",
                    "position": np.asarray(pose[0], dtype=np.float64),
                    "quat_xyzw": np.asarray(pose[1], dtype=np.float64),
                }
            )
    return targets


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> dict[str, Any]:
    if not frames:
        return {"path": None, "frame_count": 0, "written": False}
    arr = np.asarray(frames, dtype=np.uint8)
    with imageio.get_writer(path, fps=fps, format="FFMPEG", codec="libx264") as writer:
        for frame in arr:
            writer.append_data(frame)
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
    parser = argparse.ArgumentParser(description="X2 IK pose tracking smoke")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--arm", type=int, default=ARM)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/x2_ik_pose_tracking_smoke"))
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--orientation-deg", type=float, default=10.0)
    parser.add_argument("--include-previous-targets", action="store_true")
    parser.add_argument(
        "--previous-summary",
        type=Path,
        default=Path("outputs/x2_preparing_lunch_box_sam2_graspnet/summary.json"),
    )
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "arm": int(args.arm),
        "tests": [],
        "roundtrip_checks": [],
        "previous_summary": str(args.previous_summary),
        "orientation_deg": float(args.orientation_deg),
        "include_previous_targets": bool(args.include_previous_targets),
    }

    try:
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=[],
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
        )
        api = X2ControlApi(env)
        env.reset()
        env.enable_video_capture(True)
        api.open_gripper(arm=args.arm)
        api.settle_robot(steps=24)

        initial = api.get_current_eef_pose(arm=args.arm)
        initial_pos = np.asarray(initial[0], dtype=np.float64)
        initial_quat = _quat_normalize(np.asarray(initial[1], dtype=np.float64))
        initial_tcp = _tcp_from_eef(api, initial, args.arm)
        tcp_offset = api.get_tcp_offset_eef(arm=args.arm)

        summary["initial_eef"] = {"position": _as_list(initial_pos), "quat_xyzw": _as_list(initial_quat)}
        summary["initial_tcp_position"] = _as_list(initial_tcp)
        summary["initial_tcp_offset_eef"] = _as_list(tcp_offset)
        summary["gripper_state_open"] = api.get_gripper_state(arm=args.arm)

        for delta in (
            np.array([0.0, 0.0, 0.025], dtype=np.float64),
            np.array([0.0, 0.0, -0.02], dtype=np.float64),
            np.array([0.025, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.02, 0.0], dtype=np.float64),
        ):
            current = api.get_current_eef_pose(arm=args.arm)
            target_pos = np.asarray(current[0], dtype=np.float64) + delta
            target_quat = _quat_normalize(np.asarray(current[1], dtype=np.float64))
            summary["tests"].append(
                _move_and_measure(
                    api,
                    (target_pos, target_quat),
                    label=f"eef_delta_{'_'.join(str(round(float(v), 3)) for v in delta)}",
                    arm=args.arm,
                    max_steps=450,
                    pos_thresh=0.006,
                    ori_thresh=0.3,
                )
            )

        current = api.get_current_eef_pose(arm=args.arm)
        current_tcp = _tcp_from_eef(api, current, args.arm)
        current_quat = _quat_normalize(np.asarray(current[1], dtype=np.float64))
        for delta in (
            np.array([0.0, 0.0, 0.02], dtype=np.float64),
            np.array([0.018, 0.0, 0.0], dtype=np.float64),
        ):
            summary["tests"].append(
                _tcp_move_and_measure(
                    api,
                    current_tcp + delta,
                    current_quat,
                    label=f"tcp_delta_{'_'.join(str(round(float(v), 3)) for v in delta)}",
                    arm=args.arm,
                )
            )

        current = api.get_current_eef_pose(arm=args.arm)
        current_pos = np.asarray(current[0], dtype=np.float64)
        current_quat = _quat_normalize(np.asarray(current[1], dtype=np.float64))
        orientation_delta_rad = math.radians(float(args.orientation_deg))
        for axis_name, axis in (
            ("local_x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            ("local_y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            ("local_z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        ):
            target_quat = _quat_multiply_xyzw(current_quat, _axis_angle_quat_xyzw(axis, orientation_delta_rad))
            summary["tests"].append(
                _move_and_measure(
                    api,
                    (current_pos, target_quat),
                    label=f"orientation_plus_{float(args.orientation_deg):g}deg_{axis_name}",
                    arm=args.arm,
                    max_steps=450,
                    pos_thresh=0.008,
                    ori_thresh=0.04,
                )
            )
            current = api.get_current_eef_pose(arm=args.arm)
            current_pos = np.asarray(current[0], dtype=np.float64)
            current_quat = _quat_normalize(np.asarray(current[1], dtype=np.float64))

        if args.include_previous_targets:
            for target in _load_previous_targets(args.previous_summary):
                summary["tests"].append(
                    _move_and_measure(
                        api,
                        (target["position"], target["quat_xyzw"]),
                        label=target["label"],
                        arm=args.arm,
                        max_steps=700,
                        pos_thresh=0.008,
                        ori_thresh=0.5,
                    )
                )

        ok_tests = []
        failed_tests = []
        for test in summary["tests"]:
            pos_ok = float(test.get("eef_pos_error_m", 1.0)) < 0.02
            ori_ok = float(test.get("eef_ori_error_rad", 1.0)) < max(0.12, float(test.get("ori_thresh", 0.0)) + 0.02)
            passed = bool(test.get("ok")) and pos_ok and ori_ok
            (ok_tests if passed else failed_tests).append(test)
        summary["ok"] = len(ok_tests) >= max(1, len(summary["tests"]) - 1)
        summary["diagnosis"] = {
            "test_count": len(summary["tests"]),
            "ok_position_and_orientation_count": len(ok_tests),
            "failed_labels": [str(t.get("label")) for t in failed_tests],
            "max_eef_pos_error_m": max(float(t.get("eef_pos_error_m", 0.0)) for t in summary["tests"]),
            "max_eef_ori_error_rad": max(float(t.get("eef_ori_error_rad", 0.0)) for t in summary["tests"]),
        }
        summary["video"] = _write_videos(args.output_dir, env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})
    except Exception:
        summary["exception"] = traceback.format_exc()
        print(summary["exception"])

    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary.get("diagnosis", {}), indent=2))
    print(f"Wrote {output_path}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
