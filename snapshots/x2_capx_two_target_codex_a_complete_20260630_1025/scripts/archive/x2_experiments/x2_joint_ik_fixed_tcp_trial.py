"""Smoke test X2 one-shot IK plus joint-space interpolation on a fixed TCP target."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.envs.tasks import CodeExecEnvConfig, get_exec_env


ARM = 1
TARGET_TCP_POS = [0.33889, -0.188279, 0.943519]
PREGRASP_LIFT_M = 0.055
TRANSPORT_DELTA = [0.03, 0.02, 0.04]

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb"],
    "sensor_kwargs": {"image_height": 384, "image_width": 384},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


POLICY_CODE = f'''
import numpy as np
import time

ARM = {ARM}
TARGET_TCP_POS = np.array({TARGET_TCP_POS}, dtype=np.float64)
PREGRASP_LIFT_M = {PREGRASP_LIFT_M}
TRANSPORT_DELTA = np.array({TRANSPORT_DELTA}, dtype=np.float64)
TCP_REACH_THRESHOLD_M = 0.025
START_TIME = time.time()
trace = []

def quat_xyzw_to_matrix(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

def quat_error_rad(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(4)
    b = np.asarray(b, dtype=np.float64).reshape(4)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)))

def pose_dict(pose):
    pos, quat = pose
    return {{
        "position": np.asarray(pos, dtype=np.float64).round(6).tolist(),
        "quat_xyzw": np.asarray(quat, dtype=np.float64).round(6).tolist(),
    }}

def current_tcp():
    eef = get_current_eef_pose(arm=ARM)
    offset = np.asarray(get_tcp_offset_eef(arm=ARM), dtype=np.float64).reshape(3)
    tcp = np.asarray(eef[0], dtype=np.float64).reshape(3) + quat_xyzw_to_matrix(eef[1]) @ offset
    return tcp, eef, offset

def record(label, **kwargs):
    tcp, eef, offset = current_tcp()
    item = {{
        "label": label,
        "wall_time_s": round(time.time() - START_TIME, 3),
        "tcp_position": tcp.round(6).tolist(),
        "eef": pose_dict(eef),
        "tcp_offset_eef": offset.round(6).tolist(),
        "joint_positions": np.asarray(get_current_joint_positions(), dtype=np.float64).round(6).tolist(),
        "gripper": get_gripper_state(arm=ARM),
    }}
    item.update(kwargs)
    trace.append(item)
    print("[x2-joint-ik]", label, kwargs, flush=True)

def run_move_tcp(label, tcp_position, tcp_quat, *, max_steps=180, max_joint_step=0.03, pos_thresh=0.012, ori_thresh=0.2):
    tcp_position = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    tcp_quat = np.asarray(tcp_quat, dtype=np.float64).reshape(4)
    target_eef = tcp_pose_to_eef_pose((tcp_position, tcp_quat), arm=ARM)
    q_before = np.asarray(get_current_joint_positions(), dtype=np.float64)
    record(
        label + "_start",
        target_tcp_position=tcp_position.round(6).tolist(),
        target_tcp_quat_xyzw=tcp_quat.round(6).tolist(),
        converted_target_eef=pose_dict(target_eef),
    )
    t0 = time.time()
    ok = move_tcp_joint_ik(
        (tcp_position, tcp_quat),
        arm=ARM,
        pos_thresh=pos_thresh,
        ori_thresh=ori_thresh,
        max_joint_step=max_joint_step,
        max_steps=max_steps,
        settle_steps=14,
    )
    settle_robot(steps=8)
    reached_tcp, reached_eef, _ = current_tcp()
    q_after = np.asarray(get_current_joint_positions(), dtype=np.float64)
    tcp_err = float(np.linalg.norm(reached_tcp - tcp_position))
    eef_pos_err = float(np.linalg.norm(np.asarray(reached_eef[0], dtype=np.float64) - np.asarray(target_eef[0], dtype=np.float64)))
    eef_ori_err = quat_error_rad(reached_eef[1], target_eef[1])
    record(
        label + "_end",
        ok=bool(ok),
        elapsed_s=round(time.time() - t0, 3),
        target_tcp_position=tcp_position.round(6).tolist(),
        reached_tcp_position=reached_tcp.round(6).tolist(),
        tcp_target_error_m=round(tcp_err, 6),
        eef_target_pos_error_m=round(eef_pos_err, 6),
        eef_target_ori_error_rad=round(eef_ori_err, 6),
        max_abs_joint_delta=round(float(np.max(np.abs(q_after - q_before))), 6),
        max_steps=int(max_steps),
        max_joint_step=float(max_joint_step),
    )
    return bool(ok), tcp_err, eef_pos_err, eef_ori_err

record("initial")
open_gripper(arm=ARM)
settle_robot(steps=12)
record("after_open")

initial_eef = get_current_eef_pose(arm=ARM)
tcp_quat = np.asarray(initial_eef[1], dtype=np.float64).reshape(4)
record(
    "joint_ik_contract",
    target_source="hardcoded_world_tcp_pose",
    target_frame="world",
    target_link="tcp/finger_center",
    ik_target_link="r_base_gripper",
    ik_method="pyroki_one_shot_14d_context_single_arm_7d_execution",
    target_tcp_position=TARGET_TCP_POS.round(6).tolist(),
    target_tcp_quat_xyzw=tcp_quat.round(6).tolist(),
)

pre_tcp = TARGET_TCP_POS + np.array([0.0, 0.0, PREGRASP_LIFT_M], dtype=np.float64)
pre_ok, pre_tcp_err, pre_eef_pos_err, pre_eef_ori_err = run_move_tcp(
    "pre_approach",
    pre_tcp,
    tcp_quat,
    max_steps=180,
    max_joint_step=0.03,
    pos_thresh=0.014,
    ori_thresh=0.22,
)

target_ok, target_tcp_err, target_eef_pos_err, target_eef_ori_err = run_move_tcp(
    "fixed_target_tcp",
    TARGET_TCP_POS,
    tcp_quat,
    max_steps=180,
    max_joint_step=0.028,
    pos_thresh=0.012,
    ori_thresh=0.2,
)

close_gripper(arm=ARM)
settle_robot(steps=14)
record("after_close_gripper")

transport_tcp = TARGET_TCP_POS + TRANSPORT_DELTA
transport_ok, transport_tcp_err, transport_eef_pos_err, transport_eef_ori_err = run_move_tcp(
    "transport_after_close",
    transport_tcp,
    tcp_quat,
    max_steps=180,
    max_joint_step=0.03,
    pos_thresh=0.018,
    ori_thresh=0.24,
)
record("final")

RESULT = {{
    "trace": trace,
    "target_source": "hardcoded_world_tcp_pose",
    "target_tcp_position_world": TARGET_TCP_POS.round(6).tolist(),
    "target_tcp_quat_xyzw_world": tcp_quat.round(6).tolist(),
    "ik_method": "pyroki_one_shot_14d_context_single_arm_7d_execution",
    "controller_config": "x2_robotiq85_joint_primitives.yaml",
    "pre_approach_ok": bool(pre_ok),
    "pre_approach_tcp_error_m": round(float(pre_tcp_err), 6),
    "fixed_target_move_ok": bool(target_ok),
    "fixed_target_tcp_error_m": round(float(target_tcp_err), 6),
    "fixed_target_eef_pos_error_m": round(float(target_eef_pos_err), 6),
    "fixed_target_eef_ori_error_rad": round(float(target_eef_ori_err), 6),
    "reached_fixed_tcp_target": bool(target_tcp_err <= TCP_REACH_THRESHOLD_M),
    "transport_ok": bool(transport_ok),
    "transport_tcp_error_m": round(float(transport_tcp_err), 6),
    "thresholds": {{"tcp_reach_threshold_m": TCP_REACH_THRESHOLD_M}},
}}
'''


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


def _write_videos(output_dir: Path, frames: Any, fps: int) -> dict[str, Any]:
    frames_by_view = frames if isinstance(frames, dict) else {"rgb": frames}
    result: dict[str, Any] = {"views": {}, "fps": fps}
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
    parser = argparse.ArgumentParser(description="Run X2 joint-space IK fixed TCP smoke")
    parser.add_argument("--output-dir", default="outputs/x2_joint_ik_fixed_tcp_trial")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "target_tcp_position_world": TARGET_TCP_POS,
        "target_meaning": "hardcoded TCP/finger-center pose in world frame",
        "policy_code": POLICY_CODE,
        "steps": {},
        "video": {},
        "errors": [],
    }
    start = time.time()
    try:
        low_level = X2BehaviourLowLevel(
            controller_cfg="x2_robotiq85_joint_primitives.yaml",
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=256,
            robot_obs_modalities=["rgb"],
        )
        exec_env_cls = get_exec_env("x2_behavior_code_env")
        exec_env = exec_env_cls(CodeExecEnvConfig(low_level=low_level, apis=["X2ControlApi"]))
        _obs, reset_info = exec_env.reset()
        summary["steps"]["reset"] = {"info_keys": sorted(reset_info.keys())}
        exec_env.enable_video_capture(True, clear=True)
        print("[x2-joint-ik-trial] executing injected policy", flush=True)
        _obs, reward, terminated, truncated, info = exec_env.step(POLICY_CODE)
        result = exec_env._exec_globals.get("RESULT")
        summary["steps"]["execution"] = {
            "result": result,
            "last_pyroki_ik_debug": getattr(low_level, "_last_pyroki_ik_debug", None),
            "last_joint_ik_move_debug": getattr(low_level, "_last_joint_ik_move_debug", None),
            "sandbox_rc": info.get("sandbox_rc"),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_completed": info.get("task_completed"),
            "stdout_tail": info.get("stdout", "")[-4000:],
            "stderr_tail": info.get("stderr", "")[-4000:],
        }
        summary["video"] = _write_videos(output_dir, exec_env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(low_level, "_last_video_sources", {})
        result_ok = isinstance(result, dict)
        reached = bool(result_ok and result.get("reached_fixed_tcp_target"))
        video_ok = bool(summary["video"].get("combined", {}).get("written", False))
        summary["ok"] = bool(info.get("sandbox_rc") == 0 and reached and video_ok)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)

    summary["elapsed_s"] = round(time.time() - start, 3)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-joint-ik-trial] wrote {summary_path}", flush=True)
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"]), "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
