"""Exercise X2 joint-space IK on multiple reachable TCP poses with orientation changes."""

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

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb"],
    "sensor_kwargs": {"image_height": 384, "image_width": 384},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


POSE_SPECS = [
    {
        "name": "anchor_hold_current_quat",
        "delta": [0.0, 0.0, 0.0],
        "use_current_quat": True,
    },
    {
        "name": "small_up_base_roll5",
        "delta": [0.0, 0.0, 0.014],
        "rpy_deg": [5.0, 0.0, 0.0],
    },
    {
        "name": "small_back_right_mid",
        "delta": [-0.012, 0.01, 0.008],
        "quat_xyzw": [0.623938, 0.172114, 0.755461, 0.101771],
    },
    {
        "name": "small_right_low_roll",
        "delta": [0.008, 0.012, -0.004],
        "quat_xyzw": [0.621289, 0.252274, 0.713342, 0.20372],
    },
    {
        "name": "small_transport_up_yaw",
        "delta": [0.016, 0.006, 0.016],
        "quat_xyzw": [0.665412, 0.082876, 0.741762, 0.012152],
    },
    {
        "name": "small_forward_base_yaw_minus6",
        "delta": [0.014, -0.006, 0.006],
        "rpy_deg": [0.0, 0.0, -6.0],
    },
]


POLICY_CODE = f'''
import numpy as np
import time

ARM = {ARM}
BASE_TCP_POS = None
POSE_SPECS = {POSE_SPECS!r}
TCP_REACH_THRESHOLD_M = 0.025
EEF_ORI_THRESHOLD_RAD = 0.24
START_TIME = time.time()
trace = []
pose_results = []

def quat_xyzw_to_matrix(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

def matrix_to_quat_xyzw(matrix):
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)

def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx

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
        "gripper": get_gripper_state(arm=ARM),
    }}
    item.update(kwargs)
    trace.append(item)
    print("[x2-joint-ik-multi]", label, kwargs, flush=True)

def target_quat_from_base(base_quat, rpy_deg):
    base_R = quat_xyzw_to_matrix(base_quat)
    delta_R = rpy_to_matrix(*np.deg2rad(np.asarray(rpy_deg, dtype=np.float64)))
    return matrix_to_quat_xyzw(base_R @ delta_R)

def spec_target_quat(spec, base_quat):
    if spec.get("use_current_quat"):
        quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
        return quat / max(float(np.linalg.norm(quat)), 1e-12)
    if "quat_xyzw" in spec:
        quat = np.asarray(spec["quat_xyzw"], dtype=np.float64).reshape(4)
        return quat / max(float(np.linalg.norm(quat)), 1e-12)
    return target_quat_from_base(base_quat, spec["rpy_deg"])

def spec_rpy_delta(spec):
    if "rpy_deg" not in spec:
        return None
    return np.asarray(spec["rpy_deg"], dtype=np.float64).round(3).tolist()

def run_pose(spec, target_quat):
    if BASE_TCP_POS is None:
        raise RuntimeError("BASE_TCP_POS must be initialized from current_tcp() before run_pose")
    name = spec["name"]
    target_tcp = BASE_TCP_POS + np.asarray(spec["delta"], dtype=np.float64).reshape(3)
    target_eef = tcp_pose_to_eef_pose((target_tcp, target_quat), arm=ARM)
    q_before = np.asarray(get_current_joint_positions(), dtype=np.float64)
    record(
        name + "_start",
        target_tcp_position=target_tcp.round(6).tolist(),
        target_tcp_quat_xyzw=target_quat.round(6).tolist(),
        target_rpy_delta_deg=spec_rpy_delta(spec),
        converted_target_eef=pose_dict(target_eef),
    )
    t0 = time.time()
    ok = move_tcp_joint_ik(
        (target_tcp, target_quat),
        arm=ARM,
        pos_thresh=0.018,
        ori_thresh=0.24,
        max_joint_step=0.022,
        max_steps=220,
        settle_steps=20,
    )
    settle_robot(steps=16)
    reached_tcp, reached_eef, _offset = current_tcp()
    q_after = np.asarray(get_current_joint_positions(), dtype=np.float64)
    tcp_err = float(np.linalg.norm(reached_tcp - target_tcp))
    eef_pos_err = float(np.linalg.norm(np.asarray(reached_eef[0], dtype=np.float64) - np.asarray(target_eef[0], dtype=np.float64)))
    eef_ori_err = quat_error_rad(reached_eef[1], target_eef[1])
    result = {{
        "name": name,
        "primitive_ok": bool(ok),
        "elapsed_s": round(time.time() - t0, 3),
        "target_tcp_position": target_tcp.round(6).tolist(),
        "target_tcp_quat_xyzw": target_quat.round(6).tolist(),
        "target_rpy_delta_deg": spec_rpy_delta(spec),
        "reached_tcp_position": reached_tcp.round(6).tolist(),
        "tcp_error_m": round(tcp_err, 6),
        "eef_pos_error_m": round(eef_pos_err, 6),
        "eef_ori_error_rad": round(eef_ori_err, 6),
        "max_abs_joint_delta": round(float(np.max(np.abs(q_after - q_before))), 6),
        "success": bool(tcp_err <= TCP_REACH_THRESHOLD_M and eef_ori_err <= EEF_ORI_THRESHOLD_RAD),
    }}
    pose_results.append(result)
    record(name + "_end", **result)
    return result

record("initial")
open_gripper(arm=ARM)
settle_robot(steps=12)
record("after_open")

base_eef = get_current_eef_pose(arm=ARM)
BASE_TCP_POS = current_tcp()[0].copy()
base_tcp_quat = np.asarray(base_eef[1], dtype=np.float64).reshape(4)
record(
    "multi_pose_contract",
    target_source="current_reset_tcp_anchor_plus_local_offsets",
    target_frame="world",
    target_link="tcp/finger_center",
    ik_target_link="r_base_gripper",
    ik_method="pyroki_one_shot_14d_context_single_arm_7d_execution",
    base_tcp_position=BASE_TCP_POS.round(6).tolist(),
    base_tcp_quat_xyzw=base_tcp_quat.round(6).tolist(),
    pose_count=len(POSE_SPECS),
)

for idx, spec in enumerate(POSE_SPECS):
    target_quat = spec_target_quat(spec, base_tcp_quat)
    run_pose(spec, target_quat)
    if idx == 1:
        close_gripper(arm=ARM)
        settle_robot(steps=10)
        record("after_close_gripper_mid_sequence")
    elif idx == 3:
        open_gripper(arm=ARM)
        settle_robot(steps=10)
        record("after_open_gripper_mid_sequence")

record("final")
RESULT = {{
    "trace": trace,
    "pose_results": pose_results,
    "target_source": "current_reset_tcp_anchor_plus_local_offsets",
    "target_frame": "world",
    "target_link": "tcp/finger_center",
    "ik_target_link": "r_base_gripper",
    "ik_method": "pyroki_one_shot_14d_context_single_arm_7d_execution",
    "controller_config": "x2_robotiq85_joint_primitives.yaml",
    "base_tcp_position_world": BASE_TCP_POS.round(6).tolist(),
    "base_tcp_quat_xyzw_world": base_tcp_quat.round(6).tolist(),
    "thresholds": {{
        "tcp_reach_threshold_m": TCP_REACH_THRESHOLD_M,
        "eef_ori_threshold_rad": EEF_ORI_THRESHOLD_RAD,
    }},
    "success_count": int(sum(1 for r in pose_results if r["success"])),
    "pose_count": int(len(pose_results)),
    "all_success": bool(len(pose_results) == len(POSE_SPECS) and all(r["success"] for r in pose_results)),
    "max_tcp_error_m": round(float(max([r["tcp_error_m"] for r in pose_results] or [999.0])), 6),
    "max_eef_ori_error_rad": round(float(max([r["eef_ori_error_rad"] for r in pose_results] or [999.0])), 6),
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
    parser = argparse.ArgumentParser(description="Run X2 joint-space IK multi-pose smoke")
    parser.add_argument("--output-dir", default="outputs/x2_joint_ik_multi_pose_trial")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "target_meaning": "multiple TCP/finger-center poses in world frame, defined as local offsets from the reset-time TCP anchor",
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
        print("[x2-joint-ik-multi-trial] executing injected policy", flush=True)
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
        video_ok = bool(summary["video"].get("combined", {}).get("written", False))
        summary["ok"] = bool(info.get("sandbox_rc") == 0 and isinstance(result, dict) and result.get("all_success") and video_ok)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)

    summary["elapsed_s"] = round(time.time() - start, 3)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-joint-ik-multi-trial] wrote {summary_path}", flush=True)
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"]), "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
