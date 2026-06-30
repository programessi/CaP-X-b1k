"""Run one CAP-X code-exec trial for the minimal X2 tabletop TCP task.

The environment is instantiated from env_configs/x2/x2_tabletop_tcp_reach.yaml.
The policy below is intentionally hand-written but injected through
exec_env.step(code), matching the path used for generated code.
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

import mediapy as media
import numpy as np

import capx.envs.simulators  # noqa: F401 - registers low-level envs
import capx.envs.tasks  # noqa: F401 - registers code-execution envs/configs
import capx.integrations  # noqa: F401 - registers APIs
from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader


POLICY_CODE = r'''
import numpy as np
import time

ARM = 1
TARGET_NAME = "x2_reach_target_marker"
GRASP_TCP_THRESHOLD_M = 0.025

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

def pose_dict(pose):
    pos, quat = pose
    return {
        "position": np.asarray(pos, dtype=np.float64).round(6).tolist(),
        "quat_xyzw": np.asarray(quat, dtype=np.float64).round(6).tolist(),
    }

def current_tcp():
    eef = get_current_eef_pose(arm=ARM)
    offset = np.asarray(get_tcp_offset_eef(arm=ARM), dtype=np.float64).reshape(3)
    tcp = np.asarray(eef[0], dtype=np.float64).reshape(3) + quat_xyzw_to_matrix(eef[1]) @ offset
    return tcp, eef, offset

def record(label, **kwargs):
    tcp, eef, offset = current_tcp()
    item = {
        "label": label,
        "wall_time_s": round(time.time() - START_TIME, 3),
        "tcp_position": tcp.round(6).tolist(),
        "eef": pose_dict(eef),
        "tcp_offset_eef": offset.round(6).tolist(),
        "gripper": get_gripper_state(arm=ARM),
    }
    item.update(kwargs)
    trace.append(item)
    print("[x2-policy]", label, kwargs, flush=True)

def run_move_tcp(label, tcp_position, tcp_quat, *, max_steps, pos_thresh=0.006, ori_thresh=0.12):
    tcp_position = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    tcp_quat = np.asarray(tcp_quat, dtype=np.float64).reshape(4)
    before_tcp, before_eef, _ = current_tcp()
    record(
        label + "_start",
        target_tcp_position=tcp_position.round(6).tolist(),
        target_tcp_quat_xyzw=tcp_quat.round(6).tolist(),
    )
    t0 = time.time()
    ok = move_tcp(
        (tcp_position, tcp_quat),
        arm=ARM,
        pos_thresh=pos_thresh,
        ori_thresh=ori_thresh,
        stop_if_stuck=True,
        stuck_patience_steps=35,
        max_steps=max_steps,
    )
    settle_robot(steps=10)
    reached_tcp, reached_eef, _ = current_tcp()
    err = float(np.linalg.norm(reached_tcp - tcp_position))
    record(
        label + "_end",
        ok=bool(ok),
        elapsed_s=round(time.time() - t0, 3),
        before_tcp_position=before_tcp.round(6).tolist(),
        target_tcp_position=tcp_position.round(6).tolist(),
        reached_tcp_position=reached_tcp.round(6).tolist(),
        tcp_target_error_m=round(err, 6),
        reached_eef=pose_dict(reached_eef),
        max_steps=int(max_steps),
    )
    return bool(ok), err

START_TIME = time.time()

record("initial")
open_gripper(arm=ARM)
settle_robot(steps=12)
record("after_open")

# Visual/object primitive: estimate target marker pose in world frame.
target_pos, target_quat = get_object_pose(TARGET_NAME)
target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
initial_eef = get_current_eef_pose(arm=ARM)
tcp_quat = np.asarray(initial_eef[1], dtype=np.float64).reshape(4)
record(
    "visual_target_pose",
    visual_pose_frame="world",
    visual_pose_target="tcp/finger-center target point",
    target_marker_position_world=target_pos.round(6).tolist(),
    target_marker_quat_xyzw=target_quat.round(6).tolist(),
    tcp_quat_source="initial_eef_quat_xyzw",
)

# Approach from above, then settle before moving to the actual grasp/TCP target.
pre_tcp = target_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64)
pre_ok, pre_err = run_move_tcp("pre_approach", pre_tcp, tcp_quat, max_steps=700, pos_thresh=0.008, ori_thresh=0.18)
record("stable_before_grasp")

grasp_ok, grasp_err = run_move_tcp("visual_grasp_tcp", target_pos, tcp_quat, max_steps=700, pos_thresh=0.006, ori_thresh=0.15)
record(
    "visual_grasp_pose_verdict",
    target_tcp_position=target_pos.round(6).tolist(),
    reached_visual_grasp_tcp=bool(grasp_err <= GRASP_TCP_THRESHOLD_M),
    visual_grasp_tcp_error_m=round(float(grasp_err), 6),
)

close_gripper(arm=ARM)
settle_robot(steps=16)
record("after_close_gripper")

# Short transport after the close. The object interaction is intentionally not
# the success criterion; this checks that the flow keeps moving under code-exec.
transport_tcp = target_pos + np.array([0.045, 0.025, 0.06], dtype=np.float64)
transport_ok, transport_err = run_move_tcp(
    "transport_after_close",
    transport_tcp,
    tcp_quat,
    max_steps=800,
    pos_thresh=0.012,
    ori_thresh=0.25,
)
record("final")

RESULT = {
    "trace": trace,
    "target_name": TARGET_NAME,
    "target_tcp_position_world": target_pos.round(6).tolist(),
    "target_tcp_quat_xyzw_world": tcp_quat.round(6).tolist(),
    "pre_approach_ok": bool(pre_ok),
    "pre_approach_error_m": round(float(pre_err), 6),
    "visual_grasp_move_ok": bool(grasp_ok),
    "visual_grasp_tcp_error_m": round(float(grasp_err), 6),
    "reached_visual_grasp_tcp": bool(grasp_err <= GRASP_TCP_THRESHOLD_M),
    "transport_ok": bool(transport_ok),
    "transport_tcp_error_m": round(float(transport_err), 6),
    "thresholds": {
        "visual_grasp_tcp_threshold_m": GRASP_TCP_THRESHOLD_M,
    },
}
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
    parser = argparse.ArgumentParser(description="Run one hand-coded CAP-X X2 tabletop TCP reach trial")
    parser.add_argument("--config-path", default="env_configs/x2/x2_tabletop_tcp_reach.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_tabletop_tcp_reach_code_exec_trial")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "config_path": args.config_path,
        "policy_code": POLICY_CODE,
        "steps": {},
        "video": {},
        "errors": [],
    }
    start = time.time()
    try:
        cfg = DictLoader.load(args.config_path)
        exec_env = instantiate(cfg["env"])
        _obs, reset_info = exec_env.reset()
        summary["steps"]["reset"] = {"info_keys": sorted(reset_info.keys())}
        exec_env.enable_video_capture(True, clear=True)
        print("[x2-trial] executing injected policy", flush=True)
        _obs, reward, terminated, truncated, info = exec_env.step(POLICY_CODE)
        result = exec_env._exec_globals.get("RESULT")
        summary["steps"]["execution"] = {
            "result": result,
            "sandbox_rc": info.get("sandbox_rc"),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_completed": info.get("task_completed"),
            "tcp_target_error_m": info.get("tcp_target_error_m"),
            "stdout_tail": info.get("stdout", "")[-4000:],
            "stderr_tail": info.get("stderr", "")[-4000:],
        }
        frames = exec_env.get_video_frames()
        summary["video"] = _write_videos(output_dir, frames, fps=args.video_fps)
        summary["video_sources"] = getattr(exec_env.low_level_env, "_last_video_sources", {})
        result_ok = isinstance(result, dict)
        reached = bool(result_ok and result.get("reached_visual_grasp_tcp"))
        video_ok = bool(summary["video"].get("combined", {}).get("written", False))
        summary["ok"] = bool(info.get("sandbox_rc") == 0 and reached and video_ok)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)

    summary["elapsed_s"] = round(time.time() - start, 3)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-trial] wrote {summary_path}", flush=True)
    for view, meta in summary["video"].get("views", {}).items():
        if meta.get("written"):
            print(f"[x2-trial] wrote {view} video {meta['path']} ({meta['frame_count']} frames)", flush=True)
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"[x2-trial] wrote combined video {meta['path']} ({meta['frame_count']} frames)", flush=True)
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"]), "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
