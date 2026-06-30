"""Run one X2 visual-pose-to-IK CAP-X code-exec trial.

The scene setup lives in this script.  The injected policy only consumes
CAP-X-style APIs: estimate an object pose from RGB-D, derive a conservative
world-frame TCP target above the object, and move there through move_tcp().
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

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.envs.tasks import CodeExecEnvConfig, get_exec_env


ARM = 1
OBJECT_NAME = "x2_visual_ik_cube"
TABLE_NAME = "x2_visual_ik_support"
OBJECT_SIZE = 0.04
OBJECT_CENTER = np.array([0.37389, -0.163279, 0.945], dtype=np.float64)
TABLE_SCALE = np.array([0.10, 0.10, 0.012], dtype=np.float64)
TABLE_CENTER = np.array([0.37389, -0.163279, 0.925], dtype=np.float64)

# From the RGB-D contract smoke for this exact camera/object layout.  This is
# a depth gate for the visual geometry estimate, not a target pose for motion.
EXPECTED_OBJECT_DEPTH_M = 1.113063
DEPTH_WINDOW_M = 0.086603
TCP_CLEARANCE_M = 0.055
PRE_APPROACH_LIFT_M = 0.035
TCP_REACH_THRESHOLD_M = 0.025
OBJECT_POSE_THRESHOLD_M = 0.025
SAFE_TCP_QUAT_XYZW = [0.648902, 0.169021, 0.73383, 0.108867]

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb", "depth_linear"],
    "sensor_kwargs": {"image_height": 384, "image_width": 384},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


POLICY_CODE = f'''
import numpy as np
import time

ARM = {ARM}
OBJECT_NAME = "{OBJECT_NAME}"
EXPECTED_OBJECT_DEPTH_M = {EXPECTED_OBJECT_DEPTH_M}
DEPTH_WINDOW_M = {DEPTH_WINDOW_M}
TCP_CLEARANCE_M = {TCP_CLEARANCE_M}
PRE_APPROACH_LIFT_M = {PRE_APPROACH_LIFT_M}
TCP_REACH_THRESHOLD_M = {TCP_REACH_THRESHOLD_M}
SAFE_TCP_QUAT_XYZW = np.array({SAFE_TCP_QUAT_XYZW}, dtype=np.float64)

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
    print("[x2-visual-ik]", label, kwargs, flush=True)

def run_move_tcp(label, tcp_position, tcp_quat, *, max_steps, pos_thresh, ori_thresh):
    tcp_position = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    tcp_quat = np.asarray(tcp_quat, dtype=np.float64).reshape(4)
    before_tcp, _before_eef, _ = current_tcp()
    target_eef = tcp_pose_to_eef_pose((tcp_position, tcp_quat), arm=ARM)
    record(
        label + "_start",
        target_tcp_position=tcp_position.round(6).tolist(),
        target_tcp_quat_xyzw=tcp_quat.round(6).tolist(),
        converted_target_eef=pose_dict(target_eef),
        before_tcp_position=before_tcp.round(6).tolist(),
    )
    t0 = time.time()
    ok = move_tcp(
        (tcp_position, tcp_quat),
        arm=ARM,
        pos_thresh=pos_thresh,
        ori_thresh=ori_thresh,
        stop_if_stuck=True,
        stuck_patience_steps=40,
        max_steps=max_steps,
    )
    settle_robot(steps=10)
    reached_tcp, reached_eef, _ = current_tcp()
    tcp_err = float(np.linalg.norm(reached_tcp - tcp_position))
    eef_err = float(np.linalg.norm(np.asarray(reached_eef[0], dtype=np.float64) - np.asarray(target_eef[0], dtype=np.float64)))
    record(
        label + "_end",
        ok=bool(ok),
        elapsed_s=round(time.time() - t0, 3),
        target_tcp_position=tcp_position.round(6).tolist(),
        reached_tcp_position=reached_tcp.round(6).tolist(),
        tcp_target_error_m=round(tcp_err, 6),
        eef_target_error_m=round(eef_err, 6),
        reached_eef=pose_dict(reached_eef),
        max_steps=int(max_steps),
    )
    return bool(ok), tcp_err, eef_err

START_TIME = time.time()
record("initial")
open_gripper(arm=ARM)
settle_robot(steps=12)
record("after_open")

object_pos, object_quat, object_extent = get_object_pose(
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

tcp_quat = SAFE_TCP_QUAT_XYZW.copy()
target_tcp = object_pos + np.array([0.0, 0.0, TCP_CLEARANCE_M], dtype=np.float64)
pre_tcp = target_tcp + np.array([0.0, 0.0, PRE_APPROACH_LIFT_M], dtype=np.float64)

record(
    "visual_pose_contract",
    visual_output_frame="world",
    visual_output_meaning="object/aabb center, not tcp and not eef",
    visual_object_position_world=object_pos.round(6).tolist(),
    visual_object_quat_xyzw=object_quat.round(6).tolist(),
    visual_object_extent=object_extent.round(6).tolist(),
    expected_depth_m=round(float(EXPECTED_OBJECT_DEPTH_M), 6),
    depth_window_m=round(float(DEPTH_WINDOW_M), 6),
    derived_target_frame="world",
    derived_target_link="tcp/finger_center",
    target_tcp_position_world=target_tcp.round(6).tolist(),
    target_tcp_quat_xyzw_world=tcp_quat.round(6).tolist(),
)

pre_ok, pre_tcp_err, pre_eef_err = run_move_tcp(
    "pre_approach_from_visual_pose",
    pre_tcp,
    tcp_quat,
    max_steps=650,
    pos_thresh=0.010,
    ori_thresh=0.20,
)
record("stable_before_visual_tcp_target")

if pre_ok:
    target_ok, target_tcp_err, target_eef_err = run_move_tcp(
        "visual_tcp_target",
        target_tcp,
        tcp_quat,
        max_steps=700,
        pos_thresh=0.008,
        ori_thresh=0.18,
    )
    record(
        "target_verdict",
        reached_visual_tcp_target=bool(target_tcp_err <= TCP_REACH_THRESHOLD_M),
        visual_tcp_target_error_m=round(float(target_tcp_err), 6),
        visual_eef_target_error_m=round(float(target_eef_err), 6),
    )

    close_gripper(arm=ARM)
    settle_robot(steps=10)
    record("after_close_gripper_hover")
    open_gripper(arm=ARM)
    settle_robot(steps=8)
else:
    target_ok = False
    target_tcp_err = 999.0
    target_eef_err = 999.0
    record("target_skipped_after_failed_pre_approach")
record("final")

RESULT = {{
    "trace": trace,
    "visual_object_position_world": object_pos.round(6).tolist(),
    "visual_object_quat_xyzw": object_quat.round(6).tolist(),
    "visual_object_extent": object_extent.round(6).tolist(),
    "target_tcp_position_world": target_tcp.round(6).tolist(),
    "target_tcp_quat_xyzw_world": tcp_quat.round(6).tolist(),
    "target_pose_semantics": {{
        "visual_pose": "T_world_object_center",
        "action_pose": "T_world_tcp/finger_center derived from visual object center plus clearance",
        "low_level_pose": "move_tcp converts T_world_tcp to T_world_eef before move_hand",
    }},
    "pre_approach_ok": bool(pre_ok),
    "pre_approach_tcp_error_m": round(float(pre_tcp_err), 6),
    "visual_target_move_ok": bool(target_ok),
    "visual_target_tcp_error_m": round(float(target_tcp_err), 6),
    "visual_target_eef_error_m": round(float(target_eef_err), 6),
    "reached_visual_tcp_target": bool(target_tcp_err <= TCP_REACH_THRESHOLD_M),
    "thresholds": {{"tcp_reach_threshold_m": TCP_REACH_THRESHOLD_M}},
}}
'''


def _to_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).round(6).tolist()


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one X2 visual-pose-to-IK code-exec trial")
    parser.add_argument("--output-dir", default="outputs/x2_visual_pose_to_ik_code_exec_trial")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "object_name": OBJECT_NAME,
        "scene_contract": {
            "object_is_visual_only": False,
            "visual_pose_source": "RGB-D get_object_pose with depth gate",
            "visual_pose_semantics": "T_world_object_center",
            "action_pose_semantics": "T_world_tcp/finger_center, derived from object center plus z clearance",
        },
        "policy_code": POLICY_CODE,
        "steps": {},
        "video": {},
        "errors": [],
    }
    start = time.time()
    low_level = None
    try:
        low_level = X2BehaviourLowLevel(
            objects=_scene_objects(),
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=384,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
        )
        exec_env_cls = get_exec_env("x2_behavior_code_env")
        exec_env = exec_env_cls(CodeExecEnvConfig(low_level=low_level, apis=["X2ControlApi"]))
        _obs, reset_info = exec_env.reset()
        summary["steps"]["reset"] = {"info_keys": sorted(reset_info.keys())}
        exec_env.enable_video_capture(True, clear=True)
        print("[x2-visual-ik-trial] executing injected policy", flush=True)
        _obs, reward, terminated, truncated, info = exec_env.step(POLICY_CODE)
        result = exec_env._exec_globals.get("RESULT")

        truth_center = None
        try:
            obj = low_level.env.scene.object_registry("name", OBJECT_NAME)
            truth_center = np.asarray(obj.aabb_center, dtype=np.float64).reshape(3)
        except Exception:
            truth_center = OBJECT_CENTER.copy()

        visual_object_error = None
        true_target_error = None
        if isinstance(result, dict):
            visual_pos = np.asarray(result.get("visual_object_position_world"), dtype=np.float64).reshape(3)
            visual_object_error = float(np.linalg.norm(visual_pos - truth_center))
            true_target_tcp = truth_center + np.array([0.0, 0.0, TCP_CLEARANCE_M], dtype=np.float64)
            trace = result.get("trace") or []
            reached_tcp = None
            for item in trace:
                if item.get("label") == "visual_tcp_target_end":
                    reached_tcp = np.asarray(item.get("reached_tcp_position"), dtype=np.float64).reshape(3)
            if reached_tcp is not None:
                true_target_error = float(np.linalg.norm(reached_tcp - true_target_tcp))
            result["truth_object_center_world"] = truth_center.round(6).tolist()
            result["visual_object_error_to_truth_m"] = None if visual_object_error is None else round(visual_object_error, 6)
            result["true_target_tcp_position_world"] = true_target_tcp.round(6).tolist()
            result["true_target_tcp_error_m"] = None if true_target_error is None else round(true_target_error, 6)

        summary["steps"]["execution"] = {
            "result": result,
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
        reached_visual = bool(isinstance(result, dict) and result.get("reached_visual_tcp_target"))
        visual_pose_ok = bool(visual_object_error is not None and visual_object_error <= OBJECT_POSE_THRESHOLD_M)
        true_target_ok = bool(true_target_error is not None and true_target_error <= TCP_REACH_THRESHOLD_M)
        video_ok = bool(summary["video"].get("combined", {}).get("written", False))
        summary["ok"] = bool(info.get("sandbox_rc") == 0 and reached_visual and visual_pose_ok and true_target_ok and video_ok)
        summary["checks"] = {
            "sandbox_ok": bool(info.get("sandbox_rc") == 0),
            "reached_visual_tcp_target": reached_visual,
            "visual_object_pose_ok": visual_pose_ok,
            "true_target_tcp_ok": true_target_ok,
            "video_ok": video_ok,
            "thresholds": {
                "object_pose_threshold_m": OBJECT_POSE_THRESHOLD_M,
                "tcp_reach_threshold_m": TCP_REACH_THRESHOLD_M,
            },
        }
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)

    summary["elapsed_s"] = round(time.time() - start, 3)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-visual-ik-trial] wrote {summary_path}", flush=True)
    for view, meta in summary["video"].get("views", {}).items():
        if meta.get("written"):
            print(f"[x2-visual-ik-trial] wrote {view} video {meta['path']} ({meta['frame_count']} frames)", flush=True)
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"[x2-visual-ik-trial] wrote combined video {meta['path']} ({meta['frame_count']} frames)", flush=True)
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"]), "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
