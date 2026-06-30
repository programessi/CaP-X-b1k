"""Run X2 visual-pose-to-joint-IK CAP-X code-exec trial.

The injected policy consumes CAP-X APIs only:

1. Estimate an object center from RGB-D in world frame.
2. Derive conservative world-frame TCP targets above that object.
3. Execute those TCP targets through move_tcp_joint_ik().
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.envs.tasks import CodeExecEnvConfig, get_exec_env
from x2_visual_pose_to_ik_code_exec_trial import (
    ARM,
    GLOBAL_CAMERA,
    OBJECT_NAME,
    OBJECT_POSE_THRESHOLD_M,
    OBJECT_SIZE,
    SAFE_TCP_QUAT_XYZW,
    TABLE_NAME,
    TABLE_SCALE,
    TCP_REACH_THRESHOLD_M,
    _write_videos,
)


OBJECT_CENTER = np.array([0.335, -0.155, 0.955], dtype=np.float64)
TABLE_CENTER = np.array([0.305, -0.140, 0.925], dtype=np.float64)
TABLE_SCALE = np.array([0.18, 0.18, 0.012], dtype=np.float64)
OBJECT_HALF_HEIGHT_M = OBJECT_SIZE / 2.0
GRASP_CENTER_CLEARANCE_M = 0.004
PRE_APPROACH_LIFT_M = 0.04
LIFT_DISTANCE_M = 0.045


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


POLICY_CODE = f'''
import numpy as np
import time

ARM = {ARM}
OBJECT_NAME = "{OBJECT_NAME}"
OBJECT_HALF_HEIGHT_M = {OBJECT_HALF_HEIGHT_M}
GRASP_CENTER_CLEARANCE_M = {GRASP_CENTER_CLEARANCE_M}
PRE_APPROACH_LIFT_M = {PRE_APPROACH_LIFT_M}
LIFT_DISTANCE_M = {LIFT_DISTANCE_M}
TCP_REACH_THRESHOLD_M = {TCP_REACH_THRESHOLD_M}
SAFE_TCP_QUAT_XYZW = np.array({SAFE_TCP_QUAT_XYZW}, dtype=np.float64)
START_TIME = time.time()
trace = []
moves = []

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

def nested_get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

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
    print("[x2-visual-joint-ik]", label, kwargs, flush=True)

def run_move_tcp_joint_ik(label, tcp_position, tcp_quat, *, max_steps=220, max_joint_step=0.022, pos_thresh=0.018, ori_thresh=0.24):
    tcp_position = np.asarray(tcp_position, dtype=np.float64).reshape(3)
    tcp_quat = np.asarray(tcp_quat, dtype=np.float64).reshape(4)
    target_eef = tcp_pose_to_eef_pose((tcp_position, tcp_quat), arm=ARM)
    q_before = np.asarray(get_current_joint_positions(), dtype=np.float64)
    record(
        label + "_start",
        target_tcp_position=tcp_position.round(6).tolist(),
        target_tcp_quat_xyzw=tcp_quat.round(6).tolist(),
        target_eef=pose_dict(target_eef),
    )
    t0 = time.time()
    primitive_ok = move_tcp_joint_ik(
        (tcp_position, tcp_quat),
        arm=ARM,
        pos_thresh=pos_thresh,
        ori_thresh=ori_thresh,
        max_joint_step=max_joint_step,
        max_steps=max_steps,
        settle_steps=20,
    )
    debug = get_last_motion_debug()
    settle_robot(steps=16)
    reached_tcp, reached_eef, _ = current_tcp()
    q_after = np.asarray(get_current_joint_positions(), dtype=np.float64)
    tcp_err = float(np.linalg.norm(reached_tcp - tcp_position))
    eef_pos_err = float(np.linalg.norm(np.asarray(reached_eef[0], dtype=np.float64) - np.asarray(target_eef[0], dtype=np.float64)))
    eef_ori_err = quat_error_rad(reached_eef[1], target_eef[1])
    result = {{
        "move_label": label,
        "primitive_ok": bool(primitive_ok),
        "elapsed_s": round(time.time() - t0, 3),
        "target_tcp_position": tcp_position.round(6).tolist(),
        "target_tcp_quat_xyzw": tcp_quat.round(6).tolist(),
        "target_eef": pose_dict(target_eef),
        "reached_tcp_position": reached_tcp.round(6).tolist(),
        "reached_eef": pose_dict(reached_eef),
        "tcp_error_m": round(tcp_err, 6),
        "eef_pos_error_m": round(eef_pos_err, 6),
        "eef_ori_error_rad": round(eef_ori_err, 6),
        "max_abs_joint_delta": round(float(np.max(np.abs(q_after - q_before))), 6),
        "ik_solve_fk_pos_error_m": nested_get(debug, ["last_joint_ik_move_debug", "solve", "solve_fk_pos_error_m"]),
        "ik_solve_fk_ori_error_rad": nested_get(debug, ["last_joint_ik_move_debug", "solve", "solve_fk_ori_error_rad"]),
        "joint_final_error_rad": nested_get(debug, ["last_joint_ik_move_debug", "joint_move", "max_final_joint_error_rad"]),
        "joint_command_max_delta_rad": nested_get(debug, ["last_joint_ik_move_debug", "joint_move", "max_joint_delta_rad"]),
        "joint_command_steps": nested_get(debug, ["last_joint_ik_move_debug", "joint_move", "steps"]),
        "success": bool(tcp_err <= TCP_REACH_THRESHOLD_M),
    }}
    moves.append(result)
    record(label + "_end", **result)
    return result

record("initial")
open_gripper(arm=ARM)
settle_robot(steps=12)
record("after_open")

camera_names = get_camera_names()
object_pos, object_quat, object_extent = get_object_pose(
    OBJECT_NAME,
    return_bbox_extent=True,
    camera_name=None,
    external=False,
    arm=ARM,
    method="aabb_center",
)
object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4)
object_extent = np.asarray(object_extent, dtype=np.float64).reshape(3)

tcp_quat = SAFE_TCP_QUAT_XYZW.copy()
corrected_object_center = object_pos.copy()
if np.isfinite(object_extent[2]) and object_extent[2] > 0.0:
    extra_height = max(0.0, float(object_extent[2]) * 0.5 - float(OBJECT_HALF_HEIGHT_M))
    corrected_object_center[2] -= min(extra_height, 0.06)
grasp_tcp = corrected_object_center + np.array([0.0, 0.0, GRASP_CENTER_CLEARANCE_M], dtype=np.float64)
pre_tcp = grasp_tcp + np.array([0.0, 0.0, PRE_APPROACH_LIFT_M], dtype=np.float64)
lift_tcp = grasp_tcp + np.array([0.0, 0.0, LIFT_DISTANCE_M], dtype=np.float64)

record(
    "visual_to_joint_ik_contract",
    visual_output_frame="world",
    visual_output_meaning="right wrist camera object/aabb center, not tcp and not eef",
    robot_camera_names=camera_names,
    visual_object_position_world=object_pos.round(6).tolist(),
    visual_object_quat_xyzw=object_quat.round(6).tolist(),
    visual_object_extent=object_extent.round(6).tolist(),
    corrected_object_center_world=corrected_object_center.round(6).tolist(),
    derived_target_frame="world",
    derived_target_link="tcp/finger_center",
    low_level_entry="move_tcp_joint_ik",
    tcp_to_eef_conversion="inside move_tcp_joint_ik via tcp_pose_to_eef_pose",
    grasp_tcp_position_world=grasp_tcp.round(6).tolist(),
    target_tcp_quat_xyzw_world=tcp_quat.round(6).tolist(),
)

pre = run_move_tcp_joint_ik("pre_approach_from_visual_pose", pre_tcp, tcp_quat)
target = run_move_tcp_joint_ik("visual_grasp_tcp_target", grasp_tcp, tcp_quat)
close_gripper(arm=ARM)
settle_robot(steps=12)
record("after_close_gripper_at_visual_grasp_target")
lift = run_move_tcp_joint_ik("lift_after_close", lift_tcp, tcp_quat, pos_thresh=0.022, ori_thresh=0.26)
open_gripper(arm=ARM)
settle_robot(steps=8)
record("final")

RESULT = {{
    "trace": trace,
    "moves": moves,
    "visual_object_position_world": object_pos.round(6).tolist(),
    "visual_object_quat_xyzw": object_quat.round(6).tolist(),
    "visual_object_extent": object_extent.round(6).tolist(),
    "corrected_object_center_world": corrected_object_center.round(6).tolist(),
    "grasp_tcp_position_world": grasp_tcp.round(6).tolist(),
    "target_tcp_quat_xyzw_world": tcp_quat.round(6).tolist(),
    "target_pose_semantics": {{
        "visual_pose": "T_world_object_center from right wrist camera",
        "action_pose": "T_world_tcp/finger_center near corrected object center",
        "low_level_pose": "move_tcp_joint_ik converts T_world_tcp to T_world_eef before PyRoKi IK",
    }},
    "pre_approach_success": bool(pre["success"]),
    "target_success": bool(target["success"]),
    "lift_success": bool(lift["success"]),
    "visual_grasp_tcp_error_m": float(target["tcp_error_m"]),
    "all_success": bool(pre["success"] and target["success"] and lift["success"]),
    "thresholds": {{"tcp_reach_threshold_m": TCP_REACH_THRESHOLD_M}},
}}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Run X2 visual-pose-to-joint-IK code-exec trial")
    parser.add_argument("--output-dir", default="outputs/x2_visual_pose_to_joint_ik_code_exec_trial")
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
            "visual_pose_source": "RGB-D get_object_pose with depth gate",
            "visual_pose_semantics": "T_world_object_center",
            "action_pose_semantics": "T_world_tcp/finger-center near corrected object center, derived from right wrist camera object pose",
            "execution_primitive": "move_tcp_joint_ik",
        },
        "policy_code": POLICY_CODE,
        "steps": {},
        "video": {},
        "errors": [],
    }
    start = time.time()
    try:
        low_level = X2BehaviourLowLevel(
            controller_cfg="x2_robotiq85_joint_primitives.yaml",
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
        print("[x2-visual-joint-ik-trial] executing injected policy", flush=True)
        _obs, reward, terminated, truncated, info = exec_env.step(POLICY_CODE)
        result = exec_env._exec_globals.get("RESULT")

        truth_center = OBJECT_CENTER.copy()
        try:
            obj = low_level.env.scene.object_registry("name", OBJECT_NAME)
            truth_center = np.asarray(obj.aabb_center, dtype=np.float64).reshape(3)
        except Exception:
            pass

        visual_object_error = None
        true_target_error = None
        if isinstance(result, dict):
            visual_pos = np.asarray(result.get("visual_object_position_world"), dtype=np.float64).reshape(3)
            visual_object_error = float(np.linalg.norm(visual_pos - truth_center))
            true_target_tcp = truth_center + np.array([0.0, 0.0, GRASP_CENTER_CLEARANCE_M], dtype=np.float64)
            reached_tcp = None
            for move in result.get("moves") or []:
                if move.get("move_label") == "visual_grasp_tcp_target":
                    reached_tcp = np.asarray(move.get("reached_tcp_position"), dtype=np.float64).reshape(3)
            if reached_tcp is not None:
                true_target_error = float(np.linalg.norm(reached_tcp - true_target_tcp))
            result["truth_object_center_world"] = truth_center.round(6).tolist()
            result["visual_object_error_to_truth_m"] = round(visual_object_error, 6)
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
        result_ok = bool(isinstance(result, dict) and result.get("all_success"))
        visual_pose_ok = bool(visual_object_error is not None and visual_object_error <= OBJECT_POSE_THRESHOLD_M)
        true_target_ok = bool(true_target_error is not None and true_target_error <= TCP_REACH_THRESHOLD_M)
        video_ok = bool(summary["video"].get("combined", {}).get("written", False))
        summary["ok"] = bool(info.get("sandbox_rc") == 0 and result_ok and video_ok)
        summary["checks"] = {
            "sandbox_ok": bool(info.get("sandbox_rc") == 0),
            "result_all_success": result_ok,
            "visual_object_pose_ok": visual_pose_ok,
            "true_target_tcp_ok": true_target_ok,
            "video_ok": video_ok,
            "ok_semantics": "pass requires injected visual-derived TCP targets to execute successfully; truth errors are diagnostics",
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
    print(f"[x2-visual-joint-ik-trial] wrote {summary_path}", flush=True)
    for view, meta in summary["video"].get("views", {}).items():
        if meta.get("written"):
            print(f"[x2-visual-joint-ik-trial] wrote {view} video {meta['path']} ({meta['frame_count']} frames)", flush=True)
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"[x2-visual-joint-ik-trial] wrote combined video {meta['path']} ({meta['frame_count']} frames)", flush=True)
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"]), "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
