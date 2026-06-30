"""Replay a saved visual grasp target and diagnose X2 joint tracking.

This script does not start the visual services. It reads the selected
``T_world_tcp`` poses from a previous visual-chain summary and replays them in
the same simple table/cube scene with configurable joint-command timing.
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

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi
from x2_chest_camera_visual_chain_smoke import ARM, OBJECT_CENTER, _jsonable, _scene_objects
from x2_chest_visual_grasp_to_joint_ik_demo import _current_tcp, _pose_summary, _quat_error_rad
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated floats, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _load_pose(candidate: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    value = candidate[key]
    return np.asarray(value[0], dtype=np.float64), np.asarray(value[1], dtype=np.float64)


def _nested_get(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _execute_tcp_target(
    api: X2ControlApi,
    label: str,
    target_tcp_pose: tuple[np.ndarray, np.ndarray],
    *,
    command_tcp_pose: tuple[np.ndarray, np.ndarray] | None = None,
    max_joint_step: float,
    max_steps: int,
    settle_steps: int,
    hold_steps_per_waypoint: int,
    repeat_attempts: int,
    repeat_tcp_threshold: float,
    repeat_ori_threshold: float,
) -> dict[str, Any]:
    target_pos = np.asarray(target_tcp_pose[0], dtype=np.float64).reshape(3)
    target_quat = np.asarray(target_tcp_pose[1], dtype=np.float64).reshape(4)
    if command_tcp_pose is None:
        command_pos, command_quat = target_pos, target_quat
    else:
        command_pos = np.asarray(command_tcp_pose[0], dtype=np.float64).reshape(3)
        command_quat = np.asarray(command_tcp_pose[1], dtype=np.float64).reshape(4)
    target_eef_pose = api.tcp_pose_to_eef_pose((target_pos, target_quat), arm=ARM)
    command_eef_pose = api.tcp_pose_to_eef_pose((command_pos, command_quat), arm=ARM)
    start_tcp, start_eef = _current_tcp(api, arm=ARM)
    q_before = np.asarray(api.get_current_joint_positions(), dtype=np.float64)
    t0 = time.time()
    attempts: list[dict[str, Any]] = []
    ok = False
    best_tcp_error = float("inf")
    max_attempts = max(1, int(repeat_attempts))
    for attempt_idx in range(max_attempts):
        attempt_ok = bool(
            api.move_tcp_joint_ik(
                (command_pos, command_quat),
                arm=ARM,
                pos_thresh=repeat_tcp_threshold,
                ori_thresh=repeat_ori_threshold,
                max_joint_step=max_joint_step,
                max_steps=max_steps,
                settle_steps=settle_steps,
                hold_steps_per_waypoint=hold_steps_per_waypoint,
            )
        )
        api.settle_robot(steps=12)
        reached_tcp, reached_eef = _current_tcp(api, arm=ARM)
        debug = api.get_last_motion_debug()
        joint_move = _nested_get(debug, ["last_joint_ik_move_debug", "joint_move"], {}) or {}
        solve = _nested_get(debug, ["last_joint_ik_move_debug", "solve"], {}) or {}
        tcp_error = float(np.linalg.norm(reached_tcp - target_pos))
        eef_pos_error = float(np.linalg.norm(np.asarray(reached_eef[0]) - np.asarray(target_eef_pose[0])))
        eef_ori_error = _quat_error_rad(np.asarray(reached_eef[1]), np.asarray(target_eef_pose[1]))
        attempts.append(
            {
                "attempt": int(attempt_idx + 1),
                "ok": bool(attempt_ok),
                "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
                "tcp_error_m": round(tcp_error, 6),
                "eef_pos_error_m": round(eef_pos_error, 6),
                "eef_ori_error_rad": round(eef_ori_error, 6),
                "ik_solve_fk_pos_error_m": solve.get("solve_fk_pos_error_m"),
                "ik_solve_fk_ori_error_rad": solve.get("solve_fk_ori_error_rad"),
                "joint_final_error_rad": joint_move.get("max_final_joint_error_rad"),
                "joint_command_max_delta_rad": joint_move.get("max_joint_delta_rad"),
                "joint_command_steps": joint_move.get("steps"),
                "joint_action_steps_sent": joint_move.get("action_steps_sent"),
                "joint_hold_steps_per_waypoint": joint_move.get("hold_steps_per_waypoint"),
                "joint_settle_steps": joint_move.get("settle_steps"),
                "estimated_command_duration_s": joint_move.get("estimated_command_duration_s"),
                "estimated_settle_duration_s": joint_move.get("estimated_settle_duration_s"),
            }
        )
        ok = bool(tcp_error <= repeat_tcp_threshold and eef_ori_error <= repeat_ori_threshold)
        if ok:
            break
        prev_best_tcp_error = best_tcp_error
        best_tcp_error = min(best_tcp_error, tcp_error)
        if attempt_idx > 0 and tcp_error > prev_best_tcp_error + 0.005:
            break
    q_after = np.asarray(api.get_current_joint_positions(), dtype=np.float64)
    reached_tcp, reached_eef = _current_tcp(api, arm=ARM)
    debug = api.get_last_motion_debug()
    joint_move = _nested_get(debug, ["last_joint_ik_move_debug", "joint_move"], {}) or {}
    solve = _nested_get(debug, ["last_joint_ik_move_debug", "solve"], {}) or {}
    return {
        "label": label,
        "ok": ok,
        "elapsed_s": round(time.time() - t0, 3),
        "start_tcp_position": np.round(start_tcp, 6).tolist(),
        "start_eef": _pose_summary(start_eef),
        "target_tcp_pose": _pose_summary((target_pos, target_quat)),
        "target_eef_pose": _pose_summary(target_eef_pose),
        "command_tcp_pose": _pose_summary((command_pos, command_quat)),
        "command_eef_pose": _pose_summary(command_eef_pose),
        "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
        "reached_eef": _pose_summary(reached_eef),
        "tcp_error_m": round(float(np.linalg.norm(reached_tcp - target_pos)), 6),
        "eef_pos_error_m": round(float(np.linalg.norm(np.asarray(reached_eef[0]) - np.asarray(target_eef_pose[0]))), 6),
        "eef_ori_error_rad": round(_quat_error_rad(np.asarray(reached_eef[1]), np.asarray(target_eef_pose[1])), 6),
        "max_abs_joint_delta_rad": round(float(np.max(np.abs(q_after - q_before))), 6),
        "ik_solve_fk_pos_error_m": solve.get("solve_fk_pos_error_m"),
        "ik_solve_fk_ori_error_rad": solve.get("solve_fk_ori_error_rad"),
        "joint_final_error_rad": joint_move.get("max_final_joint_error_rad"),
        "joint_command_max_delta_rad": joint_move.get("max_joint_delta_rad"),
        "joint_command_steps": joint_move.get("steps"),
        "joint_action_steps_sent": joint_move.get("action_steps_sent"),
        "joint_hold_steps_per_waypoint": joint_move.get("hold_steps_per_waypoint"),
        "joint_settle_steps": joint_move.get("settle_steps"),
        "estimated_command_duration_s": joint_move.get("estimated_command_duration_s"),
        "estimated_settle_duration_s": joint_move.get("estimated_settle_duration_s"),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "last_motion_debug": _jsonable(debug),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay saved X2 visual TCP targets with configurable joint timing")
    parser.add_argument("--source-summary", default="outputs/x2_chest_visual_grasp_to_joint_ik_demo_v2/summary.json")
    parser.add_argument("--output-dir", default="outputs/x2_replay_visual_target_joint_tracking")
    parser.add_argument("--config", default="x2_robotiq85_joint_primitives.yaml")
    parser.add_argument("--condition-name", default="baseline")
    parser.add_argument("--max-joint-step", type=float, default=0.022)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--hold-steps-per-waypoint", type=int, default=1)
    parser.add_argument("--repeat-attempts", type=int, default=1)
    parser.add_argument("--repeat-tcp-threshold", type=float, default=0.02)
    parser.add_argument("--repeat-ori-threshold", type=float, default=0.26)
    parser.add_argument("--pregrasp-command-bias", type=_parse_vec3, default=np.zeros(3, dtype=np.float64))
    parser.add_argument("--grasp-command-bias", type=_parse_vec3, default=np.zeros(3, dtype=np.float64))
    parser.add_argument("--lift-command-bias", type=_parse_vec3, default=np.zeros(3, dtype=np.float64))
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary_path = Path(args.source_summary)
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    candidate = source["selected_candidate"]

    summary: dict[str, Any] = {
        "ok": False,
        "condition": {
            "name": args.condition_name,
            "max_joint_step": args.max_joint_step,
            "max_steps": args.max_steps,
            "settle_steps": args.settle_steps,
            "hold_steps_per_waypoint": args.hold_steps_per_waypoint,
            "repeat_attempts": args.repeat_attempts,
            "repeat_tcp_threshold": args.repeat_tcp_threshold,
            "repeat_ori_threshold": args.repeat_ori_threshold,
            "pregrasp_command_bias": np.round(args.pregrasp_command_bias, 6).tolist(),
            "grasp_command_bias": np.round(args.grasp_command_bias, 6).tolist(),
            "lift_command_bias": np.round(args.lift_command_bias, 6).tolist(),
        },
        "source_summary": str(source_summary_path),
        "object_center_world": np.round(OBJECT_CENTER, 6).tolist(),
        "target_contract": "saved candidate poses are T_world_tcp; move_tcp_joint_ik converts T_world_tcp to T_world_eef",
        "motions": [],
        "after_close": {},
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
            robot_obs_modalities=["rgb"],
            robot_camera_resolution=256,
            chest_camera=True,
            chest_camera_resolution=256,
            save_video=False,
        )
        api = X2ControlApi(env, use_vision_models=False, use_graspnet=False)
        summary["control_timing"] = api.get_control_timing()
        summary["selected_candidate"] = _jsonable(candidate)

        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        command_bias_by_label = {
            "pregrasp": np.asarray(args.pregrasp_command_bias, dtype=np.float64),
            "grasp": np.asarray(args.grasp_command_bias, dtype=np.float64),
            "lift": np.asarray(args.lift_command_bias, dtype=np.float64),
        }
        for label, key in (("pregrasp", "pregrasp_tcp_pose"), ("grasp", "grasp_tcp_pose")):
            target_pose = _load_pose(candidate, key)
            command_pose = (target_pose[0] + command_bias_by_label[label], target_pose[1])
            result = _execute_tcp_target(
                api,
                label,
                target_pose,
                command_tcp_pose=command_pose,
                max_joint_step=args.max_joint_step,
                max_steps=args.max_steps,
                settle_steps=args.settle_steps,
                hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                repeat_attempts=args.repeat_attempts,
                repeat_tcp_threshold=args.repeat_tcp_threshold,
                repeat_ori_threshold=args.repeat_ori_threshold,
            )
            summary["motions"].append(result)

        grasp_target = _load_pose(candidate, "grasp_tcp_pose")
        before_close_tcp, before_close_eef = _current_tcp(api, arm=ARM)
        api.close_gripper(arm=ARM)
        api.settle_robot(steps=16)
        after_close_tcp, after_close_eef = _current_tcp(api, arm=ARM)
        summary["after_close"] = {
            "target_grasp_tcp_pose": _pose_summary(grasp_target),
            "before_close_tcp_position": np.round(before_close_tcp, 6).tolist(),
            "before_close_eef": _pose_summary(before_close_eef),
            "before_close_tcp_error_m": round(float(np.linalg.norm(before_close_tcp - grasp_target[0])), 6),
            "after_close_tcp_position": np.round(after_close_tcp, 6).tolist(),
            "after_close_eef": _pose_summary(after_close_eef),
            "after_close_tcp_error_m": round(float(np.linalg.norm(after_close_tcp - grasp_target[0])), 6),
            "note": "after_close TCP can shift because the TCP/finger-center offset changes when the gripper closes",
        }

        result = _execute_tcp_target(
            api,
            "lift",
            _load_pose(candidate, "lift_tcp_pose"),
            command_tcp_pose=(_load_pose(candidate, "lift_tcp_pose")[0] + command_bias_by_label["lift"], _load_pose(candidate, "lift_tcp_pose")[1]),
            max_joint_step=args.max_joint_step,
            max_steps=args.max_steps,
            settle_steps=args.settle_steps,
            hold_steps_per_waypoint=args.hold_steps_per_waypoint,
            repeat_attempts=args.repeat_attempts,
            repeat_tcp_threshold=args.repeat_tcp_threshold,
            repeat_ori_threshold=args.repeat_ori_threshold,
        )
        summary["motions"].append(result)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        env._record_frame()
        summary["video"] = _write_videos(output_dir, env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})
        summary["ok"] = all(bool(item["ok"]) for item in summary["motions"])
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], file=sys.stderr, flush=True)
    finally:
        summary["elapsed_s"] = round(time.time() - start, 3)
        (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"ok": summary["ok"], "condition": summary["condition"], "output_dir": str(output_dir)}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
