"""Execute a fixed X2 grasp target with PyRoKi precontact planning.

No vision is used here. The target grasp TCP pose is fixed in world frame.
The demo:

1. Plans and executes a PyRoKi collision-aware joint trajectory to a
   precontact TCP pose.
2. Inserts along the TCP approach axis through small TCP waypoints.
3. Closes the gripper on the fixed red cube, holds briefly, and reopens.

The summary records TCP errors, contact pairs, and video paths.
"""

from __future__ import annotations

import argparse
import json
import os
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
from x2_chest_camera_visual_chain_smoke import (
    ARM,
    OBJECT_CENTER,
    OBJECT_NAME,
    OBJECT_SIZE,
    TABLE_CENTER,
    TABLE_NAME,
    TABLE_SCALE,
    _jsonable,
    _scene_objects,
)
from x2_chest_visual_grasp_to_joint_ik_demo import _current_tcp, _pose_summary, _quat_error_rad
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos
from x2_replay_visual_target_joint_tracking import _execute_tcp_target
from x2_replay_visual_target_orientation_sweep import (
    BASE_QUATS,
    _axis_angle_quat_xyzw,
    _quat_multiply_xyzw,
    _quat_normalize,
    _quat_to_matrix_xyzw,
)


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated floats, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _quat_for_x_angle(angle: float) -> np.ndarray:
    return _quat_normalize(
        _quat_multiply_xyzw(
            BASE_QUATS["visual_candidate"],
            _axis_angle_quat_xyzw(np.array([1.0, 0.0, 0.0], dtype=np.float64), angle),
        )
    )


def _object_contact_summary(
    env: X2BehaviourLowLevel,
    object_name: str = OBJECT_NAME,
) -> dict[str, Any]:
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI

        obj = env.env.scene.object_registry("name", object_name)
        if obj is None:
            return {"ok": False, "error": f"object {object_name!r} not found"}
        query_set = set(env.robot.link_prim_paths)
        with_set = set(obj.link_prim_paths)
        current_pairs = RigidContactAPI.get_contact_pairs(env.env.scene.idx, query_set, with_set, current_only=True)
        accumulated_pairs = RigidContactAPI.get_contact_pairs(env.env.scene.idx, query_set, with_set, current_only=False)
        return {
            "ok": True,
            "object_name": object_name,
            "current_contact_count": int(len(current_pairs)),
            "current_contact_pairs": sorted([list(pair) for pair in current_pairs]),
            "accumulated_contact_count": int(len(accumulated_pairs)),
            "accumulated_contact_pairs": sorted([list(pair) for pair in accumulated_pairs]),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _obstacles(object_center: np.ndarray, table_center: np.ndarray, cube_margin: float) -> list[dict[str, Any]]:
    cube_extent = np.full(3, float(OBJECT_SIZE) + 2.0 * float(cube_margin), dtype=np.float64)
    table_extent = np.asarray(TABLE_SCALE, dtype=np.float64).reshape(3) + np.array([0.02, 0.02, 0.006], dtype=np.float64)
    return [
        {
            "type": "box",
            "name": OBJECT_NAME,
            "position": np.asarray(object_center, dtype=np.float64).reshape(3).tolist(),
            "extent": cube_extent.tolist(),
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "type": "box",
            "name": TABLE_NAME,
            "position": np.asarray(table_center, dtype=np.float64).reshape(3).tolist(),
            "extent": table_extent.tolist(),
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    ]


def _execute_joint_trajectory_with_contacts(
    env: X2BehaviourLowLevel,
    trajectory: list[np.ndarray],
    *,
    max_joint_step: float,
    max_steps_per_waypoint: int,
    settle_steps: int,
    hold_steps_per_waypoint: int,
) -> dict[str, Any]:
    waypoint_results: list[dict[str, Any]] = []
    joint_tracking_ok = True
    for idx, q_target in enumerate(trajectory):
        waypoint_ok = env._move_to_joint_positions(
            q_target,
            arm=ARM,
            max_joint_step=max_joint_step,
            max_steps=max_steps_per_waypoint,
            settle_steps=0,
            hold_steps_per_waypoint=hold_steps_per_waypoint,
        )
        debug = dict(getattr(env, "_last_joint_position_move_debug", {}) or {})
        debug["waypoint_index"] = int(idx)
        debug["joint_tracking_ok"] = bool(waypoint_ok)
        debug["object_contact"] = _object_contact_summary(env)
        waypoint_results.append(_jsonable(debug))
        joint_tracking_ok = bool(joint_tracking_ok and waypoint_ok)
    env.settle_robot_steps(steps=settle_steps)
    return {
        "joint_tracking_ok": bool(joint_tracking_ok),
        "waypoint_count": int(len(trajectory)),
        "waypoints": waypoint_results,
        "post_settle_contact": _object_contact_summary(env),
    }


def _target_error(api: X2ControlApi, target_tcp_pose: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    reached_tcp, reached_eef = _current_tcp(api, arm=ARM)
    target_eef = api.tcp_pose_to_eef_pose(target_tcp_pose, arm=ARM)
    return {
        "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
        "reached_eef": _pose_summary(reached_eef),
        "target_tcp_pose": _pose_summary(target_tcp_pose),
        "target_eef_pose": _pose_summary(target_eef),
        "tcp_error_m": round(float(np.linalg.norm(reached_tcp - np.asarray(target_tcp_pose[0], dtype=np.float64))), 6),
        "eef_pos_error_m": round(float(np.linalg.norm(np.asarray(reached_eef[0]) - np.asarray(target_eef[0]))), 6),
        "eef_ori_error_rad": round(_quat_error_rad(np.asarray(reached_eef[1]), np.asarray(target_eef[1])), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 fixed-target PyRoKi precontact + insertion grasp demo")
    parser.add_argument("--output-dir", default="outputs/x2_pyroki_precontact_insert_grasp_demo")
    parser.add_argument("--object-center", type=_parse_vec3, default=OBJECT_CENTER)
    parser.add_argument("--table-center", type=_parse_vec3, default=TABLE_CENTER)
    parser.add_argument("--grasp-target", type=_parse_vec3, default=OBJECT_CENTER)
    parser.add_argument("--grasp-x-angle", type=float, default=90.0)
    parser.add_argument("--precontact-distance", type=float, default=0.06)
    parser.add_argument("--cube-margin", type=float, default=0.01)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--insert-waypoints", type=int, default=6)
    parser.add_argument("--max-joint-step", type=float, default=0.018)
    parser.add_argument("--insert-max-joint-step", type=float, default=0.014)
    parser.add_argument("--hold-steps-per-waypoint", type=int, default=4)
    parser.add_argument("--insert-hold-steps-per-waypoint", type=int, default=6)
    parser.add_argument("--settle-steps", type=int, default=24)
    parser.add_argument("--close-hold-steps", type=int, default=36)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ok": False, "errors": []}
    start = time.time()
    env = None
    try:
        objects = _scene_objects()
        for obj in objects:
            if obj.get("name") == OBJECT_NAME:
                obj["position"] = np.asarray(args.object_center, dtype=np.float64).tolist()
            elif obj.get("name") == TABLE_NAME:
                obj["position"] = np.asarray(args.table_center, dtype=np.float64).tolist()
        env = X2BehaviourLowLevel(
            controller_cfg="x2_robotiq85_joint_primitives.yaml",
            objects=objects,
            external_sensors=[GLOBAL_CAMERA],
            robot_obs_modalities=["rgb"],
            robot_camera_resolution=256,
            chest_camera=True,
            chest_camera_resolution=256,
            save_video=False,
        )
        api = X2ControlApi(env, use_vision_models=False, use_graspnet=False)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        env.enable_video_capture(True, clear=True)

        grasp_pos = np.asarray(args.grasp_target, dtype=np.float64).reshape(3)
        grasp_quat = _quat_for_x_angle(float(args.grasp_x_angle))
        tcp_axis_world = _quat_to_matrix_xyzw(grasp_quat) @ np.asarray(api.get_tcp_offset_eef(arm=ARM), dtype=np.float64)
        tcp_axis_world = tcp_axis_world / max(float(np.linalg.norm(tcp_axis_world)), 1e-12)
        precontact_pos = grasp_pos - tcp_axis_world * float(args.precontact_distance)
        grasp_tcp_pose = (grasp_pos, grasp_quat)
        precontact_tcp_pose = (precontact_pos, grasp_quat)
        obstacles = _obstacles(np.asarray(args.object_center), np.asarray(args.table_center), float(args.cube_margin))

        start_tcp, start_eef = _current_tcp(api, arm=ARM)
        plan = api.plan_tcp_pyroki_trajopt(
            precontact_tcp_pose,
            arm=ARM,
            obstacles_world=obstacles,
            timesteps=int(args.timesteps),
            dt=float(args.dt),
        )
        precontact_motion = _execute_joint_trajectory_with_contacts(
            env,
            plan["joint_trajectory"],
            max_joint_step=float(args.max_joint_step),
            max_steps_per_waypoint=80,
            settle_steps=int(args.settle_steps),
            hold_steps_per_waypoint=int(args.hold_steps_per_waypoint),
        )
        precontact_error = _target_error(api, precontact_tcp_pose)
        precontact_contact = _object_contact_summary(env)

        insertion_results: list[dict[str, Any]] = []
        for idx, distance in enumerate(np.linspace(float(args.precontact_distance), 0.0, int(args.insert_waypoints) + 1)[1:]):
            waypoint_pos = grasp_pos - tcp_axis_world * float(distance)
            waypoint_pose = (waypoint_pos, grasp_quat)
            label = "grasp" if idx == int(args.insert_waypoints) - 1 else f"insert_{idx:02d}"
            result = _execute_tcp_target(
                api,
                label,
                waypoint_pose,
                max_joint_step=float(args.insert_max_joint_step),
                max_steps=240,
                settle_steps=int(args.settle_steps),
                hold_steps_per_waypoint=int(args.insert_hold_steps_per_waypoint),
                repeat_attempts=1 if label != "grasp" else 2,
                repeat_tcp_threshold=0.022,
                repeat_ori_threshold=0.35,
            )
            result["object_contact"] = _object_contact_summary(env)
            insertion_results.append(_jsonable(result))

        before_close_error = _target_error(api, grasp_tcp_pose)
        before_close_contact = _object_contact_summary(env)
        api.close_gripper(arm=ARM)
        api.settle_robot(steps=int(args.close_hold_steps))
        after_close_contact = _object_contact_summary(env)
        after_close_error = _target_error(api, grasp_tcp_pose)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        after_open_contact = _object_contact_summary(env)

        env._record_frame()
        precontact_clear = precontact_contact.get("current_contact_count", 1) == 0
        before_close_reached = float(before_close_error["tcp_error_m"]) <= 0.025
        summary.update(
            {
                "ok": bool(precontact_clear and before_close_reached),
                "target_contract": "fixed target is T_world_tcp; PyRoKi plans to precontact T_world_tcp converted internally to T_world_eef",
                "object_fixed": True,
                "object_center": np.round(np.asarray(args.object_center, dtype=np.float64), 6).tolist(),
                "table_center": np.round(np.asarray(args.table_center, dtype=np.float64), 6).tolist(),
                "grasp_tcp_pose": _pose_summary(grasp_tcp_pose),
                "precontact_tcp_pose": _pose_summary(precontact_tcp_pose),
                "tcp_axis_world": np.round(tcp_axis_world, 6).tolist(),
                "start_tcp_position": np.round(start_tcp, 6).tolist(),
                "start_eef": _pose_summary(start_eef),
                "obstacles_world": obstacles,
                "plan_debug": plan.get("debug"),
                "precontact_motion": precontact_motion,
                "precontact_error": precontact_error,
                "precontact_contact": precontact_contact,
                "insertion_results": insertion_results,
                "before_close_error": before_close_error,
                "before_close_contact": before_close_contact,
                "after_close_error": after_close_error,
                "after_close_contact": after_close_contact,
                "after_open_contact": after_open_contact,
                "motion_debug": api.get_last_motion_debug(),
                "video": _write_videos(output_dir, env.get_video_frames(), fps=int(args.video_fps)),
            }
        )
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)
    finally:
        summary["elapsed_s"] = round(time.time() - start, 3)
        (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": summary.get("ok", False),
                "precontact_tcp_error_m": (summary.get("precontact_error") or {}).get("tcp_error_m"),
                "before_close_tcp_error_m": (summary.get("before_close_error") or {}).get("tcp_error_m"),
                "precontact_contact_count": (summary.get("precontact_contact") or {}).get("current_contact_count"),
                "before_close_contact_count": (summary.get("before_close_contact") or {}).get("current_contact_count"),
                "after_close_contact_count": (summary.get("after_close_contact") or {}).get("current_contact_count"),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
