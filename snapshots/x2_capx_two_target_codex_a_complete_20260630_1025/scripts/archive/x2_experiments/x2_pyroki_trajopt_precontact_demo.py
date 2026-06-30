"""Smoke test X2 PyRoKi collision-aware trajectory optimization.

This script validates the planning route before wiring it into the visual
grasp demo. It plans only to a precontact TCP pose near the red cube while the
cube and tabletop are represented as simple PyRoKi box obstacles.
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
            _axis_angle_quat_xyzw(np.array([1.0, 0.0, 0.0]), angle),
        )
    )


def _object_contact_summary(env: X2BehaviourLowLevel, object_name: str = OBJECT_NAME) -> dict[str, Any]:
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI

        obj = env.env.scene.object_registry("name", object_name)
        if obj is None:
            return {"ok": False, "error": f"object {object_name!r} not found"}
        pairs = RigidContactAPI.get_contact_pairs(
            env.env.scene.idx,
            set(env.robot.link_prim_paths),
            set(obj.link_prim_paths),
            current_only=True,
        )
        return {
            "ok": True,
            "object_name": object_name,
            "current_contact_count": int(len(pairs)),
            "current_contact_pairs": sorted([list(pair) for pair in pairs]),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 PyRoKi trajopt precontact smoke")
    parser.add_argument("--output-dir", default="outputs/x2_pyroki_trajopt_precontact_demo")
    parser.add_argument("--object-center", type=_parse_vec3, default=OBJECT_CENTER)
    parser.add_argument("--table-center", type=_parse_vec3, default=TABLE_CENTER)
    parser.add_argument("--grasp-target", type=_parse_vec3, default=OBJECT_CENTER)
    parser.add_argument("--grasp-x-angle", type=float, default=90.0)
    parser.add_argument("--precontact-distance", type=float, default=0.04)
    parser.add_argument("--cube-margin", type=float, default=0.01)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--max-joint-step", type=float, default=0.018)
    parser.add_argument("--hold-steps-per-waypoint", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=24)
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
        target_tcp_pose = (precontact_pos, grasp_quat)
        obstacles = _obstacles(np.asarray(args.object_center), np.asarray(args.table_center), float(args.cube_margin))

        start_tcp, start_eef = _current_tcp(api, arm=ARM)
        plan = api.plan_tcp_pyroki_trajopt(
            target_tcp_pose,
            arm=ARM,
            obstacles_world=obstacles,
            timesteps=int(args.timesteps),
            dt=float(args.dt),
        )
        move_ok = bool(
            env._move_through_joint_trajectory(
                plan["joint_trajectory"],
                arm=ARM,
                max_joint_step=float(args.max_joint_step),
                max_steps_per_waypoint=80,
                settle_steps=int(args.settle_steps),
                hold_steps_per_waypoint=int(args.hold_steps_per_waypoint),
            )
        )
        reached_tcp, reached_eef = _current_tcp(api, arm=ARM)
        tcp_error = float(np.linalg.norm(reached_tcp - precontact_pos))
        eef_target = api.tcp_pose_to_eef_pose(target_tcp_pose, arm=ARM)
        eef_pos_error = float(np.linalg.norm(np.asarray(reached_eef[0]) - np.asarray(eef_target[0])))
        eef_ori_error = _quat_error_rad(np.asarray(reached_eef[1]), np.asarray(eef_target[1]))
        contact = _object_contact_summary(env)
        task_ok = bool(tcp_error <= 0.025 and contact.get("current_contact_count", 1) == 0)
        env._record_frame()
        summary.update(
            {
                "ok": task_ok,
                "target_contract": "target_tcp_pose is T_world_tcp; PyRoKi plans to converted T_world_eef in X2 base frame",
                "grasp_tcp_pose": _pose_summary((grasp_pos, grasp_quat)),
                "precontact_tcp_pose": _pose_summary(target_tcp_pose),
                "tcp_axis_world": np.round(tcp_axis_world, 6).tolist(),
                "start_tcp_position": np.round(start_tcp, 6).tolist(),
                "start_eef": _pose_summary(start_eef),
                "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
                "reached_eef": _pose_summary(reached_eef),
                "tcp_error_m": round(tcp_error, 6),
                "eef_pos_error_m": round(eef_pos_error, 6),
                "eef_ori_error_rad": round(eef_ori_error, 6),
                "joint_tracking_ok": move_ok,
                "object_contact": contact,
                "obstacles_world": obstacles,
                "plan_debug": plan.get("debug"),
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
                "tcp_error_m": summary.get("tcp_error_m"),
                "contact_count": (summary.get("object_contact") or {}).get("current_contact_count"),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
