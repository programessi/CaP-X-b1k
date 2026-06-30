"""Sweep TCP orientations for the saved X2 visual grasp target using joint IK."""

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

import numpy as np

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi
from x2_chest_camera_visual_chain_smoke import ARM, _jsonable, _scene_objects
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos
from x2_replay_visual_target_joint_tracking import _execute_tcp_target, _load_pose


BASE_QUATS = {
    "visual_candidate": np.array([0.6488835215568542, 0.16876192390918732, 0.7338562607765198, 0.10919998586177826], dtype=np.float64),
    "fixed_target_reached": np.array([0.623938, 0.172114, 0.755461, 0.101771], dtype=np.float64),
    "mid_roll": np.array([0.621289, 0.252274, 0.713342, 0.20372], dtype=np.float64),
    "transport_yaw": np.array([0.665412, 0.082876, 0.741762, 0.012152], dtype=np.float64),
}


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return q / max(float(np.linalg.norm(q)), 1e-12)


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


def _axis_angle_quat_xyzw(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * math.radians(float(angle_deg))
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _quat_to_matrix_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _candidate_quats() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    seen: set[tuple[float, ...]] = set()
    for base_name, base_quat in BASE_QUATS.items():
        for axis_name, axis in axes.items():
            for angle in (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0):
                if angle == 0.0 and axis_name != "x":
                    continue
                quat = _quat_multiply_xyzw(base_quat, _axis_angle_quat_xyzw(axis, angle))
                key = tuple(np.round(quat, 5))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "label": f"{base_name}_{axis_name}_{angle:+.0f}",
                        "quat_xyzw": quat,
                        "base": base_name,
                        "axis": axis_name,
                        "angle_deg": float(angle),
                    }
                )
    return candidates


def _visual_x_angle_candidates(angles: list[float]) -> list[dict[str, Any]]:
    base_quat = BASE_QUATS["visual_candidate"]
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return [
        {
            "label": f"visual_candidate_x_{angle:+.1f}",
            "quat_xyzw": _quat_multiply_xyzw(base_quat, _axis_angle_quat_xyzw(axis, angle)),
            "base": "visual_candidate",
            "axis": "x",
            "angle_deg": float(angle),
        }
        for angle in angles
    ]


def _restore(env: X2BehaviourLowLevel, root_pose: tuple[Any, Any], q: Any, qd: Any) -> None:
    env.robot.set_position_orientation(position=root_pose[0], orientation=root_pose[1])
    env.robot.set_joint_positions(q, drive=False)
    env.robot.set_joint_velocities(qd, drive=False)
    env._set_arm(ARM)
    env._hold_current_hand_target(env._arm_name(ARM))
    env.settle_robot_steps(steps=4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep X2 saved visual grasp TCP orientations through joint IK")
    parser.add_argument("--source-summary", default="outputs/x2_chest_visual_grasp_to_joint_ik_demo_v2/summary.json")
    parser.add_argument("--output-dir", default="outputs/x2_replay_visual_target_orientation_sweep")
    parser.add_argument("--config", default="x2_robotiq85_joint_primitives.yaml")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--visual-x-angles", type=_parse_float_list, default=None)
    parser.add_argument("--hold-steps-per-waypoint", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--repeat-attempts", type=int, default=2)
    parser.add_argument("--max-joint-step", type=float, default=0.022)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))
    candidate = source["selected_candidate"]
    grasp_pos, _grasp_quat = _load_pose(candidate, "grasp_tcp_pose")
    pre_pos, _pre_quat = _load_pose(candidate, "pregrasp_tcp_pose")

    summary: dict[str, Any] = {
        "ok": False,
        "source_summary": args.source_summary,
        "target_grasp_position_world": np.round(grasp_pos, 6).tolist(),
        "target_pregrasp_position_world": np.round(pre_pos, 6).tolist(),
        "results": [],
        "best": None,
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
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        root_pose = env.robot.get_position_orientation()
        q0 = env.robot.get_joint_positions().clone()
        qd0 = env.robot.get_joint_velocities().clone()
        tcp_offset = np.asarray(api.get_tcp_offset_eef(arm=ARM), dtype=np.float64)
        env.enable_video_capture(True, clear=True)

        candidates = (
            _visual_x_angle_candidates(args.visual_x_angles)
            if args.visual_x_angles is not None
            else _candidate_quats()[: max(1, int(args.max_candidates))]
        )
        for idx, item in enumerate(candidates):
            _restore(env, root_pose, q0, qd0)
            quat = _quat_normalize(np.asarray(item["quat_xyzw"], dtype=np.float64))
            print(f"[x2-orientation-joint-sweep] {idx + 1}/{len(candidates)} {item['label']}", flush=True)
            pre = _execute_tcp_target(
                api,
                f"{item['label']}_pregrasp",
                (pre_pos, quat),
                max_joint_step=args.max_joint_step,
                max_steps=240,
                settle_steps=args.settle_steps,
                hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                repeat_attempts=args.repeat_attempts,
                repeat_tcp_threshold=0.025,
                repeat_ori_threshold=0.35,
            )
            grasp = _execute_tcp_target(
                api,
                f"{item['label']}_grasp",
                (grasp_pos, quat),
                max_joint_step=args.max_joint_step,
                max_steps=240,
                settle_steps=args.settle_steps,
                hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                repeat_attempts=args.repeat_attempts,
                repeat_tcp_threshold=0.025,
                repeat_ori_threshold=0.35,
            )
            world_offset = _quat_to_matrix_xyzw(quat) @ tcp_offset
            result = {
                **{k: _jsonable(v) for k, v in item.items() if k != "quat_xyzw"},
                "quat_xyzw": np.round(quat, 6).tolist(),
                "tcp_offset_world": np.round(world_offset, 6).tolist(),
                "pregrasp": _jsonable(pre),
                "grasp": _jsonable(grasp),
                "score": float(grasp["tcp_error_m"]) + 0.25 * float(pre["tcp_error_m"]),
            }
            summary["results"].append(result)

        ranked = sorted(summary["results"], key=lambda x: (float(x["grasp"]["tcp_error_m"]), float(x["pregrasp"]["tcp_error_m"])))
        summary["best"] = ranked[0] if ranked else None
        summary["ok"] = bool(ranked and float(ranked[0]["grasp"]["tcp_error_m"]) <= 0.025)
        env._record_frame()
        summary["video"] = _write_videos(output_dir, env.get_video_frames(), fps=args.video_fps)
        summary["video_sources"] = getattr(env, "_last_video_sources", {})
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], file=sys.stderr, flush=True)
    finally:
        summary["elapsed_s"] = round(time.time() - start, 3)
        (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"ok": summary["ok"], "best": None if summary["best"] is None else {
        "label": summary["best"]["label"],
        "grasp_tcp_error_m": summary["best"]["grasp"]["tcp_error_m"],
        "pregrasp_tcp_error_m": summary["best"]["pregrasp"]["tcp_error_m"],
    }, "output_dir": str(output_dir)}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
