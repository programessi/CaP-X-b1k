"""Test pregrasp/grasp orientation combinations for the saved X2 target."""

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
from x2_chest_camera_visual_chain_smoke import ARM, OBJECT_NAME, OBJECT_SIZE, TABLE_NAME, _jsonable, _scene_objects
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos
from x2_replay_visual_target_joint_tracking import _execute_tcp_target, _load_pose
from x2_replay_visual_target_orientation_sweep import (
    BASE_QUATS,
    _axis_angle_quat_xyzw,
    _quat_multiply_xyzw,
    _quat_normalize,
    _quat_to_matrix_xyzw,
)
from x2_replay_visual_target_joint_tracking import _parse_vec3


def _quat_for_x_angle(angle: float) -> np.ndarray:
    return _quat_multiply_xyzw(BASE_QUATS["visual_candidate"], _axis_angle_quat_xyzw(np.array([1.0, 0.0, 0.0]), angle))


def _parse_combos(value: str) -> list[tuple[float, float]]:
    combos: list[tuple[float, float]] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        pre, grasp = item.split(":")
        combos.append((float(pre), float(grasp)))
    return combos


def _parse_optional_vec3(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated floats, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _restore(env: X2BehaviourLowLevel, root_pose: tuple[Any, Any], q: Any, qd: Any) -> None:
    env.robot.set_position_orientation(position=root_pose[0], orientation=root_pose[1])
    env.robot.set_joint_positions(q, drive=False)
    env.robot.set_joint_velocities(qd, drive=False)
    env._set_arm(ARM)
    env._hold_current_hand_target(env._arm_name(ARM))
    env.settle_robot_steps(steps=4)


def _object_contact_summary(env: X2BehaviourLowLevel, object_name: str = OBJECT_NAME) -> dict[str, Any]:
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI

        obj = env.env.scene.object_registry("name", object_name)
        if obj is None:
            return {"ok": False, "error": f"object {object_name!r} not found"}
        query_set = set(env.robot.link_prim_paths)
        with_set = set(obj.link_prim_paths)
        current_pairs = RigidContactAPI.get_contact_pairs(env.env.scene.idx, query_set, with_set, current_only=True)
        return {
            "ok": True,
            "object_name": object_name,
            "current_contact_count": int(len(current_pairs)),
            "current_contact_pairs": sorted([list(pair) for pair in current_pairs]),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _inside_inflated_box(point: np.ndarray, center: np.ndarray, size: float, margin: float) -> bool:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    half = 0.5 * float(size) + float(margin)
    return bool(np.all(np.abs(point - center) <= half))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test X2 pregrasp/grasp TCP orientation combos")
    parser.add_argument("--source-summary", default="outputs/x2_chest_visual_grasp_to_joint_ik_demo_v2/summary.json")
    parser.add_argument("--output-dir", default="outputs/x2_replay_visual_target_orientation_combo")
    parser.add_argument("--combos", type=_parse_combos, default=_parse_combos("80:90;90:90;110:90;80:80"))
    parser.add_argument("--hold-steps-per-waypoint", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--repeat-attempts", type=int, default=3)
    parser.add_argument("--pregrasp-repeat-attempts", type=int, default=None)
    parser.add_argument("--grasp-repeat-attempts", type=int, default=None)
    parser.add_argument("--grasp-command-bias", type=_parse_vec3, default=np.zeros(3, dtype=np.float64))
    parser.add_argument("--table-z-offset", type=float, default=0.0)
    parser.add_argument("--object-center", type=_parse_optional_vec3, default=None)
    parser.add_argument("--table-center", type=_parse_optional_vec3, default=None)
    parser.add_argument("--grasp-target", type=_parse_optional_vec3, default=None)
    parser.add_argument("--pregrasp-lift", type=float, default=0.06)
    parser.add_argument("--pregrasp-mode", choices=("world_z", "tcp_axis"), default="world_z")
    parser.add_argument("--approach-distance", type=float, default=0.09)
    parser.add_argument("--cartesian-approach", action="store_true")
    parser.add_argument("--approach-waypoints", type=int, default=5)
    parser.add_argument("--precontact-distance", type=float, default=0.03)
    parser.add_argument("--final-insertion-waypoints", type=int, default=0)
    parser.add_argument("--guard-margin", type=float, default=0.008)
    parser.add_argument("--gripper-proxy-guard", action="store_true")
    parser.add_argument("--gripper-proxy-radius", type=float, default=0.012)
    parser.add_argument("--gripper-proxy-margin", type=float, default=0.012)
    parser.add_argument("--max-joint-step", type=float, default=0.022)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.source_summary).read_text(encoding="utf-8"))
    candidate = source["selected_candidate"]
    source_grasp_pos, _ = _load_pose(candidate, "grasp_tcp_pose")
    source_pre_pos, _ = _load_pose(candidate, "pregrasp_tcp_pose")
    grasp_pos = np.asarray(args.grasp_target, dtype=np.float64) if args.grasp_target is not None else source_grasp_pos
    pre_pos = grasp_pos + np.array([0.0, 0.0, float(args.pregrasp_lift)], dtype=np.float64) if args.grasp_target is not None else source_pre_pos

    summary: dict[str, Any] = {"ok": False, "results": [], "best": None, "video": {}, "errors": []}
    start = time.time()
    env = None
    try:
        objects = _scene_objects()
        if args.object_center is not None:
            for obj in objects:
                if obj.get("name") == OBJECT_NAME:
                    obj["position"] = np.asarray(args.object_center, dtype=np.float64).tolist()
        if args.table_center is not None:
            for obj in objects:
                if obj.get("name") == TABLE_NAME:
                    obj["position"] = np.asarray(args.table_center, dtype=np.float64).tolist()
        if abs(float(args.table_z_offset)) > 1e-9:
            for obj in objects:
                if obj.get("name") == TABLE_NAME:
                    obj["position"] = list(obj["position"])
                    obj["position"][2] = float(obj["position"][2]) + float(args.table_z_offset)
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
        root_pose = env.robot.get_position_orientation()
        q0 = env.robot.get_joint_positions().clone()
        qd0 = env.robot.get_joint_velocities().clone()
        env.enable_video_capture(True, clear=True)

        for idx, (pre_angle, grasp_angle) in enumerate(args.combos):
            _restore(env, root_pose, q0, qd0)
            pre_quat = _quat_normalize(_quat_for_x_angle(pre_angle))
            grasp_quat = _quat_normalize(_quat_for_x_angle(grasp_angle))
            if args.pregrasp_mode == "tcp_axis":
                tcp_axis_world = _quat_to_matrix_xyzw(pre_quat) @ np.asarray(api.get_tcp_offset_eef(arm=ARM), dtype=np.float64)
                tcp_axis_world = tcp_axis_world / max(float(np.linalg.norm(tcp_axis_world)), 1e-12)
                pre_target_pos = grasp_pos - tcp_axis_world * float(args.approach_distance)
            else:
                tcp_axis_world = None
                pre_target_pos = pre_pos
            label = f"pre_x_{pre_angle:+.1f}_grasp_x_{grasp_angle:+.1f}"
            print(f"[x2-orientation-combo] {idx + 1}/{len(args.combos)} {label}", flush=True)
            pre = _execute_tcp_target(
                api,
                label + "_pregrasp",
                (pre_target_pos, pre_quat),
                max_joint_step=args.max_joint_step,
                max_steps=240,
                settle_steps=args.settle_steps,
                hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                repeat_attempts=args.pregrasp_repeat_attempts or args.repeat_attempts,
                repeat_tcp_threshold=0.025,
                repeat_ori_threshold=0.45,
            )
            pre["object_contact"] = _object_contact_summary(env)
            approach_results: list[dict[str, Any]] = []
            final_insertion_results: list[dict[str, Any]] = []
            guard_violations: list[dict[str, Any]] = []
            proxy_guard_violations: list[dict[str, Any]] = []
            guarded_plan: dict[str, Any] | None = None
            if args.cartesian_approach:
                guarded_plan = api.plan_x2_guarded_grasp_approach(
                    (grasp_pos, grasp_quat),
                    grasp_pos,
                    object_size=OBJECT_SIZE,
                    arm=ARM,
                    approach_distance=float(args.approach_distance),
                    precontact_distance=float(args.precontact_distance),
                    num_waypoints=int(args.approach_waypoints),
                    guard_margin=float(args.guard_margin),
                    gripper_proxy_radius=float(args.gripper_proxy_radius),
                    gripper_proxy_margin=float(args.gripper_proxy_margin),
                    stop_at_first_proxy_collision=True,
                    final_insertion_waypoints=int(args.final_insertion_waypoints),
                )
                guard_violations = _jsonable(guarded_plan.get("guard_violations", []))
                proxy_guard_violations = _jsonable(guarded_plan.get("proxy_guard_violations", []))
                for waypoint_idx, waypoint in enumerate(guarded_plan.get("approach_waypoints", [])):
                    approach = _execute_tcp_target(
                        api,
                        f"{label}_approach_{waypoint_idx:02d}",
                        waypoint["tcp_pose"],
                        max_joint_step=args.max_joint_step,
                        max_steps=240,
                        settle_steps=args.settle_steps,
                        hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                        repeat_attempts=1,
                        repeat_tcp_threshold=0.025,
                        repeat_ori_threshold=0.35,
                    )
                    approach["object_contact"] = _object_contact_summary(env)
                    approach_results.append(_jsonable(approach))
                for waypoint_idx, waypoint in enumerate(guarded_plan.get("final_insertion_waypoints", [])):
                    insertion_label = label + "_grasp" if waypoint_idx == len(guarded_plan.get("final_insertion_waypoints", [])) - 1 else f"{label}_insert_{waypoint_idx:02d}"
                    insertion = _execute_tcp_target(
                        api,
                        insertion_label,
                        waypoint["tcp_pose"],
                        max_joint_step=args.max_joint_step,
                        max_steps=240,
                        settle_steps=args.settle_steps,
                        hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                        repeat_attempts=1 if insertion_label != label + "_grasp" else (args.grasp_repeat_attempts or args.repeat_attempts),
                        repeat_tcp_threshold=0.02,
                        repeat_ori_threshold=0.35,
                    )
                    insertion["object_contact"] = _object_contact_summary(env)
                    final_insertion_results.append(_jsonable(insertion))
            if final_insertion_results:
                grasp = final_insertion_results[-1]
            else:
                grasp = _execute_tcp_target(
                    api,
                    label + "_grasp",
                    (grasp_pos, grasp_quat),
                    command_tcp_pose=(grasp_pos + np.asarray(args.grasp_command_bias, dtype=np.float64), grasp_quat),
                    max_joint_step=args.max_joint_step,
                    max_steps=240,
                    settle_steps=args.settle_steps,
                    hold_steps_per_waypoint=args.hold_steps_per_waypoint,
                    repeat_attempts=args.grasp_repeat_attempts or args.repeat_attempts,
                    repeat_tcp_threshold=0.02,
                    repeat_ori_threshold=0.35,
                )
                grasp["object_contact"] = _object_contact_summary(env)
            result = {
                "label": label,
                "pre_angle_deg": pre_angle,
                "grasp_angle_deg": grasp_angle,
                "pre_quat_xyzw": np.round(pre_quat, 6).tolist(),
                "grasp_quat_xyzw": np.round(grasp_quat, 6).tolist(),
                "pregrasp_mode": args.pregrasp_mode,
                "approach_distance": float(args.approach_distance),
                "tcp_axis_world": None if tcp_axis_world is None else np.round(tcp_axis_world, 6).tolist(),
                "cartesian_approach": bool(args.cartesian_approach),
                "guarded_plan": _jsonable(guarded_plan),
                "approach_waypoints": approach_results,
                "final_insertion_waypoints": final_insertion_results,
                "guard_violations": guard_violations,
                "gripper_proxy_guard": bool(args.gripper_proxy_guard),
                "proxy_guard_violations": proxy_guard_violations,
                "pregrasp": _jsonable(pre),
                "grasp": _jsonable(grasp),
                "score": float(grasp["tcp_error_m"]) + 0.25 * float(pre["tcp_error_m"]),
            }
            summary["results"].append(result)

        ranked = sorted(summary["results"], key=lambda x: (float(x["grasp"]["tcp_error_m"]), float(x["pregrasp"]["tcp_error_m"])))
        summary["best"] = ranked[0] if ranked else None
        summary["ok"] = bool(ranked and float(ranked[0]["grasp"]["tcp_error_m"]) <= 0.02)
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
        "pregrasp_tcp_error_m": summary["best"]["pregrasp"]["tcp_error_m"],
        "grasp_tcp_error_m": summary["best"]["grasp"]["tcp_error_m"],
    }, "output_dir": str(output_dir)}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
