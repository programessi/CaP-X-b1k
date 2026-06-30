"""Closed-loop smoke test for the packaged X2 visual grasp API.

This script still creates the test env and records video, but the visual grasp
pipeline and motion execution are deliberately routed through the packaged
``X2ControlApi`` methods:

1. ``plan_visual_grasp_tcp_pose``
2. ``execute_tcp_grasp_plan``

Use this to verify the API path after changing the wrapper, not as generated
CaP-X task code.
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
from x2_chest_camera_visual_chain_smoke import (
    ARM,
    OBJECT_CENTER,
    OBJECT_NAME,
    OBJECT_SIZE,
    PROMPTS,
    TABLE_CENTER,
    WORKSPACE_BOUNDS,
    _jsonable,
    _scene_objects,
)
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos
from x2_pyroki_precontact_insert_grasp_demo import _object_contact_summary, _obstacles, _quat_for_x_angle


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated floats, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _pose_summary(pose: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    pos, quat = pose
    return {
        "position": np.round(np.asarray(pos, dtype=np.float64).reshape(3), 6).tolist(),
        "quat_xyzw": np.round(np.asarray(quat, dtype=np.float64).reshape(4), 6).tolist(),
    }


def _strip_large_visual_fields(plan: dict[str, Any]) -> dict[str, Any]:
    compact = dict(plan)
    visual = dict(compact.get("visual", {}) or {})
    visual.pop("mask", None)
    mask_result = dict(visual.get("mask_result", {}) or {})
    mask_result.pop("mask", None)
    visual["mask_result"] = mask_result
    compact["visual"] = visual
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 packaged visual grasp API closed-loop smoke test")
    parser.add_argument("--config", default="x2_robotiq85_joint_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_visual_grasp_api_closed_loop_smoke")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--owlvit-device", default="cpu")
    parser.add_argument("--owlvit-threshold", type=float, default=0.03)
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--graspnet-device", default="cuda")
    parser.add_argument("--object-center", type=_parse_vec3, default=np.array([0.32, -0.04, 0.921], dtype=np.float64))
    parser.add_argument("--table-center", type=_parse_vec3, default=np.array([0.32, -0.04, 0.895], dtype=np.float64))
    parser.add_argument("--grasp-x-angle", type=float, default=90.0)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "ok": False,
        "mode": "x2_packaged_visual_grasp_api_closed_loop",
        "object_name": OBJECT_NAME,
        "configured_object_center_world": np.round(np.asarray(args.object_center, dtype=np.float64), 6).tolist(),
        "configured_table_center_world": np.round(np.asarray(args.table_center, dtype=np.float64), 6).tolist(),
        "script_contract": "env setup and video are test harness; visual planning and execution use packaged X2ControlApi methods",
        "checks": [],
        "errors": [],
    }
    start = time.time()
    env = None
    try:
        objects = _scene_objects()
        for obj in objects:
            if obj.get("name") == OBJECT_NAME:
                obj["position"] = np.asarray(args.object_center, dtype=np.float64).tolist()
            elif obj.get("name") == "x2_chest_table":
                obj["position"] = np.asarray(args.table_center, dtype=np.float64).tolist()

        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=objects,
            external_sensors=[GLOBAL_CAMERA],
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
            robot_camera_resolution=args.image_size,
            chest_camera=True,
            chest_camera_resolution=args.image_size,
            save_video=False,
        )
        api = X2ControlApi(
            env,
            use_vision_models=True,
            owlvit_device=args.owlvit_device,
            owlvit_threshold=args.owlvit_threshold,
            sam2_device=args.sam2_device,
            use_graspnet=True,
            graspnet_device=args.graspnet_device,
        )
        api.settle_robot(steps=8)
        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)

        reference_quat = _quat_for_x_angle(float(args.grasp_x_angle))
        plan = api.plan_visual_grasp_tcp_pose(
            OBJECT_NAME,
            prompts=PROMPTS,
            camera_name=api.get_chest_camera_name(),
            arm=ARM,
            external=False,
            orientation_quat_xyzw=reference_quat,
            object_pose_method="aabb_center",
            graspnet_forward_passes=4,
            graspnet_max_retries=30,
            graspnet_max_candidates=24,
            min_mask_pixels=12,
            workspace_bounds=WORKSPACE_BOUNDS,
            precontact_distance=0.08,
            insert_waypoints=10,
            candidate_index=0,
            proxy_guard_object_size=OBJECT_SIZE,
        )
        if not plan.get("ok"):
            raise RuntimeError(f"plan_visual_grasp_tcp_pose failed: {plan.get('error')}")

        object_pos = np.asarray((plan.get("pose_estimate") or {}).get("position_world"), dtype=np.float64).reshape(3)
        visual_object_error = float(np.linalg.norm(object_pos - np.asarray(args.object_center, dtype=np.float64).reshape(3)))
        obstacles = _obstacles(np.asarray(args.object_center, dtype=np.float64), np.asarray(args.table_center, dtype=np.float64), 0.012)
        before_execute_contact = _object_contact_summary(env)
        print(
            "[x2-api-smoke] selected target "
            f"visual_error={visual_object_error:.4f}m "
            f"grasp_tcp={np.round(np.asarray(plan['grasp_tcp_pose'][0]), 4).tolist()} "
            f"tcp_axis_world={np.round(np.asarray(plan['tcp_axis_world']), 4).tolist()}",
            flush=True,
        )

        execution = api.execute_tcp_grasp_plan(
            plan,
            arm=ARM,
            obstacles_world=obstacles,
            timesteps=18,
            dt=0.08,
            max_joint_step=0.022,
            insert_max_joint_step=0.011,
            settle_steps=16,
            hold_steps_per_waypoint=2,
            insert_hold_steps_per_waypoint=5,
            close_hold_steps=30,
            final_tcp_threshold=0.025,
            final_ori_threshold=0.25,
        )
        after_close_contact = _object_contact_summary(env)
        video = _write_videos(output_dir, env.get_video_frames(), fps=int(args.video_fps))
        before_close_error = execution.get("before_close_error") or {}
        print(
            "[x2-api-smoke] before close "
            f"tcp_error={float(before_close_error.get('tcp_error_m', float('nan'))):.4f}m "
            f"ori_error={float(before_close_error.get('ori_error_rad', float('nan'))):.4f}rad "
            f"after_close_contact={after_close_contact.get('current_contact_count')}",
            flush=True,
        )

        summary.update(
            {
                "ok": bool(execution.get("ok")) and int(after_close_contact.get("current_contact_count", 0)) > 0,
                "reference_quat_xyzw": np.round(reference_quat, 6).tolist(),
                "visual_object_error_m": round(visual_object_error, 6),
                "plan": _jsonable(_strip_large_visual_fields(plan)),
                "selected_grasp_tcp_pose": _pose_summary(plan.get("grasp_tcp_pose")),
                "selected_precontact_tcp_pose": _pose_summary(plan.get("precontact_tcp_pose")),
                "execution": _jsonable(execution),
                "before_execute_contact": before_execute_contact,
                "after_close_contact": after_close_contact,
                "video": video,
            }
        )
        summary["checks"].extend(
            [
                "pass" if visual_object_error <= 0.08 else f"fail: visual object error {visual_object_error:.4f}m",
                "pass" if plan.get("ok") else f"fail: plan {plan.get('error')}",
                "pass" if execution.get("before_close_reached") else f"fail: before close {before_close_error}",
                "pass" if int(after_close_contact.get("current_contact_count", 0)) > 0 else "fail: no contact after close",
            ]
        )
        summary["ok"] = all(item == "pass" for item in summary["checks"])
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], file=sys.stderr, flush=True)
    finally:
        summary["elapsed_s"] = round(time.time() - start, 3)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    print(
        json.dumps(
            _jsonable(
                {
                    "ok": summary.get("ok"),
                    "checks": summary.get("checks"),
                    "visual_object_error_m": summary.get("visual_object_error_m"),
                    "selected_grasp_tcp_pose": summary.get("selected_grasp_tcp_pose"),
                    "before_close_error": (summary.get("execution") or {}).get("before_close_error"),
                    "after_close_contact_count": (summary.get("after_close_contact") or {}).get("current_contact_count"),
                    "output_dir": str(output_dir),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
