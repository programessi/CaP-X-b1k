"""Run the X2 visual grasp chain through the PyRoKi precontact executor.

This bridges the previously validated pieces:

1. Chest-camera OWL-ViT + SAM2 mask.
2. RGB-D object pose and Contact-GraspNet candidate generation.
3. X2 execution-plan conversion to a ``T_world_tcp`` grasp target.
4. PyRoKi collision-aware precontact planning plus slow TCP insertion.

The red cube is fixed, so the physical check is: reach the visual TCP target,
avoid object contact before close, then contact the object after closing.
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
import torch

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
    _as_numpy,
    _as_rgb_u8,
    _detect_and_segment,
    _jsonable,
    _mask_depth_hint,
    _save_depth_preview,
    _save_mask_overlay,
    _save_rgb,
    _scene_objects,
    _summarize_plan,
)
from x2_chest_visual_grasp_to_joint_ik_demo import _current_tcp, _pose_summary
from x2_code_exec_grasp_only_demo import GLOBAL_CAMERA, _write_videos
from x2_pyroki_precontact_insert_grasp_demo import (
    _execute_joint_trajectory_with_contacts,
    _object_contact_summary,
    _obstacles,
    _quat_for_x_angle,
    _target_error,
)
from x2_replay_visual_target_joint_tracking import _execute_tcp_target
from x2_replay_visual_target_orientation_sweep import _quat_to_matrix_xyzw

DEFAULT_GRASPNET_RAW_TO_X2_TCP_POS = np.array([0.003604, 0.002704, -0.059466], dtype=np.float64)
DEFAULT_GRASPNET_RAW_TO_X2_TCP_QUAT = np.array([0.488425, 0.670197, 0.342387, -0.441642], dtype=np.float64)


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated floats, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _parse_vec4(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Expected four comma-separated floats, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _candidate_pose(candidate: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = candidate[key]
    return np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(quat, dtype=np.float64).reshape(4)


def _quat_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64).reshape(4)
    qb = np.asarray(b, dtype=np.float64).reshape(4)
    qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
    qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (mat[2, 1] - mat[1, 2]) / scale,
                (mat[0, 2] - mat[2, 0]) / scale,
                (mat[1, 0] - mat[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        scale = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (mat[0, 1] + mat[1, 0]) / scale,
                (mat[0, 2] + mat[2, 0]) / scale,
                (mat[2, 1] - mat[1, 2]) / scale,
            ],
            dtype=np.float64,
        )
    elif mat[1, 1] > mat[2, 2]:
        scale = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        quat = np.array(
            [
                (mat[0, 1] + mat[1, 0]) / scale,
                0.25 * scale,
                (mat[1, 2] + mat[2, 1]) / scale,
                (mat[0, 2] - mat[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        quat = np.array(
            [
                (mat[0, 2] + mat[2, 0]) / scale,
                (mat[1, 2] + mat[2, 1]) / scale,
                0.25 * scale,
                (mat[1, 0] - mat[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _pose_to_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix_xyzw(quat_xyzw)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


def _matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return mat[:3, 3].copy(), _matrix_to_quat_xyzw(mat[:3, :3])


def _transform_pose(
    pose: tuple[np.ndarray, np.ndarray],
    right_transform: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pose_pos, pose_quat = pose
    tf_pos, tf_quat = right_transform
    return _matrix_to_pose(_pose_to_matrix(pose_pos, pose_quat) @ _pose_to_matrix(tf_pos, tf_quat))


def _sim_object_pose(env: X2BehaviourLowLevel, object_name: str = OBJECT_NAME) -> dict[str, Any]:
    obj = env.env.scene.object_registry("name", object_name)
    if obj is None:
        return {"ok": False, "error": f"object {object_name!r} not found"}
    pos, quat = obj.get_position_orientation()
    return {
        "ok": True,
        "object_name": object_name,
        "position_world": np.round(np.asarray(_as_numpy(pos), dtype=np.float64), 6).tolist(),
        "quat_xyzw_world": np.round(np.asarray(_as_numpy(quat), dtype=np.float64), 6).tolist(),
    }


def _select_candidate(
    candidates: list[dict[str, Any]],
    *,
    index: int,
    object_center: np.ndarray,
    max_xy_error: float,
    max_z_error: float,
) -> tuple[int, dict[str, Any]]:
    if not candidates:
        raise RuntimeError("x2 execution plan produced no candidates")
    obj = np.asarray(object_center, dtype=np.float64).reshape(3)
    checked: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        pos, _quat = _candidate_pose(candidate, "grasp_tcp_pose")
        xy_err = float(np.linalg.norm(pos[:2] - obj[:2]))
        z_err = float(abs(pos[2] - obj[2]))
        record = {
            "index": int(idx),
            "xy_error_m": xy_err,
            "z_error_m": z_err,
            "score": candidate.get("score"),
        }
        checked.append(record)
        if xy_err <= float(max_xy_error) and z_err <= float(max_z_error):
            if index <= 0:
                candidate["_selection_checked"] = checked
                return idx, candidate
            index -= 1
    best_idx = min(range(len(candidates)), key=lambda i: float(np.linalg.norm(_candidate_pose(candidates[i], "grasp_tcp_pose")[0] - obj)))
    candidates[best_idx]["_selection_checked"] = checked
    return best_idx, candidates[best_idx]


def _select_raw_graspnet_candidate(
    candidates: list[dict[str, Any]],
    *,
    index: int,
    object_center: np.ndarray,
    max_xy_error: float,
    max_z_above_object: float,
    reference_quat: np.ndarray,
    tcp_offset_eef: np.ndarray,
) -> tuple[int, dict[str, Any]]:
    raw_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("variant") == "raw_graspnet_quat" and candidate.get("raw_graspnet_pose") is not None
    ]
    if not raw_candidates:
        raise RuntimeError("grasp_plan produced no raw GraspNet orientation candidates")
    obj = np.asarray(object_center, dtype=np.float64).reshape(3)
    reference_quat = np.asarray(reference_quat, dtype=np.float64).reshape(4)
    ref_axis = _quat_to_matrix_xyzw(reference_quat) @ np.asarray(tcp_offset_eef, dtype=np.float64).reshape(3)
    ref_axis = ref_axis / max(float(np.linalg.norm(ref_axis)), 1e-12)
    checked: list[dict[str, Any]] = []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, candidate in enumerate(raw_candidates):
        pos, quat = _candidate_pose(candidate, "raw_graspnet_pose")
        xy_err = float(np.linalg.norm(pos[:2] - obj[:2]))
        z_above = float(pos[2] - obj[2])
        quat_err = _quat_error_rad(quat, reference_quat)
        axis = _quat_to_matrix_xyzw(quat) @ np.asarray(tcp_offset_eef, dtype=np.float64).reshape(3)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        axis_err = float(np.arccos(np.clip(float(np.dot(axis, ref_axis)), -1.0, 1.0)))
        pos_err = float(np.linalg.norm(pos - obj))
        outside_penalty = 0.0
        if xy_err > float(max_xy_error):
            outside_penalty += 10.0 * (xy_err - float(max_xy_error))
        if z_above < -0.02:
            outside_penalty += 10.0 * (-0.02 - z_above)
        if z_above > float(max_z_above_object):
            outside_penalty += 10.0 * (z_above - float(max_z_above_object))
        score = quat_err + 0.5 * axis_err + 2.0 * pos_err + outside_penalty
        record = {
            "raw_index": int(idx),
            "name": candidate.get("name"),
            "xy_error_m": xy_err,
            "z_above_object_m": z_above,
            "position_error_m": pos_err,
            "quat_error_to_validated_rad": quat_err,
            "axis_error_to_validated_rad": axis_err,
            "selection_score": score,
            "score": candidate.get("score"),
        }
        checked.append(record)
        scored.append((score, idx, candidate))
    scored.sort(key=lambda item: item[0])
    selected_score, selected_idx, selected = scored[min(max(0, int(index)), len(scored) - 1)]
    selected["_raw_selection_checked"] = checked
    selected["_raw_selection_ranked"] = sorted(checked, key=lambda item: float(item["selection_score"]))[:10]
    selected["_raw_selection_score"] = float(selected_score)
    selected["_raw_selection_reference_quat_xyzw"] = np.round(reference_quat, 6).tolist()
    selected["_raw_selection_reference_axis_world"] = np.round(ref_axis, 6).tolist()
    return selected_idx, selected


def _select_adapted_graspnet_candidate(
    candidates: list[dict[str, Any]],
    *,
    index: int,
    object_center: np.ndarray,
    max_xy_error: float,
    max_z_error: float,
    reference_quat: np.ndarray,
    adapter_pose: tuple[np.ndarray, np.ndarray],
    tcp_offset_eef: np.ndarray,
    position_weight: float,
    quat_weight: float,
    axis_weight: float,
    proxy_guard_fn: Any | None = None,
    proxy_collision_penalty: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    raw_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("variant") == "raw_graspnet_quat" and candidate.get("raw_graspnet_pose") is not None
    ]
    if not raw_candidates:
        raise RuntimeError("grasp_plan produced no raw GraspNet orientation candidates")
    obj = np.asarray(object_center, dtype=np.float64).reshape(3)
    reference_quat = np.asarray(reference_quat, dtype=np.float64).reshape(4)
    ref_axis = _quat_to_matrix_xyzw(reference_quat) @ np.asarray(tcp_offset_eef, dtype=np.float64).reshape(3)
    ref_axis = ref_axis / max(float(np.linalg.norm(ref_axis)), 1e-12)
    checked: list[dict[str, Any]] = []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, candidate in enumerate(raw_candidates):
        raw_pose = _candidate_pose(candidate, "raw_graspnet_pose")
        adapted_pose = _transform_pose(raw_pose, adapter_pose)
        pos, quat = adapted_pose
        xy_err = float(np.linalg.norm(pos[:2] - obj[:2]))
        z_err = float(abs(pos[2] - obj[2]))
        quat_err = _quat_error_rad(quat, reference_quat)
        axis = _quat_to_matrix_xyzw(quat) @ np.asarray(tcp_offset_eef, dtype=np.float64).reshape(3)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        axis_err = float(np.arccos(np.clip(float(np.dot(axis, ref_axis)), -1.0, 1.0)))
        pos_err = float(np.linalg.norm(pos - obj))
        proxy_guard: dict[str, Any] | None = None
        proxy_collision_count = 0
        if proxy_guard_fn is not None:
            proxy_guard = proxy_guard_fn(adapted_pose)
            proxy_collision_count = len(proxy_guard.get("proxy_guard_violations", []) or []) + len(
                proxy_guard.get("guard_violations", []) or []
            )
        outside_penalty = 0.0
        if xy_err > float(max_xy_error):
            outside_penalty += 10.0 * (xy_err - float(max_xy_error))
        if z_err > float(max_z_error):
            outside_penalty += 10.0 * (z_err - float(max_z_error))
        score = (
            float(quat_weight) * quat_err
            + float(axis_weight) * axis_err
            + float(position_weight) * pos_err
            + outside_penalty
            + float(proxy_collision_penalty) * float(proxy_collision_count)
        )
        record = {
            "raw_index": int(idx),
            "name": candidate.get("name"),
            "adapted_position": np.round(pos, 6).tolist(),
            "adapted_quat_xyzw": np.round(quat, 6).tolist(),
            "xy_error_m": xy_err,
            "z_error_m": z_err,
            "position_error_m": pos_err,
            "quat_error_to_validated_rad": quat_err,
            "axis_error_to_validated_rad": axis_err,
            "proxy_collision_count": int(proxy_collision_count),
            "selection_score": score,
            "selection_weights": {
                "position": float(position_weight),
                "quat": float(quat_weight),
                "axis": float(axis_weight),
                "proxy_collision": float(proxy_collision_penalty),
            },
            "proxy_guard": _jsonable(proxy_guard) if proxy_guard is not None else None,
            "score": candidate.get("score"),
        }
        checked.append(record)
        scored.append((score, idx, candidate))
    scored.sort(key=lambda item: item[0])
    selected_score, selected_idx, selected = scored[min(max(0, int(index)), len(scored) - 1)]
    selected["_adapted_grasp_tcp_pose"] = _transform_pose(_candidate_pose(selected, "raw_graspnet_pose"), adapter_pose)
    selected["_adapted_selection_checked"] = checked
    selected["_adapted_selection_ranked"] = sorted(checked, key=lambda item: float(item["selection_score"]))[:10]
    selected["_adapted_selection_score"] = float(selected_score)
    selected["_adapted_selection_reference_quat_xyzw"] = np.round(reference_quat, 6).tolist()
    selected["_adapted_selection_reference_axis_world"] = np.round(ref_axis, 6).tolist()
    selected["_graspnet_raw_to_x2_tcp_adapter"] = {
        "position": np.round(np.asarray(adapter_pose[0], dtype=np.float64), 6).tolist(),
        "quat_xyzw": np.round(np.asarray(adapter_pose[1], dtype=np.float64), 6).tolist(),
    }
    return selected_idx, selected


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 visual target + PyRoKi precontact insertion grasp demo")
    parser.add_argument("--config", default="x2_robotiq85_joint_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_visual_pyroki_precontact_insert_grasp_demo")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--owlvit-device", default="cpu")
    parser.add_argument("--owlvit-threshold", type=float, default=0.03)
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--graspnet-device", default="cuda")
    parser.add_argument("--graspnet-forward-passes", type=int, default=4)
    parser.add_argument("--graspnet-max-retries", type=int, default=30)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--object-center", type=_parse_vec3, default=OBJECT_CENTER)
    parser.add_argument("--table-center", type=_parse_vec3, default=TABLE_CENTER)
    parser.add_argument("--grasp-x-angle", type=float, default=90.0)
    parser.add_argument("--use-current-eef-orientation", action="store_true")
    parser.add_argument("--target-source", choices=["x2_plan", "raw_graspnet_pose", "adapted_graspnet_pose"], default="x2_plan")
    parser.add_argument("--max-raw-grasp-z-above-object", type=float, default=0.14)
    parser.add_argument("--graspnet-raw-to-x2-tcp-pos", type=_parse_vec3, default=DEFAULT_GRASPNET_RAW_TO_X2_TCP_POS)
    parser.add_argument("--graspnet-raw-to-x2-tcp-quat", type=_parse_vec4, default=DEFAULT_GRASPNET_RAW_TO_X2_TCP_QUAT)
    parser.add_argument("--adapted-selection-position-weight", type=float, default=12.0)
    parser.add_argument("--adapted-selection-quat-weight", type=float, default=1.0)
    parser.add_argument("--adapted-selection-axis-weight", type=float, default=0.5)
    parser.add_argument("--disable-adapted-selection-proxy-guard", action="store_true")
    parser.add_argument("--adapted-selection-proxy-collision-penalty", type=float, default=1.0)
    parser.add_argument("--adapted-selection-proxy-guard-margin", type=float, default=0.008)
    parser.add_argument("--adapted-selection-gripper-proxy-radius", type=float, default=0.012)
    parser.add_argument("--adapted-selection-gripper-proxy-margin", type=float, default=0.012)
    parser.add_argument("--dynamic-object", action="store_true")
    parser.add_argument("--lift-after-close", action="store_true")
    parser.add_argument("--lift-distance", type=float, default=0.06)
    parser.add_argument("--lift-object-threshold", type=float, default=0.015)
    parser.add_argument("--max-candidate-xy-error", type=float, default=0.08)
    parser.add_argument("--max-candidate-z-error", type=float, default=0.05)
    parser.add_argument("--precontact-distance", type=float, default=0.08)
    parser.add_argument("--cube-margin", type=float, default=0.012)
    parser.add_argument("--timesteps", type=int, default=18)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--insert-waypoints", type=int, default=10)
    parser.add_argument("--max-joint-step", type=float, default=0.022)
    parser.add_argument("--insert-max-joint-step", type=float, default=0.011)
    parser.add_argument("--hold-steps-per-waypoint", type=int, default=2)
    parser.add_argument("--insert-hold-steps-per-waypoint", type=int, default=5)
    parser.add_argument("--grasp-repeat-attempts", type=int, default=3)
    parser.add_argument("--grasp-repeat-tcp-threshold", type=float, default=0.015)
    parser.add_argument("--insert-repeat-tcp-threshold", type=float, default=0.018)
    parser.add_argument("--insert-repeat-ori-threshold", type=float, default=0.25)
    parser.add_argument("--settle-steps", type=int, default=16)
    parser.add_argument("--close-hold-steps", type=int, default=30)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = output_dir / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "mode": "visual_to_pyroki_precontact_insert",
        "object_name": OBJECT_NAME,
        "configured_object_center_world": np.round(np.asarray(args.object_center, dtype=np.float64), 6).tolist(),
        "configured_table_center_world": np.round(np.asarray(args.table_center, dtype=np.float64), 6).tolist(),
        "camera_contract": "chest camera RGB-D pose is T_world_camera; visual object pose is T_world_object",
        "action_contract": "selected visual candidate grasp_tcp_pose is T_world_tcp; PyRoKi converts it internally to T_world_eef",
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
                if args.dynamic_object:
                    obj["fixed_base"] = False
                    obj.pop("kinematic_only", None)
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
        object_pose_after_initial_settle = _sim_object_pose(env)

        chest_name = api.get_chest_camera_name()
        obs = api.get_chest_camera_observation()
        rgb = _as_rgb_u8(obs["rgb"])
        depth = obs.get("depth_linear", obs.get("depth"))
        if depth is None:
            raise RuntimeError("Chest camera observation has no depth/depth_linear")
        cam_pos, cam_quat = api.get_chest_camera_pose()
        K = api.get_chest_camera_intrinsics()
        _save_rgb(camera_dir / "chest_rgb.png", rgb)
        np.save(camera_dir / "chest_depth_linear.npy", _as_numpy(depth))
        _save_depth_preview(camera_dir / "chest_depth_linear.preview.png", depth)

        visual = _detect_and_segment(api, rgb, PROMPTS)
        _save_mask_overlay(
            camera_dir / "chest_sam2_mask_overlay.png",
            rgb,
            visual["detection"]["box"],
            visual["mask"],
            f"{visual['detection'].get('prompt', '')} {float(visual['detection'].get('score', 0.0)):.2f}",
        )
        expected_depth, depth_window = _mask_depth_hint(visual["mask"], depth)
        object_pos, object_quat, bbox_extent = api.get_object_pose(
            visual["detection"].get("prompt", OBJECT_NAME),
            return_bbox_extent=True,
            mask=visual["mask"],
            camera_name=chest_name,
            external=False,
            method="aabb_center",
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        object_pos = np.asarray(object_pos, dtype=np.float64)
        object_quat = np.asarray(object_quat, dtype=np.float64)
        bbox_extent = None if bbox_extent is None else np.asarray(bbox_extent, dtype=np.float64)
        configured_object_center = np.asarray(args.object_center, dtype=np.float64).reshape(3)
        visual_pose_error = float(np.linalg.norm(object_pos - configured_object_center))

        current_eef_quat = np.asarray(api.get_current_eef_pose(arm=ARM)[1], dtype=np.float64).reshape(4)
        if args.use_current_eef_orientation:
            execution_quat = current_eef_quat
            execution_quat_source = "current_eef_quat"
        else:
            execution_quat = _quat_for_x_angle(float(args.grasp_x_angle))
            execution_quat_source = f"validated_visual_candidate_x_{float(args.grasp_x_angle):.1f}"
        grasp_plan = api.sample_grasp_pose_graspnet(
            OBJECT_NAME,
            mask=visual["mask"],
            camera_name=chest_name,
            arm=ARM,
            external=False,
            orientation_quat_xyzw=execution_quat,
            include_simple_fallback=False,
            max_candidates=24,
            min_mask_pixels=12,
            expected_depth=expected_depth,
            depth_window=depth_window,
            workspace_bounds=WORKSPACE_BOUNDS,
            forward_passes=args.graspnet_forward_passes,
            max_retries=args.graspnet_max_retries,
        )
        x2_plan = api.plan_x2_grasp_execution(
            grasp_plan,
            object_pos,
            bbox_extent=bbox_extent,
            arm=ARM,
            orientation_quat_xyzw=execution_quat,
            workspace_bounds=WORKSPACE_BOUNDS,
            max_candidates=12,
        )
        summary.update(
            {
                "camera": {
                    "name": chest_name,
                    "position_world": np.round(np.asarray(cam_pos, dtype=np.float64), 6).tolist(),
                    "quat_xyzw_world": np.round(np.asarray(cam_quat, dtype=np.float64), 6).tolist(),
                    "intrinsic_matrix": np.round(K, 6).tolist(),
                },
                "artifacts": {
                    "rgb": str(camera_dir / "chest_rgb.png"),
                    "depth": str(camera_dir / "chest_depth_linear.npy"),
                    "depth_preview": str(camera_dir / "chest_depth_linear.preview.png"),
                    "mask_overlay": str(camera_dir / "chest_sam2_mask_overlay.png"),
                },
                "visual": {
                    "prompts": PROMPTS,
                    "detection": _jsonable(visual["detection"]),
                    "detection_count": len(visual["detections"]),
                    "mask_pixels": visual["mask_pixels"],
                    "expected_depth": expected_depth,
                    "depth_window": depth_window,
                },
                "pose_estimate": {
                    "meaning": "T_world_object estimated from SAM2 mask and chest RGB-D depth",
                    "position_world": np.round(object_pos, 6).tolist(),
                    "quat_xyzw_world": np.round(object_quat, 6).tolist(),
                    "bbox_extent": None if bbox_extent is None else np.round(bbox_extent, 6).tolist(),
                    "position_error_to_configured_center_m": round(visual_pose_error, 6),
                    "last_pose_estimate": _jsonable(getattr(api, "_last_object_pose_estimate", {})),
                },
                "graspnet": _summarize_plan(grasp_plan),
                "x2_execution_plan": {
                    "meaning": "candidate poses are T_world_tcp; selected grasp_tcp_pose is sent to PyRoKi precontact executor",
                    "ok": bool(x2_plan.get("ok", False)),
                    "strategy": x2_plan.get("strategy"),
                    "execution_quat_source": execution_quat_source,
                    "current_eef_quat_xyzw": np.round(current_eef_quat, 6).tolist(),
                    "execution_quat_xyzw": np.round(execution_quat, 6).tolist(),
                    "candidate_count": x2_plan.get("candidate_count"),
                    "input_candidate_count": x2_plan.get("input_candidate_count"),
                    "error": x2_plan.get("error"),
                    "filtered": _jsonable(x2_plan.get("filtered", [])[:20]),
                    "candidates": _jsonable(list(x2_plan.get("candidates", []) or [])[:5]),
                },
            }
        )
        x2_candidates = list(x2_plan.get("candidates", []) or [])
        adapter_pose = (
            np.asarray(args.graspnet_raw_to_x2_tcp_pos, dtype=np.float64).reshape(3),
            np.asarray(args.graspnet_raw_to_x2_tcp_quat, dtype=np.float64).reshape(4),
        )
        if args.target_source == "raw_graspnet_pose":
            raw_idx, raw_selected = _select_raw_graspnet_candidate(
                list(grasp_plan.get("candidates", []) or []),
                index=int(args.candidate_index),
                object_center=object_pos,
                max_xy_error=float(args.max_candidate_xy_error),
                max_z_above_object=float(args.max_raw_grasp_z_above_object),
                reference_quat=_quat_for_x_angle(float(args.grasp_x_angle)),
                tcp_offset_eef=np.asarray(api.get_tcp_offset_eef(arm=ARM), dtype=np.float64),
            )
            selected_idx = int(raw_idx)
            selected = raw_selected
            grasp_tcp_pose = _candidate_pose(selected, "raw_graspnet_pose")
            precontact_tcp_pose_from_candidate = selected.get("pregrasp_pose")
        elif args.target_source == "adapted_graspnet_pose":
            adapted_proxy_guard_fn = None
            if not bool(args.disable_adapted_selection_proxy_guard):
                first_insert_distance = float(args.precontact_distance) * (
                    float(max(0, int(args.insert_waypoints) - 1)) / float(max(1, int(args.insert_waypoints)))
                )

                def adapted_proxy_guard_fn(pose: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
                    return api.plan_x2_guarded_grasp_approach(
                        pose,
                        object_pos,
                        object_size=OBJECT_SIZE,
                        arm=ARM,
                        approach_distance=float(args.precontact_distance),
                        precontact_distance=first_insert_distance,
                        num_waypoints=2,
                        guard_margin=float(args.adapted_selection_proxy_guard_margin),
                        gripper_proxy_radius=float(args.adapted_selection_gripper_proxy_radius),
                        gripper_proxy_margin=float(args.adapted_selection_gripper_proxy_margin),
                        stop_at_first_proxy_collision=False,
                        final_insertion_waypoints=0,
                    )

            adapted_idx, adapted_selected = _select_adapted_graspnet_candidate(
                list(grasp_plan.get("candidates", []) or []),
                index=int(args.candidate_index),
                object_center=object_pos,
                max_xy_error=float(args.max_candidate_xy_error),
                max_z_error=float(args.max_candidate_z_error),
                reference_quat=_quat_for_x_angle(float(args.grasp_x_angle)),
                adapter_pose=adapter_pose,
                tcp_offset_eef=np.asarray(api.get_tcp_offset_eef(arm=ARM), dtype=np.float64),
                position_weight=float(args.adapted_selection_position_weight),
                quat_weight=float(args.adapted_selection_quat_weight),
                axis_weight=float(args.adapted_selection_axis_weight),
                proxy_guard_fn=adapted_proxy_guard_fn,
                proxy_collision_penalty=float(args.adapted_selection_proxy_collision_penalty),
            )
            selected_idx = int(adapted_idx)
            selected = adapted_selected
            grasp_tcp_pose = selected["_adapted_grasp_tcp_pose"]
            precontact_tcp_pose_from_candidate = None
        else:
            selected_idx, selected = _select_candidate(
                x2_candidates,
                index=int(args.candidate_index),
                object_center=object_pos,
                max_xy_error=float(args.max_candidate_xy_error),
                max_z_error=float(args.max_candidate_z_error),
            )
            grasp_tcp_pose = _candidate_pose(selected, "grasp_tcp_pose")
            precontact_tcp_pose_from_candidate = None
        grasp_pos, grasp_quat = grasp_tcp_pose

        tcp_axis_world = _quat_to_matrix_xyzw(grasp_quat) @ np.asarray(api.get_tcp_offset_eef(arm=ARM), dtype=np.float64)
        tcp_axis_world = tcp_axis_world / max(float(np.linalg.norm(tcp_axis_world)), 1e-12)
        if args.target_source == "raw_graspnet_pose" and precontact_tcp_pose_from_candidate is not None:
            precontact_tcp_pose = _candidate_pose({"pose": precontact_tcp_pose_from_candidate}, "pose")
            precontact_pos = precontact_tcp_pose[0]
        else:
            precontact_pos = grasp_pos - tcp_axis_world * float(args.precontact_distance)
            precontact_tcp_pose = (precontact_pos, grasp_quat)
        obstacles = _obstacles(configured_object_center, np.asarray(args.table_center), float(args.cube_margin))
        print(
            "[x2-visual-pyroki] selected target "
            f"visual_error={visual_pose_error:.4f}m "
            f"target_source={args.target_source} "
            f"quat_source={execution_quat_source} "
            f"tcp_axis_world={np.round(tcp_axis_world, 4).tolist()} "
            f"grasp_tcp={np.round(grasp_pos, 4).tolist()} "
            f"precontact_tcp={np.round(precontact_pos, 4).tolist()}",
            flush=True,
        )

        summary.update(
            {
                "camera": {
                    "name": chest_name,
                    "position_world": np.round(np.asarray(cam_pos, dtype=np.float64), 6).tolist(),
                    "quat_xyzw_world": np.round(np.asarray(cam_quat, dtype=np.float64), 6).tolist(),
                    "intrinsic_matrix": np.round(K, 6).tolist(),
                },
                "artifacts": {
                    "rgb": str(camera_dir / "chest_rgb.png"),
                    "depth": str(camera_dir / "chest_depth_linear.npy"),
                    "depth_preview": str(camera_dir / "chest_depth_linear.preview.png"),
                    "mask_overlay": str(camera_dir / "chest_sam2_mask_overlay.png"),
                },
                "visual": {
                    "prompts": PROMPTS,
                    "detection": _jsonable(visual["detection"]),
                    "detection_count": len(visual["detections"]),
                    "mask_pixels": visual["mask_pixels"],
                    "expected_depth": expected_depth,
                    "depth_window": depth_window,
                },
                "pose_estimate": {
                    "meaning": "T_world_object estimated from SAM2 mask and chest RGB-D depth",
                    "position_world": np.round(object_pos, 6).tolist(),
                    "quat_xyzw_world": np.round(object_quat, 6).tolist(),
                    "bbox_extent": None if bbox_extent is None else np.round(bbox_extent, 6).tolist(),
                    "position_error_to_configured_center_m": round(visual_pose_error, 6),
                    "last_pose_estimate": _jsonable(getattr(api, "_last_object_pose_estimate", {})),
                },
                "graspnet": _summarize_plan(grasp_plan),
                "x2_execution_plan": {
                    "meaning": "candidate poses are T_world_tcp; selected grasp_tcp_pose is sent to PyRoKi precontact executor",
                    "ok": bool(x2_plan.get("ok", False)),
                    "strategy": x2_plan.get("strategy"),
                    "execution_quat_source": execution_quat_source,
                    "current_eef_quat_xyzw": np.round(current_eef_quat, 6).tolist(),
                    "execution_quat_xyzw": np.round(execution_quat, 6).tolist(),
                    "candidate_count": x2_plan.get("candidate_count"),
                    "input_candidate_count": x2_plan.get("input_candidate_count"),
                    "error": x2_plan.get("error"),
                    "filtered": _jsonable(x2_plan.get("filtered", [])[:20]),
                    "candidates": _jsonable(x2_candidates[:5]),
                },
                "selected_target_source": args.target_source,
                "selected_candidate_index": int(selected_idx),
                "selected_candidate": _jsonable(selected),
                "grasp_tcp_pose": _pose_summary(grasp_tcp_pose),
                "precontact_tcp_pose": _pose_summary(precontact_tcp_pose),
                "tcp_axis_world": np.round(tcp_axis_world, 6).tolist(),
                "adapted_selection_weights": {
                    "position": float(args.adapted_selection_position_weight),
                    "quat": float(args.adapted_selection_quat_weight),
                    "axis": float(args.adapted_selection_axis_weight),
                },
                "obstacles_world": obstacles,
            }
        )
        summary["checks"].extend(
            [
                "pass" if visual["mask_pixels"] >= 50 else f"fail: mask too small {visual['mask_pixels']}",
                "pass" if visual_pose_error <= 0.08 else f"fail: pose error too large {visual_pose_error:.4f}m",
                "pass" if grasp_plan.get("ok") else f"fail: graspnet {grasp_plan.get('error')}",
                "pass" if x2_plan.get("ok") else f"fail: x2 plan {x2_plan.get('error')}",
            ]
        )

        start_tcp, start_eef = _current_tcp(api, arm=ARM)
        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)

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
        print(
            "[x2-visual-pyroki] precontact "
            f"joint_ok={precontact_motion.get('joint_tracking_ok')} "
            f"tcp_error={precontact_error['tcp_error_m']:.4f}m "
            f"ori_error={precontact_error['eef_ori_error_rad']:.4f}rad "
            f"contact={precontact_contact.get('current_contact_count')}",
            flush=True,
        )

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
                repeat_attempts=1 if label != "grasp" else int(args.grasp_repeat_attempts),
                repeat_tcp_threshold=float(args.insert_repeat_tcp_threshold)
                if label != "grasp"
                else float(args.grasp_repeat_tcp_threshold),
                repeat_ori_threshold=float(args.insert_repeat_ori_threshold),
            )
            result["object_contact"] = _object_contact_summary(env)
            insertion_results.append(_jsonable(result))
            print(
                "[x2-visual-pyroki] waypoint "
                f"{label} ok={result.get('ok')} "
                f"tcp_error={float(result.get('tcp_error_m', float('nan'))):.4f}m "
                f"ori_error={float(result.get('eef_ori_error_rad', float('nan'))):.4f}rad "
                f"joint_error={float(result.get('joint_final_error_rad', float('nan'))):.4f}rad "
                f"contact={(result.get('object_contact') or {}).get('current_contact_count')}",
                flush=True,
            )

        before_close_error = _target_error(api, grasp_tcp_pose)
        before_close_contact = _object_contact_summary(env)
        object_pose_before_close = _sim_object_pose(env)
        print(
            "[x2-visual-pyroki] before close "
            f"tcp_error={before_close_error['tcp_error_m']:.4f}m "
            f"ori_error={before_close_error['eef_ori_error_rad']:.4f}rad "
            f"contact={before_close_contact.get('current_contact_count')}",
            flush=True,
        )
        api.close_gripper(arm=ARM)
        api.settle_robot(steps=int(args.close_hold_steps))
        after_close_contact = _object_contact_summary(env)
        after_close_error = _target_error(api, grasp_tcp_pose)
        object_pose_after_close = _sim_object_pose(env)
        print(
            "[x2-visual-pyroki] after close "
            f"tcp_error={after_close_error['tcp_error_m']:.4f}m "
            f"contact={after_close_contact.get('current_contact_count')}",
            flush=True,
        )
        lift_result: dict[str, Any] | None = None
        object_pose_after_lift: dict[str, Any] | None = None
        object_lift_delta_z = 0.0
        if args.lift_after_close:
            lift_tcp_pose = (grasp_pos + np.array([0.0, 0.0, float(args.lift_distance)], dtype=np.float64), grasp_quat)
            lift_result = _execute_tcp_target(
                api,
                "post_close_lift",
                lift_tcp_pose,
                max_joint_step=float(args.insert_max_joint_step),
                max_steps=320,
                settle_steps=int(args.settle_steps),
                hold_steps_per_waypoint=int(args.insert_hold_steps_per_waypoint),
                repeat_attempts=1,
                repeat_tcp_threshold=0.025,
                repeat_ori_threshold=0.35,
            )
            object_pose_after_lift = _sim_object_pose(env)
            before_lift_pos = np.asarray((object_pose_after_close or {}).get("position_world", [np.nan, np.nan, np.nan]), dtype=np.float64)
            after_lift_pos = np.asarray((object_pose_after_lift or {}).get("position_world", [np.nan, np.nan, np.nan]), dtype=np.float64)
            if np.all(np.isfinite(before_lift_pos)) and np.all(np.isfinite(after_lift_pos)):
                object_lift_delta_z = float(after_lift_pos[2] - before_lift_pos[2])
            print(
                "[x2-visual-pyroki] post-close lift "
                f"ok={lift_result.get('ok')} "
                f"tcp_error={float(lift_result.get('tcp_error_m', float('nan'))):.4f}m "
                f"object_dz={object_lift_delta_z:.4f}m",
                flush=True,
            )
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=12)
        after_open_contact = _object_contact_summary(env)
        object_pose_after_open = _sim_object_pose(env)
        env._record_frame()

        insertion_contacts = [
            item
            for item in insertion_results
            if int(((item.get("object_contact") or {}).get("current_contact_count") or 0)) > 0
        ]
        precontact_clear = precontact_contact.get("current_contact_count", 1) == 0
        before_close_reached = float(before_close_error["tcp_error_m"]) <= 0.025
        before_close_clear = before_close_contact.get("current_contact_count", 1) == 0
        after_close_touched = after_close_contact.get("current_contact_count", 0) > 0
        object_lifted = bool(args.lift_after_close and object_lift_delta_z >= float(args.lift_object_threshold))
        insertion_clear = len(insertion_contacts) == 0
        summary.update(
            {
                "object_dynamic": bool(args.dynamic_object),
                "object_pose_after_initial_settle": object_pose_after_initial_settle,
                "start_tcp_position": np.round(start_tcp, 6).tolist(),
                "start_eef": _pose_summary(start_eef),
                "plan_debug": plan.get("debug"),
                "precontact_motion": precontact_motion,
                "precontact_error": precontact_error,
                "precontact_contact": precontact_contact,
                "insertion_results": insertion_results,
                "insertion_contact_events": insertion_contacts,
                "before_close_error": before_close_error,
                "before_close_contact": before_close_contact,
                "object_pose_before_close": object_pose_before_close,
                "after_close_error": after_close_error,
                "after_close_contact": after_close_contact,
                "object_pose_after_close": object_pose_after_close,
                "lift_after_close": {
                    "enabled": bool(args.lift_after_close),
                    "target_lift_distance_m": float(args.lift_distance),
                    "object_lift_threshold_m": float(args.lift_object_threshold),
                    "motion": _jsonable(lift_result),
                    "object_pose_after_lift": object_pose_after_lift,
                    "object_lift_delta_z_m": round(float(object_lift_delta_z), 6),
                    "object_lifted": bool(object_lifted),
                },
                "after_open_contact": after_open_contact,
                "object_pose_after_open": object_pose_after_open,
                "motion_debug": api.get_last_motion_debug(),
                "video": _write_videos(output_dir, env.get_video_frames(), fps=int(args.video_fps)),
            }
        )
        post_close_success = object_lifted if args.lift_after_close else after_close_touched
        post_close_failure = (
            f"fail: object lift delta {object_lift_delta_z:.4f}m < {float(args.lift_object_threshold):.4f}m"
            if args.lift_after_close
            else "fail: no object contact after close"
        )
        summary["checks"].extend(
            [
                "pass" if precontact_clear else "fail: contact at precontact",
                "pass" if insertion_clear else f"fail: insertion contact events {len(insertion_contacts)}",
                "pass" if before_close_clear else "fail: contact before close",
                "pass" if before_close_reached else f"fail: before-close tcp error {before_close_error['tcp_error_m']}",
                "pass" if post_close_success else post_close_failure,
            ]
        )
        summary["ok"] = all(str(item) == "pass" for item in summary["checks"])
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], file=sys.stderr, flush=True)
    finally:
        summary["elapsed_s"] = round(time.time() - start, 3)
        (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": summary.get("ok", False),
                "checks": summary.get("checks"),
                "visual_object_error_m": (summary.get("pose_estimate") or {}).get("position_error_to_configured_center_m"),
                "selected_grasp_tcp_pose": summary.get("grasp_tcp_pose"),
                "precontact_tcp_error_m": (summary.get("precontact_error") or {}).get("tcp_error_m"),
                "before_close_tcp_error_m": (summary.get("before_close_error") or {}).get("tcp_error_m"),
                "precontact_contact_count": (summary.get("precontact_contact") or {}).get("current_contact_count"),
                "before_close_contact_count": (summary.get("before_close_contact") or {}).get("current_contact_count"),
                "after_close_contact_count": (summary.get("after_close_contact") or {}).get("current_contact_count"),
                "object_lift_delta_z_m": ((summary.get("lift_after_close") or {}).get("object_lift_delta_z_m")),
                "object_lifted": ((summary.get("lift_after_close") or {}).get("object_lifted")),
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary.get("ok", False) else 1)


if __name__ == "__main__":
    main()
