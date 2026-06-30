"""X2 tabletop lunch-box smoke using real OWL-ViT + SAM2 visual outputs.

This is a small BEHAVIOR-style subtask of preparing_lunch_box: locate a
food-like tabletop object and a lunch-box target with OWL-ViT, segment them
with SAM2, convert the SAM2 masks through RGB-D geometry into world-frame
poses, then execute the X2 fixed-base action chain.  The scene setup stays in
this task smoke; X2ControlApi only supplies visual and motion primitives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.control import X2ControlApi


TASK_SOURCE = "preparing_lunch_box tabletop subtask"
FOOD_NAME = "lunch_food_red_cube"
BOX_NAME = "lunch_box_blue_target"
TABLE_NAME = "lunch_tabletop"
ARM = 1

FOOD_SIZE = 0.035
FOOD_CYLINDER_RADIUS = 0.018
FOOD_CYLINDER_HEIGHT = 0.075
BOX_EXTENT = np.array([0.055, 0.045, 0.018], dtype=np.float64)
STABLE_GRASP_EEF_POS = np.array([0.286943, -0.251023, 0.895185], dtype=np.float64)
STABLE_GRASP_EEF_QUAT = np.array([0.648934, 0.168974, 0.733815, 0.10885], dtype=np.float64)
OPEN_FINGER_CENTER_OFFSET_EEF = np.array([0.0, 0.0, 0.1046], dtype=np.float64)
CLOSED_FINGER_CENTER_OFFSET_EEF = np.array([0.0, 0.0, 0.1182], dtype=np.float64)


def _quat_xyzw_apply(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


FOOD_POS = STABLE_GRASP_EEF_POS + _quat_xyzw_apply(STABLE_GRASP_EEF_QUAT, OPEN_FINGER_CENTER_OFFSET_EEF)
SUPPORT_TOP_Z = float(FOOD_POS[2] - FOOD_SIZE / 2.0)
BOX_POS = np.array([FOOD_POS[0] + 0.055, FOOD_POS[1] + 0.035, SUPPORT_TOP_Z + BOX_EXTENT[2] / 2.0], dtype=np.float64)
TABLE_POS = np.array([FOOD_POS[0], FOOD_POS[1], SUPPORT_TOP_Z - 0.005], dtype=np.float64)
TABLE_SCALE = np.array([0.09, 0.055, 0.01], dtype=np.float64)

FOOD_PROMPTS = ["red cube", "red block", "red box", "cube", "food"]
FOOD_CYLINDER_PROMPTS = ["red cylinder", "cylinder", "red can", "can", "red pillar"]
BOX_PROMPTS = ["blue box", "box", "container"]
STUCK_PATIENCE_STEPS = 45
GRASPNET_WORKSPACE_BOUNDS = {"x": (0.16, 0.46), "y": (-0.46, -0.04), "z": (0.72, 1.08)}
GRASPNET_PREGRASP_DISTANCE = 0.08
GRASPNET_GRASP_OFFSET_M = 0.05
GRASPNET_FORWARD_PASSES = 4
GRASPNET_MAX_RETRIES = 30
X2_EXEC_PREGRASP_DISTANCE = 0.065
X2_EXEC_PREGRASP_LIFT = 0.025
X2_GRASPNET_CONTACT_BLEND = 0.25
MOTION_STEP_SCALE = 1.0

GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb", "depth_linear"],
    "sensor_kwargs": {"image_height": 384, "image_width": 384},
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_rgb_u8(value: Any) -> np.ndarray:
    arr = _as_numpy(value)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_list(value: Any) -> list[float]:
    return [round(float(v), 6) for v in _as_numpy(value).reshape(-1)]


def _log_stage(message: str) -> None:
    print(f"[x2-smoke] {message}", flush=True)


def _scaled_motion_steps(max_steps: int) -> int:
    return max(1, int(round(float(max_steps) * float(MOTION_STEP_SCALE))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _pose_summary(pose: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    pos, quat = pose
    return {
        "position": np.round(np.asarray(pos, dtype=np.float64), 6).tolist(),
        "quat_xyzw": np.round(np.asarray(quat, dtype=np.float64), 6).tolist(),
    }


def _summarize_grasp_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": candidate.get("name"),
        "source": candidate.get("source"),
        "variant": candidate.get("variant"),
        "rank": candidate.get("rank"),
        "grasp_index": candidate.get("grasp_index"),
        "score": None if candidate.get("score") is None else round(float(candidate.get("score", 0.0)), 6),
        "pregrasp_pose": _pose_summary(candidate.get("pregrasp_pose")),
        "grasp_pose": _pose_summary(candidate.get("grasp_pose")),
        "raw_graspnet_pose": _pose_summary(candidate.get("raw_graspnet_pose")),
        "approach_dir_world": None
        if candidate.get("approach_dir_world") is None
        else np.round(np.asarray(candidate["approach_dir_world"], dtype=np.float64), 6).tolist(),
        "contact_point_world": None
        if candidate.get("contact_point_world") is None
        else np.round(np.asarray(candidate["contact_point_world"], dtype=np.float64), 6).tolist(),
    }


def _summarize_grasp_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(plan.get("ok", False)),
        "source": plan.get("source"),
        "error": plan.get("error"),
        "fallback_used": plan.get("fallback_used"),
        "segmap_id": plan.get("segmap_id"),
        "mask_pixels": plan.get("mask_pixels"),
        "valid_mask_pixels": plan.get("valid_mask_pixels"),
        "depth_filter": _jsonable(plan.get("depth_filter")),
        "z_range": plan.get("z_range"),
        "raw_grasp_count": plan.get("raw_grasp_count"),
        "filtered_grasp_count": plan.get("filtered_grasp_count"),
        "candidate_count": plan.get("candidate_count"),
        "planner_kwargs": _jsonable(plan.get("planner_kwargs")),
        "camera_name": plan.get("camera_name"),
        "external": plan.get("external"),
        "depth_key": plan.get("depth_key"),
        "mask_source": _jsonable(plan.get("mask_source")),
        "candidates": [_summarize_grasp_candidate(candidate) for candidate in plan.get("candidates", [])],
    }


def _check(condition: bool, message: str) -> str:
    return "pass" if condition else f"fail: {message}"


def _camera_config(image_size: int) -> dict[str, Any]:
    camera = dict(GLOBAL_CAMERA)
    sensor_kwargs = dict(camera["sensor_kwargs"])
    sensor_kwargs["image_height"] = int(image_size)
    sensor_kwargs["image_width"] = int(image_size)
    camera["sensor_kwargs"] = sensor_kwargs
    return camera


def _clamp_box(box: list[float], shape: tuple[int, int]) -> list[float]:
    height, width = shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, x1 + 1, width - 1))
    y2 = float(np.clip(y2, y1 + 1, height - 1))
    return [x1, y1, x2, y2]


def _best_by_score(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return max(items, key=lambda item: float(item.get("score", 0.0)))


def _mask_depth_hint(mask: np.ndarray, depth: np.ndarray) -> tuple[float | None, float | None]:
    values = np.squeeze(depth)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None, None
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    depth_window = max(0.025, float(q75 - q25) * 1.5 + 0.01)
    return float(q50), depth_window


def _save_rgb(path: Path, value: Any) -> None:
    Image.fromarray(_as_rgb_u8(value)).save(path)


def _save_mask_overlay(path: Path, rgb: np.ndarray, box: list[float], mask: np.ndarray, label: str) -> None:
    base = _as_rgb_u8(rgb)
    overlay = base.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=3)
    draw.text((max(0, x1), max(0, y1 - 14)), label, fill=(0, 255, 255))
    image.save(path)


def _pose_position(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    try:
        if isinstance(pose, dict):
            return np.asarray(pose["position"], dtype=np.float64).reshape(3)
        return np.asarray(pose[0], dtype=np.float64).reshape(3)
    except Exception:
        return None


def _pose_quat(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    try:
        if isinstance(pose, dict):
            return np.asarray(pose["quat_xyzw"], dtype=np.float64).reshape(4)
        return np.asarray(pose[1], dtype=np.float64).reshape(4)
    except Exception:
        return None


def _draw_axes_3d(ax, position: np.ndarray, quat_xyzw: np.ndarray, *, scale: float, alpha: float = 0.8) -> None:
    R = x2_vision.quat_xyzw_to_matrix(quat_xyzw)
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for axis_idx, color in enumerate(colors):
        direction = R[:, axis_idx] * float(scale)
        ax.quiver(
            position[0],
            position[1],
            position[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            alpha=alpha,
            linewidth=1.2,
            arrow_length_ratio=0.25,
        )


def _draw_double_axis_3d(
    ax,
    position: np.ndarray,
    direction: np.ndarray,
    *,
    scale: float,
    color: str,
    alpha: float = 0.85,
    linewidth: float = 1.8,
) -> None:
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        return
    delta = direction / norm * float(scale)
    start = np.asarray(position, dtype=np.float64).reshape(3) - delta
    end = np.asarray(position, dtype=np.float64).reshape(3) + delta
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
    )


def _grasp_axis_metrics(candidate: dict[str, Any], world_long_axis: np.ndarray) -> dict[str, Any] | None:
    pose = candidate.get("raw_graspnet_pose") or candidate.get("grasp_pose")
    pos = _pose_position(pose)
    quat = _pose_quat(pose)
    if pos is None or quat is None:
        return None
    R = x2_vision.quat_xyzw_to_matrix(quat)
    long_axis = np.asarray(world_long_axis, dtype=np.float64).reshape(3)
    long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-8)
    approach = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    approach = approach / max(float(np.linalg.norm(approach)), 1e-8)
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    x_perp = 1.0 - abs(float(np.dot(x_axis, long_axis)))
    y_perp = 1.0 - abs(float(np.dot(y_axis, long_axis)))
    approach_side = 1.0 - abs(float(np.dot(approach, long_axis)))
    return {
        "position": pos,
        "quat_xyzw": quat,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "approach": approach,
        "x_perp_score": x_perp,
        "y_perp_score": y_perp,
        "approach_side_score": approach_side,
        "best_closing_axis": "X" if x_perp >= y_perp else "Y",
        "best_closing_score": max(x_perp, y_perp),
        "side_grasp_score": approach_side * max(x_perp, y_perp),
    }


def _set_axes_equal_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float64)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * float(np.max(limits[:, 1] - limits[:, 0]))
    radius = max(radius, 0.05)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _save_grasp_visualizations(
    output_dir: Path,
    grasp_plan: dict[str, Any],
    x2_execution_plan: dict[str, Any],
    object_position: np.ndarray,
    bbox_extent: np.ndarray | None,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    paths: dict[str, str] = {}
    object_position = np.asarray(object_position, dtype=np.float64).reshape(3)
    half_extent = (
        np.asarray(bbox_extent, dtype=np.float64).reshape(3) * 0.5
        if bbox_extent is not None
        else np.array([FOOD_SIZE, FOOD_SIZE, FOOD_SIZE], dtype=np.float64) * 0.5
    )
    raw_candidates = list(grasp_plan.get("candidates", []) or [])
    x2_candidates = list(x2_execution_plan.get("candidates", []) or [])

    topdown_path = output_dir / "graspnet_candidates_topdown.png"
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=160)
    rect = plt.Rectangle(
        (object_position[0] - half_extent[0], object_position[1] - half_extent[1]),
        2.0 * half_extent[0],
        2.0 * half_extent[1],
        facecolor="#ffdddd",
        edgecolor="#cc2222",
        linewidth=2,
        label="object bbox xy",
    )
    ax.add_patch(rect)
    ax.scatter([object_position[0]], [object_position[1]], c="#cc2222", s=45, marker="x", label="object center")

    for idx, candidate in enumerate(raw_candidates[:8]):
        grasp_pos = _pose_position(candidate.get("grasp_pose") or candidate.get("raw_graspnet_pose"))
        contact = candidate.get("contact_point_world")
        contact_pos = None if contact is None else np.asarray(contact, dtype=np.float64).reshape(3)
        approach = candidate.get("approach_dir_world")
        approach_dir = None if approach is None else np.asarray(approach, dtype=np.float64).reshape(3)
        if grasp_pos is not None:
            ax.scatter([grasp_pos[0]], [grasp_pos[1]], c="#1f77b4", s=28)
            ax.text(grasp_pos[0], grasp_pos[1], f"G{idx}", fontsize=8, color="#1f77b4")
        if contact_pos is not None:
            ax.scatter([contact_pos[0]], [contact_pos[1]], c="#ff7f0e", s=22, marker="o")
        if grasp_pos is not None and approach_dir is not None:
            xy = approach_dir[:2]
            norm = float(np.linalg.norm(xy))
            if norm > 1e-8:
                xy = xy / norm * 0.025
                ax.arrow(
                    grasp_pos[0],
                    grasp_pos[1],
                    xy[0],
                    xy[1],
                    head_width=0.004,
                    head_length=0.006,
                    fc="#1f77b4",
                    ec="#1f77b4",
                    alpha=0.6,
                    length_includes_head=True,
                )

    for idx, candidate in enumerate(x2_candidates[:8]):
        tcp_pose = candidate.get("grasp_tcp_pose")
        tcp_pos = _pose_position(tcp_pose)
        pre_pos = _pose_position(candidate.get("pregrasp_tcp_pose"))
        if tcp_pos is None:
            continue
        ax.scatter([tcp_pos[0]], [tcp_pos[1]], c="#2ca02c", s=28, marker="s")
        ax.text(tcp_pos[0], tcp_pos[1], f"X{idx}", fontsize=8, color="#2ca02c")
        if pre_pos is not None:
            ax.plot([pre_pos[0], tcp_pos[0]], [pre_pos[1], tcp_pos[1]], color="#2ca02c", alpha=0.35, linewidth=1.2)

    ax.set_title("GraspNet raw candidates and X2 topdown TCP targets")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(topdown_path)
    plt.close(fig)
    paths["topdown"] = str(topdown_path)

    fig = plt.figure(figsize=(8.0, 7.0), dpi=160)
    ax3 = fig.add_subplot(111, projection="3d")
    mins = object_position - half_extent
    maxs = object_position + half_extent
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )
    faces = [
        [corners[i] for i in face]
        for face in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))
    ]
    ax3.add_collection3d(Poly3DCollection(faces, facecolors="#ffdddd", edgecolors="#cc2222", alpha=0.25))
    ax3.scatter([object_position[0]], [object_position[1]], [object_position[2]], c="#cc2222", marker="x", s=45)

    for idx, candidate in enumerate(raw_candidates[:6]):
        pose = candidate.get("raw_graspnet_pose") or candidate.get("grasp_pose")
        grasp_pos = _pose_position(pose)
        grasp_quat = _pose_quat(pose)
        pre_pos = _pose_position(candidate.get("pregrasp_pose"))
        if grasp_pos is None:
            continue
        ax3.scatter([grasp_pos[0]], [grasp_pos[1]], [grasp_pos[2]], c="#1f77b4", s=25)
        ax3.text(grasp_pos[0], grasp_pos[1], grasp_pos[2], f"G{idx}", color="#1f77b4", fontsize=8)
        if pre_pos is not None:
            ax3.plot([pre_pos[0], grasp_pos[0]], [pre_pos[1], grasp_pos[1]], [pre_pos[2], grasp_pos[2]], color="#1f77b4", alpha=0.45)
        if grasp_quat is not None:
            _draw_axes_3d(ax3, grasp_pos, grasp_quat, scale=0.025, alpha=0.65)

    for idx, candidate in enumerate(x2_candidates[:6]):
        grasp_pos = _pose_position(candidate.get("grasp_tcp_pose"))
        pre_pos = _pose_position(candidate.get("pregrasp_tcp_pose"))
        lift_pos = _pose_position(candidate.get("lift_tcp_pose"))
        grasp_quat = _pose_quat(candidate.get("grasp_tcp_pose"))
        if grasp_pos is None:
            continue
        ax3.scatter([grasp_pos[0]], [grasp_pos[1]], [grasp_pos[2]], c="#2ca02c", s=28, marker="s")
        ax3.text(grasp_pos[0], grasp_pos[1], grasp_pos[2], f"X{idx}", color="#2ca02c", fontsize=8)
        if pre_pos is not None:
            ax3.plot([pre_pos[0], grasp_pos[0]], [pre_pos[1], grasp_pos[1]], [pre_pos[2], grasp_pos[2]], color="#2ca02c", alpha=0.55)
        if lift_pos is not None:
            ax3.plot([grasp_pos[0], lift_pos[0]], [grasp_pos[1], lift_pos[1]], [grasp_pos[2], lift_pos[2]], color="#2ca02c", linestyle="--", alpha=0.55)
        if grasp_quat is not None:
            _draw_axes_3d(ax3, grasp_pos, grasp_quat, scale=0.02, alpha=0.45)

    ax3.set_title("3D grasp poses: raw GraspNet (G) vs X2 TCP plan (X)")
    ax3.set_xlabel("world x (m)")
    ax3.set_ylabel("world y (m)")
    ax3.set_zlabel("world z (m)")
    _set_axes_equal_3d(ax3)
    fig.tight_layout()
    side_path = output_dir / "graspnet_candidates_3d.png"
    fig.savefig(side_path)
    plt.close(fig)
    paths["3d"] = str(side_path)

    world_long_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    metrics: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for idx, candidate in enumerate(raw_candidates[:8]):
        metric = _grasp_axis_metrics(candidate, world_long_axis)
        if metric is not None:
            metrics.append((idx, candidate, metric))

    if metrics:
        metrics_sorted = sorted(metrics, key=lambda item: float(item[2]["side_grasp_score"]), reverse=True)
        axis_path = output_dir / "graspnet_side_grasp_axes_3d.png"
        fig = plt.figure(figsize=(9.0, 7.4), dpi=160)
        axg = fig.add_subplot(111, projection="3d")
        axg.add_collection3d(Poly3DCollection(faces, facecolors="#ffdddd", edgecolors="#cc2222", alpha=0.20))
        axg.plot(
            [object_position[0], object_position[0]],
            [object_position[1], object_position[1]],
            [mins[2], maxs[2]],
            color="#cc2222",
            linewidth=3.0,
            alpha=0.65,
            label="object long axis",
        )
        for idx, candidate, metric in metrics_sorted[:6]:
            pos = metric["position"]
            approach = metric["approach"]
            axg.scatter([pos[0]], [pos[1]], [pos[2]], c="#1f77b4", s=30)
            label = (
                f"G{idx} {metric['best_closing_axis']} "
                f"side={metric['approach_side_score']:.2f} "
                f"close={metric['best_closing_score']:.2f}"
            )
            axg.text(pos[0], pos[1], pos[2], label, color="#1f77b4", fontsize=7)
            axg.quiver(
                pos[0],
                pos[1],
                pos[2],
                approach[0] * 0.045,
                approach[1] * 0.045,
                approach[2] * 0.045,
                color="#1f77b4",
                alpha=0.85,
                linewidth=2.0,
                arrow_length_ratio=0.25,
            )
            _draw_double_axis_3d(axg, pos, metric["x_axis"], scale=0.018, color="#ff7f0e", alpha=0.88, linewidth=2.2)
            _draw_double_axis_3d(axg, pos, metric["y_axis"], scale=0.018, color="#9467bd", alpha=0.88, linewidth=2.2)

        axg.plot([], [], color="#1f77b4", linewidth=2.0, label="approach (-Z)")
        axg.plot([], [], color="#ff7f0e", linewidth=2.2, label="candidate closing axis X")
        axg.plot([], [], color="#9467bd", linewidth=2.2, label="candidate closing axis Y")
        axg.set_title("Side-grasp diagnostic: approach and candidate closing axes")
        axg.set_xlabel("world x (m)")
        axg.set_ylabel("world y (m)")
        axg.set_zlabel("world z (m)")
        axg.legend(loc="best", fontsize=8)
        _set_axes_equal_3d(axg)
        fig.tight_layout()
        fig.savefig(axis_path)
        plt.close(fig)
        paths["side_grasp_axes_3d"] = str(axis_path)

        score_path = output_dir / "graspnet_side_grasp_scores.png"
        labels = [f"G{idx}" for idx, _candidate, _metric in metrics]
        x_scores = [float(metric["x_perp_score"]) for _idx, _candidate, metric in metrics]
        y_scores = [float(metric["y_perp_score"]) for _idx, _candidate, metric in metrics]
        approach_scores = [float(metric["approach_side_score"]) for _idx, _candidate, metric in metrics]
        side_scores = [float(metric["side_grasp_score"]) for _idx, _candidate, metric in metrics]
        x_positions = np.arange(len(labels))
        width = 0.20
        fig, axb = plt.subplots(figsize=(9.5, 4.8), dpi=160)
        axb.bar(x_positions - 1.5 * width, approach_scores, width, label="approach horizontal", color="#1f77b4")
        axb.bar(x_positions - 0.5 * width, x_scores, width, label="X perpendicular to long axis", color="#ff7f0e")
        axb.bar(x_positions + 0.5 * width, y_scores, width, label="Y perpendicular to long axis", color="#9467bd")
        axb.bar(x_positions + 1.5 * width, side_scores, width, label="combined side-grasp score", color="#2ca02c")
        axb.set_ylim(0.0, 1.05)
        axb.set_xticks(x_positions)
        axb.set_xticklabels(labels)
        axb.set_ylabel("score, higher is better")
        axb.set_title("Raw GraspNet candidates for vertical object side grasp")
        axb.grid(True, axis="y", alpha=0.25)
        axb.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(score_path)
        plt.close(fig)
        paths["side_grasp_scores"] = str(score_path)

    return paths


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


def _write_videos(output_dir: Path, frames_by_view: dict[str, list[np.ndarray]], fps: int) -> dict[str, Any]:
    result: dict[str, Any] = {"views": {}, "fps": fps}
    for view, frames in frames_by_view.items():
        result["views"][view] = _write_video(output_dir / f"{view}.mp4", frames, fps=fps)
    views = [view for view in ["global", "robot", "rgb"] if frames_by_view.get(view)]
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


def _run_owlvit_sam2(
    api: X2ControlApi,
    rgb: np.ndarray,
    prompts: list[str],
    *,
    min_mask_pixels: int,
) -> dict[str, Any]:
    all_detections: list[dict[str, Any]] = []
    for prompt in prompts:
        for det in api.detect_object_owlvit(rgb, prompt):
            det = dict(det)
            det["prompt"] = prompt
            det["box"] = _clamp_box(det["box"], rgb.shape[:2])
            all_detections.append(det)
    detection = _best_by_score(all_detections)
    if detection is None:
        raise ValueError(f"OWL-ViT returned no detections for prompts={prompts}")

    masks = api.segment_sam2(rgb, box=detection["box"], max_masks=3)
    mask_result = _best_by_score(masks)
    if mask_result is None:
        raise ValueError(f"SAM2 returned no masks for detection={detection}")
    mask = np.asarray(mask_result["mask"], dtype=bool)
    mask_pixels = int(mask.sum())
    if mask_pixels < min_mask_pixels:
        raise ValueError(f"SAM2 mask too small: {mask_pixels} pixels for detection={detection}")
    return {
        "prompts": prompts,
        "detections": all_detections,
        "detection": detection,
        "masks": masks,
        "mask_result": mask_result,
        "mask": mask,
        "mask_pixels": mask_pixels,
    }


def _estimate_pose_from_model_mask(
    api: X2ControlApi,
    query: str,
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    camera_name: str = "global_camera",
    method: str = "aabb_center",
) -> dict[str, Any]:
    expected_depth, depth_window = _mask_depth_hint(mask, depth)
    position, quat, extent = api.get_object_pose(
        query,
        return_bbox_extent=True,
        mask=mask,
        camera_name=camera_name,
        external=True,
        method=method,
        expected_depth=expected_depth,
        depth_window=depth_window,
    )
    return {
        "position": np.asarray(position, dtype=np.float64),
        "quat_xyzw": np.asarray(quat, dtype=np.float64),
        "bbox_extent": None if extent is None else np.asarray(extent, dtype=np.float64),
        "expected_depth": expected_depth,
        "depth_window": depth_window,
        "last_pose_estimate": getattr(api, "_last_object_pose_estimate", {}),
    }


def _move_tcp(
    api: X2ControlApi,
    tcp_position: np.ndarray,
    tcp_quat_xyzw: np.ndarray,
    *,
    arm: int = ARM,
    max_steps: int = 1200,
) -> dict[str, Any]:
    target_pose = api.tcp_pose_to_eef_pose((tcp_position, tcp_quat_xyzw), arm=arm)
    effective_max_steps = _scaled_motion_steps(max_steps)
    _log_stage(
        f"move_tcp start arm={arm} max_steps={effective_max_steps} "
        f"target_tcp={np.round(tcp_position, 4).tolist()}"
    )
    ok = api.move_hand(
        target_pose,
        arm=arm,
        pos_thresh=0.005,
        ori_thresh=0.1,
        stop_if_stuck=True,
        stuck_patience_steps=STUCK_PATIENCE_STEPS,
        max_steps=effective_max_steps,
    )
    reached_eef = api.get_current_eef_pose(arm=arm)
    reached_tcp = _as_numpy(reached_eef[0]) + x2_vision.quat_xyzw_to_matrix(_as_numpy(reached_eef[1])) @ api.get_tcp_offset_eef(arm=arm)
    tcp_error = float(np.linalg.norm(reached_tcp - tcp_position))
    _log_stage(f"move_tcp done ok={bool(ok)} tcp_error={tcp_error:.6f}m")
    return {
        "ok": bool(ok),
        "max_steps": effective_max_steps,
        "target_tcp_position": np.round(tcp_position, 6).tolist(),
        "target_tcp_quat_xyzw": np.round(tcp_quat_xyzw, 6).tolist(),
        "target_eef_position": np.round(target_pose[0], 6).tolist(),
        "target_eef_quat_xyzw": np.round(target_pose[1], 6).tolist(),
        "reached_eef": {"position": _as_list(reached_eef[0]), "quat_xyzw": _as_list(reached_eef[1])},
        "reached_tcp_position": np.round(reached_tcp, 6).tolist(),
        "tcp_target_error_m": round(tcp_error, 6),
    }


def _move_eef(
    api: X2ControlApi,
    eef_position: np.ndarray,
    eef_quat_xyzw: np.ndarray,
    *,
    arm: int = ARM,
    max_steps: int = 1200,
    pos_thresh: float = 0.005,
    ori_thresh: float = 0.1,
) -> dict[str, Any]:
    eef_position = np.asarray(eef_position, dtype=np.float64).reshape(3)
    eef_quat_xyzw = np.asarray(eef_quat_xyzw, dtype=np.float64).reshape(4)
    effective_max_steps = _scaled_motion_steps(max_steps)
    _log_stage(
        f"move_eef start arm={arm} max_steps={effective_max_steps} "
        f"target_eef={np.round(eef_position, 4).tolist()}"
    )
    ok = api.move_hand(
        (eef_position, eef_quat_xyzw),
        arm=arm,
        pos_thresh=pos_thresh,
        ori_thresh=ori_thresh,
        stop_if_stuck=True,
        stuck_patience_steps=STUCK_PATIENCE_STEPS,
        max_steps=effective_max_steps,
    )
    reached_eef = api.get_current_eef_pose(arm=arm)
    eef_error = float(np.linalg.norm(_as_numpy(reached_eef[0]) - eef_position))
    _log_stage(f"move_eef done ok={bool(ok)} eef_error={eef_error:.6f}m")
    return {
        "ok": bool(ok),
        "max_steps": effective_max_steps,
        "target_eef_position": np.round(eef_position, 6).tolist(),
        "target_eef_quat_xyzw": np.round(eef_quat_xyzw, 6).tolist(),
        "reached_eef": {"position": _as_list(reached_eef[0]), "quat_xyzw": _as_list(reached_eef[1])},
        "eef_target_error_m": round(eef_error, 6),
    }


def _object_state(env: X2BehaviourLowLevel, object_name: str) -> dict[str, Any]:
    obj = env.env.scene.object_registry("name", object_name)
    if obj is None:
        return {"found": False}
    pos, quat = obj.get_position_orientation()
    result = {
        "found": True,
        "position": _as_list(pos),
        "quat_xyzw": _as_list(quat),
    }
    for attr in ("aabb_center", "aabb_extent"):
        try:
            result[attr] = _as_list(getattr(obj, attr))
        except Exception:
            pass
    return result


def _check_object_in_hand(api: X2ControlApi, arm: int = ARM) -> bool:
    try:
        return bool(api.check_object_in_hand(arm=arm))
    except Exception:
        return False


def _candidate_pose(candidate: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    pose = candidate[key]
    pos, quat = pose
    return np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(quat, dtype=np.float64).reshape(4)


def _quat_from_x2_side_grasp_axes(approach_dir_world: np.ndarray) -> np.ndarray:
    approach = np.asarray(approach_dir_world, dtype=np.float64).reshape(3)
    approach[2] = 0.0
    approach_norm = float(np.linalg.norm(approach))
    if not np.isfinite(approach_norm) or approach_norm < 1e-8:
        approach = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        approach = approach / approach_norm

    z_axis = -approach
    y_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), approach)
    y_norm = float(np.linalg.norm(y_axis))
    if not np.isfinite(y_norm) or y_norm < 1e-8:
        y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-8)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return x2_vision.matrix_to_quat_xyzw(R)


def _select_side_grasp_candidate(grasp_plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scored: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for candidate in grasp_plan.get("candidates", []) or []:
        if candidate.get("variant") != "raw_graspnet_quat":
            continue
        metric = _grasp_axis_metrics(candidate, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if metric is None:
            continue
        score = float(metric["approach_side_score"]) * float(metric["x_perp_score"])
        scored.append((score, float(candidate.get("score", 0.0)), candidate, metric))
    if not scored:
        raise ValueError("No raw GraspNet candidates are available for single side-grasp attempt")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, _graspnet_score, candidate, metric = scored[0]
    metric = dict(metric)
    metric["x2_side_grasp_score"] = score
    return candidate, metric


def _run_single_side_grasp_once(
    env: X2BehaviourLowLevel,
    api: X2ControlApi,
    grasp_plan: dict[str, Any],
    food_position: np.ndarray,
    food_extent: np.ndarray | None,
    box_position: np.ndarray,
) -> dict[str, Any]:
    del food_extent
    _log_stage("single-side grasp: selecting GraspNet side candidate")
    selected, metric = _select_side_grasp_candidate(grasp_plan)
    object_initial = _object_state(env, FOOD_NAME)
    initial_eef = api.get_current_eef_pose(arm=ARM)
    approach = np.asarray(metric["approach"], dtype=np.float64).reshape(3)
    approach[2] = 0.0
    approach_norm = float(np.linalg.norm(approach))
    if not np.isfinite(approach_norm) or approach_norm < 1e-8:
        approach = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        approach = approach / approach_norm
    tcp_quat = _quat_from_x2_side_grasp_axes(approach)

    contact = selected.get("contact_point_world")
    contact_arr = None if contact is None else np.asarray(contact, dtype=np.float64).reshape(3)
    tcp_target = np.asarray(food_position, dtype=np.float64).reshape(3).copy()
    if contact_arr is not None and np.all(np.isfinite(contact_arr)):
        tcp_target[:2] = 0.85 * tcp_target[:2] + 0.15 * contact_arr[:2]
    tcp_target[2] = np.asarray(food_position, dtype=np.float64).reshape(3)[2]
    pregrasp_tcp = tcp_target - approach * 0.075
    pregrasp_tcp[2] = tcp_target[2]

    _log_stage("single-side grasp: open gripper and settle")
    api.open_gripper(arm=ARM)
    api.settle_robot(steps=8)
    _log_stage("single-side grasp: move to pregrasp TCP")
    pregrasp_move = _move_tcp(api, pregrasp_tcp, tcp_quat, arm=ARM, max_steps=700)
    _log_stage("single-side grasp: move to grasp TCP")
    grasp_move = _move_tcp(api, tcp_target, tcp_quat, arm=ARM, max_steps=650)
    before_close = _object_state(env, FOOD_NAME)
    _log_stage("single-side grasp: close gripper")
    api.close_gripper(arm=ARM)
    api.settle_robot(steps=18)
    after_close = _object_state(env, FOOD_NAME)
    in_hand_after_close = _check_object_in_hand(api, arm=ARM)

    current_eef = api.get_current_eef_pose(arm=ARM)
    place_delta = np.asarray(box_position, dtype=np.float64) - np.asarray(food_position, dtype=np.float64)
    place_delta[2] = 0.02
    place_eef = np.asarray(current_eef[0], dtype=np.float64) + place_delta
    _log_stage("single-side grasp: move closed gripper toward box target")
    place_move = _move_eef(
        api,
        place_eef,
        np.asarray(current_eef[1], dtype=np.float64),
        arm=ARM,
        max_steps=900,
        pos_thresh=0.012,
        ori_thresh=0.35,
    )
    api.settle_robot(steps=18)
    after_place = _object_state(env, FOOD_NAME)

    return {
        "mode": "single_x2_side_grasp_once_then_move_to_target",
        "object_initial": object_initial,
        "initial_eef": {"position": _as_list(initial_eef[0]), "quat_xyzw": _as_list(initial_eef[1])},
        "grasp_plan": _summarize_grasp_plan(grasp_plan),
        "selected_candidate": _summarize_grasp_candidate(selected),
        "selection_metric": _jsonable(metric),
        "x2_side_grasp_pose": {
            "approach_dir_world": np.round(approach, 6).tolist(),
            "tcp_target": _pose_summary((tcp_target, tcp_quat)),
            "pregrasp_tcp_target": _pose_summary((pregrasp_tcp, tcp_quat)),
            "strategy": "GraspNet side approach/contact + X2 horizontal closing axis",
            "local_minus_z_is_approach": True,
            "local_y_is_horizontal_closing_axis": True,
        },
        "pregrasp_move": pregrasp_move,
        "grasp_move": grasp_move,
        "before_close_object": before_close,
        "after_close_object": after_close,
        "in_hand_after_close": bool(in_hand_after_close),
        "place_move": place_move,
        "after_place_object": after_place,
        "release_near_box": False,
        "grasp_success": bool(in_hand_after_close),
        "success_attempt_name": selected.get("name"),
    }


def _x2_reachable_pregrasp(
    api: X2ControlApi,
    grasp_eef: np.ndarray,
    *,
    arm: int = ARM,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_eef = api.get_current_eef_pose(arm=arm)
    current_pos = np.asarray(current_eef[0], dtype=np.float64).reshape(3)
    grasp_eef = np.asarray(grasp_eef, dtype=np.float64).reshape(3)
    direction = grasp_eef - current_pos
    distance = float(np.linalg.norm(direction))
    if not np.isfinite(distance) or distance < 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        distance = 0.0
    else:
        direction = direction / distance
    retreat = min(float(X2_EXEC_PREGRASP_DISTANCE), max(0.025, distance * 0.45))
    pregrasp = grasp_eef - direction * retreat
    pregrasp[2] = max(pregrasp[2], grasp_eef[2] + float(X2_EXEC_PREGRASP_LIFT))
    return pregrasp, {
        "mode": "x2_current_to_grasp_line",
        "current_eef_position": np.round(current_pos, 6).tolist(),
        "direction_world": np.round(direction, 6).tolist(),
        "current_to_grasp_distance_m": round(distance, 6),
        "retreat_m": round(float(retreat), 6),
        "lift_m": round(float(X2_EXEC_PREGRASP_LIFT), 6),
    }


def _run_visual_grasp_sequence(
    env: X2BehaviourLowLevel,
    api: X2ControlApi,
    grasp_plan: dict[str, Any],
    food_position: np.ndarray,
    food_extent: np.ndarray | None,
    box_position: np.ndarray,
) -> dict[str, Any]:
    attempts = []
    success_attempt = None
    object_initial = _object_state(env, FOOD_NAME)
    initial_z = float(np.asarray(object_initial.get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)[2])
    initial_eef = api.get_current_eef_pose(arm=ARM)
    initial_quat = np.asarray(initial_eef[1], dtype=np.float64)
    execution_plan = api.plan_x2_grasp_execution(
        grasp_plan,
        food_position,
        bbox_extent=food_extent,
        arm=ARM,
        orientation_quat_xyzw=initial_quat,
        contact_blends=(0.0, X2_GRASPNET_CONTACT_BLEND, 0.5),
        z_biases=(0.0, 0.01, 0.02),
        pregrasp_lifts=(0.06, 0.08),
        lift_distance=0.06,
        max_candidates=12,
        workspace_bounds=GRASPNET_WORKSPACE_BOUNDS,
    )
    execution_items: list[dict[str, Any]] = list(execution_plan.get("candidates", []))

    for item in execution_items:
        candidate = item.get("graspnet_candidate", {})
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=6)

        pregrasp_eef, pregrasp_quat = _candidate_pose(item, "pregrasp_pose")
        grasp_eef, grasp_quat = _candidate_pose(item, "grasp_pose")
        planned_pregrasp_eef = pregrasp_eef.copy()
        pregrasp_adapter = {
            "mode": "x2_execution_plan_pregrasp",
            "pregrasp_lift": item.get("pregrasp_lift"),
            "descent_dir_world": _jsonable(item.get("descent_dir_world")),
        }

        pregrasp_move = _move_eef(api, pregrasp_eef, pregrasp_quat, arm=ARM, max_steps=550, ori_thresh=0.5)
        pregrasp_reached = bool(pregrasp_move["ok"] or pregrasp_move["eef_target_error_m"] < 0.035)
        grasp_move = (
            _move_eef(api, grasp_eef, grasp_quat, arm=ARM, max_steps=500, ori_thresh=0.5)
            if pregrasp_reached
            else None
        )
        grasp_reached = bool(
            grasp_move is not None and (grasp_move["ok"] or grasp_move["eef_target_error_m"] < 0.022)
        )

        before_close = _object_state(env, FOOD_NAME)
        if grasp_reached:
            api.close_gripper(arm=ARM)
            api.settle_robot(steps=12)
        after_close = _object_state(env, FOOD_NAME)
        in_hand_after_close = _check_object_in_hand(api, arm=ARM)

        lift_ok = False
        lift_move = None
        after_lift = after_close
        in_hand_after_lift = False
        lifted_delta = 0.0
        if grasp_reached:
            lift_eef, lift_quat = _candidate_pose(item, "lift_pose")
            lift_move = _move_eef(api, lift_eef, lift_quat, arm=ARM, max_steps=550, ori_thresh=0.5)
            lift_ok = bool(lift_move["ok"] or lift_move["eef_target_error_m"] < 0.018)
            api.settle_robot(steps=12)
            after_lift = _object_state(env, FOOD_NAME)
            in_hand_after_lift = _check_object_in_hand(api, arm=ARM)
            lifted_z = float(np.asarray(after_lift.get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)[2])
            before_z = float(np.asarray(before_close.get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)[2])
            lifted_delta = lifted_z - before_z

        attempt_result = {
            "name": item.get("name"),
            "strategy": item.get("strategy"),
            "candidate": _summarize_grasp_candidate(candidate),
            "execution_candidate": _jsonable(item),
            "tcp_target": _pose_summary(item.get("grasp_tcp_pose")),
            "pregrasp_tcp_target": _pose_summary(item.get("pregrasp_tcp_pose")),
            "lift_tcp_target": _pose_summary(item.get("lift_tcp_pose")),
            "tcp_target_quat_xyzw": None
            if item.get("grasp_tcp_pose") is None
            else np.round(np.asarray(item["grasp_tcp_pose"][1], dtype=np.float64), 6).tolist(),
            "contact_blend": item.get("contact_blend"),
            "z_bias": item.get("z_bias"),
            "pregrasp_lift": item.get("pregrasp_lift"),
            "planned_pregrasp_eef": np.round(planned_pregrasp_eef, 6).tolist(),
            "executed_pregrasp_eef": np.round(pregrasp_eef, 6).tolist(),
            "pregrasp_adapter": pregrasp_adapter,
            "pregrasp_move": pregrasp_move,
            "grasp_move": grasp_move,
            "grasp_reached": bool(grasp_reached),
            "before_close_object": before_close,
            "after_close_object": after_close,
            "after_lift_object": after_lift,
            "in_hand_after_close": bool(in_hand_after_close),
            "in_hand_after_lift": bool(in_hand_after_lift),
            "lift_ok": bool(lift_ok),
            "lift_move": lift_move,
            "lifted_delta_m": round(float(lifted_delta), 6),
        }
        attempts.append(attempt_result)
        if bool(in_hand_after_lift) or lifted_delta > 0.025:
            success_attempt = attempt_result
            break

    place_move = None
    after_release = _object_state(env, FOOD_NAME)
    release_ok = False
    if success_attempt is not None:
        current_eef = api.get_current_eef_pose(arm=ARM)
        place_delta = np.asarray(box_position, dtype=np.float64) - np.asarray(food_position, dtype=np.float64)
        place_delta[2] = 0.01
        place_eef = np.asarray(current_eef[0], dtype=np.float64) + place_delta
        place_move = _move_eef(
            api,
            place_eef,
            np.asarray(current_eef[1], dtype=np.float64),
            arm=ARM,
            max_steps=1200,
            pos_thresh=0.01,
            ori_thresh=0.25,
        )
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=24)
        after_release = _object_state(env, FOOD_NAME)
        final_pos = np.asarray(after_release.get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)
        release_ok = bool(np.all(np.isfinite(final_pos)) and np.linalg.norm(final_pos[:2] - np.asarray(box_position)[:2]) < 0.12)

    final_pos = np.asarray(after_release.get("position", [np.nan, np.nan, np.nan]), dtype=np.float64)
    return {
        "mode": "visual_graspnet_dynamic_grasp_place",
        "object_initial": object_initial,
        "initial_eef": {"position": _as_list(initial_eef[0]), "quat_xyzw": _as_list(initial_eef[1])},
        "grasp_plan": _summarize_grasp_plan(grasp_plan),
        "x2_execution_plan": _jsonable(execution_plan),
        "attempts": attempts,
        "success_attempt_name": None if success_attempt is None else success_attempt["name"],
        "grasp_success": success_attempt is not None,
        "place_move": place_move,
        "after_release_object": after_release,
        "object_lift_from_initial_m": None
        if not np.all(np.isfinite(final_pos)) or not np.isfinite(initial_z)
        else round(float(final_pos[2] - initial_z), 6),
        "release_near_box": bool(release_ok),
    }


def _food_extent_config(food_primitive: str, food_radius: float, food_height: float) -> np.ndarray:
    if food_primitive == "cylinder":
        return np.array([2.0 * food_radius, 2.0 * food_radius, food_height], dtype=np.float64)
    return np.array([FOOD_SIZE, FOOD_SIZE, FOOD_SIZE], dtype=np.float64)


def _food_prompts(food_primitive: str) -> list[str]:
    if food_primitive == "cylinder":
        return FOOD_CYLINDER_PROMPTS
    return FOOD_PROMPTS


def _scene_layout(food_primitive: str, food_radius: float, food_height: float) -> dict[str, np.ndarray | float]:
    food_extent = _food_extent_config(food_primitive, food_radius, food_height)
    support_top_z = float(FOOD_POS[2] - food_extent[2] / 2.0)
    box_pos = np.array(
        [FOOD_POS[0] + 0.055, FOOD_POS[1] + 0.035, support_top_z + BOX_EXTENT[2] / 2.0],
        dtype=np.float64,
    )
    table_pos = np.array([FOOD_POS[0], FOOD_POS[1], support_top_z - 0.005], dtype=np.float64)
    return {
        "food_position": FOOD_POS.copy(),
        "food_extent": food_extent,
        "support_top_z": support_top_z,
        "box_position": box_pos,
        "table_position": table_pos,
    }


def _scene_objects(food_primitive: str, food_radius: float, food_height: float) -> list[dict[str, Any]]:
    layout = _scene_layout(food_primitive, food_radius, food_height)
    if food_primitive == "cylinder":
        food_object = {
            "type": "PrimitiveObject",
            "name": FOOD_NAME,
            "primitive_type": "Cylinder",
            "radius": food_radius,
            "height": food_height,
            "position": layout["food_position"].tolist(),
            "orientation": [0, 0, 0, 1],
            "fixed_base": False,
            "rgba": [1.0, 0.04, 0.03, 1.0],
        }
    else:
        food_object = {
            "type": "PrimitiveObject",
            "name": FOOD_NAME,
            "primitive_type": "Cube",
            "size": FOOD_SIZE,
            "position": layout["food_position"].tolist(),
            "orientation": [0, 0, 0, 1],
            "fixed_base": False,
            "rgba": [1.0, 0.04, 0.03, 1.0],
        }
    return [
        {
            "type": "PrimitiveObject",
            "name": TABLE_NAME,
            "primitive_type": "Cube",
            "size": 1.0,
            "scale": TABLE_SCALE.tolist(),
            "position": layout["table_position"].tolist(),
            "orientation": [0, 0, 0, 1],
            "fixed_base": True,
            "rgba": [0.42, 0.42, 0.42, 1.0],
        },
        food_object,
        {
            "type": "PrimitiveObject",
            "name": BOX_NAME,
            "primitive_type": "Cube",
            "size": 1.0,
            "scale": BOX_EXTENT.tolist(),
            "position": layout["box_position"].tolist(),
            "orientation": [0, 0, 0, 1],
            "fixed_base": True,
            "kinematic_only": True,
            "rgba": [0.02, 0.20, 1.0, 1.0],
        },
    ]


def main() -> int:
    global MOTION_STEP_SCALE

    parser = argparse.ArgumentParser(description="X2 preparing_lunch_box tabletop SAM2 smoke")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_preparing_lunch_box_sam2_grasp")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--pre-hold-steps", type=int, default=24)
    parser.add_argument("--post-hold-steps", type=int, default=36)
    parser.add_argument("--owlvit-device", default="cpu")
    parser.add_argument("--owlvit-threshold", type=float, default=0.03)
    parser.add_argument("--sam2-device", default="cuda")
    parser.add_argument("--graspnet-device", default="cuda")
    parser.add_argument("--max-grasp-candidates", type=int, default=4)
    parser.add_argument("--graspnet-pregrasp-distance", type=float, default=GRASPNET_PREGRASP_DISTANCE)
    parser.add_argument("--graspnet-grasp-offset", type=float, default=GRASPNET_GRASP_OFFSET_M)
    parser.add_argument("--food-primitive", choices=["cube", "cylinder"], default="cube")
    parser.add_argument("--food-radius", type=float, default=FOOD_CYLINDER_RADIUS)
    parser.add_argument("--food-height", type=float, default=FOOD_CYLINDER_HEIGHT)
    parser.add_argument(
        "--motion-step-scale",
        type=float,
        default=1.0,
        help="Scale motion primitive max_steps for bounded diagnostic runs; 1.0 preserves default behavior.",
    )
    parser.add_argument("--visualize-only", action="store_true", help="Stop after visual/GraspNet/X2 grasp-plan visualization")
    parser.add_argument(
        "--single-side-grasp-once",
        action="store_true",
        help="Execute one selected vertical-object side grasp, close, then move to target regardless of success",
    )
    args = parser.parse_args()
    MOTION_STEP_SCALE = max(0.01, float(args.motion_step_scale))

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    camera_dir = output_dir / "cameras"
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)
    food_prompts = _food_prompts(args.food_primitive)
    scene_layout = _scene_layout(args.food_primitive, args.food_radius, args.food_height)
    food_extent_config = np.asarray(scene_layout["food_extent"], dtype=np.float64)
    support_top_z = float(scene_layout["support_top_z"])
    food_description = (
        f"dynamic red cylinder radius={args.food_radius:.3f} height={args.food_height:.3f}"
        if args.food_primitive == "cylinder"
        else "dynamic red cube"
    )

    summary: dict[str, Any] = {
        "ok": False,
        "task_source": TASK_SOURCE,
        "model_output_source": "OWL-ViT boxes + SAM2 masks + Contact-GraspNet grasp candidates",
        "food_name": FOOD_NAME,
        "box_name": BOX_NAME,
        "food_prompts": food_prompts,
        "box_prompts": BOX_PROMPTS,
        "global_camera": _camera_config(args.image_size),
        "scene": {
            "food_position_config": np.asarray(scene_layout["food_position"]).round(6).tolist(),
            "food_extent_config": food_extent_config.round(6).tolist(),
            "box_position_config": np.asarray(scene_layout["box_position"]).round(6).tolist(),
            "support_top_z": round(support_top_z, 6),
            "table_scale": TABLE_SCALE.round(6).tolist(),
            "food_primitive": food_description,
            "food_size": FOOD_SIZE,
            "food_radius": args.food_radius if args.food_primitive == "cylinder" else None,
            "food_height": args.food_height if args.food_primitive == "cylinder" else None,
            "food_fixed_base": False,
            "reachable_scene_seed_eef_pos": STABLE_GRASP_EEF_POS.round(6).tolist(),
            "reachable_scene_seed_eef_quat": STABLE_GRASP_EEF_QUAT.round(6).tolist(),
            "open_finger_center_offset_eef": OPEN_FINGER_CENTER_OFFSET_EEF.round(6).tolist(),
            "grasp_pose_source": "Contact-GraspNet from SAM2 mask/depth",
            "graspnet_workspace_bounds": {axis: list(bounds) for axis, bounds in GRASPNET_WORKSPACE_BOUNDS.items()},
            "graspnet_pregrasp_distance": args.graspnet_pregrasp_distance,
            "graspnet_grasp_offset": args.graspnet_grasp_offset,
            "graspnet_forward_passes": GRASPNET_FORWARD_PASSES,
            "graspnet_max_retries": GRASPNET_MAX_RETRIES,
            "x2_graspnet_contact_blend": X2_GRASPNET_CONTACT_BLEND,
        },
        "steps": {},
        "checks": [],
        "video": {},
        "errors": [],
    }

    try:
        _log_stage("create OmniGibson X2 environment")
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=_scene_objects(args.food_primitive, args.food_radius, args.food_height),
            external_sensors=[_camera_config(args.image_size)],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=args.image_size,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
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
        _log_stage("reset environment and settle")
        env.reset()
        env.enable_video_capture(True, clear=True)
        api.open_gripper(arm=ARM)
        api.settle_robot(steps=args.pre_hold_steps)

        _log_stage("capture global RGB-D")
        initial_eef = api.get_current_eef_pose(arm=ARM)
        initial_quat = np.asarray(initial_eef[1], dtype=np.float64)
        camera_obs = api.get_external_camera_observation("global_camera")
        rgb = _as_rgb_u8(camera_obs["rgb"])
        depth = np.squeeze(_as_numpy(camera_obs["depth_linear"]))
        _save_rgb(camera_dir / "global_camera_rgb.png", rgb)
        np.save(camera_dir / "global_camera_depth_linear.npy", depth)

        _log_stage("run OWL-ViT + SAM2 for food")
        food_visual = _run_owlvit_sam2(api, rgb, food_prompts, min_mask_pixels=12)
        _log_stage("run OWL-ViT + SAM2 for box")
        box_visual = _run_owlvit_sam2(api, rgb, BOX_PROMPTS, min_mask_pixels=20)
        np.save(camera_dir / "food_sam2_mask.npy", food_visual["mask"])
        np.save(camera_dir / "box_sam2_mask.npy", box_visual["mask"])
        _save_mask_overlay(
            camera_dir / "food_owlvit_sam2_overlay.png",
            rgb,
            food_visual["detection"]["box"],
            food_visual["mask"],
            f"{food_visual['detection']['prompt']} {food_visual['detection']['score']:.2f}",
        )
        _save_mask_overlay(
            camera_dir / "box_owlvit_sam2_overlay.png",
            rgb,
            box_visual["detection"]["box"],
            box_visual["mask"],
            f"{box_visual['detection']['prompt']} {box_visual['detection']['score']:.2f}",
        )

        _log_stage("estimate food pose from SAM2 mask + depth")
        food_pose = _estimate_pose_from_model_mask(
            api,
            food_visual["detection"]["prompt"],
            food_visual["mask"],
            depth,
        )
        _log_stage("estimate box pose from SAM2 mask + depth")
        box_pose = _estimate_pose_from_model_mask(
            api,
            box_visual["detection"]["prompt"],
            box_visual["mask"],
            depth,
        )
        food_position = np.asarray(food_pose["position"], dtype=np.float64)
        box_position = np.asarray(box_pose["position"], dtype=np.float64)

        _log_stage("run Contact-GraspNet grasp sampling")
        grasp_plan = api.sample_grasp_pose_graspnet(
            FOOD_NAME,
            mask=food_visual["mask"],
            camera_name="global_camera",
            arm=ARM,
            external=True,
            orientation_quat_xyzw=initial_quat,
            include_simple_fallback=False,
            max_candidates=args.max_grasp_candidates,
            pregrasp_distance=args.graspnet_pregrasp_distance,
            grasp_offset_m=args.graspnet_grasp_offset,
            min_mask_pixels=12,
            expected_depth=food_pose["expected_depth"],
            depth_window=food_pose["depth_window"],
            workspace_bounds=GRASPNET_WORKSPACE_BOUNDS,
            forward_passes=GRASPNET_FORWARD_PASSES,
            max_retries=GRASPNET_MAX_RETRIES,
        )

        _log_stage("plan X2 execution candidates")
        x2_execution_plan_preview = api.plan_x2_grasp_execution(
            grasp_plan,
            food_position,
            bbox_extent=food_pose["bbox_extent"],
            arm=ARM,
            orientation_quat_xyzw=initial_quat,
            contact_blends=(0.0, X2_GRASPNET_CONTACT_BLEND, 0.5),
            z_biases=(0.0, 0.01, 0.02),
            pregrasp_lifts=(0.06, 0.08),
            lift_distance=0.06,
            max_candidates=12,
            workspace_bounds=GRASPNET_WORKSPACE_BOUNDS,
        )
        _log_stage("write GraspNet/X2 grasp visualizations")
        grasp_visualization_paths = _save_grasp_visualizations(
            output_dir,
            grasp_plan,
            x2_execution_plan_preview,
            food_position,
            food_pose["bbox_extent"],
        )

        if args.visualize_only:
            action_chain = {
                "mode": "visualize_only",
                "skipped": True,
                "reason": "requested --visualize-only; stopped before executing grasp motions",
                "grasp_plan": _summarize_grasp_plan(grasp_plan),
                "x2_execution_plan": _jsonable(x2_execution_plan_preview),
                "grasp_success": False,
                "success_attempt_name": None,
                "place_move": None,
                "release_near_box": False,
            }
        elif args.single_side_grasp_once:
            _log_stage("execute single-side grasp action chain")
            action_chain = _run_single_side_grasp_once(
                env,
                api,
                grasp_plan,
                food_position,
                food_pose["bbox_extent"],
                box_position,
            )
            api.settle_robot(steps=args.post_hold_steps)
        else:
            _log_stage("execute visual grasp action chain")
            action_chain = _run_visual_grasp_sequence(
                env,
                api,
                grasp_plan,
                food_position,
                food_pose["bbox_extent"],
                box_position,
            )
            api.settle_robot(steps=args.post_hold_steps)

        summary["steps"]["vision_model_outputs"] = {
            "food": {
                "detections": _jsonable(food_visual["detections"]),
                "selected_detection": _jsonable(food_visual["detection"]),
                "selected_mask_score": float(food_visual["mask_result"].get("score", 0.0)),
                "mask_pixels": food_visual["mask_pixels"],
                "pose": {
                    "position": np.round(food_position, 6).tolist(),
                    "quat_xyzw": np.round(food_pose["quat_xyzw"], 6).tolist(),
                    "bbox_extent": None if food_pose["bbox_extent"] is None else np.round(food_pose["bbox_extent"], 6).tolist(),
                    "expected_depth": food_pose["expected_depth"],
                    "depth_window": food_pose["depth_window"],
                    "last_pose_estimate": _jsonable(food_pose["last_pose_estimate"]),
                },
                "graspnet": _summarize_grasp_plan(grasp_plan),
                "x2_execution_plan_preview": _jsonable(x2_execution_plan_preview),
                "grasp_visualizations": grasp_visualization_paths,
            },
            "box": {
                "detections": _jsonable(box_visual["detections"]),
                "selected_detection": _jsonable(box_visual["detection"]),
                "selected_mask_score": float(box_visual["mask_result"].get("score", 0.0)),
                "mask_pixels": box_visual["mask_pixels"],
                "pose": {
                    "position": np.round(box_position, 6).tolist(),
                    "quat_xyzw": np.round(box_pose["quat_xyzw"], 6).tolist(),
                    "bbox_extent": None if box_pose["bbox_extent"] is None else np.round(box_pose["bbox_extent"], 6).tolist(),
                    "expected_depth": box_pose["expected_depth"],
                    "depth_window": box_pose["depth_window"],
                    "last_pose_estimate": _jsonable(box_pose["last_pose_estimate"]),
                },
            },
            "saved_files": sorted(str(path) for path in camera_dir.iterdir()),
        }
        action_chain["initial_eef"] = {"position": _as_list(initial_eef[0]), "quat_xyzw": _as_list(initial_eef[1])}
        action_chain["final_gripper_state"] = api.get_gripper_state(arm=ARM)
        summary["steps"]["action_chain"] = action_chain

        if args.visualize_only:
            summary["video"] = {"skipped": True, "reason": "--visualize-only"}
            summary["video_sources"] = {}
        else:
            _log_stage("write captured videos")
            summary["video"] = _write_videos(output_dir, env.get_video_frames(), fps=args.video_fps)
            summary["video_sources"] = getattr(env, "_last_video_sources", {})

        base_checks = [
            _check(len(food_visual["detections"]) > 0, "OWL-ViT did not detect the food target"),
            _check(food_visual["mask_pixels"] > 12, "SAM2 food mask is empty or too small"),
            _check(np.asarray(food_position).shape == (3,), "food pose position is not 3D"),
            _check(bool(grasp_plan.get("ok")), f"Contact-GraspNet did not produce executable candidates: {grasp_plan.get('error')}"),
            _check(int(grasp_plan.get("raw_grasp_count", 0)) > 0, "Contact-GraspNet returned no raw grasps"),
            _check(len(box_visual["detections"]) > 0, "OWL-ViT did not detect the box target"),
            _check(box_visual["mask_pixels"] > 20, "SAM2 box mask is empty or too small"),
            _check(np.asarray(box_position).shape == (3,), "box pose position is not 3D"),
            _check(bool(x2_execution_plan_preview.get("ok")), f"X2 execution plan preview failed: {x2_execution_plan_preview.get('error')}"),
            _check(all(Path(path).exists() for path in grasp_visualization_paths.values()), "grasp visualizations were not written"),
        ]
        action_checks = [
            _check(bool(action_chain["grasp_success"]), "dynamic food object was not grasped/lifted"),
            _check(
                action_chain["success_attempt_name"] is not None,
                "no grasp attempt reached a lifted object state",
            ),
            _check(
                action_chain["place_move"] is not None and bool(action_chain["place_move"]["ok"]),
                "place move_hand returned False or was skipped",
            ),
            _check(bool(action_chain["release_near_box"]), "released object is not near the box target"),
            _check(summary["video"].get("combined", {}).get("written", False), "combined video was not written"),
        ]
        single_attempt_checks = [
            _check(action_chain.get("selected_candidate") is not None, "no side-grasp candidate was selected"),
            _check(action_chain.get("pregrasp_move") is not None, "pregrasp move was skipped"),
            _check(action_chain.get("grasp_move") is not None, "grasp move was skipped"),
            _check(bool(action_chain.get("grasp_success")), "side grasp did not secure the dynamic object"),
            _check(bool(action_chain.get("in_hand_after_close")), "object was not detected in hand after close"),
            _check(action_chain.get("place_move") is not None, "target move after close was skipped"),
            _check(summary["video"].get("combined", {}).get("written", False), "combined video was not written"),
        ]
        if args.visualize_only:
            checks = base_checks
        elif args.single_side_grasp_once:
            checks = base_checks + single_attempt_checks
        else:
            checks = base_checks + action_checks
        summary["checks"] = checks
        summary["ok"] = all(check == "pass" for check in checks)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"Wrote combined video {meta['path']} ({meta['frame_count']} frames)")
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
