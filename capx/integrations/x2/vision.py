"""X2 RGB-D visual geometry helpers.

These functions are intentionally model-free.  They convert camera RGB-D
geometry and a 2D mask into world-frame 3D points / object centers.  Text
detection and segmentation can be layered on top later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensors / array-likes to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion into a 3x3 rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Return a 4x4 world-from-camera transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into an xyzw quaternion."""
    R = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s
    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def project_world_points(
    points_world: np.ndarray,
    intrinsic_matrix: np.ndarray,
    T_world_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image coordinates.

    OmniGibson VisionSensor depth follows an OpenGL-style camera frame:
    camera looks along -Z, +X is right, +Y is up, and image v is down.
    """
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    T_cam_world = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4))
    points_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    points_cam = (T_cam_world @ points_h.T).T[:, :3]

    depth = -points_cam[:, 2]
    valid = depth > 1e-6
    u = K[0, 0] * points_cam[:, 0] / depth + K[0, 2]
    v = -K[1, 1] * points_cam[:, 1] / depth + K[1, 2]
    return np.stack([u, v], axis=1), valid


def expected_depth_for_world_point(point_world: np.ndarray, T_world_cam: np.ndarray) -> float:
    """Return OpenGL-style positive camera depth for one world point."""
    point_h = np.array([*np.asarray(point_world, dtype=np.float64).reshape(3), 1.0], dtype=np.float64)
    point_cam = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)) @ point_h
    return float(-point_cam[2])


def backproject_mask_to_world(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    T_world_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject masked depth pixels into world-frame points.

    Returns:
        ``(points_world, depths)`` where ``points_world`` has shape ``(N, 3)``
        and ``depths`` are the corresponding positive camera depths.
    """
    mask = np.asarray(mask, dtype=bool)
    depth = np.squeeze(np.asarray(depth, dtype=np.float64))
    K = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D after squeeze, got shape {depth.shape}")
    if mask.shape != depth.shape:
        raise ValueError(f"mask shape {mask.shape} does not match depth shape {depth.shape}")

    vs, us = np.nonzero(mask)
    depths = depth[vs, us]
    valid = np.isfinite(depths) & (depths > 0)
    vs = vs[valid]
    us = us[valid]
    depths = depths[valid]
    if len(depths) == 0:
        return np.empty((0, 3), dtype=np.float64), depths

    xs = (us - K[0, 2]) * depths / K[0, 0]
    ys = -(vs - K[1, 2]) * depths / K[1, 1]
    zs = -depths
    points_cam_h = np.stack([xs, ys, zs, np.ones_like(depths)], axis=1)
    points_world = (T_world_cam @ points_cam_h.T).T[:, :3]
    return points_world, depths


def estimate_position_from_points(
    points_world: np.ndarray,
    depths: np.ndarray,
    *,
    expected_depth: float | None = None,
    depth_window: float | None = None,
) -> dict[str, Any]:
    """Estimate one robust world-frame position from masked point samples."""
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    result: dict[str, Any] = {
        "raw_point_count": int(len(points_world)),
        "expected_depth": None if expected_depth is None else float(expected_depth),
        "depth_window": None if depth_window is None else float(depth_window),
        "filtered_point_count": 0,
        "fallback_used": None,
        "position": None,
    }
    if len(points_world) == 0:
        return result

    filtered = points_world
    if expected_depth is not None and depth_window is not None:
        keep = np.isfinite(depths) & (np.abs(depths - float(expected_depth)) <= float(depth_window))
        filtered = points_world[keep]
        result["filtered_point_count"] = int(len(filtered))
        if len(filtered) == 0:
            filtered = points_world
            result["fallback_used"] = "all_points"
    else:
        result["filtered_point_count"] = int(len(filtered))

    position = np.median(filtered, axis=0)
    result["position"] = position
    return result


def _statistical_outlier_filter(points_world: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Apply Open3D statistical outlier filtering when available."""
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(points_world) < max(4, nb_neighbors):
        return points_world
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_world)
        cloud, _indices = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        filtered = np.asarray(cloud.points, dtype=np.float64)
        return filtered if len(filtered) > 0 else points_world
    except Exception:
        return points_world


def estimate_pose_from_points(
    points_world: np.ndarray,
    depths: np.ndarray,
    *,
    expected_depth: float | None = None,
    depth_window: float | None = None,
    method: str = "obb_center",
    remove_outliers: bool = True,
) -> dict[str, Any]:
    """Estimate an R1Pro-style object pose from masked world-frame points.

    ``method="obb_center"`` mirrors R1Pro's visual primitive path: mask/depth
    points are filtered, converted to an Open3D point cloud when available,
    and summarized by an oriented bounding box center and orientation.
    """
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    depths = np.asarray(depths, dtype=np.float64).reshape(-1)
    result = estimate_position_from_points(
        points_world,
        depths,
        expected_depth=expected_depth,
        depth_window=depth_window,
    )
    result.update(
        {
            "method": method,
            "orientation_quat_xyzw": None,
            "bbox_extent": None,
            "pose_points_world": np.empty((0, 3), dtype=np.float64),
        }
    )
    if len(points_world) == 0:
        return result

    if expected_depth is not None and depth_window is not None:
        keep = np.isfinite(depths) & (np.abs(depths - float(expected_depth)) <= float(depth_window))
        pose_points = points_world[keep]
        if len(pose_points) == 0:
            pose_points = points_world
    else:
        pose_points = points_world

    if remove_outliers:
        pose_points = _statistical_outlier_filter(pose_points)
    result["pose_points_world"] = pose_points
    result["filtered_point_count"] = int(len(pose_points))
    if len(pose_points) == 0:
        return result

    method = str(method)
    if method == "median":
        result["position"] = np.median(pose_points, axis=0)
        result["orientation_quat_xyzw"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        result["bbox_extent"] = np.ptp(pose_points, axis=0)
        return result
    if method == "aabb_center":
        lo = pose_points.min(axis=0)
        hi = pose_points.max(axis=0)
        result["position"] = (lo + hi) / 2.0
        result["orientation_quat_xyzw"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        result["bbox_extent"] = hi - lo
        return result
    if method != "obb_center":
        raise ValueError(f"Unsupported pose estimation method: {method!r}")

    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pose_points)
        obb = cloud.get_oriented_bounding_box()
        result["position"] = np.asarray(obb.center, dtype=np.float64)
        result["orientation_quat_xyzw"] = matrix_to_quat_xyzw(np.asarray(obb.R, dtype=np.float64))
        result["bbox_extent"] = np.asarray(obb.extent, dtype=np.float64)
    except Exception:
        centered = pose_points - np.mean(pose_points, axis=0)
        if len(pose_points) >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            R = vh.T
            if np.linalg.det(R) < 0:
                R[:, -1] *= -1
        else:
            R = np.eye(3, dtype=np.float64)
        local = centered @ R
        lo = local.min(axis=0)
        hi = local.max(axis=0)
        result["position"] = np.mean(pose_points, axis=0) + ((lo + hi) / 2.0) @ R.T
        result["orientation_quat_xyzw"] = matrix_to_quat_xyzw(R)
        result["bbox_extent"] = hi - lo
        result["fallback_used"] = "pca_obb"
    return result


def estimate_position_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_xyzw: np.ndarray,
    *,
    expected_depth: float | None = None,
    depth_window: float | None = None,
) -> dict[str, Any]:
    """Estimate a world-frame object position from a binary image mask."""
    T_world_cam = pose_to_matrix(camera_position, camera_quat_xyzw)
    points_world, depths = backproject_mask_to_world(mask, depth, intrinsic_matrix, T_world_cam)
    result = estimate_position_from_points(
        points_world,
        depths,
        expected_depth=expected_depth,
        depth_window=depth_window,
    )
    result["points_world"] = points_world
    result["depths"] = depths
    return result


def estimate_pose_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_xyzw: np.ndarray,
    *,
    expected_depth: float | None = None,
    depth_window: float | None = None,
    method: str = "obb_center",
    remove_outliers: bool = True,
) -> dict[str, Any]:
    """Estimate an object pose from a binary mask using R1Pro-style geometry."""
    T_world_cam = pose_to_matrix(camera_position, camera_quat_xyzw)
    points_world, depths = backproject_mask_to_world(mask, depth, intrinsic_matrix, T_world_cam)
    result = estimate_pose_from_points(
        points_world,
        depths,
        expected_depth=expected_depth,
        depth_window=depth_window,
        method=method,
        remove_outliers=remove_outliers,
    )
    result["points_world"] = points_world
    result["depths"] = depths
    return result


def make_projected_aabb_mask(
    center: np.ndarray,
    extent: np.ndarray,
    intrinsic_matrix: np.ndarray,
    T_world_cam: np.ndarray,
    image_shape: tuple[int, int],
    *,
    margin_px: int = 0,
    min_half_extent: float = 0.025,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Make an oracle 2D mask by projecting a world-frame AABB."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    extent = np.asarray(extent, dtype=np.float64).reshape(3)
    half = np.maximum(extent / 2.0, float(min_half_extent))
    offsets = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    ) * half
    corners = center[None, :] + offsets
    uv, valid = project_world_points(corners, intrinsic_matrix, T_world_cam)
    valid_uv = uv[valid & np.all(np.isfinite(uv), axis=1)]
    height, width = image_shape
    mask = np.zeros((height, width), dtype=bool)
    detail: dict[str, Any] = {
        "projected_corners_uv": np.round(uv, 3).tolist(),
        "num_valid_projected_corners": int(len(valid_uv)),
    }
    if len(valid_uv) == 0:
        return mask, detail

    lo = np.floor(valid_uv.min(axis=0)).astype(int) - int(margin_px)
    hi = np.ceil(valid_uv.max(axis=0)).astype(int) + int(margin_px)
    x0 = int(np.clip(lo[0], 0, width - 1))
    y0 = int(np.clip(lo[1], 0, height - 1))
    x1 = int(np.clip(hi[0], 0, width - 1))
    y1 = int(np.clip(hi[1], 0, height - 1))
    if x1 >= x0 and y1 >= y0:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    detail["bbox_xyxy"] = [x0, y0, x1, y1]
    detail["mask_pixels"] = int(mask.sum())
    return mask, detail
