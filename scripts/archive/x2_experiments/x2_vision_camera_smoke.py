"""Smoke test X2 wrist-camera observations for future visual primitives.

This script intentionally stops before detector / segmenter models.  It checks
whether the X2 BEHAVIOR wrapper exposes the camera data needed by visual pose
primitives: RGB, metric depth, camera intrinsics, and camera world poses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi


DEFAULT_MODALITIES = ["rgb", "depth", "depth_linear"]
DEFAULT_GLOBAL_CAMERA_MODALITIES = ["rgb", "depth_linear"]
DEFAULT_OBJECT_NAME = "cologne"
GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": DEFAULT_GLOBAL_CAMERA_MODALITIES,
    "sensor_kwargs": {
        "image_height": 480,
        "image_width": 480,
    },
    # Fixed torso/ego-style view for the current fixed-base X2 phase.
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}
WRIST_CAMERA_TOKENS = (
    "wrist",
    "gripper",
    "eef",
    "hand",
    "l_base_gripper",
    "r_base_gripper",
    "left_gripper",
    "right_gripper",
)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "root"


def _array_stats(value: Any) -> dict[str, Any]:
    arr = _as_numpy(value)
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size == 0:
        return stats
    if np.issubdtype(arr.dtype, np.number):
        numeric = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(numeric)
        stats["finite_count"] = int(finite.sum())
        stats["nan_count"] = int(np.isnan(numeric).sum())
        if finite.any():
            finite_values = numeric[finite]
            stats["min"] = float(finite_values.min())
            stats["max"] = float(finite_values.max())
            stats["mean"] = float(finite_values.mean())
            stats["std"] = float(finite_values.std())
    return stats


def _walk_observation(obs: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(obs, dict):
        leaves: list[tuple[tuple[str, ...], Any]] = []
        for key, value in obs.items():
            leaves.extend(_walk_observation(value, (*prefix, str(key))))
        return leaves
    return [(prefix, obs)]


def _save_rgb(path: Path, value: Any) -> None:
    arr = _as_numpy(value)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim < 3:
        return
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(arr)).save(path)


def _save_depth_preview(path: Path, value: Any) -> None:
    arr = _as_numpy(value).astype(np.float32)
    arr = np.squeeze(arr)
    finite = np.isfinite(arr)
    if arr.ndim != 2 or not finite.any():
        return
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi <= lo:
        return
    vis = np.zeros_like(arr, dtype=np.float32)
    vis[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(path)


def _parse_modalities(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _global_camera_config(image_size: int) -> dict[str, Any]:
    camera = dict(GLOBAL_CAMERA)
    sensor_kwargs = dict(camera["sensor_kwargs"])
    sensor_kwargs["image_height"] = int(image_size)
    sensor_kwargs["image_width"] = int(image_size)
    camera["sensor_kwargs"] = sensor_kwargs
    return camera


def _sensor_summary(sensor: Any, *, include_camera_params: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "class": sensor.__class__.__name__,
        "modalities": sorted(getattr(sensor, "modalities", [])),
        "prim_path": str(getattr(sensor, "prim_path", "")),
        "name": str(getattr(sensor, "name", "")),
    }
    for attr in ("image_width", "image_height", "focal_length", "horizontal_aperture"):
        try:
            summary[attr] = _jsonable(getattr(sensor, attr))
        except Exception as exc:
            summary[f"{attr}_error"] = repr(exc)
    try:
        pos, quat = sensor.get_position_orientation()
        summary["world_position"] = _jsonable(pos)
        summary["world_quat_xyzw"] = _jsonable(quat)
        summary["has_world_pose"] = True
    except Exception as exc:
        summary["has_world_pose"] = False
        summary["pose_error"] = repr(exc)

    if include_camera_params:
        try:
            if hasattr(sensor, "intrinsic_matrix"):
                summary["intrinsic_matrix"] = _jsonable(sensor.intrinsic_matrix)
                summary["has_intrinsic_matrix"] = True
                summary["intrinsic_source"] = "sensor.intrinsic_matrix"
        except Exception as exc:
            summary["has_intrinsic_matrix"] = False
            summary["intrinsic_error"] = repr(exc)
            fallback = _fallback_intrinsic_matrix(sensor)
            if fallback is not None:
                summary["intrinsic_matrix"] = _jsonable(fallback)
                summary["has_intrinsic_matrix"] = True
                summary["intrinsic_source"] = "focal_length_horizontal_aperture_fallback"
        try:
            if hasattr(sensor, "camera_parameters"):
                camera_params = sensor.camera_parameters
                summary["camera_parameters_keys"] = sorted(str(k) for k in camera_params.keys())
                for key in ("cameraProjection", "cameraViewTransform", "renderProductResolution"):
                    if key in camera_params:
                        summary[key] = _jsonable(camera_params[key])
        except Exception as exc:
            summary["camera_parameters_error"] = repr(exc)

    return summary


def _fallback_intrinsic_matrix(sensor: Any) -> np.ndarray | None:
    try:
        width = float(getattr(sensor, "image_width"))
        height = float(getattr(sensor, "image_height"))
        focal_length = float(getattr(sensor, "focal_length"))
        horizontal_aperture = float(getattr(sensor, "horizontal_aperture"))
    except Exception:
        return None
    if width <= 0 or height <= 0 or focal_length <= 0 or horizontal_aperture <= 0:
        return None
    fx = focal_length / horizontal_aperture * width
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _is_wrist_camera_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in WRIST_CAMERA_TOKENS)


def _check(condition: bool, message: str) -> str:
    return "pass" if condition else f"fail: {message}"


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-X X2 camera / visual primitive smoke test")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--output-dir", default="outputs/x2_vision_camera_smoke")
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--skip-object", action="store_true")
    parser.add_argument("--skip-camera-params", action="store_true")
    parser.add_argument("--with-global-camera", action="store_true", help="Inject a fixed external global_camera.")
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    camera_dir = output_dir / "cameras"
    camera_dir.mkdir(parents=True, exist_ok=True)

    modalities = _parse_modalities(args.modalities)
    global_camera = _global_camera_config(args.image_size)
    summary: dict[str, Any] = {
        "ok": False,
        "config": args.config,
        "requested_modalities": modalities,
        "with_global_camera": bool(args.with_global_camera),
        "global_camera": global_camera if args.with_global_camera else None,
        "output_dir": str(output_dir),
        "steps": {},
        "verdicts": {},
        "errors": [],
    }

    env = None
    try:
        objects = None
        if not args.skip_object:
            objects = [
                {
                    "type": "DatasetObject",
                    "name": DEFAULT_OBJECT_NAME,
                    "category": "bottle_of_cologne",
                    "model": "lyipur",
                    "position": [0.6, 0.3, 0.8],
                    "orientation": [0, 0, 0, 1],
                    "fixed_base": True,
                    "kinematic_only": True,
                }
            ]

        print("Creating X2BehaviourLowLevel with robot camera modalities:", modalities)
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=objects,
            external_sensors=[global_camera] if args.with_global_camera else None,
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_resolution=args.image_size,
            robot_obs_modalities=modalities,
        )
        api = X2ControlApi(env)
        obs, info = env.reset()
        for _ in range(max(0, int(args.settle_steps))):
            env.step(env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=False)))
        obs = api.get_env_observation()

        robot = env.robot
        sensors: dict[str, Any] = {}
        for sensor_name, sensor in sorted(robot.sensors.items()):
            sensors[sensor_name] = _sensor_summary(
                sensor,
                include_camera_params=not args.skip_camera_params,
            )
        external_sensors: dict[str, Any] = {}
        for sensor_name, sensor in sorted((getattr(env.env, "external_sensors", {}) or {}).items()):
            external_sensors[sensor_name] = _sensor_summary(
                sensor,
                include_camera_params=not args.skip_camera_params,
            )

        observation_leaves: dict[str, Any] = {}
        external_observation_leaves: dict[str, Any] = {}
        rgb_paths: list[str] = []
        depth_paths: list[str] = []
        depth_linear_paths: list[str] = []
        external_rgb_paths: list[str] = []
        external_depth_paths: list[str] = []
        external_depth_linear_paths: list[str] = []
        saved_files: list[str] = []

        def record_observation_leaf(
            leaf_name: str,
            suffix: str,
            value: Any,
            leaf_stats: dict[str, Any],
            rgb_accumulator: list[str],
            depth_accumulator: list[str],
            depth_linear_accumulator: list[str],
        ) -> None:
            leaf_stats[leaf_name] = _array_stats(value)
            safe_leaf_name = _safe_name(leaf_name)
            if suffix == "rgb":
                out_path = camera_dir / f"{safe_leaf_name}.png"
                _save_rgb(out_path, value)
                rgb_accumulator.append(leaf_name)
                saved_files.append(str(out_path))
            elif suffix in ("depth", "depth_linear"):
                arr = _as_numpy(value)
                npy_path = camera_dir / f"{safe_leaf_name}.npy"
                preview_path = camera_dir / f"{safe_leaf_name}.preview.png"
                np.save(npy_path, arr)
                _save_depth_preview(preview_path, arr)
                saved_files.append(str(npy_path))
                if preview_path.exists():
                    saved_files.append(str(preview_path))
                if suffix == "depth":
                    depth_accumulator.append(leaf_name)
                else:
                    depth_linear_accumulator.append(leaf_name)

        for path, value in _walk_observation(obs):
            leaf_name = "/".join(path)
            suffix = path[-1] if path else ""
            record_observation_leaf(
                leaf_name,
                suffix,
                value,
                observation_leaves,
                rgb_paths,
                depth_paths,
                depth_linear_paths,
            )

        for sensor_name in sorted(external_sensors.keys()):
            camera_obs = api.get_external_camera_observation(sensor_name)
            for key, value in camera_obs.items():
                if not isinstance(value, np.ndarray):
                    continue
                record_observation_leaf(
                    f"external/{sensor_name}/{key}",
                    key,
                    value,
                    external_observation_leaves,
                    external_rgb_paths,
                    external_depth_paths,
                    external_depth_linear_paths,
                )

        sensor_names = sorted(sensors.keys())
        wrist_sensor_names = [name for name in sensor_names if _is_wrist_camera_name(name)]
        pose_sensor_names = [
            name for name, item in sensors.items() if item.get("has_world_pose")
        ]
        intrinsic_sensor_names = [
            name for name, item in sensors.items() if item.get("has_intrinsic_matrix")
        ]

        summary["steps"]["camera_inventory"] = {
            "robot_name": robot.name,
            "robot_model": robot.model,
            "robot_obs_modalities": sorted(getattr(robot, "obs_modalities", [])),
            "sensor_names": sensor_names,
            "wrist_like_sensor_names": wrist_sensor_names,
            "sensors": sensors,
            "external_sensor_names": sorted(external_sensors.keys()),
            "external_sensors": external_sensors,
            "reset_info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
        }
        summary["steps"]["observation"] = {
            "top_level_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
            "observation_leaves": observation_leaves,
            "external_observation_leaves": external_observation_leaves,
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths,
            "depth_linear_paths": depth_linear_paths,
            "external_rgb_paths": external_rgb_paths,
            "external_depth_paths": external_depth_paths,
            "external_depth_linear_paths": external_depth_linear_paths,
            "saved_files": saved_files,
        }

        if not args.skip_object:
            try:
                obj_pos, obj_quat = api.get_object_pose(DEFAULT_OBJECT_NAME)
                summary["steps"]["oracle_object_pose"] = {
                    "object_name": DEFAULT_OBJECT_NAME,
                    "position": _jsonable(obj_pos),
                    "quat_xyzw": _jsonable(obj_quat),
                    "note": "This is registry/oracle pose, not visual pose.",
                }
            except Exception as exc:
                summary["steps"]["oracle_object_pose"] = {"error": repr(exc)}

        api_primitive_checks: list[str] = []
        api_primitive_detail: dict[str, Any] = {}
        try:
            api_camera_names = api.get_camera_names()
            api_external_camera_names = api.get_external_camera_names()
            api_primitive_detail["camera_names"] = api_camera_names
            api_primitive_detail["external_camera_names"] = api_external_camera_names
            api_primitive_checks.append(_check(api_camera_names == sensor_names, "API camera names differ from robot sensors"))
            api_primitive_checks.append(
                _check(
                    api_external_camera_names == sorted(external_sensors.keys()),
                    "API external camera names differ from env external sensors",
                )
            )
            for arm, label in ((0, "left"), (1, "right")):
                wrist_obs = api.get_wrist_camera_observation(arm=arm)
                pos, quat = api.get_camera_pose(arm=arm)
                K = api.get_camera_intrinsics(arm=arm)
                api_primitive_detail[label] = {
                    "camera_name": wrist_obs.get("camera_name"),
                    "obs_keys": sorted(wrist_obs.keys()),
                    "rgb": _array_stats(wrist_obs["rgb"]) if "rgb" in wrist_obs else None,
                    "depth_linear": _array_stats(wrist_obs["depth_linear"])
                    if "depth_linear" in wrist_obs
                    else None,
                    "pose": {"position": _jsonable(pos), "quat_xyzw": _jsonable(quat)},
                    "intrinsic_matrix": _jsonable(K),
                }
                api_primitive_checks.extend(
                    [
                        _check("rgb" in wrist_obs, f"{label} wrist API observation has no rgb"),
                        _check(
                            "depth_linear" in wrist_obs or "depth" in wrist_obs,
                            f"{label} wrist API observation has no depth/depth_linear",
                        ),
                        _check(np.asarray(pos).shape == (3,), f"{label} camera position shape is not (3,)"),
                        _check(np.asarray(quat).shape == (4,), f"{label} camera quat shape is not (4,)"),
                        _check(np.asarray(K).shape == (3, 3), f"{label} camera intrinsics shape is not (3,3)"),
                    ]
                )
            if args.with_global_camera:
                global_obs = api.get_external_camera_observation("global_camera")
                global_pos, global_quat = api.get_external_camera_pose("global_camera")
                global_K = api.get_external_camera_intrinsics("global_camera")
                api_primitive_detail["global_camera"] = {
                    "camera_name": global_obs.get("camera_name"),
                    "obs_keys": sorted(global_obs.keys()),
                    "rgb": _array_stats(global_obs["rgb"]) if "rgb" in global_obs else None,
                    "depth_linear": _array_stats(global_obs["depth_linear"])
                    if "depth_linear" in global_obs
                    else None,
                    "pose": {"position": _jsonable(global_pos), "quat_xyzw": _jsonable(global_quat)},
                    "intrinsic_matrix": _jsonable(global_K),
                }
                api_primitive_checks.extend(
                    [
                        _check("global_camera" in api_external_camera_names, "API has no global_camera"),
                        _check("rgb" in global_obs, "global_camera API observation has no rgb"),
                        _check("depth_linear" in global_obs, "global_camera API observation has no depth_linear"),
                        _check(np.asarray(global_pos).shape == (3,), "global_camera position shape is not (3,)"),
                        _check(np.asarray(global_quat).shape == (4,), "global_camera quat shape is not (4,)"),
                        _check(np.asarray(global_K).shape == (3, 3), "global_camera intrinsics shape is not (3,3)"),
                    ]
                )
        except Exception as exc:
            api_primitive_detail["error"] = repr(exc)
            api_primitive_checks.append(f"fail: X2ControlApi visual primitive call failed: {exc!r}")
        summary["steps"]["api_visual_primitives"] = api_primitive_detail
        summary["verdicts"]["api_visual_primitives"] = {
            "passed": all(item.startswith("pass") for item in api_primitive_checks),
            "checks": api_primitive_checks,
        }

        checks = [
            _check(len(sensor_names) > 0, "robot exposes no sensors"),
            _check(len(wrist_sensor_names) >= 2, f"expected >=2 wrist/gripper-like sensors, got {wrist_sensor_names}"),
            _check(len(rgb_paths) > 0, "observation has no rgb leaves"),
            _check(
                len(depth_linear_paths) > 0 or len(depth_paths) > 0,
                "observation has no depth/depth_linear leaves; metric visual pose is not ready",
            ),
            _check(len(pose_sensor_names) > 0, "no camera sensor world pose could be read"),
            _check(
                bool(args.skip_camera_params)
                or all(name in intrinsic_sensor_names for name in wrist_sensor_names),
                f"not all wrist cameras have intrinsics; intrinsic_sensor_names={intrinsic_sensor_names}",
            ),
        ]
        if args.with_global_camera:
            external_pose_sensor_names = [
                name for name, item in external_sensors.items() if item.get("has_world_pose")
            ]
            external_intrinsic_sensor_names = [
                name for name, item in external_sensors.items() if item.get("has_intrinsic_matrix")
            ]
            checks.extend(
                [
                    _check("global_camera" in external_sensors, "global_camera was not injected"),
                    _check(len(external_rgb_paths) > 0, "global_camera has no saved rgb observation"),
                    _check(
                        len(external_depth_linear_paths) > 0,
                        "global_camera has no saved depth_linear observation",
                    ),
                    _check(
                        "global_camera" in external_pose_sensor_names,
                        f"global_camera has no readable world pose; readable={external_pose_sensor_names}",
                    ),
                    _check(
                        bool(args.skip_camera_params) or "global_camera" in external_intrinsic_sensor_names,
                        f"global_camera has no intrinsics; readable={external_intrinsic_sensor_names}",
                    ),
                ]
            )
        summary["verdicts"]["camera_smoke"] = {
            "passed": all(item.startswith("pass") for item in checks),
            "checks": checks,
            "pose_sensor_names": pose_sensor_names,
            "intrinsic_sensor_names": intrinsic_sensor_names,
        }
        summary["ok"] = bool(
            summary["verdicts"]["camera_smoke"]["passed"]
            and summary["verdicts"]["api_visual_primitives"]["passed"]
        )

    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])
    finally:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {summary_path}")
        print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
