"""Move one red cube through candidate positions and measure right wrist visibility."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import capx.envs.simulators  # noqa: F401
import capx.envs.tasks  # noqa: F401
import capx.integrations  # noqa: F401
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi


ARM = 1
OBJECT_NAME = "x2_red_visibility_target"
TABLE_NAME = "x2_red_visibility_support"
OBJECT_SIZE = 0.04
TABLE_CENTER = np.array([0.34, -0.08, 0.935], dtype=np.float64)
TABLE_SCALE = np.array([0.26, 0.26, 0.012], dtype=np.float64)

CANDIDATES = [
    [0.300, -0.160, 0.955],
    [0.335, -0.155, 0.955],
    [0.370, -0.150, 0.955],
    [0.300, -0.080, 0.955],
    [0.335, -0.080, 0.955],
    [0.370, -0.080, 0.955],
    [0.300, 0.000, 0.955],
    [0.335, 0.000, 0.955],
    [0.370, 0.000, 0.955],
    [0.300, -0.080, 0.985],
    [0.335, -0.080, 0.985],
    [0.370, -0.080, 0.985],
    [0.335, 0.000, 0.985],
]


def _scene_objects() -> list[dict[str, Any]]:
    return [
        {
            "type": "PrimitiveObject",
            "name": TABLE_NAME,
            "primitive_type": "Cube",
            "size": 1.0,
            "scale": TABLE_SCALE.tolist(),
            "position": TABLE_CENTER.tolist(),
            "orientation": [0, 0, 0, 1],
            "fixed_base": True,
            "rgba": [0.35, 0.35, 0.35, 1.0],
        },
        {
            "type": "PrimitiveObject",
            "name": OBJECT_NAME,
            "primitive_type": "Cube",
            "size": OBJECT_SIZE,
            "position": CANDIDATES[0],
            "orientation": [0, 0, 0, 1],
            "fixed_base": True,
            "kinematic_only": True,
            "rgba": [1.0, 0.0, 0.0, 1.0],
        },
    ]


def _rgb_u8(value: Any) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, :, 0] > 140) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)


def _score(rgb: np.ndarray) -> dict[str, Any]:
    mask = _red_mask(rgb)
    ys, xs = np.nonzero(mask)
    result: dict[str, Any] = {"red_pixels": int(mask.sum())}
    if len(xs) == 0:
        result.update({"centroid_xy": None, "bbox_xyxy": None, "center_score": 0.0})
        return result
    h, w = mask.shape
    centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    image_center = np.array([0.5 * (w - 1), 0.5 * (h - 1)], dtype=np.float64)
    dist = float(np.linalg.norm((centroid - image_center) / np.array([w, h], dtype=np.float64)))
    result.update(
        {
            "centroid_xy": np.round(centroid, 3).tolist(),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "center_score": round(float(mask.sum()) / (1.0 + 250.0 * dist), 6),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan one red cube through right wrist camera candidate positions")
    parser.add_argument("--output-dir", default="outputs/x2_wrist_red_position_scan")
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {"ok": False, "errors": [], "results": []}
    start = time.time()
    try:
        low_level = X2BehaviourLowLevel(
            controller_cfg="x2_robotiq85_joint_primitives.yaml",
            objects=_scene_objects(),
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=ARM,
            robot_camera_resolution=384,
            robot_obs_modalities=["rgb", "depth", "depth_linear"],
        )
        api = X2ControlApi(low_level)
        obj = low_level.env.scene.object_registry("name", OBJECT_NAME)
        if obj is None:
            raise RuntimeError(f"{OBJECT_NAME} not found")
        for idx, pos in enumerate(CANDIDATES):
            obj.set_position_orientation(position=np.asarray(pos, dtype=np.float64), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
            low_level.settle_robot_steps(steps=4)
            obs = api.get_wrist_camera_observation(arm=ARM)
            rgb = _rgb_u8(obs["rgb"])
            image_path = frames_dir / f"candidate_{idx:02d}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            item = {
                "index": idx,
                "position": [float(v) for v in pos],
                "image_path": str(image_path),
                **_score(rgb),
            }
            summary["results"].append(item)
            print("[x2-red-scan]", item, flush=True)
        visible = [item for item in summary["results"] if int(item["red_pixels"]) > 20]
        best = max(visible, key=lambda item: (float(item["center_score"]), int(item["red_pixels"]))) if visible else None
        summary.update({"ok": True, "camera_name": api.get_wrist_camera_observation(arm=ARM).get("camera_name"), "best": best})
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)
    summary["elapsed_s"] = round(time.time() - start, 3)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-red-scan] wrote {path}", flush=True)
    print(json.dumps({"ok": summary["ok"], "best": summary.get("best"), "errors": len(summary["errors"])}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
