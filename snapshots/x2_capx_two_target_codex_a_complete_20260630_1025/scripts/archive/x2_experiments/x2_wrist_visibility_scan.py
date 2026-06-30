"""Scan candidate tabletop object positions for right wrist camera visibility."""

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
OBJECT_SIZE = 0.04
TABLE_NAME = "x2_wrist_scan_support"
TABLE_CENTER = np.array([0.305, -0.14, 0.925], dtype=np.float64)
TABLE_SCALE = np.array([0.18, 0.18, 0.012], dtype=np.float64)

CANDIDATES = [
    {"name": "red_mid_low", "position": [0.300, -0.160, 0.945], "rgba": [1.0, 0.02, 0.02, 1.0], "color": "red"},
    {"name": "green_mid", "position": [0.315, -0.125, 0.945], "rgba": [0.02, 1.0, 0.02, 1.0], "color": "green"},
    {"name": "blue_center", "position": [0.285, -0.100, 0.945], "rgba": [0.02, 0.12, 1.0, 1.0], "color": "blue"},
    {"name": "yellow_high", "position": [0.335, -0.155, 0.955], "rgba": [1.0, 0.9, 0.02, 1.0], "color": "yellow"},
    {"name": "magenta_center_high", "position": [0.305, -0.075, 0.955], "rgba": [1.0, 0.02, 1.0, 1.0], "color": "magenta"},
    {"name": "cyan_inner_high", "position": [0.255, -0.130, 0.955], "rgba": [0.02, 1.0, 1.0, 1.0], "color": "cyan"},
]

COLOR_RULES = {
    "red": lambda rgb: (rgb[:, :, 0] > 140) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100),
    "green": lambda rgb: (rgb[:, :, 1] > 140) & (rgb[:, :, 0] < 120) & (rgb[:, :, 2] < 120),
    "blue": lambda rgb: (rgb[:, :, 2] > 140) & (rgb[:, :, 0] < 120) & (rgb[:, :, 1] < 140),
    "yellow": lambda rgb: (rgb[:, :, 0] > 140) & (rgb[:, :, 1] > 130) & (rgb[:, :, 2] < 100),
    "magenta": lambda rgb: (rgb[:, :, 0] > 140) & (rgb[:, :, 2] > 140) & (rgb[:, :, 1] < 120),
    "cyan": lambda rgb: (rgb[:, :, 1] > 140) & (rgb[:, :, 2] > 140) & (rgb[:, :, 0] < 120),
}


def _scene_objects() -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = [
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
        }
    ]
    for candidate in CANDIDATES:
        objects.append(
            {
                "type": "PrimitiveObject",
                "name": candidate["name"],
                "primitive_type": "Cube",
                "size": OBJECT_SIZE,
                "position": candidate["position"],
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "kinematic_only": True,
                "rgba": candidate["rgba"],
            }
        )
    return objects


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan X2 right wrist camera candidate object visibility")
    parser.add_argument("--output-dir", default="outputs/x2_wrist_visibility_scan")
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ok": False, "candidates": CANDIDATES, "errors": []}
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
        low_level.settle_robot_steps(steps=12)
        obs = api.get_wrist_camera_observation(arm=ARM)
        rgb = _rgb_u8(obs["rgb"])
        image_path = output_dir / "right_wrist_rgb.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        counts = {}
        for candidate in CANDIDATES:
            color = candidate["color"]
            mask = COLOR_RULES[color](rgb)
            counts[candidate["name"]] = {
                "color": color,
                "pixel_count": int(mask.sum()),
                "position": candidate["position"],
            }
        summary.update(
            {
                "ok": True,
                "camera_name": obs.get("camera_name"),
                "image_path": str(image_path),
                "color_counts": counts,
                "best": max(counts.items(), key=lambda kv: kv[1]["pixel_count"]),
            }
        )
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1], flush=True)
    summary["elapsed_s"] = round(time.time() - start, 3)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[x2-wrist-visibility-scan] wrote {path}", flush=True)
    print(json.dumps({"ok": summary["ok"], "best": summary.get("best"), "errors": len(summary["errors"])}, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
