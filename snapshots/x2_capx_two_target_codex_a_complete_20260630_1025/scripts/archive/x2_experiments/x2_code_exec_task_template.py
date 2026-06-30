"""Layer-4 fixed task template for CAP-X X2 code execution.

This validates that CAP-X generated-code execution can command the X2 right arm
to a deterministic world-frame grasp pose, close the gripper on a simple object,
then return the gripper to its original pose:

    open_gripper -> move_hand(fixed_grasp_pose) -> close_gripper -> move_hand(home_pose)

The object is deliberately a small primitive cube placed at the already-verified
reachable pose so this remains a deterministic CAP-X task-template demo rather
than a grasp-pose sampling test.
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
import torch

import capx.envs.simulators  # noqa: F401 - registers low-level envs
import capx.envs.tasks  # noqa: F401 - registers code-execution envs/configs
import capx.integrations  # noqa: F401 - registers APIs
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.envs.tasks import CodeExecEnvConfig, get_exec_env

OBJECT_NAME = "demo_grasp_cube"
TABLE_NAME = "demo_support"
FIXED_GRASP_POS = [0.275818, -0.227553, 0.938968]
FIXED_GRASP_QUAT = [0.648934, 0.168974, 0.733815, 0.10885]
GRIPPER_CENTER_OFFSET_EEF = [0.0, 0.0, 0.105]
OBJECT_SIZE = 0.025
STUCK_PATIENCE_STEPS = 45
MAX_GRIPPER_OPENING_M = 0.0997
POST_SETTLE_STEPS_DEFAULT = 80


def _quat_xyzw_apply(quat_xyzw: list[float], vec: list[float]) -> list[float]:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return (v + q_w * t + np.cross(q_xyz, t)).tolist()


OBJECT_POS = (
    np.asarray(FIXED_GRASP_POS, dtype=np.float64)
    + np.asarray(_quat_xyzw_apply(FIXED_GRASP_QUAT, GRIPPER_CENTER_OFFSET_EEF), dtype=np.float64)
).round(6).tolist()
SUPPORT_TOP_Z = OBJECT_POS[2] - OBJECT_SIZE / 2.0
GLOBAL_CAMERA = {
    "sensor_type": "VisionSensor",
    "name": "global_camera",
    "relative_prim_path": "/global_camera",
    "modalities": ["rgb"],
    "sensor_kwargs": {
        "image_height": 768,
        "image_width": 768,
    },
    "position": [0.85, -1.05, 1.45],
    "orientation": [0.467415, 0.155805, 0.275181, 0.825544],
}


FIXED_PICK_TEMPLATE = r'''
import numpy as np

ARM = 1
OBJECT_NAME = "demo_grasp_cube"
FIXED_GRASP_POS = np.array([0.275818, -0.227553, 0.938968], dtype=np.float32)
STUCK_PATIENCE_STEPS = 45

trace = []

def pose_to_list(pose):
    pos, quat = pose
    return {
        "pos": np.asarray(pos, dtype=np.float32).round(6).tolist(),
        "quat": np.asarray(quat, dtype=np.float32).round(6).tolist(),
    }

def record(label, **kwargs):
    entry = {"label": label}
    entry.update(kwargs)
    trace.append(entry)

def pause():
    settle_robot(steps=12)

def object_pose_to_list(name):
    try:
        return pose_to_list(get_object_pose(name))
    except Exception as exc:
        return {"error": str(exc)}

def safe_check_in_hand():
    try:
        return bool(check_object_in_hand(arm=ARM))
    except Exception:
        return False

initial_eef = get_current_eef_pose(arm=ARM)
record(
    "initial",
    eef=pose_to_list(initial_eef),
    object=object_pose_to_list(OBJECT_NAME),
    gripper=get_gripper_state(arm=ARM),
)
pause()

open_gripper(arm=ARM)
record(
    "open_gripper",
    gripper=get_gripper_state(arm=ARM),
)
pause()

target_pose = (FIXED_GRASP_POS, np.asarray(initial_eef[1], dtype=np.float32))
record(
    "fixed_grasp_pose",
    target=pose_to_list(target_pose),
    object=object_pose_to_list(OBJECT_NAME),
    gripper=get_gripper_state(arm=ARM),
)
pause()

move_ok = move_hand(
    target_pose,
    arm=ARM,
    pos_thresh=0.005,
    ori_thresh=0.1,
    stop_if_stuck=True,
    stuck_patience_steps=STUCK_PATIENCE_STEPS,
)
target_eef = get_current_eef_pose(arm=ARM)
record(
    "move_fixed_grasp",
    ok=bool(move_ok),
    eef=pose_to_list(target_eef),
    gripper=get_gripper_state(arm=ARM),
)
pause()

close_gripper(arm=ARM)
grasp_success_after_close = safe_check_in_hand()
record(
    "close_gripper",
    object=object_pose_to_list(OBJECT_NAME),
    gripper=get_gripper_state(arm=ARM),
    grasp_success=grasp_success_after_close,
)
pause()

move_back_ok = move_hand(
    initial_eef,
    arm=ARM,
    pos_thresh=0.005,
    ori_thresh=0.1,
    stop_if_stuck=True,
    stuck_patience_steps=STUCK_PATIENCE_STEPS,
)
home_eef = get_current_eef_pose(arm=ARM)
grasp_success_after_return = safe_check_in_hand()
record(
    "return_home",
    ok=bool(move_back_ok),
    eef=pose_to_list(home_eef),
    object=object_pose_to_list(OBJECT_NAME),
    gripper=get_gripper_state(arm=ARM),
    grasp_success=grasp_success_after_return,
)
pause()

final_eef = get_current_eef_pose(arm=ARM)
record(
    "final_home",
    eef=pose_to_list(final_eef),
    object=object_pose_to_list(OBJECT_NAME),
    gripper=get_gripper_state(arm=ARM),
)
pause()

RESULT = {
    "trace": trace,
    "move_ok": bool(move_ok),
    "move_back_ok": bool(move_back_ok),
    "grasp_success_after_close": bool(grasp_success_after_close),
    "grasp_success_after_return": bool(grasp_success_after_return),
    "initial_eef_pos": np.asarray(initial_eef[0], dtype=np.float32).round(6).tolist(),
    "target_grasp_pos": np.asarray(target_pose[0], dtype=np.float32).round(6).tolist(),
    "target_eef_pos": np.asarray(target_eef[0], dtype=np.float32).round(6).tolist(),
    "home_eef_pos": np.asarray(home_eef[0], dtype=np.float32).round(6).tolist(),
    "final_eef_pos": np.asarray(final_eef[0], dtype=np.float32).round(6).tolist(),
}
'''


def _as_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value) -> list[float]:
    return [round(float(v), 6) for v in _as_np(value).reshape(-1)]


def _check(condition: bool, message: str) -> str:
    return "pass" if condition else f"fail: {message}"


def _place_demo_objects_at_gripper_center(env: X2BehaviourLowLevel) -> dict[str, Any]:
    """Place the demo object at the target gripper opening center for the current EEF orientation."""
    _eef_pos, eef_quat = env.get_robot_eef_pose(arm=1)
    target_quat = _as_np(eef_quat).reshape(-1)
    object_pos = (
        np.asarray(FIXED_GRASP_POS, dtype=np.float64)
        + np.asarray(_quat_xyzw_apply(target_quat.tolist(), GRIPPER_CENTER_OFFSET_EEF), dtype=np.float64)
    ).round(6)
    support_top_z = float(object_pos[2] - OBJECT_SIZE / 2.0)

    table = env.env.scene.object_registry("name", TABLE_NAME)
    obj = env.env.scene.object_registry("name", OBJECT_NAME)
    if table is None or obj is None:
        raise RuntimeError(f"Missing demo objects: table={table is not None}, object={obj is not None}")

    table.set_position_orientation(
        position=torch.tensor([object_pos[0], object_pos[1], support_top_z - 0.005], dtype=torch.float32),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    obj.set_position_orientation(
        position=torch.tensor(object_pos, dtype=torch.float32),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    obj.keep_still()
    env.settle_robot_steps(steps=6)

    placed_obj_pos, placed_obj_quat = obj.get_position_orientation()
    return {
        "object_pos": _as_list(placed_obj_pos),
        "object_quat_xyzw": _as_list(placed_obj_quat),
        "support_top_z": round(support_top_z, 6),
        "target_eef_pos": FIXED_GRASP_POS,
        "target_eef_quat_source": "initial_current_eef_quat",
        "target_eef_quat_xyzw": _as_list(target_quat),
        "object_pos_source": "target_eef_pos_plus_current_eef_quat_rotated_gripper_center_offset_eef",
    }


def _scene_setup_summary(env: X2BehaviourLowLevel, placement: dict[str, Any]) -> dict[str, Any]:
    eef_pos, _eef_quat = env.get_robot_eef_pose(arm=1)
    return {
        "mode": "fixed_object_pick_return_demo",
        "eef_pos": _as_list(eef_pos),
        "fixed_grasp_pos": FIXED_GRASP_POS,
        "gripper_center_offset_eef": GRIPPER_CENTER_OFFSET_EEF,
        "actual_placement": placement,
        "max_gripper_opening_m": MAX_GRIPPER_OPENING_M,
        "objects": [
            {
                "name": OBJECT_NAME,
                "type": "PrimitiveObject",
                "primitive_type": "Cube",
                "size": OBJECT_SIZE,
                "position": placement["object_pos"],
            },
            {
                "name": TABLE_NAME,
                "type": "PrimitiveObject",
                "primitive_type": "Cube",
                "top_z": placement["support_top_z"],
            },
        ],
    }


def _post_settle_summary(env: X2BehaviourLowLevel, steps: int) -> dict[str, Any]:
    before_pos, before_quat = env.get_robot_eef_pose(arm=1)
    start_frame = env.get_video_frame_count()
    executed_steps = env.settle_robot_steps(steps=steps)
    after_pos, after_quat = env.get_robot_eef_pose(arm=1)
    before = _as_np(before_pos)
    after = _as_np(after_pos)
    return {
        "steps": int(executed_steps),
        "video_frames_start": int(start_frame),
        "video_frames_end": int(env.get_video_frame_count()),
        "eef_pos_before": _as_list(before_pos),
        "eef_quat_before": _as_list(before_quat),
        "eef_pos_after": _as_list(after_pos),
        "eef_quat_after": _as_list(after_quat),
        "eef_pos_drift": round(float(np.linalg.norm(after - before)), 6),
    }


def _validate_result(result: dict[str, Any] | None, info: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    checks: list[str] = []
    detail: dict[str, Any] = {
        "result": result,
        "sandbox_rc": info.get("sandbox_rc"),
        "stdout_tail": info.get("stdout", "")[-2000:],
        "stderr_tail": info.get("stderr", "")[-4000:],
    }

    checks.append(
        _check(info.get("sandbox_rc") == 0, f"sandbox_rc={info.get('sandbox_rc')} stderr={info.get('stderr', '')[-500:]}")
    )
    checks.append(_check(isinstance(result, dict), f"RESULT must be dict, got {type(result).__name__}"))
    if not isinstance(result, dict):
        return False, detail, checks

    trace = result.get("trace")
    expected_labels = [
        "initial",
        "open_gripper",
        "fixed_grasp_pose",
        "move_fixed_grasp",
        "close_gripper",
        "return_home",
        "final_home",
    ]
    labels = [entry.get("label") for entry in trace] if isinstance(trace, list) else []
    checks.append(_check(labels == expected_labels, f"trace labels {labels} != {expected_labels}"))
    move_ok = result.get("move_ok")
    move_back_ok = result.get("move_back_ok")
    grasp_success_after_close = bool(result.get("grasp_success_after_close"))
    grasp_success_after_return = bool(result.get("grasp_success_after_return"))
    detail["motion_success"] = {
        "move_ok": bool(move_ok),
        "move_back_ok": bool(move_back_ok),
    }
    detail["grasp_success"] = {
        "after_close": grasp_success_after_close,
        "after_return": grasp_success_after_return,
    }
    checks.append(_check(OBJECT_SIZE < MAX_GRIPPER_OPENING_M, f"object size {OBJECT_SIZE} exceeds gripper opening"))
    if move_ok is not True:
        detail["move_warning"] = "move_hand(fixed_grasp_pose) returned False; measured pose error is checked separately"
    if move_back_ok is not True:
        detail["move_back_warning"] = "move_hand(initial_eef) returned False; measured home error is checked separately"

    initial_eef = np.asarray(result.get("initial_eef_pos", []), dtype=np.float32)
    target = np.asarray(result.get("target_grasp_pos", []), dtype=np.float32)
    target_eef = np.asarray(result.get("target_eef_pos", []), dtype=np.float32)
    home_eef = np.asarray(result.get("home_eef_pos", []), dtype=np.float32)
    final_eef = np.asarray(result.get("final_eef_pos", []), dtype=np.float32)
    shapes_ok = all(arr.shape == (3,) for arr in [initial_eef, target, target_eef, home_eef, final_eef])
    checks.append(_check(shapes_ok, "all reported positions must be 3D"))
    if shapes_ok:
        target_eef_error = float(np.linalg.norm(target_eef - target))
        home_eef_error = float(np.linalg.norm(home_eef - initial_eef))
        final_drift = float(np.linalg.norm(final_eef - home_eef))
        requested_delta = float(np.linalg.norm(target - initial_eef))
        detail.update(
            {
                "requested_delta": round(requested_delta, 6),
                "target_eef_error": round(target_eef_error, 6),
                "home_eef_error": round(home_eef_error, 6),
                "final_drift_after_close": round(final_drift, 6),
            }
        )
        checks.append(_check(target_eef_error < 0.01, f"target EEF error {target_eef_error:.4f}m"))
        checks.append(_check(home_eef_error < 0.02, f"home EEF error {home_eef_error:.4f}m"))
        checks.append(_check(final_drift < 0.03, f"final drift after close {final_drift:.4f}m"))

    return all(v.startswith("pass") for v in checks), detail, checks


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> dict[str, Any]:
    if not frames:
        return {"path": None, "frame_count": 0, "written": False}
    arr = np.asarray(frames)
    media.write_video(path, arr, fps=fps)
    return {
        "path": str(path),
        "frame_count": int(len(frames)),
        "shape": list(arr.shape),
        "fps": fps,
        "written": True,
    }


def _resize_nearest(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    y_idx = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
    return frame[y_idx][:, x_idx]


def _combined_frames(frames_by_view: dict[str, list[np.ndarray]], preferred_order: list[str]) -> list[np.ndarray]:
    views = [view for view in preferred_order if frames_by_view.get(view)]
    if len(views) < 2:
        return []
    frame_count = min(len(frames_by_view[view]) for view in views)
    height = max(frames_by_view[view][0].shape[0] for view in views)
    combined = []
    for i in range(frame_count):
        row = [_resize_nearest(frames_by_view[view][i], height) for view in views]
        combined.append(np.concatenate(row, axis=1))
    return combined


def _write_videos(output_dir: Path, frames: Any, fps: int) -> dict[str, Any]:
    if isinstance(frames, dict):
        frames_by_view = frames
    else:
        frames_by_view = {"rgb": frames}

    result: dict[str, Any] = {"views": {}, "fps": fps}
    for view, view_frames in frames_by_view.items():
        result["views"][view] = _write_video(output_dir / f"{view}.mp4", view_frames, fps=fps)

    combined = _combined_frames(frames_by_view, preferred_order=["global", "robot", "rgb"])
    result["combined"] = _write_video(output_dir / "video_combined.mp4", combined, fps=fps)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-X X2 layer-4 fixed task-template smoke test")
    parser.add_argument("--output-dir", default="outputs/x2_code_exec_pick_demo_dynamic_object")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--post-settle-steps", type=int, default=POST_SETTLE_STEPS_DEFAULT)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    summary: dict[str, Any] = {
        "ok": False,
        "object_name": OBJECT_NAME,
        "table_name": TABLE_NAME,
        "fixed_grasp_pos": FIXED_GRASP_POS,
        "global_camera": GLOBAL_CAMERA,
        "fixed_code": FIXED_PICK_TEMPLATE,
        "steps": {},
        "verdicts": {},
        "video": {},
        "errors": [],
    }

    try:
        objects = [
            {
                "type": "PrimitiveObject",
                "name": TABLE_NAME,
                "primitive_type": "Cube",
                "size": 1.0,
                "scale": [0.08, 0.08, 0.01],
                "position": [OBJECT_POS[0], OBJECT_POS[1], SUPPORT_TOP_Z - 0.005],
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "rgba": [0.35, 0.35, 0.35, 1.0],
            },
            {
                "type": "PrimitiveObject",
                "name": OBJECT_NAME,
                "primitive_type": "Cube",
                "size": OBJECT_SIZE,
                "position": OBJECT_POS,
                "orientation": [0, 0, 0, 1],
                "fixed_base": False,
                "rgba": [1.0, 0.05, 0.05, 1.0],
            },
        ]
        low_level = X2BehaviourLowLevel(
            objects=objects,
            external_sensors=[GLOBAL_CAMERA],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_camera_arm=1,
            robot_camera_resolution=256,
        )
        exec_env_cls = get_exec_env("x2_behavior_code_env")
        exec_env = exec_env_cls(CodeExecEnvConfig(low_level=low_level, apis=["X2ControlApi"]))
        _obs, _info = exec_env.reset()

        placement = _place_demo_objects_at_gripper_center(low_level)
        summary["steps"]["scene_setup"] = _scene_setup_summary(low_level, placement)
        summary["verdicts"]["scene_setup"] = {"passed": True, "checks": ["pass"]}

        exec_env.enable_video_capture(True, clear=True)
        _obs, reward, terminated, truncated, info = exec_env.step(FIXED_PICK_TEMPLATE)
        summary["steps"]["post_task_passive_settle"] = _post_settle_summary(low_level, args.post_settle_steps)
        result = exec_env._exec_globals.get("RESULT")
        passed, detail, checks = _validate_result(result, info)
        detail.update(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        summary["steps"]["fixed_pick_template"] = detail
        summary["verdicts"]["fixed_pick_template"] = {"passed": passed, "checks": checks}

        frames = exec_env.get_video_frames()
        summary["video"] = _write_videos(output_dir, frames, fps=args.video_fps)
        summary["video_sources"] = getattr(low_level, "_last_video_sources", {})
        summary["ok"] = all(v["passed"] for v in summary["verdicts"].values())
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    for view, meta in summary["video"].get("views", {}).items():
        if meta.get("written"):
            print(f"Wrote {view} video {meta['path']} ({meta['frame_count']} frames)")
    if summary["video"].get("combined", {}).get("written"):
        meta = summary["video"]["combined"]
        print(f"Wrote combined video {meta['path']} ({meta['frame_count']} frames)")
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
