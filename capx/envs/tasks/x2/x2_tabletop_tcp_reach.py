from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.tasks.base import CodeExecutionEnvBase


TARGET_NAME = "x2_reach_target_marker"
ARM = 1
SUCCESS_THRESHOLD_M = 0.02


PROMPT = f"""
You are controlling a fixed-base X2 robot in a simple tabletop reach task.
Goal: move the right gripper TCP / finger center to the red target marker named "{TARGET_NAME}".
The scene, robot, target marker, support surface, and camera are already created by the environment.
Do not create or reset the simulator. Only use the injected APIs.
The visual/object pose API returns world-frame object poses. For this task, command a world-frame TCP pose using move_tcp(), or convert TCP to EEF with tcp_pose_to_eef_pose() before move_hand().
Use arm={ARM}. Keep the final TCP near the target marker.
You may write python code comments for reasoning but ONLY write executable Python code and do not write it in code fences.
If you want to use numpy, scipy for spatial transformations, opencv, pytorch, or any other libraries, import them explicitly.
The functions (APIs) below are already imported to the environment.
"""


ORACLE_CODE = f"""
import numpy as np

ARM = {ARM}
TARGET_NAME = "{TARGET_NAME}"

trace = []

def pose_to_dict(pose):
    pos, quat = pose
    return {{
        "position": np.asarray(pos, dtype=np.float64).round(6).tolist(),
        "quat_xyzw": np.asarray(quat, dtype=np.float64).round(6).tolist(),
    }}

def record(label, **kwargs):
    item = {{"label": label}}
    item.update(kwargs)
    trace.append(item)

def quat_xyzw_to_matrix(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

initial_eef = get_current_eef_pose(arm=ARM)
initial_tcp_offset = get_tcp_offset_eef(arm=ARM)
open_gripper(arm=ARM)
settle_robot(steps=12)

target_pos, target_quat = get_object_pose(TARGET_NAME)
target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
target_tcp_quat = np.asarray(initial_eef[1], dtype=np.float64).reshape(4)
target_tcp_pose = (target_pos, target_tcp_quat)

record(
    "target_from_env",
    target_position=target_pos.round(6).tolist(),
    target_quat_xyzw=np.asarray(target_quat, dtype=np.float64).round(6).tolist(),
    initial_eef=pose_to_dict(initial_eef),
    tcp_offset_eef=np.asarray(initial_tcp_offset, dtype=np.float64).round(6).tolist(),
)

move_ok = move_tcp(
    target_tcp_pose,
    arm=ARM,
    pos_thresh=0.005,
    ori_thresh=0.1,
    stop_if_stuck=True,
    stuck_patience_steps=45,
    max_steps=1200,
)
settle_robot(steps=12)
reached_eef = get_current_eef_pose(arm=ARM)
offset = np.asarray(get_tcp_offset_eef(arm=ARM), dtype=np.float64)
reached_tcp = np.asarray(reached_eef[0], dtype=np.float64) + quat_xyzw_to_matrix(reached_eef[1]) @ offset
tcp_error = float(np.linalg.norm(reached_tcp - target_pos))

record(
    "move_tcp",
    ok=bool(move_ok),
    target_tcp_position=target_pos.round(6).tolist(),
    reached_eef=pose_to_dict(reached_eef),
    reached_tcp_position=reached_tcp.round(6).tolist(),
    tcp_error_m=round(tcp_error, 6),
)

close_gripper(arm=ARM)
settle_robot(steps=8)
open_gripper(arm=ARM)
settle_robot(steps=8)

final_eef = get_current_eef_pose(arm=ARM)
final_offset = np.asarray(get_tcp_offset_eef(arm=ARM), dtype=np.float64)
final_tcp = np.asarray(final_eef[0], dtype=np.float64) + quat_xyzw_to_matrix(final_eef[1]) @ final_offset
final_tcp_error = float(np.linalg.norm(final_tcp - target_pos))

record(
    "final",
    final_eef=pose_to_dict(final_eef),
    final_tcp_position=final_tcp.round(6).tolist(),
    final_tcp_error_m=round(final_tcp_error, 6),
    gripper=get_gripper_state(arm=ARM),
)

RESULT = {{
    "trace": trace,
    "move_ok": bool(move_ok),
    "target_tcp_position": target_pos.round(6).tolist(),
    "reached_tcp_position": reached_tcp.round(6).tolist(),
    "tcp_error_m": round(tcp_error, 6),
    "final_tcp_position": final_tcp.round(6).tolist(),
    "final_tcp_error_m": round(final_tcp_error, 6),
}}
"""


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class X2TabletopTcpReachCodeEnv(CodeExecutionEnvBase):
    """Minimal X2 task: move the right TCP to a fixed tabletop target marker."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE

    def _target_position(self) -> np.ndarray | None:
        try:
            obj = self.low_level_env.env.scene.object_registry("name", TARGET_NAME)
            if obj is None:
                return None
            if hasattr(obj, "aabb_center"):
                return _as_numpy(obj.aabb_center).astype(np.float64).reshape(3)
            pos, _quat = obj.get_position_orientation()
            return _as_numpy(pos).astype(np.float64).reshape(3)
        except Exception:
            return None

    def _current_tcp_position(self) -> np.ndarray | None:
        try:
            eef_pos, eef_quat = self.low_level_env.get_robot_eef_pose(arm=ARM)
            gripper_state = self.low_level_env.get_gripper_state(arm=ARM)
            offset = np.asarray(gripper_state.get("finger_center_eef") or [0.0, 0.0, 0.1046], dtype=np.float64)
            return _as_numpy(eef_pos).astype(np.float64).reshape(3) + _quat_xyzw_to_matrix(
                _as_numpy(eef_quat).astype(np.float64).reshape(4)
            ) @ offset.reshape(3)
        except Exception:
            return None

    def tcp_target_error(self) -> float | None:
        target = self._target_position()
        tcp = self._current_tcp_position()
        if target is None or tcp is None:
            return None
        return float(np.linalg.norm(tcp - target))

    def compute_reward(self) -> float:
        err = self.tcp_target_error()
        return 1.0 if err is not None and err <= SUCCESS_THRESHOLD_M else 0.0

    def step(self, action: str):
        obs, reward, terminated, truncated, info = super().step(action)
        err = self.tcp_target_error()
        info["tcp_target_error_m"] = err
        info["task_completed"] = bool(err is not None and err <= SUCCESS_THRESHOLD_M)
        return obs, reward, terminated, truncated, info


__all__ = [
    "X2TabletopTcpReachCodeEnv",
]
