"""Diagnose X2 orientation IK separately from position IK."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _list(value: Any, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in _as_np(value).reshape(-1)]


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = _quat_normalize(a)
    bx, by, bz, bw = _quat_normalize(b)
    return _quat_normalize(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )
    )


def _axis_angle_quat_xyzw(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * float(angle_rad)
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


def _quat_to_mat_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _quat_normalize(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_from_mat(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    denom = 2.0 * math.sin(angle)
    axis = np.array(
        [
            (rot[2, 1] - rot[1, 2]) / denom,
            (rot[0, 2] - rot[2, 0]) / denom,
            (rot[1, 0] - rot[0, 1]) / denom,
        ],
        dtype=np.float64,
    )
    return axis * angle


def _quat_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(_quat_normalize(a), _quat_normalize(b))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _restore_robot(robot, root_pose, q, qd) -> None:
    robot.set_position_orientation(position=root_pose[0], orientation=root_pose[1])
    robot.set_joint_positions(q, drive=False)
    robot.set_joint_velocities(qd, drive=False)


def _finite_difference_rot_jacobian(robot, arm: str, dof_idx: np.ndarray, eps: float) -> np.ndarray:
    root_pose = robot.get_position_orientation()
    q0 = robot.get_joint_positions().clone()
    qd0 = robot.get_joint_velocities().clone()
    fd = np.zeros((3, len(dof_idx)), dtype=np.float64)

    for col, dof in enumerate(dof_idx):
        q_plus = q0.clone()
        q_minus = q0.clone()
        q_plus[int(dof)] += eps
        q_minus[int(dof)] -= eps

        _restore_robot(robot, root_pose, q_plus, qd0)
        _, quat_plus = robot.get_relative_eef_pose(arm=arm)
        _restore_robot(robot, root_pose, q_minus, qd0)
        _, quat_minus = robot.get_relative_eef_pose(arm=arm)

        r_plus = _quat_to_mat_xyzw(_as_np(quat_plus))
        r_minus = _quat_to_mat_xyzw(_as_np(quat_minus))
        fd[:, col] = _axis_angle_from_mat(r_plus @ r_minus.T) / (2.0 * eps)

    _restore_robot(robot, root_pose, q0, qd0)
    return fd


def _controller_rot_jacobian(robot, arm: str, dof_idx: np.ndarray) -> np.ndarray:
    from omnigibson.utils.usd_utils import ControllableObjectViewAPI

    routing_path = robot.articulation_root_path
    row = int(ControllableObjectViewAPI.get_member_view_indices(routing_path, [routing_path])[0])
    all_q = ControllableObjectViewAPI.get_all_joint_positions(routing_path)
    jac_all = ControllableObjectViewAPI.get_all_relative_jacobians(routing_path)
    jac_link_name = robot.jacobian_link_names[arm]
    jac_body_idx = ControllableObjectViewAPI.get_link_index(routing_path, jac_link_name)
    jac_row = int(jac_body_idx) - 1
    jac_col_offset = int(jac_all.shape[-1] - all_q.shape[-1])
    jac_dof_idx = dof_idx + jac_col_offset
    return _as_np(jac_all[row, jac_row, 3:, :][:, jac_dof_idx]).astype(np.float64)


def _run_orientation_tracking(
    env: X2BehaviourLowLevel,
    arm_idx: int,
    axis_name: str,
    axis: np.ndarray,
    angle_rad: float,
    max_steps: int,
    ori_thresh: float,
) -> dict[str, Any]:
    import omnigibson.utils.transform_utils as T

    env._set_arm(arm_idx)
    arm_name = env._arm_name(arm_idx)
    start_pos, start_quat = env.get_robot_eef_pose(arm=arm_name)
    start_pos_np = _as_np(start_pos).astype(np.float64)
    start_quat_np = _quat_normalize(_as_np(start_quat).astype(np.float64))
    target_quat = _quat_multiply_xyzw(start_quat_np, _axis_angle_quat_xyzw(axis, angle_rad))
    target_pose = (
        torch.tensor(start_pos_np, dtype=torch.float32),
        torch.tensor(target_quat, dtype=torch.float32),
    )
    target_rel_pos, target_rel_quat = env.controller._world_pose_to_robot_pose(target_pose)

    trace = []
    ok = True
    error_message = None
    steps = 0
    try:
        gen = env._move_hand_direct_ik_with_stuck_patience(
            target_pose,
            pos_thresh=0.008,
            ori_thresh=ori_thresh,
            stop_if_stuck=True,
            stuck_patience_steps=20,
            stuck_pos_thresh=0.0001,
            stuck_ori_thresh=0.001,
            max_steps=max_steps,
        )
        for steps, action in enumerate(gen, start=1):
            if action is not None:
                env.step(action)
            current_rel_pos, current_rel_quat = env.controller._world_pose_to_robot_pose(
                (env.robot.get_eef_position(arm_name), env.robot.get_eef_orientation(arm_name))
            )
            pos_err = float(torch.linalg.norm(target_rel_pos - current_rel_pos))
            ori_err = float(T.get_orientation_diff_in_radian(current_rel_quat, target_rel_quat))
            if steps <= 10 or steps % 10 == 0:
                trace.append({"step": steps, "pos_error_m": round(pos_err, 6), "ori_error_rad": round(ori_err, 6)})
    except Exception as exc:  # noqa: BLE001 - diagnostic script records failures
        ok = False
        error_message = f"{type(exc).__name__}: {exc}"

    final_pos, final_quat = env.get_robot_eef_pose(arm=arm_name)
    final_pos_np = _as_np(final_pos).astype(np.float64)
    final_quat_np = _quat_normalize(_as_np(final_quat).astype(np.float64))
    return {
        "arm": arm_name,
        "axis": axis_name,
        "target_angle_rad": round(float(angle_rad), 6),
        "ok": ok,
        "error_message": error_message,
        "steps": int(steps),
        "start_pos": _list(start_pos_np),
        "final_pos": _list(final_pos_np),
        "position_drift_m": round(float(np.linalg.norm(final_pos_np - start_pos_np)), 6),
        "target_quat_xyzw": _list(target_quat),
        "final_quat_xyzw": _list(final_quat_np),
        "actual_angle_delta_rad": round(float(_quat_angle_error(final_quat_np, start_quat_np)), 6),
        "final_ori_error_rad": round(float(_quat_angle_error(final_quat_np, target_quat)), 6),
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/x2_orientation_ik_diagnostic"))
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--angle-deg", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--ori-thresh", type=float, default=0.04)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The diagnostic only needs physics / kinematics; avoid constructing the
    # default viewer camera in headless runs.
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm

    gm.RENDER_VIEWER_CAMERA = False

    summary: dict[str, Any] = {"ok": False, "rot_jacobian": [], "tracking": []}
    try:
        env = X2BehaviourLowLevel(
            objects=[],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_obs_modalities=[],
        )
        env.reset()
        env.settle_robot_steps(steps=24)

        for arm_idx in (0, 1):
            arm_name = env._arm_name(arm_idx)
            dof_idx = _as_np(env.robot.arm_control_idx[arm_name]).astype(int)
            fd_rot = _finite_difference_rot_jacobian(env.robot, arm_name, dof_idx, args.eps)
            ctrl_rot = _controller_rot_jacobian(env.robot, arm_name, dof_idx)
            diff = ctrl_rot - fd_rot
            summary["rot_jacobian"].append(
                {
                    "arm": arm_name,
                    "dof_idx": dof_idx.tolist(),
                    "joint_names": [list(env.robot.joints.keys())[int(i)] for i in dof_idx],
                    "jacobian_link_name": env.robot.jacobian_link_names[arm_name],
                    "eef_link_name": env.robot.eef_link_names[arm_name],
                    "fd_rot_jacobian": np.round(fd_rot, 6).tolist(),
                    "controller_rot_jacobian": np.round(ctrl_rot, 6).tolist(),
                    "fro_error": round(float(np.linalg.norm(diff)), 6),
                    "max_abs_error": round(float(np.max(np.abs(diff))), 6),
                    "fd_norm": round(float(np.linalg.norm(fd_rot)), 6),
                    "controller_norm": round(float(np.linalg.norm(ctrl_rot)), 6),
                }
            )

            # Restore after finite differences before running tracking.
            env.settle_robot_steps(steps=12)
            for axis_name, axis in (
                ("local_x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                ("local_y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                ("local_z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
            ):
                summary["tracking"].append(
                    _run_orientation_tracking(
                        env,
                        arm_idx=arm_idx,
                        axis_name=axis_name,
                        axis=axis,
                        angle_rad=math.radians(args.angle_deg),
                        max_steps=args.max_steps,
                        ori_thresh=args.ori_thresh,
                    )
                )

        summary["ok"] = all(item["max_abs_error"] < 0.05 for item in summary["rot_jacobian"]) and all(
            item["ok"] and item["final_ori_error_rad"] < 0.08 for item in summary["tracking"]
        )
    except Exception:
        summary["exception"] = traceback.format_exc()
        print(summary["exception"])

    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "output": str(output_path)}, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
