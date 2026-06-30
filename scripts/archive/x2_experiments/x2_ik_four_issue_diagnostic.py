"""Separate the four likely failure modes for X2 orientation IK."""

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


def _rounded(value: Any, digits: int = 6) -> list[float]:
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


def _quat_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(_quat_normalize(a), _quat_normalize(b))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _mat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return _quat_normalize(
            np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[1, 0] - rot[0, 1]) / s,
                    0.25 * s,
                ]
            )
        )

    idx = int(np.argmax(np.diag(rot)))
    if idx == 0:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        q = [(0.25 * s), (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[2, 1] - rot[1, 2]) / s]
    elif idx == 1:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        q = [(rot[0, 1] + rot[1, 0]) / s, (0.25 * s), (rot[1, 2] + rot[2, 1]) / s, (rot[0, 2] - rot[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        q = [(rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, (0.25 * s), (rot[1, 0] - rot[0, 1]) / s]
    return _quat_normalize(np.asarray(q, dtype=np.float64))


def _restore_robot(robot, root_pose, q, qd) -> None:
    robot.set_position_orientation(position=root_pose[0], orientation=root_pose[1])
    robot.set_joint_positions(q, drive=False)
    robot.set_joint_velocities(qd, drive=False)


def _controller_group(env: X2BehaviourLowLevel, arm_name: str):
    from omnigibson.controllers.controller_view import ControllerView

    group_key, controller_idx = env.robot.controllers[f"arm_{arm_name}"]
    return group_key, controller_idx, ControllerView._controller_groups[group_key]


def _joint_order_debug(env: X2BehaviourLowLevel, controller) -> dict[str, Any]:
    from omnigibson.utils.usd_utils import ControllableObjectViewAPI, get_robot_kinematic_tree_pattern

    view_api = ControllableObjectViewAPI._VIEWS_BY_PATTERN[
        get_robot_kinematic_tree_pattern(controller.routing_path)
    ]
    view = view_api._view
    meta = view.get_metatype(0)
    debug: dict[str, Any] = {
        "robot_joint_names": list(env.robot.joints.keys()),
        "view_type": type(view).__name__,
        "metatype_type": type(meta).__name__,
    }
    for attr in ("dof_names", "joint_names", "dof_paths", "joint_paths", "link_names"):
        try:
            value = getattr(meta, attr)
        except Exception as exc:  # noqa: BLE001 - best-effort diagnostics
            debug[attr] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            debug[attr] = list(value)
        except TypeError:
            debug[attr] = repr(value)
    return debug


def _eef_rel_pose(env: X2BehaviourLowLevel, arm_name: str):
    return env.controller._world_pose_to_robot_pose(
        (env.robot.get_eef_position(arm_name), env.robot.get_eef_orientation(arm_name))
    )


def _build_action_and_target(env: X2BehaviourLowLevel, arm_name: str, target_pos, target_quat):
    import omnigibson.utils.transform_utils as T

    target_rel_pos, target_rel_quat = env.controller._world_pose_to_robot_pose(
        (
            torch.tensor(_as_np(target_pos), dtype=torch.float32),
            torch.tensor(_as_np(target_quat), dtype=torch.float32),
        )
    )
    env.controller._arm_targets[f"arm_{arm_name}"] = (target_rel_pos, T.quat2axisangle(target_rel_quat))
    action = env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=True))
    action_idx = env.robot.controller_action_idx[f"arm_{arm_name}"]
    partial_action = action[action_idx]
    return action, partial_action, target_rel_pos, target_rel_quat


def _analyze_case(env: X2BehaviourLowLevel, arm_idx: int, axis_name: str, axis: np.ndarray, angle_rad: float) -> dict:
    import omnigibson.utils.transform_utils as T
    from omnigibson.controllers.controller_view import ControllerView
    from omnigibson.utils.backend_utils import _compute_backend as cb
    from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

    arm_name = env._arm_name(arm_idx)
    root_pose = env.robot.get_position_orientation()
    q0_all = env.robot.get_joint_positions().clone()
    qd0_all = env.robot.get_joint_velocities().clone()
    start_pos, start_quat = env.get_robot_eef_pose(arm=arm_name)
    start_pos_np = _as_np(start_pos).astype(np.float64)
    start_quat_np = _quat_normalize(_as_np(start_quat).astype(np.float64))
    target_quat = _quat_multiply_xyzw(start_quat_np, _axis_angle_quat_xyzw(axis, angle_rad))

    group_key, controller_idx, controller = _controller_group(env, arm_name)
    dof_idx = _as_np(ControllerView.get_dof_idx(group_key)).astype(int)

    action, partial_action, target_rel_pos, target_rel_quat = _build_action_and_target(
        env, arm_name, start_pos_np, target_quat
    )

    preprocessed = controller._preprocess_command(cb.from_torch(partial_action))
    goal = controller._update_goal(controller_idx, preprocessed)
    reconstructed_quat = _mat_to_quat_xyzw(_as_np(cb.to_torch(goal["target_ori_mat"])))
    target_rel_quat_np = _quat_normalize(_as_np(target_rel_quat))
    current_rel_pos, current_rel_quat = _eef_rel_pose(env, arm_name)
    current_rel_quat_np = _quat_normalize(_as_np(current_rel_quat))

    controller.update_goal(controller_idx, cb.from_torch(partial_action))
    control_batch = controller.compute_control(controller._goals)
    control_batch = controller.clip_control(control_batch)
    target_q = _as_np(cb.to_torch(control_batch[controller_idx])).astype(np.float64)
    q0 = _as_np(q0_all[dof_idx]).astype(np.float64)
    delta_q = target_q - q0

    # Reproduce the controller's task Jacobian after its pose-link point correction.
    all_q = cb.to_torch(__import__("omnigibson.utils.usd_utils", fromlist=["ControllableObjectViewAPI"]).ControllableObjectViewAPI.get_all_joint_positions(controller.routing_path))
    jac_all = __import__("omnigibson.utils.usd_utils", fromlist=["ControllableObjectViewAPI"]).ControllableObjectViewAPI.get_all_relative_jacobians(controller.routing_path)
    rows = controller.view_row_indices
    row = int(_as_np(rows)[controller_idx])
    jac_link = controller._link_name
    pose_link = controller._pose_link_name
    api = __import__("omnigibson.utils.usd_utils", fromlist=["ControllableObjectViewAPI"]).ControllableObjectViewAPI
    jac_body_idx = int(api.get_link_index(controller.routing_path, jac_link))
    jac_row = jac_body_idx - 1
    jac_col_offset = int(jac_all.shape[-1] - all_q.shape[-1])
    jac_dof_idx = dof_idx + jac_col_offset
    j_eef = _as_np(jac_all[row, jac_row, :, :][:, jac_dof_idx]).astype(np.float64)
    if jac_link != pose_link:
        pose_pos_all, _ = api.get_all_link_relative_position_orientation(controller.routing_path, pose_link)
        jac_pos_all, _ = api.get_all_link_relative_position_orientation(controller.routing_path, jac_link)
        r = _as_np(pose_pos_all[row] - jac_pos_all[row]).astype(np.float64)
        j_rot = j_eef[3:, :].copy()
        j_eef[0, :] += r[2] * j_rot[1, :] - r[1] * j_rot[2, :]
        j_eef[1, :] += -r[2] * j_rot[0, :] + r[0] * j_rot[2, :]
        j_eef[2, :] += r[1] * j_rot[0, :] - r[0] * j_rot[1, :]

    ee_pos_np = _as_np(current_rel_pos).astype(np.float64)
    goal_pos_np = _as_np(cb.to_torch(goal["target_pos"])).astype(np.float64)
    ee_mat = _as_np(T.quat2mat(current_rel_quat)).astype(np.float64)
    goal_mat = _as_np(cb.to_torch(goal["target_ori_mat"])).astype(np.float64)
    pos_err = goal_pos_np - ee_pos_np
    ori_err = _as_np(T.orientation_error(torch.tensor(goal_mat[None], dtype=torch.float32), torch.tensor(ee_mat[None], dtype=torch.float32))[0]).astype(np.float64)
    err = np.concatenate([pos_err, ori_err])
    predicted_task_step = j_eef @ delta_q
    singular_values = np.linalg.svd(j_eef, compute_uv=False)
    cond = float(singular_values[0] / max(singular_values[-1], 1e-12))

    q_lower = _as_np(cb.to_torch(controller._q_lower[0])).astype(np.float64)
    q_upper = _as_np(cb.to_torch(controller._q_upper[0])).astype(np.float64)
    clipped_joints = np.where((np.isclose(target_q, q_lower, atol=1e-6)) | (np.isclose(target_q, q_upper, atol=1e-6)))[0]

    collision_before = bool(detect_robot_collision_in_sim(env.robot))
    env.step(action)
    collision_after = bool(detect_robot_collision_in_sim(env.robot))
    q1_all = env.robot.get_joint_positions().clone()
    q1 = _as_np(q1_all[dof_idx]).astype(np.float64)
    actual_delta_q = q1 - q0
    after_pos, after_quat = _eef_rel_pose(env, arm_name)

    _restore_robot(env.robot, root_pose, q0_all, qd0_all)
    direct_fk = []
    for scale in (1.0, 0.5, 0.25, 0.1):
        q_test = q0_all.clone()
        q_test[dof_idx] = torch.tensor(q0 + scale * delta_q, dtype=q_test.dtype, device=q_test.device)
        env.robot.set_joint_positions(q_test, drive=False)
        fk_pos, fk_quat = _eef_rel_pose(env, arm_name)
        direct_fk.append(
            {
                "scale": scale,
                "pos_error_m": round(float(np.linalg.norm(_as_np(fk_pos) - goal_pos_np)), 6),
                "ori_error_rad": round(float(T.get_orientation_diff_in_radian(fk_quat, target_rel_quat)), 6),
            }
        )

    result = {
        "arm": arm_name,
        "axis": axis_name,
        "target_angle_rad": round(float(angle_rad), 6),
        "controller_mode": controller.mode,
        "jacobian_link_name": jac_link,
        "pose_link_name": pose_link,
        "joint_order_debug": _joint_order_debug(env, controller),
        "arm_dof_idx": [int(i) for i in dof_idx],
        "arm_joint_names_from_robot_order": [list(env.robot.joints.keys())[int(i)] for i in dof_idx],
        "partial_action": _rounded(partial_action),
        "preprocessed_action": _rounded(cb.to_torch(preprocessed)),
        "action_passthrough_max_abs_diff": round(float(np.max(np.abs(_as_np(partial_action) - _as_np(cb.to_torch(preprocessed))))), 9),
        "issue_1_error_frame_direction": {
            "initial_ori_error_rad": round(float(T.get_orientation_diff_in_radian(current_rel_quat, target_rel_quat)), 6),
            "primitive_ori_cmd_norm": round(float(np.linalg.norm(_as_np(partial_action)[3:])), 6),
            "ik_ori_err_norm": round(float(np.linalg.norm(ori_err)), 6),
            "reconstructed_target_error_rad": round(float(_quat_angle_error(reconstructed_quat, target_rel_quat_np)), 9),
            "ori_err_dot_action_ori": round(float(np.dot(ori_err, _as_np(partial_action)[3:])), 9),
        },
        "issue_2_step_size_pinv": {
            "singular_values": _rounded(singular_values),
            "condition_number": round(cond, 6),
            "delta_q_norm": round(float(np.linalg.norm(delta_q)), 6),
            "delta_q_max_abs": round(float(np.max(np.abs(delta_q))), 6),
            "err_norm": round(float(np.linalg.norm(err)), 6),
            "predicted_task_step_norm": round(float(np.linalg.norm(predicted_task_step)), 6),
            "predicted_residual_norm": round(float(np.linalg.norm(err - predicted_task_step)), 6),
            "predicted_ori_step_dot_err": round(float(np.dot(predicted_task_step[3:], ori_err)), 9),
            "direct_fk_scaled_steps": direct_fk,
        },
        "issue_3_execution_semantics": {
            "commanded_delta_q": _rounded(delta_q),
            "actual_delta_q_one_env_step": _rounded(actual_delta_q),
            "actual_to_commanded_delta_ratio": round(float(np.linalg.norm(actual_delta_q) / max(np.linalg.norm(delta_q), 1e-12)), 6),
            "actual_delta_dot_commanded": round(float(np.dot(actual_delta_q, delta_q)), 9),
            "after_one_step_pos_error_m": round(float(np.linalg.norm(_as_np(after_pos) - goal_pos_np)), 6),
            "after_one_step_ori_error_rad": round(float(T.get_orientation_diff_in_radian(after_quat, target_rel_quat)), 6),
        },
        "issue_4_limits_collision": {
            "q_margin_lower_min": round(float(np.min(q0 - q_lower)), 6),
            "q_margin_upper_min": round(float(np.min(q_upper - q0)), 6),
            "target_margin_lower_min": round(float(np.min(target_q - q_lower)), 6),
            "target_margin_upper_min": round(float(np.min(q_upper - target_q)), 6),
            "clipped_joint_local_indices": [int(i) for i in clipped_joints],
            "collision_before": collision_before,
            "collision_after_one_step": collision_after,
        },
    }

    _restore_robot(env.robot, root_pose, q0_all, qd0_all)
    env.controller._arm_targets.pop(f"arm_{arm_name}", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/x2_ik_four_issue_diagnostic"))
    parser.add_argument("--angle-deg", type=float, default=10.0)
    parser.add_argument("--axes", nargs="*", default=["local_x", "local_y", "local_z"])
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # This diagnostic only needs physics and kinematics. Avoid constructing the
    # default viewer camera, which can trip Vulkan in headless runs.
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm

    gm.RENDER_VIEWER_CAMERA = False

    axes = {
        "local_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "local_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "local_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    summary: dict[str, Any] = {"ok": False, "cases": []}
    try:
        env = X2BehaviourLowLevel(
            objects=[],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_obs_modalities=[],
        )
        env.reset()
        env.settle_robot_steps(steps=24)
        for arm_idx in (0, 1):
            for axis_name in args.axes:
                summary["cases"].append(
                    _analyze_case(env, arm_idx, axis_name, axes[axis_name], math.radians(args.angle_deg))
                )
        summary["ok"] = True
    except Exception:
        summary["exception"] = traceback.format_exc()
        print(summary["exception"])

    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "output": str(output_path)}, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
