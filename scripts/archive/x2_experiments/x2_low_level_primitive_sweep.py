"""Sweep X2 low-level manipulation primitives over local reachable targets.

This script avoids visual models and generated code. It directly tests the
low-level X2 wrapper's arm IK, gripper, settle, and object-in-hand primitives.
"""

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


def _as_list(value: Any, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in _as_np(value).reshape(-1)]


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


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


def _pose_dict(pose: tuple[Any, Any]) -> dict[str, list[float]]:
    return {
        "position": _as_list(pose[0]),
        "quat_xyzw": _as_list(pose[1]),
    }


def _move_and_measure(
    env: X2BehaviourLowLevel,
    target_pose: tuple[np.ndarray, np.ndarray],
    *,
    arm: int,
    label: str,
    max_steps: int,
    pos_thresh: float,
    ori_thresh: float,
    stuck_patience_steps: int,
) -> dict[str, Any]:
    target_pos = np.asarray(target_pose[0], dtype=np.float64).reshape(3)
    target_quat = _quat_normalize(np.asarray(target_pose[1], dtype=np.float64).reshape(4))
    before = env.get_robot_eef_pose(arm=arm)
    before_pos = _as_np(before[0]).astype(np.float64).reshape(3)
    before_quat = _quat_normalize(_as_np(before[1]).astype(np.float64).reshape(4))
    target_t = (
        torch.tensor(target_pos, dtype=torch.float32),
        torch.tensor(target_quat, dtype=torch.float32),
    )

    trace: list[dict[str, Any]] = []
    steps = 0
    ok = True
    error_message = None
    try:
        env._set_arm(arm)
        gen = env._move_hand_direct_ik_with_stuck_patience(
            target_t,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            stop_if_stuck=True,
            stuck_patience_steps=stuck_patience_steps,
            stuck_pos_thresh=0.0001,
            stuck_ori_thresh=0.001,
            max_steps=max_steps,
        )
        for steps, action in enumerate(gen, start=1):
            if action is not None:
                env.step(action)
            if steps <= 8 or steps % 25 == 0:
                cur_pos, cur_quat = env.get_robot_eef_pose(arm=arm)
                cur_pos_np = _as_np(cur_pos).astype(np.float64).reshape(3)
                cur_quat_np = _quat_normalize(_as_np(cur_quat).astype(np.float64).reshape(4))
                trace.append(
                    {
                        "step": int(steps),
                        "pos_error_m": round(float(np.linalg.norm(cur_pos_np - target_pos)), 6),
                        "ori_error_rad": round(float(_quat_angle_error(cur_quat_np, target_quat)), 6),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - diagnostic records failures
        ok = False
        error_message = f"{type(exc).__name__}: {exc}"

    after = env.get_robot_eef_pose(arm=arm)
    after_pos = _as_np(after[0]).astype(np.float64).reshape(3)
    after_quat = _quat_normalize(_as_np(after[1]).astype(np.float64).reshape(4))
    pos_error = float(np.linalg.norm(after_pos - target_pos))
    ori_error = float(_quat_angle_error(after_quat, target_quat))
    return {
        "label": label,
        "arm": env._arm_name(arm),
        "ok": bool(ok),
        "error_message": error_message,
        "steps": int(steps),
        "target": {"position": _as_list(target_pos), "quat_xyzw": _as_list(target_quat)},
        "before": {"position": _as_list(before_pos), "quat_xyzw": _as_list(before_quat)},
        "after": {"position": _as_list(after_pos), "quat_xyzw": _as_list(after_quat)},
        "requested_delta_m": _as_list(target_pos - before_pos),
        "pos_error_m": round(pos_error, 6),
        "ori_error_rad": round(ori_error, 6),
        "pos_thresh": float(pos_thresh),
        "ori_thresh": float(ori_thresh),
        "passed": bool(ok and pos_error < 0.02 and ori_error < max(0.12, ori_thresh + 0.03)),
        "trace": trace,
    }


def _test_gripper(env: X2BehaviourLowLevel, arm: int) -> dict[str, Any]:
    env._set_arm(arm)
    before = env.get_gripper_state(arm=arm)
    open_steps = env._open_close_gripper(arm=arm, open=True)
    opened = env.get_gripper_state(arm=arm)
    env.settle_robot_steps(steps=8)
    close_steps = env._open_close_gripper(arm=arm, open=False)
    closed = env.get_gripper_state(arm=arm)
    env.settle_robot_steps(steps=8)
    reopen_steps = env._open_close_gripper(arm=arm, open=True)
    reopened = env.get_gripper_state(arm=arm)
    open_span = opened.get("finger_span_y_eef")
    closed_span = closed.get("finger_span_y_eef")
    reopened_span = reopened.get("finger_span_y_eef")
    span_ok = (
        isinstance(open_span, (int, float))
        and isinstance(closed_span, (int, float))
        and isinstance(reopened_span, (int, float))
        and closed_span < open_span
        and reopened_span > closed_span
    )
    return {
        "arm": env._arm_name(arm),
        "before": before,
        "opened": opened,
        "closed": closed,
        "reopened": reopened,
        "open_steps": int(open_steps),
        "close_steps": int(close_steps),
        "reopen_steps": int(reopen_steps),
        "passed": bool(span_ok),
    }


def _run_arm_sweep(
    env: X2BehaviourLowLevel,
    *,
    arm: int,
    max_steps: int,
    orientation_deg: float,
    pos_thresh: float,
    ori_thresh: float,
    stuck_patience_steps: int,
) -> dict[str, Any]:
    arm_name = env._arm_name(arm)
    env._set_arm(arm)
    env.settle_robot_steps(steps=24)
    home = env.get_robot_eef_pose(arm=arm)
    home_pos = _as_np(home[0]).astype(np.float64).reshape(3)
    home_quat = _quat_normalize(_as_np(home[1]).astype(np.float64).reshape(4))

    position_deltas = [
        ("x_plus_4cm", [0.04, 0.0, 0.0]),
        ("x_minus_3cm", [-0.03, 0.0, 0.0]),
        ("y_plus_4cm", [0.0, 0.04, 0.0]),
        ("y_minus_4cm", [0.0, -0.04, 0.0]),
        ("z_plus_4cm", [0.0, 0.0, 0.04]),
        ("z_minus_3cm", [0.0, 0.0, -0.03]),
        ("diag_forward_up", [0.035, 0.025, 0.025]),
        ("diag_side_down", [0.025, -0.035, -0.02]),
        ("diag_back_side", [-0.025, 0.03, 0.015]),
    ]
    angle = math.radians(float(orientation_deg))
    orientation_targets = [
        ("roll_plus", [1.0, 0.0, 0.0], angle),
        ("roll_minus", [1.0, 0.0, 0.0], -angle),
        ("pitch_plus", [0.0, 1.0, 0.0], angle),
        ("pitch_minus", [0.0, 1.0, 0.0], -angle),
        ("yaw_plus", [0.0, 0.0, 1.0], angle),
        ("yaw_minus", [0.0, 0.0, 1.0], -angle),
    ]

    results: list[dict[str, Any]] = []
    for name, delta in position_deltas:
        delta_np = np.asarray(delta, dtype=np.float64)
        results.append(
            _move_and_measure(
                env,
                (home_pos + delta_np, home_quat),
                arm=arm,
                label=f"{arm_name}_{name}",
                max_steps=max_steps,
                pos_thresh=pos_thresh,
                ori_thresh=ori_thresh,
                stuck_patience_steps=stuck_patience_steps,
            )
        )
        results.append(
            _move_and_measure(
                env,
                (home_pos, home_quat),
                arm=arm,
                label=f"{arm_name}_return_home_after_{name}",
                max_steps=max_steps,
                pos_thresh=pos_thresh,
                ori_thresh=ori_thresh,
                stuck_patience_steps=stuck_patience_steps,
            )
        )

    for name, axis, delta_angle in orientation_targets:
        q_delta = _axis_angle_quat_xyzw(np.asarray(axis, dtype=np.float64), delta_angle)
        target_quat = _quat_multiply_xyzw(home_quat, q_delta)
        results.append(
            _move_and_measure(
                env,
                (home_pos, target_quat),
                arm=arm,
                label=f"{arm_name}_{name}_{orientation_deg:g}deg",
                max_steps=max_steps,
                pos_thresh=0.008,
                ori_thresh=0.04,
                stuck_patience_steps=stuck_patience_steps,
            )
        )
        results.append(
            _move_and_measure(
                env,
                (home_pos, home_quat),
                arm=arm,
                label=f"{arm_name}_return_home_after_{name}",
                max_steps=max_steps,
                pos_thresh=pos_thresh,
                ori_thresh=ori_thresh,
                stuck_patience_steps=stuck_patience_steps,
            )
        )

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    return {
        "arm": arm_name,
        "home": {"position": _as_list(home_pos), "quat_xyzw": _as_list(home_quat)},
        "gripper": _test_gripper(env, arm),
        "tests": results,
        "summary": {
            "count": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "failed_labels": [r["label"] for r in failed],
            "max_pos_error_m": round(max(float(r["pos_error_m"]) for r in results), 6),
            "max_ori_error_rad": round(max(float(r["ori_error_rad"]) for r in results), 6),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="X2 low-level primitive reachable sweep")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--arms", default="0,1", help="Comma-separated arm indices, e.g. 1 or 0,1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/x2_low_level_primitive_sweep"))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--orientation-deg", type=float, default=10.0)
    parser.add_argument("--pos-thresh", type=float, default=0.006)
    parser.add_argument("--ori-thresh", type=float, default=0.12)
    parser.add_argument("--stuck-patience-steps", type=int, default=45)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        arms = [int(x.strip()) for x in args.arms.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --arms value {args.arms!r}: {exc}") from exc

    report: dict[str, Any] = {
        "ok": False,
        "config": args.config,
        "arms": arms,
        "arm_results": [],
    }
    try:
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=[],
            load_object_categories=["floors", "ceilings", "walls"],
            robot_obs_modalities=[],
        )
        env.reset()
        env.settle_robot_steps(steps=24)
        report["initial_joint_positions"] = _as_list(env.get_joint_positions())
        report["initial_object_in_hand"] = {
            str(arm): bool(env.check_object_in_hand(arm=arm)) for arm in arms
        }
        for arm in arms:
            report["arm_results"].append(
                _run_arm_sweep(
                    env,
                    arm=arm,
                    max_steps=args.max_steps,
                    orientation_deg=args.orientation_deg,
                    pos_thresh=args.pos_thresh,
                    ori_thresh=args.ori_thresh,
                    stuck_patience_steps=args.stuck_patience_steps,
                )
            )
        report["diagnosis"] = {
            "total_tests": sum(r["summary"]["count"] for r in report["arm_results"]),
            "passed_tests": sum(r["summary"]["passed"] for r in report["arm_results"]),
            "failed_tests": sum(r["summary"]["failed"] for r in report["arm_results"]),
            "failed_labels": [
                label
                for arm_result in report["arm_results"]
                for label in arm_result["summary"]["failed_labels"]
            ],
            "gripper_failed_arms": [
                r["arm"] for r in report["arm_results"] if not r["gripper"]["passed"]
            ],
            "max_pos_error_m": max(r["summary"]["max_pos_error_m"] for r in report["arm_results"]),
            "max_ori_error_rad": max(r["summary"]["max_ori_error_rad"] for r in report["arm_results"]),
        }
        report["ok"] = (
            report["diagnosis"]["failed_tests"] == 0
            and len(report["diagnosis"]["gripper_failed_arms"]) == 0
        )
    except Exception:
        report["exception"] = traceback.format_exc()
        print(report["exception"])

    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("diagnosis", {}), indent=2))
    print(f"Wrote {output_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
