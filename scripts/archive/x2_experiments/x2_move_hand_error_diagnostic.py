"""Diagnose X2 CAP-X move_hand residual error.

The CAP-X smoke test previously saw about 1.36 cm residual after a 5 cm +x
hand move. This script separates a loose success threshold from IK / dynamics
tracking issues by driving the same target with different direct-IK thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
from capx.integrations.x2.control import X2ControlApi


def _as_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value) -> list[float]:
    return [round(float(v), 6) for v in _as_np(value).reshape(-1)]


def _err(pos, target_pos) -> float:
    return float(np.linalg.norm(_as_np(pos) - _as_np(target_pos)))


def _eef_np(env: X2BehaviourLowLevel, arm: int) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = env.get_robot_eef_pose(arm=arm)
    return _as_np(pos).copy(), _as_np(quat).copy()


def _step_with_controller_action(env: X2BehaviourLowLevel, action) -> None:
    if action is not None:
        env.step(action)


def _run_direct_ik(
    env: X2BehaviourLowLevel,
    target_pose: tuple[np.ndarray, np.ndarray],
    arm: int,
    *,
    label: str,
    pos_thresh: float,
    ori_thresh: float,
    stop_if_stuck: bool,
) -> dict[str, Any]:
    env._set_arm(arm)
    target_pos, target_quat = target_pose
    target_t = (
        torch.tensor(target_pos, dtype=torch.float32),
        torch.tensor(target_quat, dtype=torch.float32),
    )
    trace: list[dict[str, Any]] = []
    steps = 0
    ok = True
    error: str | None = None

    try:
        gen = env.controller._move_hand_direct_ik(
            target_t,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            stop_if_stuck=stop_if_stuck,
        )
        for steps, action in enumerate(gen, start=1):
            _step_with_controller_action(env, action)
            pos, _quat = _eef_np(env, arm)
            if steps <= 10 or steps % 10 == 0:
                trace.append(
                    {
                        "step": steps,
                        "pos": _as_list(pos),
                        "error": round(_err(pos, target_pos), 6),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - record diagnostic failure details
        ok = False
        error = f"{type(exc).__name__}: {exc}"

    final_pos, final_quat = _eef_np(env, arm)
    return {
        "label": label,
        "ok": ok,
        "error_message": error,
        "pos_thresh": pos_thresh,
        "ori_thresh": ori_thresh,
        "stop_if_stuck": stop_if_stuck,
        "steps": steps,
        "final_pos": _as_list(final_pos),
        "final_quat": _as_list(final_quat),
        "final_error": round(_err(final_pos, target_pos), 6),
        "trace": trace,
    }


def _run_empty_followup(
    env: X2BehaviourLowLevel,
    target_pos: np.ndarray,
    arm: int,
    *,
    steps: int,
    label: str,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    for i in range(1, steps + 1):
        action = env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=True))
        env.step(action)
        if i <= 10 or i % 10 == 0:
            pos, _quat = _eef_np(env, arm)
            trace.append({"step": i, "pos": _as_list(pos), "error": round(_err(pos, target_pos), 6)})

    final_pos, final_quat = _eef_np(env, arm)
    return {
        "label": label,
        "steps": steps,
        "final_pos": _as_list(final_pos),
        "final_quat": _as_list(final_quat),
        "final_error": round(_err(final_pos, target_pos), 6),
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int, default=1, help="Arm index: 0 left, 1 right")
    parser.add_argument("--move-distance", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/x2_move_hand_error_diagnostic"))
    parser.add_argument("--empty-followup-steps", type=int, default=80)
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "ok": False,
        "arm": args.arm,
        "move_distance": args.move_distance,
        "phases": [],
    }

    try:
        env = X2BehaviourLowLevel(
            objects=[],
            load_object_categories=["floors", "ceilings", "walls"],
        )
        api = X2ControlApi(env)
        initial_pos, initial_quat = api.get_current_eef_pose(arm=args.arm)
        target_pos = np.asarray(initial_pos, dtype=np.float32) + np.array([args.move_distance, 0.0, 0.0], dtype=np.float32)
        target_pose = (target_pos, np.asarray(initial_quat, dtype=np.float32))

        summary["initial_pos"] = _as_list(initial_pos)
        summary["initial_quat"] = _as_list(initial_quat)
        summary["target_pos"] = _as_list(target_pos)

        default_phase = _run_direct_ik(
            env,
            target_pose,
            args.arm,
            label="direct_ik_default_2cm_threshold",
            pos_thresh=0.02,
            ori_thresh=0.4,
            stop_if_stuck=True,
        )
        summary["phases"].append(default_phase)
        print(json.dumps(default_phase, indent=2))

        followup_phase = _run_empty_followup(
            env,
            target_pos,
            args.arm,
            steps=args.empty_followup_steps,
            label="empty_action_follow_existing_target",
        )
        summary["phases"].append(followup_phase)
        print(json.dumps(followup_phase, indent=2))

        strict_phase = _run_direct_ik(
            env,
            target_pose,
            args.arm,
            label="direct_ik_strict_5mm_threshold",
            pos_thresh=0.005,
            ori_thresh=0.1,
            stop_if_stuck=False,
        )
        summary["phases"].append(strict_phase)
        print(json.dumps(strict_phase, indent=2))

        final_err = strict_phase["final_error"]
        summary["ok"] = bool(default_phase["ok"] and strict_phase["ok"] and final_err < 0.006)
        summary["diagnosis"] = (
            "loose_threshold"
            if default_phase["final_error"] > 0.01 and strict_phase["final_error"] < 0.006
            else "needs_further_ik_or_dynamics_debug"
        )
    except Exception:
        summary["exception"] = traceback.format_exc()
        print(summary["exception"])

    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps({"ok": summary["ok"], "diagnosis": summary.get("diagnosis")}, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
