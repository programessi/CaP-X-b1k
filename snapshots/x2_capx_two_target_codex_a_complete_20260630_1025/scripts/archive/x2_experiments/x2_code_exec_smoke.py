"""Layer-3 smoke test for CAP-X X2 generated-code execution.

This does not call X2ControlApi methods directly. Instead it creates a CAP-X
CodeExecutionEnv, injects X2 APIs into the exec namespace, and executes fixed
Python code through env.step(code), matching the path used for LLM-generated
programs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import capx.envs.simulators  # noqa: F401 - registers low-level envs
import capx.envs.tasks  # noqa: F401 - registers code-execution envs/configs
import capx.integrations  # noqa: F401 - registers APIs
from capx.envs.tasks import CodeExecEnvConfig, get_config, get_exec_env, list_configs, list_exec_envs
from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel

OBJECT_NAME = "cologne"


FIXED_CODE_STEP1 = r'''
import numpy as np

required = [
    "get_env_observation",
    "get_object_pose",
    "get_current_joint_positions",
    "get_current_eef_pose",
    "get_robot_relative_eef_pose",
    "move_hand",
    "open_gripper",
    "close_gripper",
    "sample_grasp_pose",
    "grasp_object",
    "check_object_in_hand",
    "lift_arm",
]

injected = {name: callable(globals().get(name)) for name in required}

obs = get_env_observation()
joints = get_current_joint_positions()
initial_pos, initial_quat = get_current_eef_pose(arm=1)
rel_pos, rel_quat = get_robot_relative_eef_pose(arm=1)

open_gripper(arm=1)
target_pos = np.asarray(initial_pos, dtype=np.float32) + np.array([0.05, 0.0, 0.0], dtype=np.float32)
move_ok = move_hand((target_pos, initial_quat), arm=1)
moved_pos, moved_quat = get_current_eef_pose(arm=1)
move_error = float(np.linalg.norm(np.asarray(moved_pos) - target_pos))
close_gripper(arm=1)

RESULT = {
    "step": "arm_motion",
    "all_required_injected": all(injected.values()),
    "injected": injected,
    "obs_type": type(obs).__name__,
    "obs_top_level_keys": sorted(list(obs.keys())),
    "joint_count": int(len(joints)),
    "initial_pos": np.asarray(initial_pos).round(6).tolist(),
    "target_pos": target_pos.round(6).tolist(),
    "moved_pos": np.asarray(moved_pos).round(6).tolist(),
    "move_ok": bool(move_ok),
    "move_error": round(move_error, 6),
    "rel_pos": np.asarray(rel_pos).round(6).tolist(),
}
'''


FIXED_CODE_STEP2 = r'''
import numpy as np

obj_pos, obj_quat = get_object_pose("cologne")
pregrasp_pose, grasp_pose = sample_grasp_pose("cologne", arm=1)
lift_ok = lift_arm(arm=1)
post_lift_pos, _ = get_current_eef_pose(arm=1)
in_hand = check_object_in_hand(arm=1)

RESULT = {
    "step": "object_sampling_and_pregrasp",
    "object_pos": np.asarray(obj_pos).round(6).tolist(),
    "sample_pregrasp_pos": np.asarray(pregrasp_pose[0]).round(6).tolist(),
    "sample_grasp_pos": np.asarray(grasp_pose[0]).round(6).tolist(),
    "lift_ok": bool(lift_ok),
    "post_lift_pos": np.asarray(post_lift_pos).round(6).tolist(),
    "object_in_hand": bool(in_hand),
    "object_in_hand_type": type(in_hand).__name__,
}
'''


def _as_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_list(value) -> list[float]:
    return [round(float(v), 6) for v in _as_np(value).reshape(-1)]


def _place_object_near_right_eef(env: X2BehaviourLowLevel) -> dict[str, Any]:
    pos, _quat = env.get_robot_eef_pose(arm=1)
    placed_pos = pos + torch.tensor([0.09, 0.0, -0.05], dtype=pos.dtype, device=pos.device)
    obj = env.env.scene.object_registry("name", OBJECT_NAME)
    if obj is None:
        raise ValueError(f"Object {OBJECT_NAME!r} not found in scene registry")
    obj.set_position_orientation(
        position=placed_pos,
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device),
    )
    obj.keep_still()
    env.env.step(env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=False)))
    actual_pos, actual_quat = obj.get_position_orientation()
    return {
        "requested_pos": _as_list(placed_pos),
        "actual_pos": _as_list(actual_pos),
        "actual_quat": _as_list(actual_quat),
    }


def _validate_step1_result(result: dict[str, Any] | None, info: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    checks: list[str] = []
    detail: dict[str, Any] = {"result": result, "sandbox_rc": info.get("sandbox_rc")}

    def check(condition: bool, message: str) -> None:
        checks.append("pass" if condition else f"fail: {message}")

    check(info.get("sandbox_rc") == 0, f"sandbox_rc={info.get('sandbox_rc')} stderr={info.get('stderr', '')[-500:]}")
    check(isinstance(result, dict), f"RESULT must be dict, got {type(result).__name__}")
    if not isinstance(result, dict):
        return False, detail, checks

    injected = result.get("injected", {})
    missing = [name for name, ok in injected.items() if not ok]
    check(result.get("all_required_injected") is True, f"missing injected callables: {missing}")
    check(result.get("joint_count", 0) >= 6, f"joint_count={result.get('joint_count')} expected >=6")
    check(result.get("move_ok") is True, "move_hand returned False")
    check(result.get("move_error", 999.0) < 0.01, f"move_error={result.get('move_error')} expected <1cm")
    return all(v.startswith("pass") for v in checks), detail, checks


def _validate_step2_result(result: dict[str, Any] | None, info: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    checks: list[str] = []
    detail: dict[str, Any] = {"result": result, "sandbox_rc": info.get("sandbox_rc")}

    def check(condition: bool, message: str) -> None:
        checks.append("pass" if condition else f"fail: {message}")

    check(info.get("sandbox_rc") == 0, f"sandbox_rc={info.get('sandbox_rc')} stderr={info.get('stderr', '')[-500:]}")
    check(isinstance(result, dict), f"RESULT must be dict, got {type(result).__name__}")
    if not isinstance(result, dict):
        return False, detail, checks

    pregrasp = np.asarray(result.get("sample_pregrasp_pos", []), dtype=np.float32)
    grasp = np.asarray(result.get("sample_grasp_pos", []), dtype=np.float32)
    obj = np.asarray(result.get("object_pos", []), dtype=np.float32)
    sample_shapes_ok = pregrasp.shape == (3,) and grasp.shape == (3,) and obj.shape == (3,)
    pregrasp_grasp_dist = float(np.linalg.norm(pregrasp - grasp)) if sample_shapes_ok else 999.0
    grasp_obj_dist = float(np.linalg.norm(grasp - obj)) if sample_shapes_ok else 999.0

    detail["pregrasp_grasp_dist"] = round(pregrasp_grasp_dist, 6)
    detail["grasp_obj_dist"] = round(grasp_obj_dist, 6)
    check(sample_shapes_ok, "sample/object positions must all be 3D")
    check(0.01 < pregrasp_grasp_dist < 0.15, f"pregrasp_grasp_dist={pregrasp_grasp_dist} expected 1-15cm")
    check(grasp_obj_dist < 0.15, f"grasp_obj_dist={grasp_obj_dist} expected <15cm")
    check(result.get("lift_ok") is True, "lift_arm returned False")
    check(result.get("object_in_hand_type") == "bool", f"object_in_hand_type={result.get('object_in_hand_type')}")
    return all(v.startswith("pass") for v in checks), detail, checks


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-X X2 fixed-code execution smoke test")
    parser.add_argument("--output-dir", default="outputs/x2_code_exec_smoke")
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"

    summary: dict[str, Any] = {
        "ok": False,
        "fixed_code_step1": FIXED_CODE_STEP1,
        "fixed_code_step2": FIXED_CODE_STEP2,
        "registry": {
            "has_x2_exec_env": "x2_behavior_code_env" in list_exec_envs(),
            "has_x2_config": "x2_behavior_code_env" in list_configs(),
        },
        "steps": {},
        "verdicts": {},
        "errors": [],
    }

    try:
        registered_cfg = get_config("x2_behavior_code_env")
        summary["registry"].update(
            {
                "registered_low_level": registered_cfg.low_level,
                "registered_apis": registered_cfg.apis,
            }
        )
        summary["verdicts"]["registry"] = {
            "passed": registered_cfg.low_level == "x2_b1k_low_level" and registered_cfg.apis == ["X2ControlApi"],
            "checks": [
                "pass" if summary["registry"]["has_x2_exec_env"] else "fail: x2_behavior_code_env not registered",
                "pass" if summary["registry"]["has_x2_config"] else "fail: x2_behavior_code_env config not registered",
                "pass"
                if registered_cfg.low_level == "x2_b1k_low_level"
                else f"fail: low_level={registered_cfg.low_level}",
                "pass" if registered_cfg.apis == ["X2ControlApi"] else f"fail: apis={registered_cfg.apis}",
            ],
        }

        objects = [
            {
                "type": "DatasetObject",
                "name": OBJECT_NAME,
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [0.6, 0.3, 0.8],
                "orientation": [0, 0, 0, 1],
                "fixed_base": True,
                "kinematic_only": True,
            }
        ]
        low_level = X2BehaviourLowLevel(
            objects=objects,
            load_object_categories=["floors", "ceilings", "walls"],
        )
        exec_env_cls = get_exec_env("x2_behavior_code_env")
        exec_env = exec_env_cls(CodeExecEnvConfig(low_level=low_level, apis=["X2ControlApi"]))
        _obs, _info = exec_env.reset()

        _obs, reward, terminated, truncated, info = exec_env.step(FIXED_CODE_STEP1)
        result = exec_env._exec_globals.get("RESULT")
        passed, detail, checks = _validate_step1_result(result, info)
        detail.update(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "stdout_tail": info.get("stdout", "")[-2000:],
                "stderr_tail": info.get("stderr", "")[-2000:],
            }
        )
        summary["steps"]["fixed_code_arm_motion"] = detail
        summary["verdicts"]["fixed_code_arm_motion"] = {
            "passed": passed,
            "checks": checks,
        }

        placement = _place_object_near_right_eef(low_level)
        summary["steps"]["object_placement"] = placement
        summary["verdicts"]["object_placement"] = {
            "passed": True,
            "checks": ["pass"],
        }

        _obs, reward, terminated, truncated, info = exec_env.step(FIXED_CODE_STEP2)
        result = exec_env._exec_globals.get("RESULT")
        passed, detail, checks = _validate_step2_result(result, info)
        detail.update(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "stdout_tail": info.get("stdout", "")[-2000:],
                "stderr_tail": info.get("stderr", "")[-2000:],
            }
        )
        summary["steps"]["fixed_code_object_pregrasp"] = detail
        summary["verdicts"]["fixed_code_object_pregrasp"] = {
            "passed": passed,
            "checks": checks,
        }

        summary["ok"] = all(v["passed"] for v in summary["verdicts"].values())
    except Exception:
        summary["errors"].append(traceback.format_exc())
        print(summary["errors"][-1])

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps({"ok": summary["ok"], "errors": len(summary["errors"])}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
