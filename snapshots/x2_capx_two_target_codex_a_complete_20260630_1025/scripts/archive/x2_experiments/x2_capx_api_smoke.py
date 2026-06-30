"""Layer-2 smoke test for the CAP-X X2 low-level API.

Each API function is exercised directly (no LLM) and the script emits a
per-step ``pass`` / ``fail`` verdict so that behaviour correctness is
automatically checkable instead of relying on manual log inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capx.envs.simulators.x2_b1k import X2BehaviourLowLevel
import capx.envs.simulators.x2_b1k as x2_b1k_module
from capx.integrations.x2.control import X2ControlApi

OBJECT_NAME = "cologne"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _as_list(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return [round(float(v), 6) for v in np.asarray(value).reshape(-1)]


def _check(condition: bool, message: str) -> str:
    """Return 'pass' or 'fail' with an explanation."""
    return "pass" if condition else f"fail: {message}"


def _quat_norm(quat: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(quat)))


def _pose_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b)))


def _gripper_values(env: X2BehaviourLowLevel, arm: int) -> list[float]:
    arm_name = env._arm_name(arm)
    q = env.robot.get_joint_positions()
    idx = env.robot.gripper_control_idx[arm_name]
    return _as_list(q[idx])


# ---------------------------------------------------------------------------
# Per-step validation helpers
# ---------------------------------------------------------------------------


def _validate_observation(obs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Check that the observation dict exposes nested camera tensors."""
    detail: dict[str, Any] = {}
    checks: list[str] = []

    detail["obs_type"] = type(obs).__name__
    is_dict = isinstance(obs, dict)
    checks.append(_check(is_dict, f"observation must be dict, got {type(obs).__name__}"))
    if not is_dict:
        return detail, checks

    detail["top_level_keys"] = sorted(obs.keys())

    rgb_paths: list[str] = []
    depth_paths: list[str] = []
    proprio_paths: list[str] = []
    bad_shapes: list[str] = []

    def visit(path: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}.{key}" if path else str(key), child)
            return

        if not isinstance(value, (torch.Tensor, np.ndarray)):
            return

        key = path.rsplit(".", 1)[-1]
        shape = tuple(value.shape)
        if key == "rgb":
            rgb_paths.append(path)
            if len(shape) < 3 or shape[-1] not in (3, 4):
                bad_shapes.append(f"{path} shape {shape} is not image-like")
        elif key == "depth":
            depth_paths.append(path)
            if len(shape) < 2:
                bad_shapes.append(f"{path} shape {shape} is not depth-like")
        elif key == "proprio":
            proprio_paths.append(path)
            if len(shape) != 1:
                bad_shapes.append(f"{path} shape {shape} is not 1D proprio")

    visit("", obs)
    detail["rgb_paths"] = rgb_paths[:8]
    detail["depth_paths"] = depth_paths[:8]
    detail["proprio_paths"] = proprio_paths[:8]
    detail["num_rgb"] = len(rgb_paths)
    detail["num_depth"] = len(depth_paths)
    detail["num_proprio"] = len(proprio_paths)

    checks.append(_check(len(rgb_paths) > 0, "observation has no nested rgb tensor"))
    checks.append(_check(not bad_shapes, "; ".join(bad_shapes)))

    return detail, checks


def _validate_pose(label: str, pos: np.ndarray, quat: np.ndarray) -> tuple[dict[str, Any], list[str]]:
    """Validate a pose tuple: shapes, quaternion norm ≈ 1, finite values."""
    detail: dict[str, Any] = {}
    checks: list[str] = []
    prefix = label

    pos = np.asarray(pos)
    quat = np.asarray(quat)

    detail[f"{prefix}_pos_shape"] = list(pos.shape)
    detail[f"{prefix}_quat_shape"] = list(quat.shape)

    checks.append(_check(pos.shape in ((3,), (3, 1)), f"{prefix} pos shape {pos.shape} != (3,)"))

    checks.append(_check(quat.shape in ((4,), (4, 1)), f"{prefix} quat shape {quat.shape} != (4,)"))

    checks.append(_check(np.all(np.isfinite(pos)), f"{prefix} pos contains non-finite values"))
    checks.append(_check(np.all(np.isfinite(quat)), f"{prefix} quat contains non-finite values"))

    qn = _quat_norm(quat)
    detail[f"{prefix}_quat_norm"] = round(qn, 6)
    checks.append(_check(abs(qn - 1.0) < 0.01, f"{prefix} quat norm {qn:.6f} not ≈ 1.0"))

    return detail, checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CAP-X X2 API layer-2 smoke test")
    parser.add_argument("--config", default="x2_robotiq85_primitives.yaml")
    parser.add_argument("--arm", type=int, choices=(0, 1), default=1)
    parser.add_argument("--move-distance", type=float, default=0.05)
    parser.add_argument("--output-dir", default="outputs/x2_capx_api_smoke")
    parser.add_argument("--skip-grasp", action="store_true", help="Skip full grasp_object attempt.")
    args = parser.parse_args()

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/og_mpl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    sample_source = inspect.getsource(X2BehaviourLowLevel._sample_grasp_pose)
    api_sample_source = inspect.getsource(X2ControlApi.sample_grasp_pose)
    report: dict[str, Any] = {
        "ok": False,
        "steps": {},
        "verdicts": {},
        "errors": [],
        "debug": {
            "x2_module_file": x2_b1k_module.__file__,
            "sample_grasp_pose_sha1": hashlib.sha1(sample_source.encode()).hexdigest(),
            "sample_uses_controller_sampler": "controller._sample_grasp_pose" in sample_source,
            "api_sample_grasp_pose_sha1": hashlib.sha1(api_sample_source.encode()).hexdigest(),
            "api_sample_calls_env_sampler": "self._env._sample_grasp_pose" in api_sample_source,
        },
    }

    def record_step(name: str, detail: dict[str, Any], verdicts: list[str]):
        passed = all(v.startswith("pass") for v in verdicts)
        report["steps"][name] = detail
        report["verdicts"][name] = {"passed": passed, "checks": verdicts}
        status = "PASS" if passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"  {name}  →  {status}")
        for v in verdicts:
            print(f"    {v}")
        return passed

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

    all_passed = True
    env = None
    try:
        # ==============================================================
        # 0. Environment creation
        # ==============================================================
        print("Creating X2BehaviourLowLevel …")
        env = X2BehaviourLowLevel(
            controller_cfg=args.config,
            objects=objects,
            load_object_categories=["floors", "ceilings", "walls"],
        )
        api = X2ControlApi(env)

        # ==============================================================
        # 1. get_env_observation
        # ==============================================================
        print("\n--- 1. get_env_observation ---")
        obs = api.get_env_observation()
        obs_detail, obs_checks = _validate_observation(obs)
        obs_detail["_keys_preview"] = obs_detail.get("top_level_keys", [])[:12]
        all_passed &= record_step("get_env_observation", obs_detail, obs_checks)

        # ==============================================================
        # 2. get_current_joint_positions
        # ==============================================================
        print("\n--- 2. get_current_joint_positions ---")
        joints = api.get_current_joint_positions()
        j_checks = [
            _check(isinstance(joints, np.ndarray), f"joints must be ndarray, got {type(joints).__name__}"),
            _check(joints.ndim == 1, f"joints must be 1D, got shape {joints.shape}"),
            _check(len(joints) >= 6, f"expected >=6 joints, got {len(joints)}"),
            _check(np.all(np.isfinite(joints)), "joints contain non-finite values"),
        ]
        all_passed &= record_step(
            "get_current_joint_positions",
            {"shape": list(joints.shape), "head": _as_list(joints[:6])},
            j_checks,
        )

        # ==============================================================
        # 3. get_current_eef_pose
        # ==============================================================
        print("\n--- 3. get_current_eef_pose ---")
        initial_pos, initial_quat = api.get_current_eef_pose(arm=args.arm)
        eef_detail, eef_checks = _validate_pose("eef", initial_pos, initial_quat)
        all_passed &= record_step("get_current_eef_pose", eef_detail, eef_checks)

        # ==============================================================
        # 4. get_robot_relative_eef_pose
        # ==============================================================
        print("\n--- 4. get_robot_relative_eef_pose ---")
        rel_pos, rel_quat = api.get_robot_relative_eef_pose(arm=args.arm)
        rel_detail, rel_checks = _validate_pose("rel_eef", rel_pos, rel_quat)
        all_passed &= record_step("get_robot_relative_eef_pose", rel_detail, rel_checks)

        # ==============================================================
        # 5. get_object_pose  (after placing object near EEF)
        # ==============================================================
        print("\n--- 5. get_object_pose ---")
        obj = env.env.scene.object_registry("name", OBJECT_NAME)
        placed_pos = np.array([0.6, 0.3, 0.8], dtype=np.float32)
        obj.set_position_orientation(
            position=torch.tensor(placed_pos, dtype=torch.float32),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        obj.keep_still()
        env.env.step(env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=False)))
        obj_pos, obj_quat = api.get_object_pose(OBJECT_NAME)
        obj_detail, obj_checks = _validate_pose("object", obj_pos, obj_quat)
        obj_err = _pose_distance(obj_pos, placed_pos)
        obj_detail["placed_pos"] = _as_list(placed_pos)
        obj_detail["reported_pos"] = _as_list(obj_pos)
        obj_detail["position_error"] = round(obj_err, 6)
        obj_checks.append(
            _check(obj_err < 0.08, f"object position error {obj_err:.4f}m > 8cm near-field threshold")
        )
        all_passed &= record_step("get_object_pose", obj_detail, obj_checks)

        # ==============================================================
        # 6. open_gripper / close_gripper
        # ==============================================================
        print("\n--- 6. open_gripper / close_gripper ---")
        api.open_gripper(arm=args.arm)
        open_vals = _gripper_values(env, args.arm)
        api.close_gripper(arm=args.arm)
        close_vals = _gripper_values(env, args.arm)

        open_max = max(open_vals)
        close_max = max(close_vals)
        gripper_detail = {
            "open_values": open_vals,
            "close_values": close_vals,
            "open_max": open_max,
            "close_max": close_max,
        }
        gripper_checks = [
            _check(
                open_max > 0.5,
                f"open gripper max {open_max:.4f} not near Robotiq open region (~0.785 rad)",
            ),
            _check(
                close_max < 0.15,
                f"close gripper max {close_max:.4f} not near closed region (~0.0 rad)",
            ),
            _check(open_max > close_max + 0.3, "open gripper max must be significantly larger than close max"),
        ]
        all_passed &= record_step("gripper_open_close", gripper_detail, gripper_checks)

        # ==============================================================
        # 7. move_hand  (+x offset)
        # ==============================================================
        print("\n--- 7. move_hand (+x) ---")
        target_pos = np.asarray(initial_pos) + np.array([args.move_distance, 0.0, 0.0], dtype=np.float32)
        move_success = api.move_hand((target_pos, initial_quat), arm=args.arm)
        moved_pos, _moved_quat = api.get_current_eef_pose(arm=args.arm)
        delta = np.asarray(moved_pos) - np.asarray(initial_pos)
        move_error = float(np.linalg.norm(np.asarray(moved_pos) - target_pos))

        move_detail = {
            "success": bool(move_success),
            "target_pos": _as_list(target_pos),
            "moved_pos": _as_list(moved_pos),
            "delta": _as_list(delta),
            "error": round(move_error, 6),
        }
        move_checks = [
            _check(move_success, "move_hand returned False"),
            _check(move_error < 0.01, f"move error {move_error:.4f}m > 1cm threshold for {args.move_distance}m move"),
            _check(
                delta[0] > 0.0 and abs(delta[0]) >= abs(delta[1]) and abs(delta[0]) >= abs(delta[2]),
                f"delta {_as_list(delta)} — +x direction not dominant",
            ),
        ]
        all_passed &= record_step("move_hand", move_detail, move_checks)

        # ==============================================================
        # 8. sample_grasp_pose  (→ key Layer-2 fix)
        # ==============================================================
        print("\n--- 8. sample_grasp_pose ---")
        api.open_gripper(arm=args.arm)
        sample_placed_pos = np.asarray(moved_pos) + np.array([0.09, 0.0, -0.05], dtype=np.float32)
        obj.set_position_orientation(
            position=torch.tensor(sample_placed_pos, dtype=torch.float32),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        obj.keep_still()
        env.env.step(env.controller._postprocess_action(env.controller._empty_action(follow_arm_targets=False)))
        obj_pos, obj_quat = api.get_object_pose(OBJECT_NAME)
        pregrasp_pose, grasp_pose = api.sample_grasp_pose(OBJECT_NAME, arm=args.arm)

        gp_pos = np.asarray(grasp_pose[0])
        gp_quat = np.asarray(grasp_pose[1])
        pp_pos = np.asarray(pregrasp_pose[0])
        pp_quat = np.asarray(pregrasp_pose[1])

        # Compute distances
        grasp_to_obj = _pose_distance(gp_pos, obj_pos)
        pregrasp_to_obj = _pose_distance(pp_pos, obj_pos)
        grasp_pregrasp_dist = _pose_distance(gp_pos, pp_pos)

        sgp_detail = {
            "sample_placed_pos": _as_list(sample_placed_pos),
            "object_pos": _as_list(obj_pos),
            "pregrasp_pos": _as_list(pp_pos),
            "grasp_pos": _as_list(gp_pos),
            "grasp_dist_to_object": round(grasp_to_obj, 6),
            "pregrasp_dist_to_object": round(pregrasp_to_obj, 6),
            "grasp_pregrasp_separation": round(grasp_pregrasp_dist, 6),
            "sampler_debug": getattr(env, "_last_sample_grasp_debug", None),
        }

        sgp_checks = [
            _check(
                grasp_to_obj < 0.15,
                f"grasp_pos {grasp_to_obj:.4f}m from object — not near-field (>=15cm)",
            ),
            _check(
                0.01 < grasp_pregrasp_dist < 0.15,
                f"pregrasp↔grasp sep {grasp_pregrasp_dist:.4f}m — expected 1-15cm",
            ),
            _check(
                gp_pos[2] > obj_pos[2],
                f"grasp z {gp_pos[2]:.4f} <= object z {obj_pos[2]:.4f} (grasp should be above object)",
            ),
            _check(
                pp_pos[2] >= gp_pos[2],
                f"pregrasp z {pp_pos[2]:.4f} < grasp z {gp_pos[2]:.4f} (pregrasp should be at or above grasp)",
            ),
        ]
        # Pose validity
        _, pose_checks = _validate_pose("grasp", gp_pos, gp_quat)
        sgp_checks.extend(pose_checks)
        _, pp_pose_checks = _validate_pose("pregrasp", pp_pos, pp_quat)
        sgp_checks.extend(pp_pose_checks)

        all_passed &= record_step("sample_grasp_pose", sgp_detail, sgp_checks)

        # ==============================================================
        # 9. lift_arm  (direct test)
        # ==============================================================
        print("\n--- 9. lift_arm ---")
        pre_lift_pos, _pre_lift_quat = api.get_current_eef_pose(arm=args.arm)
        lift_ok = api.lift_arm(arm=args.arm)
        post_lift_pos, _post_lift_quat = api.get_current_eef_pose(arm=args.arm)
        lift_dz = float(np.asarray(post_lift_pos)[2] - np.asarray(pre_lift_pos)[2])
        lift_detail = {
            "success": bool(lift_ok),
            "pre_lift_z": round(float(np.asarray(pre_lift_pos)[2]), 6),
            "post_lift_z": round(float(np.asarray(post_lift_pos)[2]), 6),
            "dz": round(lift_dz, 6),
        }
        lift_checks = [
            _check(lift_ok, "lift_arm returned False"),
            _check(lift_dz > 0.01, f"lift dz {lift_dz:.4f}m — expected >1cm upward"),
        ]
        all_passed &= record_step("lift_arm", lift_detail, lift_checks)

        # ==============================================================
        # 10. check_object_in_hand  (direct, pre-grasp)
        # ==============================================================
        print("\n--- 10. check_object_in_hand (pre-grasp) ---")
        in_hand_before = api.check_object_in_hand(arm=args.arm)
        coih_detail: dict[str, Any] = {"pre_grasp": bool(in_hand_before)}
        coih_checks: list[str] = [
            _check(
                not in_hand_before,
                "check_object_in_hand returned True before any grasp — expected False",
            ),
        ]

        # ==============================================================
        # 11. grasp_object  (full attempt)
        # ==============================================================
        print("\n--- 11. grasp_object ---")
        if not args.skip_grasp:
            grasp_success = api.grasp_object(pregrasp_pose, grasp_pose, OBJECT_NAME, arm=args.arm)
            in_hand_after = api.check_object_in_hand(arm=args.arm)
            coih_detail["post_grasp"] = bool(in_hand_after)
            go_detail = {
                "success": bool(grasp_success),
                "object_in_hand_after": bool(in_hand_after),
                "object_name": OBJECT_NAME,
            }
            go_checks = [
                _check(
                    grasp_success is not None,
                    "grasp_object returned None (should be bool)",
                ),
            ]
            # Note: grasp success depends on many factors; we do not require True.
            # We only verify the API executed without crashes and returned a bool.
            all_passed &= record_step("grasp_object", go_detail, go_checks)

            coih_checks.append(
                _check(
                    isinstance(in_hand_after, bool),
                    f"check_object_in_hand returned {type(in_hand_after).__name__}, expected bool",
                ),
            )
        else:
            go_detail = {"skipped": True}
            go_checks = ["pass (skipped)"]
            all_passed &= record_step("grasp_object", go_detail, go_checks)

        all_passed &= record_step("check_object_in_hand", coih_detail, coih_checks)

        # ==============================================================
        # 12. Unsupported APIs  (solve_ik, move_to_joint_positions)
        # ==============================================================
        print("\n--- 12. unsupported APIs ---")

        def _check_not_implemented(label, fn):
            nonlocal all_passed
            detail: dict[str, Any] = {}
            checks: list[str] = []
            try:
                fn()
            except NotImplementedError as e:
                detail["status"] = "expected_not_implemented"
                detail["message"] = str(e)
                checks.append("pass")
                print(f"  {label}: NotImplementedError (expected)")
            except Exception as e:
                detail["status"] = "unexpected_exception"
                detail["message"] = repr(e)
                checks.append(f"fail: unexpected {type(e).__name__}: {e}")
                print(f"  {label}: unexpected {type(e).__name__}")
            else:
                detail["status"] = "no_exception"
                checks.append("fail: expected NotImplementedError but call returned")
                print(f"  {label}: FAIL — no exception raised")
            all_passed &= record_step(label, detail, checks)

        _check_not_implemented(
            "solve_ik",
            lambda: api.solve_ik(
                np.asarray(initial_pos), np.asarray(initial_quat)[[3, 0, 1, 2]], arm=args.arm
            ),
        )
        _check_not_implemented(
            "move_to_joint_positions",
            lambda: api.move_to_joint_positions(joints),
        )

        report["ok"] = all_passed

    except Exception as e:
        report["errors"].append(
            {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        )
        print("\n" + report["errors"][-1]["traceback"])
        raise
    finally:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {summary_path}")

        # Print final tally
        verdicts = report.get("verdicts", {})
        passed = sum(1 for v in verdicts.values() if v.get("passed"))
        total = len(verdicts)
        print(f"\n{'='*60}")
        print(f"  Layer-2 verdict: {passed}/{total} steps passed")
        print(f"  Overall: {'OK' if report.get('ok') else 'FAILED'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
