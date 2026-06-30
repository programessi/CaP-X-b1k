"""Audit X2 visual-chain pose contracts from existing smoke summaries.

This script does not start OmniGibson or invoke vision models. It reads the
summary.json files produced by the X2 visual smoke scripts and checks the frame
contract between:

1. RGB-D visual object pose estimates,
2. visual/grasp planner TCP targets, and
3. EEF world poses passed to move_hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SUMMARIES = [
    "outputs/x2_visual_to_tcp_pose_smoke/summary.json",
    "outputs/x2_visual_mask_to_move_hand_smoke_tcp/summary.json",
    "outputs/x2_preparing_lunch_box_sam2_smoke/summary.json",
    "outputs/x2_preparing_lunch_box_sam2_graspnet_cylinder_viz_512/summary.json",
    "outputs/x2_preparing_lunch_box_sam2_cylinder_single_side_grasp_once/summary.json",
    "outputs/x2_preparing_lunch_box_sam2_graspnet_full_20260619_2104_256/summary.json",
]


def _arr(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if shape is not None:
        try:
            result = result.reshape(shape)
        except Exception:
            return None
    return result


def _round_list(value: np.ndarray | None, ndigits: int = 6) -> list[float] | None:
    if value is None:
        return None
    return np.round(np.asarray(value, dtype=np.float64).reshape(-1), ndigits).tolist()


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


def _pose_pair(pose: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if pose is None:
        return None, None
    if isinstance(pose, dict):
        return _arr(pose.get("position"), (3,)), _arr(pose.get("quat_xyzw"), (4,))
    if isinstance(pose, (list, tuple)) and len(pose) == 2:
        return _arr(pose[0], (3,)), _arr(pose[1], (4,))
    return None, None


def _pose_pair_tcp_eef_check(
    name: str,
    tcp_pose: Any,
    eef_pose: Any,
    *,
    tolerance_m: float = 0.002,
) -> dict[str, Any] | None:
    tcp_pos, tcp_quat = _pose_pair(tcp_pose)
    eef_pos, eef_quat = _pose_pair(eef_pose)
    if tcp_pos is None or tcp_quat is None or eef_pos is None:
        return None

    rot = _quat_xyzw_to_matrix(tcp_quat)
    inferred_offset = rot.T @ (tcp_pos - eef_pos)
    reconstructed_eef = tcp_pos - rot @ inferred_offset
    conversion_error = float(np.linalg.norm(reconstructed_eef - eef_pos))
    quat_error = None
    if eef_quat is not None:
        quat_error = min(float(np.linalg.norm(tcp_quat - eef_quat)), float(np.linalg.norm(tcp_quat + eef_quat)))

    xy_offset = float(np.linalg.norm(inferred_offset[:2]))
    z_offset = float(inferred_offset[2])
    plausible_x2_tcp_offset = bool(xy_offset <= 0.02 and 0.08 <= z_offset <= 0.13)
    return {
        "name": name,
        "target_tcp_position_world": _round_list(tcp_pos),
        "target_tcp_quat_xyzw_world": _round_list(tcp_quat),
        "target_eef_position_world": _round_list(eef_pos),
        "target_eef_quat_xyzw_world": _round_list(eef_quat),
        "inferred_tcp_offset_eef_m": _round_list(inferred_offset),
        "expected_tcp_offset_eef_m": None,
        "tcp_to_eef_conversion_error_m": round(conversion_error, 9),
        "tcp_and_eef_quat_match_error": None if quat_error is None else round(quat_error, 9),
        "plausible_x2_tcp_offset": plausible_x2_tcp_offset,
        "ok": bool(
            conversion_error <= tolerance_m
            and (quat_error is None or quat_error <= 1e-5)
            and plausible_x2_tcp_offset
        ),
        "move_ok": None,
        "reported_tcp_target_error_m": None,
        "reported_eef_target_error_m": None,
    }


def _movement_tcp_eef_check(
    name: str,
    move: dict[str, Any],
    *,
    expected_offset_eef: np.ndarray | None = None,
    tolerance_m: float = 0.002,
) -> dict[str, Any] | None:
    tcp_pos = _arr(move.get("target_tcp_position"), (3,))
    tcp_quat = _arr(move.get("target_tcp_quat_xyzw"), (4,))
    eef_pos = _arr(move.get("target_eef_position"), (3,))
    eef_quat = _arr(move.get("target_eef_quat_xyzw"), (4,))
    if tcp_pos is None or tcp_quat is None or eef_pos is None:
        return None

    rot = _quat_xyzw_to_matrix(tcp_quat)
    inferred_offset = rot.T @ (tcp_pos - eef_pos)
    offset = expected_offset_eef if expected_offset_eef is not None else inferred_offset
    reconstructed_eef = tcp_pos - rot @ offset
    conversion_error = float(np.linalg.norm(reconstructed_eef - eef_pos))
    quat_error = None
    if eef_quat is not None:
        quat_error = min(float(np.linalg.norm(tcp_quat - eef_quat)), float(np.linalg.norm(tcp_quat + eef_quat)))

    return {
        "name": name,
        "target_tcp_position_world": _round_list(tcp_pos),
        "target_tcp_quat_xyzw_world": _round_list(tcp_quat),
        "target_eef_position_world": _round_list(eef_pos),
        "target_eef_quat_xyzw_world": _round_list(eef_quat),
        "inferred_tcp_offset_eef_m": _round_list(inferred_offset),
        "expected_tcp_offset_eef_m": _round_list(expected_offset_eef),
        "tcp_to_eef_conversion_error_m": round(conversion_error, 9),
        "tcp_and_eef_quat_match_error": None if quat_error is None else round(quat_error, 9),
        "ok": bool(conversion_error <= tolerance_m and (quat_error is None or quat_error <= 1e-5)),
        "move_ok": move.get("ok"),
        "reported_tcp_target_error_m": move.get("tcp_target_error_m"),
        "reported_eef_target_error_m": move.get("eef_target_error_m"),
    }


def _record_visual_pose(name: str, item: dict[str, Any]) -> dict[str, Any]:
    detection = item.get("selected_detection") or item.get("detection") or {}
    pose = item.get("pose") or {}
    return {
        "name": name,
        "role": "object_pose_estimate_from_rgbd_mask",
        "frame": "world",
        "link_or_point": "object/mask point cloud, not EEF and not TCP",
        "quat_format": "xyzw",
        "mask_pixels": item.get("mask_pixels"),
        "selected_detection": {
            "prompt": detection.get("prompt"),
            "score": detection.get("score"),
            "box_xyxy_pixels": detection.get("box"),
        },
        "pose": {
            "position_world": pose.get("position"),
            "quat_xyzw_world": pose.get("quat_xyzw"),
            "bbox_extent_m": pose.get("bbox_extent"),
            "expected_depth_m": pose.get("expected_depth"),
            "depth_window_m": pose.get("depth_window"),
        },
    }


def _audit_visual_to_tcp(data: dict[str, Any]) -> dict[str, Any]:
    steps = data.get("steps", {})
    tcp_step = steps.get("tcp_to_action_target", {})
    motion = steps.get("motion_result", {})
    tcp_pose = tcp_step.get("tcp_target_pose") or {}
    eef_pose = tcp_step.get("eef_target_pose_for_move_hand") or {}
    move = {
        "ok": motion.get("move_ok"),
        "target_tcp_position": tcp_pose.get("position"),
        "target_tcp_quat_xyzw": tcp_pose.get("quat_xyzw"),
        "target_eef_position": eef_pose.get("position"),
        "target_eef_quat_xyzw": eef_pose.get("quat_xyzw"),
        "tcp_target_error_m": motion.get("tcp_target_error_m"),
        "eef_target_error_m": motion.get("eef_target_error_m"),
    }
    return {
        "visual_outputs": [
            {
                "name": data.get("object_name"),
                "role": "deterministic visual object pose used as TCP target",
                "frame": "world",
                "link_or_point": "TCP target point",
                "quat_format": "xyzw",
                "pose": steps.get("visual_object_pose", {}),
            }
        ],
        "tcp_to_eef_checks": [
            _movement_tcp_eef_check(
                "visual_tcp_target_to_move_hand_eef",
                move,
                expected_offset_eef=_arr(tcp_step.get("tcp_offset_eef"), (3,)),
            )
        ],
    }


def _audit_mask_to_move(data: dict[str, Any]) -> dict[str, Any]:
    steps = data.get("steps", {})
    motion = steps.get("motion", {})
    target = motion.get("target_pose_from_visual", {})
    move = {
        "ok": motion.get("move_ok"),
        "target_tcp_position": target.get("visual_position"),
        "target_tcp_quat_xyzw": target.get("quat_xyzw"),
        "target_eef_position": target.get("eef_position"),
        "target_eef_quat_xyzw": target.get("quat_xyzw"),
        "tcp_target_error_m": motion.get("finger_center_to_visual_target_error_m"),
        "eef_target_error_m": motion.get("eef_target_error_m"),
    }
    return {
        "visual_outputs": [
            {
                "name": data.get("object_name"),
                "role": "mask centroid / RGB-D position used as desired TCP/finger-center point",
                "frame": "world",
                "link_or_point": "TCP/finger center target point, not EEF",
                "quat_format": "xyzw",
                "pose": steps.get("visual_geometry", {}).get("estimate", {}),
            }
        ],
        "tcp_to_eef_checks": [
            _movement_tcp_eef_check(
                "visual_mask_point_to_move_hand_eef",
                move,
                expected_offset_eef=_arr(motion.get("finger_center_offset_eef"), (3,)),
            )
        ],
    }


def _collect_action_moves(action_chain: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key, value in action_chain.items():
        if isinstance(value, dict):
            check = _movement_tcp_eef_check(key, value)
            if check is not None:
                checks.append(check)
    for idx, attempt in enumerate(action_chain.get("attempts", []) or []):
        if not isinstance(attempt, dict):
            continue
        for key in ("pregrasp_move", "grasp_move", "lift_move"):
            value = attempt.get(key)
            if isinstance(value, dict):
                check = _movement_tcp_eef_check(f"attempt_{idx}.{key}", value)
                if check is not None:
                    checks.append(check)
    return checks


def _collect_execution_plan_pose_pairs(name: str, plan: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    candidates = list(plan.get("candidates", []) or [])[: max(0, int(limit))]
    pose_pairs = (
        ("pregrasp_tcp_pose", "pregrasp_pose"),
        ("grasp_tcp_pose", "grasp_pose"),
        ("lift_tcp_pose", "lift_pose"),
    )
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        candidate_name = candidate.get("name") or f"candidate_{idx}"
        for tcp_key, eef_key in pose_pairs:
            check = _pose_pair_tcp_eef_check(
                f"{name}.{candidate_name}.{tcp_key}_to_{eef_key}",
                candidate.get(tcp_key),
                candidate.get(eef_key),
            )
            if check is not None:
                checks.append(check)
    return checks


def _audit_lunchbox(data: dict[str, Any]) -> dict[str, Any]:
    steps = data.get("steps", {})
    vision = steps.get("vision_model_outputs", {})
    action = steps.get("action_chain", {})
    visual_outputs = []
    for name in ("food", "box"):
        if isinstance(vision.get(name), dict):
            visual_outputs.append(_record_visual_pose(name, vision[name]))

    grasp_outputs: dict[str, Any] = {}
    food = vision.get("food", {}) if isinstance(vision.get("food"), dict) else {}
    if isinstance(food.get("graspnet"), dict):
        grasp_outputs["graspnet"] = {
            "role": "Contact-GraspNet candidates transformed to world",
            "frame": "world",
            "candidate_pose_meaning": "raw candidate gripper/TCP-like grasp pose, not direct move_hand EEF input",
            "quat_format": "xyzw",
            "ok": food["graspnet"].get("ok"),
            "raw_grasp_count": food["graspnet"].get("raw_grasp_count"),
            "candidate_count": food["graspnet"].get("candidate_count"),
            "mask_pixels": food["graspnet"].get("mask_pixels"),
            "valid_mask_pixels": food["graspnet"].get("valid_mask_pixels"),
        }
    if isinstance(action, dict) and action.get("selected_candidate") is not None:
        selected = action.get("selected_candidate") or {}
        grasp_outputs["selected_graspnet_candidate"] = {
            "role": "selected raw GraspNet world grasp candidate",
            "frame": "world",
            "pose_meaning": "candidate grasp frame before X2-specific TCP rewrite",
            "quat_format": "xyzw",
            "name": selected.get("name"),
            "score": selected.get("score"),
            "grasp_pose": selected.get("grasp_pose"),
            "pregrasp_pose": selected.get("pregrasp_pose"),
            "approach_dir_world": selected.get("approach_dir_world"),
            "contact_point_world": selected.get("contact_point_world"),
        }
    if isinstance(action, dict) and action.get("x2_side_grasp_pose") is not None:
        side = action.get("x2_side_grasp_pose") or {}
        grasp_outputs["x2_side_grasp_pose"] = {
            "role": "X2-specific executable side grasp target",
            "frame": "world",
            "pose_meaning": "TCP pose. Must be converted to EEF world pose before move_hand.",
            "quat_format": "xyzw",
            "tcp_target": side.get("tcp_target"),
            "pregrasp_tcp_target": side.get("pregrasp_tcp_target"),
            "approach_dir_world": side.get("approach_dir_world"),
            "local_minus_z_is_approach": side.get("local_minus_z_is_approach"),
            "local_y_is_horizontal_closing_axis": side.get("local_y_is_horizontal_closing_axis"),
        }
    if isinstance(action, dict) and action.get("x2_execution_plan") is not None:
        plan = action.get("x2_execution_plan") or {}
        first = None
        candidates = plan.get("candidates") or []
        if candidates:
            candidate = candidates[0]
            first = {
                "name": candidate.get("name"),
                "grasp_tcp_pose": candidate.get("grasp_tcp_pose"),
                "grasp_pose": candidate.get("grasp_pose"),
                "meaning": "grasp_tcp_pose is TCP world target; grasp_pose is converted EEF world target",
            }
        grasp_outputs["x2_execution_plan"] = {
            "role": "X2 executable TCP and EEF pose pairs",
            "frame": "world",
            "quat_format": "xyzw",
            "candidate_count": plan.get("candidate_count"),
            "strategy": plan.get("strategy"),
            "first_candidate": first,
        }

    tcp_to_eef_checks = _collect_action_moves(action) if isinstance(action, dict) else []
    preview = food.get("x2_execution_plan_preview")
    if isinstance(preview, dict):
        tcp_to_eef_checks.extend(_collect_execution_plan_pose_pairs("vision_preview", preview))
    executed_plan = action.get("x2_execution_plan") if isinstance(action, dict) else None
    if isinstance(executed_plan, dict):
        tcp_to_eef_checks.extend(_collect_execution_plan_pose_pairs("action_plan", executed_plan))

    return {
        "visual_outputs": visual_outputs,
        "grasp_outputs": grasp_outputs,
        "tcp_to_eef_checks": tcp_to_eef_checks,
        "action_result": {
            "mode": action.get("mode") if isinstance(action, dict) else None,
            "grasp_success": action.get("grasp_success") if isinstance(action, dict) else None,
            "success_attempt_name": action.get("success_attempt_name") if isinstance(action, dict) else None,
            "release_near_box": action.get("release_near_box") if isinstance(action, dict) else None,
        },
    }


def _audit_one(path: Path, root: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    result = {
        "path": rel,
        "summary_ok": data.get("ok"),
        "model_output_source": data.get("model_output_source"),
        "errors_count": len(data.get("errors", []) or []),
        "visual_outputs": [],
        "grasp_outputs": {},
        "tcp_to_eef_checks": [],
        "action_result": {},
    }
    if "x2_visual_to_tcp_pose_smoke" in rel:
        result.update(_audit_visual_to_tcp(data))
    elif "x2_visual_mask_to_move_hand_smoke" in rel:
        result.update(_audit_mask_to_move(data))
    else:
        result.update(_audit_lunchbox(data))

    checks = [check for check in result.get("tcp_to_eef_checks", []) if check is not None]
    failed = [check for check in checks if not check.get("ok")]
    result["contract_verdict"] = {
        "tcp_to_eef_check_count": len(checks),
        "tcp_to_eef_failed_count": len(failed),
        "tcp_to_eef_conversion_ok": bool(checks and not failed) if checks else None,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit X2 visual-chain frame and TCP/EEF pose contracts")
    parser.add_argument("--root", default=".", help="CaP-X-b1k root directory")
    parser.add_argument("--output", default="outputs/x2_visual_chain_contract_audit/summary.json")
    parser.add_argument("summaries", nargs="*", help="summary.json paths relative to --root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_paths = args.summaries or DEFAULT_SUMMARIES
    report = {
        "contract": {
            "move_hand_input": {
                "frame": "world",
                "link": "EEF",
                "pose_symbol": "T_world_eef",
                "quat_format": "xyzw",
            },
            "visual_object_pose": {
                "frame": "world",
                "meaning": "object or mask point-cloud pose; not directly a move_hand target",
                "pose_symbol": "T_world_object",
                "quat_format": "xyzw",
            },
            "visual_grasp_pose": {
                "frame": "world",
                "link": "TCP / finger center / grasp frame unless explicitly named *_eef*",
                "pose_symbol": "T_world_tcp",
                "quat_format": "xyzw",
            },
            "required_conversion": "eef_pos_world = tcp_pos_world - R(tcp_quat_xyzw) @ tcp_offset_eef; eef_quat_xyzw = tcp_quat_xyzw",
        },
        "summaries": [],
    }

    for item in summary_paths:
        path = (root / item).resolve()
        if not path.exists():
            report["summaries"].append({"path": item, "error": "summary file not found"})
            continue
        report["summaries"].append(_audit_one(path, root))

    all_checks = [
        check
        for summary in report["summaries"]
        for check in summary.get("tcp_to_eef_checks", []) or []
        if check is not None
    ]
    report["overall"] = {
        "summary_count": len(report["summaries"]),
        "tcp_to_eef_check_count": len(all_checks),
        "tcp_to_eef_failed_count": sum(1 for check in all_checks if not check.get("ok")),
        "max_tcp_to_eef_conversion_error_m": None
        if not all_checks
        else round(max(float(check["tcp_to_eef_conversion_error_m"]) for check in all_checks), 9),
        "note": "A passing contract check means pose frames agree. It does not imply the grasp/task succeeded physically.",
    }

    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Wrote {output_path}")
    return 0 if report["overall"]["tcp_to_eef_failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
