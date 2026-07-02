"""ASPIRE-style trace utilities for X2 CaP-X pick-place runs.

The functions here are deliberately offline-friendly: they operate on saved
CaP-X trial folders and X2 visual artifacts, without importing Isaac Sim,
OmniGibson, PyRoKi, or model servers.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RESULT_RE = re.compile(
    r"X2_(?:TWO_TARGET_|TWO_OBJECT_|PICK_PLACE_)?RESULT\s+"
    r"(?P<fields>.*?)(?=\nX2_|\n\s*Stderr:|\Z)",
    re.DOTALL,
)
FIELD_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
TRIAL_RE = re.compile(r"_reward_([0-9.]+)_taskcompleted_([01])")


FAILURE_REPAIR_TAGS = {
    "perception_no_detection": ["broaden_visual_prompts", "move_camera_or_reobserve"],
    "segmentation_bad_mask": ["tighten_detection_box", "retry_sam2_or_reobserve"],
    "depth_too_sparse": ["reobserve_from_precontact", "relax_depth_window", "move_camera_for_depth"],
    "grasp_pose_unreachable": ["try_next_grasp_candidate", "prefer_reachable_reference_orientation"],
    "planner_collision_or_no_path": ["increase_precontact_clearance", "use_rgbd_obstacle_boxes", "try_next_grasp_candidate"],
    "preclose_pose_not_reached": ["try_next_grasp_candidate", "increase_fine_align_steps", "reduce_joint_step"],
    "object_not_in_hand_after_close": ["increase_grasp_tcp_axis_offsets", "try_next_grasp_candidate", "slow_down_close"],
    "place_pre_pose_not_reached": ["relax_place_orientation", "retry_place_prepose", "move_place_target_closer"],
    "place_error_too_large": ["slow_vertical_place_descent", "hold_before_release", "use_visual_grasp_pose_place_offset"],
    "timeout": ["reduce_candidate_count", "use_validated_candidate_indices", "shorten_debug_search"],
    "unknown": ["inspect_trace_bundle", "add_more_primitive_instrumentation"],
}


@dataclass(frozen=True)
class TraceBuildResult:
    trial_dir: Path
    bundle_dir: Path
    ok: bool
    primary_failure: str | None


def jsonable(value: Any) -> Any:
    """Return a JSON-serializable version of saved X2 result data."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def parse_fields(text: str) -> dict[str, str]:
    return {key: value for key, value in FIELD_RE.findall(text or "")}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_trial_result_fields(trial_dir: Path) -> dict[str, str]:
    for filename in ("summary.txt", "raw_response.sh"):
        match = RESULT_RE.search(read_text(trial_dir / filename))
        if match:
            return parse_fields(match.group("fields"))
    return {}


def trial_timed_out(trial_dir: Path) -> bool:
    text = "\n".join(
        read_text(trial_dir / filename)
        for filename in ("summary.txt", "raw_response.sh")
    )
    return "TimeoutError" in text or re.search(r"exceeded\s+\d+\s+seconds", text) is not None


def parse_trial_reward(trial_dir: Path) -> tuple[float | None, bool | None]:
    match = TRIAL_RE.search(trial_dir.name)
    if not match:
        return None, None
    return float(match.group(1)), bool(int(match.group(2)))


def find_trial_dirs(path: Path) -> list[Path]:
    if path.is_file():
        return []
    if path.name.startswith("trial_"):
        return [path]
    return sorted(item for item in path.rglob("trial_*") if item.is_dir())


def artifact_match_keys(run_name: str) -> set[str]:
    keys = {run_name}
    normalized = run_name
    for prefix in ("x2_pick_place_", "x2_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            keys.add(normalized)
    if "x2_pick_place_" in run_name:
        keys.add(run_name.replace("x2_pick_place_", ""))
    return {key for key in keys if key}


def find_visual_artifact_dirs(repo_root: Path, run_name: str, visual_artifact_root: Path | None = None) -> list[Path]:
    if visual_artifact_root is not None:
        root = visual_artifact_root
        if root.exists():
            return sorted(path.parent.resolve() for path in root.rglob("grasp_summary.json"))

    roots = [repo_root / "outputs" / "x2_visual_artifacts"]
    keys = artifact_match_keys(run_name)
    seen: set[Path] = set()
    result: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("grasp_summary.json")):
            parent = path.parent.resolve()
            if parent in seen:
                continue
            if any(key in str(path) for key in keys):
                seen.add(parent)
                result.append(parent)
    return result


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"", "none", "null", "nan"}:
            return None
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if value is None:
        return None
    return bool(value)


def _nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_metrics(
    result_summary: dict[str, Any] | None,
    trial_fields: dict[str, str] | None = None,
    *,
    reward: float | None = None,
    task_completed: bool | None = None,
) -> dict[str, Any]:
    result_summary = result_summary or {}
    trial_fields = trial_fields or {}
    execution = result_summary.get("execution") if isinstance(result_summary.get("execution"), dict) else {}
    place = execution.get("place") if isinstance(execution.get("place"), dict) else {}
    before_close_error = execution.get("before_close_error") if isinstance(execution.get("before_close_error"), dict) else {}
    final_close_attempt = execution.get("final_close_attempt") if isinstance(execution.get("final_close_attempt"), dict) else {}
    reobserve = execution.get("precontact_reobserve") if isinstance(execution.get("precontact_reobserve"), dict) else {}
    place_pre = {}
    transfer_waypoints = place.get("transfer_waypoints")
    if isinstance(transfer_waypoints, list):
        for waypoint in transfer_waypoints:
            if isinstance(waypoint, dict) and waypoint.get("name") == "place_pre_descent":
                place_pre = waypoint
                break
    active_grasp_reachability = _nested_get(execution, "active_plan_reachability", "grasp")
    if not isinstance(active_grasp_reachability, dict):
        active_grasp_reachability = {}
    preclose_motion_debug = before_close_error.get("motion_debug") if isinstance(before_close_error, dict) else {}
    if not isinstance(preclose_motion_debug, dict):
        preclose_motion_debug = {}
    preclose_joint_ik_debug = _nested_get(preclose_motion_debug, "last_joint_ik_move_debug") or {}
    if not isinstance(preclose_joint_ik_debug, dict):
        preclose_joint_ik_debug = {}
    preclose_solve_debug = preclose_joint_ik_debug.get("solve") if isinstance(preclose_joint_ik_debug.get("solve"), dict) else {}
    preclose_joint_move_debug = (
        preclose_joint_ik_debug.get("joint_move") if isinstance(preclose_joint_ik_debug.get("joint_move"), dict) else {}
    )

    return {
        "ok": _to_bool(result_summary.get("ok", trial_fields.get("ok"))),
        "reward": reward,
        "task_completed": task_completed,
        "object_name": result_summary.get("object_name"),
        "obstacle_source": result_summary.get("obstacle_source", trial_fields.get("obstacle_source")),
        "place_offset_source": place.get("place_offset_source", trial_fields.get("place_offset_source")),
        "place_orientation_source": place.get(
            "place_orientation_source",
            execution.get("place_orientation_source", trial_fields.get("place_orientation_source")),
        ),
        "reobserve_adopted": _to_bool(reobserve.get("adopted", trial_fields.get("reobserve_adopted"))),
        "reobserve_reason": reobserve.get("reason", trial_fields.get("reobserve_reason")),
        "before_close_tcp_error_m": _to_float(
            before_close_error.get("tcp_error_m", trial_fields.get("before_close_tcp_error_m"))
        ),
        "before_close_ori_error_rad": _to_float(
            before_close_error.get("ori_error_rad", trial_fields.get("before_close_ori_error_rad"))
        ),
        "final_close_axis_offset_m": _to_float(
            final_close_attempt.get("axis_offset_m", trial_fields.get("final_close_axis_offset_m"))
        ),
        "before_close_reached": _to_bool(execution.get("before_close_reached")),
        "precontact_ok": _to_bool(execution.get("precontact_ok")),
        "object_in_hand_after_close": _to_bool(
            execution.get("object_in_hand_after_close", trial_fields.get("object_in_hand_after_close"))
        ),
        "place_error_m": _to_float(place.get("place_error_m", trial_fields.get("place_error_m"))),
        "place_pre_tcp_error_m": _to_float(place_pre.get("tcp_error_m", trial_fields.get("place_pre_tcp_error_m"))),
        "place_pre_ori_error_rad": _to_float(place_pre.get("ori_error_rad", trial_fields.get("place_pre_ori_error_rad"))),
        "place_descent_waypoints": place.get("place_descent_waypoints", trial_fields.get("place_descent_waypoints")),
        "rgbd_obstacles_sim_truth": (
            (result_summary.get("obstacle_plan") or {}).get("sim_truth")
            if isinstance(result_summary.get("obstacle_plan"), dict)
            else None
        ),
        "active_grasp_ik_reachable": _to_bool(active_grasp_reachability.get("ok")),
        "active_grasp_ik_fk_pos_error_m": _to_float(active_grasp_reachability.get("fk_pos_error_m")),
        "active_grasp_ik_fk_ori_error_rad": _to_float(active_grasp_reachability.get("fk_ori_error_rad")),
        "preclose_ik_solve_fk_pos_error_m": _to_float(preclose_solve_debug.get("solve_fk_pos_error_m")),
        "preclose_ik_solve_fk_ori_error_rad": _to_float(preclose_solve_debug.get("solve_fk_ori_error_rad")),
        "preclose_joint_final_error_rad": _to_float(preclose_joint_move_debug.get("max_final_joint_error_rad")),
        "preclose_joint_ok": _to_bool(preclose_joint_ik_debug.get("joint_ok")),
        "preclose_motion_final_pos_error_m": _to_float(preclose_joint_ik_debug.get("final_pos_error_m")),
        "preclose_motion_final_ori_error_rad": _to_float(preclose_joint_ik_debug.get("final_ori_error_rad")),
    }


def classify_failure(
    result_summary: dict[str, Any] | None,
    trial_fields: dict[str, str] | None = None,
    *,
    reward: float | None = None,
    task_completed: bool | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Classify an X2 pick-place run into an ASPIRE-style failure report."""
    metrics = extract_metrics(result_summary, trial_fields, reward=reward, task_completed=task_completed)
    evidence: dict[str, Any] = dict(metrics)
    primary_failure: str | None = None
    status = "success"
    result_summary = result_summary or {}
    plan = result_summary.get("plan") if isinstance(result_summary.get("plan"), dict) else {}
    obstacle_plan = result_summary.get("obstacle_plan") if isinstance(result_summary.get("obstacle_plan"), dict) else {}
    execution = result_summary.get("execution") if isinstance(result_summary.get("execution"), dict) else {}
    place = execution.get("place") if isinstance(execution.get("place"), dict) else {}

    if timed_out:
        primary_failure = "timeout"
    elif metrics["ok"] is True or task_completed is True:
        primary_failure = None
    elif metrics.get("before_close_reached") is False:
        if metrics.get("active_grasp_ik_reachable") is False:
            primary_failure = "grasp_pose_unreachable"
        else:
            primary_failure = "preclose_pose_not_reached"
    elif metrics.get("object_in_hand_after_close") is False and (
        (not execution or execution.get("close_attempted") is not False)
        and (metrics.get("before_close_tcp_error_m") is not None or metrics.get("before_close_reached") is True)
    ):
        primary_failure = "object_not_in_hand_after_close"
    elif not plan or plan.get("ok") is False:
        error = str(plan.get("error") or result_summary.get("error") or "")
        evidence["plan_error"] = error
        if "No detections" in error:
            primary_failure = "perception_no_detection"
        elif "SAM2" in error or "mask" in error.lower():
            primary_failure = "segmentation_bad_mask"
        elif "reach" in error.lower() or "ik" in error.lower():
            primary_failure = "grasp_pose_unreachable"
        else:
            primary_failure = "grasp_pose_unreachable"
    elif obstacle_plan and obstacle_plan.get("ok") is False:
        errors = obstacle_plan.get("errors") or []
        evidence["obstacle_errors"] = errors
        text = " ".join(str(item) for item in errors).lower()
        if "depth" in text or "point" in text:
            primary_failure = "depth_too_sparse"
        else:
            primary_failure = "planner_collision_or_no_path"
    elif metrics.get("place_error_m") is not None:
        threshold = _to_float(result_summary.get("place_position_threshold")) or 0.10
        if float(metrics["place_error_m"]) > threshold:
            primary_failure = "place_error_too_large"
    elif place and place.get("requested") is True and place.get("place_pre_ok") is False:
        primary_failure = "place_pre_pose_not_reached"
    elif execution and execution.get("precontact_ok") is False:
        primary_failure = "planner_collision_or_no_path"
    if primary_failure is None and not (metrics["ok"] is True or task_completed is True):
        primary_failure = "unknown"

    if primary_failure is not None:
        status = "failure"

    suggested = [] if primary_failure is None else FAILURE_REPAIR_TAGS.get(primary_failure, FAILURE_REPAIR_TAGS["unknown"])
    return {
        "status": status,
        "primary_failure": primary_failure,
        "suggested_repair_tags": suggested,
        "evidence": evidence,
    }


def load_default_skills() -> list[dict[str, Any]]:
    path = Path(__file__).with_name("aspire_skills.json")
    return json.loads(path.read_text(encoding="utf-8"))


def select_applicable_skills(metrics: dict[str, Any], failure_report: dict[str, Any]) -> list[dict[str, Any]]:
    skills = load_default_skills()
    selected: list[dict[str, Any]] = []
    if metrics.get("obstacle_source") == "rgbd_visual":
        selected.extend(skill for skill in skills if skill["id"] == "rgbd_tabletop_obstacle_box_planning")
    if metrics.get("reobserve_adopted") is not None:
        selected.extend(skill for skill in skills if skill["id"] == "precontact_reobserve_with_quality_gates")
    if (_to_int(metrics.get("place_descent_waypoints")) or 0) > 1:
        selected.extend(skill for skill in skills if skill["id"] == "slow_vertical_place_descent")
    if failure_report.get("primary_failure") in {"preclose_pose_not_reached", "object_not_in_hand_after_close", "grasp_pose_unreachable"}:
        selected.extend(skill for skill in skills if skill["id"] == "try_next_grasp_candidate_when_preclose_unreached")
    seen: set[str] = set()
    deduped = []
    for skill in selected:
        if skill["id"] not in seen:
            seen.add(skill["id"])
            deduped.append(skill)
    return deduped


def _pose_record(stage: str, name: str, pose: Any, *, source: str) -> dict[str, Any] | None:
    if not isinstance(pose, (list, tuple)) or len(pose) != 2:
        return None
    return {
        "stage": stage,
        "name": name,
        "source": source,
        "position_world": pose[0],
        "quat_xyzw_world": pose[1],
        "meaning": "T_world_tcp",
    }


def extract_motion_trace(result_summary: dict[str, Any] | None) -> dict[str, Any]:
    result_summary = result_summary or {}
    plan = result_summary.get("plan") if isinstance(result_summary.get("plan"), dict) else {}
    execution = result_summary.get("execution") if isinstance(result_summary.get("execution"), dict) else {}
    planned: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for key in ("precontact_tcp_pose", "grasp_tcp_pose"):
        record = _pose_record("visual_plan", key, plan.get(key), source="plan")
        if record:
            planned.append(record)

    for item in execution.get("insertion_results") or []:
        target = _pose_record("insert", str(item.get("name")), item.get("target_tcp_pose"), source="target")
        reached = _pose_record("insert", str(item.get("name")), item.get("reached_tcp_pose"), source="reached")
        if target:
            planned.append(target)
        if reached:
            executed.append(reached)
        if "tcp_error_m" in item or "ori_error_rad" in item:
            errors.append({"stage": "insert", "name": item.get("name"), "tcp_error_m": item.get("tcp_error_m"), "ori_error_rad": item.get("ori_error_rad")})

    before_close = execution.get("before_close_error") if isinstance(execution.get("before_close_error"), dict) else {}
    record = _pose_record("preclose", "before_close_reached_tcp_pose", before_close.get("reached_tcp_pose"), source="reached")
    if record:
        executed.append(record)
    if before_close:
        errors.append({"stage": "preclose", "name": "before_close", "tcp_error_m": before_close.get("tcp_error_m"), "ori_error_rad": before_close.get("ori_error_rad")})

    for item in execution.get("preclose_attempts") or []:
        target = _pose_record("close", f"axis_offset_{item.get('axis_offset_m')}", item.get("close_target_tcp_pose"), source="target")
        if target:
            planned.append(target)
        close_error = item.get("close_error") if isinstance(item.get("close_error"), dict) else {}
        reached = _pose_record("close", f"axis_offset_{item.get('axis_offset_m')}", close_error.get("reached_tcp_pose"), source="reached")
        if reached:
            executed.append(reached)
        if close_error:
            errors.append({"stage": "close", "name": f"axis_offset_{item.get('axis_offset_m')}", "tcp_error_m": close_error.get("tcp_error_m"), "ori_error_rad": close_error.get("ori_error_rad")})

    place = execution.get("place") if isinstance(execution.get("place"), dict) else {}
    for item in place.get("transfer_waypoints") or []:
        target = _pose_record("transfer", str(item.get("name")), item.get("target_tcp_pose"), source="target")
        reached = _pose_record("transfer", str(item.get("name")), item.get("reached_tcp_pose"), source="reached")
        if target:
            planned.append(target)
        if reached:
            executed.append(reached)
        if "tcp_error_m" in item or "ori_error_rad" in item:
            errors.append(
                {
                    "stage": "transfer",
                    "name": item.get("name"),
                    "tcp_error_m": item.get("tcp_error_m"),
                    "ori_error_rad": item.get("ori_error_rad"),
                }
            )
    for item in place.get("descent_waypoints") or []:
        target = _pose_record("place_descent", str(item.get("name")), item.get("target_tcp_pose"), source="target")
        reached = _pose_record("place_descent", str(item.get("name")), item.get("reached_tcp_pose"), source="reached")
        if target:
            planned.append(target)
        if reached:
            executed.append(reached)
        if "tcp_error_m" in item or "ori_error_rad" in item:
            errors.append({"stage": "place_descent", "name": item.get("name"), "tcp_error_m": item.get("tcp_error_m"), "ori_error_rad": item.get("ori_error_rad")})

    return {
        "planned_tcp_path": planned,
        "executed_tcp_path": executed,
        "tracking_errors": errors,
        "joint_targets": [],
        "joint_actuals": [],
        "joint_trace_note": "Full joint time-series is not yet exported by the X2 low-level controller; TCP waypoints and last_motion_debug are preserved.",
        "initial_plan_reachability": execution.get("initial_plan_reachability"),
        "active_plan_reachability": execution.get("active_plan_reachability"),
        "precontact_motion_debug": execution.get("precontact_motion_debug"),
        "fine_align_motion_debug": execution.get("fine_align_motion_debug"),
        "last_motion_debug": execution.get("last_motion_debug"),
    }


def build_primitive_calls(
    result_summary: dict[str, Any] | None,
    grasp_summaries: list[dict[str, Any]],
    visual_obstacles: dict[str, Any] | None,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    result_summary = result_summary or {}
    plan = result_summary.get("plan") if isinstance(result_summary.get("plan"), dict) else {}
    execution = result_summary.get("execution") if isinstance(result_summary.get("execution"), dict) else {}
    calls: list[dict[str, Any]] = [
        {
            "name": "pick_and_place_visual_object",
            "kind": "task_primitive",
            "inputs": {
                "object_name": result_summary.get("object_name"),
                "place_position_world": result_summary.get("place_position_world"),
                "obstacle_source": result_summary.get("obstacle_source"),
                "place_offset_source": metrics.get("place_offset_source"),
                "candidate_indices_observed": [attempt.get("candidate_index") for attempt in result_summary.get("attempts") or []],
            },
                "outputs": metrics,
                "attempts": result_summary.get("attempts"),
                "status": "success" if metrics.get("ok") else "failure",
            }
        ]
    if grasp_summaries:
        calls.append(
            {
                "name": "plan_visual_grasp_tcp_pose",
                "kind": "vision_grasp_primitive",
                "inputs": {"prompts": (plan.get("visual") or {}).get("prompts")},
                "outputs": {
                    "ok": plan.get("ok"),
                    "selected_T_world_tcp": plan.get("grasp_tcp_pose"),
                    "precontact_T_world_tcp": plan.get("precontact_tcp_pose"),
                    "grasp_summaries": grasp_summaries,
                },
                "status": "success" if plan.get("ok") else "failure",
            }
        )
    if visual_obstacles:
        calls.append(
            {
                "name": "get_rgbd_visual_tabletop_obstacles",
                "kind": "rgbd_obstacle_primitive",
                "inputs": {"source_plan": "plan_visual_grasp_tcp_pose"},
                "outputs": visual_obstacles,
                "status": "success" if visual_obstacles.get("ok") else "failure",
            }
        )
    if execution:
        calls.append(
            {
                "name": "execute_tcp_grasp_plan",
                "kind": "action_primitive",
                "inputs": {
                    "grasp_tcp_pose": plan.get("grasp_tcp_pose"),
                    "precontact_tcp_pose": plan.get("precontact_tcp_pose"),
                    "obstacles_world": result_summary.get("obstacles_world"),
                },
                "outputs": {
                    "ok": execution.get("ok"),
                    "precontact_ok": execution.get("precontact_ok"),
                    "before_close_error": execution.get("before_close_error"),
                    "object_in_hand_after_close": execution.get("object_in_hand_after_close"),
                    "place": execution.get("place"),
                    "candidate_attempts": result_summary.get("attempts"),
                },
                "status": "success" if execution.get("ok") else "failure",
            }
        )
    return calls


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_trace_bundle(
    *,
    repo_root: Path,
    trial_dir: Path,
    output_root: Path,
    visual_artifact_root: Path | None = None,
) -> TraceBuildResult:
    trial_dir = trial_dir.resolve()
    run_name = trial_dir.parent.name
    bundle_dir = (output_root / f"{run_name}__{trial_dir.name}").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    visual_dirs = find_visual_artifact_dirs(repo_root, run_name, visual_artifact_root)
    main_visual_dir = next((path for path in visual_dirs if (path / "pick_place_result_summary.json").exists()), None)
    result_summary = read_json(main_visual_dir / "pick_place_result_summary.json") if main_visual_dir else {}
    trial_fields = parse_trial_result_fields(trial_dir)
    reward, task_completed = parse_trial_reward(trial_dir)
    timed_out = trial_timed_out(trial_dir)
    metrics = extract_metrics(result_summary, trial_fields, reward=reward, task_completed=task_completed)
    failure_report = classify_failure(
        result_summary,
        trial_fields,
        reward=reward,
        task_completed=task_completed,
        timed_out=timed_out,
    )
    skills = select_applicable_skills(metrics, failure_report)

    copy_if_exists(trial_dir / "code.py", bundle_dir / "generated_code.py")
    copy_if_exists(trial_dir / "summary.txt", bundle_dir / "run_summary.txt")
    copy_if_exists(trial_dir / "raw_response.sh", bundle_dir / "raw_response.sh")
    for video in sorted(trial_dir.glob("video_*.mp4")):
        copy_if_exists(video, bundle_dir / "videos" / video.name)

    grasp_summaries: list[dict[str, Any]] = []
    visual_obstacles: dict[str, Any] | None = None
    for idx, visual_dir in enumerate(visual_dirs):
        label = "initial" if visual_dir == main_visual_dir else f"observation_{idx:02d}"
        target_dir = bundle_dir / "visual" / label
        for artifact in sorted(visual_dir.iterdir()):
            if artifact.is_file() and artifact.suffix.lower() in {".png", ".json", ".npy"}:
                copy_if_exists(artifact, target_dir / artifact.name)
        grasp_path = visual_dir / "grasp_summary.json"
        if grasp_path.exists():
            grasp = read_json(grasp_path)
            grasp["artifact_label"] = label
            grasp["artifact_dir"] = str(visual_dir)
            grasp_summaries.append(grasp)
        obstacle_path = visual_dir / "visual_obstacles.json"
        if obstacle_path.exists() and visual_obstacles is None:
            visual_obstacles = read_json(obstacle_path)

    motion_trace = extract_motion_trace(result_summary)
    write_json(bundle_dir / "motion" / "planned_tcp_path.json", motion_trace["planned_tcp_path"])
    write_json(bundle_dir / "motion" / "executed_tcp_path.json", motion_trace["executed_tcp_path"])
    write_json(bundle_dir / "motion" / "tracking_errors.json", motion_trace["tracking_errors"])
    write_json(bundle_dir / "motion" / "joint_targets.json", motion_trace["joint_targets"])
    write_json(bundle_dir / "motion" / "joint_actuals.json", motion_trace["joint_actuals"])
    write_json(bundle_dir / "motion" / "motion_trace_meta.json", {k: v for k, v in motion_trace.items() if k not in {"planned_tcp_path", "executed_tcp_path", "tracking_errors", "joint_targets", "joint_actuals"}})

    primitive_calls = build_primitive_calls(result_summary, grasp_summaries, visual_obstacles, metrics)
    with (bundle_dir / "primitive_calls.jsonl").open("w", encoding="utf-8") as stream:
        for call in primitive_calls:
            stream.write(json.dumps(jsonable(call), ensure_ascii=True, sort_keys=True) + "\n")

    run_meta = {
        "run_name": run_name,
        "trial_dir": str(trial_dir),
        "visual_artifact_dirs": [str(path) for path in visual_dirs],
        "main_visual_artifact_dir": None if main_visual_dir is None else str(main_visual_dir),
        "trace_bundle_schema": "x2_aspire_trace_v1",
        "contract": "Offline ASPIRE-style trace bundle derived from CaP-X trial artifacts and X2 visual artifacts.",
    }
    write_json(bundle_dir / "run_meta.json", run_meta)
    write_json(bundle_dir / "metrics.json", metrics)
    write_json(bundle_dir / "failure_report.json", failure_report)
    write_json(bundle_dir / "skill_library.json", {"skills": skills, "all_default_skills": load_default_skills()})

    return TraceBuildResult(
        trial_dir=trial_dir,
        bundle_dir=bundle_dir,
        ok=bool(metrics.get("ok") or task_completed),
        primary_failure=failure_report.get("primary_failure"),
    )


def build_trace_bundles(
    *,
    repo_root: Path,
    paths: Iterable[Path],
    output_root: Path,
    visual_artifact_root: Path | None = None,
) -> list[TraceBuildResult]:
    results: list[TraceBuildResult] = []
    for path in paths:
        for trial_dir in find_trial_dirs(path):
            results.append(
                build_trace_bundle(
                    repo_root=repo_root,
                    trial_dir=trial_dir,
                    output_root=output_root,
                    visual_artifact_root=visual_artifact_root,
                )
            )
    return results
