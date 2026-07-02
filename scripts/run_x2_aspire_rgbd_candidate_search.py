#!/usr/bin/env python3
"""ASPIRE-style candidate search harness for X2 RGB-D pick-place.

The harness searches over a small, controlled parameter space for the existing
high-level primitive call. It does not let the model rewrite low-level X2
control code. In ``--execute`` mode it runs debug seeds, selects the best
candidate by saved trace metrics, then runs held-out validation seeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ASPIRE_PATH = REPO_ROOT / "capx" / "integrations" / "x2" / "aspire.py"
_ASPIRE_SPEC = importlib.util.spec_from_file_location("x2_aspire_offline", _ASPIRE_PATH)
if _ASPIRE_SPEC is None or _ASPIRE_SPEC.loader is None:
    raise RuntimeError(f"Could not load X2 ASPIRE utilities from {_ASPIRE_PATH}")
_ASPIRE = importlib.util.module_from_spec(_ASPIRE_SPEC)
sys.modules[_ASPIRE_SPEC.name] = _ASPIRE
_ASPIRE_SPEC.loader.exec_module(_ASPIRE)
build_trace_bundles = _ASPIRE.build_trace_bundles
classify_failure = _ASPIRE.classify_failure
jsonable = _ASPIRE.jsonable
read_json = _ASPIRE.read_json


TARGET_OBJECT = "x2_pick_place_blue_cube"
DISTRACTOR_OBJECT = "x2_pick_place_red_cube"
RIGHT_TARGET = "x2_pick_place_right_target_marker"


DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "stable_rgbd_v1",
        "description": "Known stable x2-rgbd-non-oracle-v1 parameters.",
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.025",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.07",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.45",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.023",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.25",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "rgbd_tabletop_obstacle_box_planning",
            "precontact_reobserve_with_quality_gates",
            "slow_vertical_place_descent",
        ],
    },
    {
        "id": "slightly_faster_descent",
        "description": "Keeps perception and candidate policy stable but tests a faster place descent.",
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.025",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.07",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.45",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.023",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.25",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "3",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.008",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "rgbd_tabletop_obstacle_box_planning",
            "precontact_reobserve_with_quality_gates",
        ],
    },
    {
        "id": "wider_grasp_candidate_search",
        "description": "Allows one lower-ranked initial candidate before falling back to validated candidates.",
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "0,1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.025",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.07",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.45",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.023",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.25",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "try_next_grasp_candidate_when_preclose_unreached",
            "precontact_reobserve_with_quality_gates",
        ],
    },
    {
        "id": "repair_wider_relaxed_preclose",
        "description": "Repair for preclose_pose_not_reached: try more visual candidates and relax quality gates modestly.",
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "0,1,2,3,4",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012,0.016",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.045",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.065",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.11",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.050",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.55",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.040",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.35",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "try_next_grasp_candidate_when_preclose_unreached",
            "precontact_reobserve_with_quality_gates",
            "slow_vertical_place_descent",
        ],
    },
    {
        "id": "repair_validated_relaxed_preclose_v2",
        "description": (
            "Repair for preclose_pose_not_reached without the wide empty-gripper search: "
            "keep validated visual candidates, keep the known close offsets, and relax only the final reach gate."
        ),
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.030",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.045",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.085",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.040",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.50",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.030",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.30",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "try_next_grasp_candidate_when_preclose_unreached",
            "precontact_reobserve_with_quality_gates",
            "slow_vertical_place_descent",
        ],
    },
    {
        "id": "repair_validated_near_miss_close_v1",
        "description": (
            "Repair for object_not_in_hand_after_close after a near-miss preclose: "
            "keep validated visual candidates, allow a slightly looser before-close TCP gate, "
            "and try a few deeper TCP-axis close offsets before moving to the next visual candidate."
        ),
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012,0.016,0.020",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.030",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.045",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.085",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.040",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.50",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.035",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.35",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "increase_grasp_tcp_axis_offsets",
            "try_next_grasp_candidate",
            "precontact_reobserve_with_quality_gates",
            "slow_vertical_place_descent",
        ],
    },
    {
        "id": "repair_place_keep_lift_orientation_v1",
        "description": (
            "Repair for place_pre_pose_not_reached after a successful grasp: keep the validated "
            "RGB-D grasp policy, but release using the measured post-lift TCP orientation instead "
            "of forcing the GraspNet grasp orientation during transfer/place."
        ),
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012,0.016,0.020",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.030",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.045",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.085",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.040",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.50",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.035",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.35",
            "CAPX_X2_RGBD_PLACE_ORIENTATION_SOURCE": "post_lift_current",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "16",
        },
        "skills": [
            "increase_grasp_tcp_axis_offsets",
            "try_next_grasp_candidate",
            "precontact_reobserve_with_quality_gates",
            "use_post_lift_place_orientation",
            "slow_vertical_place_descent",
        ],
    },
    {
        "id": "repair_slow_fine_align_v1",
        "description": (
            "Repair for reachable preclose targets that fail due to joint execution tracking: "
            "keep validated visual candidates but slow final insert/fine-align and retry fine-align."
        ),
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "1,2",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0,0.004,0.008,0.012",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "1",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.08",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.030",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.045",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.085",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.040",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.50",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.023",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.25",
            "CAPX_X2_RGBD_MAX_JOINT_STEP": "0.020",
            "CAPX_X2_RGBD_INSERT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_TRANSFER_MAX_JOINT_STEP": "0.020",
            "CAPX_X2_RGBD_PLACE_INSERT_MAX_JOINT_STEP": "0.008",
            "CAPX_X2_RGBD_SETTLE_STEPS": "36",
            "CAPX_X2_RGBD_HOLD_STEPS": "3",
            "CAPX_X2_RGBD_INSERT_HOLD_STEPS": "12",
            "CAPX_X2_RGBD_FINE_ALIGN_RETRIES": "3",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "4",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.006",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "8",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "20",
        },
        "skills": [
            "try_next_grasp_candidate_when_preclose_unreached",
            "precontact_reobserve_with_quality_gates",
            "repeat_slow_fine_align_for_joint_tracking",
            "slow_vertical_place_descent",
        ],
    },
    {
        "id": "controlled_failure_fast_no_reobserve",
        "description": "Failure-seeking baseline: shallow close offsets and a fast single descent.",
        "params": {
            "CAPX_X2_RGBD_CANDIDATE_INDICES": "0",
            "CAPX_X2_RGBD_GRASP_TCP_AXIS_OFFSETS_M": "0.0",
            "CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT": "0",
            "CAPX_X2_RGBD_REOBSERVE_DISTANCE_M": "0.05",
            "CAPX_X2_RGBD_REOBSERVE_MAX_OBJECT_SHIFT_M": "0.025",
            "CAPX_X2_RGBD_REOBSERVE_MAX_GRASP_SHIFT_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_PRECONTACT_SHIFT_M": "0.07",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_POS_ERROR_M": "0.035",
            "CAPX_X2_RGBD_REOBSERVE_MAX_IK_ORI_ERROR_RAD": "0.45",
            "CAPX_X2_RGBD_FINAL_TCP_THRESHOLD_M": "0.023",
            "CAPX_X2_RGBD_FINAL_ORI_THRESHOLD_RAD": "0.25",
            "CAPX_X2_RGBD_PLACE_DESCENT_WAYPOINTS": "1",
            "CAPX_X2_RGBD_PLACE_DESCENT_MAX_JOINT_STEP": "0.012",
            "CAPX_X2_RGBD_PLACE_DESCENT_HOLD_STEPS": "2",
            "CAPX_X2_RGBD_PLACE_PRE_RELEASE_SETTLE_STEPS": "0",
        },
        "expected_failure": True,
        "skills": [],
    },
]


DEBUG_SEEDS: list[dict[str, Any]] = [
    {
        "id": "debug_nominal",
        "target_position": [0.32, -0.08, 0.921],
        "distractor_position": [0.225, 0.035, 0.921],
        "place_position": [0.37, 0.055, 0.921],
    },
    {
        "id": "debug_slight_left",
        "target_position": [0.30, -0.075, 0.921],
        "distractor_position": [0.225, 0.040, 0.921],
        "place_position": [0.37, 0.055, 0.921],
    },
    {
        "id": "debug_near_distractor",
        "target_position": [0.335, -0.095, 0.921],
        "distractor_position": [0.278, -0.082, 0.921],
        "place_position": [0.37, 0.055, 0.921],
    },
]


VALIDATION_SEEDS: list[dict[str, Any]] = [
    {
        "id": "val_nominal_shift",
        "target_position": [0.318, -0.078, 0.921],
        "distractor_position": [0.225, 0.035, 0.921],
        "place_position": [0.37, 0.055, 0.921],
    },
    {
        "id": "val_centered",
        "target_position": [0.305, -0.065, 0.921],
        "distractor_position": [0.225, 0.045, 0.921],
        "place_position": [0.37, 0.055, 0.921],
    },
    {
        "id": "val_right_shift",
        "target_position": [0.342, -0.083, 0.921],
        "distractor_position": [0.225, 0.025, 0.921],
        "place_position": [0.37, 0.055, 0.921],
    },
]


def _set_object_position(config: dict[str, Any], name: str, position: list[float]) -> None:
    objects = config["env"]["cfg"]["low_level"]["objects"]
    for obj in objects:
        if obj.get("name") == name:
            obj["position"] = [float(value) for value in position]
            return
    raise KeyError(f"Object {name!r} not found in config")


def _write_seed_config(base_config: Path, output_path: Path, seed: dict[str, Any]) -> None:
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    _set_object_position(config, TARGET_OBJECT, seed["target_position"])
    _set_object_position(config, DISTRACTOR_OBJECT, seed["distractor_position"])
    _set_object_position(config, RIGHT_TARGET, seed["place_position"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _effective_output_dir(output_dir: Path, model: str) -> Path:
    return output_dir.parent / model.replace("/", "_") / output_dir.name


def _load_metrics_from_bundle(bundle_dir: Path) -> dict[str, Any]:
    metrics_path = bundle_dir / "metrics.json"
    failure_path = bundle_dir / "failure_report.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    failure = read_json(failure_path) if failure_path.exists() else {"primary_failure": "unknown"}
    return {"metrics": metrics, "failure_report": failure}


def _score(metrics: dict[str, Any], failure_report: dict[str, Any]) -> float:
    completed = 1.0 if metrics.get("task_completed") or metrics.get("ok") else 0.0
    reward = float(metrics.get("reward") or 0.0)
    tcp_error = float(metrics.get("before_close_tcp_error_m") or 0.20)
    ori_error = float(metrics.get("before_close_ori_error_rad") or 1.0)
    place_error = float(metrics.get("place_error_m") or 0.20)
    failure_penalty = 0.0 if failure_report.get("primary_failure") is None else 50.0
    return 200.0 * completed + 100.0 * reward - 60.0 * tcp_error - 10.0 * ori_error - 60.0 * place_error - failure_penalty


def _mean_metric(results: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for item in results:
        value = item.get("metrics", {}).get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _bundle_has_video(bundle_dir: str | None) -> bool:
    if not bundle_dir:
        return False
    return any(Path(bundle_dir).glob("videos/*.mp4"))


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """Terminate a spawned simulation subprocess and its descendants."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _run_command_with_group_cleanup(command: list[str], *, cwd: Path, env: dict[str, str], timeout_s: int) -> tuple[int, bool]:
    """Run a command while guaranteeing Ctrl-C / timeout cleans the child group."""
    proc = subprocess.Popen(command, cwd=cwd, env=env, text=True, start_new_session=True)
    try:
        return proc.wait(timeout=int(timeout_s)), False
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        return 124, True
    except KeyboardInterrupt:
        _kill_process_group(proc)
        raise


def _run_one(
    *,
    repo_root: Path,
    runner: Path,
    candidate: dict[str, Any],
    seed: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    visual_dir: Path,
    trace_dir: Path,
    timeout_s: int,
    model: str,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in candidate.get("params", {}).items()})
    env.update(
        {
            "CONFIG_PATH": str(config_path),
            "OUTPUT_DIR": str(output_dir),
            "VISUAL_ARTIFACT_DIR": str(visual_dir),
            "TRACE_OUTPUT_DIR": str(trace_dir),
            "TIMEOUT_SECONDS": str(timeout_s),
            "STAMP": f"{candidate['id']}_{seed['id']}",
        }
    )
    command = [str(runner)]
    started = time.time()
    effective_output = _effective_output_dir(output_dir, model)
    reused_existing = bool(reuse_existing and effective_output.exists())
    timed_out = False
    if reused_existing:
        returncode = 0
    else:
        returncode, timed_out = _run_command_with_group_cleanup(
            command,
            cwd=repo_root,
            env=env,
            timeout_s=int(timeout_s) + 180,
        )
    elapsed = time.time() - started
    trace_build_error = None
    if effective_output.exists():
        try:
            build_trace_bundles(
                repo_root=repo_root,
                paths=[effective_output],
                output_root=trace_dir,
                visual_artifact_root=visual_dir,
            )
        except Exception as exc:
            trace_build_error = f"{type(exc).__name__}: {exc}"
    bundles = sorted(trace_dir.glob("*trial_*"))
    latest_bundle = bundles[-1] if bundles else None
    metrics = {}
    failure_report = classify_failure(None, {}, timed_out=timed_out or returncode == 124)
    if latest_bundle is not None:
        loaded = _load_metrics_from_bundle(latest_bundle)
        metrics = loaded["metrics"]
        failure_report = loaded["failure_report"]
    return {
        "candidate_id": candidate["id"],
        "seed_id": seed["id"],
        "returncode": returncode,
        "elapsed_s": elapsed,
        "reused_existing": reused_existing,
        "output_dir": str(effective_output),
        "visual_artifact_dir": str(visual_dir),
        "trace_dir": str(trace_dir),
        "bundle_dir": None if latest_bundle is None else str(latest_bundle),
        "metrics": metrics,
        "failure_report": failure_report,
        "trace_build_error": trace_build_error,
        "score": _score(metrics, failure_report),
    }


def _summarize_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"score": float("-inf"), "successes": 0, "trials": 0}
    successes = sum(1 for item in results if item.get("metrics", {}).get("task_completed") or item.get("metrics", {}).get("ok"))
    bundle_count = sum(1 for item in results if item.get("bundle_dir") and Path(str(item["bundle_dir"])).exists())
    video_count = sum(1 for item in results if _bundle_has_video(item.get("bundle_dir")))
    return {
        "score": sum(float(item.get("score", 0.0)) for item in results) / len(results),
        "successes": successes,
        "trials": len(results),
        "primary_failures": [item.get("failure_report", {}).get("primary_failure") for item in results],
        "avg_before_close_tcp_error_m": _mean_metric(results, "before_close_tcp_error_m"),
        "avg_before_close_ori_error_rad": _mean_metric(results, "before_close_ori_error_rad"),
        "avg_place_error_m": _mean_metric(results, "place_error_m"),
        "trace_bundles": bundle_count,
        "videos": video_count,
    }


def _fmt_optional(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "candidate_search_report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = ["# X2 ASPIRE RGB-D Candidate Search", ""]
    lines.append(f"- mode: `{report['mode']}`")
    lines.append(f"- best_candidate: `{report.get('best_candidate_id') or '-'}`")
    lines.append("")
    lines.append("| split | candidate | successes | trials | score | avg TCP m | avg ori rad | avg place m | traces | videos | failures |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for split in ("debug", "validation"):
        for candidate_id, summary in sorted((report.get(f"{split}_summaries") or {}).items()):
            failures = ", ".join(str(item) for item in summary.get("primary_failures", []))
            lines.append(
                f"| {split} | {candidate_id} | {summary.get('successes', 0)} | "
                f"{summary.get('trials', 0)} | {summary.get('score', 0.0):.3f} | "
                f"{_fmt_optional(summary.get('avg_before_close_tcp_error_m'))} | "
                f"{_fmt_optional(summary.get('avg_before_close_ori_error_rad'))} | "
                f"{_fmt_optional(summary.get('avg_place_error_m'))} | "
                f"{summary.get('trace_bundles', 0)} | {summary.get('videos', 0)} | {failures} |"
            )
    (report_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_candidate_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        raise ValueError(f"Candidate file must contain a list or a {{'candidates': [...]}} object: {path}")
    allowed_param_keys = {key for candidate in DEFAULT_CANDIDATES for key in candidate.get("params", {})}
    loaded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate #{index} must be an object.")
        candidate_id = candidate.get("id")
        params = candidate.get("params")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"Candidate #{index} is missing a non-empty string id.")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate id in {path}: {candidate_id}")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"Candidate {candidate_id!r} is missing params.")
        unknown = sorted(set(params) - allowed_param_keys)
        if unknown:
            raise ValueError(f"Candidate {candidate_id!r} uses unsupported params: {unknown}")
        loaded.append(
            {
                "id": candidate_id,
                "description": str(candidate.get("description") or "External ASPIRE candidate."),
                "params": {str(key): str(value) for key, value in params.items()},
                "skills": [str(item) for item in candidate.get("skills", [])],
                "source": str(candidate.get("source") or path),
            }
        )
        seen_ids.add(candidate_id)
    return loaded


def _select_candidates(raw: str | None, *, candidate_file: Path | None = None) -> list[dict[str, Any]]:
    available = _load_candidate_file(candidate_file) if candidate_file is not None else list(DEFAULT_CANDIDATES)
    if not raw:
        return available
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    candidates = [candidate for candidate in available if candidate["id"] in requested]
    missing = requested - {candidate["id"] for candidate in candidates}
    if missing:
        valid = ", ".join(candidate["id"] for candidate in available)
        raise ValueError(f"Unknown candidate id(s): {sorted(missing)}. Valid: {valid}")
    return candidates


def _filter_seeds(seeds: list[dict[str, Any]], raw: str | None, *, split: str) -> list[dict[str, Any]]:
    if not raw:
        return seeds
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    by_id = {seed["id"]: seed for seed in seeds}
    missing = [seed_id for seed_id in requested if seed_id not in by_id]
    if missing:
        valid = ", ".join(seed["id"] for seed in seeds)
        raise ValueError(f"Unknown {split} seed id(s): {missing}. Valid: {valid}")
    return [by_id[seed_id] for seed_id in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--base-config", default="env_configs/x2/x2_pick_place_two_objects_blue_right_rgbd_visual.yaml")
    parser.add_argument("--runner", default="scripts/run_x2_two_object_blue_right_rgbd_visual_codex_a_non_oracle_smoke.sh")
    parser.add_argument("--output-root", default=None, help="Search output root. Defaults to outputs/x2_aspire_candidate_search/<stamp>.")
    parser.add_argument("--model", default="codex-a", help="Model directory inserted by CaP-X runner.")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--execute", action="store_true", help="Actually run CaP-X. Without this, only write the search plan.")
    parser.add_argument("--debug-limit", type=int, default=len(DEBUG_SEEDS))
    parser.add_argument("--validation-limit", type=int, default=len(VALIDATION_SEEDS))
    parser.add_argument("--debug-seed-ids", default=None, help="Optional comma-separated debug seed ids to run before --debug-limit is applied.")
    parser.add_argument("--validation-seed-ids", default=None, help="Optional comma-separated validation seed ids to run before --validation-limit is applied.")
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate ids to include in debug search.")
    parser.add_argument("--candidate-file", default=None, help="Optional JSON file containing externally proposed candidates.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing run directories under --output-root when present.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else repo_root / "outputs" / "x2_aspire_candidate_search" / stamp
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    base_config = repo_root / args.base_config
    runner = repo_root / args.runner
    debug_seeds = _filter_seeds(DEBUG_SEEDS, args.debug_seed_ids, split="debug")[: max(0, int(args.debug_limit))]
    validation_seeds = _filter_seeds(VALIDATION_SEEDS, args.validation_seed_ids, split="validation")[
        : max(0, int(args.validation_limit))
    ]
    candidate_file = Path(args.candidate_file).resolve() if args.candidate_file else None
    candidates = _select_candidates(args.candidates, candidate_file=candidate_file)

    plan_rows: list[dict[str, Any]] = []
    for split, seeds in (("debug", debug_seeds), ("validation", validation_seeds)):
        for seed in seeds:
            config_path = output_root / "configs" / split / f"{seed['id']}.yaml"
            _write_seed_config(base_config, config_path, seed)
            for candidate in (candidates if split == "debug" else [candidates[0]]):
                run_root = output_root / split / candidate["id"] / seed["id"]
                plan_rows.append(
                    {
                        "split": split,
                        "candidate_id": candidate["id"],
                        "seed_id": seed["id"],
                        "config_path": str(config_path),
                        "output_dir": str(run_root / "run"),
                        "visual_artifact_dir": str(run_root / "visual_artifacts"),
                        "trace_dir": str(run_root / "trace"),
                        "params": candidate["params"],
                    }
                )

    report: dict[str, Any] = {
        "mode": "execute" if args.execute else "plan_only",
        "candidates": candidates,
        "candidate_file": None if candidate_file is None else str(candidate_file),
        "debug_seeds": debug_seeds,
        "validation_seeds": validation_seeds,
        "plan": plan_rows,
        "debug_results": [],
        "validation_results": [],
        "debug_summaries": {},
        "validation_summaries": {},
        "best_candidate_id": None,
    }

    if args.execute:
        debug_results: list[dict[str, Any]] = []
        for candidate in candidates:
            for seed in debug_seeds:
                run_root = output_root / "debug" / candidate["id"] / seed["id"]
                result = _run_one(
                    repo_root=repo_root,
                    runner=runner,
                    candidate=candidate,
                    seed=seed,
                    config_path=output_root / "configs" / "debug" / f"{seed['id']}.yaml",
                    output_dir=run_root / "run",
                    visual_dir=run_root / "visual_artifacts",
                    trace_dir=run_root / "trace",
                    timeout_s=int(args.timeout_seconds),
                    model=str(args.model),
                    reuse_existing=bool(args.reuse_existing),
                )
                debug_results.append(result)
                partial_by_candidate = {
                    item["id"]: [result for result in debug_results if result["candidate_id"] == item["id"]]
                    for item in candidates
                }
                report["debug_results"] = debug_results
                report["debug_summaries"] = {
                    key: _summarize_candidate(value) for key, value in partial_by_candidate.items()
                }
                _write_report(output_root, report)
        report["debug_results"] = debug_results
        debug_by_candidate = {
            candidate["id"]: [item for item in debug_results if item["candidate_id"] == candidate["id"]]
            for candidate in candidates
        }
        report["debug_summaries"] = {key: _summarize_candidate(value) for key, value in debug_by_candidate.items()}
        best_candidate = max(candidates, key=lambda candidate: report["debug_summaries"][candidate["id"]]["score"])
        report["best_candidate_id"] = best_candidate["id"]

        validation_results = []
        for seed in validation_seeds:
            run_root = output_root / "validation" / best_candidate["id"] / seed["id"]
            validation_results.append(
                _run_one(
                    repo_root=repo_root,
                    runner=runner,
                    candidate=best_candidate,
                    seed=seed,
                    config_path=output_root / "configs" / "validation" / f"{seed['id']}.yaml",
                    output_dir=run_root / "run",
                    visual_dir=run_root / "visual_artifacts",
                    trace_dir=run_root / "trace",
                    timeout_s=int(args.timeout_seconds),
                    model=str(args.model),
                    reuse_existing=bool(args.reuse_existing),
                )
            )
        report["validation_results"] = validation_results
        report["validation_summaries"] = {best_candidate["id"]: _summarize_candidate(validation_results)}

    _write_report(output_root, report)
    print(json.dumps(jsonable({"output_root": str(output_root), "best_candidate_id": report.get("best_candidate_id"), "mode": report["mode"]}), indent=2))


if __name__ == "__main__":
    main()
