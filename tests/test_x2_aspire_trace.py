from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from capx.integrations.x2.aspire import build_trace_bundles, classify_failure


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_classify_failure_reports_object_not_in_hand():
    report = classify_failure(
        {
            "ok": False,
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": True,
                "before_close_reached": True,
                "object_in_hand_after_close": False,
                "before_close_error": {"tcp_error_m": 0.01, "ori_error_rad": 0.02},
            },
        }
    )

    assert report["status"] == "failure"
    assert report["primary_failure"] == "object_not_in_hand_after_close"
    assert "increase_grasp_tcp_axis_offsets" in report["suggested_repair_tags"]


def test_classify_failure_prefers_empty_gripper_after_close_over_precontact_flag():
    report = classify_failure(
        {
            "ok": False,
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": False,
                "close_attempted": True,
                "before_close_reached": True,
                "object_in_hand_after_close": False,
                "before_close_error": {"tcp_error_m": 0.008, "ori_error_rad": 0.024},
            },
        }
    )

    assert report["primary_failure"] == "object_not_in_hand_after_close"


def test_classify_failure_prefers_place_failures_after_successful_grasp():
    report = classify_failure(
        {
            "ok": False,
            "place_position_threshold": 0.10,
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": False,
                "close_attempted": True,
                "before_close_reached": True,
                "object_in_hand_after_close": True,
                "before_close_error": {"tcp_error_m": 0.009, "ori_error_rad": 0.03},
                "place": {"requested": True, "place_pre_ok": False, "place_error_m": 0.129},
            },
        }
    )

    assert report["primary_failure"] == "place_error_too_large"

    report = classify_failure(
        {
            "ok": False,
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": False,
                "close_attempted": True,
                "before_close_reached": True,
                "object_in_hand_after_close": True,
                "before_close_error": {"tcp_error_m": 0.009, "ori_error_rad": 0.03},
                "place": {"requested": True, "place_pre_ok": False, "place_error_m": None},
            },
        }
    )

    assert report["primary_failure"] == "place_pre_pose_not_reached"


def test_classify_preclose_failure_uses_static_ik_reachability():
    report = classify_failure(
        {
            "ok": False,
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": True,
                "before_close_reached": False,
                "before_close_error": {"tcp_error_m": 0.08, "ori_error_rad": 0.4},
                "active_plan_reachability": {
                    "grasp": {
                        "ok": False,
                        "fk_pos_error_m": 0.12,
                        "fk_ori_error_rad": 0.8,
                    }
                },
            },
        }
    )

    assert report["primary_failure"] == "grasp_pose_unreachable"
    assert "try_next_grasp_candidate" in report["suggested_repair_tags"]

    report = classify_failure(
        {
            "ok": False,
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": True,
                "before_close_reached": False,
                "before_close_error": {
                    "tcp_error_m": 0.08,
                    "ori_error_rad": 0.4,
                    "motion_debug": {
                        "last_joint_ik_move_debug": {
                            "solve": {"solve_fk_pos_error_m": 0.001, "solve_fk_ori_error_rad": 0.01},
                            "joint_move": {"max_final_joint_error_rad": 0.05},
                            "joint_ok": False,
                            "final_pos_error_m": 0.08,
                            "final_ori_error_rad": 0.4,
                        }
                    },
                },
                "active_plan_reachability": {
                    "grasp": {
                        "ok": True,
                        "fk_pos_error_m": 0.001,
                        "fk_ori_error_rad": 0.01,
                    }
                },
            },
        }
    )

    assert report["primary_failure"] == "preclose_pose_not_reached"
    assert report["evidence"]["active_grasp_ik_reachable"] is True
    assert report["evidence"]["preclose_ik_solve_fk_pos_error_m"] == 0.001
    assert report["evidence"]["preclose_joint_final_error_rad"] == 0.05


def test_build_trace_bundle_classifies_trial_timeout(tmp_path):
    repo = tmp_path
    run_name = "x2_pick_place_two_objects_blue_right_rgbd_visual_oracle_timeout"
    trial = repo / "outputs" / "oracle" / run_name / "trial_01_sandboxrc_1_reward_0.000_taskcompleted_0"
    trial.mkdir(parents=True)
    (trial / "code.py").write_text("RESULT = pick_and_place_visual_object('x2_pick_place_blue_cube', [0.37, 0.055, 0.921])\n", encoding="utf-8")
    (trial / "summary.txt").write_text(
        "Environment response:\n"
        "  Stderr: TimeoutError: Trial 1 exceeded 1000 seconds\n"
        "  Reward: 0.0\n"
        "  Task Completed: False\n",
        encoding="utf-8",
    )

    results = build_trace_bundles(repo_root=repo, paths=[trial.parent], output_root=repo / "outputs" / "x2_aspire_traces")

    assert len(results) == 1
    failure = json.loads((results[0].bundle_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure["primary_failure"] == "timeout"
    assert "reduce_candidate_count" in failure["suggested_repair_tags"]


def test_build_trace_bundle_from_saved_trial_and_visual_artifacts(tmp_path):
    repo = tmp_path
    run_name = "x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_stamp"
    trial = repo / "outputs" / "codex-a" / run_name / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial.mkdir(parents=True)
    (trial / "code.py").write_text("RESULT = pick_and_place_visual_object('x2_pick_place_blue_cube', [0.37, 0.055, 0.921])\n", encoding="utf-8")
    (trial / "summary.txt").write_text(
        "X2_TWO_OBJECT_RESULT ok=True obstacle_source=rgbd_visual place_offset_source=visual_grasp_pose "
        "reobserve_adopted=True reobserve_reason=quality_gates_passed before_close_tcp_error_m=0.01 "
        "before_close_ori_error_rad=0.02 object_in_hand_after_close=True place_error_m=0.03 place_descent_waypoints=4\n",
        encoding="utf-8",
    )
    (trial / "video_combined_global.mp4").write_bytes(b"video")

    visual = repo / "outputs" / "x2_visual_artifacts" / run_name / "x2_pick_place_blue_cube_001"
    _write_json(
        visual / "grasp_summary.json",
        {
            "object_name": "x2_pick_place_blue_cube",
            "mask_pixels": 1200,
            "grasp_tcp_pose": [[0.32, -0.08, 0.93], [0.0, 0.0, 0.0, 1.0]],
            "precontact_tcp_pose": [[0.32, -0.16, 0.96], [0.0, 0.0, 0.0, 1.0]],
        },
    )
    _write_json(
        visual / "visual_obstacles.json",
        {
            "ok": True,
            "sim_truth": False,
            "obstacles_world": [{"name": "x2_pick_place_blue_cube", "source": "rgbd_object_mask_aabb"}],
        },
    )
    _write_json(
        visual / "pick_place_result_summary.json",
        {
            "ok": True,
            "source": "x2_visual_pick_place",
            "object_name": "x2_pick_place_blue_cube",
            "place_position_world": [0.37, 0.055, 0.921],
            "place_position_threshold": 0.1,
            "obstacle_source": "rgbd_visual",
            "obstacle_plan": {"ok": True, "sim_truth": False},
            "obstacles_world": [{"name": "x2_pick_place_blue_cube", "source": "rgbd_object_mask_aabb"}],
            "plan": {
                "ok": True,
                "grasp_tcp_pose": [[0.32, -0.08, 0.93], [0.0, 0.0, 0.0, 1.0]],
                "precontact_tcp_pose": [[0.32, -0.16, 0.96], [0.0, 0.0, 0.0, 1.0]],
                "selection": {"selected_rank": 1, "selected_raw_index": 6},
            },
            "execution": {
                "ok": True,
                "precontact_ok": True,
                "before_close_reached": True,
                "before_close_error": {
                    "tcp_error_m": 0.01,
                    "ori_error_rad": 0.02,
                    "reached_tcp_pose": [[0.32, -0.08, 0.93], [0.0, 0.0, 0.0, 1.0]],
                },
                "precontact_reobserve": {"enabled": True, "adopted": True, "reason": "quality_gates_passed"},
                "object_in_hand_after_close": True,
                "place": {
                    "place_offset_source": "visual_grasp_pose",
                    "place_orientation_source": "post_lift_current",
                    "place_error_m": 0.03,
                    "place_descent_waypoints": 4,
                    "transfer_waypoints": [
                        {
                            "name": "place_pre_descent",
                            "target_tcp_pose": [[0.37, 0.055, 1.02], [0.0, 0.0, 0.0, 1.0]],
                            "reached_tcp_pose": [[0.372, 0.054, 1.019], [0.0, 0.0, 0.0, 1.0]],
                            "tcp_error_m": 0.0024,
                            "ori_error_rad": 0.01,
                        }
                    ],
                    "descent_waypoints": [
                        {
                            "name": "release",
                            "target_tcp_pose": [[0.37, 0.055, 0.94], [0.0, 0.0, 0.0, 1.0]],
                            "reached_tcp_pose": [[0.371, 0.055, 0.941], [0.0, 0.0, 0.0, 1.0]],
                            "tcp_error_m": 0.0014,
                            "ori_error_rad": 0.0,
                        }
                    ],
                },
            },
            "attempts": [{"candidate_index": 1, "ok": True}],
        },
    )

    results = build_trace_bundles(repo_root=repo, paths=[trial.parent], output_root=repo / "outputs" / "x2_aspire_traces")

    assert len(results) == 1
    bundle = results[0].bundle_dir
    assert (bundle / "run_meta.json").exists()
    assert (bundle / "generated_code.py").exists()
    assert (bundle / "primitive_calls.jsonl").exists()
    assert (bundle / "metrics.json").exists()
    assert (bundle / "failure_report.json").exists()
    assert (bundle / "visual" / "initial" / "grasp_summary.json").exists()
    assert (bundle / "motion" / "planned_tcp_path.json").exists()
    assert (bundle / "videos" / "video_combined_global.mp4").exists()

    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    failure = json.loads((bundle / "failure_report.json").read_text(encoding="utf-8"))
    skills = json.loads((bundle / "skill_library.json").read_text(encoding="utf-8"))
    calls = (bundle / "primitive_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert metrics["obstacle_source"] == "rgbd_visual"
    assert metrics["place_orientation_source"] == "post_lift_current"
    assert metrics["place_pre_tcp_error_m"] == 0.0024
    assert metrics["place_pre_ori_error_rad"] == 0.01
    assert metrics["rgbd_obstacles_sim_truth"] is False
    assert failure["status"] == "success"
    assert any(skill["id"] == "rgbd_tabletop_obstacle_box_planning" for skill in skills["skills"])
    assert any('"pick_and_place_visual_object"' in line for line in calls)
    tracking_errors = json.loads((bundle / "motion" / "tracking_errors.json").read_text(encoding="utf-8"))
    assert any(item["stage"] == "transfer" and item["name"] == "place_pre_descent" for item in tracking_errors)


def test_build_trace_bundle_handles_failed_empty_gripper_trial(tmp_path):
    repo = tmp_path
    run_name = "x2_pick_place_two_objects_blue_right_rgbd_visual_oracle_failed"
    trial = repo / "outputs" / "oracle" / run_name / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial.mkdir(parents=True)
    (trial / "code.py").write_text("RESULT = pick_and_place_visual_object('x2_pick_place_blue_cube', [0.37, 0.055, 0.921])\n", encoding="utf-8")
    (trial / "summary.txt").write_text(
        "X2_TWO_OBJECT_RESULT ok=False obstacle_source=rgbd_visual place_offset_source=visual_grasp_pose "
        "reobserve_adopted=True reobserve_reason=quality_gates_passed before_close_tcp_error_m=0.008 "
        "before_close_ori_error_rad=0.024 object_in_hand_after_close=False place_error_m=None "
        "place_descent_waypoints=None\n",
        encoding="utf-8",
    )

    visual = repo / "outputs" / "x2_visual_artifacts" / run_name / "x2_pick_place_blue_cube_001"
    _write_json(
        visual / "grasp_summary.json",
        {"object_name": "x2_pick_place_blue_cube", "mask_pixels": 1200},
    )
    _write_json(
        visual / "visual_obstacles.json",
        {"ok": True, "sim_truth": False, "obstacles_world": []},
    )
    _write_json(
        visual / "pick_place_result_summary.json",
        {
            "ok": False,
            "object_name": "x2_pick_place_blue_cube",
            "obstacle_source": "rgbd_visual",
            "obstacle_plan": {"ok": True, "sim_truth": False},
            "plan": {"ok": True},
            "execution": {
                "ok": False,
                "precontact_ok": False,
                "close_attempted": True,
                "before_close_reached": True,
                "before_close_error": {"tcp_error_m": 0.008, "ori_error_rad": 0.024},
                "object_in_hand_after_close": False,
                "place": {"place_offset_source": "visual_grasp_pose", "place_error_m": None},
            },
        },
    )

    results = build_trace_bundles(repo_root=repo, paths=[trial.parent], output_root=repo / "outputs" / "x2_aspire_traces")

    assert len(results) == 1
    failure = json.loads((results[0].bundle_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure["primary_failure"] == "object_not_in_hand_after_close"


def test_candidate_search_plan_only_writes_debug_validation_configs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_x2_aspire_rgbd_candidate_search.py"
    spec = importlib.util.spec_from_file_location("x2_candidate_search", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    base = tmp_path / "base.yaml"
    base.write_text(
        """
env:
  cfg:
    low_level:
      objects:
        - name: x2_pick_place_blue_cube
          position: [0.32, -0.08, 0.921]
        - name: x2_pick_place_red_cube
          position: [0.24, -0.09, 0.921]
        - name: x2_pick_place_right_target_marker
          position: [0.37, 0.055, 0.921]
""",
        encoding="utf-8",
    )
    out = tmp_path / "debug_nominal.yaml"
    module._write_seed_config(base, out, module.DEBUG_SEEDS[0])
    text = out.read_text(encoding="utf-8")
    assert "x2_pick_place_blue_cube" in text
    assert "0.32" in text
    assert module.DEFAULT_CANDIDATES[0]["params"]["CAPX_X2_RGBD_CANDIDATE_INDICES"] == "1,2"
    assert module.DEFAULT_CANDIDATES[0]["params"]["CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT"] == "1"
    assert module.DEFAULT_CANDIDATES[-1]["params"]["CAPX_X2_RGBD_REOBSERVE_AT_PRECONTACT"] == "0"


def test_aspire_acceptance_audit_passes_complete_report():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_x2_aspire_acceptance.py"
    spec = importlib.util.spec_from_file_location("x2_aspire_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = {
        "mode": "execute_aggregate",
        "candidates": [{"id": "stable_rgbd_v1"}, {"id": "repair_v2"}],
        "debug_seeds": [{"id": "d0"}, {"id": "d1"}, {"id": "d2"}],
        "validation_seeds": [{"id": "v0"}, {"id": "v1"}, {"id": "v2"}],
        "best_candidate_id": "repair_v2",
        "debug_summaries": {
            "stable_rgbd_v1": {"trials": 3, "successes": 1},
            "repair_v2": {"trials": 3, "successes": 3},
        },
        "validation_summaries": {
            "repair_v2": {
                "trials": 3,
                "successes": 3,
                "avg_before_close_tcp_error_m": 0.012,
                "avg_before_close_ori_error_rad": 0.03,
                "avg_place_error_m": 0.02,
                "trace_bundles": 3,
                "videos": 3,
            }
        },
        "debug_results": [
            {
                "metrics": {"obstacle_source": "rgbd_visual", "rgbd_obstacles_sim_truth": False},
                "failure_report": {"primary_failure": "object_not_in_hand_after_close"},
            },
            {
                "metrics": {"obstacle_source": "rgbd_visual", "rgbd_obstacles_sim_truth": False},
                "failure_report": {"primary_failure": None},
            },
        ],
        "validation_results": [
            {
                "metrics": {"obstacle_source": "rgbd_visual", "rgbd_obstacles_sim_truth": False},
                "failure_report": {"primary_failure": None},
            }
        ],
    }

    audit = module.audit_report(report)
    assert audit["ok"] is True


def test_aggregate_x2_aspire_evidence_from_trace_bundles(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_x2_aspire_evidence.py"
    spec = importlib.util.spec_from_file_location("x2_aspire_aggregate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = tmp_path / "search"
    bundle = root / "debug" / "stable_rgbd_v1" / "debug_nominal" / "trace" / "run__trial_01"
    _write_json(
        bundle / "metrics.json",
        {
            "ok": True,
            "task_completed": True,
            "reward": 1.0,
            "before_close_tcp_error_m": 0.01,
            "before_close_ori_error_rad": 0.02,
            "place_error_m": 0.03,
            "obstacle_source": "rgbd_visual",
            "rgbd_obstacles_sim_truth": False,
        },
    )
    _write_json(bundle / "failure_report.json", {"status": "success", "primary_failure": None})
    (bundle / "videos").mkdir(parents=True)
    (bundle / "videos" / "video_combined_global.mp4").write_bytes(b"video")

    failed = root / "debug" / "controlled_failure_fast_no_reobserve" / "debug_nominal" / "trace" / "run__trial_02"
    _write_json(
        failed / "metrics.json",
        {
            "ok": False,
            "task_completed": False,
            "reward": 0.0,
            "before_close_tcp_error_m": 0.008,
            "before_close_ori_error_rad": 0.024,
            "object_in_hand_after_close": False,
            "obstacle_source": "rgbd_visual",
            "rgbd_obstacles_sim_truth": False,
        },
    )
    _write_json(
        failed / "failure_report.json",
        {
            "status": "failure",
            "primary_failure": "object_not_in_hand_after_close",
            "suggested_repair_tags": ["increase_grasp_tcp_axis_offsets"],
        },
    )

    report = module.aggregate_evidence([root])
    assert report["mode"] == "execute_aggregate"
    assert len(report["debug_results"]) == 2
    assert report["debug_summaries"]["stable_rgbd_v1"]["successes"] == 1
    assert report["debug_summaries"]["controlled_failure_fast_no_reobserve"]["primary_failures"] == [
        "object_not_in_hand_after_close"
    ]


def test_aspire_acceptance_audit_fails_incomplete_report():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_x2_aspire_acceptance.py"
    spec = importlib.util.spec_from_file_location("x2_aspire_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    audit = module.audit_report(
        {
            "mode": "execute",
            "candidates": [{"id": "stable_rgbd_v1"}, {"id": "repair_v2"}],
            "debug_seeds": [{"id": "d0"}, {"id": "d1"}, {"id": "d2"}],
            "validation_seeds": [{"id": "v0"}, {"id": "v1"}, {"id": "v2"}],
            "debug_summaries": {"stable_rgbd_v1": {"trials": 1, "successes": 1}},
            "validation_summaries": {},
            "debug_results": [
                {
                    "metrics": {"obstacle_source": "rgbd_visual", "rgbd_obstacles_sim_truth": False},
                    "failure_report": {"primary_failure": None},
                }
            ],
        }
    )

    failed = {item["name"] for item in audit["checks"] if not item["ok"]}
    assert audit["ok"] is False
    assert "debug_candidate_search_ran" in failed
    assert "best_candidate_selected" in failed
    assert "validation_trials_ran" in failed
    assert "failure_taxonomy_evidence" in failed
