from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_x2_runs.py"
    spec = importlib.util.spec_from_file_location("summarize_x2_runs", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_acceptance_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_x2_acceptance.py"
    spec = importlib.util.spec_from_file_location("check_x2_acceptance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_audit_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_x2_capx_integration.py"
    spec = importlib.util.spec_from_file_location("audit_x2_capx_integration", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_trial(
    tmp_path: Path,
    *,
    run_name: str,
    target: str,
    tcp_error: float = 0.012,
    ori_error: float = 0.034,
    place_error: float = 0.056,
    ok: bool = True,
):
    trial_dir = (
        tmp_path
        / "outputs"
        / "stability"
        / run_name
        / f"trial_01_sandboxrc_0_reward_{1.0 if ok else 0.0:.3f}_taskcompleted_{1 if ok else 0}"
    )
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.txt").write_text(
        "\n".join(
            [
                "Environment response:",
                f"  Stdout: X2_TWO_TARGET_RESULT ok={ok} target={target} "
                f"before_close_tcp_error_m={tcp_error} before_close_ori_error_rad={ori_error} "
                f"object_in_hand_after_close={ok} place_error_m={place_error}",
                "X2_TWO_TARGET_ATTEMPT candidate_index=0 ok=True before_close_reached=True "
                f"object_in_hand_after_close={ok} before_close_tcp_error_m={tcp_error} "
                f"before_close_ori_error_rad={ori_error}",
                f"  Reward: {1.0 if ok else 0.0}",
                f"  Task Completed: {ok}",
            ]
        ),
        encoding="utf-8",
    )
    (trial_dir / "code.py").write_text(
        f"RESULT = pick_and_place_red_cube_to_{target}_target()\n",
        encoding="utf-8",
    )
    (trial_dir / "video_combined_global.mp4").write_bytes(b"fake")
    visual_dir = (
        tmp_path
        / "outputs"
        / "x2_visual_artifacts"
        / "stability"
        / run_name
        / "x2_pick_place_red_cube_000"
    )
    visual_dir.mkdir(parents=True)
    (visual_dir / "grasp_summary.json").write_text("{}", encoding="utf-8")
    return trial_dir


def test_summarize_x2_runs_parses_trial_metrics_and_artifacts(tmp_path):
    module = _load_summary_module()
    run_name = "two_targets_right_api_stability_stamp_run01"
    trial_dir = (
        tmp_path
        / "outputs"
        / "stability"
        / run_name
        / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    )
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.txt").write_text(
        "\n".join(
            [
                "Environment response:",
                "  Stdout: X2_TWO_TARGET_RESULT ok=True target=right "
                "before_close_tcp_error_m=0.012 before_close_ori_error_rad=0.034 "
                "object_in_hand_after_close=True place_error_m=0.056",
                "X2_TWO_TARGET_ATTEMPT candidate_index=0 ok=True before_close_reached=True "
                "object_in_hand_after_close=True before_close_tcp_error_m=0.012 "
                "before_close_ori_error_rad=0.034",
                "  Reward: 1.0",
                "  Task Completed: True",
            ]
        ),
        encoding="utf-8",
    )
    (trial_dir / "code.py").write_text("RESULT = pick_and_place_red_cube_to_right_target()\n", encoding="utf-8")
    (trial_dir / "video_combined_global.mp4").write_bytes(b"fake")
    visual_dir = (
        tmp_path
        / "outputs"
        / "x2_visual_artifacts"
        / "stability"
        / run_name
        / "x2_pick_place_red_cube_000"
    )
    visual_dir.mkdir(parents=True)
    (visual_dir / "grasp_summary.json").write_text("{}", encoding="utf-8")

    rows = [module._parse_trial(tmp_path, trial_dir)]

    assert rows[0]["reward"] == "1.000"
    assert rows[0]["task_completed"] == "1"
    assert rows[0]["result"]["ok"] == "True"
    assert rows[0]["result"]["target"] == "right"
    assert rows[0]["result"]["before_close_tcp_error_m"] == "0.012"
    assert rows[0]["attempts"] == [
        {
            "candidate_index": "0",
            "ok": "True",
            "before_close_reached": "True",
            "object_in_hand_after_close": "True",
            "before_close_tcp_error_m": "0.012",
            "before_close_ori_error_rad": "0.034",
        }
    ]
    assert rows[0]["videos"] == [str(trial_dir / "video_combined_global.mp4")]
    assert rows[0]["visual_artifact_dirs"] == [str(visual_dir)]


def test_summarize_x2_runs_parses_two_object_result(tmp_path):
    module = _load_summary_module()
    run_name = "two_objects_blue_right_oracle_smoke"
    trial_dir = (
        tmp_path
        / "outputs"
        / "oracle"
        / run_name
        / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    )
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.txt").write_text(
        "\n".join(
            [
                "Environment response:",
                "  Stdout: X2_TWO_OBJECT_RESULT ok=True object=x2_pick_place_blue_cube target=right "
                "before_close_tcp_error_m=0.009 before_close_ori_error_rad=0.020 "
                "object_in_hand_after_close=True place_error_m=0.031",
                "X2_TWO_OBJECT_ATTEMPT candidate_index=1 ok=True before_close_reached=True "
                "object_in_hand_after_close=True before_close_tcp_error_m=0.009 "
                "before_close_ori_error_rad=0.020",
            ]
        ),
        encoding="utf-8",
    )

    row = module._parse_trial(tmp_path, trial_dir)

    assert row["result"]["ok"] == "True"
    assert row["result"]["object"] == "x2_pick_place_blue_cube"
    assert row["result"]["target"] == "right"
    assert row["result"]["before_close_tcp_error_m"] == "0.009"
    assert row["attempts"][0]["candidate_index"] == "1"


def test_summarize_x2_runs_matches_two_object_visual_artifact_prefix(tmp_path):
    module = _load_summary_module()
    run_name = "x2_pick_place_two_objects_blue_right_rgbd_visual_stamp"
    trial_dir = (
        tmp_path
        / "outputs"
        / "oracle"
        / run_name
        / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    )
    trial_dir.mkdir(parents=True)
    (trial_dir / "summary.txt").write_text(
        "X2_TWO_OBJECT_RESULT ok=True object=x2_pick_place_blue_cube target=right "
        "before_close_tcp_error_m=0.009 before_close_ori_error_rad=0.020 "
        "object_in_hand_after_close=True place_error_m=0.031",
        encoding="utf-8",
    )
    visual_dir = (
        tmp_path
        / "outputs"
        / "x2_visual_artifacts"
        / "two_objects_blue_right_rgbd_visual_stamp"
        / "x2_pick_place_blue_cube_000"
    )
    visual_dir.mkdir(parents=True)
    (visual_dir / "grasp_summary.json").write_text("{}", encoding="utf-8")

    row = module._parse_trial(tmp_path, trial_dir)

    assert row["visual_artifact_dirs"] == [str(visual_dir)]


def test_check_x2_acceptance_passes_right_and_left_runs(tmp_path):
    module = _load_acceptance_module()
    right = _write_trial(tmp_path, run_name="two_targets_right_api_stability_stamp_run01", target="right")
    left = _write_trial(tmp_path, run_name="two_targets_left_api_stability_stamp_run01", target="left")

    class Args:
        paths = [str(right.parent), str(left.parent)]
        repo_root = str(tmp_path)
        require_targets = "right,left"
        max_tcp_error_m = 0.02
        max_ori_error_rad = 0.10
        max_place_error_m = 0.10
        allow_missing_video = False
        allow_missing_visual = False
        allow_missing_code = False

    ok, checked = module.check_acceptance(Args)

    assert ok is True
    assert len(checked) == 2


def test_check_x2_acceptance_fails_missing_required_target(tmp_path):
    module = _load_acceptance_module()
    right = _write_trial(tmp_path, run_name="two_targets_right_api_stability_stamp_run01", target="right")

    class Args:
        paths = [str(right.parent)]
        repo_root = str(tmp_path)
        require_targets = "right,left"
        max_tcp_error_m = 0.02
        max_ori_error_rad = 0.10
        max_place_error_m = 0.10
        allow_missing_video = False
        allow_missing_visual = False
        allow_missing_code = False

    ok, checked = module.check_acceptance(Args)

    assert ok is False
    assert any("missing required target" in "; ".join(item["errors"]) for item in checked)


def test_check_x2_acceptance_reads_full_code_and_rejects_extra_primitive_call(tmp_path):
    module = _load_acceptance_module()
    right = _write_trial(tmp_path, run_name="two_targets_right_api_stability_stamp_run01", target="right")
    left = _write_trial(tmp_path, run_name="two_targets_left_api_stability_stamp_run01", target="left")
    (right / "code.py").write_text(
        "\n".join(
            [
                "RESULT = pick_and_place_red_cube_to_right_target()",
                *(f"# filler {idx}" for idx in range(20)),
                "OTHER = pick_and_place_red_cube_to_left_target()",
            ]
        ),
        encoding="utf-8",
    )

    class Args:
        paths = [str(right.parent), str(left.parent)]
        repo_root = str(tmp_path)
        require_targets = "right,left"
        max_tcp_error_m = 0.02
        max_ori_error_rad = 0.10
        max_place_error_m = 0.10
        allow_missing_video = False
        allow_missing_visual = False
        allow_missing_code = False

    ok, checked = module.check_acceptance(Args)

    assert ok is False
    assert any("exactly one target primitive" in "; ".join(item["errors"]) for item in checked)


def test_audit_reports_direct_api_pending_when_only_oracle_evidence_exists(tmp_path, monkeypatch):
    module = _load_audit_module()
    required_paths = [
        "capx/integrations/x2/control.py",
        "docs/x2-capx-integration-status.md",
    ]
    monkeypatch.setattr(module, "REQUIRED_PATHS", required_paths)
    monkeypatch.setattr(
        module,
        "ORACLE_EVIDENCE_PATHS",
        [
            "outputs/oracle/stability/oracle/two_targets_right",
            "outputs/oracle/stability/oracle/two_targets_left",
        ],
    )
    monkeypatch.setattr(module, "NON_ORACLE_EVIDENCE_GLOBS", ["outputs/stability/two_targets_*_api_stability_*_run*"])

    control = tmp_path / "capx" / "integrations" / "x2" / "control.py"
    control.parent.mkdir(parents=True)
    control.write_text(
        "\n".join(
            [
                "class X2PickPlaceApi: pass",
                '"pick_and_place_red_cube"',
                '"pick_and_place_red_cube_to_left_target"',
                '"pick_and_place_red_cube_to_right_target"',
                '"pick_and_place_visual_object"',
                "T_world_tcp",
            ]
        ),
        encoding="utf-8",
    )
    status_doc = tmp_path / "docs" / "x2-capx-integration-status.md"
    status_doc.parent.mkdir(parents=True)
    status_doc.write_text(
        "\n".join(
            [
                "docs/x2-accepted-baseline-20260630.md",
                "Accepted non-oracle",
                "scripts/run_x2_two_target_api_stability_and_check.sh",
                "scripts/run_x2_two_target_codex_a_stability_and_check.sh",
                "scripts/check_x2_acceptance.py",
            ]
        ),
        encoding="utf-8",
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    source_summary = Path(__file__).resolve().parents[1] / "scripts" / "summarize_x2_runs.py"
    source_acceptance = Path(__file__).resolve().parents[1] / "scripts" / "check_x2_acceptance.py"
    (scripts_dir / "summarize_x2_runs.py").write_text(source_summary.read_text(encoding="utf-8"), encoding="utf-8")
    (scripts_dir / "check_x2_acceptance.py").write_text(source_acceptance.read_text(encoding="utf-8"), encoding="utf-8")

    _write_trial(tmp_path, run_name="two_targets_right", target="right")
    _write_trial(tmp_path, run_name="two_targets_left", target="left")
    oracle_root = tmp_path / "outputs" / "oracle" / "stability" / "oracle"
    right_trial = tmp_path / "outputs" / "stability" / "two_targets_right"
    left_trial = tmp_path / "outputs" / "stability" / "two_targets_left"
    for source, target in [
        (right_trial, oracle_root / "two_targets_right"),
        (left_trial, oracle_root / "two_targets_left"),
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    result = module.audit(tmp_path)

    assert result["required_local_baseline_ok"] is True
    assert result["direct_api_evidence"]["status"] == "pending"
    assert result["complete"] is False
