from pathlib import Path

from capx.integrations.x2.control import X2PickPlaceApi
from capx.envs.tasks.x2.x2_pick_place_red_cube_two_targets import (
    X2PickPlaceRedCubeTwoTargetsCodeEnv,
    X2PickPlaceRedCubeTwoTargetsLeftCodeEnv,
)
from capx.envs.tasks.x2.x2_pick_place_two_objects import X2PickPlaceTwoObjectsBlueRightCodeEnv
from capx.envs.tasks.x2.x2_pick_place_two_objects import X2PickPlaceTwoObjectsBlueRightRgbdVisualCodeEnv


class _DummyEnv:
    pass


def test_x2_pick_place_api_exposes_only_reduced_primitives():
    api = X2PickPlaceApi(_DummyEnv())

    assert set(api.functions()) == {
        "pick_and_place_red_cube",
        "pick_and_place_red_cube_to_left_target",
        "pick_and_place_red_cube_to_right_target",
        "pick_and_place_visual_object",
    }


def test_x2_pick_place_api_prompt_docs_include_pose_contract():
    docs = X2PickPlaceApi(_DummyEnv()).combined_doc()

    assert "pick_and_place_red_cube" in docs
    assert "pick_and_place_visual_object" in docs
    assert "T_world_tcp" in docs
    assert "place_position" in docs
    assert "sim-known" in docs
    assert "obstacle_source" in docs
    assert "rgbd_visual" in docs
    assert "place_offset_source" in docs
    assert "visual_grasp_pose" in docs


def test_x2_pick_place_api_does_not_expose_debug_motion_or_camera_helpers():
    names = set(X2PickPlaceApi(_DummyEnv()).functions())

    assert "get_chest_camera_name" not in names
    assert "plan_visual_grasp_tcp_pose" not in names
    assert "execute_tcp_grasp_plan" not in names
    assert "move_tcp_joint_ik" not in names
    assert "open_gripper" not in names


def test_x2_two_target_task_variants_select_expected_primitives():
    assert X2PickPlaceRedCubeTwoTargetsCodeEnv.target_name == "right"
    assert "pick_and_place_red_cube_to_right_target()" in X2PickPlaceRedCubeTwoTargetsCodeEnv.prompt
    assert X2PickPlaceRedCubeTwoTargetsCodeEnv.oracle_code.strip().splitlines()[0] == (
        "RESULT = pick_and_place_red_cube_to_right_target()"
    )

    assert X2PickPlaceRedCubeTwoTargetsLeftCodeEnv.target_name == "left"
    assert "pick_and_place_red_cube_to_left_target()" in X2PickPlaceRedCubeTwoTargetsLeftCodeEnv.prompt
    assert X2PickPlaceRedCubeTwoTargetsLeftCodeEnv.oracle_code.strip().splitlines()[0] == (
        "RESULT = pick_and_place_red_cube_to_left_target()"
    )


def test_x2_two_object_task_uses_generic_visual_pick_place_primitive():
    prompt = X2PickPlaceTwoObjectsBlueRightCodeEnv.prompt
    oracle = X2PickPlaceTwoObjectsBlueRightCodeEnv.oracle_code

    assert X2PickPlaceTwoObjectsBlueRightCodeEnv.target_object_name == "x2_pick_place_blue_cube"
    assert X2PickPlaceTwoObjectsBlueRightCodeEnv.distractor_object_name == "x2_pick_place_red_cube"
    assert 'pick_and_place_visual_object(' in prompt
    assert 'pick_and_place_visual_object(' in oracle
    assert 'prompts=["blue cube", "blue block", "blue box"]' in oracle
    assert "pick_and_place_red_cube_to_right_target" not in oracle
    assert "plan_visual_grasp_tcp_pose" not in oracle
    assert "execute_tcp_grasp_plan" not in oracle


def test_x2_two_object_rgbd_visual_variant_selects_experimental_route():
    prompt = X2PickPlaceTwoObjectsBlueRightRgbdVisualCodeEnv.prompt
    oracle = X2PickPlaceTwoObjectsBlueRightRgbdVisualCodeEnv.oracle_code

    assert 'pick_and_place_visual_object(' in prompt
    assert 'pick_and_place_visual_object(' in oracle
    assert 'obstacle_source="rgbd_visual"' in oracle
    assert 'place_offset_source="visual_grasp_pose"' in oracle
    assert "sim_place_correction_steps=0" in oracle
    assert "reobserve_at_precontact=True" in oracle
    assert "candidate_indices=(1, 2)" in oracle
    assert "place_descent_waypoints=4" in oracle
    assert "get_sim_known_tabletop_obstacles" not in oracle
    assert "execute_tcp_grasp_plan" not in oracle


def test_x2_integration_status_documents_current_acceptance_boundary():
    repo_root = Path(__file__).resolve().parents[1]
    status_doc = (repo_root / "docs" / "x2-capx-integration-status.md").read_text(encoding="utf-8")
    scripts_readme = (repo_root / "scripts" / "README.md").read_text(encoding="utf-8")

    assert "X2PickPlaceApi" in status_doc
    assert "T_world_tcp" in status_doc
    assert "docs/x2-accepted-baseline-20260630.md" in status_doc
    assert "Accepted non-oracle" in status_doc
    assert "X2_INTEGRATION_AUDIT COMPLETE" in status_doc
    assert "scripts/run_x2_two_target_api_stability_smoke.sh" in scripts_readme
    assert "scripts/archive/x2_experiments/" in scripts_readme
