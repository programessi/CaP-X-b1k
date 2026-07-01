from __future__ import annotations

import numpy as np

from capx.integrations.x2.control import X2ControlApi


class _SceneTrapEnv:
    @property
    def scene(self):  # pragma: no cover - should never be touched in these tests
        raise AssertionError("RGB-D visual obstacle path must not access env.scene")


def _synthetic_visual_grasp_plan(tmp_path):
    height = 40
    width = 40
    cx = width / 2.0
    cy = height / 2.0
    intrinsic = np.array([[100.0, 0.0, cx], [0.0, 100.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Camera is at z=1.0 looking along its OpenGL -Z axis.  Table pixels at
    # depth 1.02 land at world z=-0.02; object-mask pixels at depth 1.0 land
    # around world z=0.0.
    depth = np.full((height, width), 1.02, dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)
    mask[17:23, 17:23] = True
    depth[mask] = 1.0

    return {
        "ok": True,
        "object_name": "x2_pick_place_blue_cube",
        "visual_artifact_dir": str(tmp_path / "visual_artifacts"),
        "camera": {
            "intrinsic_matrix": intrinsic,
            "position_world": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "quat_xyzw_world": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        },
        "pose_estimate": {
            "object_name": "x2_pick_place_blue_cube",
            "position_world": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "bbox_extent": np.array([0.04, 0.04, 0.04], dtype=np.float64),
        },
        "visual": {
            "mask": mask,
            "depth": depth,
            "expected_depth": 1.0,
            "depth_window": 0.05,
        },
    }


def test_rgbd_visual_tabletop_obstacles_are_estimated_without_scene_truth(tmp_path):
    api = X2ControlApi(_SceneTrapEnv())
    plan = _synthetic_visual_grasp_plan(tmp_path)

    result = api.get_rgbd_visual_tabletop_obstacles(
        plan,
        object_name="x2_pick_place_blue_cube",
        table_name="x2_pick_place_table",
        workspace_bounds={"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (-0.2, 0.1)},
        object_margin=0.01,
        table_margin_xy=0.02,
        table_margin_z=0.006,
        table_thickness=0.024,
    )

    assert result["ok"] is True
    assert result["sim_truth"] is False
    assert result["errors"] == []
    assert result["point_counts"]["object_points"] >= 16
    assert result["point_counts"]["table_plane_points"] >= 16

    obstacles = result["obstacles_world"]
    assert [obstacle["source"] for obstacle in obstacles] == [
        "rgbd_object_mask_aabb",
        "rgbd_table_plane_aabb",
    ]
    assert obstacles[0]["name"] == "x2_pick_place_blue_cube"
    assert np.allclose(obstacles[0]["position"], [0.0, 0.0, 0.0])
    assert np.all(np.asarray(obstacles[0]["extent"], dtype=np.float64) >= np.array([0.06, 0.06, 0.06]))
    assert obstacles[1]["name"] == "x2_pick_place_table"
    assert obstacles[1]["position"][2] < 0.0

    artifact_dir = tmp_path / "visual_artifacts"
    assert (artifact_dir / "visual_obstacles.json").exists()
    assert (artifact_dir / "object_obstacle_points_world.npy").exists()
    assert (artifact_dir / "table_obstacle_points_world.npy").exists()


class _PickPlaceSwitchApi(X2ControlApi):
    def __init__(self):
        super().__init__(_SceneTrapEnv())
        self.sim_known_called = False
        self.execute_kwargs = None

    def get_chest_camera_name(self):
        return "x2_chest_camera"

    def settle_robot(self, steps: int = 8):
        return True

    def open_gripper(self, arm: int = 1):
        return True

    def get_sim_known_tabletop_obstacles(self, *args, **kwargs):  # pragma: no cover - should not be called
        self.sim_known_called = True
        raise AssertionError("rgbd_visual path must not call sim-known obstacle primitive")

    def plan_visual_grasp_tcp_pose(self, *args, **kwargs):
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return {
            "ok": True,
            "object_name": "x2_pick_place_blue_cube",
            "pose_estimate": {
                "position_world": np.array([0.30, -0.08, 0.92], dtype=np.float64),
            },
            "graspnet": {"candidates": []},
            "selection": {"selected_rank": 0, "selected_raw_index": 0},
            "grasp_tcp_pose": (np.array([0.32, -0.08, 0.92], dtype=np.float64), quat),
            "precontact_tcp_pose": (np.array([0.32, -0.08, 1.00], dtype=np.float64), quat),
            "insertion_waypoints": [],
        }

    def get_rgbd_visual_tabletop_obstacles(self, *args, **kwargs):
        return {
            "ok": True,
            "source": "rgbd_visual_tabletop_obstacles",
            "obstacles_world": [
                {
                    "type": "box",
                    "name": "x2_pick_place_blue_cube",
                    "position": [0.30, -0.08, 0.92],
                    "extent": [0.06, 0.06, 0.06],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "source": "rgbd_object_mask_aabb",
                },
                {
                    "type": "box",
                    "name": "x2_pick_place_table",
                    "position": [0.32, -0.01, 0.88],
                    "extent": [0.24, 0.24, 0.04],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "source": "rgbd_table_plane_aabb",
                },
            ],
            "point_counts": {"object_points": 36, "table_candidate_points": 1200, "table_plane_points": 900},
            "visual_artifact_dir": "/tmp/x2_visual_artifacts",
            "sim_truth": False,
        }

    def execute_tcp_grasp_plan(self, plan, **kwargs):
        self.execute_kwargs = kwargs
        return {
            "ok": True,
            "before_close_reached": True,
            "object_in_hand_after_close": True,
            "before_close_error": {"tcp_error_m": 0.005, "ori_error_rad": 0.02},
            "place": {"place_error_m": 0.03},
        }


def test_pick_place_rgbd_visual_route_passes_real_work_friendly_options():
    api = _PickPlaceSwitchApi()

    result = api.pick_and_place_visual_object(
        "x2_pick_place_blue_cube",
        [0.37, 0.055, 0.921],
        prompts=["blue cube"],
        table_name="x2_pick_place_table",
        obstacle_source="rgbd_visual",
        place_offset_source="visual_grasp_pose",
    )

    assert result["ok"] is True
    assert result["obstacle_source"] == "rgbd_visual"
    assert result["real_work_friendly"]["rgbd_visual_obstacles"] is True
    assert result["real_work_friendly"]["visual_grasp_pose_place_offset"] is True
    assert result["sim_only"]["sim_known_obstacles"] is False
    assert result["sim_only"]["after_close_sim_known_place_offset"] is False
    assert result["sim_only"]["sim_place_correction_steps"] == 0
    assert api.sim_known_called is False

    assert api.execute_kwargs is not None
    assert api.execute_kwargs["place_offset_source"] == "visual_grasp_pose"
    assert api.execute_kwargs["read_after_close_sim_object_pose"] is False
    assert api.execute_kwargs["place_object_correction_steps"] == 0
    assert api.execute_kwargs["reobserve_at_precontact"] is False
    assert api.execute_kwargs["place_descent_waypoints"] == 1
    assert [obstacle["source"] for obstacle in api.execute_kwargs["obstacles_world"]] == [
        "rgbd_object_mask_aabb",
        "rgbd_table_plane_aabb",
    ]


class _ReobserveExecutorApi(X2ControlApi):
    def __init__(self, reobserve_plan):
        super().__init__(_SceneTrapEnv())
        self.reobserve_plan = reobserve_plan
        self.current_tcp_pose = None
        self.moved_targets = []

    def open_gripper(self, arm: int = 1):
        return True

    def close_gripper(self, arm: int = 1):
        return True

    def settle_robot(self, steps: int = 8):
        return int(steps)

    def move_tcp_pyroki_trajopt(self, target_tcp_pose, **kwargs):
        self.current_tcp_pose = target_tcp_pose
        self.moved_targets.append(("precontact", target_tcp_pose))
        return True

    def move_tcp_joint_ik(self, target_tcp_pose, **kwargs):
        self.current_tcp_pose = target_tcp_pose
        self.moved_targets.append(("joint_ik", target_tcp_pose))
        return True

    def get_current_tcp_pose(self, arm: int = 1):
        if self.current_tcp_pose is None:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), quat
        return self.current_tcp_pose

    def check_object_in_hand(self, arm: int = 1):
        return True

    def get_tcp_offset_eef(self, arm: int = 1):
        return np.zeros(3, dtype=np.float64)

    def plan_visual_grasp_tcp_pose(self, *args, **kwargs):
        return self.reobserve_plan

    def _check_tcp_pose_ik_reachability(self, *args, **kwargs):
        return {"ok": True, "fk_pos_error_m": 0.002, "fk_ori_error_rad": 0.01}

    def get_last_motion_debug(self):
        return {}


def _minimal_tcp_plan(grasp_pos, object_pos, *, mask_pixels=1600, depth_value=1.0):
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    grasp_pos = np.asarray(grasp_pos, dtype=np.float64)
    object_pos = np.asarray(object_pos, dtype=np.float64)
    precontact_pos = grasp_pos + np.array([0.0, 0.0, 0.08], dtype=np.float64)
    side = max(1, int(np.ceil(np.sqrt(mask_pixels))))
    mask = np.zeros((side, side), dtype=bool)
    mask.flat[:mask_pixels] = True
    depth = np.full((side, side), float(depth_value), dtype=np.float64)
    return {
        "ok": True,
        "object_name": "x2_pick_place_blue_cube",
        "visual_artifact_dir": None,
        "pose_estimate": {
            "object_name": "x2_pick_place_blue_cube",
            "position_world": object_pos,
        },
        "selection": {"selected_rank": 0, "selected_raw_index": 0},
        "visual": {
            "mask": mask,
            "depth": depth,
            "mask_pixels": int(mask.sum()),
            "expected_depth": float(depth_value),
            "depth_window": 0.05,
        },
        "grasp_tcp_pose": (grasp_pos, quat),
        "precontact_tcp_pose": (precontact_pos, quat),
        "tcp_axis_world": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "insertion_waypoints": [
            {"name": "grasp", "distance_m": 0.0, "tcp_pose": (grasp_pos, quat)},
        ],
    }


def test_execute_tcp_grasp_plan_adopts_quality_gated_precontact_reobserve():
    initial = _minimal_tcp_plan([0.320, -0.080, 0.930], [0.300, -0.080, 0.920])
    reobserved = _minimal_tcp_plan([0.326, -0.081, 0.931], [0.306, -0.081, 0.921])
    api = _ReobserveExecutorApi(reobserved)

    result = api.execute_tcp_grasp_plan(
        initial,
        place_position=None,
        reobserve_at_precontact=True,
        reobserve_prompts=["blue cube"],
        reobserve_candidate_index=0,
        reobserve_min_mask_pixels=1000,
        reobserve_min_depth_points=64,
    )

    assert result["precontact_reobserve"]["adopted"] is True
    assert result["precontact_reobserve"]["reason"] == "quality_gates_passed"
    assert np.allclose(result["active_plan_summary"]["grasp_tcp_pose"][0], reobserved["grasp_tcp_pose"][0])
    assert any(np.allclose(target[1][0], reobserved["grasp_tcp_pose"][0]) for target in api.moved_targets)


def test_execute_tcp_grasp_plan_falls_back_when_reobserve_mask_is_poor():
    initial = _minimal_tcp_plan([0.320, -0.080, 0.930], [0.300, -0.080, 0.920])
    poor_reobserved = _minimal_tcp_plan(
        [0.326, -0.081, 0.931],
        [0.306, -0.081, 0.921],
        mask_pixels=16,
    )
    api = _ReobserveExecutorApi(poor_reobserved)

    result = api.execute_tcp_grasp_plan(
        initial,
        place_position=None,
        reobserve_at_precontact=True,
        reobserve_prompts=["blue cube"],
        reobserve_candidate_index=0,
        reobserve_min_mask_pixels=1000,
        reobserve_min_depth_points=64,
    )

    assert result["precontact_reobserve"]["adopted"] is False
    assert result["precontact_reobserve"]["reason"] == "quality_gate_failed"
    assert "mask_pixels_below_min" in result["precontact_reobserve"]["gate_failures"]
    assert np.allclose(result["active_plan_summary"]["grasp_tcp_pose"][0], initial["grasp_tcp_pose"][0])
