from .base import (
    CodeExecEnvConfig,
    CodeExecutionEnvBase,
    get_config,
    get_exec_env,
    list_configs,
    list_exec_envs,
    register_config,
    register_exec_env,
)

# ---------------------------------------------------------------------------
# R1Pro Behavior Environments
# ---------------------------------------------------------------------------
from .r1pro.r1pro_behavior import R1ProBehaviorCodeEnv
register_exec_env("r1pro_behavior_code_env", R1ProBehaviorCodeEnv)
register_config(
    "r1pro_behavior_code_env",
    CodeExecEnvConfig(
        low_level="r1pro_b1k_low_level",
        apis=["R1ProControlApi"],
    ),
)

from .r1pro.r1pro_pickup_radio import R1ProRadioCodeEnv
register_exec_env("r1pro_radio_code_env", R1ProRadioCodeEnv)
register_config(
    "r1pro_radio_code_env",
    CodeExecEnvConfig(
        low_level="r1pro_b1k_low_level",
        apis=["R1ProControlApi"],
    ),
)

from .r1pro.r1pro_pickup_trash import R1ProTrashCodeEnv
register_exec_env("r1pro_trash_code_env", R1ProTrashCodeEnv)
register_config(
    "r1pro_trash_code_env",
    CodeExecEnvConfig(
        low_level="r1pro_b1k_low_level",
        apis=["R1ProControlApi"],
    ),
)

# ---------------------------------------------------------------------------
# X2 Behavior Environments
# ---------------------------------------------------------------------------
from .x2.x2_behavior import X2BehaviorCodeEnv
register_exec_env("x2_behavior_code_env", X2BehaviorCodeEnv)
register_config(
    "x2_behavior_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2ControlApi"],
    ),
)

from .x2.x2_tabletop_tcp_reach import X2TabletopTcpReachCodeEnv
register_exec_env("x2_tabletop_tcp_reach_code_env", X2TabletopTcpReachCodeEnv)
register_config(
    "x2_tabletop_tcp_reach_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2ControlApi"],
    ),
)

from .x2.x2_pick_place_red_cube import X2PickPlaceRedCubeCodeEnv
register_exec_env("x2_pick_place_red_cube_code_env", X2PickPlaceRedCubeCodeEnv)
register_config(
    "x2_pick_place_red_cube_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2PickPlaceApi"],
    ),
)

from .x2.x2_pick_place_red_cube_two_targets import (
    X2PickPlaceRedCubeTwoTargetsCodeEnv,
    X2PickPlaceRedCubeTwoTargetsLeftCodeEnv,
)
register_exec_env("x2_pick_place_red_cube_two_targets_code_env", X2PickPlaceRedCubeTwoTargetsCodeEnv)
register_config(
    "x2_pick_place_red_cube_two_targets_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2PickPlaceApi"],
    ),
)
register_exec_env("x2_pick_place_red_cube_two_targets_left_code_env", X2PickPlaceRedCubeTwoTargetsLeftCodeEnv)
register_config(
    "x2_pick_place_red_cube_two_targets_left_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2PickPlaceApi"],
    ),
)

from .x2.x2_pick_place_two_objects import (
    X2PickPlaceTwoObjectsBlueRightCodeEnv,
    X2PickPlaceTwoObjectsBlueRightRgbdVisualCodeEnv,
)
register_exec_env("x2_pick_place_two_objects_blue_right_code_env", X2PickPlaceTwoObjectsBlueRightCodeEnv)
register_config(
    "x2_pick_place_two_objects_blue_right_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2PickPlaceApi"],
    ),
)
register_exec_env(
    "x2_pick_place_two_objects_blue_right_rgbd_visual_code_env",
    X2PickPlaceTwoObjectsBlueRightRgbdVisualCodeEnv,
)
register_config(
    "x2_pick_place_two_objects_blue_right_rgbd_visual_code_env",
    CodeExecEnvConfig(
        low_level="x2_b1k_low_level",
        apis=["X2PickPlaceApi"],
    ),
)
