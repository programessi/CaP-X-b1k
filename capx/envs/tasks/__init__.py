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
