# import all environments here to register them!
from capx.envs.base import list_envs, register_env


from .franka_real import FrankaRealLowLevel
register_env("franka_real_low_level", FrankaRealLowLevel)

try:
    from .r1pro_b1k import R1ProBehaviourLowLevel
    register_env("r1pro_b1k_low_level", R1ProBehaviourLowLevel)
except Exception:
    print("R1Pro not installed!")
