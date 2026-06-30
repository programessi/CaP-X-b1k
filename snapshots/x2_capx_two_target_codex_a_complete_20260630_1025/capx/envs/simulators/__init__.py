# import all environments here to register them!
from capx.envs.base import list_envs, register_env


try:
    from .franka_real import FrankaRealLowLevel
    register_env("franka_real_low_level", FrankaRealLowLevel)
except Exception:
    print("Franka real not installed!")

try:
    from .r1pro_b1k import R1ProBehaviourLowLevel
    register_env("r1pro_b1k_low_level", R1ProBehaviourLowLevel)
except Exception:
    print("R1Pro not installed!")

try:
    from .x2_b1k import X2BehaviourLowLevel
    register_env("x2_b1k_low_level", X2BehaviourLowLevel)
except Exception:
    print("X2 not installed!")
