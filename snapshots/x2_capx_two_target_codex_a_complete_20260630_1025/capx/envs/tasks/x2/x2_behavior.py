from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a fixed-base X2 dual-arm robot with API described below.
Goal: Complete the task described in the environment.
The robot cannot navigate or move its base. Use only arm, gripper, observation, and object-pose APIs.
You may write python code comments for reasoning but ONLY write executable Python code and do not write it in code fences.
If you want to use numpy, scipy for spatial transformations, opencv, pytorch, or any other libraries, you need to import them explicitly.
Note that API may fail. Make sure the code is fault tolerant.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""


class X2BehaviorCodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for fixed-base X2 BEHAVIOR tasks."""

    prompt = PROMPT
    oracle_code = None


__all__ = [
    "X2BehaviorCodeEnv",
]
