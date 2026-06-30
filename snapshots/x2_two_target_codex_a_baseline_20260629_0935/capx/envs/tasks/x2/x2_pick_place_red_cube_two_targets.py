from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.tasks.base import CodeExecutionEnvBase


ARM = 1
OBJECT_NAME = "x2_pick_place_red_cube"
TABLE_NAME = "x2_pick_place_table"
LEFT_MARKER_NAME = "x2_pick_place_left_target_marker"
RIGHT_MARKER_NAME = "x2_pick_place_right_target_marker"
LEFT_PLACE_POSITION = np.array([0.27, 0.055, 0.921], dtype=np.float64)
RIGHT_PLACE_POSITION = np.array([0.37, 0.055, 0.921], dtype=np.float64)
SUCCESS_THRESHOLD_M = 0.10


PROMPT = f"""
You are controlling a fixed-base X2 robot in a simple tabletop pick-place task.
Goal: pick up the movable red cube named "{OBJECT_NAME}" and place it at the RIGHT target marker named "{RIGHT_MARKER_NAME}".
There are two target markers in the scene:
- left target marker: "{LEFT_MARKER_NAME}"
- right target marker: "{RIGHT_MARKER_NAME}"

The robot, table, movable cube, target markers, chest camera, and video recording are already created by the environment.
Do not create, reset, or reconfigure the simulator. Only use the injected APIs.
Use the right arm: arm={ARM}.

The visual/action pose contract is:
- pick_and_place_red_cube_to_right_target() internally plans X2 executable TCP poses in world frame, T_world_tcp.
- It internally converts TCP targets to X2 EEF/joint targets as needed.
- Do not execute raw GraspNet poses directly.
- Do not call lower-level planning or execution functions; this reduced API exposes the task-level primitives for you.

This short-term simulation task may use sim-known table/cube box obstacles for PyRoKi planning.
This obstacle source is for simulation integration only, not a 2real perception primitive.
The default pre-release object-center correction is also sim-only.

Write exactly one call to pick_and_place_red_cube_to_right_target(); do not call it more than once.
Recommended code:
RESULT = pick_and_place_red_cube_to_right_target()
execution = RESULT.get("execution", {{}})
before_close_error = execution.get("before_close_error", {{}})
place = execution.get("place", {{}})
print("X2_TWO_TARGET_RESULT "
      + "ok=" + str(bool(RESULT.get("ok"))) + " "
      + "target=" + str(RESULT.get("target_name")) + " "
      + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
      + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
      + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
      + "place_error_m=" + str(place.get("place_error_m")))

Do not include reasoning text, function signatures, markdown fences, or <think> tags.
ONLY write executable Python code.
If you want to use numpy or other libraries, import them explicitly.
The functions (APIs) below are already imported to the environment.
"""


ORACLE_CODE = """
RESULT = pick_and_place_red_cube_to_right_target()
execution = RESULT.get("execution", {})
before_close_error = execution.get("before_close_error", {})
place = execution.get("place", {})
print("X2_TWO_TARGET_RESULT "
      + "ok=" + str(bool(RESULT.get("ok"))) + " "
      + "target=" + str(RESULT.get("target_name")) + " "
      + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
      + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
      + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
      + "place_error_m=" + str(place.get("place_error_m")))
"""


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


class X2PickPlaceRedCubeTwoTargetsCodeEnv(CodeExecutionEnvBase):
    """X2 CaP-X code task for placing a red cube at one of two tabletop targets."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE

    def _object_position(self, object_name: str) -> np.ndarray | None:
        try:
            obj = self.low_level_env.env.scene.object_registry("name", object_name)
            if obj is None:
                return None
            if hasattr(obj, "aabb_center"):
                return _as_numpy(obj.aabb_center).astype(np.float64).reshape(3)
            pos, _quat = obj.get_position_orientation()
            return _as_numpy(pos).astype(np.float64).reshape(3)
        except Exception:
            return None

    def place_error(self) -> float | None:
        pos = self._object_position(OBJECT_NAME)
        if pos is None:
            return None
        return float(np.linalg.norm(pos - RIGHT_PLACE_POSITION))

    def _last_execution_result(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        result = self._exec_globals.get("RESULT")
        if not isinstance(result, dict):
            return None, None
        execution = result.get("execution")
        if not isinstance(execution, dict):
            execution = result
        return result, execution

    def _last_execution_success(self) -> bool:
        result, execution = self._last_execution_result()
        if result is None or execution is None:
            return False
        if result.get("ok") is not True:
            return False
        if result.get("target_name") != "right":
            return False
        if execution.get("object_in_hand_after_close") is not True:
            return False
        return True

    def _task_success(self) -> bool:
        err = self.place_error()
        return bool(
            err is not None
            and err <= SUCCESS_THRESHOLD_M
            and self._last_execution_success()
        )

    def compute_reward(self) -> float:
        return 1.0 if self._task_success() else 0.0

    def step(self, action: str):
        obs, reward, terminated, truncated, info = super().step(action)
        err = self.place_error()
        result, execution = self._last_execution_result()
        info["right_place_error_m"] = err
        info["visual_pick_place_ok"] = None if result is None else result.get("ok")
        info["target_name"] = None if result is None else result.get("target_name")
        info["execution_ok"] = None if execution is None else execution.get("ok")
        info["object_in_hand_after_close"] = (
            None if execution is None else execution.get("object_in_hand_after_close")
        )
        info["task_completed"] = self._task_success()
        return obs, reward, terminated, truncated, info


__all__ = [
    "X2PickPlaceRedCubeTwoTargetsCodeEnv",
]
