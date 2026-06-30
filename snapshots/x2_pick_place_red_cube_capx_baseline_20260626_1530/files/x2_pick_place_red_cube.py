from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.tasks.base import CodeExecutionEnvBase


ARM = 1
OBJECT_NAME = "x2_pick_place_red_cube"
TABLE_NAME = "x2_pick_place_table"
PLACE_MARKER_NAME = "x2_pick_place_target_marker"
PLACE_POSITION = np.array([0.32, 0.055, 0.921], dtype=np.float64)
SUCCESS_THRESHOLD_M = 0.10


PROMPT = f"""
You are controlling a fixed-base X2 robot in a simple tabletop pick-place task.
Goal: pick up the movable red cube named "{OBJECT_NAME}" and place it at the target marker named "{PLACE_MARKER_NAME}".
The robot, table, movable cube, target marker, chest camera, and video recording are already created by the environment.
Do not create, reset, or reconfigure the simulator. Only use the injected APIs.
Use the right arm: arm={ARM}.

The visual/action pose contract is:
- pick_and_place_red_cube() internally plans X2 executable TCP poses in world frame, T_world_tcp.
- It internally converts TCP targets to X2 EEF/joint targets as needed.
- Do not execute raw GraspNet poses directly.
- Do not call lower-level planning or execution functions; this reduced API exposes the task-level primitive for you.

This short-term simulation task may use sim-known table/cube box obstacles for PyRoKi planning:
pick_and_place_red_cube() uses the task table/object obstacle boxes internally.
This obstacle source is for simulation integration only, not a 2real perception primitive.
The default pre-release object-center correction is also sim-only.

Write exactly one call to pick_and_place_red_cube(); do not call it more than once.
Recommended code:
RESULT = pick_and_place_red_cube()
execution = RESULT.get("execution", {{}})
before_close_error = execution.get("before_close_error", {{}})
place = execution.get("place", {{}})
print("X2_PICK_PLACE_RESULT "
      + "ok=" + str(bool(RESULT.get("ok"))) + " "
      + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
      + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
      + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
      + "place_error_m=" + str(place.get("place_error_m")))

Do not include reasoning text, function signatures, markdown fences, or <think> tags.
ONLY write executable Python code.
If you want to use numpy or other libraries, import them explicitly.
The functions (APIs) below are already imported to the environment.
"""


ORACLE_CODE = f"""
import numpy as np

ARM = {ARM}
OBJECT_NAME = "{OBJECT_NAME}"
TABLE_NAME = "{TABLE_NAME}"
PLACE_POSITION = np.array({PLACE_POSITION.round(6).tolist()}, dtype=np.float64)
PROMPTS = ["red cube", "red block", "red box", "cube"]

# Orientation prior validated in previous X2 tabletop visual grasp smoke tests.
reference_quat_xyzw = np.array([0.53604597, 0.63824742, 0.39958203, -0.38161388], dtype=np.float64)

result = pick_and_place_visual_object(
    OBJECT_NAME,
    PLACE_POSITION,
    prompts=PROMPTS,
    camera_name=get_chest_camera_name(),
    arm=ARM,
    table_name=TABLE_NAME,
    orientation_quat_xyzw=reference_quat_xyzw,
    workspace_bounds={{"x": (0.16, 0.50), "y": (-0.38, 0.12), "z": (0.78, 1.12)}},
    place_position_threshold={SUCCESS_THRESHOLD_M},
    use_sim_known_obstacles=True,
    sim_place_correction_steps=2,
)

execution = result.get("execution") or {{}}
before_close_error = execution.get("before_close_error") or {{}}
place = execution.get("place") or {{}}
print(
    "X2_PICK_PLACE_RESULT "
    f"ok={{bool(result.get('ok'))}} "
    f"before_close_tcp_error_m={{before_close_error.get('tcp_error_m')}} "
    f"before_close_ori_error_rad={{before_close_error.get('ori_error_rad')}} "
    f"object_in_hand_after_close={{execution.get('object_in_hand_after_close')}} "
    f"place_error_m={{place.get('place_error_m')}}"
)
RESULT = result
"""


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


class X2PickPlaceRedCubeCodeEnv(CodeExecutionEnvBase):
    """X2 CaP-X code task for visual red-cube tabletop pick-place."""

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
        return float(np.linalg.norm(pos - PLACE_POSITION))

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
        info["place_error_m"] = err
        info["visual_pick_place_ok"] = None if result is None else result.get("ok")
        info["execution_ok"] = None if execution is None else execution.get("ok")
        info["object_in_hand_after_close"] = (
            None if execution is None else execution.get("object_in_hand_after_close")
        )
        info["task_completed"] = self._task_success()
        return obs, reward, terminated, truncated, info


__all__ = [
    "X2PickPlaceRedCubeCodeEnv",
]
