from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.tasks.base import CodeExecutionEnvBase


ARM = 1
TARGET_OBJECT_NAME = "x2_pick_place_blue_cube"
DISTRACTOR_OBJECT_NAME = "x2_pick_place_red_cube"
TABLE_NAME = "x2_pick_place_table"
RIGHT_MARKER_NAME = "x2_pick_place_right_target_marker"
RIGHT_PLACE_POSITION = np.array([0.37, 0.055, 0.921], dtype=np.float64)
REFERENCE_TCP_QUAT_XYZW = np.array([0.53604597, 0.63824742, 0.39958203, -0.38161388], dtype=np.float64)
WORKSPACE_BOUNDS = {"x": (0.16, 0.50), "y": (-0.38, 0.12), "z": (0.78, 1.12)}
CANDIDATE_INDICES = (0, 1, 2, 3, 4, 5)
SIM_PLACE_CORRECTION_STEPS = 4
SUCCESS_THRESHOLD_M = 0.10


PROMPT = f"""
You are controlling a fixed-base X2 robot in a simple tabletop pick-place task.
Goal: pick up the movable BLUE cube named "{TARGET_OBJECT_NAME}" and place it at the right target marker named "{RIGHT_MARKER_NAME}".
There is also a red cube named "{DISTRACTOR_OBJECT_NAME}" in the scene. Do not pick or move the red cube.

The robot, table, movable cubes, target marker, chest camera, and video recording are already created by the environment.
Do not create, reset, or reconfigure the simulator. Only use the injected APIs.
Use the right arm: arm={ARM}.

The visual/action pose contract is:
- pick_and_place_visual_object(...) internally detects the requested object, produces X2 executable TCP poses in world frame, T_world_tcp, and executes the pick-place.
- It internally converts TCP targets to X2 EEF/joint targets as needed.
- place_position is the desired world-frame object-center position, not an EEF or TCP pose.
- Do not execute raw GraspNet poses directly.
- Do not call lower-level planning, IK, camera, gripper, or execution functions.

This short-term simulation task may use sim-known table/cube box obstacles for PyRoKi planning.
This obstacle source is for simulation integration only, not a 2real perception primitive.
The default pre-release object-center correction is also sim-only.

Write exactly one call to pick_and_place_visual_object(...); do not call it more than once.
Use object_name="{TARGET_OBJECT_NAME}" and table_name="{TABLE_NAME}".
Use visual prompts that specifically identify the blue cube, such as ["blue cube", "blue block", "blue box"].

Recommended code:
RESULT = pick_and_place_visual_object(
    "{TARGET_OBJECT_NAME}",
    [{RIGHT_PLACE_POSITION[0]:.3f}, {RIGHT_PLACE_POSITION[1]:.3f}, {RIGHT_PLACE_POSITION[2]:.3f}],
    prompts=["blue cube", "blue block", "blue box"],
    table_name="{TABLE_NAME}",
    orientation_quat_xyzw=[
        {REFERENCE_TCP_QUAT_XYZW[0]:.8f},
        {REFERENCE_TCP_QUAT_XYZW[1]:.8f},
        {REFERENCE_TCP_QUAT_XYZW[2]:.8f},
        {REFERENCE_TCP_QUAT_XYZW[3]:.8f},
    ],
    workspace_bounds={{"x": ({WORKSPACE_BOUNDS["x"][0]:.2f}, {WORKSPACE_BOUNDS["x"][1]:.2f}), "y": ({WORKSPACE_BOUNDS["y"][0]:.2f}, {WORKSPACE_BOUNDS["y"][1]:.2f}), "z": ({WORKSPACE_BOUNDS["z"][0]:.2f}, {WORKSPACE_BOUNDS["z"][1]:.2f})}},
    candidate_indices={CANDIDATE_INDICES},
    sim_place_correction_steps={SIM_PLACE_CORRECTION_STEPS},
)

Do not include reasoning text, function signatures, markdown fences, or <think> tags.
ONLY write executable Python code.
If you want to use numpy or other libraries, import them explicitly.
The functions (APIs) below are already imported to the environment.
"""


ORACLE_CODE = f"""
RESULT = pick_and_place_visual_object(
    "{TARGET_OBJECT_NAME}",
    [{RIGHT_PLACE_POSITION[0]:.3f}, {RIGHT_PLACE_POSITION[1]:.3f}, {RIGHT_PLACE_POSITION[2]:.3f}],
    prompts=["blue cube", "blue block", "blue box"],
    table_name="{TABLE_NAME}",
    orientation_quat_xyzw=[
        {REFERENCE_TCP_QUAT_XYZW[0]:.8f},
        {REFERENCE_TCP_QUAT_XYZW[1]:.8f},
        {REFERENCE_TCP_QUAT_XYZW[2]:.8f},
        {REFERENCE_TCP_QUAT_XYZW[3]:.8f},
    ],
    workspace_bounds={{"x": ({WORKSPACE_BOUNDS["x"][0]:.2f}, {WORKSPACE_BOUNDS["x"][1]:.2f}), "y": ({WORKSPACE_BOUNDS["y"][0]:.2f}, {WORKSPACE_BOUNDS["y"][1]:.2f}), "z": ({WORKSPACE_BOUNDS["z"][0]:.2f}, {WORKSPACE_BOUNDS["z"][1]:.2f})}},
    candidate_indices={CANDIDATE_INDICES},
    sim_place_correction_steps={SIM_PLACE_CORRECTION_STEPS},
)
"""


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


class X2PickPlaceTwoObjectsBlueRightCodeEnv(CodeExecutionEnvBase):
    """X2 task: choose the blue cube among two cubes and place it right."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE
    target_object_name = TARGET_OBJECT_NAME
    distractor_object_name = DISTRACTOR_OBJECT_NAME
    target_name = "right"

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
        pos = self._object_position(TARGET_OBJECT_NAME)
        if pos is None:
            return None
        return float(np.linalg.norm(pos - RIGHT_PLACE_POSITION))

    def distractor_displacement(self) -> float | None:
        pos = self._object_position(DISTRACTOR_OBJECT_NAME)
        if pos is None:
            return None
        initial = np.array([0.24, -0.09, 0.921], dtype=np.float64)
        return float(np.linalg.norm(pos - initial))

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
        if result.get("object_name") != TARGET_OBJECT_NAME:
            return False
        if execution.get("object_in_hand_after_close") is not True:
            return False
        return True

    def _summary_lines(self) -> list[str]:
        result, execution = self._last_execution_result()
        if result is None or execution is None:
            return []
        before_close_error = execution.get("before_close_error")
        if not isinstance(before_close_error, dict):
            before_close_error = {}
        place = execution.get("place")
        if not isinstance(place, dict):
            place = {}
        attempts = result.get("attempts")
        if not isinstance(attempts, list):
            attempts = []

        lines = [
            "X2_TWO_OBJECT_RESULT "
            + "ok=" + str(bool(result.get("ok"))) + " "
            + "object=" + str(result.get("object_name")) + " "
            + "target=right "
            + "before_close_tcp_error_m=" + str(before_close_error.get("tcp_error_m")) + " "
            + "before_close_ori_error_rad=" + str(before_close_error.get("ori_error_rad")) + " "
            + "object_in_hand_after_close=" + str(execution.get("object_in_hand_after_close")) + " "
            + "place_error_m=" + str(place.get("place_error_m"))
        ]
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_error = attempt.get("before_close_error")
            if not isinstance(attempt_error, dict):
                attempt_error = {}
            lines.append(
                "X2_TWO_OBJECT_ATTEMPT "
                + "candidate_index=" + str(attempt.get("candidate_index")) + " "
                + "ok=" + str(attempt.get("ok")) + " "
                + "before_close_reached=" + str(attempt.get("before_close_reached")) + " "
                + "object_in_hand_after_close=" + str(attempt.get("object_in_hand_after_close")) + " "
                + "before_close_tcp_error_m=" + str(attempt_error.get("tcp_error_m")) + " "
                + "before_close_ori_error_rad=" + str(attempt_error.get("ori_error_rad"))
            )
        return lines

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
        summary_lines = self._summary_lines()
        if summary_lines:
            summary_text = "\n".join(summary_lines) + "\n"
            print(summary_text, end="")
            info["stdout"] = str(info.get("stdout", "")) + summary_text
        info["target_object_name"] = TARGET_OBJECT_NAME
        info["target_name"] = self.target_name
        info["place_error_m"] = err
        info["distractor_displacement_m"] = self.distractor_displacement()
        info["visual_pick_place_ok"] = None if result is None else result.get("ok")
        info["picked_object_name"] = None if result is None else result.get("object_name")
        info["execution_ok"] = None if execution is None else execution.get("ok")
        info["object_in_hand_after_close"] = (
            None if execution is None else execution.get("object_in_hand_after_close")
        )
        info["task_completed"] = self._task_success()
        return obs, reward, terminated, truncated, info


__all__ = [
    "X2PickPlaceTwoObjectsBlueRightCodeEnv",
]
