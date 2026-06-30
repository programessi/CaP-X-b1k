"""Example of CaP-X-style injected X2 visual grasp code.

This file intentionally does not create an env, configure a task, record
video, or start vision services.  It assumes the caller already injected an
X2ControlApi instance into a running X2 BEHAVIOR environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def grasp_visible_object_with_x2_api(
    api: Any,
    object_name: str,
    *,
    orientation_quat_xyzw: np.ndarray | None = None,
    obstacles_world: list[dict[str, Any]] | None = None,
    arm: int = 1,
) -> dict[str, Any]:
    """Plan and execute a visual GraspNet grasp with the current X2 env/API.

    Args:
        api: Existing ``X2ControlApi`` from the running env.
        object_name: Text query and scene object name, e.g. ``"red cube"``.
        orientation_quat_xyzw: Optional validated X2 reference orientation.
            If omitted, the API uses the current EEF orientation as reference.
        obstacles_world: Optional PyRoKi obstacle list for precontact planning.
        arm: X2 arm index. ``1`` is the right arm.

    Returns:
        A dictionary containing the visual TCP plan and execution diagnostics.
    """
    plan = api.plan_visual_grasp_tcp_pose(
        object_name,
        camera_name=api.get_chest_camera_name(),
        arm=arm,
        external=False,
        orientation_quat_xyzw=orientation_quat_xyzw,
        precontact_distance=0.08,
        insert_waypoints=10,
    )
    if not plan.get("ok"):
        return {"ok": False, "stage": "plan_visual_grasp_tcp_pose", "plan": plan}

    execution = api.execute_tcp_grasp_plan(
        plan,
        arm=arm,
        obstacles_world=obstacles_world,
        timesteps=18,
        dt=0.08,
        max_joint_step=0.022,
        insert_max_joint_step=0.011,
        settle_steps=16,
        hold_steps_per_waypoint=2,
        insert_hold_steps_per_waypoint=5,
        close_hold_steps=30,
    )
    return {
        "ok": bool(execution.get("ok")),
        "stage": "execute_tcp_grasp_plan",
        "plan": plan,
        "execution": execution,
    }
