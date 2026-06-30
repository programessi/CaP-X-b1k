"""X2 BEHAVIOR low-level environment for CAP-X.

This wrapper mirrors the R1Pro BEHAVIOR low-level layer, but routes control
through the X2 OmniGibson IK / gripper primitives that are available without
CuRobo or mobile-base navigation.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from capx.envs.base import BaseEnv

og = None


X2_PYROKI_SOURCE_URDF = (
    Path(__file__).resolve().parents[4]
    / "BEHAVIOR-1K"
    / "datasets"
    / "omnigibson-robot-assets"
    / "objects"
    / "robot"
    / "x2_ultra_plus_omnipicker_omnipicker_robotiq85"
    / "urdf"
    / "x2_ultra_plus_omnipicker_omnipicker_robotiq85_source.urdf"
)
X2_PYROKI_PATCHED_URDF = (
    X2_PYROKI_SOURCE_URDF.parent
    / "x2_ultra_plus_omnipicker_omnipicker_robotiq85_pyroki_ik_fixed_grippers.urdf"
)


def _ensure_x2_pyroki_ik_urdf() -> str:
    """Return a PyRoKi-loadable X2 URDF with gripper joints fixed."""
    if X2_PYROKI_PATCHED_URDF.exists():
        return str(X2_PYROKI_PATCHED_URDF)
    if not X2_PYROKI_SOURCE_URDF.exists():
        raise FileNotFoundError(f"X2 PyRoKi source URDF not found: {X2_PYROKI_SOURCE_URDF}")

    root = ET.parse(X2_PYROKI_SOURCE_URDF).getroot()
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if "finger" in name or "knuckle" in name:
            joint.set("type", "fixed")
            for child in list(joint):
                if child.tag in {"limit", "dynamics", "mimic", "safety_controller"}:
                    joint.remove(child)
    X2_PYROKI_PATCHED_URDF.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(X2_PYROKI_PATCHED_URDF, encoding="utf-8", xml_declaration=True)
    return str(X2_PYROKI_PATCHED_URDF)


def _load_omnigibson():
    global og
    if og is None:
        try:
            import omnigibson as og_module
            from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
            from omnigibson.macros import gm
        except ModuleNotFoundError as e:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "BEHAVIOR / OmniGibson is not available; install or expose it on PYTHONPATH."
            ) from e

        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.HEADLESS = True
        og = og_module
        return og_module, StarterSemanticActionPrimitives

    from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

    return og, StarterSemanticActionPrimitives


def _make_x2_starter_class(starter_cls):
    class X2StarterSemanticActionPrimitives(starter_cls):
        """Starter primitives with an explicitly selectable arm."""

        def __init__(self, *args, default_arm: str = "right", **kwargs) -> None:
            self._active_arm = default_arm
            super().__init__(*args, **kwargs)

        @property
        def arm(self) -> str:
            return self._active_arm

        def set_arm(self, arm: int | str) -> None:
            if isinstance(arm, int):
                self._active_arm = self.robot.arm_names[arm]
            else:
                if arm not in self.robot.arm_names:
                    raise ValueError(f"Unknown arm {arm!r}; expected one of {self.robot.arm_names}")
                self._active_arm = arm

    return X2StarterSemanticActionPrimitives


class X2BehaviourLowLevel(BaseEnv):
    """CAP-X low-level environment wrapper for fixed-base X2 BEHAVIOR control."""

    def __init__(
        self,
        controller_cfg: str = "x2_robotiq85_primitives.yaml",
        privileged: bool = False,
        save_video: bool = False,
        activity_name: str | None = None,
        objects: list[dict[str, Any]] | None = None,
        external_sensors: list[dict[str, Any]] | None = None,
        load_object_categories: list[str] | None = None,
        robot_camera_arm: int | str = 1,
        robot_camera_resolution: int | None = None,
        robot_obs_modalities: list[str] | None = None,
        chest_camera: bool = True,
        chest_camera_resolution: int | None = None,
        chest_camera_modalities: list[str] | None = None,
        chest_camera_link: str = "lidar_chest_front",
        ik_pose_delta_pos_gain: float = 1.0,
        ik_pose_delta_ori_gain: float = 0.2,
        arm_controller_override: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.privileged = privileged
        self.save_video = save_video
        self._step_count = 0
        self.cur_observation = None
        self._record_frames = False
        self._frame_buffer: list[dict[str, np.ndarray]] = []
        self._last_video_sources: dict[str, str] = {}
        self._ik_pose_delta_pos_gain = float(ik_pose_delta_pos_gain)
        self._ik_pose_delta_ori_gain = float(ik_pose_delta_ori_gain)
        self._x2_pyroki_ctx = None
        self._x2_pyroki_joint_names: list[str] | None = None
        self._x2_joint_name_to_index: dict[str, int] | None = None

        og_module, starter_cls = _load_omnigibson()
        x2_starter_cls = _make_x2_starter_class(starter_cls)

        config_filename = os.path.join(og_module.example_config_path, controller_cfg)
        with open(config_filename, "r") as f:
            self.controller_cfg = yaml.load(f, Loader=yaml.FullLoader)

        if activity_name is not None and "task" in self.controller_cfg:
            self.controller_cfg["task"]["activity_name"] = activity_name
        if objects is not None:
            self.controller_cfg["objects"] = objects
        if external_sensors is not None:
            self.controller_cfg["env"]["external_sensors"] = external_sensors
        if load_object_categories is not None and "scene" in self.controller_cfg:
            self.controller_cfg["scene"]["load_object_categories"] = load_object_categories
        if robot_obs_modalities is not None:
            for robot_cfg in self.controller_cfg.get("robots", []):
                robot_cfg["obs_modalities"] = list(robot_obs_modalities)
        if arm_controller_override:
            for robot_cfg in self.controller_cfg.get("robots", []):
                controller_config = robot_cfg.setdefault("controller_config", {})
                for arm_key in ("arm_left", "arm_right"):
                    controller_config.setdefault(arm_key, {"name": "InverseKinematicsController"}).update(
                        dict(arm_controller_override)
                    )
        if robot_camera_resolution is not None:
            for robot_cfg in self.controller_cfg.get("robots", []):
                sensor_kwargs = (
                    robot_cfg.setdefault("sensor_config", {})
                    .setdefault("VisionSensor", {})
                    .setdefault("sensor_kwargs", {})
                )
                sensor_kwargs["image_height"] = int(robot_camera_resolution)
                sensor_kwargs["image_width"] = int(robot_camera_resolution)

        self.env = og_module.Environment(configs=self.controller_cfg)
        self.robot = self.env.robots[0]
        if chest_camera:
            self._install_chest_camera(
                preferred_link=chest_camera_link,
                image_size=chest_camera_resolution or robot_camera_resolution,
                modalities=chest_camera_modalities,
            )
        self._video_robot_arm = self._arm_name(robot_camera_arm)
        self.controller = x2_starter_cls(
            self.env,
            self.robot,
            default_arm="right",
            enable_head_tracking=False,
            skip_curobo_initilization=True,
        )

        if not og_module.sim.is_playing():
            og_module.sim.play()
        for _ in range(5):
            og_module.sim.step()

    def _install_chest_camera(
        self,
        preferred_link: str = "lidar_chest_front",
        image_size: int | None = None,
        modalities: list[str] | None = None,
    ) -> str:
        """Attach a torso-mounted RGBD camera to the X2 robot and register it as a robot sensor."""
        from omnigibson.sensors import create_sensor
        from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative

        link_name = preferred_link if preferred_link in self.robot.links else "torso_link"
        if link_name not in self.robot.links:
            raise ValueError(f"Cannot install chest camera; available links do not include {preferred_link!r} or 'torso_link'")

        sensor_name = f"{self.robot.name}:{link_name}:Camera:0"
        if sensor_name in self.robot.sensors:
            return sensor_name

        import omnigibson.lazy as lazy

        camera_prim_path = f"{self.robot.links[link_name].prim_path}/Camera"
        with og.sim.editing_usd():
            camera_prim = lazy.pxr.UsdGeom.Camera.Define(og.sim.stage, camera_prim_path).GetPrim()
            self._set_usd_attr(camera_prim, "focalLength", lazy.pxr.Sdf.ValueTypeNames.Float, 17.0)
            self._set_usd_attr(
                camera_prim,
                "clippingRange",
                lazy.pxr.Sdf.ValueTypeNames.Float2,
                lazy.pxr.Gf.Vec2f(0.001, 1000000.0),
            )
            self._set_usd_attr(
                camera_prim,
                "xformOp:translate",
                lazy.pxr.Sdf.ValueTypeNames.Double3,
                lazy.pxr.Gf.Vec3d(0.06, 0.0, 0.0),
            )
            self._set_usd_attr(
                camera_prim,
                "xformOp:orient",
                lazy.pxr.Sdf.ValueTypeNames.Quatd,
                # Aim the chest camera forward (+X in world) with a small downward pitch toward tabletop targets.
                lazy.pxr.Gf.Quatd(-0.133719, -0.694348, -0.133719, 0.694348),
            )
            self._set_usd_attr(
                camera_prim,
                "xformOp:scale",
                lazy.pxr.Sdf.ValueTypeNames.Double3,
                lazy.pxr.Gf.Vec3d(1.0, 1.0, 1.0),
            )
            self._set_usd_attr(
                camera_prim,
                "xformOpOrder",
                lazy.pxr.Sdf.ValueTypeNames.TokenArray,
                ["xformOp:translate", "xformOp:orient", "xformOp:scale"],
            )

        sensor_kwargs = {
            "image_height": int(image_size or 384),
            "image_width": int(image_size or 384),
        }
        sensor = create_sensor(
            sensor_type="Camera",
            relative_prim_path=absolute_prim_path_to_scene_relative(self.robot.scene, camera_prim_path),
            name=sensor_name,
            modalities=modalities or ["rgb", "depth", "depth_linear"],
            sensor_kwargs=sensor_kwargs,
        )
        sensor.load(self.robot.scene)
        if self.robot.initialized:
            sensor.initialize()
        self.robot.sensors[sensor_name] = sensor
        self.robot._obs_modalities = set(self.robot.obs_modalities).union(sensor.modalities)
        self.env.load_observation_space()
        return sensor_name

    @staticmethod
    def _set_usd_attr(prim, name: str, type_name, value) -> None:
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)

    def _arm_name(self, arm: int | str = 1) -> str:
        if isinstance(arm, int):
            return self.robot.arm_names[arm]
        if arm not in self.robot.arm_names:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {self.robot.arm_names}")
        return arm

    def _set_arm(self, arm: int | str = 1) -> str:
        arm_name = self._arm_name(arm)
        self.controller.set_arm(arm_name)
        return arm_name

    def _to_tensor_pose(self, target_pose: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]):
        pos, quat = target_pose
        pos_t = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos, dtype=torch.float32)
        quat_t = quat if isinstance(quat, torch.Tensor) else torch.tensor(quat, dtype=torch.float32)
        return pos_t, quat_t

    def _hold_current_hand_target(self, arm: int | str = 1) -> None:
        """Refresh the saved IK target so later no-op actions hold the current pose."""
        from omnigibson.controllers.controller_view import ControllerView
        from omnigibson.controllers.ik_controller import InverseKinematicsController

        import omnigibson.utils.transform_utils as T

        arm_name = self._arm_name(arm)
        controller_name = f"arm_{arm_name}"
        group_key, _controller_idx = self.robot.controllers[controller_name]
        if ControllerView.is_controller_type(group_key, InverseKinematicsController):
            current_pose = self.controller._world_pose_to_robot_pose(
                (self.robot.get_eef_position(arm_name), self.robot.get_eef_orientation(arm_name))
            )
            current_pos, current_orn = current_pose
            self.controller._arm_targets[controller_name] = (current_pos, T.quat2axisangle(current_orn))
        else:
            self.controller._arm_targets[controller_name] = self.robot.get_joint_positions()[ControllerView.get_dof_idx(group_key)]

    @staticmethod
    def _debug_list(value) -> list[float]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return [round(float(v), 6) for v in np.asarray(value).reshape(-1)]

    @staticmethod
    def _quat_xyzw_to_matrix(quat_xyzw) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        x, y, z, w = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_quat_xyzw(matrix) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([x, y, z, w], dtype=np.float64)
        return quat / max(float(np.linalg.norm(quat)), 1e-12)

    @classmethod
    def _pose_to_matrix(cls, pos, quat_xyzw) -> np.ndarray:
        T_mat = np.eye(4, dtype=np.float64)
        T_mat[:3, :3] = cls._quat_xyzw_to_matrix(quat_xyzw)
        T_mat[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        return T_mat

    @classmethod
    def _matrix_to_pose(cls, T_mat) -> tuple[np.ndarray, np.ndarray]:
        T_mat = np.asarray(T_mat, dtype=np.float64).reshape(4, 4)
        return T_mat[:3, 3].copy(), cls._matrix_to_quat_xyzw(T_mat[:3, :3])

    @staticmethod
    def _orientation_error_rad(quat_a_xyzw, quat_b_xyzw) -> float:
        qa = np.asarray(quat_a_xyzw, dtype=np.float64).reshape(4)
        qb = np.asarray(quat_b_xyzw, dtype=np.float64).reshape(4)
        qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
        qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
        dot = abs(float(np.dot(qa, qb)))
        return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def _execute_controller(self, ctrl_gen) -> int:
        steps = 0
        for steps, action in enumerate(ctrl_gen, start=1):
            if action is not None:
                self.step(action)
        return steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_count = 0
        obs, info = self.env.reset()
        self.cur_observation = obs
        info["task_prompt"] = "Control the X2 robot."
        return self.get_observation(), info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cur_observation = obs
        self._record_frame()
        return obs, reward, terminated, truncated, info

    def get_observation(self) -> dict[str, Any]:
        if self.cur_observation is not None:
            return self.cur_observation
        obs, _ = self.env.get_obs()
        self.cur_observation = obs
        return obs

    def compute_reward(self) -> float:
        if hasattr(self.env, "task") and hasattr(self.env.task, "get_reward"):
            reward, _ = self.env.task.get_reward(self.env)
            return float(reward)
        return 0.0

    def task_completed(self) -> bool:
        if hasattr(self.env, "task") and hasattr(self.env.task, "success"):
            return bool(self.env.task.success)
        return False

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")

        frame = self._external_rgb_frame()
        if frame is None:
            frame = self._first_rgb_from_obs(self.get_observation())
        if frame is None:
            raise RuntimeError("No RGB camera frame is available for X2 render().")
        return self._frame_to_numpy(frame)

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> dict[str, list[np.ndarray]]:
        if len(self._frame_buffer) == 0:
            self._record_frame()
        keys = sorted({key for frame in self._frame_buffer for key in frame})
        frames = {
            key: [frame[key].copy() for frame in self._frame_buffer if key in frame]
            for key in keys
        }
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> dict[str, list[np.ndarray]]:
        selected = self._frame_buffer[start:end]
        keys = sorted({key for frame in selected for key in frame})
        return {
            key: [frame[key].copy() for frame in selected if key in frame]
            for key in keys
        }

    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        try:
            frame: dict[str, np.ndarray] = {}
            global_frame = self._external_rgb_frame()
            if global_frame is not None:
                frame["global"] = self._frame_to_numpy(global_frame)
                self._last_video_sources["global"] = "external_sensors.global_camera"
            robot_path, robot_frame = self._robot_rgb_from_obs_with_path(self.get_observation())
            if robot_frame is not None:
                frame["robot"] = self._frame_to_numpy(robot_frame)
                self._last_video_sources["robot"] = robot_path
            if not frame:
                frame["rgb"] = self.render()
                self._last_video_sources["rgb"] = "render()"
            self._frame_buffer.append(frame)
        except Exception as e:
            print("X2 video frame capture failed:", e)

    def _external_rgb_frame(self) -> Any | None:
        if not getattr(self.env, "external_sensors", None):
            return None
        sensor = self.env.external_sensors.get("global_camera")
        if sensor is None:
            sensor = next(iter(self.env.external_sensors.values()))
        try:
            if og is not None:
                og.sim.render()
            return sensor.get_obs()[0].get("rgb")
        except Exception as e:
            print("X2 external camera capture failed:", e)
            return None

    @classmethod
    def _first_rgb_from_obs(cls, obs: Any) -> Any:
        _path, frame = cls._first_rgb_from_obs_with_path(obs)
        return frame

    @classmethod
    def _first_rgb_from_obs_with_path(cls, obs: Any, path: str = "") -> tuple[str, Any]:
        if isinstance(obs, dict):
            if "rgb" in obs:
                return f"{path}.rgb".lstrip("."), obs["rgb"]
            for key, value in obs.items():
                frame_path, frame = cls._first_rgb_from_obs_with_path(
                    value,
                    f"{path}.{key}" if path else str(key),
                )
                if frame is not None:
                    return frame_path, frame
        return "", None

    def _robot_rgb_from_obs_with_path(self, obs: Any) -> tuple[str, Any]:
        candidates = self._all_rgb_from_obs_with_path(obs)
        if not candidates:
            return "", None

        preferred_tokens = {
            "left": ["l_base_gripper", "left_gripper", "gripper_left"],
            "right": ["r_base_gripper", "right_gripper", "gripper_right"],
        }.get(self._video_robot_arm, [])
        for token in preferred_tokens:
            for path, frame in candidates:
                if token in path:
                    return path, frame
        return candidates[0]

    @classmethod
    def _all_rgb_from_obs_with_path(cls, obs: Any, path: str = "") -> list[tuple[str, Any]]:
        frames: list[tuple[str, Any]] = []
        if isinstance(obs, dict):
            if "rgb" in obs:
                frames.append((f"{path}.rgb".lstrip("."), obs["rgb"]))
            for key, value in obs.items():
                frames.extend(cls._all_rgb_from_obs_with_path(value, f"{path}.{key}" if path else str(key)))
        return frames

    @staticmethod
    def _frame_to_numpy(frame: Any) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            max_value = float(np.nanmax(frame)) if frame.size else 0.0
            if max_value <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def get_joint_positions(self) -> torch.Tensor:
        return self.robot.get_joint_positions()

    def get_robot_eef_pose(self, arm: int | str = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.get_eef_pose(arm=self._arm_name(arm))

    def get_robot_relative_eef_pose(self, arm: int | str = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.get_relative_eef_pose(arm=self._arm_name(arm))

    def _settle_robot(self) -> int:
        return self.settle_robot_steps(steps=12)

    def settle_robot_steps(self, steps: int = 12) -> int:
        for _ in range(max(0, int(steps))):
            self.step(self.controller._postprocess_action(self.controller._empty_action(follow_arm_targets=False)))
        return max(0, int(steps))

    def get_gripper_state(self, arm: int | str = 1) -> dict[str, Any]:
        arm_name = self._arm_name(arm)
        q = self.robot.get_joint_positions()
        joint_indices = self.robot.gripper_control_idx[arm_name]
        q_gripper = q[joint_indices]
        state = {
            "arm": arm_name,
            "joint_names": [list(self.robot.joints.keys())[int(i)] for i in self._debug_list(joint_indices)],
            "qpos": self._debug_list(q_gripper),
            "qpos_min": round(float(torch.min(q_gripper).detach().cpu()), 6),
            "qpos_max": round(float(torch.max(q_gripper).detach().cpu()), 6),
            "finger_span_y_eef": None,
        }

        finger_links = self.robot.finger_links.get(arm_name, [])
        if len(finger_links) >= 2:
            try:
                import omnigibson.utils.transform_utils as T

                eef_pos, eef_quat = self.robot.eef_links[arm_name].get_position_orientation()
                world_to_eef = T.pose_inv(T.pose2mat((eef_pos, eef_quat)))
                positions_eef = []
                for link in finger_links:
                    pos, _ = link.get_position_orientation()
                    pos_h = torch.cat((pos, torch.ones(1, dtype=pos.dtype, device=pos.device)))
                    pos_eef = world_to_eef @ pos_h
                    positions_eef.append(pos_eef[:3].detach().cpu())
                positions = torch.stack(positions_eef)
                ys = positions[:, 1].tolist()
                state["finger_span_y_eef"] = round(max(ys) - min(ys), 6)
                state["finger_positions_eef"] = [[round(float(v), 6) for v in row] for row in positions.tolist()]
                state["finger_center_eef"] = [round(float(v), 6) for v in torch.mean(positions, dim=0).tolist()]
            except Exception as e:
                state["finger_span_error"] = str(e)
        return state

    def _open_close_gripper(self, arm: int | str = 1, max_steps: int = 250, open: bool = True) -> int:
        arm_name = self._set_arm(arm)
        self._hold_current_hand_target(arm_name)
        # The X2 Robotiq85 model has a wide finger span at the lower joint limit
        # and a narrow span at the upper joint limit.
        limit = "lower" if open else "upper"
        steps = 0
        for steps, action in enumerate(self.controller._move_fingers_to_limit(limit), start=1):
            if steps > max_steps:
                break
            if action is not None:
                self.step(action)
        self._hold_current_hand_target(arm_name)
        return steps

    def _move_hand_direct_ik(
        self,
        target_pose: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
        arm: int | str = 1,
        pos_thresh: float = 0.02,
        ori_thresh: float = 0.4,
        stop_on_contact: bool = False,
        ignore_failure: bool = False,
        stop_if_stuck: bool = False,
        stuck_patience_steps: int = 30,
        stuck_pos_thresh: float = 0.0003,
        stuck_ori_thresh: float = 0.01,
        max_steps: int = 1000,
    ) -> bool:
        arm_name = self._set_arm(arm)
        try:
            self._execute_controller(
                self._move_hand_direct_ik_with_stuck_patience(
                    self._to_tensor_pose(target_pose),
                    pos_thresh=pos_thresh,
                    ori_thresh=ori_thresh,
                    stop_on_contact=stop_on_contact,
                    ignore_failure=ignore_failure,
                    stop_if_stuck=stop_if_stuck,
                    stuck_patience_steps=stuck_patience_steps,
                    stuck_pos_thresh=stuck_pos_thresh,
                    stuck_ori_thresh=stuck_ori_thresh,
                    max_steps=max_steps,
                )
            )
            return True
        except TimeoutError:
            raise
        except Exception as e:
            print("X2 direct IK hand move failed:", e)
            return False
        finally:
            self._hold_current_hand_target(arm_name)

    def _move_hand_direct_ik_with_stuck_patience(
        self,
        target_pose: tuple[torch.Tensor, torch.Tensor],
        pos_thresh: float = 0.02,
        ori_thresh: float = 0.4,
        stop_on_contact: bool = False,
        ignore_failure: bool = False,
        stop_if_stuck: bool = False,
        stuck_patience_steps: int = 30,
        stuck_pos_thresh: float = 0.0003,
        stuck_ori_thresh: float = 0.01,
        max_steps: int = 1000,
    ):
        """Run OG direct IK, but require consecutive stalled steps before stuck failure."""
        import omnigibson.utils.transform_utils as T
        from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim

        controller_config = self.robot._controller_config["arm_" + self.controller.arm]
        assert controller_config["name"] == "InverseKinematicsController", "Controller must be InverseKinematicsController"
        assert controller_config["mode"] in ("pose_absolute_ori", "pose_delta_ori"), (
            "Controller must be in pose_absolute_ori or pose_delta_ori mode"
        )

        target_pose = self.controller._world_pose_to_robot_pose(target_pose)
        target_pos, target_orn = target_pose
        target_orn_axisangle = T.quat2axisangle(target_orn)
        self.controller._arm_targets[f"arm_{self.controller.arm}"] = (target_pos, target_orn_axisangle)

        prev_pos = None
        prev_orn = None
        stalled_steps = 0
        patience = max(1, int(stuck_patience_steps))

        for _ in range(max(1, int(max_steps))):
            current_pose = self.controller._world_pose_to_robot_pose(
                (self.robot.get_eef_position(self.controller.arm), self.robot.get_eef_orientation(self.controller.arm))
            )
            current_pos, current_orn = current_pose
            target_pos_diff = torch.linalg.norm(target_pos - current_pos)
            target_orn_diff = T.get_orientation_diff_in_radian(current_orn, target_orn)
            if target_pos_diff < pos_thresh and target_orn_diff < ori_thresh:
                return

            if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                return

            if stop_if_stuck and prev_pos is not None and prev_orn is not None:
                pos_step = torch.linalg.norm(prev_pos - current_pos)
                orn_step = T.get_orientation_diff_in_radian(current_orn, prev_orn)
                if pos_step < stuck_pos_thresh and orn_step < stuck_ori_thresh:
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                if stalled_steps >= patience:
                    raise RuntimeError(
                        "Hand is stuck: "
                        f"stalled_steps={stalled_steps}, "
                        f"target_pos_error={float(target_pos_diff):.6f}, "
                        f"target_ori_error={float(target_orn_diff):.6f}"
                    )

            prev_pos = current_pos
            prev_orn = current_orn

            action = self.controller._empty_action()
            if controller_config["mode"] == "pose_delta_ori":
                action_idx = self.robot.controller_action_idx[f"arm_{self.controller.arm}"]
                action[action_idx[:3]] *= self._ik_pose_delta_pos_gain
                action[action_idx[3:6]] *= self._ik_pose_delta_ori_gain
            yield self.controller._postprocess_action(action)

        if not ignore_failure:
            raise RuntimeError("Your hand was obstructed from moving to the desired joint position")

    def _move_hand(
        self,
        target_pose: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
        arm: int | str = 1,
        pos_thresh: float = 0.005,
        ori_thresh: float = 0.1,
        stop_on_contact: bool = False,
        ignore_failure: bool = False,
        stop_if_stuck: bool = True,
        stuck_patience_steps: int = 30,
        stuck_pos_thresh: float = 0.0003,
        stuck_ori_thresh: float = 0.01,
        max_steps: int = 1000,
        **kwargs,
    ) -> bool:
        arm_name = self._set_arm(arm)
        if kwargs:
            print(f"X2 hand move ignoring unsupported motion kwargs: {sorted(kwargs)}")
        try:
            self.settle_robot_steps(steps=12)
            self._execute_controller(
                self._move_hand_direct_ik_with_stuck_patience(
                    self._to_tensor_pose(target_pose),
                    pos_thresh=pos_thresh,
                    ori_thresh=ori_thresh,
                    stop_on_contact=stop_on_contact,
                    ignore_failure=ignore_failure,
                    stop_if_stuck=stop_if_stuck,
                    stuck_patience_steps=stuck_patience_steps,
                    stuck_pos_thresh=stuck_pos_thresh,
                    stuck_ori_thresh=stuck_ori_thresh,
                    max_steps=max_steps,
                )
            )
            return True
        except TimeoutError:
            raise
        except Exception as e:
            print("X2 hand move failed:", e)
            return False
        finally:
            self._hold_current_hand_target(arm_name)

    def _sample_grasp_pose(self, obj_name: str, object_obb: Any | None = None, arm: int | str = 1):
        del object_obb
        self._set_arm(arm)
        obj = self.env.scene.object_registry("name", obj_name)
        if obj is None:
            raise ValueError(f"Object {obj_name!r} not found in scene registry")

        obj_pos, _ = obj.get_position_orientation()
        eef_pos, eef_quat = self.get_robot_eef_pose(arm=arm)
        dtype = eef_pos.dtype
        device = eef_pos.device

        obj_pos = obj_pos.to(dtype=dtype, device=device)
        try:
            extent = torch.as_tensor(obj.aabb_extent, dtype=dtype, device=device)
            object_half_height = torch.clamp(0.5 * extent[2], min=0.015, max=0.08)
        except Exception:
            extent = torch.tensor([0.0, 0.0, 0.06], dtype=dtype, device=device)
            object_half_height = torch.tensor(0.03, dtype=dtype, device=device)

        grasp_pos = obj_pos.clone()
        grasp_pos[2] = obj_pos[2] + object_half_height

        approach_dir = grasp_pos - eef_pos
        approach_dir[2] = 0.0
        approach_norm = torch.linalg.norm(approach_dir)
        if approach_norm < 1e-6:
            approach_dir = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        else:
            approach_dir = approach_dir / approach_norm

        pregrasp_pos = grasp_pos - 0.06 * approach_dir
        pregrasp_pos[2] = grasp_pos[2] + 0.03
        self._last_sample_grasp_debug = {
            "object_name": obj_name,
            "object_pos_read_in_sampler": self._debug_list(obj_pos),
            "eef_pos_read_in_sampler": self._debug_list(eef_pos),
            "aabb_extent_read_in_sampler": self._debug_list(extent),
            "object_half_height": round(float(object_half_height.detach().cpu()), 6),
            "approach_dir": self._debug_list(approach_dir),
            "pregrasp_pos_returned": self._debug_list(pregrasp_pos),
            "grasp_pos_returned": self._debug_list(grasp_pos),
        }
        return (pregrasp_pos, eef_quat.clone()), (grasp_pos, eef_quat.clone())

    def _execute_release(self, arm: int | str = 1) -> int:
        self._set_arm(arm)
        return self._execute_controller(self.controller._execute_release())

    def check_object_in_hand(self, arm: int | str = 1) -> bool:
        self._set_arm(arm)
        self._settle_robot()
        return self.controller._get_obj_in_hand() is not None

    def _lift_arm(self, arm: int | str = 1, distance: float = 0.05) -> bool:
        pos, quat = self.get_robot_eef_pose(arm=arm)
        target_pos = pos + torch.tensor([0.0, 0.0, distance], dtype=pos.dtype, device=pos.device)
        return self._move_hand((target_pos, quat), arm=arm)

    def _robot_joint_name_to_index(self) -> dict[str, int]:
        if self._x2_joint_name_to_index is None:
            self._x2_joint_name_to_index = {str(name): idx for idx, name in enumerate(self.robot.joints.keys())}
        return self._x2_joint_name_to_index

    def _arm_joint_names(self, arm: int | str = 1) -> list[str]:
        arm_name = self._arm_name(arm)
        names = getattr(self.robot, "arm_joint_names", None)
        if isinstance(names, dict) and arm_name in names:
            return [str(name) for name in names[arm_name]]
        prefix = f"{arm_name}_"
        return [name for name in self._robot_joint_name_to_index() if name.startswith(prefix) and "finger" not in name]

    def _get_x2_pyroki_context(self):
        if self._x2_pyroki_ctx is None:
            from capx.integrations.motion.pyroki_context import get_pyroki_context

            self._x2_pyroki_ctx = get_pyroki_context(_ensure_x2_pyroki_ik_urdf(), target_link_name="r_base_gripper")
            self._x2_pyroki_joint_names = [str(name) for name in self._x2_pyroki_ctx.robot.joints.actuated_names]
        return self._x2_pyroki_ctx

    def _current_pyroki_q(self) -> np.ndarray:
        ctx = self._get_x2_pyroki_context()
        q_all = self.robot.get_joint_positions().detach().cpu().numpy()
        name_to_idx = self._robot_joint_name_to_index()
        values = []
        for joint_name in ctx.robot.joints.actuated_names:
            if str(joint_name) not in name_to_idx:
                raise KeyError(f"PyRoKi joint {joint_name!r} is not present in OG X2 joints")
            values.append(float(q_all[name_to_idx[str(joint_name)]]))
        return np.asarray(values, dtype=np.float64)

    def _pyroki_fk_pose_base(self, q_pyroki: np.ndarray, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        ctx = self._get_x2_pyroki_context()
        fk = ctx.robot.forward_kinematics(np.asarray(q_pyroki, dtype=np.float64).reshape(-1))
        link_idx = ctx.robot.links.names.index(link_name)
        pose = np.asarray(fk[link_idx], dtype=np.float64).reshape(7)
        quat_wxyz = pose[:4]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        return pose[4:].copy(), quat_xyzw / max(float(np.linalg.norm(quat_xyzw)), 1e-12)

    def _solve_pyroki_eef_joint_target(
        self,
        target_pose_world: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
        arm: int | str = 1,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import jax.numpy as jnp
        from capx.integrations.motion import pyroki_snippets as pks

        arm_name = self._arm_name(arm)
        target_link_name = self.robot.eef_link_names[arm_name]
        target_pos_world, target_quat_world = self._to_tensor_pose(target_pose_world)
        target_pos_world_np = target_pos_world.detach().cpu().numpy().astype(np.float64)
        target_quat_world_np = target_quat_world.detach().cpu().numpy().astype(np.float64)

        ctx = self._get_x2_pyroki_context()
        current_q_pyroki = self._current_pyroki_q()
        cur_base_pos, cur_base_quat = self._pyroki_fk_pose_base(current_q_pyroki, target_link_name)
        cur_world_pos, cur_world_quat = self.get_robot_eef_pose(arm=arm_name)
        cur_world_pos_np = cur_world_pos.detach().cpu().numpy().astype(np.float64)
        cur_world_quat_np = cur_world_quat.detach().cpu().numpy().astype(np.float64)

        T_world_eef_current = self._pose_to_matrix(cur_world_pos_np, cur_world_quat_np)
        T_base_eef_current = self._pose_to_matrix(cur_base_pos, cur_base_quat)
        T_world_base = T_world_eef_current @ np.linalg.inv(T_base_eef_current)
        T_world_eef_target = self._pose_to_matrix(target_pos_world_np, target_quat_world_np)
        T_base_eef_target = np.linalg.inv(T_world_base) @ T_world_eef_target
        target_base_pos, target_base_quat = self._matrix_to_pose(T_base_eef_target)
        target_base_wxyz = np.array(
            [target_base_quat[3], target_base_quat[0], target_base_quat[1], target_base_quat[2]],
            dtype=np.float64,
        )

        # PyRoKi sees both arms, but only the selected arm's 7 joints are allowed to change in execution.
        rest_cost_weights = jnp.zeros(ctx.robot.joints.num_actuated_joints)
        solved_q_pyroki = pks.solve_ik_rest(
            robot=ctx.robot,
            target_link_name=target_link_name,
            target_position=target_base_pos,
            target_wxyz=target_base_wxyz,
            rest_cost_weights=rest_cost_weights,
            initial_q=current_q_pyroki,
        )

        q_target_all = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float64).copy()
        robot_name_to_idx = self._robot_joint_name_to_index()
        pyroki_name_to_idx = {str(name): idx for idx, name in enumerate(ctx.robot.joints.actuated_names)}
        selected_joint_names = self._arm_joint_names(arm_name)
        for joint_name in selected_joint_names:
            q_target_all[robot_name_to_idx[joint_name]] = solved_q_pyroki[pyroki_name_to_idx[joint_name]]

        fk_solved_pos, fk_solved_quat = self._pyroki_fk_pose_base(solved_q_pyroki, target_link_name)
        T_world_eef_solved = T_world_base @ self._pose_to_matrix(fk_solved_pos, fk_solved_quat)
        solved_world_pos, solved_world_quat = self._matrix_to_pose(T_world_eef_solved)
        debug = {
            "source_urdf": str(X2_PYROKI_SOURCE_URDF),
            "patched_urdf": str(X2_PYROKI_PATCHED_URDF),
            "arm": arm_name,
            "target_link_name": target_link_name,
            "pyroki_num_actuated": int(ctx.robot.joints.num_actuated_joints),
            "pyroki_joint_names": [str(name) for name in ctx.robot.joints.actuated_names],
            "selected_joint_names": selected_joint_names,
            "target_world_position": self._debug_list(target_pos_world_np),
            "target_world_quat_xyzw": self._debug_list(target_quat_world_np),
            "target_base_position": self._debug_list(target_base_pos),
            "target_base_quat_xyzw": self._debug_list(target_base_quat),
            "current_world_position": self._debug_list(cur_world_pos_np),
            "current_world_quat_xyzw": self._debug_list(cur_world_quat_np),
            "solved_world_position": self._debug_list(solved_world_pos),
            "solved_world_quat_xyzw": self._debug_list(solved_world_quat),
            "solve_fk_pos_error_m": float(np.linalg.norm(solved_world_pos - target_pos_world_np)),
            "solve_fk_ori_error_rad": self._orientation_error_rad(solved_world_quat, target_quat_world_np),
            "current_pyroki_q": self._debug_list(current_q_pyroki),
            "solved_pyroki_q": self._debug_list(solved_q_pyroki),
        }
        self._last_pyroki_ik_debug = debug
        return q_target_all, debug

    def _make_pyroki_world_collision(
        self,
        obstacles_world: list[dict[str, Any]] | None,
        T_base_world: np.ndarray,
    ) -> list[Any]:
        if not obstacles_world:
            return []
        import pyroki as pk

        world_coll = []
        for obs in obstacles_world:
            if obs.get("type") != "box":
                continue
            extent = np.asarray(obs.get("extent", obs.get("size")), dtype=np.float64).reshape(3)
            pos_world = np.asarray(obs.get("position", obs.get("center")), dtype=np.float64).reshape(3)
            quat_world = np.asarray(obs.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(4)
            T_world_obs = self._pose_to_matrix(pos_world, quat_world)
            T_base_obs = T_base_world @ T_world_obs
            pos_base, quat_base = self._matrix_to_pose(T_base_obs)
            quat_base_wxyz = np.asarray(
                [quat_base[3], quat_base[0], quat_base[1], quat_base[2]],
                dtype=np.float64,
            )
            world_coll.append(pk.collision.Box.from_extent(extent=extent, position=pos_base, wxyz=quat_base_wxyz))
        return world_coll

    def _solve_pyroki_eef_trajopt(
        self,
        target_pose_world: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
        arm: int | str = 1,
        obstacles_world: list[dict[str, Any]] | None = None,
        timesteps: int = 16,
        dt: float = 0.08,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        import jax.numpy as jnp
        from capx.integrations.motion import pyroki_snippets as pks

        arm_name = self._arm_name(arm)
        target_link_name = self.robot.eef_link_names[arm_name]
        target_pos_world, target_quat_world = self._to_tensor_pose(target_pose_world)
        target_pos_world_np = target_pos_world.detach().cpu().numpy().astype(np.float64)
        target_quat_world_np = target_quat_world.detach().cpu().numpy().astype(np.float64)

        ctx = self._get_x2_pyroki_context()
        current_q_pyroki = self._current_pyroki_q()
        cur_base_pos, cur_base_quat = self._pyroki_fk_pose_base(current_q_pyroki, target_link_name)
        cur_world_pos, cur_world_quat = self.get_robot_eef_pose(arm=arm_name)
        cur_world_pos_np = cur_world_pos.detach().cpu().numpy().astype(np.float64)
        cur_world_quat_np = cur_world_quat.detach().cpu().numpy().astype(np.float64)

        T_world_eef_current = self._pose_to_matrix(cur_world_pos_np, cur_world_quat_np)
        T_base_eef_current = self._pose_to_matrix(cur_base_pos, cur_base_quat)
        T_world_base = T_world_eef_current @ np.linalg.inv(T_base_eef_current)
        T_base_world = np.linalg.inv(T_world_base)
        T_world_eef_target = self._pose_to_matrix(target_pos_world_np, target_quat_world_np)
        T_base_eef_target = T_base_world @ T_world_eef_target
        target_base_pos, target_base_quat = self._matrix_to_pose(T_base_eef_target)
        target_base_wxyz = np.array(
            [target_base_quat[3], target_base_quat[0], target_base_quat[1], target_base_quat[2]],
            dtype=np.float64,
        )

        rest_cost_weights = jnp.zeros(ctx.robot.joints.num_actuated_joints)
        end_q_pyroki = pks.solve_ik_rest(
            robot=ctx.robot,
            target_link_name=target_link_name,
            target_position=target_base_pos,
            target_wxyz=target_base_wxyz,
            rest_cost_weights=rest_cost_weights,
            initial_q=current_q_pyroki,
        )
        world_coll = self._make_pyroki_world_collision(obstacles_world, T_base_world)
        traj_pyroki = pks.solve_joint_trajopt(
            robot=ctx.robot,
            robot_coll=ctx.robot_coll,
            world_coll=world_coll,
            start_cfg=current_q_pyroki,
            end_cfg=end_q_pyroki,
            timesteps=max(2, int(timesteps)),
            dt=float(dt),
        )
        traj_pos_base = []
        traj_wxyz_base = []
        for q_pyroki in np.asarray(traj_pyroki, dtype=np.float64):
            pos_base, quat_base = self._pyroki_fk_pose_base(q_pyroki, target_link_name)
            traj_pos_base.append(pos_base)
            traj_wxyz_base.append(np.array([quat_base[3], quat_base[0], quat_base[1], quat_base[2]], dtype=np.float64))
        traj_pos_base = np.asarray(traj_pos_base, dtype=np.float64)
        traj_wxyz_base = np.asarray(traj_wxyz_base, dtype=np.float64)

        q_current_all = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float64)
        robot_name_to_idx = self._robot_joint_name_to_index()
        pyroki_name_to_idx = {str(name): idx for idx, name in enumerate(ctx.robot.joints.actuated_names)}
        selected_joint_names = self._arm_joint_names(arm_name)
        q_targets_all: list[np.ndarray] = []
        for q_pyroki in np.asarray(traj_pyroki, dtype=np.float64):
            q_target_all = q_current_all.copy()
            for joint_name in selected_joint_names:
                q_target_all[robot_name_to_idx[joint_name]] = q_pyroki[pyroki_name_to_idx[joint_name]]
            q_targets_all.append(q_target_all)

        fk_final_pos_base, fk_final_quat_base = self._pyroki_fk_pose_base(np.asarray(traj_pyroki[-1]), target_link_name)
        T_world_eef_final = T_world_base @ self._pose_to_matrix(fk_final_pos_base, fk_final_quat_base)
        final_world_pos, final_world_quat = self._matrix_to_pose(T_world_eef_final)
        debug = {
            "source": "pyroki_joint_trajopt_world_collision",
            "planner": "solve_joint_trajopt_fixed_endpoints",
            "arm": arm_name,
            "target_link_name": target_link_name,
            "timesteps": int(timesteps),
            "dt": float(dt),
            "world_collision_count": int(len(world_coll)),
            "selected_joint_names": selected_joint_names,
            "target_world_position": self._debug_list(target_pos_world_np),
            "target_world_quat_xyzw": self._debug_list(target_quat_world_np),
            "target_base_position": self._debug_list(target_base_pos),
            "target_base_quat_xyzw": self._debug_list(target_base_quat),
            "current_pyroki_q": self._debug_list(current_q_pyroki),
            "end_ik_pyroki_q": self._debug_list(end_q_pyroki),
            "traj_first_pyroki_q": self._debug_list(np.asarray(traj_pyroki[0])),
            "traj_final_pyroki_q": self._debug_list(np.asarray(traj_pyroki[-1])),
            "traj_pos_base": [self._debug_list(row) for row in np.asarray(traj_pos_base)],
            "traj_wxyz_base": [self._debug_list(row) for row in np.asarray(traj_wxyz_base)],
            "final_world_position": self._debug_list(final_world_pos),
            "final_world_quat_xyzw": self._debug_list(final_world_quat),
            "final_fk_pos_error_m": float(np.linalg.norm(final_world_pos - target_pos_world_np)),
            "final_fk_ori_error_rad": self._orientation_error_rad(final_world_quat, target_quat_world_np),
        }
        self._last_pyroki_trajopt_debug = debug
        return q_targets_all, debug

    def _joint_position_action(self, q_target_all: np.ndarray) -> torch.Tensor:
        from omnigibson.controllers.controller_view import ControllerView

        q_target = torch.as_tensor(q_target_all, dtype=torch.float32, device=self.robot.get_joint_positions().device)
        action = self.controller._empty_action(follow_arm_targets=False)
        for arm_name in self.robot.arm_names:
            controller_name = f"arm_{arm_name}"
            group_key, _controller_idx = self.robot.controllers[controller_name]
            action_idx = self.robot.controller_action_idx[controller_name]
            dof_idx = ControllerView.get_dof_idx(group_key)
            command = q_target[dof_idx]
            action[action_idx] = ControllerView.reverse_preprocess_command(group_key, command)
        return self.controller._postprocess_action(action)

    def _move_to_joint_positions(
        self,
        target_joint_positions,
        arm: int | str | None = None,
        max_joint_step: float = 0.035,
        max_steps: int = 180,
        settle_steps: int = 12,
        pos_tol: float = 0.015,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        q_current = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float64)
        q_target = np.asarray(target_joint_positions, dtype=np.float64).reshape(-1)
        if q_target.shape[0] != q_current.shape[0]:
            raise ValueError(f"target_joint_positions must have shape ({q_current.shape[0]},), got {q_target.shape}")

        if arm is None:
            move_indices = np.arange(q_current.shape[0], dtype=np.int64)
        else:
            arm_name = self._arm_name(arm)
            name_to_idx = self._robot_joint_name_to_index()
            move_indices = np.asarray([name_to_idx[name] for name in self._arm_joint_names(arm_name)], dtype=np.int64)
            keep = np.ones(q_current.shape[0], dtype=bool)
            keep[move_indices] = False
            q_target[keep] = q_current[keep]

        max_delta = float(np.max(np.abs(q_target[move_indices] - q_current[move_indices]))) if move_indices.size else 0.0
        steps = min(max(1, int(np.ceil(max_delta / max(float(max_joint_step), 1e-6)))), max(1, int(max_steps)))
        hold_steps = max(1, int(hold_steps_per_waypoint))
        for step_idx in range(1, steps + 1):
            alpha = step_idx / steps
            q_cmd = q_current + alpha * (q_target - q_current)
            action = self._joint_position_action(q_cmd)
            for _ in range(hold_steps):
                self.step(action)
        self.settle_robot_steps(steps=settle_steps)

        q_reached = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float64)
        err = float(np.max(np.abs(q_reached[move_indices] - q_target[move_indices]))) if move_indices.size else 0.0
        action_frequency = float(self.controller_cfg.get("env", {}).get("action_frequency", 0.0) or 0.0)
        physics_frequency = float(self.controller_cfg.get("env", {}).get("physics_frequency", 0.0) or 0.0)
        self._last_joint_position_move_debug = {
            "arm": None if arm is None else self._arm_name(arm),
            "move_indices": move_indices.astype(int).tolist(),
            "max_joint_delta_rad": max_delta,
            "steps": int(steps),
            "hold_steps_per_waypoint": int(hold_steps),
            "action_steps_sent": int(steps * hold_steps),
            "settle_steps": int(settle_steps),
            "max_final_joint_error_rad": err,
            "max_joint_step": float(max_joint_step),
            "action_frequency_hz": action_frequency,
            "physics_frequency_hz": physics_frequency,
            "seconds_per_action_step": (1.0 / action_frequency) if action_frequency > 0.0 else None,
            "estimated_command_duration_s": (float(steps * hold_steps) / action_frequency) if action_frequency > 0.0 else None,
            "estimated_settle_duration_s": (float(settle_steps) / action_frequency) if action_frequency > 0.0 else None,
        }
        return bool(err <= float(pos_tol))

    def _move_through_joint_trajectory(
        self,
        joint_trajectory: list[np.ndarray] | np.ndarray,
        arm: int | str | None = None,
        max_joint_step: float = 0.035,
        max_steps_per_waypoint: int = 80,
        settle_steps: int = 12,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        traj = np.asarray(joint_trajectory, dtype=np.float64)
        if traj.ndim != 2:
            raise ValueError(f"joint_trajectory must be 2D, got shape {traj.shape}")
        waypoint_results = []
        ok = True
        for idx, q_target in enumerate(traj):
            waypoint_ok = self._move_to_joint_positions(
                q_target,
                arm=arm,
                max_joint_step=max_joint_step,
                max_steps=max_steps_per_waypoint,
                settle_steps=0,
                hold_steps_per_waypoint=hold_steps_per_waypoint,
            )
            waypoint_debug = dict(getattr(self, "_last_joint_position_move_debug", {}) or {})
            waypoint_debug["waypoint_index"] = int(idx)
            waypoint_debug["ok"] = bool(waypoint_ok)
            waypoint_results.append(waypoint_debug)
            ok = bool(ok and waypoint_ok)
        self.settle_robot_steps(steps=settle_steps)
        self._last_joint_trajectory_move_debug = {
            "arm": None if arm is None else self._arm_name(arm),
            "waypoint_count": int(traj.shape[0]),
            "max_joint_step": float(max_joint_step),
            "max_steps_per_waypoint": int(max_steps_per_waypoint),
            "settle_steps": int(settle_steps),
            "hold_steps_per_waypoint": int(hold_steps_per_waypoint),
            "waypoints": waypoint_results,
            "success": bool(ok),
        }
        return bool(ok)

    def _move_hand_joint_ik(
        self,
        target_pose: tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor],
        arm: int | str = 1,
        pos_thresh: float = 0.01,
        ori_thresh: float = 0.18,
        max_joint_step: float = 0.035,
        max_steps: int = 180,
        settle_steps: int = 12,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        arm_name = self._set_arm(arm)
        q_target, solve_debug = self._solve_pyroki_eef_joint_target(target_pose, arm=arm_name)
        joint_ok = self._move_to_joint_positions(
            q_target,
            arm=arm_name,
            max_joint_step=max_joint_step,
            max_steps=max_steps,
            settle_steps=settle_steps,
            hold_steps_per_waypoint=hold_steps_per_waypoint,
        )
        reached_pos, reached_quat = self.get_robot_eef_pose(arm=arm_name)
        target_pos, target_quat = self._to_tensor_pose(target_pose)
        reached_pos_np = reached_pos.detach().cpu().numpy().astype(np.float64)
        reached_quat_np = reached_quat.detach().cpu().numpy().astype(np.float64)
        target_pos_np = target_pos.detach().cpu().numpy().astype(np.float64)
        target_quat_np = target_quat.detach().cpu().numpy().astype(np.float64)
        pos_err = float(np.linalg.norm(reached_pos_np - target_pos_np))
        ori_err = self._orientation_error_rad(reached_quat_np, target_quat_np)
        self._last_joint_ik_move_debug = {
            "solve": solve_debug,
            "joint_move": getattr(self, "_last_joint_position_move_debug", {}),
            "reached_world_position": self._debug_list(reached_pos_np),
            "reached_world_quat_xyzw": self._debug_list(reached_quat_np),
            "final_pos_error_m": pos_err,
            "final_ori_error_rad": ori_err,
            "joint_ok": bool(joint_ok),
            "success": bool(joint_ok and pos_err <= pos_thresh and ori_err <= ori_thresh),
        }
        return bool(self._last_joint_ik_move_debug["success"])
