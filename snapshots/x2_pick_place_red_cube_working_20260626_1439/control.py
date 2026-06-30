"""CAP-X control API for fixed-base X2 BEHAVIOR manipulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.x2 import vision as x2_vision
from capx.integrations.x2.vision import estimate_position_from_mask, estimate_pose_from_mask


DEFAULT_GRASPNET_RAW_TO_X2_TCP_POS = np.array([0.003604, 0.002704, -0.059466], dtype=np.float64)
DEFAULT_GRASPNET_RAW_TO_X2_TCP_QUAT_XYZW = np.array([0.488425, 0.670197, 0.342387, -0.441642], dtype=np.float64)


class X2ControlApi(ApiBase):
    """Robot control helpers for X2.

    This API intentionally exposes only fixed-base manipulation functions.
    Base navigation / base rotation are not part of the current X2 phase.
    """

    def __init__(
        self,
        env: BaseEnv,
        *,
        use_vision_models: bool = False,
        owlvit_device: str = "cpu",
        owlvit_threshold: float = 0.05,
        sam2_device: str = "cuda",
        use_graspnet: bool = False,
        graspnet_device: str = "cuda",
    ) -> None:
        super().__init__(env)
        self.use_vision_models = bool(use_vision_models)
        self.use_graspnet = bool(use_graspnet)
        self.graspnet_device = graspnet_device
        self.owl_vit_det_fn = None
        self.sam2_seg_fn = None
        self.grasp_net_plan_fn = None
        if self.use_vision_models:
            from capx.integrations.vision.owlvit import init_owlvit
            from capx.integrations.vision.sam2 import init_sam2

            self.owl_vit_det_fn = init_owlvit(device=owlvit_device, threshold=owlvit_threshold)
            self.sam2_seg_fn = init_sam2(device=sam2_device)
        if self.use_graspnet:
            self._ensure_graspnet()

    def functions(self) -> dict[str, Any]:
        """Return the dictionary of functions exposed to generated code."""
        return {
            "get_env_observation": self.get_env_observation,
            "get_camera_names": self.get_camera_names,
            "get_camera_observation": self.get_camera_observation,
            "get_wrist_camera_observation": self.get_wrist_camera_observation,
            "get_chest_camera_name": self.get_chest_camera_name,
            "get_chest_camera_observation": self.get_chest_camera_observation,
            "get_chest_camera_pose": self.get_chest_camera_pose,
            "get_chest_camera_intrinsics": self.get_chest_camera_intrinsics,
            "get_camera_pose": self.get_camera_pose,
            "get_camera_intrinsics": self.get_camera_intrinsics,
            "get_external_camera_names": self.get_external_camera_names,
            "get_external_camera_observation": self.get_external_camera_observation,
            "get_external_camera_pose": self.get_external_camera_pose,
            "get_external_camera_intrinsics": self.get_external_camera_intrinsics,
            "point_prompt_molmo": self.point_prompt_molmo,
            "segment_sam3_text_prompt": self.segment_sam3_text_prompt,
            "segment_sam3_point_prompt": self.segment_sam3_point_prompt,
            "get_sam3_mask": self.get_sam3_mask,
            "detect_object_owlvit": self.detect_object_owlvit,
            "segment_sam2": self.segment_sam2,
            "get_object_mask_sam2": self.get_object_mask_sam2,
            "get_sam2_mask": self.get_sam2_mask,
            "estimate_position_from_mask": self.estimate_position_from_mask,
            "get_object_pose": self.get_object_pose,
            "save_current_observation": self.save_current_observation,
            "get_current_joint_positions": self.get_current_joint_positions,
            "get_current_eef_pose": self.get_current_eef_pose,
            "get_robot_relative_eef_pose": self.get_robot_relative_eef_pose,
            "move_hand": self.move_hand,
            "settle_robot": self.settle_robot,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_gripper_state": self.get_gripper_state,
            "get_last_motion_debug": self.get_last_motion_debug,
            "get_control_timing": self.get_control_timing,
            "get_tcp_offset_eef": self.get_tcp_offset_eef,
            "get_current_tcp_pose": self.get_current_tcp_pose,
            "tcp_pose_to_eef_pose": self.tcp_pose_to_eef_pose,
            "move_tcp": self.move_tcp,
            "move_hand_joint_ik": self.move_hand_joint_ik,
            "move_tcp_joint_ik": self.move_tcp_joint_ik,
            "plan_tcp_pyroki_trajopt": self.plan_tcp_pyroki_trajopt,
            "move_tcp_pyroki_trajopt": self.move_tcp_pyroki_trajopt,
            "adapt_graspnet_raw_pose_to_x2_tcp": self.adapt_graspnet_raw_pose_to_x2_tcp,
            "select_adapted_graspnet_tcp_candidate": self.select_adapted_graspnet_tcp_candidate,
            "plan_visual_grasp_tcp_pose": self.plan_visual_grasp_tcp_pose,
            "pick_and_place_visual_object": self.pick_and_place_visual_object,
            "execute_tcp_grasp_plan": self.execute_tcp_grasp_plan,
            "sample_grasp_pose": self.sample_grasp_pose,
            "sample_grasp_pose_graspnet": self.sample_grasp_pose_graspnet,
            "plan_graspnet_from_mask": self.plan_graspnet_from_mask,
            "plan_x2_grasp_execution": self.plan_x2_grasp_execution,
            "plan_x2_guarded_grasp_approach": self.plan_x2_guarded_grasp_approach,
            "get_sim_known_tabletop_obstacles": self.get_sim_known_tabletop_obstacles,
            "grasp_object": self.grasp_object,
            "check_object_in_hand": self.check_object_in_hand,
            "move_to_joint_positions": self.move_to_joint_positions,
            "solve_ik": self.solve_ik,
            "lift_arm": self.lift_arm,
        }

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _rgb_u8(cls, value: Any) -> np.ndarray:
        rgb = cls._to_numpy(value)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise ValueError(f"Expected an RGB/RGBA image, got shape {rgb.shape}")
        rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
            if max_value <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _best_by_score(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        return max(items, key=lambda item: float(item.get("score", 0.0)))

    @staticmethod
    def _translation_matrix(offset_xyz: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(offset_xyz, dtype=np.float64).reshape(3)
        return T

    @staticmethod
    def _pose_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return T[:3, 3].copy(), x2_vision.matrix_to_quat_xyzw(T[:3, :3])

    @staticmethod
    def _quat_error_rad(a: np.ndarray, b: np.ndarray) -> float:
        qa = np.asarray(a, dtype=np.float64).reshape(4)
        qb = np.asarray(b, dtype=np.float64).reshape(4)
        qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
        qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
        return float(2.0 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))

    @staticmethod
    def _mask_depth_hint(mask: np.ndarray, depth: np.ndarray) -> tuple[float | None, float | None]:
        mask_bool = np.asarray(mask, dtype=bool)
        depth_arr = np.squeeze(np.asarray(depth, dtype=np.float64))
        if mask_bool.shape != depth_arr.shape:
            return None, None
        values = depth_arr[mask_bool & np.isfinite(depth_arr) & (depth_arr > 0.0)]
        if values.size == 0:
            return None, None
        median = float(np.median(values))
        spread = float(np.percentile(values, 90.0) - np.percentile(values, 10.0)) if values.size >= 8 else 0.08
        return median, max(0.035, min(0.18, 0.5 * spread + 0.03))

    @staticmethod
    def _pose_right_transform(
        pose: tuple[np.ndarray, np.ndarray],
        right_transform: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        pose_pos, pose_quat = pose
        tf_pos, tf_quat = right_transform
        T_pose = x2_vision.pose_to_matrix(pose_pos, pose_quat)
        T_tf = x2_vision.pose_to_matrix(tf_pos, tf_quat)
        return X2ControlApi._pose_from_matrix(T_pose @ T_tf)

    @staticmethod
    def _convert_T_cam_cv_to_cam_gl(T_cam_cv_grasp: np.ndarray) -> np.ndarray:
        """Convert a Contact-GraspNet CV-camera pose into OG's GL camera frame."""
        A = np.eye(4, dtype=np.float64)
        A[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return A @ np.asarray(T_cam_cv_grasp, dtype=np.float64).reshape(4, 4) @ A

    @staticmethod
    def _point_cam_cv_to_world(point_cam_cv: np.ndarray, T_world_cam_gl: np.ndarray) -> np.ndarray:
        point_cv = np.asarray(point_cam_cv, dtype=np.float64).reshape(3)
        point_gl = np.array([point_cv[0], -point_cv[1], -point_cv[2], 1.0], dtype=np.float64)
        return (np.asarray(T_world_cam_gl, dtype=np.float64).reshape(4, 4) @ point_gl)[:3]

    @staticmethod
    def _clean_depth_for_graspnet(depth: np.ndarray) -> np.ndarray:
        depth_arr = np.squeeze(np.asarray(depth, dtype=np.float32)).copy()
        if depth_arr.ndim != 2:
            raise ValueError(f"depth must be 2D after squeeze, got shape {depth_arr.shape}")
        depth_arr[~np.isfinite(depth_arr)] = 0.0
        depth_arr[depth_arr <= 0.0] = 0.0
        return depth_arr

    @staticmethod
    def _within_workspace(position: np.ndarray, workspace_bounds: dict[str, tuple[float, float]] | None) -> bool:
        if workspace_bounds is None:
            return True
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        for idx, axis in enumerate(("x", "y", "z")):
            bounds = workspace_bounds.get(axis)
            if bounds is not None and not (float(bounds[0]) <= float(pos[idx]) <= float(bounds[1])):
                return False
        return True

    def _ensure_graspnet(self):
        if self.grasp_net_plan_fn is None:
            from capx.integrations.vision.graspnet import init_contact_graspnet

            self.grasp_net_plan_fn = init_contact_graspnet(device=self.graspnet_device)
        return self.grasp_net_plan_fn

    def get_env_observation(self) -> dict[str, Any]:
        """Get the latest environment observation."""
        return self._env.get_observation()

    def get_camera_names(self) -> list[str]:
        """Return all camera sensor names attached to the X2 robot."""
        return sorted(str(name) for name in self._env.robot.sensors.keys())

    def get_external_camera_names(self) -> list[str]:
        """Return all external camera sensor names injected into the scene."""
        external_sensors = getattr(self._env.env, "external_sensors", {}) or {}
        return sorted(str(name) for name in external_sensors.keys())

    def _camera_name_for_arm(self, arm: int = 1) -> str:
        arm_name = self._env._arm_name(arm) if hasattr(self._env, "_arm_name") else ("left" if arm == 0 else "right")
        tokens = {
            "left": ("l_base_gripper", "left_gripper", "gripper_left", "left"),
            "right": ("r_base_gripper", "right_gripper", "gripper_right", "right"),
        }.get(arm_name, (arm_name,))
        sensor_names = self.get_camera_names()
        for token in tokens:
            for name in sensor_names:
                if token in name:
                    return name
        raise ValueError(f"No wrist camera found for arm {arm!r}; available cameras: {sensor_names}")

    def get_chest_camera_name(self) -> str:
        """Return the torso-mounted X2 camera name."""
        sensor_names = self.get_camera_names()
        for token in ("lidar_chest_front", "chest", "torso"):
            for name in sensor_names:
                if token in name:
                    return name
        raise ValueError(f"No chest camera found; available cameras: {sensor_names}")

    def _get_camera_sensor(self, camera_name: str | None = None, arm: int = 1):
        if camera_name is None:
            camera_name = self._camera_name_for_arm(arm=arm)
        sensors = self._env.robot.sensors
        if camera_name in sensors:
            return camera_name, sensors[camera_name]
        matches = [name for name in sensors if camera_name in str(name)]
        if len(matches) == 1:
            return matches[0], sensors[matches[0]]
        raise ValueError(f"Camera {camera_name!r} not found; available cameras: {self.get_camera_names()}")

    def _get_external_camera_sensor(self, camera_name: str | None = None):
        sensors = getattr(self._env.env, "external_sensors", {}) or {}
        if not sensors:
            raise ValueError("No external cameras are available in this X2 environment")
        if camera_name is None:
            camera_name = "global_camera" if "global_camera" in sensors else next(iter(sorted(sensors.keys())))
        if camera_name in sensors:
            return camera_name, sensors[camera_name]
        matches = [name for name in sensors if camera_name in str(name)]
        if len(matches) == 1:
            return matches[0], sensors[matches[0]]
        raise ValueError(f"External camera {camera_name!r} not found; available cameras: {self.get_external_camera_names()}")

    @classmethod
    def _find_camera_obs(cls, obs: Any, camera_name: str) -> dict[str, Any] | None:
        if not isinstance(obs, dict):
            return None
        if camera_name in obs and isinstance(obs[camera_name], dict):
            return obs[camera_name]
        for value in obs.values():
            result = cls._find_camera_obs(value, camera_name)
            if result is not None:
                return result
        return None

    @staticmethod
    def _bbox_to_mask(box: list[float] | tuple[float, float, float, float], shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(box[2]), x1, width - 1))
        y2 = int(np.clip(np.ceil(box[3]), y1, height - 1))
        mask = np.zeros((height, width), dtype=bool)
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
        return mask

    @staticmethod
    def _camera_intrinsics_fallback(sensor) -> np.ndarray | None:
        try:
            width = float(sensor.image_width)
            height = float(sensor.image_height)
            focal_length = float(sensor.focal_length)
            horizontal_aperture = float(sensor.horizontal_aperture)
        except Exception:
            return None
        if width <= 0 or height <= 0 or focal_length <= 0 or horizontal_aperture <= 0:
            return None
        fx = focal_length / horizontal_aperture * width
        fy = fx
        return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def get_camera_observation(self, camera_name: str | None = None, arm: int = 1) -> dict[str, np.ndarray]:
        """Return RGB / depth observation leaves for one camera.

        Args:
            camera_name: Exact OmniGibson camera sensor name. If omitted, uses
                the wrist camera for ``arm``.
            arm: Arm index used when ``camera_name`` is omitted. ``0`` is left,
                ``1`` is right.
        """
        resolved_name, _sensor = self._get_camera_sensor(camera_name=camera_name, arm=arm)
        camera_obs = self._find_camera_obs(self.get_env_observation(), resolved_name)
        if camera_obs is None:
            raise RuntimeError(f"Observation has no entry for camera {resolved_name!r}")
        result = {"camera_name": resolved_name}
        for key, value in camera_obs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                result[key] = self._to_numpy(value)
        return result

    def get_wrist_camera_observation(self, arm: int = 1) -> dict[str, np.ndarray]:
        """Return RGB / depth observation leaves for the selected wrist camera."""
        return self.get_camera_observation(camera_name=None, arm=arm)

    def get_chest_camera_observation(self) -> dict[str, np.ndarray]:
        """Return RGB / depth observation leaves for the torso-mounted camera."""
        return self.get_camera_observation(camera_name=self.get_chest_camera_name())

    def get_external_camera_observation(self, camera_name: str | None = None) -> dict[str, np.ndarray]:
        """Return RGB / depth observation leaves for an injected external camera."""
        resolved_name, sensor = self._get_external_camera_sensor(camera_name=camera_name)
        camera_obs = sensor.get_obs()[0]
        result = {"camera_name": resolved_name}
        for key, value in camera_obs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                result[key] = self._to_numpy(value)
        return result

    def get_camera_pose(self, camera_name: str | None = None, arm: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return camera world pose as ``(position, quaternion_xyzw)``."""
        _resolved_name, sensor = self._get_camera_sensor(camera_name=camera_name, arm=arm)
        pos, quat = sensor.get_position_orientation()
        return self._to_numpy(pos), self._to_numpy(quat)

    def get_chest_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return torso-mounted camera world pose as ``(position, quaternion_xyzw)``."""
        return self.get_camera_pose(camera_name=self.get_chest_camera_name())

    def get_external_camera_pose(self, camera_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return external camera world pose as ``(position, quaternion_xyzw)``."""
        _resolved_name, sensor = self._get_external_camera_sensor(camera_name=camera_name)
        pos, quat = sensor.get_position_orientation()
        return self._to_numpy(pos), self._to_numpy(quat)

    def get_camera_intrinsics(self, camera_name: str | None = None, arm: int = 1) -> np.ndarray:
        """Return a 3x3 pinhole camera intrinsic matrix for one X2 camera."""
        _resolved_name, sensor = self._get_camera_sensor(camera_name=camera_name, arm=arm)
        try:
            return self._to_numpy(sensor.intrinsic_matrix)
        except Exception:
            fallback = self._camera_intrinsics_fallback(sensor)
            if fallback is not None:
                return fallback
            raise

    def get_chest_camera_intrinsics(self) -> np.ndarray:
        """Return a 3x3 pinhole intrinsic matrix for the torso-mounted camera."""
        return self.get_camera_intrinsics(camera_name=self.get_chest_camera_name())

    def get_external_camera_intrinsics(self, camera_name: str | None = None) -> np.ndarray:
        """Return a 3x3 pinhole intrinsic matrix for an injected external camera."""
        _resolved_name, sensor = self._get_external_camera_sensor(camera_name=camera_name)
        try:
            return self._to_numpy(sensor.intrinsic_matrix)
        except Exception:
            fallback = self._camera_intrinsics_fallback(sensor)
            if fallback is not None:
                return fallback
            raise

    def _camera_bundle(
        self,
        camera_name: str | None = None,
        arm: int = 1,
        external: bool = False,
        depth_key: str = "depth_linear",
    ) -> dict[str, Any]:
        if external:
            resolved_name, _sensor = self._get_external_camera_sensor(camera_name=camera_name)
            obs = self.get_external_camera_observation(resolved_name)
            K = self.get_external_camera_intrinsics(resolved_name)
            cam_pos, cam_quat = self.get_external_camera_pose(resolved_name)
        else:
            resolved_name, _sensor = self._get_camera_sensor(camera_name=camera_name, arm=arm)
            obs = self.get_camera_observation(resolved_name, arm=arm)
            K = self.get_camera_intrinsics(resolved_name, arm=arm)
            cam_pos, cam_quat = self.get_camera_pose(resolved_name, arm=arm)
        if depth_key not in obs:
            fallback_depth_key = "depth_linear" if "depth_linear" in obs else "depth" if "depth" in obs else None
            if fallback_depth_key is None:
                raise ValueError(f"Camera {resolved_name!r} observation has no depth/depth_linear keys: {sorted(obs.keys())}")
            depth_key = fallback_depth_key
        depth = self._to_numpy(obs[depth_key])
        if depth.ndim == 3:
            depth = np.squeeze(depth)
        return {
            "camera_name": resolved_name,
            "external": bool(external),
            "observation": obs,
            "depth": depth,
            "depth_key": depth_key,
            "intrinsic_matrix": K,
            "camera_position": cam_pos,
            "camera_quat_xyzw": cam_quat,
        }

    def _oracle_mask_for_object(
        self,
        object_name: str,
        bundle: dict[str, Any],
        *,
        margin_px: int = 2,
        min_half_extent: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any], Any]:
        obj = self._env.env.scene.object_registry("name", object_name)
        if obj is None:
            matches = []
            try:
                objects = getattr(self._env.env.scene, "objects", [])
                matches = [o for o in objects if object_name in getattr(o, "name", "")]
            except Exception:
                matches = []
            if len(matches) == 1:
                obj = matches[0]
        if obj is None:
            raise ValueError(f"Object {object_name!r} not found in scene registry for oracle visual mock")
        obj_pos, _obj_quat = obj.get_position_orientation()
        aabb_center = self._to_numpy(getattr(obj, "aabb_center", obj_pos)).astype(np.float64)
        aabb_extent = self._to_numpy(getattr(obj, "aabb_extent", np.array([0.05, 0.05, 0.05]))).astype(np.float64)
        T_world_cam = x2_vision.pose_to_matrix(bundle["camera_position"], bundle["camera_quat_xyzw"])
        mask, detail = x2_vision.make_projected_aabb_mask(
            aabb_center,
            aabb_extent,
            bundle["intrinsic_matrix"],
            T_world_cam,
            bundle["depth"].shape[:2],
            margin_px=margin_px,
            min_half_extent=min_half_extent,
        )
        detail.update({"source": "oracle_projected_aabb", "aabb_center": aabb_center, "aabb_extent": aabb_extent})
        return mask, detail, obj

    def point_prompt_molmo(self, image: np.ndarray, text_prompt: str) -> dict[str, tuple[int | None, int | None]]:
        """R1Pro-compatible placeholder for Molmo point prompting.

        Vision models are intentionally not invoked in the current X2 phase.
        The caller should pass handcrafted masks / boxes or use oracle object
        masks in simulation.
        """
        del image
        return {text_prompt: (None, None)}

    def detect_object_owlvit(self, rgb: np.ndarray, text: str) -> list[dict[str, Any]]:
        """Run OWL-ViT when enabled; otherwise keep the X2 mock behavior."""
        if self.use_vision_models:
            if self.owl_vit_det_fn is None:
                raise RuntimeError("OWL-ViT is enabled but detector function was not initialized")
            return self.owl_vit_det_fn(self._rgb_u8(rgb), texts=[[text]])
        del rgb, text
        return []

    def segment_sam2(
        self,
        rgb: np.ndarray,
        box: list[float] | tuple[float, float, float, float] | None = None,
        max_masks: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run SAM2 when enabled; otherwise keep the rectangular mock mask.

        If a handcrafted box is supplied, this returns a rectangular mask with
        the same result schema as SAM2. Otherwise it returns no masks.
        """
        if self.use_vision_models:
            if self.sam2_seg_fn is None:
                raise RuntimeError("SAM2 is enabled but segmentation function was not initialized")
            results = self.sam2_seg_fn(self._rgb_u8(rgb), box=box, max_masks=max_masks)
            for result in results:
                result.setdefault("box", None if box is None else list(box))
                result.setdefault("source", "sam2")
            return results
        if box is None:
            return []
        height, width = np.asarray(rgb).shape[:2]
        return [{"mask": self._bbox_to_mask(box, (height, width)), "box": list(box), "score": 1.0, "source": "mock_box"}]

    def segment_sam3_text_prompt(self, rgb: np.ndarray, text_prompt: str) -> list[dict[str, Any]]:
        """R1Pro-compatible placeholder for SAM3 text-prompt segmentation."""
        del rgb, text_prompt
        return []

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """R1Pro-compatible placeholder for SAM3 point-prompt segmentation."""
        height, width = np.asarray(rgb).shape[:2]
        x, y = point_coords
        radius = max(3, int(round(min(height, width) * 0.02)))
        box = [x - radius, y - radius, x + radius, y + radius]
        return [{"mask": self._bbox_to_mask(box, (height, width)), "score": 1.0, "point_coords": point_coords, "source": "mock_point"}]

    def get_object_mask_sam2(
        self,
        object_name: str,
        camera_name: str | None = None,
        arm: int = 1,
        external: bool | None = None,
        max_masks: int = 1,
    ) -> dict[str, Any]:
        """Detect an object with OWL-ViT and segment it with SAM2.

        This is the real-model replacement for the X2 oracle 2D mask path.
        It returns the selected detection, selected mask, and camera metadata
        so callers can pass ``mask`` into ``get_object_pose``.
        """
        if not self.use_vision_models:
            raise RuntimeError("get_object_mask_sam2 requires X2ControlApi(use_vision_models=True)")
        if external is None:
            external = bool(self.get_external_camera_names())
        obs = (
            self.get_external_camera_observation(camera_name)
            if external
            else self.get_camera_observation(camera_name=camera_name, arm=arm)
        )
        rgb = self._rgb_u8(obs["rgb"])
        detections = self.detect_object_owlvit(rgb, object_name)
        detection = self._best_by_score(detections)
        if detection is None:
            raise ValueError(f"No OWL-ViT detections for {object_name!r}")
        masks = self.segment_sam2(rgb, box=detection["box"], max_masks=max_masks)
        mask_result = self._best_by_score(masks)
        if mask_result is None:
            raise ValueError(f"No SAM2 masks for {object_name!r} from OWL-ViT box {detection['box']}")
        return {
            "object_name": object_name,
            "camera_name": obs["camera_name"],
            "external": bool(external),
            "rgb": rgb,
            "detections": detections,
            "detection": detection,
            "masks": masks,
            "mask_result": mask_result,
            "mask": np.asarray(mask_result["mask"], dtype=bool),
        }

    def get_sam2_mask(self, object_name: str) -> int:
        """Return SAM2 mask area when enabled, otherwise oracle mask area."""
        if self.use_vision_models:
            return int(np.asarray(self.get_object_mask_sam2(object_name)["mask"], dtype=bool).sum())
        return self.get_sam3_mask(object_name)

    def get_sam3_mask(self, object_name: str) -> int:
        """Return oracle mask area for an object in simulation.

        This preserves R1Pro's return type (mask pixel count) while avoiding
        model invocation until the visual frontend is connected.
        """
        bundle = self._camera_bundle(external=True if self.get_external_camera_names() else False)
        mask, _detail, _obj = self._oracle_mask_for_object(object_name, bundle)
        return int(mask.sum())

    def estimate_position_from_mask(
        self,
        mask: np.ndarray,
        camera_name: str | None = None,
        arm: int = 1,
        external: bool = False,
        depth_key: str = "depth_linear",
        expected_depth: float | None = None,
        depth_window: float | None = None,
    ) -> dict[str, Any]:
        """Estimate a world-frame 3D position from a camera mask.

        Args:
            mask: Binary image mask with shape ``(H, W)``. True pixels are
                backprojected with the selected camera depth image.
            camera_name: Camera sensor name. If omitted, uses the selected
                wrist camera for robot sensors or ``global_camera`` for
                external sensors.
            arm: Arm index used only when ``external`` is False and
                ``camera_name`` is omitted. ``0`` is left, ``1`` is right.
            external: If True, read from injected external sensors such as the
                temporary ``global_camera``.
            depth_key: Preferred depth observation key, usually
                ``depth_linear``.
            expected_depth: Optional positive camera depth used to filter
                background points.
            depth_window: Optional depth tolerance in meters.

        Returns:
            A dictionary containing ``position`` as a ``(3,)`` numpy array,
            point counts, camera metadata, and the raw masked point cloud.
        """
        if external:
            bundle = self._camera_bundle(camera_name=camera_name, arm=arm, external=True, depth_key=depth_key)
        else:
            bundle = self._camera_bundle(camera_name=camera_name, arm=arm, external=False, depth_key=depth_key)

        result = estimate_position_from_mask(
            mask,
            bundle["depth"],
            bundle["intrinsic_matrix"],
            bundle["camera_position"],
            bundle["camera_quat_xyzw"],
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        result.update(
            {
                "camera_name": bundle["camera_name"],
                "external": bundle["external"],
                "depth_key": bundle["depth_key"],
                "intrinsic_matrix": bundle["intrinsic_matrix"],
                "camera_position": bundle["camera_position"],
                "camera_quat_xyzw": bundle["camera_quat_xyzw"],
            }
        )
        return result

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
        mask: np.ndarray | None = None,
        camera_name: str | None = None,
        arm: int = 1,
        external: bool | None = None,
        depth_key: str = "depth_linear",
        method: str = "obb_center",
        expected_depth: float | None = None,
        depth_window: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray]:
        """Get an object's pose using the R1Pro visual-primitive shape.

        Args:
            object_name: Object query / name.
            return_bbox_extent: Whether to return OBB extent.
            mask: Optional handcrafted/model mask. If omitted in simulation,
                an oracle projected AABB mask is used as a temporary stand-in
                for the visual model output.

        Returns:
            ``(position, quaternion_xyzw, bbox_extent)`` when
            ``return_bbox_extent`` is true, otherwise ``(position,
            quaternion_xyzw)``.
        """
        if external is None:
            external = bool(self.get_external_camera_names())
        bundle = self._camera_bundle(camera_name=camera_name, arm=arm, external=bool(external), depth_key=depth_key)
        oracle_detail = None
        if mask is None:
            mask, oracle_detail, _obj = self._oracle_mask_for_object(object_name, bundle)
        pose = estimate_pose_from_mask(
            mask,
            bundle["depth"],
            bundle["intrinsic_matrix"],
            bundle["camera_position"],
            bundle["camera_quat_xyzw"],
            expected_depth=expected_depth,
            depth_window=depth_window,
            method=method,
            remove_outliers=True,
        )
        position = pose["position"]
        quat = pose.get("orientation_quat_xyzw")
        extent = pose.get("bbox_extent")
        if position is None:
            raise RuntimeError(f"Could not estimate object pose for {object_name!r}; mask points were empty")
        if quat is None:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._last_object_pose_estimate = {
            **pose,
            "object_name": object_name,
            "camera_name": bundle["camera_name"],
            "external": bundle["external"],
            "depth_key": bundle["depth_key"],
            "oracle_mask": oracle_detail,
        }
        if return_bbox_extent:
            return self._to_numpy(position), self._to_numpy(quat), None if extent is None else self._to_numpy(extent)
        return self._to_numpy(position), self._to_numpy(quat)

    def save_current_observation(self, name) -> None:
        """R1Pro-compatible observation save hook.

        X2 smoke scripts handle artifact writing directly, so this function
        records the request and keeps the generated-code API compatible.
        """
        self._last_saved_observation_name = str(name)

    def get_current_joint_positions(self) -> np.ndarray:
        """Get current X2 joint positions."""
        return self._to_numpy(self._env.get_joint_positions())

    def get_current_eef_pose(self, arm: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose.

        Args:
            arm: Arm index. `0` is left, `1` is right.

        Returns:
            A tuple `(position, quaternion_xyzw)` in world frame.
        """
        pos, quat = self._env.get_robot_eef_pose(arm=arm)
        return self._to_numpy(pos), self._to_numpy(quat)

    def get_robot_relative_eef_pose(self, arm: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose relative to the robot frame.

        Args:
            arm: Arm index. `0` is left, `1` is right.

        Returns:
            A tuple `(position, quaternion_xyzw)` in robot frame.
        """
        pos, quat = self._env.get_robot_relative_eef_pose(arm=arm)
        return self._to_numpy(pos), self._to_numpy(quat)

    def move_hand(
        self,
        target_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        pos_thresh: float = 0.005,
        ori_thresh: float = 0.1,
        stop_if_stuck: bool = True,
        stuck_patience_steps: int = 30,
        stuck_pos_thresh: float = 0.0003,
        stuck_ori_thresh: float = 0.01,
        max_steps: int = 1000,
    ) -> bool:
        """Move an X2 hand to a target world pose with OmniGibson IK.

        Args:
            target_pose: Tuple `(position, quaternion_xyzw)`.
            arm: Arm index. `0` is left, `1` is right.
            pos_thresh: Position tolerance in meters for declaring success.
            ori_thresh: Orientation tolerance in radians for declaring success.
            stop_if_stuck: Whether to fail when the EEF stops making progress.
            stuck_patience_steps: Consecutive stalled steps before stuck failure.
            stuck_pos_thresh: Per-step position progress threshold in meters.
            stuck_ori_thresh: Per-step orientation progress threshold in radians.
            max_steps: Maximum IK control steps.

        Returns:
            Whether the low-level move completed without an exception.
        """
        pos, quat = target_pose
        return bool(
            self._env._move_hand(
                (np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)),
                arm=arm,
                pos_thresh=pos_thresh,
                ori_thresh=ori_thresh,
                stop_if_stuck=stop_if_stuck,
                stuck_patience_steps=stuck_patience_steps,
                stuck_pos_thresh=stuck_pos_thresh,
                stuck_ori_thresh=stuck_ori_thresh,
                max_steps=max_steps,
            )
        )

    def settle_robot(self, steps: int = 12) -> int:
        """Hold the robot still for a fixed number of simulator steps."""
        return int(self._env.settle_robot_steps(steps=steps))

    def open_gripper(self, arm: int = 1) -> None:
        """Open the Robotiq gripper.

        Args:
            arm: Arm index. `0` is left, `1` is right.
        """
        self._env._open_close_gripper(arm=arm, open=True)

    def close_gripper(self, arm: int = 1) -> None:
        """Close the Robotiq gripper.

        Args:
            arm: Arm index. `0` is left, `1` is right.
        """
        self._env._open_close_gripper(arm=arm, open=False)

    def get_gripper_state(self, arm: int = 1) -> dict[str, Any]:
        """Get diagnostic gripper joint / finger-span state for the selected arm."""
        return self._env.get_gripper_state(arm=arm)

    def get_last_motion_debug(self) -> dict[str, Any]:
        """Return diagnostics from the most recent X2 motion primitive call."""
        return {
            "last_pyroki_ik_debug": getattr(self._env, "_last_pyroki_ik_debug", None),
            "last_pyroki_trajopt_debug": getattr(self._env, "_last_pyroki_trajopt_debug", None),
            "last_joint_position_move_debug": getattr(self._env, "_last_joint_position_move_debug", None),
            "last_joint_trajectory_move_debug": getattr(self._env, "_last_joint_trajectory_move_debug", None),
            "last_joint_ik_move_debug": getattr(self._env, "_last_joint_ik_move_debug", None),
        }

    def get_control_timing(self) -> dict[str, Any]:
        """Return configured action / physics rates for timing diagnostics."""
        env_cfg = getattr(self._env, "controller_cfg", {}).get("env", {})
        action_frequency = float(env_cfg.get("action_frequency", 0.0) or 0.0)
        physics_frequency = float(env_cfg.get("physics_frequency", 0.0) or 0.0)
        return {
            "action_frequency_hz": action_frequency,
            "physics_frequency_hz": physics_frequency,
            "seconds_per_action_step": (1.0 / action_frequency) if action_frequency > 0.0 else None,
            "physics_steps_per_action": (physics_frequency / action_frequency) if action_frequency > 0.0 else None,
        }

    def get_tcp_offset_eef(self, arm: int = 1) -> np.ndarray:
        """Return the current TCP/finger-center offset in the EEF frame.

        X2's ``move_hand`` consumes an EEF pose, while visual grasp reasoning
        usually reasons about the gripper TCP / finger center.  This offset
        bridges those two definitions.
        """
        gripper_state = self.get_gripper_state(arm=arm)
        finger_center = gripper_state.get("finger_center_eef")
        if finger_center is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(finger_center, dtype=np.float64).reshape(3)

    def get_current_tcp_pose(self, arm: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return the current gripper TCP / finger-center pose in world frame."""
        eef_pos, eef_quat = self.get_current_eef_pose(arm=arm)
        offset = self.get_tcp_offset_eef(arm=arm)
        tcp_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3) + x2_vision.quat_xyzw_to_matrix(eef_quat) @ offset
        return tcp_pos, np.asarray(eef_quat, dtype=np.float64).reshape(4)

    def tcp_pose_to_eef_pose(
        self,
        tcp_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        tcp_offset_eef: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a desired TCP pose into the EEF pose required by ``move_hand``.

        Args:
            tcp_pose: ``(tcp_position_world, tcp_quaternion_xyzw)``.
            arm: Arm index used to read the current gripper TCP offset.
            tcp_offset_eef: Optional explicit TCP offset in the EEF frame.

        Returns:
            ``(eef_position_world, eef_quaternion_xyzw)`` suitable for
            ``move_hand``.
        """
        tcp_pos, tcp_quat = tcp_pose
        tcp_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(3)
        tcp_quat = np.asarray(tcp_quat, dtype=np.float64).reshape(4)
        offset = self.get_tcp_offset_eef(arm=arm) if tcp_offset_eef is None else np.asarray(tcp_offset_eef, dtype=np.float64).reshape(3)
        eef_pos = tcp_pos - x2_vision.quat_xyzw_to_matrix(tcp_quat) @ offset
        return eef_pos, tcp_quat

    def move_tcp(
        self,
        target_tcp_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        pos_thresh: float = 0.005,
        ori_thresh: float = 0.1,
        stop_if_stuck: bool = True,
        stuck_patience_steps: int = 30,
        stuck_pos_thresh: float = 0.0003,
        stuck_ori_thresh: float = 0.01,
        max_steps: int = 1000,
    ) -> bool:
        """Move the X2 TCP / finger center to a target world pose.

        This is the visual-action friendly wrapper: visual grasp logic usually
        produces ``(tcp_position_world, tcp_quaternion_xyzw)``.  The low-level
        X2 controller still consumes an EEF world pose, so this method converts
        TCP -> EEF and delegates to ``move_hand``.
        """
        eef_pose = self.tcp_pose_to_eef_pose(target_tcp_pose, arm=arm)
        return self.move_hand(
            eef_pose,
            arm=arm,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            stop_if_stuck=stop_if_stuck,
            stuck_patience_steps=stuck_patience_steps,
            stuck_pos_thresh=stuck_pos_thresh,
            stuck_ori_thresh=stuck_ori_thresh,
            max_steps=max_steps,
        )

    def move_hand_joint_ik(
        self,
        target_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        pos_thresh: float = 0.01,
        ori_thresh: float = 0.18,
        max_joint_step: float = 0.035,
        max_steps: int = 180,
        settle_steps: int = 12,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        """Move an EEF world pose via one-shot PyRoKi IK and joint-space interpolation."""
        pos, quat = target_pose
        return bool(
            self._env._move_hand_joint_ik(
                (np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)),
                arm=arm,
                pos_thresh=pos_thresh,
                ori_thresh=ori_thresh,
                max_joint_step=max_joint_step,
                max_steps=max_steps,
                settle_steps=settle_steps,
                hold_steps_per_waypoint=hold_steps_per_waypoint,
            )
        )

    def move_tcp_joint_ik(
        self,
        target_tcp_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        pos_thresh: float = 0.012,
        ori_thresh: float = 0.2,
        max_joint_step: float = 0.035,
        max_steps: int = 180,
        settle_steps: int = 12,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        """Move a TCP/finger-center world pose through the joint-space IK backend."""
        eef_pose = self.tcp_pose_to_eef_pose(target_tcp_pose, arm=arm)
        return self.move_hand_joint_ik(
            eef_pose,
            arm=arm,
            pos_thresh=pos_thresh,
            ori_thresh=ori_thresh,
            max_joint_step=max_joint_step,
            max_steps=max_steps,
            settle_steps=settle_steps,
            hold_steps_per_waypoint=hold_steps_per_waypoint,
        )

    def plan_tcp_pyroki_trajopt(
        self,
        target_tcp_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        obstacles_world: list[dict[str, Any]] | None = None,
        timesteps: int = 16,
        dt: float = 0.08,
    ) -> dict[str, Any]:
        """Plan a PyRoKi collision-aware joint trajectory to a TCP world pose."""
        eef_pose = self.tcp_pose_to_eef_pose(target_tcp_pose, arm=arm)
        trajectory, debug = self._env._solve_pyroki_eef_trajopt(
            eef_pose,
            arm=arm,
            obstacles_world=obstacles_world,
            timesteps=timesteps,
            dt=dt,
        )
        return {
            "ok": bool(trajectory),
            "joint_trajectory": [np.asarray(q, dtype=np.float64).copy() for q in trajectory],
            "debug": debug,
        }

    def move_tcp_pyroki_trajopt(
        self,
        target_tcp_pose: tuple[np.ndarray, np.ndarray],
        arm: int = 1,
        obstacles_world: list[dict[str, Any]] | None = None,
        timesteps: int = 16,
        dt: float = 0.08,
        max_joint_step: float = 0.02,
        max_steps_per_waypoint: int = 80,
        settle_steps: int = 12,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        """Plan and execute a PyRoKi collision-aware joint trajectory to a TCP pose."""
        plan = self.plan_tcp_pyroki_trajopt(
            target_tcp_pose,
            arm=arm,
            obstacles_world=obstacles_world,
            timesteps=timesteps,
            dt=dt,
        )
        if not plan["ok"]:
            return False
        return bool(
            self._env._move_through_joint_trajectory(
                plan["joint_trajectory"],
                arm=arm,
                max_joint_step=max_joint_step,
                max_steps_per_waypoint=max_steps_per_waypoint,
                settle_steps=settle_steps,
                hold_steps_per_waypoint=hold_steps_per_waypoint,
            )
        )

    def plan_graspnet_from_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        intrinsic_matrix: np.ndarray,
        camera_position: np.ndarray,
        camera_quat_xyzw: np.ndarray,
        *,
        orientation_quat_xyzw: np.ndarray | None = None,
        max_candidates: int = 8,
        pregrasp_distance: float = 0.08,
        grasp_offset_m: float = 0.05,
        min_mask_pixels: int = 8,
        expected_depth: float | None = None,
        depth_window: float | None = None,
        segmap_id: int = 1,
        z_range: list[float] | None = None,
        workspace_bounds: dict[str, tuple[float, float]] | None = None,
        include_raw_orientation: bool = True,
        local_regions: bool = True,
        filter_grasps: bool = True,
        skip_border_objects: bool = False,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> dict[str, Any]:
        """Plan X2 grasp candidates from a segmentation mask with Contact-GraspNet.

        This mirrors R1Pro's model boundary: the segmentation model produces
        an integer ``segmap`` and ``segmap_id``; Contact-GraspNet predicts
        camera-frame grasp matrices; X2 then converts them to world-frame
        executable pose candidates.
        """
        depth_clean = self._clean_depth_for_graspnet(depth)
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != depth_clean.shape:
            raise ValueError(f"mask shape {mask_bool.shape} does not match depth shape {depth_clean.shape}")

        valid_mask = mask_bool & (depth_clean > 0.0)
        depth_filter: dict[str, Any] = {
            "expected_depth": None if expected_depth is None else float(expected_depth),
            "depth_window": None if depth_window is None else float(depth_window),
            "used": False,
            "valid_mask_pixels_before_filter": int(valid_mask.sum()),
            "valid_mask_pixels_after_filter": int(valid_mask.sum()),
        }
        if expected_depth is not None and depth_window is not None:
            depth_keep = np.abs(depth_clean - float(expected_depth)) <= float(depth_window)
            filtered_mask = valid_mask & depth_keep
            depth_filter["valid_mask_pixels_after_filter"] = int(filtered_mask.sum())
            if int(filtered_mask.sum()) >= int(min_mask_pixels):
                valid_mask = filtered_mask
                depth_filter["used"] = True
        segmap = np.zeros(depth_clean.shape, dtype=np.int32)
        segmap[valid_mask] = int(segmap_id)
        result: dict[str, Any] = {
            "ok": False,
            "source": "contact_graspnet",
            "segmap_id": int(segmap_id),
            "mask_pixels": int(mask_bool.sum()),
            "valid_mask_pixels": int(valid_mask.sum()),
            "depth_filter": depth_filter,
            "depth_shape": list(depth_clean.shape),
            "raw_grasp_count": 0,
            "filtered_grasp_count": 0,
            "candidate_count": 0,
            "planner_kwargs": {
                "local_regions": bool(local_regions),
                "filter_grasps": bool(filter_grasps),
                "skip_border_objects": bool(skip_border_objects),
                "forward_passes": int(forward_passes),
                "max_retries": int(max_retries),
            },
            "candidates": [],
            "error": None,
        }
        if int(valid_mask.sum()) < int(min_mask_pixels):
            result["error"] = (
                f"mask has only {int(valid_mask.sum())} valid depth pixels; "
                f"minimum is {int(min_mask_pixels)}"
            )
            return result

        plan_fn = self._ensure_graspnet()
        if z_range is None:
            target_depths = depth_clean[valid_mask]
            if target_depths.size:
                lo, hi = np.percentile(target_depths, [1.0, 99.0])
                z_range = [max(0.05, float(lo) - 0.25), float(hi) + 0.25]
            else:
                z_range = [0.2, 2.0]
        result["z_range"] = [float(z_range[0]), float(z_range[1])]

        try:
            grasp_samples, grasp_scores, grasp_contact_pts = plan_fn(
                depth_clean,
                np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3),
                segmap,
                int(segmap_id),
                local_regions=bool(local_regions),
                filter_grasps=bool(filter_grasps),
                skip_border_objects=bool(skip_border_objects),
                z_range=z_range,
                forward_passes=int(forward_passes),
                max_retries=int(max_retries),
            )
        except Exception as exc:
            result["error"] = str(exc)
            return result

        grasps = np.asarray(grasp_samples, dtype=np.float64)
        if grasps.size == 0:
            result["error"] = "Contact-GraspNet returned no grasps"
            return result
        grasps = grasps.reshape(-1, 4, 4)
        scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
        if scores.size != len(grasps):
            scores = np.zeros(len(grasps), dtype=np.float64)
        contact_pts = np.asarray(grasp_contact_pts, dtype=np.float64)
        if contact_pts.size == 0:
            contact_pts = np.empty((0, 3), dtype=np.float64)
        else:
            contact_pts = contact_pts.reshape(-1, 3)

        result["raw_grasp_count"] = int(len(grasps))
        T_world_cam_gl = x2_vision.pose_to_matrix(camera_position, camera_quat_xyzw)
        T_grasp_offset = self._translation_matrix(np.array([0.0, 0.0, float(grasp_offset_m)], dtype=np.float64))
        order = np.argsort(-scores)[: max(1, int(max_candidates))]
        base_quat = None if orientation_quat_xyzw is None else np.asarray(orientation_quat_xyzw, dtype=np.float64).reshape(4)
        candidate_items: list[dict[str, Any]] = []
        filtered_count = 0

        for rank, grasp_idx in enumerate(order):
            T_cam_cv = grasps[int(grasp_idx)] @ T_grasp_offset
            T_cam_gl = self._convert_T_cam_cv_to_cam_gl(T_cam_cv)
            T_world_grasp = T_world_cam_gl @ T_cam_gl
            grasp_pos = T_world_grasp[:3, 3].astype(np.float64)
            raw_quat = x2_vision.matrix_to_quat_xyzw(T_world_grasp[:3, :3])
            if not self._within_workspace(grasp_pos, workspace_bounds):
                filtered_count += 1
                continue

            raw_approach_dir = x2_vision.quat_xyzw_to_matrix(raw_quat) @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
            approach_norm = float(np.linalg.norm(raw_approach_dir))
            if not np.isfinite(approach_norm) or approach_norm < 1e-8:
                raw_approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            else:
                raw_approach_dir = raw_approach_dir / approach_norm

            contact_point_world = None
            if len(contact_pts) > int(grasp_idx):
                contact_point_world = self._point_cam_cv_to_world(contact_pts[int(grasp_idx)], T_world_cam_gl)

            variants: list[tuple[str, np.ndarray, np.ndarray]] = []
            if include_raw_orientation:
                variants.append(("raw_graspnet_quat", raw_quat, raw_approach_dir))
            if base_quat is not None:
                variants.append(("x2_current_quat", base_quat, raw_approach_dir))

            for variant_name, quat, approach_dir in variants:
                pregrasp_pos = grasp_pos - approach_dir * float(pregrasp_distance)
                candidate_items.append(
                    {
                        "name": f"graspnet_{variant_name}_{rank}",
                        "source": "contact_graspnet",
                        "variant": variant_name,
                        "rank": int(rank),
                        "grasp_index": int(grasp_idx),
                        "score": float(scores[int(grasp_idx)]),
                        "grasp_pose": (grasp_pos.copy(), quat.copy()),
                        "pregrasp_pose": (pregrasp_pos.astype(np.float64), quat.copy()),
                        "raw_graspnet_pose": (grasp_pos.copy(), raw_quat.copy()),
                        "approach_dir_world": approach_dir.copy(),
                        "contact_point_world": None if contact_point_world is None else contact_point_world.copy(),
                    }
                )

        result["filtered_grasp_count"] = int(filtered_count)
        result["candidate_count"] = int(len(candidate_items))
        result["candidates"] = candidate_items
        result["ok"] = len(candidate_items) > 0
        if not result["ok"] and result["error"] is None:
            result["error"] = "all Contact-GraspNet grasps were filtered out"
        return result

    def sample_grasp_pose_graspnet(
        self,
        object_name: str,
        *,
        mask: np.ndarray | None = None,
        camera_name: str | None = None,
        arm: int = 1,
        external: bool | None = None,
        depth_key: str = "depth_linear",
        orientation_quat_xyzw: np.ndarray | None = None,
        include_simple_fallback: bool = True,
        **plan_kwargs,
    ) -> dict[str, Any]:
        """Sample grasp candidates using OWL-ViT/SAM2-oracle mask plus GraspNet."""
        if external is None:
            external = bool(self.get_external_camera_names())
        bundle = self._camera_bundle(camera_name=camera_name, arm=arm, external=bool(external), depth_key=depth_key)
        mask_source: dict[str, Any]
        if mask is None:
            if self.use_vision_models:
                mask_result = self.get_object_mask_sam2(
                    object_name,
                    camera_name=bundle["camera_name"],
                    arm=arm,
                    external=bool(external),
                    max_masks=1,
                )
                mask = np.asarray(mask_result["mask"], dtype=bool)
                mask_source = {
                    "source": "owlvit_sam2",
                    "detection": mask_result.get("detection"),
                    "mask_pixels": int(mask.sum()),
                }
            else:
                mask, oracle_detail, _obj = self._oracle_mask_for_object(object_name, bundle)
                mask_source = {
                    "source": "oracle_projected_aabb",
                    "detail": oracle_detail,
                    "mask_pixels": int(np.asarray(mask, dtype=bool).sum()),
                }
        else:
            mask = np.asarray(mask, dtype=bool)
            mask_source = {"source": "provided_mask", "mask_pixels": int(mask.sum())}

        if orientation_quat_xyzw is None:
            orientation_quat_xyzw = self.get_current_eef_pose(arm=arm)[1]

        plan = self.plan_graspnet_from_mask(
            mask,
            bundle["depth"],
            bundle["intrinsic_matrix"],
            bundle["camera_position"],
            bundle["camera_quat_xyzw"],
            orientation_quat_xyzw=orientation_quat_xyzw,
            **plan_kwargs,
        )
        plan.update(
            {
                "object_name": object_name,
                "camera_name": bundle["camera_name"],
                "external": bundle["external"],
                "depth_key": bundle["depth_key"],
                "mask_source": mask_source,
            }
        )
        if not plan["candidates"] and include_simple_fallback:
            try:
                pregrasp_pose, grasp_pose = self.sample_grasp_pose(object_name, arm=arm)
                plan["candidates"] = [
                    {
                        "name": "x2_simple_fallback",
                        "source": "x2_heuristic_sampler",
                        "variant": "simple_fallback",
                        "rank": 0,
                        "score": 0.0,
                        "pregrasp_pose": pregrasp_pose,
                        "grasp_pose": grasp_pose,
                    }
                ]
                plan["candidate_count"] = 1
                plan["fallback_used"] = "x2_heuristic_sampler"
            except Exception as exc:
                plan["fallback_error"] = str(exc)
        return plan

    def plan_x2_grasp_execution(
        self,
        grasp_plan: dict[str, Any],
        object_position: np.ndarray,
        *,
        bbox_extent: np.ndarray | None = None,
        arm: int = 1,
        orientation_quat_xyzw: np.ndarray | None = None,
        contact_blends: tuple[float, ...] = (0.0, 0.25, 0.5),
        z_biases: tuple[float, ...] = (0.0, 0.01, 0.02),
        pregrasp_lifts: tuple[float, ...] = (0.06, 0.08),
        lift_distance: float = 0.06,
        max_candidates: int = 12,
        workspace_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Convert GraspNet candidates into X2-friendly TCP execution poses.

        Contact-GraspNet proposes where contact is plausible; this planner
        rewrites those proposals into short, near-vertical TCP motions that are
        better matched to X2's fixed base and Robotiq gripper geometry.  It is
        intentionally pure planning: no robot motion is executed here.
        """
        object_pos = np.asarray(object_position, dtype=np.float64).reshape(3)
        if orientation_quat_xyzw is None:
            orientation_quat_xyzw = self.get_current_eef_pose(arm=arm)[1]
            orientation_source = "current_eef_quat"
        else:
            orientation_source = "provided_quat"
        tcp_quat = np.asarray(orientation_quat_xyzw, dtype=np.float64).reshape(4)
        extent = None if bbox_extent is None else np.asarray(bbox_extent, dtype=np.float64).reshape(3)

        raw_candidates = list(grasp_plan.get("candidates", []) or [])
        raw_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        result: dict[str, Any] = {
            "ok": False,
            "source": "x2_grasp_execution_planner",
            "strategy": "x2_topdown_tcp_from_graspnet_contact",
            "input_source": grasp_plan.get("source"),
            "input_candidate_count": int(len(raw_candidates)),
            "candidate_count": 0,
            "object_position": object_pos.copy(),
            "bbox_extent": None if extent is None else extent.copy(),
            "orientation_source": orientation_source,
            "orientation_quat_xyzw": tcp_quat.copy(),
            "planner_kwargs": {
                "contact_blends": [float(v) for v in contact_blends],
                "z_biases": [float(v) for v in z_biases],
                "pregrasp_lifts": [float(v) for v in pregrasp_lifts],
                "lift_distance": float(lift_distance),
                "max_candidates": int(max_candidates),
                "workspace_bounds": workspace_bounds,
            },
            "candidates": [],
            "filtered": [],
            "error": None,
        }
        if not raw_candidates:
            result["error"] = "grasp_plan has no candidates"
            return result

        planned: list[dict[str, Any]] = []
        seen_keys: set[tuple[float, ...]] = set()
        for parent_rank, candidate in enumerate(raw_candidates):
            contact = candidate.get("contact_point_world")
            contact_arr = None if contact is None else np.asarray(contact, dtype=np.float64).reshape(3)
            contact_valid = contact_arr is not None and bool(np.all(np.isfinite(contact_arr)))
            blends = tuple(float(v) for v in contact_blends if contact_valid or float(v) == 0.0)
            for blend in blends:
                for z_bias in z_biases:
                    for pregrasp_lift in pregrasp_lifts:
                        tcp_target = object_pos.copy()
                        if contact_valid:
                            tcp_target[:2] = (1.0 - blend) * object_pos[:2] + blend * contact_arr[:2]
                        tcp_target[2] = object_pos[2] + float(z_bias)
                        pregrasp_tcp = tcp_target + np.array([0.0, 0.0, float(pregrasp_lift)], dtype=np.float64)
                        lift_tcp = tcp_target + np.array([0.0, 0.0, float(lift_distance)], dtype=np.float64)

                        key = tuple(np.round(np.concatenate([tcp_target, [blend, z_bias, pregrasp_lift]]), 5))
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        grasp_eef = self.tcp_pose_to_eef_pose((tcp_target, tcp_quat), arm=arm)
                        pregrasp_eef = self.tcp_pose_to_eef_pose((pregrasp_tcp, tcp_quat), arm=arm)
                        lift_eef = self.tcp_pose_to_eef_pose((lift_tcp, tcp_quat), arm=arm)
                        if not self._within_workspace(tcp_target, workspace_bounds):
                            result["filtered"].append(
                                {
                                    "parent_name": candidate.get("name"),
                                    "reason": "tcp_target_outside_workspace",
                                    "tcp_target": tcp_target.copy(),
                                }
                            )
                            continue

                        planned.append(
                            {
                                "name": f"x2_topdown_tcp_{len(planned)}",
                                "source": "x2_grasp_execution_planner",
                                "strategy": "x2_topdown_tcp_from_graspnet_contact",
                                "rank": int(len(planned)),
                                "parent_rank": int(parent_rank),
                                "parent_name": candidate.get("name"),
                                "score": None if candidate.get("score") is None else float(candidate.get("score", 0.0)),
                                "graspnet_candidate": candidate,
                                "contact_point_world": None if not contact_valid else contact_arr.copy(),
                                "object_position": object_pos.copy(),
                                "bbox_extent": None if extent is None else extent.copy(),
                                "contact_blend": float(blend),
                                "z_bias": float(z_bias),
                                "pregrasp_lift": float(pregrasp_lift),
                                "lift_distance": float(lift_distance),
                                "descent_dir_world": np.array([0.0, 0.0, -1.0], dtype=np.float64),
                                "grasp_tcp_pose": (tcp_target.copy(), tcp_quat.copy()),
                                "pregrasp_tcp_pose": (pregrasp_tcp.copy(), tcp_quat.copy()),
                                "lift_tcp_pose": (lift_tcp.copy(), tcp_quat.copy()),
                                "grasp_pose": grasp_eef,
                                "pregrasp_pose": pregrasp_eef,
                                "lift_pose": lift_eef,
                            }
                        )
                        if len(planned) >= int(max_candidates):
                            break
                    if len(planned) >= int(max_candidates):
                        break
                if len(planned) >= int(max_candidates):
                    break
            if len(planned) >= int(max_candidates):
                break

        result["candidates"] = planned
        result["candidate_count"] = int(len(planned))
        result["ok"] = bool(planned)
        if not planned and result["error"] is None:
            result["error"] = "all X2 execution candidates were filtered out"
        return result

    @staticmethod
    def _sphere_intersects_box(point: np.ndarray, center: np.ndarray, size: float, margin: float, radius: float) -> bool:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        center = np.asarray(center, dtype=np.float64).reshape(3)
        half = 0.5 * float(size) + float(margin)
        closest = np.minimum(np.maximum(point, center - half), center + half)
        return bool(np.linalg.norm(point - closest) <= float(radius))

    def _gripper_proxy_points_eef(self, arm: int = 1) -> list[dict[str, Any]]:
        state = self.get_gripper_state(arm=arm)
        points: list[dict[str, Any]] = []
        center = np.asarray(state.get("finger_center_eef", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        points.append({"name": "tcp_finger_center", "point_eef": center, "solid": False})
        for idx, point in enumerate(state.get("finger_positions_eef", []) or []):
            finger = np.asarray(point, dtype=np.float64).reshape(3)
            points.append({"name": f"finger_tip_{idx}", "point_eef": finger, "solid": True})
            points.append({"name": f"finger_gap_mid_{idx}", "point_eef": 0.5 * (finger + center), "solid": False})
        return points

    def _gripper_proxy_collisions_for_tcp(
        self,
        tcp_pose: tuple[np.ndarray, np.ndarray],
        proxy_points_eef: list[dict[str, Any]],
        object_center: np.ndarray,
        *,
        object_size: float,
        margin: float,
        proxy_radius: float,
        arm: int = 1,
        solid_only: bool = True,
    ) -> list[dict[str, Any]]:
        tcp_pos, tcp_quat = tcp_pose
        eef_pos, eef_quat = self.tcp_pose_to_eef_pose((tcp_pos, tcp_quat), arm=arm)
        eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
        rot = x2_vision.quat_xyzw_to_matrix(np.asarray(eef_quat, dtype=np.float64).reshape(4))
        collisions: list[dict[str, Any]] = []
        for item in proxy_points_eef:
            if solid_only and not bool(item.get("solid", True)):
                continue
            point_world = eef_pos + rot @ np.asarray(item["point_eef"], dtype=np.float64).reshape(3)
            if self._sphere_intersects_box(point_world, object_center, object_size, margin, proxy_radius):
                collisions.append(
                    {
                        "name": item["name"],
                        "point_world": point_world.copy(),
                        "proxy_radius": float(proxy_radius),
                        "solid": bool(item.get("solid", True)),
                    }
                )
        return collisions

    def plan_x2_guarded_grasp_approach(
        self,
        grasp_tcp_pose: tuple[np.ndarray, np.ndarray],
        object_center: np.ndarray,
        *,
        object_size: float = 0.04,
        arm: int = 1,
        approach_distance: float = 0.09,
        precontact_distance: float = 0.03,
        num_waypoints: int = 6,
        guard_margin: float = 0.008,
        gripper_proxy_radius: float = 0.012,
        gripper_proxy_margin: float = 0.012,
        stop_at_first_proxy_collision: bool = True,
        final_insertion_waypoints: int = 0,
    ) -> dict[str, Any]:
        """Plan a lightweight guarded Cartesian TCP approach without cuRobo.

        This is a task-specific mini planner. It does not solve full robot-world
        collision planning. Instead, it creates TCP waypoints along the gripper
        approach axis and filters pre-grasp waypoints where a simple open-gripper
        proxy would intersect an inflated object box.
        """
        grasp_pos = np.asarray(grasp_tcp_pose[0], dtype=np.float64).reshape(3)
        grasp_quat = np.asarray(grasp_tcp_pose[1], dtype=np.float64).reshape(4)
        object_center = np.asarray(object_center, dtype=np.float64).reshape(3)
        proxy_points = self._gripper_proxy_points_eef(arm=arm)
        tcp_offset = np.asarray(self.get_tcp_offset_eef(arm=arm), dtype=np.float64).reshape(3)
        approach_axis = x2_vision.quat_xyzw_to_matrix(grasp_quat) @ tcp_offset
        approach_axis = approach_axis / max(float(np.linalg.norm(approach_axis)), 1e-12)
        pregrasp_pos = grasp_pos - approach_axis * float(approach_distance)

        waypoints: list[dict[str, Any]] = []
        guard_violations: list[dict[str, Any]] = []
        proxy_guard_violations: list[dict[str, Any]] = []
        distances = np.linspace(float(approach_distance), float(precontact_distance), max(2, int(num_waypoints)))
        for idx, distance in enumerate(distances):
            waypoint_pos = grasp_pos - approach_axis * float(distance)
            tcp_box_inside = bool(np.all(np.abs(waypoint_pos - object_center) <= (0.5 * float(object_size) + float(guard_margin))))
            if tcp_box_inside:
                guard_violations.append(
                    {
                        "index": int(idx),
                        "distance_m": float(distance),
                        "waypoint_position": waypoint_pos.copy(),
                        "reason": "tcp_waypoint_inside_inflated_object_box_before_final_grasp",
                    }
                )
                continue
            proxy_collisions = self._gripper_proxy_collisions_for_tcp(
                (waypoint_pos, grasp_quat),
                proxy_points,
                object_center,
                object_size=object_size,
                margin=gripper_proxy_margin,
                proxy_radius=gripper_proxy_radius,
                arm=arm,
            )
            if proxy_collisions:
                proxy_guard_violations.append(
                    {
                        "index": int(idx),
                        "distance_m": float(distance),
                        "waypoint_position": waypoint_pos.copy(),
                        "reason": "gripper_proxy_intersects_inflated_object_box_before_final_grasp",
                        "collisions": proxy_collisions,
                    }
                )
                if stop_at_first_proxy_collision:
                    break
                continue
            waypoints.append(
                {
                    "name": f"guarded_approach_{len(waypoints)}",
                    "distance_m": float(distance),
                    "tcp_pose": (waypoint_pos.copy(), grasp_quat.copy()),
                    "eef_pose": self.tcp_pose_to_eef_pose((waypoint_pos, grasp_quat), arm=arm),
                }
            )

        insertion_waypoints: list[dict[str, Any]] = []
        num_insert = max(0, int(final_insertion_waypoints))
        if num_insert > 0:
            start_distance = float(waypoints[-1]["distance_m"]) if waypoints else float(approach_distance)
            for idx, distance in enumerate(np.linspace(start_distance, 0.0, num_insert + 1)[1:]):
                waypoint_pos = grasp_pos - approach_axis * float(distance)
                insertion_waypoints.append(
                    {
                        "name": f"guarded_final_insertion_{idx}",
                        "distance_m": float(distance),
                        "tcp_pose": (waypoint_pos.copy(), grasp_quat.copy()),
                        "eef_pose": self.tcp_pose_to_eef_pose((waypoint_pos, grasp_quat), arm=arm),
                    }
                )

        return {
            "ok": True,
            "source": "x2_guarded_cartesian_approach_planner",
            "strategy": "tcp_axis_cartesian_waypoints_with_gripper_proxy_guard",
            "grasp_tcp_pose": (grasp_pos.copy(), grasp_quat.copy()),
            "pregrasp_tcp_pose": (pregrasp_pos.copy(), grasp_quat.copy()),
            "pregrasp_pose": self.tcp_pose_to_eef_pose((pregrasp_pos, grasp_quat), arm=arm),
            "approach_axis_world": approach_axis.copy(),
            "approach_waypoints": waypoints,
            "final_insertion_waypoints": insertion_waypoints,
            "guard_violations": guard_violations,
            "proxy_guard_violations": proxy_guard_violations,
            "gripper_proxy_points_eef": proxy_points,
            "params": {
                "object_center": object_center.copy(),
                "object_size": float(object_size),
                "approach_distance": float(approach_distance),
                "precontact_distance": float(precontact_distance),
                "num_waypoints": int(num_waypoints),
                "guard_margin": float(guard_margin),
                "gripper_proxy_radius": float(gripper_proxy_radius),
                "gripper_proxy_margin": float(gripper_proxy_margin),
                "stop_at_first_proxy_collision": bool(stop_at_first_proxy_collision),
                "final_insertion_waypoints": int(final_insertion_waypoints),
            },
        }

    def adapt_graspnet_raw_pose_to_x2_tcp(
        self,
        raw_graspnet_pose: tuple[np.ndarray, np.ndarray],
        *,
        raw_to_x2_tcp_pos: np.ndarray | None = None,
        raw_to_x2_tcp_quat_xyzw: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a raw Contact-GraspNet world pose into X2's executable TCP pose.

        The returned pose is ``T_world_tcp``.  It is the pose that
        ``move_tcp_joint_ik`` and ``plan_tcp_pyroki_trajopt`` consume after
        internally converting TCP to EEF.
        """
        adapter = (
            DEFAULT_GRASPNET_RAW_TO_X2_TCP_POS
            if raw_to_x2_tcp_pos is None
            else np.asarray(raw_to_x2_tcp_pos, dtype=np.float64).reshape(3),
            DEFAULT_GRASPNET_RAW_TO_X2_TCP_QUAT_XYZW
            if raw_to_x2_tcp_quat_xyzw is None
            else np.asarray(raw_to_x2_tcp_quat_xyzw, dtype=np.float64).reshape(4),
        )
        raw_pos, raw_quat = raw_graspnet_pose
        return self._pose_right_transform(
            (np.asarray(raw_pos, dtype=np.float64).reshape(3), np.asarray(raw_quat, dtype=np.float64).reshape(4)),
            adapter,
        )

    def select_adapted_graspnet_tcp_candidate(
        self,
        grasp_plan: dict[str, Any],
        object_position: np.ndarray,
        *,
        arm: int = 1,
        index: int = 0,
        reference_quat_xyzw: np.ndarray | None = None,
        max_xy_error: float = 0.08,
        max_z_error: float = 0.05,
        position_weight: float = 12.0,
        quat_weight: float = 1.0,
        axis_weight: float = 0.5,
        proxy_collision_penalty: float = 1.0,
        proxy_guard: bool = True,
        proxy_guard_object_size: float = 0.04,
        proxy_guard_approach_distance: float = 0.08,
        proxy_guard_first_insert_fraction: float = 0.9,
        proxy_guard_margin: float = 0.008,
        gripper_proxy_radius: float = 0.012,
        gripper_proxy_margin: float = 0.012,
        raw_to_x2_tcp_pos: np.ndarray | None = None,
        raw_to_x2_tcp_quat_xyzw: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Select one raw GraspNet candidate after adapting it to X2 TCP.

        This is the reusable form of the v11 demo selection rule: raw
        Contact-GraspNet candidates are first transformed to ``T_world_tcp``,
        then ranked by X2 reachability-oriented geometry and a lightweight
        open-gripper proxy guard.
        """
        raw_candidates = [
            candidate
            for candidate in list(grasp_plan.get("candidates", []) or [])
            if candidate.get("variant") == "raw_graspnet_quat" and candidate.get("raw_graspnet_pose") is not None
        ]
        if not raw_candidates:
            return {
                "ok": False,
                "source": "x2_adapted_graspnet_tcp_selector",
                "error": "grasp_plan has no raw_graspnet_quat candidates",
                "candidates": [],
            }

        object_pos = np.asarray(object_position, dtype=np.float64).reshape(3)
        reference_quat = (
            np.asarray(self.get_current_eef_pose(arm=arm)[1], dtype=np.float64).reshape(4)
            if reference_quat_xyzw is None
            else np.asarray(reference_quat_xyzw, dtype=np.float64).reshape(4)
        )
        tcp_offset = self.get_tcp_offset_eef(arm=arm)
        reference_axis = x2_vision.quat_xyzw_to_matrix(reference_quat) @ tcp_offset
        reference_axis = reference_axis / max(float(np.linalg.norm(reference_axis)), 1e-12)
        first_insert_distance = float(proxy_guard_approach_distance) * float(proxy_guard_first_insert_fraction)

        scored: list[tuple[float, int, dict[str, Any]]] = []
        records: list[dict[str, Any]] = []
        for idx, candidate in enumerate(raw_candidates):
            raw_pos, raw_quat = candidate["raw_graspnet_pose"]
            adapted_pose = self.adapt_graspnet_raw_pose_to_x2_tcp(
                (raw_pos, raw_quat),
                raw_to_x2_tcp_pos=raw_to_x2_tcp_pos,
                raw_to_x2_tcp_quat_xyzw=raw_to_x2_tcp_quat_xyzw,
            )
            adapted_pos, adapted_quat = adapted_pose
            xy_err = float(np.linalg.norm(adapted_pos[:2] - object_pos[:2]))
            z_err = float(abs(adapted_pos[2] - object_pos[2]))
            pos_err = float(np.linalg.norm(adapted_pos - object_pos))
            quat_err = self._quat_error_rad(adapted_quat, reference_quat)
            axis = x2_vision.quat_xyzw_to_matrix(adapted_quat) @ tcp_offset
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            axis_err = float(np.arccos(np.clip(float(np.dot(axis, reference_axis)), -1.0, 1.0)))

            guard_plan = None
            proxy_collision_count = 0
            if proxy_guard:
                guard_plan = self.plan_x2_guarded_grasp_approach(
                    adapted_pose,
                    object_pos,
                    object_size=float(proxy_guard_object_size),
                    arm=arm,
                    approach_distance=float(proxy_guard_approach_distance),
                    precontact_distance=first_insert_distance,
                    num_waypoints=2,
                    guard_margin=float(proxy_guard_margin),
                    gripper_proxy_radius=float(gripper_proxy_radius),
                    gripper_proxy_margin=float(gripper_proxy_margin),
                    stop_at_first_proxy_collision=False,
                    final_insertion_waypoints=0,
                )
                proxy_collision_count = len(guard_plan.get("proxy_guard_violations", []) or []) + len(
                    guard_plan.get("guard_violations", []) or []
                )

            outside_penalty = 0.0
            if xy_err > float(max_xy_error):
                outside_penalty += 10.0 * (xy_err - float(max_xy_error))
            if z_err > float(max_z_error):
                outside_penalty += 10.0 * (z_err - float(max_z_error))
            score = (
                float(position_weight) * pos_err
                + float(quat_weight) * quat_err
                + float(axis_weight) * axis_err
                + outside_penalty
                + float(proxy_collision_penalty) * float(proxy_collision_count)
            )
            record = {
                "raw_index": int(idx),
                "name": candidate.get("name"),
                "score": candidate.get("score"),
                "adapted_grasp_tcp_pose": adapted_pose,
                "adapted_position": adapted_pos.copy(),
                "adapted_quat_xyzw": adapted_quat.copy(),
                "xy_error_m": xy_err,
                "z_error_m": z_err,
                "position_error_m": pos_err,
                "quat_error_to_reference_rad": quat_err,
                "axis_error_to_reference_rad": axis_err,
                "proxy_collision_count": int(proxy_collision_count),
                "proxy_guard": guard_plan,
                "selection_score": float(score),
                "raw_candidate": candidate,
            }
            records.append(record)
            scored.append((float(score), idx, record))

        scored.sort(key=lambda item: item[0])
        selected_rank = min(max(0, int(index)), len(scored) - 1)
        selected_score, selected_raw_index, selected = scored[selected_rank]
        selected_pose = selected["adapted_grasp_tcp_pose"]
        return {
            "ok": True,
            "source": "x2_adapted_graspnet_tcp_selector",
            "strategy": "raw_graspnet_pose_right_multiplied_by_x2_tcp_adapter",
            "selected_rank": int(selected_rank),
            "selected_raw_index": int(selected_raw_index),
            "selected_score": float(selected_score),
            "selected_candidate": selected,
            "grasp_tcp_pose": selected_pose,
            "tcp_axis_world": (
                x2_vision.quat_xyzw_to_matrix(selected_pose[1]) @ tcp_offset
            )
            / max(float(np.linalg.norm(x2_vision.quat_xyzw_to_matrix(selected_pose[1]) @ tcp_offset)), 1e-12),
            "ranked_candidates": [item[2] for item in scored[:10]],
            "reference_quat_xyzw": reference_quat.copy(),
            "reference_axis_world": reference_axis.copy(),
            "adapter": {
                "position": (
                    DEFAULT_GRASPNET_RAW_TO_X2_TCP_POS
                    if raw_to_x2_tcp_pos is None
                    else np.asarray(raw_to_x2_tcp_pos, dtype=np.float64).reshape(3)
                ).copy(),
                "quat_xyzw": (
                    DEFAULT_GRASPNET_RAW_TO_X2_TCP_QUAT_XYZW
                    if raw_to_x2_tcp_quat_xyzw is None
                    else np.asarray(raw_to_x2_tcp_quat_xyzw, dtype=np.float64).reshape(4)
                ).copy(),
            },
        }

    def plan_visual_grasp_tcp_pose(
        self,
        object_name: str,
        *,
        prompts: list[str] | tuple[str, ...] | None = None,
        camera_name: str | None = None,
        arm: int = 1,
        external: bool = False,
        orientation_quat_xyzw: np.ndarray | None = None,
        object_pose_method: str = "aabb_center",
        graspnet_forward_passes: int = 4,
        graspnet_max_retries: int = 30,
        graspnet_max_candidates: int = 24,
        min_mask_pixels: int = 12,
        workspace_bounds: dict[str, tuple[float, float]] | None = None,
        precontact_distance: float = 0.08,
        insert_waypoints: int = 10,
        candidate_index: int = 0,
        proxy_guard_object_size: float = 0.04,
        **selection_kwargs,
    ) -> dict[str, Any]:
        """Run OWL-ViT + SAM2 + GraspNet and return an X2 ``T_world_tcp`` grasp plan.

        This method assumes the env and task already exist. It performs no
        scene setup and no robot motion, so it is suitable for CaP-X generated
        code that is injected into an existing X2 BEHAVIOR environment.

        Args:
            object_name: Name of the target object already present in the
                BEHAVIOR scene.
            prompts: Text prompts used by OWL-ViT. Use short visual phrases
                such as ``["red cube", "red block"]``.
            camera_name: Camera sensor to use. For the current X2 tabletop
                tasks, prefer ``get_chest_camera_name()``.
            arm: X2 arm index. Current tabletop tasks use right arm ``1``.
            orientation_quat_xyzw: Optional X2 TCP orientation prior in world
                frame. If omitted, the current EEF orientation is used.

        Returns:
            Dict with ``ok`` and, on success, ``grasp_tcp_pose``,
            ``precontact_tcp_pose``, and ``insertion_waypoints``. All target
            poses are ``T_world_tcp``. Do not execute raw GraspNet poses
            directly; this function adapts them into X2 TCP targets.
        """
        if not self.use_vision_models:
            raise RuntimeError("plan_visual_grasp_tcp_pose requires X2ControlApi(use_vision_models=True)")
        if not self.use_graspnet:
            raise RuntimeError("plan_visual_grasp_tcp_pose requires X2ControlApi(use_graspnet=True)")

        bundle = self._camera_bundle(camera_name=camera_name, arm=arm, external=bool(external), depth_key="depth_linear")
        rgb = self._rgb_u8(bundle["observation"]["rgb"])
        depth = bundle["depth"]
        prompt_list = [object_name] if prompts is None else list(prompts)
        detections: list[dict[str, Any]] = []
        for prompt in prompt_list:
            for detection in self.detect_object_owlvit(rgb, str(prompt)):
                det = dict(detection)
                det.setdefault("prompt", str(prompt))
                detections.append(det)
        detection = self._best_by_score(detections)
        if detection is None:
            return {"ok": False, "source": "x2_visual_grasp_tcp_planner", "error": f"No detections for {prompt_list!r}"}
        masks = self.segment_sam2(rgb, box=detection["box"], max_masks=1)
        mask_result = self._best_by_score(masks)
        if mask_result is None:
            return {"ok": False, "source": "x2_visual_grasp_tcp_planner", "error": "SAM2 returned no masks"}
        mask = np.asarray(mask_result["mask"], dtype=bool)
        expected_depth, depth_window = self._mask_depth_hint(mask, depth)

        object_pos, object_quat, bbox_extent = self.get_object_pose(
            object_name,
            return_bbox_extent=True,
            mask=mask,
            camera_name=bundle["camera_name"],
            arm=arm,
            external=bool(external),
            method=object_pose_method,
            expected_depth=expected_depth,
            depth_window=depth_window,
        )
        reference_quat = (
            np.asarray(self.get_current_eef_pose(arm=arm)[1], dtype=np.float64).reshape(4)
            if orientation_quat_xyzw is None
            else np.asarray(orientation_quat_xyzw, dtype=np.float64).reshape(4)
        )
        grasp_plan = self.sample_grasp_pose_graspnet(
            object_name,
            mask=mask,
            camera_name=bundle["camera_name"],
            arm=arm,
            external=bool(external),
            orientation_quat_xyzw=reference_quat,
            include_simple_fallback=False,
            max_candidates=int(graspnet_max_candidates),
            min_mask_pixels=int(min_mask_pixels),
            expected_depth=expected_depth,
            depth_window=depth_window,
            workspace_bounds=workspace_bounds,
            forward_passes=int(graspnet_forward_passes),
            max_retries=int(graspnet_max_retries),
        )
        selection = self.select_adapted_graspnet_tcp_candidate(
            grasp_plan,
            object_pos,
            arm=arm,
            index=int(candidate_index),
            reference_quat_xyzw=reference_quat,
            proxy_guard_object_size=float(proxy_guard_object_size),
            proxy_guard_approach_distance=float(precontact_distance),
            proxy_guard_first_insert_fraction=float(max(0, int(insert_waypoints) - 1)) / float(max(1, int(insert_waypoints))),
            **selection_kwargs,
        )
        if not selection.get("ok"):
            return {
                "ok": False,
                "source": "x2_visual_grasp_tcp_planner",
                "error": selection.get("error"),
                "visual": {"detections": detections, "detection": detection, "mask_pixels": int(mask.sum())},
                "pose_estimate": {"position_world": object_pos, "quat_xyzw_world": object_quat, "bbox_extent": bbox_extent},
                "graspnet": grasp_plan,
                "selection": selection,
            }

        grasp_tcp_pose = selection["grasp_tcp_pose"]
        tcp_axis_world = np.asarray(selection["tcp_axis_world"], dtype=np.float64).reshape(3)
        precontact_tcp_pose = (
            np.asarray(grasp_tcp_pose[0], dtype=np.float64).reshape(3) - tcp_axis_world * float(precontact_distance),
            np.asarray(grasp_tcp_pose[1], dtype=np.float64).reshape(4),
        )
        insertion_waypoints = []
        for idx, distance in enumerate(np.linspace(float(precontact_distance), 0.0, int(insert_waypoints) + 1)[1:]):
            insertion_waypoints.append(
                {
                    "name": "grasp" if idx == int(insert_waypoints) - 1 else f"insert_{idx:02d}",
                    "distance_m": float(distance),
                    "tcp_pose": (
                        np.asarray(grasp_tcp_pose[0], dtype=np.float64).reshape(3) - tcp_axis_world * float(distance),
                        np.asarray(grasp_tcp_pose[1], dtype=np.float64).reshape(4),
                    ),
                }
            )
        return {
            "ok": True,
            "source": "x2_visual_grasp_tcp_planner",
            "strategy": "owlvit_sam2_graspnet_raw_pose_adapted_to_x2_tcp",
            "camera": {
                "name": bundle["camera_name"],
                "external": bundle["external"],
                "position_world": bundle["camera_position"],
                "quat_xyzw_world": bundle["camera_quat_xyzw"],
                "intrinsic_matrix": bundle["intrinsic_matrix"],
            },
            "visual": {
                "prompts": prompt_list,
                "detections": detections,
                "detection": detection,
                "mask_result": mask_result,
                "mask": mask,
                "mask_pixels": int(mask.sum()),
                "expected_depth": expected_depth,
                "depth_window": depth_window,
            },
            "pose_estimate": {
                "meaning": "T_world_object estimated from SAM2 mask and RGB-D depth",
                "object_name": object_name,
                "position_world": object_pos,
                "quat_xyzw_world": object_quat,
                "bbox_extent": bbox_extent,
            },
            "graspnet": grasp_plan,
            "selection": selection,
            "grasp_tcp_pose": grasp_tcp_pose,
            "precontact_tcp_pose": precontact_tcp_pose,
            "insertion_waypoints": insertion_waypoints,
            "tcp_axis_world": tcp_axis_world,
            "object_name": object_name,
            "contract": "grasp_tcp_pose, precontact_tcp_pose, and insertion_waypoints are T_world_tcp; X2 motion converts TCP to EEF internally",
        }

    def _scene_object_pose_extent(self, object_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obj = self._env.env.scene.object_registry("name", object_name)
        if obj is None:
            raise ValueError(f"Object {object_name!r} not found in scene registry")
        if hasattr(obj, "aabb_center"):
            pos = self._to_numpy(obj.aabb_center).astype(np.float64).reshape(3)
        else:
            obj_pos, _obj_quat = obj.get_position_orientation()
            pos = self._to_numpy(obj_pos).astype(np.float64).reshape(3)
        obj_pos, obj_quat = obj.get_position_orientation()
        quat = self._to_numpy(obj_quat).astype(np.float64).reshape(4)
        try:
            extent = self._to_numpy(obj.aabb_extent).astype(np.float64).reshape(3)
        except Exception:
            extent = np.full(3, 0.04, dtype=np.float64)
        return pos, quat, extent

    def get_sim_known_tabletop_obstacles(
        self,
        object_name: str,
        table_name: str,
        *,
        object_margin: float = 0.012,
        table_margin_xy: float = 0.02,
        table_margin_z: float = 0.006,
        include_object: bool = True,
        include_table: bool = True,
    ) -> list[dict[str, Any]]:
        """Build short-term simulation-known box obstacles for X2 PyRoKi planning.

        This is explicitly a simulation-task helper. It reads the current
        OmniGibson scene pose and AABB extent for the target object and support
        table, inflates them by engineering margins, and returns PyRoKi box
        obstacles in world frame. It is not a 2real perception primitive.

        Args:
            object_name: Scene object to protect during precontact approach.
            table_name: Scene table/support object to avoid.

        Returns:
            List of PyRoKi world-frame box obstacle dictionaries. The source is
            sim-known scene state, not vision.
        """
        obstacles: list[dict[str, Any]] = []
        if include_object:
            obj_pos, _obj_quat, obj_extent = self._scene_object_pose_extent(object_name)
            obstacles.append(
                {
                    "type": "box",
                    "name": object_name,
                    "position": obj_pos.tolist(),
                    "extent": (obj_extent + 2.0 * float(object_margin)).tolist(),
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "source": "sim_known_scene_aabb",
                }
            )
        if include_table:
            table_pos, _table_quat, table_extent = self._scene_object_pose_extent(table_name)
            table_margin = np.array(
                [float(table_margin_xy), float(table_margin_xy), float(table_margin_z)],
                dtype=np.float64,
            )
            obstacles.append(
                {
                    "type": "box",
                    "name": table_name,
                    "position": table_pos.tolist(),
                    "extent": (table_extent + table_margin).tolist(),
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "source": "sim_known_scene_aabb",
                }
            )
        return obstacles

    @staticmethod
    def _without_obstacle_named(obstacles_world: list[dict[str, Any]] | None, name: str | None) -> list[dict[str, Any]] | None:
        if obstacles_world is None:
            return None
        if not name:
            return [dict(obs) for obs in obstacles_world]
        return [dict(obs) for obs in obstacles_world if str(obs.get("name")) != str(name)]

    @staticmethod
    def _coerce_pose(pose: Any) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = pose
        return np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(quat, dtype=np.float64).reshape(4)

    def execute_tcp_grasp_plan(
        self,
        plan: dict[str, Any],
        *,
        arm: int = 1,
        place_position: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        place_tcp_pose: tuple[np.ndarray, np.ndarray] | None = None,
        obstacles_world: list[dict[str, Any]] | None = None,
        transfer_obstacles_world: list[dict[str, Any]] | None = None,
        release: bool = True,
        retreat_after_release: bool = True,
        retreat_height: float = 0.08,
        place_approach_height: float = 0.08,
        timesteps: int = 18,
        dt: float = 0.08,
        max_joint_step: float = 0.022,
        insert_max_joint_step: float = 0.011,
        transfer_max_joint_step: float = 0.022,
        place_insert_max_joint_step: float = 0.011,
        settle_steps: int = 16,
        hold_steps_per_waypoint: int = 2,
        insert_hold_steps_per_waypoint: int = 5,
        close_hold_steps: int = 30,
        release_hold_steps: int = 24,
        final_tcp_threshold: float = 0.025,
        final_ori_threshold: float = 0.25,
        place_position_threshold: float = 0.08,
        require_object_in_hand_for_place: bool = True,
        skip_place_if_no_object_in_hand: bool = True,
        place_object_correction_steps: int = 0,
        place_object_correction_threshold: float = 0.025,
        place_object_correction_max_step: float = 0.05,
    ) -> dict[str, Any]:
        """Execute a visual X2 TCP grasp plan, optionally as a full pick-place.

        Args:
            plan: Result from ``plan_visual_grasp_tcp_pose``. Its TCP poses are
                all ``T_world_tcp``.
            place_position: Optional desired final object-center position in
                world frame. If provided without ``place_tcp_pose``, the method
                keeps the visual grasp TCP-to-object offset and computes a
                matching place TCP pose.
            place_tcp_pose: Optional explicit release TCP pose in world frame.
            obstacles_world: Optional PyRoKi world obstacles for the precontact
                move. In the short-term CaP-X sim task these are sim-known
                inflated boxes for the cube and table.
            transfer_obstacles_world: Optional obstacles for lift/transfer/place
                moves. Defaults to ``obstacles_world`` with the grasped object
                removed, so the held cube is not treated as a fixed obstacle.
            place_position_threshold: Task-level success radius in meters for
                the released object center.
            require_object_in_hand_for_place: If true, a pick-place execution
                only succeeds when the object is detected in the gripper after
                closing. This prevents false positives where the object is
                merely pushed near the target.
            skip_place_if_no_object_in_hand: If true, abort the place leg when
                the gripper is empty after close, then retreat to precontact.
            place_object_correction_steps: Optional sim-only pre-release
                correction count. When greater than zero, the API reads the
                sim-known object center before opening the gripper and nudges
                the TCP by the clipped object-center error. This is not a
                2real primitive.

        Returns:
            Execution summary with before-close TCP error and, when requested,
            place/release diagnostics. ``ok`` is task-level success:
            before-close TCP target reached and, if placing, the object was
            actually detected in hand after close and released near the target.
        """
        if not plan.get("ok"):
            return {"ok": False, "source": "x2_tcp_grasp_executor", "error": "input plan is not ok"}
        if obstacles_world is None:
            obstacles_world = plan.get("obstacles_world")
        object_name = plan.get("object_name") or (plan.get("pose_estimate") or {}).get("object_name")
        transfer_obstacles = (
            self._without_obstacle_named(obstacles_world, object_name)
            if transfer_obstacles_world is None
            else transfer_obstacles_world
        )
        self.open_gripper(arm=arm)
        self.settle_robot(steps=12)
        precontact_ok = self.move_tcp_pyroki_trajopt(
            plan["precontact_tcp_pose"],
            arm=arm,
            obstacles_world=obstacles_world,
            timesteps=int(timesteps),
            dt=float(dt),
            max_joint_step=float(max_joint_step),
            settle_steps=int(settle_steps),
            hold_steps_per_waypoint=int(hold_steps_per_waypoint),
        )
        insertion_results = []
        for waypoint in plan.get("insertion_waypoints", []) or []:
            ok = self.move_tcp_joint_ik(
                waypoint["tcp_pose"],
                arm=arm,
                pos_thresh=0.018 if waypoint.get("name") != "grasp" else 0.015,
                ori_thresh=0.25,
                max_joint_step=float(insert_max_joint_step),
                max_steps=240,
                settle_steps=int(settle_steps),
                hold_steps_per_waypoint=int(insert_hold_steps_per_waypoint),
            )
            reached_tcp_pose = self.get_current_tcp_pose(arm=arm)
            target_pos = np.asarray(waypoint["tcp_pose"][0], dtype=np.float64).reshape(3)
            target_quat = np.asarray(waypoint["tcp_pose"][1], dtype=np.float64).reshape(4)
            insertion_results.append(
                {
                    "name": waypoint.get("name"),
                    "ok": bool(ok),
                    "target_tcp_pose": waypoint["tcp_pose"],
                    "reached_tcp_pose": reached_tcp_pose,
                    "tcp_error_m": float(np.linalg.norm(reached_tcp_pose[0] - target_pos)),
                    "ori_error_rad": self._quat_error_rad(reached_tcp_pose[1], target_quat),
                }
            )
        before_close_tcp_pose = self.get_current_tcp_pose(arm=arm)
        grasp_pos = np.asarray(plan["grasp_tcp_pose"][0], dtype=np.float64).reshape(3)
        grasp_quat = np.asarray(plan["grasp_tcp_pose"][1], dtype=np.float64).reshape(4)
        before_close_error = {
            "tcp_error_m": float(np.linalg.norm(before_close_tcp_pose[0] - grasp_pos)),
            "ori_error_rad": self._quat_error_rad(before_close_tcp_pose[1], grasp_quat),
            "reached_tcp_pose": before_close_tcp_pose,
        }
        final_reached = bool(
            before_close_error["tcp_error_m"] <= float(final_tcp_threshold)
            and before_close_error["ori_error_rad"] <= float(final_ori_threshold)
        )
        self.close_gripper(arm=arm)
        self.settle_robot(steps=int(close_hold_steps))
        after_close_tcp_pose = self.get_current_tcp_pose(arm=arm)
        object_after_close = None
        object_in_hand = None
        if object_name:
            try:
                pos, quat, extent = self._scene_object_pose_extent(str(object_name))
                object_after_close = {"position_world": pos, "quat_xyzw_world": quat, "bbox_extent": extent}
            except Exception as exc:
                object_after_close = {"error": repr(exc)}
        try:
            object_in_hand = bool(self.check_object_in_hand(arm=arm))
        except Exception as exc:
            object_in_hand = {"error": repr(exc)}

        place_summary: dict[str, Any] | None = None
        place_requested = place_position is not None or place_tcp_pose is not None
        skip_place_due_to_empty_gripper = bool(
            place_requested
            and bool(require_object_in_hand_for_place)
            and bool(skip_place_if_no_object_in_hand)
            and object_in_hand is not True
        )
        if skip_place_due_to_empty_gripper:
            self.open_gripper(arm=arm)
            self.settle_robot(steps=max(8, int(settle_steps)))
            retreat_to_precontact_ok = self.move_tcp_joint_ik(
                plan["precontact_tcp_pose"],
                arm=arm,
                pos_thresh=0.025,
                ori_thresh=0.35,
                max_joint_step=float(insert_max_joint_step),
                max_steps=220,
                settle_steps=int(settle_steps),
                hold_steps_per_waypoint=int(hold_steps_per_waypoint),
            )
            place_summary = {
                "requested": True,
                "skipped": "object_not_in_hand_after_close",
                "release": False,
                "retreat_after_release": False,
                "target_object_position_world": None if place_position is None else np.asarray(place_position, dtype=np.float64).reshape(3),
                "place_error_m": None,
                "place_position_threshold": float(place_position_threshold),
                "retreat_to_precontact_ok": bool(retreat_to_precontact_ok),
            }
        elif place_requested:
            current_tcp_pos, current_tcp_quat = self._coerce_pose(after_close_tcp_pose)
            lifted_tcp_pose = (current_tcp_pos + np.array([0.0, 0.0, float(retreat_height)], dtype=np.float64), current_tcp_quat)
            lift_ok = self.move_tcp_joint_ik(
                lifted_tcp_pose,
                arm=arm,
                pos_thresh=0.02,
                ori_thresh=0.3,
                max_joint_step=float(insert_max_joint_step),
                max_steps=260,
                settle_steps=int(settle_steps),
                hold_steps_per_waypoint=int(insert_hold_steps_per_waypoint),
            )

            grasp_pos, grasp_quat = self._coerce_pose(plan["grasp_tcp_pose"])
            if place_tcp_pose is None:
                target_object_pos = np.asarray(place_position, dtype=np.float64).reshape(3)
                if isinstance(object_after_close, dict) and "position_world" in object_after_close:
                    close_object_pos = np.asarray(object_after_close["position_world"], dtype=np.float64).reshape(3)
                    tcp_from_object = current_tcp_pos - close_object_pos
                    tcp_from_object_source = "after_close_sim_known_object_pose"
                else:
                    visual_object_pos = np.asarray(
                        (plan.get("pose_estimate") or {}).get("position_world", grasp_pos),
                        dtype=np.float64,
                    ).reshape(3)
                    tcp_from_object = grasp_pos - visual_object_pos
                    tcp_from_object_source = "visual_plan_pose_estimate"
                release_tcp_pose = (target_object_pos + tcp_from_object, grasp_quat.copy())
            else:
                release_tcp_pose = self._coerce_pose(place_tcp_pose)
                target_object_pos = None if place_position is None else np.asarray(place_position, dtype=np.float64).reshape(3)
                tcp_from_object = None
                tcp_from_object_source = "explicit_place_tcp_pose"

            release_pos, release_quat = self._coerce_pose(release_tcp_pose)
            place_pre_tcp_pose = (
                release_pos + np.array([0.0, 0.0, float(place_approach_height)], dtype=np.float64),
                release_quat.copy(),
            )
            place_pre_ok = self.move_tcp_pyroki_trajopt(
                place_pre_tcp_pose,
                arm=arm,
                obstacles_world=transfer_obstacles,
                timesteps=int(timesteps),
                dt=float(dt),
                max_joint_step=float(transfer_max_joint_step),
                settle_steps=int(settle_steps),
                hold_steps_per_waypoint=int(hold_steps_per_waypoint),
            )
            object_corrections = []
            if target_object_pos is not None and object_name and int(place_object_correction_steps) > 0:
                elevated_target_object_pos = target_object_pos + np.array(
                    [0.0, 0.0, float(place_approach_height)],
                    dtype=np.float64,
                )
                for correction_idx in range(int(place_object_correction_steps)):
                    try:
                        obj_pos, obj_quat, obj_extent = self._scene_object_pose_extent(str(object_name))
                        object_error = elevated_target_object_pos - np.asarray(obj_pos, dtype=np.float64).reshape(3)
                        object_error_m = float(np.linalg.norm(object_error))
                        correction_record: dict[str, Any] = {
                            "idx": correction_idx,
                            "stage": "pre_release_above_target",
                            "target_object_position_world": elevated_target_object_pos,
                            "object_position_world": obj_pos,
                            "object_quat_xyzw_world": obj_quat,
                            "object_bbox_extent": obj_extent,
                            "object_error": object_error,
                            "object_error_m": object_error_m,
                        }
                        if object_error_m <= float(place_object_correction_threshold):
                            correction_record["ok"] = True
                            correction_record["skipped"] = "already_within_threshold"
                            object_corrections.append(correction_record)
                            break
                        correction_delta = np.clip(
                            object_error,
                            -float(place_object_correction_max_step),
                            float(place_object_correction_max_step),
                        )
                        correction_tcp_pose = self.get_current_tcp_pose(arm=arm)
                        corrected_tcp_pose = (
                            np.asarray(correction_tcp_pose[0], dtype=np.float64).reshape(3) + correction_delta,
                            np.asarray(correction_tcp_pose[1], dtype=np.float64).reshape(4),
                        )
                        correction_ok = self.move_tcp_joint_ik(
                            corrected_tcp_pose,
                            arm=arm,
                            pos_thresh=0.018,
                            ori_thresh=0.35,
                            max_joint_step=float(place_insert_max_joint_step),
                            max_steps=180,
                            settle_steps=int(settle_steps),
                            hold_steps_per_waypoint=int(insert_hold_steps_per_waypoint),
                        )
                        correction_record.update(
                            {
                                "ok": bool(correction_ok),
                                "correction_delta": correction_delta,
                                "target_tcp_pose": corrected_tcp_pose,
                                "reached_tcp_pose": self.get_current_tcp_pose(arm=arm),
                            }
                        )
                        object_corrections.append(correction_record)
                    except Exception as exc:
                        object_corrections.append({"idx": correction_idx, "ok": False, "error": repr(exc)})
                        break
            pre_descent_tcp_pose = self.get_current_tcp_pose(arm=arm)
            if place_tcp_pose is None:
                pre_descent_pos = np.asarray(pre_descent_tcp_pose[0], dtype=np.float64).reshape(3)
                pre_descent_quat = np.asarray(pre_descent_tcp_pose[1], dtype=np.float64).reshape(4)
                release_tcp_pose = (
                    pre_descent_pos - np.array([0.0, 0.0, float(place_approach_height)], dtype=np.float64),
                    pre_descent_quat.copy(),
                )
                release_pos, release_quat = self._coerce_pose(release_tcp_pose)
            place_insert_ok = self.move_tcp_joint_ik(
                release_tcp_pose,
                arm=arm,
                pos_thresh=0.02,
                ori_thresh=0.3,
                max_joint_step=float(place_insert_max_joint_step),
                max_steps=260,
                settle_steps=int(settle_steps),
                hold_steps_per_waypoint=int(insert_hold_steps_per_waypoint),
            )
            before_release_tcp_pose = self.get_current_tcp_pose(arm=arm)
            before_release_error = {
                "tcp_error_m": float(np.linalg.norm(before_release_tcp_pose[0] - release_pos)),
                "ori_error_rad": self._quat_error_rad(before_release_tcp_pose[1], release_quat),
                "reached_tcp_pose": before_release_tcp_pose,
                "target_tcp_pose": release_tcp_pose,
            }
            if release:
                self.open_gripper(arm=arm)
                self.settle_robot(steps=int(release_hold_steps))
            object_after_release = None
            place_error_m = None
            if object_name:
                try:
                    pos, quat, extent = self._scene_object_pose_extent(str(object_name))
                    object_after_release = {"position_world": pos, "quat_xyzw_world": quat, "bbox_extent": extent}
                    if target_object_pos is not None:
                        place_error_m = float(np.linalg.norm(pos - target_object_pos))
                except Exception as exc:
                    object_after_release = {"error": repr(exc)}
            retreat_after_release_pose = None
            retreat_after_release_ok = None
            if retreat_after_release:
                current_release_tcp_pose = self.get_current_tcp_pose(arm=arm)
                retreat_after_release_pose = (
                    np.asarray(current_release_tcp_pose[0], dtype=np.float64).reshape(3)
                    + np.array([0.0, 0.0, float(retreat_height)], dtype=np.float64),
                    np.asarray(current_release_tcp_pose[1], dtype=np.float64).reshape(4),
                )
                retreat_after_release_ok = self.move_tcp_joint_ik(
                    retreat_after_release_pose,
                    arm=arm,
                    pos_thresh=0.025,
                    ori_thresh=0.35,
                    max_joint_step=float(insert_max_joint_step),
                    max_steps=220,
                    settle_steps=int(settle_steps),
                    hold_steps_per_waypoint=int(hold_steps_per_waypoint),
                )
            place_summary = {
                "requested": True,
                "release": bool(release),
                "retreat_after_release": bool(retreat_after_release),
                "target_object_position_world": None if place_position is None else np.asarray(place_position, dtype=np.float64).reshape(3),
                "tcp_from_object_source": tcp_from_object_source,
                "tcp_from_object": tcp_from_object,
                "release_tcp_pose": release_tcp_pose,
                "place_pre_tcp_pose": place_pre_tcp_pose,
                "pre_descent_tcp_pose": pre_descent_tcp_pose,
                "lift_ok": bool(lift_ok),
                "place_pre_ok": bool(place_pre_ok),
                "place_insert_ok": bool(place_insert_ok),
                "object_corrections": object_corrections,
                "before_release_error": before_release_error,
                "object_after_release": object_after_release,
                "place_error_m": place_error_m,
                "place_position_threshold": float(place_position_threshold),
                "retreat_after_release_pose": retreat_after_release_pose,
                "retreat_after_release_ok": bool(retreat_after_release_ok),
            }
        place_ok = True
        grasp_ok = True
        place_motion_steps_ok = None
        if place_summary is not None:
            grasp_ok = (
                object_in_hand is True
                if bool(require_object_in_hand_for_place)
                else True
            )
            if place_position is None:
                place_error_ok = (
                    place_summary["place_error_m"] is None
                    or place_summary["place_error_m"] <= float(place_position_threshold)
                )
            else:
                place_error_ok = (
                    place_summary["place_error_m"] is not None
                    and place_summary["place_error_m"] <= float(place_position_threshold)
                )
            place_motion_steps_ok = bool(
                place_summary.get("lift_ok", False)
                and place_summary.get("place_pre_ok", False)
                and place_summary.get("place_insert_ok", False)
            )
            # A true pick-place success needs both grasp evidence and final
            # placement. Per-segment motion return flags remain in the summary
            # for debugging controller tracking.
            place_ok = bool(grasp_ok and place_error_ok)
        return {
            "ok": bool(final_reached and place_ok),
            "source": "x2_tcp_grasp_executor",
            "success_condition": "before-close TCP pose reached within final thresholds; if place requested, object detected in hand after close and released near target",
            "final_tcp_threshold": float(final_tcp_threshold),
            "final_ori_threshold": float(final_ori_threshold),
            "place_position_threshold": float(place_position_threshold),
            "require_object_in_hand_for_place": bool(require_object_in_hand_for_place),
            "skip_place_if_no_object_in_hand": bool(skip_place_if_no_object_in_hand),
            "precontact_ok": bool(precontact_ok),
            "insertion_results": insertion_results,
            "before_close_error": before_close_error,
            "before_close_reached": final_reached,
            "grasp_ok": bool(grasp_ok),
            "place_motion_steps_ok": place_motion_steps_ok,
            "after_close_tcp_pose": after_close_tcp_pose,
            "object_name": object_name,
            "object_in_hand_after_close": object_in_hand,
            "object_after_close": object_after_close,
            "place": place_summary or {"requested": False},
            "obstacles_world": obstacles_world,
            "transfer_obstacles_world": transfer_obstacles,
            "contract": "grasp, precontact, insertion, and place TCP poses are T_world_tcp; place_position is a world-frame object-center target",
            "last_motion_debug": self.get_last_motion_debug(),
        }

    def pick_and_place_visual_object(
        self,
        object_name: str,
        place_position: np.ndarray | list[float] | tuple[float, float, float],
        *,
        prompts: list[str] | tuple[str, ...] | None = None,
        camera_name: str | None = None,
        arm: int = 1,
        table_name: str | None = None,
        orientation_quat_xyzw: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        workspace_bounds: dict[str, tuple[float, float]] | None = None,
        candidate_indices: list[int] | tuple[int, ...] | None = None,
        place_position_threshold: float = 0.10,
        use_sim_known_obstacles: bool = True,
        sim_place_correction_steps: int = 2,
    ) -> dict[str, Any]:
        """Pick an object with vision and place it at a world-frame target.

        This is the recommended short-term CaP-X task-level primitive for X2
        tabletop simulation tasks. It assumes the environment, robot, cameras,
        object, table, and task have already been created by CaP-X / BEHAVIOR.
        Generated code should call this API instead of creating or resetting
        the simulator.

        The visual/action contract is:
        - vision produces X2 executable TCP targets, ``T_world_tcp``;
        - motion consumes those same ``T_world_tcp`` targets;
        - TCP-to-EEF conversion is handled internally by the X2 API.

        Args:
            object_name: Scene object to pick, e.g. ``"x2_pick_place_red_cube"``.
            place_position: Desired final object-center position in world
                frame, ``[x, y, z]`` meters.
            prompts: OWL-ViT text prompts, e.g. ``["red cube", "red block"]``.
            camera_name: Camera to use. If omitted, uses the X2 chest camera.
            arm: X2 arm index; current tabletop tasks use right arm ``1``.
            table_name: Optional support table name. When provided with
                ``use_sim_known_obstacles=True``, the API uses sim-known table
                and object AABBs as PyRoKi box obstacles.
            orientation_quat_xyzw: Optional world-frame TCP orientation prior.
            workspace_bounds: Optional visual grasp workspace filter.
            candidate_indices: Ranked adapted GraspNet candidate indices to
                try. If a candidate reaches the TCP target but closes on empty
                gripper, the API can replan with the next candidate.
            place_position_threshold: Success radius in meters for final
                object-center placement.
            use_sim_known_obstacles: If true, use sim-known obstacle boxes.
                This is acceptable for short-term simulation integration only.
            sim_place_correction_steps: Number of sim-only pre-release object
                center correction steps. Set to ``0`` to disable.

        Returns:
            Dict with ``ok``, ``plan``, ``obstacles_world``, and ``execution``.
            ``ok`` means the grasp TCP target was reached, the object was
            detected in hand after closing, and the released object center is
            within ``place_position_threshold``.
        """
        prompt_list = [object_name] if prompts is None else list(prompts)
        selected_camera_name = self.get_chest_camera_name() if camera_name is None else camera_name
        reference_quat = (
            None
            if orientation_quat_xyzw is None
            else np.asarray(orientation_quat_xyzw, dtype=np.float64).reshape(4)
        )

        self.settle_robot(steps=8)
        self.open_gripper(arm=arm)
        self.settle_robot(steps=12)

        obstacles_world = None
        if use_sim_known_obstacles and table_name:
            obstacles_world = self.get_sim_known_tabletop_obstacles(
                object_name,
                table_name,
                object_margin=0.012,
                table_margin_xy=0.02,
                table_margin_z=0.006,
            )

        indices = [0] if candidate_indices is None else [int(idx) for idx in candidate_indices]
        attempts: list[dict[str, Any]] = []
        last_plan: dict[str, Any] | None = None
        last_execution: dict[str, Any] | None = None
        for attempt_idx, candidate_index in enumerate(indices):
            plan = self.plan_visual_grasp_tcp_pose(
                object_name,
                prompts=prompt_list,
                camera_name=selected_camera_name,
                arm=arm,
                external=False,
                orientation_quat_xyzw=reference_quat,
                object_pose_method="aabb_center",
                graspnet_forward_passes=4,
                graspnet_max_retries=30,
                graspnet_max_candidates=24,
                min_mask_pixels=12,
                workspace_bounds=workspace_bounds,
                precontact_distance=0.08,
                insert_waypoints=10,
                candidate_index=int(candidate_index),
                proxy_guard_object_size=0.04,
            )
            last_plan = plan
            if not plan.get("ok"):
                attempts.append(
                    {
                        "attempt": int(attempt_idx),
                        "candidate_index": int(candidate_index),
                        "ok": False,
                        "stage": "plan_visual_grasp_tcp_pose",
                        "plan_error": plan.get("error"),
                    }
                )
                continue

            execution = self.execute_tcp_grasp_plan(
                plan,
                arm=arm,
                place_position=np.asarray(place_position, dtype=np.float64).reshape(3),
                obstacles_world=obstacles_world,
                release=True,
                retreat_after_release=False,
                retreat_height=0.08,
                place_approach_height=0.08,
                timesteps=18,
                dt=0.08,
                max_joint_step=0.022,
                insert_max_joint_step=0.011,
                transfer_max_joint_step=0.022,
                place_insert_max_joint_step=0.011,
                settle_steps=16,
                hold_steps_per_waypoint=2,
                insert_hold_steps_per_waypoint=5,
                close_hold_steps=30,
                release_hold_steps=24,
                final_tcp_threshold=0.035,
                final_ori_threshold=0.25,
                place_position_threshold=float(place_position_threshold),
                require_object_in_hand_for_place=True,
                skip_place_if_no_object_in_hand=True,
                place_object_correction_steps=int(sim_place_correction_steps),
                place_object_correction_threshold=0.025,
                place_object_correction_max_step=0.05,
            )
            last_execution = execution
            attempts.append(
                {
                    "attempt": int(attempt_idx),
                    "candidate_index": int(candidate_index),
                    "ok": bool(execution.get("ok")),
                    "before_close_reached": bool(execution.get("before_close_reached")),
                    "object_in_hand_after_close": execution.get("object_in_hand_after_close"),
                    "before_close_error": execution.get("before_close_error"),
                    "place": execution.get("place"),
                }
            )
            if execution.get("ok"):
                break
            if execution.get("object_in_hand_after_close") is True:
                break

        if last_plan is None or not last_plan.get("ok"):
            return {
                "ok": False,
                "source": "x2_visual_pick_place",
                "stage": "plan_visual_grasp_tcp_pose",
                "plan": last_plan,
                "attempts": attempts,
                "contract": "visual and action targets are T_world_tcp",
            }
        execution = last_execution or {"ok": False, "error": "no execution attempt completed"}
        return {
            "ok": bool(execution.get("ok")),
            "source": "x2_visual_pick_place",
            "object_name": object_name,
            "place_position_world": np.asarray(place_position, dtype=np.float64).reshape(3),
            "place_position_threshold": float(place_position_threshold),
            "plan": last_plan,
            "obstacles_world": obstacles_world,
            "execution": execution,
            "attempts": attempts,
            "contract": "visual grasp plan and action targets are T_world_tcp; place_position is world-frame object center",
            "sim_only": {
                "sim_known_obstacles": bool(use_sim_known_obstacles and table_name),
                "sim_place_correction_steps": int(sim_place_correction_steps),
            },
        }

    def sample_grasp_pose(self, object_name: str, arm: int = 1) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Sample a near-field top-down grasp pose for an object.

        Uses the X2 low-level wrapper's custom sampler that computes
        grasp/pregrasp poses from the object's current position and AABB
        extent, which is appropriate for fixed-base manipulation.

        Args:
            object_name: Exact OmniGibson object name.
            arm: Arm index. ``0`` is left, ``1`` is right.

        Returns:
            ``(pregrasp_pose, grasp_pose)``, each as ``(position, quaternion_xyzw)``.
        """
        pregrasp_pose, grasp_pose = self._env._sample_grasp_pose(object_name, arm=arm)
        return (
            (self._to_numpy(pregrasp_pose[0]), self._to_numpy(pregrasp_pose[1])),
            (self._to_numpy(grasp_pose[0]), self._to_numpy(grasp_pose[1])),
        )

    def grasp_object(
        self,
        pregrasp_pose: tuple[np.ndarray, np.ndarray],
        grasp_pose: tuple[np.ndarray, np.ndarray],
        object_name: str,
        arm: int = 1,
    ) -> bool:
        """Execute a fixed-base grasp attempt.

        Args:
            pregrasp_pose: Pregrasp pose `(position, quaternion_xyzw)`.
            grasp_pose: Grasp pose `(position, quaternion_xyzw)`.
            object_name: Exact OmniGibson object name. Used for logging / API parity.
            arm: Arm index. `0` is left, `1` is right.

        Returns:
            Whether OmniGibson reports an object in hand after the attempt.
        """
        del object_name
        self.open_gripper(arm=arm)
        if pregrasp_pose is not None:
            self.move_hand(pregrasp_pose, arm=arm)
        self.move_hand(grasp_pose, arm=arm)
        self.close_gripper(arm=arm)
        self.lift_arm(arm=arm)
        return self.check_object_in_hand(arm=arm)

    def check_object_in_hand(self, arm: int = 1) -> bool:
        """Check whether OmniGibson reports an object in the selected hand."""
        return bool(self._env.check_object_in_hand(arm=arm))

    def lift_arm(self, arm: int = 1) -> bool:
        """Lift the selected hand upward a short distance."""
        return bool(self._env._lift_arm(arm=arm))

    def move_to_joint_positions(
        self,
        target_joint_positions,
        arm: int | None = None,
        max_joint_step: float = 0.035,
        max_steps: int = 180,
        settle_steps: int = 12,
        hold_steps_per_waypoint: int = 1,
    ) -> bool:
        """Move to joint positions.

        In the joint-controller X2 path, this sends absolute joint-position
        targets through joint-space interpolation. If `arm` is set, only that
        arm's seven joints are allowed to change.
        """
        return self._env._move_to_joint_positions(
            target_joint_positions,
            arm=arm,
            max_joint_step=max_joint_step,
            max_steps=max_steps,
            settle_steps=settle_steps,
            hold_steps_per_waypoint=hold_steps_per_waypoint,
        )

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        arm: int = 1,
        offset_translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> np.ndarray:
        """Solve one-shot PyRoKi IK for an EEF world pose and return full X2 qpos.

        Args:
            position: Target EEF position in world frame.
            quaternion_wxyz: Target EEF orientation in world frame, `[w, x, y, z]`.
            arm: Arm index. Only this arm's seven joints are changed in the returned target.
            offset_translation: Accepted for R1Pro API compatibility; unused for X2.
        """
        del offset_translation
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        q_target, _debug = self._env._solve_pyroki_eef_joint_target(
            (np.asarray(position, dtype=np.float64).reshape(3), quat_xyzw),
            arm=arm,
        )
        return q_target


class X2PickPlaceApi(X2ControlApi):
    """Small LLM-facing API for the accepted X2 tabletop pick-place baseline.

    This wrapper intentionally exposes only the task-level primitive and a few
    safe helpers. The full ``X2ControlApi`` remains available for debugging and
    lower-level experiments, but ordinary CaP-X task code should prefer this
    reduced API so generated programs do not need to reason about cameras,
    GraspNet frames, TCP-to-EEF conversion, IK, or trajectory optimization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x2_pick_place_red_cube_result: dict[str, Any] | None = None

    def functions(self) -> dict[str, Any]:
        """Return the small function set exposed to generated pick-place code."""
        return {
            "pick_and_place_red_cube": self.pick_and_place_red_cube,
            "pick_and_place_visual_object": self.pick_and_place_visual_object,
            "get_chest_camera_name": self.get_chest_camera_name,
            "settle_robot": self.settle_robot,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_last_motion_debug": self.get_last_motion_debug,
        }

    def pick_and_place_red_cube(
        self,
        place_position: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        *,
        place_position_threshold: float = 0.10,
        force: bool = False,
    ) -> dict[str, Any]:
        """Pick the task red cube and place it at the green marker.

        This is the narrow LLM-facing primitive for
        ``x2_pick_place_red_cube.yaml``. It fixes the task object, table,
        camera, prompts, grasp-orientation prior, workspace bounds, and
        sim-only stabilizers that were validated for the accepted X2 tabletop
        pick-place baseline.
        """
        if self._x2_pick_place_red_cube_result is not None and not bool(force):
            cached = dict(self._x2_pick_place_red_cube_result)
            cached["cached"] = True
            return cached

        target = (
            np.array([0.32, 0.055, 0.921], dtype=np.float64)
            if place_position is None
            else np.asarray(place_position, dtype=np.float64).reshape(3)
        )
        result = self.pick_and_place_visual_object(
            "x2_pick_place_red_cube",
            target,
            prompts=["red cube", "red block", "red box", "cube"],
            camera_name=self.get_chest_camera_name(),
            arm=1,
            table_name="x2_pick_place_table",
            orientation_quat_xyzw=np.array(
                [0.53604597, 0.63824742, 0.39958203, -0.38161388],
                dtype=np.float64,
            ),
            workspace_bounds={"x": (0.16, 0.50), "y": (-0.38, 0.12), "z": (0.78, 1.12)},
            candidate_indices=(0, 1, 2),
            place_position_threshold=float(place_position_threshold),
            use_sim_known_obstacles=True,
            sim_place_correction_steps=4,
        )
        result = dict(result)
        result["task_primitive"] = "pick_and_place_red_cube"
        result["cached"] = False
        self._x2_pick_place_red_cube_result = result
        return result
