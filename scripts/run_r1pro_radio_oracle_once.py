from __future__ import annotations

from pathlib import Path

import mediapy as media
import numpy as np
import torch

from capx.envs.simulators.r1pro_b1k import (
    R1ProBehaviourLowLevel,
    holonomic_base_command_in_world_frame,
)


def main() -> None:
    out_dir = Path("outputs/r1pro_radio_oracle_once")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = R1ProBehaviourLowLevel(save_video=True)
    env.reset(options={"trial": 0})

    radio_state = env.get_object_state("radio")
    initial_radio_pose = radio_state[0].clone()
    radio_pose = radio_state[0]

    grasp_obj = env.env.scene.object_registry("name", "radio_89")
    table_obj = env.env.scene.object_registry("name", "coffee_table_koagbh_0")
    table_pose, _ = table_obj.get_position_orientation()
    table_a, table_b = table_obj.aabb
    table_width = table_b[0] - table_a[0]
    table_length = table_b[1] - table_a[1]

    short_side_center = (table_pose[0], table_pose[1] - table_length / 2)
    long_side_center = (table_pose[0] + table_width / 2, table_pose[1])
    dist_to_short_side = np.linalg.norm(np.array(short_side_center) - np.array(radio_pose[:2]))
    dist_to_long_side = np.linalg.norm(np.array(long_side_center) - np.array(radio_pose[:2]))

    buffer_dist = 0.1
    if dist_to_short_side < dist_to_long_side:
        robot_target = (radio_pose[0], short_side_center[1] - buffer_dist, np.pi / 2)
    else:
        robot_target = (long_side_center[0] + buffer_dist, radio_pose[1], np.pi)

    reward = 0.0
    env._navigate_to_pose(robot_target)

    robot_pos, _ = env.robot.get_position_orientation()
    radio_pos, _ = grasp_obj.get_position_orientation()
    if np.linalg.norm(np.array(robot_pos[:2]) - np.array(radio_pos[:2])) < 0.3:
        reward += 1 / 3

    try:
        grasp_success = env._grasp_obj("radio_89")
    except Exception as exc:
        grasp_success = False
        print("Grasp failed:", exc)
    if grasp_success:
        reward += 1 / 3

    cur_joint_positions = env.get_joint_positions()
    target_joint_positions = cur_joint_positions.clone()
    target_joint_positions[10] -= 0.4
    action = env.robot.q_to_action(
        holonomic_base_command_in_world_frame(env.robot, target_joint_positions)
    )
    for _ in range(20):
        env.step(action)
        current_joint_positions = env.robot.get_joint_positions()
        if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
            break

    radio_pos, _ = grasp_obj.get_position_orientation()
    if env.controller._get_obj_in_hand() == grasp_obj and radio_pos[2] > initial_radio_pose[2] + 0.005:
        reward += 1 / 3

    frames = env.get_video_frames()
    for name, frame_list in frames.items():
        if len(frame_list) == 0:
            continue
        path = out_dir / f"{name}.mp4"
        media.write_video(path, np.asarray(frame_list), fps=30)
        print(f"Saved {path} ({len(frame_list)} frames)")

    print(f"Reward: {reward:.3f}")


if __name__ == "__main__":
    main()
