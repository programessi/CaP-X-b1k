# X2 仿真控制时间尺度说明

本文记录 X2 在 OmniGibson / Isaac 仿真里的三个时间尺度，以及它们和当前 one-shot IK + joint position controller 路线之间的关系。

## 结论摘要

当前 X2 配置不是“发一次 joint target 后机械臂立刻到位”。更准确地说：

1. Python 侧每调用一次 `env.step(action)`，仿真环境前进一个 action step。
2. 当前配置中 action step 是 30 Hz，即每 33.33 ms 接收一次新的高层 action / joint command。
3. Isaac physics 是 120 Hz，即每 8.33 ms 做一次物理积分和 joint drive / PD drive 更新。
4. 因为 `120 / 30 = 4`，默认情况下，一个 action command 只会被 Isaac 物理和驱动器执行 4 个 physics tick，然后就可能被下一个 command 替换。
5. 如果我们在 Python 侧快速连续发布一串插值 joint target，每个 target 只保持一个 action step，那么控制器可能还没把关节拉到当前 target，就已经收到下一个 target。

所以这里至少有三层“步”：

| 层级 | 当前值 | 含义 |
| --- | ---: | --- |
| Python / environment action step | 30 Hz | `env.step(action)` 的节奏，也是我们现在发布 joint target 的节奏 |
| Isaac physics step | 120 Hz | 物理积分、接触、关节 drive / PD 效果真正生效的离散时间步 |
| joint command hold / controller effective time | 默认每个 waypoint 1 个 action step | 一个给定 joint target 被保持多久，决定它实际经历多少个 physics tick |

默认情况下：

```text
seconds_per_action_step = 1 / 30 = 0.03333 s
physics_ticks_per_action = 120 / 30 = 4
```

如果每个插值 waypoint 只发布一次，那么每个 waypoint 只被执行：

```text
1 action step = 0.03333 s = 4 physics ticks
```

如果设置 `hold_steps_per_waypoint = 4`，同一个 waypoint 会重复发布 4 个 action step，那么它会被执行：

```text
4 action steps = 0.13333 s = 16 physics ticks
```

## 当前 X2 joint IK 执行方式

当前 `move_tcp_joint_ik()` 的路线是：

```text
T_world_tcp
  -> tcp_pose_to_eef_pose()
  -> T_world_eef
  -> PyRoKi one-shot IK 求 q_target
  -> joint-space linear interpolation
  -> 每个 q_cmd 通过 env.step(action) 发布给 JointController
```

关键点是，IK 解算和执行跟踪是两件事：

- IK 解算回答：“有没有一组关节角能让目标 EEF 位姿成立？”
- joint controller 执行回答：“在当前仿真、关节 drive、接触、限位、发布频率和保持时间下，真实关节是否跟到了这组关节角？”

当前 `_move_to_joint_positions()` 的基本逻辑是：

```text
q_current = 当前关节角
q_target = IK 解出来的目标关节角
steps = ceil(max_abs(q_target - q_current) / max_joint_step)

for i in 1..steps:
    q_cmd = q_current + alpha * (q_target - q_current)
    env.step(joint_position_action(q_cmd))

settle_robot_steps(settle_steps)
```

也就是说，默认每个插值点只保持一次 `env.step()`。如果 `max_joint_delta` 很大，路径会被拆成很多 waypoint，但每个 waypoint 被保持的真实物理时间仍然很短。

## 为什么仿真里要特别考虑三者关系

实物机器人通常也有三层频率：

- 上层规划或策略发布目标的频率，例如 10-100 Hz。
- 控制器内部 servo loop，例如 250 Hz、500 Hz、1 kHz。
- 机械系统本身的连续动力学响应。

仿真里这些都被离散化了，而且更显式：

- `env.step()` 是你给仿真系统输入 action 的节奏。
- `physics_frequency` 是仿真内部积分和 actuator drive 生效的节奏。
- command 被保持多久，决定 actuator 有多少 physics tick 去产生执行效果。

如果上层 command 更新太快，而每个 command 只保持很短时间，就会出现：

```text
q_target_1 还没跟上 -> q_target_2 已发布
q_target_2 还没跟上 -> q_target_3 已发布
...
最终关节一直落后于目标轨迹
```

这在大范围关节运动时更明显，因为每一步都存在跟踪滞后，误差会沿整段轨迹积累。

## 本次 X2 视觉抓取回放实验

我们复用了视觉链条得到的同一个目标，不重新跑 OWL-ViT / SAM2 / GraspNet，只测试 joint target 的执行跟踪。

目标来源：

```text
outputs/x2_chest_visual_grasp_to_joint_ik_demo_v2/summary.json
```

目标语义：

```text
selected candidate poses are T_world_tcp
move_tcp_joint_ik() converts T_world_tcp to T_world_eef
PyRoKi solves target joint positions for r_base_gripper / EEF
```

当前视觉估计本身很好：

```text
configured object center: [0.42, -0.04, 0.921]
estimated object position: [0.41794, -0.03955, 0.921867]
visual position error: 0.00228 m
```

所以这次 10 cm 级误差不是视觉目标位置错了。

## Baseline：每个 waypoint 只保持 1 个 action step

配置：

```text
action_frequency = 30 Hz
physics_frequency = 120 Hz
hold_steps_per_waypoint = 1
settle_steps = 20
max_joint_step = 0.022 rad
```

结果：

| 阶段 | TCP 误差 | EEF 位置误差 | EEF 姿态误差 | 最终最大关节误差 | command 时长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pregrasp | 0.122555 m | 0.099184 m | 0.315440 rad | 0.293911 rad | 1.433 s |
| grasp | 0.109980 m | 0.088624 m | 0.316426 rad | 0.382174 rad | 0.633 s |
| lift | 0.041433 m | 0.026958 m | 0.139304 rad | 0.083834 rad | 0.533 s |

闭爪前后：

```text
before_close_tcp_error_m = 0.109980
after_close_tcp_error_m  = 0.118023
```

这说明闭爪时 TCP 没有到预定抓取位姿。视频里看到“手到方块旁边闭爪”，和数据一致。

同时，IK 解算本身并没有 10 cm 级误差：

```text
pregrasp IK solve FK pos error ~= 5.9e-08 m
grasp    IK solve FK pos error ~= 0.00489 m
lift     IK solve FK pos error ~= 4.9e-08 m
```

因此 baseline 的主问题不是“PyRoKi 算不出目标”，而是“解出来的 joint target 没有被仿真关节实际跟上”。

输出：

```text
outputs/x2_replay_visual_target_joint_tracking_baseline_v1/summary.json
outputs/x2_replay_visual_target_joint_tracking_baseline_v1/video_combined.mp4
```

## Hold4：每个 waypoint 保持 4 个 action step

配置：

```text
action_frequency = 30 Hz
physics_frequency = 120 Hz
hold_steps_per_waypoint = 4
settle_steps = 20
max_joint_step = 0.022 rad
```

这会让每个 waypoint 从：

```text
0.033 s, 4 physics ticks
```

增加到：

```text
0.133 s, 16 physics ticks
```

结果：

| 阶段 | TCP 误差 | EEF 位置误差 | EEF 姿态误差 | 最终最大关节误差 | command 时长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pregrasp | 0.065933 m | 0.039141 m | 0.421943 rad | 0.368026 rad | 5.733 s |
| grasp | 0.042482 m | 0.029921 m | 0.164263 rad | 0.094213 rad | 1.600 s |
| lift | 0.006833 m | 0.004596 m | 0.019514 rad | 0.010158 rad | 1.867 s |

闭爪前后：

```text
before_close_tcp_error_m = 0.042482
after_close_tcp_error_m  = 0.051463
```

输出：

```text
outputs/x2_replay_visual_target_joint_tracking_hold4_v1/summary.json
outputs/x2_replay_visual_target_joint_tracking_hold4_v1/video_combined.mp4
```

## 对实验结果的解释

Hold4 明显降低了 grasp 和 lift 的误差：

```text
grasp TCP error: 0.109980 m -> 0.042482 m
lift  TCP error: 0.041433 m -> 0.006833 m
```

这说明 command 保持时间 / controller 执行时间确实是主要影响因素之一。原来的发布节奏太快时，joint drive 没有足够 physics tick 去跟踪每个插值目标。

但是 Hold4 仍然没有完全解决 pregrasp 和 grasp：

```text
pregrasp TCP error = 0.065933 m
grasp    TCP error = 0.042482 m
```

所以不能把全部问题简单归因于“发布频率太快”。剩余误差可能来自：

1. 当前目标位姿处在 X2 右臂较吃力的工作区。
2. 目标姿态导致某些关节接近限位或姿态跟踪困难。
3. 桌板、方块、手爪或机械臂之间存在接触 / 阻挡。
4. Joint drive 的 kp / kd / effort limit / velocity limit 使得实际响应不足。
5. 插值路径是关节空间直线，不保证中间过程远离碰撞或动态困难区域。

## 为什么以前 IK 测试能到毫米级

之前成功的毫米级测试目标大致在：

```text
target TCP: [0.260286, -0.214361, 0.949974]
```

那次 grasp 阶段：

```text
TCP error ~= 0.001072 m
joint final error ~= 0.004645 rad
joint command max delta ~= 0.166778 rad
joint command steps = 8
```

这次视觉链条的目标大致在：

```text
target TCP: [0.417940, -0.039550, 0.921867]
```

baseline pregrasp：

```text
joint command max delta ~= 0.926900 rad
joint command steps = 43
joint final error ~= 0.293911 rad
```

也就是说，两次不是同一个难度：

- 之前目标更靠近右臂自然工作区，所需关节变化更小。
- 这次目标更靠中间、更远，所需关节变化更大，姿态也更难。
- 大范围运动下，每个 waypoint 只保持 33 ms 的问题被放大了。

因此“之前毫米级”和“这次 10 cm 级”并不矛盾。之前证明的是某些目标点上 IK + joint controller 可以工作；这次暴露的是视觉抓取目标附近的执行跟踪和工作区问题。

## 后续诊断建议

建议把后续诊断分成四类，不要只看最终 TCP 误差：

1. IK 解算误差

检查：

```text
ik_solve_fk_pos_error_m
ik_solve_fk_ori_error_rad
```

如果这里已经很大，说明目标不可达、姿态不合适，或者 IK 配置 / URDF / link frame 有问题。

2. joint tracking 误差

检查：

```text
joint_final_error_rad
joint_command_max_delta_rad
joint_command_steps
joint_action_steps_sent
hold_steps_per_waypoint
estimated_command_duration_s
```

如果 IK solve 很好，但 `joint_final_error_rad` 很大，说明执行层没有跟上目标关节角。

3. command 与 physics 时间关系

检查：

```text
action_frequency_hz
physics_frequency_hz
physics_steps_per_action
seconds_per_action_step
```

对当前配置：

```text
30 Hz action, 120 Hz physics, 4 physics ticks per action
```

如果一个 waypoint 只保持一个 action step，就只有 4 个 physics tick 的执行时间。

4. 接触 / 阻挡 / 工作区

如果增加 hold 后仍然残留明显误差，需要检查：

- 目标是否太靠近桌板。
- EEF / TCP 目标姿态是否让手爪、腕部或前臂撞到桌板。
- 关节是否接近限位。
- 同一个 TCP 位置下，换几组姿态是否更容易跟踪。
- 同一个目标姿态下，把目标点移回之前成功区域是否恢复毫米级。

## 实用调参方向

短期可以先调 execution timing：

```text
hold_steps_per_waypoint: 1 -> 2 -> 4
settle_steps: 20 -> 60 -> 120
max_joint_step: 0.022 -> 0.015 或 0.010
```

但注意：

- 更小的 `max_joint_step` 会增加 waypoint 数量。
- 如果每个 waypoint 仍只保持 1 个 action step，不一定更稳。
- `hold_steps_per_waypoint` 增加的是每个 waypoint 的实际物理执行时间。
- `settle_steps` 只增加最终目标附近的等待，不改善中间路径跟踪。

更系统的路线是：

1. 先用 one-shot IK 得到目标关节角。
2. 用关节空间轨迹生成器生成速度 / 加速度更平滑的轨迹。
3. 每个轨迹点保持足够的 action / physics 时间。
4. 对难目标补充碰撞检查或换抓取姿态。
5. 再考虑调 JointController 的 kp / kd / effort / velocity 限制。

## 当前推荐判断

对这次视觉抓取 demo，当前证据支持：

```text
视觉目标位置基本正确。
TCP/EEF 坐标契约基本正确。
PyRoKi one-shot IK 对目标 EEF 位姿多数能解出来。
主要失败点在 joint target 执行跟踪层。
发布/保持时间是重要因素，但不是唯一因素。
```

所以下一步不应继续重复验证视觉链条，而应围绕同一个目标做：

```text
timing sweep + 姿态 sweep + 接触/桌板 clearance 检查
```

目标是找出 X2 右臂在该视觉工作区内稳定可执行的 TCP 姿态集合，再把 GraspNet 候选筛选到这个集合附近。
