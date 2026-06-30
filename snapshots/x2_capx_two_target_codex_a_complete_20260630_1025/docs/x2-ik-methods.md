# X2 IK Methods Notes

本文记录当前 X2 中并行存在的两条末端控制路线：

1. 任务空间增量 IK 伺服：反复计算末端位姿误差，再通过雅可比映射成关节命令。
2. 单次全局 IK 求解：一次性求出目标关节角，再用关节空间插值发给关节位置控制器。

这里的目标位姿统一按 CAP-X 包装后的语义理解：

- `move_hand(...)` / `move_hand_joint_ik(...)` 的目标是 EEF link 相对于 world 的位姿。
- `move_tcp(...)` / `move_tcp_joint_ik(...)` 的目标是 TCP / finger-center 相对于 world 的位姿。
- TCP 目标会先经过 `tcp_pose_to_eef_pose()` 转换成 EEF 目标，再进入底层控制。

## Route A: Task-Space Iterative IK Servo

对应当前旧路线：

- Public API: `move_hand(...)`, `move_tcp(...)`
- Low-level implementation: `X2BehaviourLowLevel._move_hand()`
- Controller: OmniGibson `InverseKinematicsController`

核心形式可以理解为任务空间 P 控制：

```text
e_x = [p_target - p_current, orientation_error(R_target, R_current)]
delta_x = K * e_x
delta_q = J(q)^# * delta_x
q_cmd = q_current + delta_q
```

其中：

- `J(q)` 是当前构型下 EEF 的几何雅可比。
- `J(q)^#` 通常是伪逆或阻尼伪逆。
- `K` 是任务空间误差增益。当前 X2 中 `pose_delta_ori` 模式下额外用了：
  - `_ik_pose_delta_pos_gain`
  - `_ik_pose_delta_ori_gain`

它不是先规划一条完整轨迹，也不是一次性求最终关节角。它是在每个仿真 step 根据当前误差重新算一个小的关节更新。

### Strengths

- 适合小范围末端修正。
- 适合视觉伺服、接近目标、微调对齐。
- 每一步都用当前状态重新计算，对小扰动有反馈。
- 不需要单独切到关节级动作接口。

### Weaknesses

- 大范围移动时不稳定，尤其是姿态变化大时。
- 接近奇异点、关节限位、雅可比条件数差时，伪逆会放大误差。
- 姿态误差和位置误差的尺度不同，增益很敏感。
- 当前没有碰撞规划；它只是在局部跟踪目标。
- 如果目标离当前构型较远，局部线性化假设会变差。

### When To Use

更适合：

- 已经在目标附近后的精修。
- 视觉链条给出目标后，做小步接近。
- 抓取前最后几厘米的姿态调整。

不适合：

- 从当前姿态直接跨很大关节距离。
- 大角度重定向 TCP。
- 需要绕障或强约束路径的移动。

## Route B: One-Shot IK + Joint-Space Interpolation

对应当前新增并行路线：

- Public API: `move_hand_joint_ik(...)`, `move_tcp_joint_ik(...)`
- Low-level implementation:
  - `_solve_pyroki_eef_joint_target(...)`
  - `_move_to_joint_positions(...)`
- Controller config: `x2_robotiq85_joint_primitives.yaml`
- Controller type: OmniGibson `JointController`
- IK solver: PyRoKi / JAX least-squares IK

核心形式是先解一个非线性优化问题：

```text
q* = argmin_q
       w_p || FK_pos(q) - p_target ||^2
     + w_R || FK_rot(q) - R_target ||^2
     + limit_constraint(q)
     + rest_cost(q)
```

当前 PyRoKi 求解器中 pose cost 的权重是：

```text
pos_weight = 50.0
ori_weight = 10.0
```

求得 `q*` 后，再做简单关节空间线性插值：

```text
q_cmd(alpha) = q_current + alpha * (q* - q_current)
alpha = 1 / N, 2 / N, ..., 1
```

当前 X2 是双臂模型，PyRoKi 看到 14 个手臂关节。但执行时只更新选中手臂的 7 个关节，另一只手臂保持当前关节位置。

### Strengths

- 对大范围目标比任务空间增量伺服更合理。
- IK 是一次性对完整运动链做非线性求解，不依赖每一步的局部误差小量假设。
- 可以清楚区分：
  - IK 是否能解出目标。
  - 关节控制器是否能跟踪解出的关节目标。
  - 最终 TCP / EEF 位姿是否到位。

### Weaknesses

- 当前只是关节空间插值，不是碰撞规划。
- 插值路径可能穿过碰撞、限位附近或动态不稳定区域。
- 执行质量强依赖关节驱动器参数，例如 Isaac drive `kp` / `kd`。
- 如果 joint controller 跟不上，PyRoKi 即使解得很准，最终末端也会偏。
- 当前 primitive 的 `primitive_ok` 同时受关节误差判据影响；实际是否可用还要看最终 TCP / EEF 位姿误差。

### When To Use

更适合：

- 从一个稳定姿态移动到另一个明确目标姿态。
- 手工任务代码中的中等距离移动。
- 视觉输出目标位姿后，先求一个完整关节目标。
- 需要调试“目标是否可达”和“执行是否跟得上”的场景。

不适合：

- 需要绕障的复杂路径。
- 与物体接触时需要柔顺控制的动作。
- 对轨迹中间状态有严格约束的动作。

## How To Diagnose A Failed Pose

不要只看最终 TCP 误差。至少分三层看：

```text
1. IK solve residual
   solve_fk_pos_error_m
   solve_fk_ori_error_rad

2. Joint execution residual
   joint_final_error_rad
   joint_command_max_delta_rad

3. Final task-space residual
   tcp_error_m
   eef_ori_error_rad
```

判断规则：

```text
IK residual large
  -> 目标可能不可达、目标姿态不适合该构型、URDF / link / frame 设置有问题，或优化没有收敛。

IK residual small, joint residual large
  -> IK 解出来了，但关节控制器没有跟上。优先看 JointController / Isaac drive kp-kd / 插值步长 / settle steps。

IK residual small, joint residual small, final TCP residual large
  -> 优先检查 TCP <-> EEF 转换、目标 frame、link name 或记录方式。

IK residual small, joint residual medium, final TCP acceptable
  -> 这个目标工程上可能可用，但 primitive_ok 可能因为关节误差阈值过严而返回 false。
```

## Current Diagnostic Result

诊断脚本：

```text
scripts/archive/x2_experiments/x2_joint_ik_failure_diagnostics.py
```

输出：

```text
outputs/x2_joint_ik_failure_diagnostics_v1/summary.json
outputs/x2_joint_ik_failure_diagnostics_v1/video_combined.mp4
```

这次故意重跑 v3 中失败的两个目标，并记录每个目标的 solve / execution / final residual。

| Target | Success | TCP error m | EEF ori error rad | IK FK pos error m | IK FK ori error rad | Joint final error rad |
|---|---:|---:|---:|---:|---:|---:|
| anchor_hold_current_quat | true | 0.000004 | 0.000000 | 1.49e-08 | 1.40e-07 | 0.000000 |
| failed_small_up_safe_x_minus15 | false | 0.032435 | 0.232063 | 1.81e-08 | 7.30e-08 | 0.244053 |
| failed_small_forward_left_mixed | false | 0.039618 | 0.251191 | 3.15e-08 | 1.95e-07 | 0.249324 |
| small_back_right_mid | true | 0.006263 | 0.035009 | 4.08e-08 | 7.30e-08 | 0.027783 |
| small_right_low_roll | true | 0.005990 | 0.080791 | 6.81e-08 | 4.06e-07 | 0.088543 |
| small_transport_up_yaw | true | 0.007898 | 0.110764 | 1.08e-07 | 1.19e-07 | 0.052675 |

结论：

- 这些失败点不是“不可达”，也不是“PyRoKi 解不出来”。失败目标的 IK FK 残差是 `1e-8 m` / `1e-7 rad` 量级，说明目标在当前模型里能解到。
- 主要问题是关节位置执行没有跟上。两个失败点的 `joint_final_error_rad` 分别约为 `0.244` 和 `0.249 rad`。
- 通过的几个点同样可能出现 `primitive_ok=false`，因为内部关节误差阈值比最终 TCP 成功判据更严格；但它们最终 TCP 误差在 `1-8 mm`，姿态误差在 `0.035-0.111 rad`，工程上是可接受的。

## Current Practical Recommendation

短期：

- 保留旧 task-space IK route，用于小范围末端修正。
- 使用 one-shot IK + joint interpolation route 作为手工任务主路线。
- 对任务点做离线白名单测试，记录稳定 TCP 姿态集合。
- 调整 joint controller 的 drive `kp` / `kd`，目标是降低 `joint_final_error_rad`。

中期：

- 给 one-shot IK route 增加更清晰的返回结构，不只返回 bool。
- 将 `primitive_ok` 拆成：
  - `ik_solved`
  - `joint_tracked`
  - `eef_reached`
  - `tcp_reached`
- 对关节插值加入速度 / 加速度限制，而不仅是线性插值。

长期：

- 如果需要可靠绕障和复杂任务，应在 one-shot IK 之后接入轨迹优化或运动规划。
- 规划可暂时不考虑碰撞，但接口上应该预留 collision-aware planner 的位置。
