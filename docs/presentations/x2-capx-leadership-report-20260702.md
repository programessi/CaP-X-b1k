# 面向数据自进化的 X2-CaP-X 技能体系汇报提纲

日期：2026-07-02

## 汇报主线

本阶段工作从 CaP-X 出发，目标是让 X2 进入 BEHAVIOR/CaP-X 任务体系，并进一步形成可积累、可诊断、可检索、可修复的 skill 数据闭环。

主线按六个阶段展开：

1. 解决本机 RTX 5090 与 CaP-X bundled BEHAVIOR 的环境不匹配问题。
2. 独立安装官网 BEHAVIOR 并适配 Isaac Sim 5.1。
3. 修复该版本 BEHAVIOR 的自定义机器人导入 pipeline，并完成 main 与 patched main 的烟测。
4. 为 X2 增加 85 型欠驱动夹爪，注册带爪/不带爪版本，使其具备桌面操作条件。
5. 在 CaP-X-b1k 中封装 X2 视觉与动作 primitive，让任务代码通过高层 API 调用 X2。
6. 引入 ASPIRE-lite 的 trace、failure taxonomy、candidate search 和 validation seed，为后续数据驱动的 skill 检索与扩展做准备。

## 当前系统状态

X2 已能在 BEHAVIOR/Isaac Sim 5.1 中加载、运动、闭合夹爪，并通过 CaP-X 高层 primitive 完成桌面 pick-place。

核心任务级 primitive：

```python
pick_and_place_visual_object(
    object_name="blue cube",
    target_name="right target",
    obstacle_source="rgbd_visual",
    place_offset_source="visual_grasp_pose",
    reobserve_at_precontact=True,
)
```

该接口内部串联：

```text
OWL-ViT target detection
-> SAM2 target segmentation
-> RGB-D point cloud object/table estimation
-> Contact-GraspNet TCP grasp generation
-> PyRoKi / joint IK approach and transfer
-> gripper close / lift / place / release
-> trace / metrics / failure report
```

关键坐标契约：

```text
视觉输出：T_world_tcp
动作输入：T_world_tcp
```

## 实验 1：X2 夹爪与低级动作验证

材料：

```text
docs/presentations/assets/x2_gripper_motion_primitives.mp4
```

展示内容：

- X2 带 85 型欠驱动夹爪版本可以在 BEHAVIOR/Isaac Sim 中加载。
- 机械臂可以执行末端位姿变化。
- 夹爪可以闭合和打开。

该实验支撑机器人接入和动作 primitive 的基础可用性。

## 实验 2：RGB-D 视觉 primitive 到机器人执行

材料：

```text
docs/presentations/assets/x2_visual_rgb.png
docs/presentations/assets/x2_visual_detection_overlay.png
docs/presentations/assets/x2_visual_sam2_mask_overlay.png
docs/presentations/assets/x2_rgbd_codex_a.mp4
```

记录到的关键数据：

```text
pose_estimate.meaning = T_world_object estimated from SAM2 mask and RGB-D depth
object_position_world = [0.3046, -0.0647, 0.9213] m
graspnet_candidate_count = 48
grasp_tcp_position_world = [0.3068, -0.0656, 0.9240] m
grasp_tcp_quat_xyzw_world = [0.4899, 0.6940, 0.3302, -0.4114]
before_close_tcp_error_m = 0.0111 m
before_close_ori_error_rad = 0.0340 rad
```

解释口径：

- 视觉链路不只输出 mask，还输出可接入动作 primitive 的 TCP 抓取位姿。
- 机器人闭合夹爪前的 TCP 误差约 1.1 cm，说明动作执行基本到达视觉生成的抓取位姿。
- RGB-D route 已经减少早期 oracle obstacle route 的仿真真值依赖，但 reward、task_completed、place_error 等评估信号仍来自仿真。

## 实验 3：ASPIRE-lite 修复策略验证

材料：

```text
docs/presentations/assets/x2_aspire_val_nominal_shift.mp4
docs/presentations/assets/x2_aspire_val_centered.mp4
docs/presentations/assets/x2_aspire_val_right_shift.mp4
```

三个 validation seed：

```text
val_nominal_shift:
  target    = [0.318, -0.078, 0.921]
  distractor= [0.225,  0.035, 0.921]

val_centered:
  target    = [0.305, -0.065, 0.921]
  distractor= [0.225,  0.045, 0.921]

val_right_shift:
  target    = [0.342, -0.083, 0.921]
  distractor= [0.225,  0.025, 0.921]
```

候选搜索过程中的成功率变化：

```text
controlled_failure_fast_no_reobserve: 0/1
repair_validated_relaxed_preclose_v2: 1/2
stable_rgbd_v1: 2/4
repair_place_keep_lift_orientation_v1: 3/3 validation
```

最终验证指标：

```text
successes = 3/3
avg_before_close_tcp_error_m = 0.0127399412
avg_before_close_ori_error_rad = 0.0420904692
avg_place_error_m = 0.0215318505
trace_bundles = 3
videos = 3
rgbd_obstacles_sim_truth = False
```

解释口径：

- 该结果属于受控候选搜索中的工程验证，不能等同于大样本统计评测。
- 早期候选暴露出 `object_not_in_hand_after_close`、`preclose_pose_not_reached`、place 阶段跳变等问题。
- 最终修复策略使用 `post_lift_current` 作为 place orientation，并在 place-pre 已到位时跳过冗余 IK move。
- 修复后在三个 held-out validation seed 上均完成任务。

## ASPIRE-lite 当前实现

当前实现覆盖以下组件：

- trace bundle：保存生成代码、primitive 调用、视觉产物、抓取候选、运动 waypoint、误差指标、视频。
- failure taxonomy：将失败归类到检测、分割、深度点云、抓取不可达、preclose 未到达、未夹住、放置误差等类别。
- skill candidate search：围绕高层 primitive 参数搜索候选策略，包括 grasp candidate、TCP offset、reobserve、place descent、place orientation。
- validation seed：将候选策略放到不同目标/干扰物布局下验证。
- evidence report：把结果写成 `candidate_search_report.json`、`findings.md`、视频和 trace bundle。

完整 ASPIRE 论文级自动技能发现闭环仍属于后续工作；当前阶段完成的是可复现实验骨架和工程证据链。

## 后续工作

1. 固化当前稳定版本：tag、README、Pages 汇报和视频证据保持同步。
2. 增加更系统的 ASPIRE-style 单报告：同一报告内包含多候选 debug、controlled failure、best candidate、3-seed validation。
3. 扩展任务集：多物体、多目标、多摆放区域和更多失败恢复策略。
4. 推进 x2-agent-lab 实物桥接：相机标定、TCP/EEF 标定、ROS trajectory/action、安全停止。
