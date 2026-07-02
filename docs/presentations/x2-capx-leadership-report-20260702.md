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
docs/presentations/assets/x2_grasp_pose_world_frame.png
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
- 抓取姿态图由 `grasp_summary.json` 重建，展示 world-frame 下的 object bbox、precontact TCP、grasp TCP、TCP 三轴和 approach axis。由于当次运行没有保存相机内参/外参，该图不是 RGB 图像投影。
- 机器人闭合夹爪前的 TCP 误差约 1.1 cm，说明动作执行基本到达视觉生成的抓取位姿。
- RGB-D route 已经减少早期 oracle obstacle route 的仿真真值依赖，但 reward、task_completed、place_error 等评估信号仍来自仿真。

## 实验 3：ASPIRE-lite 修复策略验证

材料：三条视频均为本轮 validation 的全局视角，便于观察桌面、方块和机械臂整体运动。

```text
docs/presentations/assets/x2_aspire_val_nominal_shift.mp4
docs/presentations/assets/x2_aspire_val_centered.mp4
docs/presentations/assets/x2_aspire_val_right_shift.mp4
```

本轮输出：

```text
outputs/x2_aspire_parameter_loop/full_strict_gate_20260702_gpu/
```

任务设置：

- 同一 RGB-D pick-place 任务使用 3 个 debug seed 和 3 个 held-out validation seed。
- seed 表示任务实例不同：目标方块、干扰物和目标放置点有小范围扰动，但仍在 X2 右臂可达区域内。
- baseline 和候选使用同一任务族；validation seed 不参与候选生成和 debug 选择。

候选搜索过程：

```text
controlled_failure_strict_reach_gate baseline:
  debug success = 1/3
  failures = preclose_pose_not_reached, preclose_pose_not_reached

LLM-generated candidates:
  llm_try_more_ranked_grasps:      debug 2/3
  llm_slow_preclose_align:         debug 3/3
  llm_enable_guarded_reobserve:    debug 1/3

selected best:
  llm_slow_preclose_align
```

最终验证指标：

```text
successes = 3/3
avg_before_close_tcp_error_m = 0.0026426902
avg_before_close_ori_error_rad = 0.0072086684
avg_place_error_m = 0.0284396451
trace_bundles = 3
videos = 3
rgbd_obstacles_sim_truth = False
```

解释口径：

- 该结果属于受控候选搜索中的工程验证，不能等同于大样本统计评测。
- baseline 的两个失败都发生在闭爪前，failure taxonomy 为 `preclose_pose_not_reached`。
- trace 显示 IK 解算误差接近 0，目标并非明显不可达；失败主要来自执行跟踪和过严的 final reach gate。
- LLM 根据失败报告生成 3 个受限参数候选。候选只改高层 primitive 白名单参数，不能改底层控制器、视觉模型、IK 或 PyRoKi。
- 最终修复策略 `llm_slow_preclose_align` 将接近/插入动作放慢，增加 hold 和 fine-align，并放宽 final gate，使末端在可抓容差内稳定闭爪。
- `llm_enable_guarded_reobserve` 是一个反例：它单独强化 reobserve，但在两个 debug seed 上变成 `object_not_in_hand_after_close`，说明 LLM 建议必须经过执行筛选。
- 修复后在三个 held-out validation seed 上均完成任务。

### 参数级 ASPIRE-lite 的作用示例

当前参数级修复的效果是：在不改底层 X2 控制器、不改视觉模型、不改 PyRoKi/IK 实现的前提下，把失败记录转化为高层 primitive 的候选参数，并通过多 seed 执行筛选出更稳定的一组策略。

以本轮 baseline 的 `preclose_pose_not_reached` 为例，失败 trace 会记录：

```text
before_close_tcp_error_m
before_close_ori_error_rad
preclose_joint_ok
preclose_joint_final_error_rad
preclose_ik_solve_fk_pos_error_m
preclose_ik_solve_fk_ori_error_rad
```

如果 IK solve FK 误差很小，但闭合前 TCP / 姿态误差仍超过严格阈值，failure report 会将主失败归类为 `preclose_pose_not_reached`，并给出类似修复方向：

```text
slow_preclose_align
fine_align_retries
relax_final_gate
```

随后 candidate search 不改程序结构，只改变高层 primitive 参数，例如：

```text
candidate_indices:              1        -> 1,2
final_tcp_threshold:             0.002    -> 0.035
final_ori_threshold:             0.010    -> 0.35
insert_max_joint_step:           faster   -> 0.006
fine_align_retries:              low      -> 2
hold / insert_hold_steps:        lower    -> 4 / 10
place_descent_waypoints:         1        -> 4
```

这些候选会先在 debug seed 中比较，再把较好的候选放到 held-out validation seed 上验证。当前记录中的候选过程从 baseline 1/3 推进到 `llm_slow_preclose_align` 的 debug 3/3 和 validation 3/3。这个结果说明参数级修复已经能完成可复验的失败归因和策略筛选，但还不是完整 ASPIRE 的代码级 skill 自修改。

## ASPIRE-lite 当前实现

当前实现覆盖以下组件：

- trace bundle：保存生成代码、primitive 调用、视觉产物、抓取候选、运动 waypoint、误差指标、视频。
- failure taxonomy：将失败归类到检测、分割、深度点云、抓取不可达、preclose 未到达、未夹住、放置误差等类别。
- LLM candidate proposer：读取历史 `candidate_search_report.json`、failure taxonomy 和 skill library，让 LLM 只生成白名单参数内的候选策略。
- skill candidate search：围绕高层 primitive 参数搜索候选策略，包括 grasp candidate、TCP offset、reobserve、place descent、place orientation。
- validation seed：将候选策略放到不同目标/干扰物布局下验证。
- evidence report：把结果写成 `candidate_search_report.json`、`findings.md`、视频和 trace bundle。

### 已实现功能

当前 ASPIRE-lite 不只是保存视频，而是把一次 CaP-X 任务运行拆成可检索的数据记录：

- 运行代码和 primitive 调用参数。
- RGB、检测框、SAM2 mask、RGB-D 点云估计和 Contact-GraspNet 候选。
- 抓取前 TCP 误差、姿态误差、place 误差。
- failure report，包括 `primary_failure` 和 `suggested_repair_tags`。
- candidate search report，包括 debug seed、validation seed、候选策略得分和最终选择。
- LLM proposal，包括 prompt、raw response、validated `candidates.json`。

### 验证方式

验证流程分三步：

1. 从历史 report 中抽取失败类型、指标和已有 skill evidence。
2. 由 LLM 生成参数级 candidate，候选只能使用白名单参数，不能改底层控制代码。
3. 在 debug seed 上运行多个候选策略，故意保留失败和半成功样本。
4. 根据 failure taxonomy 判断主要失败原因，例如没夹住、preclose 没到、place 阶段跳变。
5. 将修复后的候选策略放到 held-out validation seed 上运行，并要求同时满足任务完成、误差指标、trace bundle 和视频记录。

### 已验证效果

当前记录到的候选搜索过程：

```text
controlled_failure_strict_reach_gate baseline: 1/3 debug
llm_try_more_ranked_grasps:                   2/3 debug
llm_slow_preclose_align:                      3/3 debug, selected best
llm_enable_guarded_reobserve:                 1/3 debug
llm_slow_preclose_align:                      3/3 held-out validation
```

最终验证：

```text
successes = 3/3
avg_before_close_tcp_error_m = 0.0026 m
avg_before_close_ori_error_rad = 0.0072 rad
avg_place_error_m = 0.0284 m
videos = 3
trace_bundles = 3
```

新增 LLM 自动候选生成和真实 execute 验证：

```text
script:
  scripts/propose_x2_aspire_skill_candidates.py
  scripts/run_x2_aspire_parameter_loop.py

input report:
  outputs/x2_aspire_parameter_loop/full_strict_gate_20260702_gpu/baseline/candidate_search_report.json

Codex CLI generated candidates:
  llm_slow_preclose_align
  llm_try_more_ranked_grasps
  llm_enable_guarded_reobserve

candidate-search execute:
  3 debug seeds
  3 validation seeds
  audit_ok = true
```

真实 execute 已完成。服务证据保存在 `service_status.json`：SAM2、OWL-ViT、Contact-GraspNet 使用 cuda 服务，PyRoKi 为进程内调用。

### 与原 CaP-X 的差异

原 CaP-X 的重点是让 LLM 在环境中调用 primitive 完成任务。当前扩展保留这一点，但增加了面向 skill 自进化的数据层：

- 原 CaP-X 主要关心一次任务是否完成；当前系统记录任务为什么成功或失败。
- 原 CaP-X 失败后主要靠人工看视频和日志；当前系统把失败写成 failure taxonomy 和 repair tag。
- 原 CaP-X 通常需要人工修改代码或参数；当前系统把修改约束成 skill candidate，并在 debug/validation seed 上比较。
- 原 CaP-X 的证据以 reward、视频为主；当前系统同时保存视觉中间结果、抓取候选、TCP 误差、place 误差、candidate report 和多 seed 视频。

完整 ASPIRE 论文级自动技能发现闭环仍属于后续工作；当前阶段完成的是可复现实验骨架和工程证据链。

## 后续工作

1. 固化当前稳定版本：tag、README、Pages 汇报和视频证据保持同步。
2. 增加更系统的 ASPIRE-style 单报告：同一报告内包含多候选 debug、controlled failure、best candidate、3-seed validation。
3. 扩展任务集：多物体、多目标、多摆放区域和更多失败恢复策略。
4. 推进 x2-agent-lab 实物桥接：相机标定、TCP/EEF 标定、ROS trajectory/action、安全停止。
