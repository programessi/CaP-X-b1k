# X2 接入 CaP-X/BEHAVIOR 汇报讲稿

日期：2026-07-02

## 建议汇报主线

建议不要按“我改了哪些文件”来讲，而是按工程闭环讲：

1. 我从 CaP-X 出发，目标是让自己的 X2 机器人进入 BEHAVIOR/CaP-X 任务体系。
2. 原始 CaP-X third_party BEHAVIOR 不适配当前 5090 环境，所以我先解决仿真底座。
3. 仿真底座通了以后，又补齐自定义机器人导入、X2 夹爪安装、X2 任务 primitive。
4. 然后把 X2 接进 CaP-X 的 LLM 代码注入模式，让 LLM 只调用高层任务 API。
5. 最后在 ASPIRE 出来后，加入 trace、失败分类和 candidate search，使任务可以迭代调试。

一句话总结：

> 这项工作不是单个抓取 demo，而是把 X2 从环境适配、机器人建模、动作/视觉原语、CaP-X 代码注入，到 ASPIRE-like 调试闭环打通了一遍。

## 3 分钟版

### 1. 背景和问题

最初目标是在 CaP-X 的 BEHAVIOR 环境里导入 X2 机器人。但很快发现 CaP-X 自带的 BEHAVIOR 包不适合当前电脑的 5090 显卡配置。

因此我转向独立安装官网 BEHAVIOR，并适配 Isaac Sim 5.1。这个环境能适配 5090，但新的 BEHAVIOR 版本在自定义机器人导入 pipeline 上有问题。我修了这个问题，对官方 main 和修改后的 main 都做了烟测，并提交了修复分支。

### 2. 让 X2 具备做任务的能力

X2 原始机器人模型不能直接完成桌面抓取任务，因为缺少可用夹爪。我参考官方 Franka + 85 型欠驱动夹爪的例子，写了 X2 安装夹爪脚本，并注册了带爪和不带爪两个版本。

这一步之后，X2 才从“能加载”变成“能干活”。

### 3. 接入 CaP-X

我形成了 `CaP-X-b1k` 分支。这里的关键不是让 LLM 写一大段仿真控制脚本，而是让环境和任务 setup 留在 config/env 层，LLM 只调用高层 primitive。

当前暴露给任务代码的核心接口是：

```python
pick_and_place_visual_object(...)
```

内部会完成视觉、抓取位姿、障碍物估计、轨迹接近、关节 IK、夹爪闭合、转运、放置。

### 4. 当前结果

已经跑通：

- X2 CaP-X two-target pick-place。
- two-object blue cube pick-place。
- RGB-D visual route：用视觉分割、深度点云和 Contact-GraspNet 产生抓取位姿。
- ASPIRE-lite validation：3 个验证场景全部成功。

最新指标：

```text
successes=3/3
avg_before_close_tcp_error_m=0.0127 m
avg_before_close_ori_error_rad=0.0421 rad
avg_place_error_m=0.0215 m
```

## 8 分钟版

### 第 1 页：目标

我的目标是把自研 X2 机器人导入 CaP-X/BEHAVIOR 体系，使它能在 CaP-X 的代码注入框架下完成操作任务。

这里有三个层次：

- 底层：X2 能在 BEHAVIOR/Isaac Sim 5.1 正确加载和控制。
- 中层：X2 有夹爪、视觉、IK、规划、抓取和放置能力。
- 上层：CaP-X / LLM 不需要关心场景搭建，只调用任务级 primitive。

### 第 2 页：为什么先做 BEHAVIOR/Isaac 适配

CaP-X 自带的 BEHAVIOR third_party 包不适配当前 5090 环境。独立安装官网 BEHAVIOR 后，Isaac Sim 5.1 可以运行，但自定义机器人导入 pipeline 有问题。

我做了：

- 修复自定义机器人导入链路。
- 对官方 main 做烟测，确认不是环境本身坏了。
- 对修改后的 main 做烟测，确认补丁有效。
- 提交了修复分支。

本地提交：

```text
0f6705681 Fix custom robot import for Isaac Sim 5.1
```

### 第 3 页：X2 机器人补齐任务能力

X2 原始模型只解决“机器人存在”的问题，不解决“机器人能抓东西”的问题。为了做 pick-place，我参考官方 Franka 例子，给 X2 增加 85 型欠驱动夹爪，并注册了两个版本：

- X2 without gripper
- X2 with gripper

这一步是后续所有任务的基础。

### 第 4 页：CaP-X-b1k 的设计原则

我没有把仿真 setup 写进 LLM 生成代码里，而是保持 CaP-X 原本的思路：

- 场景、机器人、物体、相机、任务目标放在 config/env 层。
- LLM 只拿到少量高层 API。
- 低层视觉和动作细节由 X2 API 内部封装。

核心 task-level primitive：

```python
pick_and_place_visual_object(...)
```

### 第 5 页：视觉到动作的链路

当前 RGB-D route 是：

```text
OWL-ViT 目标检测
-> SAM2 目标分割
-> RGB-D 点云估计目标和桌面障碍物
-> Contact-GraspNet 生成抓取 TCP 位姿
-> PyRoKi / IK 规划接近
-> joint IK 执行
-> gripper close
-> lift / transfer / place / release
```

重要坐标契约：

```text
视觉输出：T_world_tcp
动作输入：T_world_tcp
```

LLM 不需要自己处理 TCP/EEF 转换。

### 第 6 页：ASPIRE-lite 做了什么

ASPIRE 之后，我没有说已经完整复现论文，而是先做了工程上有价值的 ASPIRE-like 子集：

- trace bundle：记录代码、primitive 调用、视觉产物、运动 waypoint、视频。
- failure taxonomy：把失败分成检测失败、分割失败、抓取不可达、接近失败、没夹住、放置失败等。
- skill library：把有效策略记录下来。
- candidate search：对高层 primitive 参数做受控搜索。

这让失败调试从“看视频猜原因”变成“有 trace 和指标支撑”。

### 第 7 页：一个具体修复案例

最近的失败点在 place 阶段：

- 机械臂抓起物体后，转运过程中其实已经到达 `place_pre_tcp_pose`。
- 原代码又重复发一次到同一个 place-pre pose 的 IK move。
- 这个冗余动作偶发导致末端跳走约 10 cm。

修复：

- 到位后跳过冗余 place-pre IK move。
- 使用 `place_orientation_source=post_lift_current`，保持抓起后已经稳定的 TCP 姿态去放置。

验证结果：

```text
validation successes=3/3
avg_before_close_tcp_error_m=0.0127 m
avg_place_error_m=0.0215 m
```

### 第 8 页：当前边界

需要讲清楚：

- CaP-X 仿真任务链路已经打通。
- RGB-D route 已经比早期 oracle route 更接近 real。
- ASPIRE-lite 的 trace/failure/candidate/validation 骨架已经有了。

但不要说过头：

- 还不是完整 ASPIRE 论文级复现。
- 还没有完成真实机器人闭环。
- 仍有部分评估信号来自仿真，例如 reward、task_completed、place_error。

### 第 9 页：下一步

建议下一步分三条线：

1. 固化当前稳定基线：tag、README、视频证据。
2. 补一个严格 ASPIRE-style 单报告：多候选 debug + controlled failure + best candidate + 3-seed validation。
3. 推进 x2-agent-lab 实物部署：相机标定、TCP/EEF 标定、ROS trajectory/action、真实安全策略。

## 建议现场播放的视频

### 1. CaP-X high-level primitive 任务

```text
docs/presentations/assets/x2_two_target_codex_a.mp4
```

讲法：这个视频说明 X2 已经在 CaP-X 高层 primitive 下完成任务，不是手写 env setup 脚本。

### 2. RGB-D visual route

```text
docs/presentations/assets/x2_rgbd_codex_a.mp4
```

讲法：这个视频说明视觉、抓取位姿、障碍物估计和动作执行已经串起来。

### 3. ASPIRE-lite validation

```text
docs/presentations/assets/x2_aspire_validation.mp4
```

讲法：这个视频说明经过 trace/candidate 调试后的策略可以在验证场景里稳定完成。

## 关键材料

GitHub Pages 入口：

```text
docs/index.html
```

网页汇报：

```text
docs/presentations/x2-capx-leadership-report-20260702.html
```

状态文档：

```text
docs/x2-capx-integration-status.md
docs/x2-aspire-lite-minimal-replication.md
docs/x2-rgbd-visual-obstacle-upgrade-20260630.md
```

代码入口：

```text
capx/integrations/x2/control.py
capx/integrations/x2/aspire.py
scripts/run_x2_aspire_rgbd_candidate_search.py
```
