# X2 Visual Grasp, RGB-D Obstacles, and Collision-Aware Planning

本文用当前 X2 RGB-D visual pick-place 路线解释一个完整问题：

```text
相机看到物体
-> 检测 / 分割 / 深度反投影
-> 生成抓取 TCP 位姿
-> 从 RGB-D 估计物体和桌面障碍物
-> PyRoKi 用这些障碍物规划到 precontact
-> 短距离插入、闭爪、转运、放置
```

重点是：障碍物最后被处理成什么数据，以及这些数据如何影响末端轨迹。

## 1. 当前路线的定位

这不是完整 SLAM，也不是把整张深度图变成稠密碰撞地图。当前 X2 tabletop 任务使用一个更简单、工程上可控的表示：

```text
目标物体     -> 一个 world-frame box
桌面支撑面   -> 一个 world-frame box
```

这两个 box 再传给 PyRoKi。PyRoKi 不直接规划一条手工写死的末端直线，而是在关节空间优化一条轨迹。末端轨迹是这条关节轨迹通过正运动学算出来的结果。

## 2. 坐标系和符号

本文使用如下记号：

```text
W: world frame
C: camera frame
B: robot base frame / PyRoKi base frame
E: EEF frame, X2 的末端 link frame
T: TCP frame, 当前近似为 gripper finger center / grasp point frame
O: object frame
```

齐次变换记号：

```text
^A T_B
```

表示把 B 系下的点变换到 A 系：

```text
p_A = ^A T_B p_B
```

展开为旋转和平移：

```text
p_A = ^A R_B p_B + ^A t_B
```

当前视觉 / 动作契约是：

```text
视觉输出抓取目标:      ^W T_T, 即 TCP 相对于 world 的位姿
动作 primitive 输入:   ^W T_T
内部运动控制目标:      ^W T_E，由 tcp_pose_to_eef_pose() 转换得到
障碍物 box:            world frame 下的 center / extent / orientation
放置目标:              world frame 下的 object center [x, y, z]
```

## 3. 从 RGB-D 到 world 点云

相机内参记为：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

像素坐标为：

```text
u = [u, v, 1]^T
```

深度图给出该像素的深度：

```text
d(u, v)
```

如果深度是沿相机光轴的 metric depth，那么相机系点为：

```text
p_C(u, v) = d(u, v) K^{-1} [u, v, 1]^T
```

展开就是：

```text
x_C = (u - cx) d / fx
y_C = (v - cy) d / fy
z_C = d
```

再用相机外参变到 world：

```text
p_W(u, v) = ^W R_C p_C(u, v) + ^W t_C
```

在代码里，`get_rgbd_visual_tabletop_obstacles()` 从 visual plan 里拿到：

```text
mask
depth
camera.intrinsic_matrix
camera.position_world
camera.quat_xyzw_world
```

然后通过 `backproject_mask_to_world()` 做上述反投影。

## 4. 目标物体 box 怎么生成

目标物体的 mask 来自：

```text
OWL-ViT detection box -> SAM2 mask
```

mask 内的有效深度点反投影到 world 后，得到目标物体可见点云：

```text
P_obj = { p_W(u, v) | M_obj(u, v) = 1, d(u, v) is valid }
```

如果有深度窗口，会再过滤一次：

```text
P_obj = { p in P_obj | |d(p) - d_expected| <= depth_window }
```

再按 workspace bounds 过滤：

```text
x_min <= p_x <= x_max
y_min <= p_y <= y_max
z_min <= p_z <= z_max
```

为了避免深度离群点把 box 拉得过大，不直接用 min / max，而是用分位数。对每个坐标轴 `j in {x, y, z}`：

```text
lo_j = percentile_2(P_obj[:, j])
hi_j = percentile_98(P_obj[:, j])
```

可见点云中心和尺寸：

```text
c_visible = (lo + hi) / 2
e_visible = hi - lo
```

当前 object box 的中心不是直接用 `c_visible`，而是用视觉估计的物体中心：

```text
c_obj = visual_pose_estimate.position_world
```

尺寸取可见尺寸、最小尺寸、视觉 bbox extent 的最大值，并加 margin：

```text
e_obj = max(e_visible, e_min, e_pose_extent) + 2 m_obj
```

其中 `m_obj` 是每个方向的安全 margin。最后得到：

```python
{
    "type": "box",
    "name": object_name,
    "position": c_obj,
    "extent": e_obj,
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    "source": "rgbd_object_mask_aabb",
}
```

注意这里的 `extent` 是完整尺寸，不是半长宽高。

这次成功 run 的实际目标物体 obstacle 大约是：

```text
source = rgbd_object_mask_aabb
raw_point_count = 8208
center = [0.3195, -0.0797, 0.9214]
extent = [0.0629, 0.0640, 0.0637]
```

蓝方块真实尺寸约 4 cm，这里变成约 6.3 cm，是因为加了 margin，并且要容忍视觉误差和控制误差。

## 5. 桌面 box 怎么生成

桌面不是从 sim table AABB 读出来的。它来自 RGB-D 里“非目标物体”的有效深度点。

先取所有有效深度点，排除目标 mask：

```text
M_table_candidate(u, v) = valid_depth(u, v) AND NOT M_obj(u, v)
```

反投影得到候选点云：

```text
P_all = { p_W(u, v) | M_table_candidate(u, v) = 1 }
```

再做 workspace 过滤。

因为当前任务是假设桌面支撑目标物体，所以桌面应该在物体中心下方附近。设视觉估计的物体中心是：

```text
c_obj = [x_obj, y_obj, z_obj]^T
```

保留物体下方一段高度范围内的支撑候选点：

```text
P_support = { p in P_all | z_obj - 0.20 <= p_z <= z_obj - 0.004 }
```

用高分位数估计桌面高度：

```text
z_table = percentile_92(P_support[:, z])
```

再保留接近这个高度的平面点：

```text
P_plane = { p in P_support | |p_z - z_table| <= table_plane_tolerance }
```

如果点太少，会退化为取支撑候选点的上层部分。

桌面 XY 范围也用分位数：

```text
xy_lo = percentile_2(P_plane[:, x:y])
xy_hi = percentile_98(P_plane[:, x:y])
xy_center = (xy_lo + xy_hi) / 2
xy_extent_visible = xy_hi - xy_lo
```

为了避免可见桌面太窄，XY 尺寸有最小值：

```text
xy_extent = max(xy_extent_visible, table_min_extent_xy)
```

桌面 box 尺寸：

```text
e_table = [
    xy_extent_x + 2 table_margin_xy,
    xy_extent_y + 2 table_margin_xy,
    table_thickness + 2 table_margin_z
]
```

桌面 box 中心：

```text
c_table = [
    xy_center_x,
    xy_center_y,
    z_table - 0.5 table_thickness
]
```

最后得到：

```python
{
    "type": "box",
    "name": table_name,
    "position": c_table,
    "extent": e_table,
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    "source": "rgbd_table_plane_aabb",
}
```

这次成功 run 的实际桌面 obstacle 大约是：

```text
source = rgbd_table_plane_aabb
plane_point_count = 44592
table_z_world = 0.9014
center = [0.3424, 0.0013, 0.8894]
extent = [0.2000, 0.2204, 0.0360]
```

## 6. 障碍物数据的完整 schema

当前传给 PyRoKi 的 obstacle 是一个 list：

```python
obstacles_world = [
    {
        "type": "box",
        "name": "x2_pick_place_blue_cube",
        "position": [0.3195, -0.0797, 0.9214],
        "extent": [0.0629, 0.0640, 0.0637],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "source": "rgbd_object_mask_aabb",
    },
    {
        "type": "box",
        "name": "x2_pick_place_table",
        "position": [0.3424, 0.0013, 0.8894],
        "extent": [0.2000, 0.2204, 0.0360],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "source": "rgbd_table_plane_aabb",
    },
]
```

这些值的语义是：

```text
position:  box center in world frame
extent:    full box size [dx, dy, dz]
quat_xyzw: box orientation in world frame
source:    box 的生成方式，方便审计是否读了 sim truth
```

这份数据不是末端轨迹本身，而是轨迹优化中的 collision object。

## 7. 从 grasp pose 到 precontact pose

视觉抓取规划最终输出的是 X2 可执行 TCP 位姿：

```text
^W T_T^grasp = (p_T^grasp, R_T^grasp)
```

其中：

```text
p_T^grasp in R^3
R_T^grasp in SO(3)
```

还会得到 TCP approach axis：

```text
a_T^W
```

这是 world frame 下的单位向量，表示从 precontact 到 grasp 的插入方向。

precontact 距离设为：

```text
d_pre = 0.08 m
```

precontact TCP 位姿：

```text
p_T^pre = p_T^grasp - d_pre a_T^W
R_T^pre = R_T^grasp
```

也就是：姿态不变，沿抓取轴反方向退 8 cm。PyRoKi 主要负责从当前状态避障到这个 precontact 位姿。

## 8. TCP 和 EEF 的转换

X2 视觉抓取语义使用 TCP，但是机器人运动学链通常以 EEF link 为目标。设 TCP 在 EEF 下的固定偏移为：

```text
^E T_T
```

已知目标 TCP 位姿：

```text
^W T_T
```

则目标 EEF 位姿是：

```text
^W T_E = ^W T_T (^E T_T)^(-1)
```

代码里由：

```text
tcp_pose_to_eef_pose()
```

完成这个转换。

这一步很重要。如果视觉输出的是 TCP pose，而动作原语误以为它是 EEF pose，末端会系统性偏移，偏移量大约就是 finger center 到 EEF link 的距离。

## 9. PyRoKi 规划的变量是什么

PyRoKi 不是直接优化一串 TCP 点：

```text
p_T(0), p_T(1), ..., p_T(N)
```

而是优化一串关节角：

```text
Q = [q_0, q_1, ..., q_N]
```

其中每个：

```text
q_i in R^n
```

对 X2 来说，PyRoKi 看到双臂 14 个手臂关节，但执行时只更新选中手臂的 7 个关节，另一只手保持当前值。

末端位姿由正运动学给出：

```text
^B T_E(q_i) = FK_E(q_i)
```

因此末端轨迹是：

```text
{ ^W T_E(q_i) } for i = 0..N
```

它是关节轨迹的结果，不是直接被手工指定的直线。

## 10. 终点 IK

规划前先解一个终点 IK。目标是让 EEF 到达由 TCP 转换得到的目标：

```text
^B T_E^target
```

可以抽象成：

```text
q_goal = argmin_q
    w_p || p_E(q) - p_E^target ||^2
  + w_R || Log(R_E^target^-1 R_E(q)) ||^2
  + C_limit(q)
  + C_rest(q)
```

其中：

```text
p_E(q):            FK 得到的 EEF 位置
R_E(q):            FK 得到的 EEF 姿态
Log(.):            SO(3) 旋转误差映射到三维向量
C_limit(q):        关节限位代价
C_rest(q):         偏离默认姿态的轻微正则项
```

如果 IK 终点本身解不出来，后面的轨迹优化就没有可靠终点。

## 11. 轨迹优化的目标函数

有了当前关节角：

```text
q_start
```

和 IK 终点：

```text
q_goal
```

PyRoKi 初始化一条线性关节插值：

```text
q_i^init = (1 - alpha_i) q_start + alpha_i q_goal
alpha_i = i / N
```

然后优化整条轨迹：

```text
Q* = argmin_Q
    C_start(Q)
  + C_goal(Q)
  + C_limit(Q)
  + C_self_collision(Q)
  + C_world_collision(Q)
  + C_smooth(Q)
```

各项含义如下。

起点约束：

```text
C_start = lambda_s || q_0 - q_start ||^2
```

终点约束：

```text
C_goal = lambda_g || q_N - q_goal ||^2
```

关节限位：

```text
C_limit = sum_i phi_limit(q_i)
```

自碰撞：

```text
C_self_collision = sum_i phi_self(q_i)
```

平滑项：

```text
C_smooth = sum_i || q_i - q_{i-1} ||^2
```

当前 PyRoKi snippets 里还对 swept collision 做了世界障碍物代价。直观写法是：

```text
C_world_collision =
    sum_i sum_k sum_m phi( distance( swept_link_k(q_i, q_{i+1}), obstacle_m ) )
```

其中：

```text
swept_link_k(q_i, q_{i+1})
```

表示机器人第 `k` 个碰撞 capsule 从 `q_i` 到 `q_{i+1}` 这一步扫过的空间。

`obstacle_m` 就是前面 RGB-D 估计出来的 object box 或 table box。

如果扫过的 capsule 离 box 很远，碰撞代价接近 0；如果进入安全距离甚至穿透 box，代价变大。

可以把碰撞惩罚函数理解成：

```text
phi(d) = max(0, margin - d)^2
```

其中：

```text
d > 0: 两个几何体之间有间隙
d = 0: 刚好接触
d < 0: 发生穿透
margin: 希望保持的安全距离
```

代码里 PyRoKi 用 SDF / collision distance 形式实现，思想等价：离障碍物太近就产生残差，优化器会试图改变中间关节姿态来减小残差。

## 12. 障碍物如何改变末端轨迹

如果没有障碍物，优化器主要考虑：

```text
从 q_start 平滑到 q_goal
```

这通常会接近关节空间直线插值：

```text
q_i ~= q_i^init
```

末端路径可能穿过桌面或方块，因为关节空间直线不关心世界几何。

加入障碍物后，某些中间状态会有大的碰撞代价：

```text
phi( distance(robot(q_i), box_table) )
phi( distance(robot(q_i), box_object) )
```

优化器为了降低总代价，会改变中间的 `q_i`：

```text
q_i^* != q_i^init
```

这会间接改变末端轨迹：

```text
^W T_E(q_i^*) != ^W T_E(q_i^init)
```

所以你在视频里看到的绕路，本质上不是末端被显式规定要绕某个圆弧，而是关节轨迹为了降低 collision cost 和保持平滑，形成了一条新的末端路径。

更准确地说：

```text
障碍物 box 影响的是整条机器人几何体的 swept collision cost；
末端轨迹只是优化后关节序列的 FK 投影结果。
```

这也是为什么“只看 EE 点有没有碰到物体”不够。真实碰撞可能来自：

```text
手指
手掌
手腕
前臂
```

PyRoKi 的 swept capsule 检查比单点 EE 检查更接近真实机器人碰撞。

## 13. 为什么只避障到 precontact

如果一直把目标方块当障碍物，同时要求 TCP 到达最终抓取位姿，会出现矛盾：

```text
抓取要求：TCP / gripper 必须接近甚至包围物体
避障要求：机器人几何体必须远离物体 box
```

所以当前策略分成两段：

```text
1. 当前状态 -> precontact
   使用 PyRoKi + object/table obstacles
   目标：不要提前撞物体和桌面

2. precontact -> grasp
   沿视觉抓取轴短距离插入
   目标：有控制地进入抓取区域
```

也就是：

```text
precontact 前：物体是障碍物
precontact 后：物体是抓取目标
```

这是抓取任务里常见的分层逻辑。不能把整个抓取过程都简单当作“远离目标物体”的避障问题。

## 14. precontact 处的二次观察

当前路线不是连续视觉伺服，而是一次半闭环修正：

```text
初始观察
-> 规划到 precontact
-> 停稳
-> 再跑一次 OWL-ViT / SAM2 / GraspNet
-> 如果质量门通过，采用新的 grasp pose
-> 否则沿用初始 grasp pose
```

质量门包括：

```text
mask_pixels >= threshold
depth_points >= threshold
object_shift <= threshold
grasp_tcp_shift <= threshold
precontact_tcp_shift <= threshold
IK reachability ok
```

数学上就是检查两次估计的一致性。设初始物体中心和二次观察物体中心为：

```text
c_obj^0, c_obj^1
```

则：

```text
|| c_obj^1 - c_obj^0 || <= epsilon_obj
```

设两次 grasp TCP 位置为：

```text
p_T^0, p_T^1
```

则：

```text
|| p_T^1 - p_T^0 || <= epsilon_grasp
```

如果差异太大，说明二次观察可能被手爪遮挡、分割漂移或深度异常污染，就不采用。

这次成功 run 中二次观察被采用：

```text
reobserve_adopted = True
object_shift_m = 0.0000149
grasp_shift_m = 0.00460
```

说明两次视觉估计高度一致。

## 15. 放置阶段和障碍物的关系

当前 accepted RGB-D route 的显式 PyRoKi collision-aware planning 主要用于：

```text
当前状态 -> grasp precontact
```

抓起之后，转运和放置使用更结构化的路径：

```text
抬高
-> 高位横移到目标上方
-> 垂直慢速下降
-> 停稳
-> 打开爪子
```

这不是每一段都重新跑 PyRoKi world-collision trajopt，而是用任务几何约束降低拖拽风险。这样做的原因是当前简单任务中，物体已经在爪子里，桌面高度已知由视觉/任务配置间接约束，慢速垂直下降比复杂重规划更稳定。

后续如果要更通用，可以把放置阶段也改成：

```text
视觉/点云重建障碍物
-> 生成 release precontact
-> collision-aware trajopt 到 release precontact
-> 垂直下降释放
```

## 16. 当前 real-work-friendly 和 sim-only 的边界

这次 RGB-D obstacle route 中，规划障碍物不是 sim truth：

```text
object obstacle: RGB-D mask point cloud AABB
table obstacle:  RGB-D support plane AABB
sim_truth:       False
```

但是以下内容仍然是仿真评估或工程先验：

```text
object_in_hand_after_close: 当前还是仿真判断
place_error_m / reward:     当前还是仿真评估
workspace_bounds:           人工任务先验
table_thickness / margin:   工程参数
camera pose / intrinsics:   仿真中来自传感器配置，真机上对应相机标定
```

因此更准确的说法是：

```text
抓取位姿和抓取前避障规划输入已经走 RGB-D 感知路线；
任务成功判定和部分任务先验仍保留仿真/工程成分。
```

## 17. 调试时应该看什么

如果视频里仍然碰撞，优先看这些 artifact：

```text
visual_obstacles.json
object_obstacle_points_world.npy
table_obstacle_points_world.npy
grasp_summary.json
pick_place_result_summary.json
```

重点检查：

```text
1. object box center 是否在物体中心附近
2. object box extent 是否过小或过大
3. table_z_world 是否接近真实桌面高度
4. table box 是否覆盖了机械臂会碰到的桌面区域
5. PyRoKi debug 里的 world_collision_count 是否为 2
6. before_close_tcp_error_m 是否足够小
7. 碰撞发生在 precontact 前，还是 insertion 阶段
```

判断方向：

```text
precontact 前碰撞:
  多半是 obstacle box 估计不准、margin 不够、PyRoKi 轨迹优化没绕开，或执行跟踪偏离轨迹。

insertion 阶段碰撞:
  可能是 GraspNet grasp pose / X2 TCP adapter / gripper proxy guard / 插入深度问题。

放置阶段拖拽物体:
  可能是 release TCP offset、下降高度、下降速度、抓取后物体在爪内位置估计问题。
```

## 18. 一句话总结

当前 X2 RGB-D visual grasp 的避障链条是：

```text
RGB-D + SAM2
-> world-frame object/table point clouds
-> robust AABB boxes
-> obstacles_world
-> PyRoKi world collision objects
-> joint-space trajectory optimization
-> FK 得到实际末端轨迹
```

障碍物不是直接指定末端轨迹，而是通过碰撞代价影响整条关节轨迹。末端轨迹是优化后的关节序列在运动学模型中的结果。

