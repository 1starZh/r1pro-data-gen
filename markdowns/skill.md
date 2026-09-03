# 技能库

本文是技能的现行合同：给闭环 Agent / LLM 选动作用，也作为我们对技能实现的记录。产品路径一步一个公开技能。新任务只加 TaskSpec（YAML + instruction），不往公开目录加技能、不写任务专属 policy。

公开目录由 `AGENT_PUBLIC_SKILLS` / `LLM_PUBLIC_SKILLS` 锁定，六个名字：

`base_navigate_to` · `prepare_workspace` · `grasp_object` · `arm_carry_object_to` · `release_object` · `push_object_to`

注册表里还有底盘微操、手臂规划、夹爪、躯干、全身过渡和查询。那些只给这六个技能内部调用，或给可信 Plan 回放。Agent 不能直接点名。

```text
Agent / LLM 每步只选一个公开技能
        │
        ▼
navigate / prepare_workspace / grasp / carry / release / push
        │
        ▼
内部：arm_move_*、align、gripper_*、torso、whole_body_*、
      support_aware_grasp、transfer_*、joint_mask、query_*、
      base_move_to / rotate / follow_path
```

观测循环每步已经给出物体位姿、接触、attachment。Query 技能不对 LLM 开放。

---

## 1. LLM 必须遵守的调用规则

- 每步恰好一个技能。不要拼 `arm_move_to` / `gripper_grasp`，也不要一条龙 `transfer_*`。
- 实体名必须是顶层场景对象（例如 `pick_cylinder`、`place_target`、`work_table`）。嵌套区域名 `place_region` 非法。
- 导航：`target_ref=scene://<object>`（靠近场景实体）或 GoalSpec 里的 `target=[x,y,yaw]`（明确底盘位姿），加 `purpose`。不要自己编世界坐标。
- `side=auto` 由运行时按几何排左右臂；只有观测或失败诊断支持时才写死 left/right。
- 上一步技能成功 ≠ 任务成功。Verifier 看冻结 GoalSpec。
- `unreachable_from_base`：改底盘接近，不要只换臂。
- 抓取失败码 `target_contact_not_established` / `contact_not_centered` / `one_finger_contact` / `grasp_not_attached`，且物体仍在够得着的范围：从**当前站位**再调 `grasp_object`，不要插一次额外导航。
- 桌面抓放顺序：`base_navigate_to(purpose=pregrasp)`（尚未到位才需要）→ `grasp_object` → `arm_carry_object_to` → `release_object`。
- 只在抓取报 `workspace_not_prepared`、物体在地面/低支撑、或站立高度明显不对时才调 `prepare_workspace`。
- 已抓住且放置区在**同一支撑面**：禁止再导航，只用 `arm_carry_object_to`。`purpose=dropoff` 只用于抓住后换到另一张支撑面。
- `prepare_workspace` 禁止关节角。`profile` 只能是 `tabletop` / `floor` / `carry` / `travel`。
- 安全违例非空时停止重复操作，发 `unsupported`。不要输出关节名、关节角、力矩或传送。

建议 JSON：

```json
{
  "schema_version": "agent_action.v1",
  "status": "act",
  "reason": "",
  "action": {
    "skill": "grasp_object",
    "parameters": {"object_name": "pick_cylinder", "side": "auto"}
  }
}
```

---

## 2. 公开技能

### 2.1 `base_navigate_to`

把底盘开到可操作站位。场景有障碍时走 2D A*，纯前进跟踪，不绕场外。

**LLM 参数**

| 参数 | 含义 |
| --- | --- |
| `target_ref` | `scene://物体`，运行时从场景几何解接近位姿 |
| `target` | `[x, y, yaw]`，只能来自冻结 GoalSpec 的底盘目标 |
| `purpose` | `pregrasp` / `dropoff` / `observe` / `park` / `navigation` / `staging` |
| `approach_side` | 可选 `west` / `east` / `south` / `north` |

二者只给一个：靠近实体用 `target_ref`，明确位姿用 `target`。

**内部**

1. 解目标站位（footprint 默认 0.25 m）。
2. 已在约 4 cm / 0.10 rad 内：0 步成功返回。
3. 否则栅格 A*，前瞻跟踪；误差 1.5 s 不再缩小则停止，避免空转进录像。
4. 终点刹停、锁轮。

**成功**：到位并刹住。  
**失败**：起点/终点在障碍里、无路、卡住未到、`same_support_navigation_forbidden`（已抓住且目标在同一支撑面）。

桌面两门房间：从 `(-3.8, -2.4)` 穿过错开门到桌西侧，近期一次 8 路点、1318 步成功。

---

### 2.2 `prepare_workspace`

把腰/上身调到命名工作高度。抓取**不会**自己俯腰。

| 参数 | 含义 |
| --- | --- |
| `profile` | 必填：`tabletop` / `floor` / `carry` / `travel` |
| `object_name` | `floor` 用；省略则取场景里最低可抓物体 |
| `side` | `auto` / `left` / `right` |

- `tabletop` / `carry` / `travel`：站立标定躯干。
- `floor`：对低物做认证全身预抓（`whole_body_pregrasp_transition`）。

桌面默认站姿够高，不必调用。地面任务先 `profile=floor`，再 `grasp_object`。  
失败：`workspace_backend_unavailable`、`floor_target_unavailable`、未知 profile。

---

### 2.3 `grasp_object`

在**当前底盘站位、当前腰高**抓住命名物体。形状用实时几何，不按物体名分支。

| 参数 | 含义 |
| --- | --- |
| `object_name` | 必填，顶层可抓物体 |
| `side` | 默认 `auto` |

**前置**：底盘已刹；物体在手臂工作空间内。不够则 `workspace_not_prepared`，本技能不俯腰、不全身预抓。

**内部（桌面当前实现，一段技能、多段轨迹）**

1. 锁轮、刹停，开夹。
2. 手臂若仍在 home 悬挂：关节空间抬到 ready（速度 0.18）。
3. 笛卡尔/螺旋一条路径到物体上方约 15 cm standoff（`TaskSpaceVerified`；碰障才 OMPL）。
4. `arm_align_gripper` 多次闭环小修正（最短约 0.28 s/次）。
5. `gripper_grasp` 闭合。要**双指都接触**并建立 attachment；几何上“在两指之间”不够。
6. 闭合失败且物体未移动：张开、退回 standoff，换下一组几何再试。`auto` 时第一臂若已开始动，不再自动换另一臂。

**成功**：attached。  
**失败码**：`workspace_not_prepared`、`unreachable_from_base`、`target_contact_not_established`、`contact_not_centered`、`one_finger_contact`、`grasp_not_attached`、`object_moved_before_grasp`、`ready_pose_failed`。

**已知问题**：对外一次调用，对内 ready / 接近 / 多次对准 / 闭合互相停顿。对准结束时垂直误差常仍约 5 cm。两门场景接近曾被拉到约 7 s（410 点），只有桌子约 1.6 s。这是录像里预抓“乱动/卡顿”的来源。

---

### 2.4 `arm_carry_object_to`

物体已抓住，放到命名区域所在支撑上。同支撑面短距离不要求再导航。

| 参数 | 含义 |
| --- | --- |
| `object_name` | 当前 attached 物体 |
| `target_region_name` | 目标区域/标记，顶层对象（桌面是 `place_target`） |
| `support_surface_name` | 区域下方支撑（桌面是 `work_table`） |
| `side` | 默认 `auto`，跟当前附着侧 |

**内部**

1. 必须 already attached。
2. 同支撑且很近：只走 `carry_extend` 到区域上方。
3. 更远：回抽 → 横移 → 伸到上方（`arm_move_through` 一条认证轨迹）。
4. 单独一条竖直下降（保持当前末端姿态，只改高度）。
5. 看**物体**是否在区域内、高度是否到。末端跟踪差一点，只要物体在 footprint 内且高度到位，也算放置成功。

**成功**：物体在区域、高度到位。  
**失败**：`object is not attached`、`attachment_lost`、`carry motion failed`、`place_descend_failed`（下降无碰路径/IK 不够且物体也没落到区域）。

**已知问题**：下降比抓取脆。只有桌子：下降执行了但末端差约 5.9 cm，靠区域内判定过关。两门：前两次 `no_collision_free_path`，第三次才落到区域。完成依赖重试和判定，不是一次下降就位。

---

### 2.5 `release_object`

开夹脱离，再抬手离开，短停给 `released` / `settled` 证据。

| 参数 | 含义 |
| --- | --- |
| `object_name` | 当前附着物体 |
| `side` | 默认 `auto` |

先开夹（拆 attachment），再笛卡尔 `+z` 约 10 cm 抬空夹爪。成功只表示开夹和抬手；是否放稳仍由 GoalSpec verifier 判定。不要在仍抓住时上提。

---

### 2.6 `push_object_to`

不抓，把可推物体推到语义目标。给 `tasks/push`。桌面抓放不用。

目标三者只给一个：`target_ref` / `target_region_name` / `target_pose`，外加 `object_name`。

---

## 3. 内部后端（不对 Agent）

这些在 `build_default_registry` 里，语义技能当积木。

**底盘**：`base_move_to`（直线，不绕障）、`base_rotate_to`、`base_follow_path`、`base_velocity_set`、`base_lock_wheels` / `base_unlock_wheels`。

**手臂**：`arm_move_to`（一个末端/grasp_center 位姿）、`arm_move_through`（多路点一条轨迹）、`arm_move_directional`、`arm_align_gripper`、`arm_joint_to`、`arm_trajectory_follow`、`arm_rotate_ee`。接近默认笛卡尔螺旋链式 IK，不改 IK seed。

**夹爪**：`gripper_set`、`gripper_grasp`（双指接触 + attachment）。

**姿态 / 全身**：`torso_move_to`、`joint_mask_lock` / `unlock`、`whole_body_pregrasp_transition`、`whole_body_hold_transition`。全身预抓只由 `prepare_workspace(profile=floor)` 触发。

**组合（禁止 Agent 用它们替代四段语义）**：`support_aware_grasp_object`、`transfer_object_between_supports`、`whole_body_transfer_object_between_supports`。

**只读**：`query_object_pose`、`query_contacts`、`query_ee_pose`、`query_joint_pos`、`query_ik_solution`、`query_arm_path`、`query_base_path`。

---

## 4. 规划与时间

- LLM / IK / 路径计算期间冻结仿真时钟，禁止 `adapter.step()`。录像只采物理步进，墙钟等待不进视频。
- 技能内部的 settle / hold / 失败后空转导航**会**进录像。卡住导航必须早停，不要把整段预算空转到视频里。

---

## 5. 和任务的关系

| 任务 | 公开技能用法 |
| --- | --- |
| `pickplace.tabletop`（两门） | 导航过门 → 抓 → 搬（可能多次）→ 放 |
| `pickplace.tabletop_complete`（只有桌子） | 短导航 → 抓 → 搬 → 放 |
| 地面到桌面 | 先 `prepare_workspace(profile=floor)`，再抓搬放 |
| `push.*` | `push_object_to`，可先导航 |

圆柱、立方体、地面物体走同一套公开名字，差在场景几何和要不要先 prepare。卡顿、放下偏、泛化差，都还在这六个技能的内部实现，不是公开目录不够。
