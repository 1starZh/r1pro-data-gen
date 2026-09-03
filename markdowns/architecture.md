# 项目说明

R1Pro 在 Isaac Sim 里做**任务无关**的移动操作数据生成。新产品任务只提供场景 YAML 和一句自然语言目标；运行时冻结 GoalSpec，由闭环 Agent 逐步调用公开技能，用物理证据验收整段 episode。不为单个任务写 Python policy，也不把 LLM 墙钟等待录进轨迹。

## 1. 目标与硬约束

- 同一套技能跑桌面抓放、过门导航、地面抓取、推移、多物体重排；新任务加 `tasks/<family>/<name>.yaml`。
- Agent 每步只选一个中层语义技能，看不见关节角、IK、网格规划。
- 验收单位是完整 episode（抓/搬/放/稳都要物理证据），不是单技能 success 或视频存在。
- 规划与计算期间仿真时间暂停，录像只含物理步进，轨迹连续。
- 不改 `kinematics.py` 的 IK seed。不锁死闲置臂。不为某一个 TaskSpec 写分叉。

## 2. 运行栈

| 层 | 现状 |
| --- | --- |
| 机器人 | 默认 `asset/r1pro/r1pro.usda`（USDA 从星海图官方仓库获取，不入库），左右臂/夹爪同一套 side-aware 技能 |
| 仿真 | Isaac Sim 5.1 + Isaac Lab 2.3 |
| 入口 | `scripts/tasks/run_task.py --task <TaskSpec id>` |
| Agent | 同进程 `AgentLoop`，DeepSeek |
| GPU | `CUDA_VISIBLE_DEVICES` 指向要用的物理卡；进程内 `cuda:0` |

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src \
  python scripts/tasks/run_task.py \
  --task pickplace.tabletop \
  --device cuda:0 --physical-gpu-id <gpu> --headless --enable_cameras
```

复用已冻结目标：`--goal-spec outputs/tasks/<prior>/goal_spec.json`。

## 3. 数据流

```text
TaskSpec (场景 + instruction)
    → GoalSpec / GoalContract（冻结，可复用）
    → AgentLoop：观测 → 一个公开技能 → 执行
    → 物理步进（step_hook 采视频与 evidence）
    → PredicateVerifier
    → complete-episode acceptance + rollout.mp4
```

LLM 或 IK 计算时 `adapter.freeze_simulation_clock()`，误步进会报错。技能内部的 settle、对准点动、失败后空转导航仍会进入录像。

## 4. 源码分层

`src/r1pro_data_gen/` 按职责分，不按任务名分：

| 包 | 职责 |
| --- | --- |
| `domain` | 内存模型：场景、GoalSpec、GraspContext、Evidence。不读文件、不碰 Isaac |
| `data` | YAML/JSON、场景加载、随机化、provenance |
| `agent` | 观测、prompt、一步一技能、局内失败反馈 |
| `planning` | Goal 编译、导航目标、LLM 合同与校验 |
| `skills` | 公开语义技能 + 内部后端，见 [`skill.md`](skill.md) |
| `methods` | A*、碰撞、笛卡尔/螺旋 IK、MPlib、全身候选 |
| `robot` | 关节映射、运动学、底盘、力矩/速度标定 |
| `execution` | Orchestrator，技能预算与安全中止 |
| `evaluation` | PredicateVerifier、acceptance（哈希、证据、视频） |
| `simulation` | Isaac 适配、视频、EvidenceRecorder |
| `tasks` | 只加载 TaskSpec catalog，没有任务实现 |

规划器不拿 simulator handle。技能通过 adapter 读写状态并 `step()`。

## 5. 任务

定义在 `tasks/<family>/<name>.yaml`，`task_spec.v2`，场景嵌在文件里。

当前已人工确认、可物理跑：

- `pickplace.tabletop`：围栏 + 两道错开门，桌面圆柱抓放。
- `pickplace.tabletop_complete`：只有桌子，约 2 m 西侧起步。

其余（地面到桌面、推移、三物体、纯导航 showcase）`scene_human_verified: false`，不能当正式物理 rollout。

## 6. 验收

一次 accepted 需要同时：

- 冻结 GoalSpec / GoalContract 哈希一致；
- 物理谓词满足（桌面抓放：`attached` → `inside_region` → `released` → `settled`）；
- 证据覆盖完整；
- 有效 RGB 视频；
- 正式 manifest。

`stage_success_complete` 要求本局每个技能都成功。Goal 谓词过了但中间技能失败过，仍是 `accepted` 且 `stage_success_complete: false`。固定场景一次通过 ≠ 随机泛化率。

产物在 `outputs/tasks/<run>/`：`result.json`、`action_trace.json`、`rollout.mp4`、`goal_spec.json`。

## 7. 测试

默认 `pytest` 只收 CPU/合同（当前约 566 条），不启 Isaac、不调 DeepSeek。物理单技能在 `tests/physical/`。完整任务只走 `run_task.py`。

## 8. 文档

现行说明只保留本目录三份：

- [`architecture.md`](architecture.md)：项目结构（本文）
- [`skill.md`](skill.md)：技能合同与实现记录
- [`current_work.md`](current_work.md)：现状、问题和下一步
