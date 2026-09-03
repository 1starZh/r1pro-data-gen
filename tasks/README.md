# TaskSpec 任务目录

`tasks/` 是项目唯一公开的任务定义目录。每个任务都是一个数据文件，所有正式入口都通过同一个 `TaskSpec` 读取场景和自然语言目标；任务目录不放 Python policy、evaluator、recorder 或动作脚本。

Agent 对任何任务都只调用同一套 6 个公开技能：`base_navigate_to`、
`prepare_workspace`、`grasp_object`、`arm_carry_object_to`、`release_object`、
`push_object_to`。桌面抓放的语义顺序是导航（若尚未到位）→ 抓 → 搬 → 放。

## 当前任务

| id | family | 场景要点 | `scene_human_verified` |
| --- | --- | --- | --- |
| `pickplace.tabletop` | pickplace | 封闭围栏 + 两道错开门，桌面圆柱抓放 | `true` |
| `pickplace.tabletop_complete` | pickplace | 只有桌子，约 2 m 西侧起步 | `true` |
| `pickplace.floor_to_table_complete` | pickplace | 地面到桌面 | `false` |
| `pickplace.holdout_floor_to_table` | pickplace | 地面到桌面 holdout | `false` |
| `pickplace.holdout_prism_on_slate` | pickplace | 几何 holdout | `false` |
| `push.box_to_region` / `push.box_to_region_complete` | push | 平面推移 | `false` |
| `rearrangement.three_objects` / `rearrangement.three_objects_complete` | rearrangement | 三物体重排 | `false` |
| `navigation.arena_route` | navigation | 两门导航 showcase | `false` |

只有 `scene_human_verified: true` 的任务能走 `run_task.py` 物理 rollout。

## 文件格式

任务文件放在 `tasks/<family>/<name>.yaml`，必须符合 `task_spec.v2`。场景数据直接
嵌入 TaskSpec，任务文件是运行所需的唯一数据入口：

```yaml
schema_version: task_spec.v2
id: pickplace.tabletop_complete
family: pickplace
scene_human_verified: false
scene:
  name: tabletop_cylinder_manipulation
  world: {}
  robot:
    asset: asset/r1pro/r1pro.usda
  objects: []
instruction: >-
  Describe the physical state that must be achieved.
tags:
  - manipulation
```

必填字段是 `schema_version`、`id`、`family`、`scene_human_verified`、`scene` 和
`instruction`；`tags` 可选。`id` 必须稳定且唯一，`scene` 必须是完整的场景映射，
不能再写外部文件路径。`scene_human_verified` 必须是严格布尔值：只有人工检查确认
场景可用于物理测试后才能设为 `true`；新任务和随机化派生场景默认设为 `false`。
未确认的场景可以做离线解析、规划和 `--prepare-only`，物理执行入口会拒绝它。
场景只声明环境事实，任务目标和任务语义写在 `instruction` 中。

## 统一入口

产品入口：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src \
  python scripts/tasks/run_task.py \
  --task pickplace.tabletop_complete \
  --device cuda:0 --physical-gpu-id <gpu>
```

`--task` 接受 TaskSpec id 或 YAML 路径。以下入口也必须使用同一参数，不再组合传入独立的 scene、instruction、instruction file 或 Python task package：

- `scripts/tasks/run_task.py`：产品闭环 AgentLoop；
- `scripts/planning/run_llm_loop.py`：external LLM 计划反馈闭环；
- `scripts/tasks/run_plan.py`：TaskSpec 驱动的计划回放；
- `scripts/benchmarks/run_llm_random_rollouts.py`：随机化 rollout 编排；
- `scripts/benchmarks/run_benchmark_suite.py`：benchmark case 只写 `task` 引用；
- `scripts/planning/generate_llm_plan.py`：离线计划生成。

随机 rollout 产生的临时 `task.yaml` 仍是同一 schema 的派生数据，并将随机化后的
完整场景再次嵌入；它不会继承源任务的人工确认状态。

## 新增任务规则

新增任务只需：

1. 添加一个包含完整 `scene` 映射的合法 TaskSpec YAML；
2. 先保持 `scene_human_verified: false`，人工检查场景后再改为 `true`；
3. 用 `run_task.py --task <id|path>` 验证入口。

不要新增任务注册表项、任务专属 policy/evaluator、专属技能或隐藏动作序列。旧任务专属实现已从工作树移除；历史过程仅可从 Git 历史和进度文档追溯，不属于公开任务入口。
