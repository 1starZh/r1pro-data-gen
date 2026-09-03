# 测试结构

测试按被测源码职责组织，运行时代码仍全部位于 `src/r1pro_data_gen/`。默认
`pytest` 只收集 CPU/合同测试；需要 Isaac Sim 的物理验证集中放在
`tests/physical/`，不会被默认回归导入。

当前默认可收集约 **566** 条 CPU/合同测试（以 `pytest --collect-only` 为准）。

```text
tests/
├── contracts/             跨层接口与机器人控制合同
├── fixtures/              稳定、可版本化的测试输入
├── integration/
│   ├── planning/          Plan 序列化与计划合同
│   ├── scenes/            场景 fixture 加载与约束
│   ├── skills/            技能注册表集成
│   └── tasks/             任务注册与 evaluator 集成
├── unit/
│   ├── agent/             AgentLoop、动作合同和观测（含规划期冻结时钟）
│   ├── domain/            Goal、场景语义和几何
│   ├── entrypoints/       CLI/批处理脚本的纯逻辑
│   ├── evaluation/        acceptance 与 predicate verifier
│   ├── execution/         Orchestrator
│   ├── methods/           A*、RRT、MPlib、笛卡尔路径
│   ├── planning/          Goal/LLM/导航规划合同
│   ├── robot/             底盘和运动学
│   ├── simulation/        adapter、evidence 和视频合同
│   ├── skills/            各通用机器人技能（含导航卡住退出、放置看物体位姿）
│   └── tasks/             TaskSpec catalog 与任务合同
├── physical/              Isaac Sim 单技能物理验证与 MP4 生成
└── support.py             共享 fake adapter 与项目根路径
```

## TaskSpec 入口回归

公开任务定义位于 `tasks/<family>/<name>.yaml`，统一采用 `task_spec.v2`。
每个 TaskSpec 自带完整的 `scene` 映射和严格布尔字段 `scene_human_verified`；
默认回归会验证任务数据可解析，物理入口只接受人工确认过的场景
（当前为 `pickplace.tabletop` 与 `pickplace.tabletop_complete`）。
入口测试覆盖 TaskSpec catalog、benchmark case、随机 rollout、产品入口和
独立 Plan replay；这些入口都必须通过 `--task <TaskSpec id|path>` 取得场景与
instruction，不能再组合传入 scene、instruction 或 task package 名。

## CPU/合同回归

在 Isaac Lab 对应的 Python 环境中：

```bash
python -m pytest
```

也可以按职责运行：

```bash
python -m pytest tests/unit/planning
python -m pytest tests/unit/skills
python -m pytest tests/integration
python -m pytest tests/contracts
```

当前基线以命令输出为准。CPU 回归不启动 Isaac Sim、不调用 DeepSeek，也不代表
GPU 物理任务成功。

## 物理技能验证

单技能验证：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src \
  python tests/physical/verify_skill.py --skill arm_move_to --scene bare \
  --headless --device cuda:0 --physical-gpu-id <gpu>
```

批量验证：

```bash
ISAACLAB_PYTHON="$(command -v python)" \
  PHYSICAL_GPU_ID=<gpu> \
  tests/physical/run_all.sh
```

物理验证产物统一写入 `outputs/skills/`，不进入版本控制。完整任务 rollout 走
`scripts/tasks/run_task.py`，产物在 `outputs/tasks/`。

## 维护规则

- 测试文件跟随被测模块归类，不再平铺到 `tests/unit/` 根目录。
- 通用 fake、stub 和项目路径统一放在 `tests/support.py`。
- `outputs/` 中的历史运行结果不能作为测试 fixture。
- `tests/fixtures/scenes/` 只放无任务入口的底层测试场景；任务场景必须以内嵌 `scene` 的 TaskSpec 形式维护在 `tasks/`。
- 已删除的旧任务专属 policy/evaluator/Plan 不再恢复为测试依赖；新任务只通过 TaskSpec 合同接入。
- 实验进度和历史成功率写入 `markdowns/current_work.md`，不写进测试说明。
