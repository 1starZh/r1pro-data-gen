# R1Pro Data Gen

在 Isaac Sim 上为 R1Pro 生成任务无关的移动操作数据。新任务只提供场景 YAML 和自然语言目标；闭环 Agent 每步调用一个公开语义技能，用物理证据验收整段 episode。

公开技能六个：`base_navigate_to`、`prepare_workspace`、`grasp_object`、`arm_carry_object_to`、`release_object`、`push_object_to`。规划与 LLM 等待期间仿真时钟冻结，录像只记录物理步进。

## 目录

```text
asset/         机器人 URDF / 网格；USDA 需自行获取（见下）
tasks/         TaskSpec（场景 + instruction）
src/           领域模型、Agent、技能、规划、仿真适配
scripts/       产品入口
benchmarks/    泛化评测套件（数据文件）
tests/         CPU 合同测试；物理单技能在 tests/physical/
markdowns/     现行说明：architecture / skill / current_work
```

## 环境

- Isaac Sim 5.1 + Isaac Lab 2.3（建议使用对应的 conda 环境）
- LLM：DeepSeek，凭证只通过环境变量 `DEEPSEEK_API_KEY` 注入
- GPU：设置 `CUDA_VISIBLE_DEVICES` 为要用的物理卡；进程内设备名为 `cuda:0`。`--physical-gpu-id` 与可见的那张物理卡编号一致。若入口脚本拒绝该编号，改 `scripts/tasks/run_task.py` 里的允许列表。

## 机器人资产

仿真默认加载 `asset/r1pro/r1pro.usda`。该 USDA 体积过大，**不随本仓库提交**。请从星海图（Galaxea）R1 Pro 官方仓库获取 Isaac 仿真用 USD/USDA，放到上述路径（文件名保持 `r1pro.usda`，或改 TaskSpec 里的 `robot.asset`）。

官方入口：

- 文档：[OpenGalaxea](https://open-galaxea.github.io/Doc/)

仓库内仍包含体积较小的 URDF、网格和 MPlib 资源，供运动学与规划使用。

## 运行任务

已人工确认、可物理执行的任务：

- `pickplace.tabletop`：围栏 + 两道门，桌面抓放
- `pickplace.tabletop_complete`：只有桌子，约 2 m 外起步

```bash
conda activate <isaac_lab_env>
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=<gpu>

python scripts/tasks/run_task.py \
  --task pickplace.tabletop \
  --device cuda:0 \
  --physical-gpu-id <gpu> \
  --headless \
  --enable_cameras
```

复用已冻结 GoalSpec：

```bash
python scripts/tasks/run_task.py \
  --task pickplace.tabletop_complete \
  --device cuda:0 \
  --physical-gpu-id <gpu> \
  --headless \
  --enable_cameras \
  --goal-spec outputs/tasks/<prior_run>/goal_spec.json \
  --output-dir outputs/tasks/<this_run>
```

产物在 `outputs/tasks/<run>/`：`result.json`、`action_trace.json`、`rollout.mp4`。该目录不入库。

## 测试

```bash
python -m pytest
```

默认只跑 CPU/合同测试，不启动 Isaac Sim。

## 说明文档

- [`markdowns/architecture.md`](markdowns/architecture.md) 项目结构
- [`markdowns/skill.md`](markdowns/skill.md) 技能合同
- [`markdowns/current_work.md`](markdowns/current_work.md) 现状与下一步
