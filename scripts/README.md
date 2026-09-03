# 项目脚本

`scripts/` 只保留和 R1Pro 数据生成主流程直接相关的入口，并按职责分组。
任务定义仍然只在顶层 `tasks/<family>/<name>.yaml` 中维护；每个 TaskSpec 内嵌完整
场景数据和 `scene_human_verified` 人工验收状态。脚本不包含任务专属
policy、evaluator 或隐藏动作序列。

产品入口是 `scripts/tasks/run_task.py`：TaskSpec → 冻结 GoalSpec → 同进程
`AgentLoop`（DeepSeek，一步一个公开语义技能）→ Evidence / PredicateVerifier。
规划与 LLM 等待期间仿真时钟冻结，录像只记录物理步进。

所有正式视频入口统一输出 30fps；`--fps` 仅接受 `30`，以保证不同任务和
不同运行器生成的录像可以直接比较。

用哪张 GPU 由调用方指定：`CUDA_VISIBLE_DEVICES=<gpu>`，`--physical-gpu-id` 与
该物理编号一致，进程内设备名为 `cuda:0`。入口脚本里有一份物理卡允许列表，
与本机编号不符时改 `scripts/tasks/run_task.py`。

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src \
  python scripts/tasks/run_task.py \
  --task pickplace.tabletop \
  --device cuda:0 --physical-gpu-id <gpu> --headless --enable_cameras
```

复用已冻结 GoalSpec（不重新规划目标）：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src \
  python scripts/tasks/run_task.py \
  --task pickplace.tabletop_complete \
  --device cuda:0 --physical-gpu-id <gpu> --headless --enable_cameras \
  --goal-spec outputs/tasks/<prior_run>/goal_spec.json \
  --output-dir outputs/tasks/<this_run>
```

## 目录

```text
scripts/
├── tasks/
│   ├── run_task.py       # 唯一产品任务入口：TaskSpec → AgentLoop → evidence
│   └── run_plan.py       # TaskSpec + Plan 回放入口
├── planning/
│   ├── generate_llm_plan.py  # 离线生成并校验 Plan
│   └── run_llm_loop.py      # 外部 LLM Plan/replay 反馈闭环
├── benchmarks/
│   ├── run_benchmark_suite.py       # benchmark suite 编排与验收
│   └── run_llm_random_rollouts.py   # TaskSpec 随机化 rollout 编排
└── infrastructure/
    ├── check_gpu_health.py          # 物理执行前的 driver-backed GPU 门禁
    └── preview_scene.py             # 场景预览（非产品验收入口）
```

## 使用关系

```text
tasks/<family>/<name>.yaml
           │
           ▼
scripts/tasks/run_task.py                  # 正式产品路径
           │
           ├── scripts/planning/run_llm_loop.py
           ├── scripts/benchmarks/run_llm_random_rollouts.py
           └── scripts/benchmarks/run_benchmark_suite.py

scripts/planning/generate_llm_plan.py      # 离线 Plan 工具
scripts/tasks/run_plan.py                  # 固定 Plan 回放工具
scripts/infrastructure/check_gpu_health.py # GPU 环境门禁
```

## 常用入口

```bash
# 唯一产品任务入口（两门房间抓放）
PYTHONPATH=src python scripts/tasks/run_task.py \
  --task pickplace.tabletop

# 桌面近距离抓放 holdout
PYTHONPATH=src python scripts/tasks/run_task.py \
  --task pickplace.tabletop_complete

# 离线生成 Plan（--output 所在目录必须已有冻结的 goal_spec.json）
PYTHONPATH=src python scripts/planning/generate_llm_plan.py \
  --task pickplace.tabletop \
  --output outputs/tasks/<run>/plan.json \
  --urdf asset/r1pro/r1_pro_with_gripper.urdf

# 运行完整 benchmark suite
PYTHONPATH=src python scripts/benchmarks/run_benchmark_suite.py \
  --suite benchmarks/complete_task_episodes.yaml \
  --output-dir outputs/benchmarks/complete_task_prepare \
  --prepare-only
```

GPU 物理任务会由 `run_task.py` 自动调用源码中的 GPU 健康探针；需要单独检查
环境时才直接运行 `scripts/infrastructure/check_gpu_health.py`。

物理入口在启动 GPU/Isaac Sim 前检查 `scene_human_verified`；未人工确认的 TaskSpec
只能用于离线规划、场景解析或 `--prepare-only`，不能执行真实 rollout。当前已确认
可物理执行的是 `pickplace.tabletop`（两门房间）和 `pickplace.tabletop_complete`
（只有桌子）。

阶段性标定、导航诊断、思考模式 wrapper 和一次性资产转换器不属于项目脚本
接口。Isaac Sim 单技能验证统一放在 `tests/physical/`，实验产物统一写入
`outputs/`，不在这里增加新的任务脚本。
