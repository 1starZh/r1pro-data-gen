# `src` 源码结构

`r1pro_data_gen` 按职责分层，而不是按某一个任务切分。任务输入是
`tasks/**/*.yaml` 中的数据化 `TaskSpec`；源码只提供通用的领域模型、规划、技能、
执行和仿真能力。新增任务应优先增加 TaskSpec 数据，不应在源码中新增任务专属
policy、evaluator 或动作序列。

产品闭环是同进程 `AgentLoop`：冻结 GoalSpec 后，每步观测 → 选一个公开语义技能
→ 物理执行 → 证据 → 再决策。Agent 只看见 6 个公开技能：`base_navigate_to`、
`prepare_workspace`、`grasp_object`、`arm_carry_object_to`、`release_object`、
`push_object_to`。其余技能在 registry 中，供语义技能内部和可信 Plan 回放使用。

LLM / Python 规划期间仿真时钟冻结（禁止 `adapter.step()`）；录像只从物理步进
采样，规划等待不会写入空档。

```text
src/r1pro_data_gen/
├── domain/              # 纯 Python 领域模型与不变量
├── data/                # 文件读写、场景加载、随机化、运行产物 provenance
├── agent/               # 闭环：观测 → 单个语义动作 → 证据 → 下一步
├── planning/            # GoalSpec、Task Plan、导航语义、运行时引用、LLM 合同
│   ├── goals/
│   ├── context/
│   ├── navigation/
│   ├── task/
│   ├── llm/
│   │   └── providers/
│   └── backends/
├── skills/              # 任务无关的可调用能力与 registry
│   ├── core/
│   ├── mobility/
│   ├── manipulation/
│   ├── observation/
│   ├── planning/
│   └── posture/
├── methods/              # skills 使用的确定性算法（碰撞、A*、笛卡尔/螺旋 IK、MPlib）
│   ├── navigation/
│   └── manipulation/
├── robot/                # R1Pro 关节、运动学、底盘和机器人标定（IK seed 不改）
├── execution/            # Plan/skill 编排与执行合同
├── evaluation/           # Evidence、PredicateVerifier、complete-episode acceptance
├── simulation/            # 仿真后端和 EvidenceRecorder；Isaac Sim 单独隔离
├── control/              # 控制接口与命令路由
├── infrastructure/       # GPU/运行环境门禁等基础设施
└── tasks/                # TaskSpec v2 的加载和 catalog 接口，不放任务实现
```

## 边界规则

- `domain` 只保存内存对象和领域规则，不读文件、不导入 Isaac Sim。
- `data` 负责把文件数据转换为 domain 对象；Plan 序列化入口是
  `r1pro_data_gen.data.plan_io`，场景加载入口是 `r1pro_data_gen.data.scenes`。
- `planning` 决定“做什么”，`methods` 解决“怎样计算可行路径”，`skills` 将方法封装为
  可执行的任务无关能力。规划器不直接操作 simulator handle。手臂接近默认先走
  SE(3) 螺旋/笛卡尔链式 IK（`TaskSpaceVerified`），碰撞才退到 OMPL。
- `agent` 是产品闭环入口的状态机；它可以调用 `planning` 生成目标/上下文，调用
  `execution` 执行技能，但不把任务流程硬编码进某个 task package。失败只在同一
  episode 内用上一步诊断重选技能，不 reset 仿真。
- `simulation` 是唯一的仿真适配边界；`freeze_simulation_clock` 包住 LLM 调用。
  纯逻辑测试不应依赖 Isaac Sim。
- `evaluation` 只根据通用 evidence 和冻结的 GoalSpec 判定结果，不按任务名分支。
  技能局部 success 不等于 `stage_success_complete`。

## 推荐导入入口

优先从包级 API 导入稳定对象；只有需要某个实现细节时才使用子模块路径：

```python
from r1pro_data_gen.agent import AgentLoop
from r1pro_data_gen.data import load_plan, load_scene
from r1pro_data_gen.domain import GoalSpec, Plan, SceneModel
from r1pro_data_gen.planning import GoalCompiler, LLMTaskPlanner
from r1pro_data_gen.skills import SkillRegistry, build_default_registry
```

旧的扁平模块路径不再是兼容入口。新的代码按能力域导入，例如使用
`skills.manipulation.arm_motion`、`planning.context.facts` 或
`methods.navigation.astar`，不要在根目录重新创建同类模块。
