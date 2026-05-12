## Agentic HotpotQA

这个仓库实现了一个面向 HotpotQA 的 **agentic multi-hop QA** 原型。

当前主线已经从早期的固定 `filter -> answer -> teacher` 三角色脚本，重构为一个更适合研究和项目展示的主动式推理流程：

- `controller -> action selection -> state update -> verify -> stop`
- 有限动作空间，便于做 trace、分析和后续蒸馏
- 保留 HotpotQA 数据处理、SFT、GRPO、distill 等训练相关脚本，作为后续工作基础

### 当前能力

- Agentic 单样本推理
- Agentic 批量评测与 trace 导出
- HotpotQA 数据适配与上下文匹配
- 旧训练脚本保留：SFT、GRPO、distill、merge
- 基础单元测试与静态检查

### Agentic Workflow

当前 inference v1 使用 6 个高层动作：

1. `decompose_question`
2. `retrieve_passages`
3. `expand_entity`
4. `extract_evidence`
5. `verify_evidence`
6. `stop_and_answer`

核心目标不是一次性直接回答，而是在有限步数内主动决定：

- 是否先拆题
- 是否继续追踪桥接实体
- 是否已经有足够证据
- 是否应该停止并输出答案

### 目录

```text
.
├── run_agentic_workflow.py
├── evaluate_agentic_workflow.py
├── hotpot_evaluate_em_f1.py
├── eval_other_models/
├── doc/
│   ├── 3agent.pdf
│   ├── 5agent.pdf
│   └── agentic_project_brief.md
├── src/
│   ├── workflow_actions.py
│   ├── workflow_runner.py
│   ├── baseline_eval.py
│   ├── config.py
│   ├── data_preprocess.py
│   ├── workflow_schema.py
│   ├── load_model.py
│   ├── logging_utils.py
│   ├── openai_compatible.py
│   ├── agent_prompts.py
│   ├── reward_for_grpo.py
│   ├── sft.py
│   ├── grpo.py
│   ├── distill.py
│   └── merge.py
└── tests/
```

### 安装

```bash
pip install -r requirements.txt
```

如果你只想做静态检查和单元测试，不需要完整模型环境。

### 环境变量

外部 API key 通过环境变量读取：

```bash
export SILICONFLOW_API_KEY=...
export TENCENT_LKE_API_KEY=...
```

### 数据

默认路径在 [src/config.py](/home/ziyao/MultiAgent_hotpotQA/src/config.py:8) 中定义：

- `hotpotqa_test_400.json`
- `hotpot_train_v1.1.json`
- `Hotpotqa_sft.json`

当前代码默认接受 HotpotQA 原始 JSON 格式，并在内部转换为统一的 `QuestionSample` / `AgentState`。

### 常用命令

1. Agentic 单样本推理

```bash
python run_agentic_workflow.py --models 123 --sample-index 0 --show-trace
```

常用参数：

- `--models 123`
  - 三位模型编号，分别对应 `filter/evidence`、`answer`、`teacher(controller+verifier)`
- `--sample-id`
  - 按样本 id 运行
- `--sample-index`
  - 按样本下标运行
- `--max-steps`
  - controller 的最大步数
- `--show-trace`
  - 直接打印完整 trace JSON

2. Agentic 批量评测

```bash
python evaluate_agentic_workflow.py --models 123 --limit 20 --max-steps 6
```

输出：

- `results/agentic_answers_<timestamp>.json`
- `results/agentic_traces_<timestamp>.json`

3. HotpotQA 结果评估

```bash
python hotpot_evaluate_em_f1.py predictions.json gold.json
```

4. SFT 训练

```bash
python src/sft.py
```

5. GRPO 训练

```bash
python src/grpo.py
```

6. 蒸馏数据构造

```bash
python src/distill.py --models 126
```

7. LoRA merge

```bash
python src/merge.py
```

8. 外部模型基线评测

```bash
python eval_other_models/eval_deepseek.py
python eval_other_models/eval_qwen2.5_14B.py
python eval_other_models/eval_qwen3_8B.py
python eval_other_models/eval_qwen3_32B.py
```

### 测试与检查

当前环境下推荐：

```bash
python -m py_compile run_agentic_workflow.py evaluate_agentic_workflow.py hotpot_evaluate_em_f1.py src/*.py eval_other_models/*.py tests/*.py
python -m unittest discover -s tests -p 'test_*.py'
```

如果本机安装了 `pytest`，也可以继续使用 `pytest`。

### 代码结构说明

- [src/workflow_schema.py](/home/ziyao/MultiAgent_hotpotQA/src/workflow_schema.py:1)
  - 定义 `QuestionSample`、`AgentState`、`AgentResult`、`AgentTraceStep`
- [src/agent_prompts.py](/home/ziyao/MultiAgent_hotpotQA/src/agent_prompts.py:1)
  - 集中管理 filter / controller / verifier / answer prompts
- [src/workflow_actions.py](/home/ziyao/MultiAgent_hotpotQA/src/workflow_actions.py:45)
  - 定义动作集合、controller、verifier 和动作执行逻辑
- [src/workflow_runner.py](/home/ziyao/MultiAgent_hotpotQA/src/workflow_runner.py:47)
  - 编排 agent loop 和 trace 记录
- [src/data_preprocess.py](/home/ziyao/MultiAgent_hotpotQA/src/data_preprocess.py:39)
  - 负责 HotpotQA 读取、legacy 适配、context 解析和匹配

### 当前定位

这个仓库现在更适合被理解为：

`A research-oriented agentic retrieval QA prototype for multi-hop reasoning`

而不是一个已经完全工业化的训练框架。

它的重点是：

- 主动式多跳推理
- 可解释 trace
- verifier-guided stopping
- 为后续蒸馏/SFT 预留轨迹数据

### 后续方向

比较自然的下一步有三类：

- 改进 controller / verifier prompt，减少动作抖动
- 将 trace 导出为蒸馏或 SFT 数据格式
- 用更强的 retrieval / rerank 机制替换当前 context matching
