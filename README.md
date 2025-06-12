## Multi-Agent HotpotQA 系统

本仓库实现了一个**多智能体**框架，用于在 [HotpotQA](https://hotpotqa.github.io/) 数据集上进行多跳推理。实验结果表明，本系统在指标上超过了 **Deepseek R1** 基线。核心思路是采用多个专门化智能体（如 3-agent 和 5-agent 配置）协同工作，以回答复杂的问题。

---

### 📂 项目结构
```
│  eval_3agent.py           # 评估 3-agent 系统
│  eval_5agent.py           # 评估 5-agent 系统
│  run_inference_3agent.py  # 使用 3-agent 配置运行推理
│  hotpot_evaluate_em_f1.py # 计算 HotpotQA 上的 EM 和 F1
│
├─ doc                    # 系统架构图
│     3agent.pdf          # 3-agent 系统架构示意图
│     5agent.pdf          # 5-agent 系统架构示意图
│
├─ eval_other_models      # 其他基线模型的评估脚本
│     eval_deepseek.py    # Deepseek R1 评估脚本
│     eval_qwen2.5_14B.py # Qwen-2.5B (14B) 评估脚本
│     eval_qwen3_8B.py    # Qwen-3 (8B) 评估脚本
│     eval_qwen3_32B.py   # Qwen-3 (32B) 评估脚本
│     extract_answer.py   # 答案提取工具
│     extract_regular.py  # 模式／正则提取脚本
│
└─ src                    # 核心模型与训练脚本
      data_preprocess.py  # 预处理 HotpotQA 数据
      load_model.py       # 加载与初始化预训练模型
      sft.py              # 监督微调（Supervised Fine-Tuning）
      distill.py          # 知识蒸馏实现
      grpo.py             # 基于 unsloth 库的通用化奖励策略优化（GRPO）
      reward_for_grpo.py  # 为 GRPO 提供的奖励函数
      merge.py            # 多智能体输出合并工具
      __init__.py
      \__pycache__\       # Python 缓存目录
```

---

### 🚀 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **数据预处理**
   ```bash
   python src/data_preprocess.py --input_dir data/raw --output_dir data/processed
   ```

3. **训练或微调智能体**
   ```bash
   python src/sft.py --config configs/sft.yaml
   python src/distill.py --config configs/distill.yaml
   ```

4. **运行推理**
   - **3-agent 配置**
     ```bash
     python run_inference_3agent.py --config configs/3agent.yaml
     ```
   - **5-agent 配置**
     ```bash
     python eval_5agent.py --config configs/5agent.yaml
     ```

5. **性能评估**
   ```bash
   python hotpot_evaluate_em_f1.py --predictions predictions.json --gold data/processed/dev.json
   ```

---

### 🏗️ 系统架构图

以下为两种多智能体系统的架构示意：

- **3-Agent 系统**
  ![3-Agent 架构](doc/3agent.pdf)

- **5-Agent 系统**
  ![5-Agent 架构](doc/5agent.pdf)

---

### 📈 实验结果
以下为 HotpotQA 自定义测试集（400 道题）上的关键实验结果：

#### 1. 基线模型性能

| 模型                            | EM     | F1     |
|---------------------------------|--------|--------|
| Deepseek R1                     | 52.25% | 68.54% |
| Qwen2.5 32B                     | 27.82% | 46.39% |
| Qwen2.5 14B                     | 22.11% | 39.16% |
| Qwen2.5 3B                      | 15.50% | 29.49% |

#### 2. MAS-Pipe 系统性能（Qwen2.5 3B）

| 配置                  | EM     | F1     |
|-----------------------|--------|--------|
| 单体 Qwen2.5 3B (Base)        | 15.50% | 29.49% |
| MAS-Pipe (所有 Base Agents)   | 17.00% | 30.48% |
| MAS-Pipe (RLHF Answer Agent)  | 34.75% | 48.56% |
| MAS-Pipe (SFT Filter + RLHF)  | 35.00% | 46.92% |

> **最佳配置**：仅对 Answer Agent 应用 RLHF，F1 达到 **48.56%**，已超越 Qwen2.5 32B (46.39%)。  

#### 3. MAS-Parallel 系统性能（Qwen2.5 7B）

| 配置                                     | EM     | F1     |
|------------------------------------------|--------|--------|
| MAS-Parallel (所有 Base Agents)          | 31.75% | 46.55% |
| MAS-Parallel (Strong Agent: GRPO-4500)   | 47.00% | 64.42% |
| MAS-Parallel (Strong Agent: GRPO-7500)   | 59.75% | 73.87% |

> **最终结果**：MAS-Parallel (GRPO-7500) 在 HotpotQA 自定义测试集上取得 **EM 59.75% / F1 73.87%**，显著超越 Deepseek R1 (F1 68.54%)。

---

### 🤝 贡献
欢迎贡献！如有问题、功能需求或改进建议，请提交 Issue 或 Pull Request。

---

### 📜 许可证
本项目采用 MIT 许可证，详情参见 [LICENSE](LICENSE)。

