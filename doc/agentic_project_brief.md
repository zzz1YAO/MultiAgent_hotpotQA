# Agentic HotpotQA Project Brief

## One-line Summary

An agentic multi-hop QA prototype that turns a fixed three-role pipeline into an active retrieval-and-verification loop on HotpotQA.

## Project Positioning

这个项目不再强调“多角色串联脚本”，而是强调：

- 主动式多跳检索与推理
- 基于动作空间的 agentic workflow
- 可记录、可分析、可蒸馏的推理轨迹

更适合的英文定位：

`Agentic Multi-hop QA with Adaptive Evidence Collection and Verifier-Guided Stopping`

或者更偏工程展示一点：

`A Low-Cost Agentic Retrieval QA Stack for Multi-hop Reasoning`

## What Was Built

项目当前包含这些核心设计：

1. 一个结构化 agent state
   - 包含 `question`、`subquestions`、`candidate_entities`、`candidate_passages`、`evidence_buffer`、`draft_answer`、`verification_status`、`trace`

2. 一个有限动作空间
   - `decompose_question`
   - `retrieve_passages`
   - `expand_entity`
   - `extract_evidence`
   - `verify_evidence`
   - `stop_and_answer`

3. 一个 LLM-driven controller
   - 每一步从有限动作中选一个
   - 输出结构化 JSON action
   - 便于调试和后续训练

4. 一个 verifier-guided stopping 机制
   - 不再默认“生成完就结束”
   - 先判断证据是否充足，再决定继续检索还是停止回答

## Why It Is Interesting

这个项目的价值不在于“再做一个 HotpotQA baseline”，而在于：

- 它把静态 QA pipeline 改造成了动态决策系统
- 它显式建模了多跳问题中的拆题、桥接实体追踪和证据验证
- 它具备 trace，可继续用于蒸馏、SFT 或行为分析

如果用于找工作，更推荐强调这些点：

- 从固定 pipeline 重构为 agentic workflow
- 设计了状态、动作、控制和停止条件
- 关注 reasoning trace，而不只是最终答案

## Resume-friendly Description

可以直接放在简历里的英文描述：

`Built an agentic multi-hop QA system on HotpotQA with structured action selection, verifier-guided evidence collection, and traceable inference loops.`

`Redesigned a fixed multi-agent pipeline into a controllable retrieval-and-reasoning workflow, improving extensibility for future distillation and supervised trace learning.`

## Interview Framing

面试里更自然的讲法：

“我一开始有一个固定的 multi-agent QA 原型，但它的流程比较僵化，检索深度和停止条件都写死了。后来我把它重构成 agentic workflow：系统不再直接回答，而是在有限预算内主动决定是否拆题、追踪桥接实体、补证据并验证覆盖度，最后再停止作答。这样项目更适合做分析、ablation 和后续蒸馏。”

## Next Research Directions

- 加强 controller prompt，减少动作选择抖动
- 引入更强的 retrieval / reranking
- 把 trace 导出成 SFT 数据
- 后续再考虑 RL 或 search-based policy optimization
