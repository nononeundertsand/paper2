# 本次更新简介

## 更新概述

本次更新围绕“面向边缘大模型的按需读写参数化记忆机制”补齐了端到端实验系统。当前代码已经从早期 synthetic feasibility 实验，扩展到真实 QA 数据、真实 general 数据和真实 LLM hidden-state 表征上的系统验证。

核心目标是验证：

> 参数记忆不应始终激活。Dense Memory 虽然能补充 tail knowledge，但会破坏 base model 已有能力；Conditional Memory 通过轻量 router 按需激活 memory，可以在低激活率下补充 tail knowledge，并保持 common/general 能力。

## 主要新增内容

1. **端到端实验系统**
   - 新增 `scripts/run_e2e_smoke.bat`
   - 新增 `scripts/run_e2e_main.bat`
   - 新增 `scripts/run_e2e_full.bat`
   - 支持 smoke/main/full 三档实验运行。

2. **真实 QA 实验增强**
   - 使用 SQuAD 构造真实 `common_fact` 和 `tail_fact`。
   - 使用 AG News 构造真实 `general`。
   - 支持候选答案排序 EM/F1、MRR、Hits@K。

3. **系统性消融实验**
   - 多随机种子实验。
   - 记忆容量实验。
   - 专家数量和 Top-K 消融。
   - Router 噪声鲁棒性实验。
   - 资源感知写入策略实验。

4. **结果汇总与聚合**
   - 新增 `collect_results.py`，用于收集实验 JSON 输出。
   - 新增 `aggregate_results.py`，用于生成 mean/std 聚合表。

5. **研究文档补充**
   - 新增端到端实验系统设计文档。
   - 更新实验结果分析报告。
   - 更新研究进展汇总报告。
   - 新增相关工作与科研诚信核查文档。

## 推荐运行方式

先运行 smoke test：

```powershell
.\scripts\run_e2e_smoke.bat D:\models\Qwen2.5-0.5B-Instruct
```

环境确认无误后运行主实验：

```powershell
.\scripts\run_e2e_main.bat D:\models\Qwen2.5-0.5B-Instruct
```

完整实验包：

```powershell
.\scripts\run_e2e_full.bat D:\models\Qwen2.5-0.5B-Instruct
```

## 输出文件

主实验输出：

```text
outputs\e2e\summary\e2e_main_summary.csv
outputs\e2e\summary\e2e_main_aggregate.csv
```

完整实验输出：

```text
outputs\e2e\summary\e2e_full_summary.csv
outputs\e2e\summary\e2e_full_aggregate.csv
```

## 当前实验意义

当前实验已经支持以下阶段性结论：

1. Base model 对 tail QA 存在明显知识缺口。
2. Dense Memory 能补充 tail knowledge，但会造成明显负迁移。
3. Conditional Memory 可以通过按需激活 memory 避免负迁移。
4. 在 SQuAD + AG News 的真实数据设置中，Conditional Memory 能在较低 activation rate 下同时保持 common、tail 和 general 表现。
5. 当前结果已经从 toy synthetic 验证推进到真实数据候选答案排序验证。

## 后续计划

后续仍需继续补强：

1. 扩展到 Natural Questions、TriviaQA、HotpotQA 等更难真实 QA 数据集。
2. 加入生成式 QA EM/F1。
3. 补充真实端侧延迟、显存和能耗测量。
4. 比较 Selective RAG、Static LoRA、Adaptive Retrieval 等更强 baseline。
5. 将资源感知写入从仿真扩展为真实闭环。

