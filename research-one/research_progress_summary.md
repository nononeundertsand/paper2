# 按需读写参数化记忆研究进展汇总

## 1. 当前研究背景

大语言模型在知识密集型任务中通常依赖两类增强方式：一类是 **RAG**，通过外部检索库获取文档证据；另一类是 **参数化记忆**，通过 LoRA、MLP Memory、Memory Layer 等方式将知识压缩进模型参数或外挂参数模块中。

RAG 的优势是知识更新灵活、可追溯性强，但在边缘设备和移动计算场景中存在明显问题：检索需要额外索引、向量库和上下文拼接，带来较高延迟、内存占用、能耗和通信开销。参数化记忆的优势是推理时不需要显式检索文档，速度更快，但现有方法通常是静态、全局、始终激活的，一旦记忆模块被调用，就可能干扰基础模型已有能力，而且知识更新、删除和资源调度不够灵活。

结合边缘智能场景，我们关注的核心问题是：

> **边缘设备上的大模型不应该一直检索，也不应该一直激活参数记忆，而应该学会什么时候读取记忆、什么时候写入记忆。**

因此，本文当前研究聚焦于 **面向边缘大模型的按需读写参数化记忆机制**。其目标是在资源受限环境下，以较低的 memory 激活率补充长尾知识，同时尽量保持基础模型已有的 common/general 能力。

## 2. 我们提出的问题

当前工作将问题拆成两个相互关联的子问题：

1. **按需读取问题**  
   给定一个基础 LLM 和外挂参数记忆模块，如何判断当前输入是否真的需要读取 memory？如果基础模型已经能够正确处理 common/general 样本，就应跳过 memory；如果遇到基础模型缺失的 tail facts，则应激活 memory。

2. **资源感知写入问题**  
   端侧设备会不断产生新知识，但不是所有知识都值得参数化。如何根据访问频率、收益、能耗、存储、隐私和删除风险，决定哪些知识值得从外部情景记忆固化为参数记忆？

这两个问题共同构成一个端侧长期记忆闭环：

- 写入机制决定 **哪些知识进入参数记忆**；
- 读取机制决定 **什么时候使用这些参数记忆**。

## 3. 相关方向现有方法与不足

| 方法方向 | 代表工作 | 主要做法 | 为什么不足以解决我们的问题 |
|---|---|---|---|
| RAG / MobileRAG | RAG, MobileRAG | 检索外部文档并拼接到上下文 | 知识更新灵活，但端侧检索、重排序、上下文编码开销高；没有回答哪些知识应参数化 |
| MLP Memory | MLP Memory | 将 kNN 检索器的输出分布蒸馏到 MLP 记忆模块 | 证明了检索知识可参数化，但 memory 通常静态、稠密、始终激活，端侧开销和干扰问题明显 |
| Memory Layers | Memory Layers at Scale | 通过稀疏 key-value memory layer 增加模型事实容量 | 更偏预训练结构扩展，未针对边缘设备上的动态读写、能耗和知识更新问题 |
| 文档参数化 | Doc-to-LoRA, SHINE | 将文档或上下文快速转换成 LoRA 参数 | 解决如何把文档变成参数，但没有考虑端侧什么时候值得固化、何时删除和如何路由 |
| 测试时记忆 | Titans, Atlas, Locas | 在推理或测试阶段写入神经记忆 | 支持动态记忆，但在线更新可能有训练开销，且没有系统建模移动端资源约束 |
| 情景到参数固化 | UniMem, MemVerse | 新知识先进入外部情景记忆，高频稳定知识再固化 | 思路接近，但主要关注记忆管理，没有把端侧能耗、存储、隐私和删除成本作为核心优化目标 |

因此，现有方法分别解决了“检索”“参数化”“长时记忆”“文档固化”等局部问题，但还没有系统回答：

> **在边缘设备上，如何按需读取参数记忆，并资源感知地写入参数记忆？**

## 4. 我们的方法设计

当前方法可以概括为：

> **On-Demand Read and Resource-Aware Write for Edge Parametric Memory**

中文可称为：

> **面向边缘大模型的按需读写参数化记忆机制**

系统由四个部分组成：

1. **Base LLM**  
   冻结的基础语言模型。在当前实验中使用 `Qwen2.5-0.5B-Instruct` 的 hidden state 作为真实 LLM 表征。

2. **Parametric Memory Bank**  
   由多个 memory experts 组成，每个 expert 是一个小型 MLP 记忆模块，用于存储基础模型缺失的 tail facts。

3. **Read Router**  
   输入当前样本的 hidden state、base logits entropy、logit gap 等信息，输出是否读取 memory 的概率 `p_read`。当 `p_read` 高于 threshold 时，激活 Top-K memory experts；否则直接使用 base 输出。

4. **Write Scheduler**  
   用于模拟资源感知写入策略。它根据知识项的访问频率、准确率收益、检索节省、写入能耗、存储成本、隐私风险和删除风险，决定哪些知识值得固化为参数记忆。

### 4.1 按需读取为什么能解决问题

Dense Memory 的问题是每个样本都激活 memory，即使基础模型已经知道答案，也会被 memory 输出干扰。我们的 Read Router 将样本分成两类：

- 对于 `common_fact` 和 `general`：跳过 memory，保留基础模型已有能力；
- 对于 `tail_fact`：激活 memory，用参数记忆补充缺失知识。

因此，该机制不仅降低了计算开销，还减少了 memory 对基础模型的负迁移。

### 4.2 资源感知写入为什么能解决问题

全量写入会导致存储增长、写入能耗和隐私风险增加；仅按频率写入又可能忽略知识稳定性和删除风险。我们的 Write Scheduler 将写入决策建模为效用优化：

```text
U(write) =
    accuracy_gain
  + retrieval_cost_saving
  + offline_availability_gain
  - write_energy
  - storage_cost
  - privacy_risk
  - deletion_cost
```

只有当一条知识的长期收益高于资源和风险成本时，才将其固化为参数记忆。

## 5. 算法流程

### 5.1 推理阶段：按需读取

```text
Input: x
1. 使用冻结 LLM 提取 hidden state h
2. Base head 计算基础预测 P_base
3. Read Router 根据 h、entropy、logit gap 计算 p_read
4. if p_read < threshold:
       直接输出 P_base
   else:
       选择 Top-K memory experts
       计算 memory 输出 P_mem
       融合 P_base 和 P_mem 得到最终预测
5. 输出答案
```

### 5.2 训练阶段：读取模块

```text
1. 构造 common_fact、tail_fact、general 三类样本
2. Base head 只在 common/general 上训练，模拟基础模型已有知识
3. 冻结 base head
4. 训练 memory experts 存储 tail facts
5. 训练 Read Router 判断是否需要读取 memory
6. 用 threshold sweep 分析准确率和激活率的权衡
```

### 5.3 写入阶段：资源感知固化

```text
Input: candidate knowledge items
1. 统计每条知识的访问频率、收益、稳定性和资源成本
2. 计算写入效用 U(write)
3. 在能耗和存储预算下选择高效用知识
4. 将选中的知识固化为 memory atom
5. 对过期、冲突或低价值知识进行降级或删除
```

## 6. 当前实验进展

当前实验分为三个阶段。

### 6.1 Toy Synthetic 实验

最初使用轻量 Bag-of-Words encoder 构造合成事实任务，验证机制是否可行。结果表明：

- Base 能处理 common/general，但无法处理 tail facts；
- Dense Memory 能补 tail facts，但严重干扰 common/general；
- Conditional Memory 能在较低激活率下同时保持三类样本性能。

该阶段证明了机制逻辑成立。

### 6.2 真实 LLM Hidden-State 实验

进一步使用冻结的 `Qwen2.5-0.5B-Instruct` hidden state，验证方法是否能迁移到真实 LLM 表征。

主实验结果：

| 方法 | Accuracy | Common Acc | Tail Acc | General Acc | Activation Rate |
|---|---:|---:|---:|---:|---:|
| Base | 0.6467 | 0.9718 | 0.0000 | 1.0000 | 0.0000 |
| Dense Memory | 0.3975 | 0.0410 | 0.9467 | 0.1763 | 1.0000 |
| Conditional Memory | 0.9725 | 0.9718 | 0.9467 | 1.0000 | 0.3442 |

结论：

- Base 对 tail facts 完全无能为力；
- Dense Memory 能存储 tail facts，但始终激活会破坏 common/general；
- Conditional Memory 只激活约 34.42% 的 memory，就能同时保持 common/general 并补充 tail facts。

这说明按需读取机制在真实 LLM 表征上是有效的。

### 6.3 全真实数据候选答案排序实验

进一步地，我们将实验从 synthetic facts 扩展到真实数据：

- `common_fact / tail_fact`：来自真实 SQuAD 问答样本；
- `general`：来自真实 AG News 文本分类样本；
- 表征模型：冻结的 `Qwen2.5-0.5B-Instruct` hidden state；
- 评估方式：除分类准确率外，额外对真实 QA 样本进行候选答案排序，并报告 EM、F1、MRR、Hits@K。

主实验结果如下：

| 方法 | Accuracy | Common Acc | Tail Acc | General Acc | Activation Rate | QA EM | QA F1 | QA MRR | Hits@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.6075 | 1.0000 | 0.0000 | 0.8706 | 0.0000 | 0.4897 | 0.4915 | 0.4952 | 0.4897 |
| Dense Memory | 0.4425 | 0.0936 | 0.9905 | 0.1995 | 1.0000 | 0.5513 | 0.5513 | 0.5866 | 0.6140 |
| Conditional Memory | 0.9525 | 0.9901 | 0.9905 | 0.8679 | 0.3567 | 0.9903 | 0.9903 | 0.9938 | 1.0000 |

分类型候选答案排序结果：

| 方法 | Common EM | Tail EM |
|---|---:|---:|
| Base | 1.0000 | 0.0000 |
| Dense Memory | 0.0936 | 0.9905 |
| Conditional Memory | 0.9901 | 0.9905 |

结论：

- Base 在 common QA 和 AG News general 上表现较好，但对 tail QA 完全失败，`rank_tail_em=0.0000`；
- Dense Memory 能将 tail QA 提升到 `rank_tail_em=0.9905`，但 common QA 和 general 几乎崩溃，说明始终激活 memory 会产生严重负迁移；
- Conditional Memory 只激活约 `35.67%` 的 memory，就能同时保持 common QA、tail QA 和真实 general，达到 `rank_qa_em=0.9903`、`rank_qa_f1=0.9903`、`rank_qa_mrr=0.9938`、`Hits@5=1.0000`。

这说明，在真实 QA 文本、真实答案和真实 general 文本上，按需读取参数记忆不仅能提高分类准确率，也能在候选答案排序 EM/F1 上显著优于 Base 和 Dense Memory。

### 6.4 资源感知写入仿真

写入策略仿真中，对比三种策略：

| 策略 | 写入数量 | Total Utility | Gain / Storage | Saving / Energy | Privacy Risk | Deletion Risk |
|---|---:|---:|---:|---:|---:|---:|
| All Write | 179 | -72.3508 | 0.1957 | 0.2866 | 0.2596 | 0.2291 |
| Frequency Only | 90 | -35.1523 | 0.2436 | 0.4850 | 0.2566 | 0.2471 |
| Resource-Aware | 21 | 7.7628 | 0.3811 | 0.8524 | 0.1201 | 0.1723 |

结论：

- Resource-Aware 写入数量最少；
- 单位存储收益和单位能耗节省最高；
- 隐私风险和删除风险最低；
- 总效用为正，说明资源感知写入更适合端侧场景。

## 7. 专业实验结果分析

### 7.1 容量实验

固定 `num_experts=8, top_k=2`，改变事实数量。

| num_facts | Base Acc | Dense Acc | Conditional Acc | Common Acc | Tail Acc | Activation Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.6467 | 0.3975 | 0.9725 | 0.9718 | 0.9467 | 0.3442 |
| 100 | 0.6258 | 0.4125 | 0.9608 | 0.8786 | 1.0000 | 0.3500 |
| 160 | 0.5875 | 0.3417 | 0.9150 | 0.8180 | 0.9313 | 0.3575 |
| 240 | 0.5517 | 0.3617 | 0.8500 | 0.6692 | 0.8765 | 0.3583 |

分析：

- Conditional Memory 在所有规模上都明显优于 Base 和 Dense Memory；
- 随着 `num_facts` 增大，性能逐步下降，说明固定 memory bank 存在容量上限；
- 激活率稳定在 34%-36%，说明 router 没有通过盲目激活 memory 来换取准确率。

### 7.2 专家结构消融

| num_experts | top_k | Conditional Acc | Common Acc | Tail Acc | Activation Rate | Expert Cost |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.9767 | 0.9718 | 0.9588 | 0.3442 | 0.3442 |
| 8 | 1 | 0.8908 | 0.9718 | 0.7094 | 0.3442 | 0.3442 |
| 8 | 2 | 0.9725 | 0.9718 | 0.9467 | 0.3442 | 0.6883 |
| 16 | 2 | 0.9858 | 0.9769 | 0.9806 | 0.3492 | 0.6983 |
| 16 | 4 | 0.9892 | 0.9744 | 0.9927 | 0.3483 | 1.3933 |

分析：

- 更大的专家池和更高 Top-K 可以提升 tail fact 记忆能力；
- `16 experts + top_k=4` 最强，但计算成本最高；
- `16 experts + top_k=2` 是较好的准确率/开销折中；
- 专家数量和 Top-K 是后续优化端侧开销的重要超参数。

### 7.3 Router 噪声鲁棒性

| Router Noise | Conditional Acc | Common Acc | Tail Acc | Activation Rate | Router F1 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.9725 | 0.9718 | 0.9467 | 0.3442 | 1.0000 |
| 0.1 | 0.9850 | 0.9718 | 0.9831 | 0.3542 | 0.9857 |
| 0.2 | 0.9875 | 0.9718 | 0.9903 | 0.3592 | 0.9787 |
| 0.3 | 0.9717 | 0.9718 | 0.9443 | 0.3342 | 0.9803 |

分析：

- Router 在 0.1-0.3 标签噪声下仍保持较高性能；
- 说明按需读取机制不完全依赖完美 router 监督；
- 该结果为后续使用真实 teacher 或 RAG-derived label 提供了鲁棒性依据。

## 8. 当前能够得出的结论

基于当前实验，可以得出以下阶段性结论：

1. **长尾知识缺口真实存在**  
   在 SQuAD + AG News 的真实数据实验中，Base 对 common QA 和 general 分类表现较好，但 tail QA 的候选答案排序 EM 为 0，说明需要外部记忆补充。

2. **始终激活参数记忆不是好方案**  
   Dense Memory 能显著补充 tail QA，但会严重破坏 common QA 和 general 分类，说明 memory 需要被选择性调用。

3. **按需读取机制有效**  
   Conditional Memory 通过 Read Router 只在需要时激活 memory，以约 34%-36% 的激活率显著提升 tail QA，并保持 common QA 和真实 general。当前真实数据实验中，Conditional Memory 达到 `rank_qa_em=0.9903`、`rank_qa_f1=0.9903`、`rank_qa_mrr=0.9938`。

4. **Memory capacity 是后续关键问题**  
   当事实数量增大时，固定规模 memory experts 性能下降，说明后续需要研究可扩展 memory bank、动态增加 experts 或资源感知写入。

5. **专家结构存在准确率/开销权衡**  
   更大专家池和更高 Top-K 提升准确率，但增加计算成本。当前较优折中是 `16 experts + top_k=2`。

6. **Router 具有初步鲁棒性**  
   在读取标签存在噪声时，方法仍能保持较好性能，说明该机制具备进一步扩展到真实任务的潜力。

## 9. 当前仍存在的问题

尽管当前结果比较有说服力，但仍有几个限制：

1. **真实 QA 实验仍处于候选答案排序阶段**  
   当前已经支持 SQuAD 或本地 QA JSONL 作为 common/tail 数据，同时支持 AG News 或本地分类 JSONL 作为真实 general 数据，并加入了候选答案排序 EM/F1。但这仍然不是自由生成式 QA，下一步需要让 memory 输出参与 LLM token 解码或候选答案重排序，并在 Natural Questions、TriviaQA、HotpotQA 等更复杂数据集上系统验证。

2. **还没有多随机种子**  
   当前实验主要是单 seed，需要补充 3-5 个 seed，验证结果稳定性。

3. **Memory 形式仍是 MLP experts**  
   当前 memory 是小型 MLP，后续可以比较 micro-LoRA、MLP Memory、memory atoms 等不同参数化形式。

4. **写入策略仍是仿真**  
   Resource-Aware Write 目前是策略仿真，还没有和真实训练过程完全闭环。

5. **缺少真实端侧系统指标**  
   当前主要报告 activation rate 和 expert cost，还需要进一步测量显存、延迟、能耗或移动端代理指标。

6. **缺少来源追踪和删除验证**  
   参数记忆写入后如何溯源、删除和验证遗忘，目前还没有展开实验。

## 10. 后续可以引出的研究内容

当前结果自然引出以下研究方向：

### 10.1 可扩展参数记忆容量

容量实验表明固定 memory bank 会随事实数量增加而性能下降。后续可以研究：

- 动态增加 memory experts；
- 按主题聚类 memory atoms；
- 稀疏专家路由与负载均衡；
- memory compression 和淘汰机制。

### 10.2 真实 QA 场景下的按需读取

将 synthetic facts 替换为真实 QA 数据。当前代码已经支持：

- HuggingFace QA 数据集，例如 SQuAD；
- 本地 JSONL QA 文件；
- HuggingFace general 数据集，例如 AG News；
- 本地 JSONL general 分类文件。

优先顺序建议为：

1. SQuAD：用于快速调通真实数据流程；
2. Natural Questions / TriviaQA：用于开放域知识问答；
3. HotpotQA：用于多跳组合事实；
4. LoCoMo：用于长程对话记忆。

目标是验证同样的模式是否存在于自然语言问答：

```text
Base 缺 tail knowledge
Dense Memory 过度干扰
Conditional Memory 低激活率补充知识
```

需要注意，当前真实 QA 实现虽然使用真实问题、真实答案和真实 general 文本，并已经加入候选答案排序 EM/F1，但还不是最终自由生成式 QA。后续可以进一步将 memory 输出接入候选答案重排序或 LLM token 生成。

### 10.3 资源感知写入闭环

将当前 Write Scheduler 仿真扩展为真实闭环：

1. 新知识先进入 episodic memory；
2. 根据访问频率和收益决定是否固化；
3. 固化为 memory expert 或 micro-LoRA；
4. 推理时由 Read Router 按需读取；
5. 低价值或过期 memory 被删除或降级。

### 10.4 端侧效率评估

后续需要将 `activation_rate` 转换为更真实的系统指标：

- TTFT；
- TPOT；
- GPU 显存；
- memory 参数读取量；
- 端侧能耗代理；
- 本地检索调用次数下降比例。

### 10.5 可删除和可溯源参数记忆

如果 memory 写入用户隐私知识，需要进一步研究：

- 每个 memory atom 的来源标记；
- 删除某条知识后如何卸载对应参数；
- 删除后是否仍能被模型恢复；
- 参数记忆泄漏风险评估。

## 11. 当前汇报建议

在组会中可以这样概括当前阶段：

> 我们围绕“边缘大模型是否应该一直读取参数记忆”这一问题，设计了按需读取的参数记忆机制。实验表明，在冻结 Qwen2.5-0.5B hidden state 上，Base 对 tail facts 准确率为 0，Dense Memory 虽能补充 tail facts 但严重破坏 common/general；而 Conditional Memory 通过 router 按需激活，在约 34%-36% 的 memory 激活率下显著提升 tail facts，并保持 common/general。进一步容量实验说明固定 memory bank 存在容量瓶颈，专家结构消融说明 memory expert 数量和 Top-K 存在准确率/开销权衡，router 噪声实验说明方法对不完美监督具有初步鲁棒性。下一步将接入真实 QA 数据，并将资源感知写入从仿真扩展为完整闭环。
> 我们围绕“边缘大模型是否应该一直读取参数记忆”这一问题，设计了按需读取的参数记忆机制。最新实验使用 SQuAD 构造真实 common/tail QA，并使用 AG News 作为真实 general 任务。在冻结 Qwen2.5-0.5B hidden state 上，Base 对 tail QA 的候选答案排序 EM 为 0；Dense Memory 虽将 tail EM 提升到 99.05%，但 common EM 降至 9.36%、general accuracy 降至 19.95%；Conditional Memory 在仅 35.67% 的 memory 激活率下达到 QA EM 99.03%、QA F1 99.03%、MRR 99.38%、Hits@5 100%，同时保持 common EM 99.01%、tail EM 99.05% 和 general accuracy 86.79%。这说明按需读取参数记忆在真实数据上能够补充长尾知识并避免负迁移。下一步将扩展到更难的开放域 QA 和生成式 QA。
