# 按需读写参数化记忆的专业实验设计

## 1. 实验目标

当前 toy synthetic 和 Qwen2.5 hidden-state 实验已经证明：在存在 common facts、tail facts 和 general samples 的场景下，始终激活参数记忆会干扰基础模型，而按需读取机制可以在较低激活率下补充长尾知识。下一阶段实验需要从“可行性验证”升级为“论文式验证”，重点回答以下问题：

1. **有效性**：按需读取能否在真实 LLM 表征上显著提升长尾知识准确率，同时保持通用能力？
2. **效率**：相比 dense memory，按需读取能否显著减少 memory 激活率和专家计算量？
3. **容量**：随着可记忆事实数量增加，memory experts 的存储能力如何变化？
4. **结构设计**：专家数量、Top-K 激活数、threshold 对准确率和开销有什么影响？
5. **鲁棒性**：在 router 监督不完美或样本分布变化时，方法是否仍然稳定？
6. **写入策略**：资源感知固化是否比全量写入或频率写入更适合端侧？

## 2. 主实验：真实 LLM Hidden-State 上的按需读取

### 2.1 研究问题

冻结真实 LLM 后，仅训练轻量 Read Router 和 memory experts，是否仍能实现“保留 common/general，补充 tail facts”的效果？

### 2.2 实验设置

- Backbone：Qwen2.5-0.5B-Instruct，冻结参数。
- Feature：最后一层 hidden state，mean pooling。
- 数据：synthetic facts，包含 common facts、tail facts、general samples。
- 推荐配置：
  - `num_facts=60`
  - `base_train_size=4000`
  - `memory_train_size=6000`
  - `test_size=1200`
  - `base_epochs=20`
  - `memory_epochs=30`
  - `num_experts=8`
  - `top_k_experts=2`

### 2.3 Baselines

1. **Base**：只训练 base head，不使用 memory。
2. **Dense Memory**：每个样本都激活 memory。
3. **Conditional Memory**：由 router 按需激活 memory。

### 2.4 指标

- `accuracy`
- `acc_common_fact`
- `acc_tail_fact`
- `acc_general`
- `activation_rate`
- `router_precision`
- `router_recall`
- `router_f1`

### 2.5 预期结论

- Base 对 common/general 表现好，但对 tail facts 表现差。
- Dense Memory 能补 tail facts，但会破坏 common/general。
- Conditional Memory 以较低激活率同时保持 common/general 并补 tail facts。

### 2.6 推荐图表

- 表格：Base / Dense / Conditional 对比。
- 柱状图：三类样本准确率对比。
- 折线图：threshold vs accuracy / activation rate。

## 3. 容量实验：可记忆事实数量扩展

### 3.1 研究问题

当 tail facts 数量增加时，固定规模 memory experts 的记忆能力如何下降？按需读取机制是否仍能保持较低干扰？

### 3.2 自变量

`num_facts`：

```text
60, 100, 160, 240
```

保持：

- `num_experts=8`
- `top_k_experts=2`
- 训练轮数固定。

### 3.3 指标

- `acc_tail_fact`
- `acc_common_fact`
- `activation_rate`
- `router_f1`
- `accuracy`

### 3.4 预期观察

- `num_facts` 增加后，tail fact 准确率可能下降。
- 如果 router 仍能保持 common/general，说明读取机制稳定。
- 如果 tail fact 下降明显，说明需要增加 expert 数量或改进 memory capacity。

### 3.5 推荐图表

- 横轴：`num_facts`
- 纵轴 1：`acc_tail_fact`
- 纵轴 2：`activation_rate`

## 4. 专家结构消融：专家数量与 Top-K 激活

### 4.1 研究问题

memory experts 的数量和每次激活专家数如何影响准确率和计算开销？

### 4.2 自变量

专家数量：

```text
num_experts = 4, 8, 16
```

激活专家数：

```text
top_k_experts = 1, 2, 4
```

建议先做小网格：

| num_experts | top_k |
|---|---|
| 4 | 1 |
| 8 | 1 |
| 8 | 2 |
| 16 | 2 |
| 16 | 4 |

### 4.3 指标

- `accuracy`
- `acc_tail_fact`
- `activation_rate`
- `expert_activation_cost = activation_rate * top_k_experts`
- 训练时间，可选。

### 4.4 预期观察

- `top_k=1` 计算最低，但可能容量不足。
- `top_k=2` 可能是较好的准确率/开销平衡点。
- `top_k=4` 准确率提升有限，但计算成本更高。

## 5. Threshold 消融：准确率和激活率权衡

### 5.1 研究问题

read threshold 如何影响过度激活和漏激活？

### 5.2 自变量

```text
threshold = 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
```

### 5.3 指标

- `accuracy`
- `acc_tail_fact`
- `activation_rate`
- `router_precision`
- `router_recall`
- `router_f1`

### 5.4 预期观察

- 低 threshold：memory 过度激活，common/general 可能下降。
- 高 threshold：漏掉 tail facts，`acc_tail_fact` 下降。
- 中间 threshold：准确率和激活率达到较优平衡。

## 6. Router 鲁棒性实验

### 6.1 研究问题

当 router 监督标签存在噪声时，按需读取是否仍然有效？

### 6.2 自变量

```text
router_label_noise = 0.0, 0.1, 0.2, 0.3
```

### 6.3 指标

- `accuracy`
- `acc_tail_fact`
- `activation_rate`
- `router_f1`

### 6.4 预期观察

- 少量噪声下，router 应该仍保持可用。
- 高噪声下，activation rate 和 tail accuracy 可能恶化。
- 该实验可以支撑方法在真实 teacher 不完美时的鲁棒性。

## 7. 写入策略实验：资源感知固化

### 7.1 研究问题

资源感知写入是否比全量写入和频率写入更适合边缘设备？

### 7.2 Baselines

1. **All Write**：预算内尽可能写入。
2. **Frequency Only**：按访问频率写入。
3. **Resource-Aware**：综合收益、能耗、存储、隐私和删除风险。

### 7.3 指标

- `num_written`
- `expected_accuracy_gain`
- `retrieval_saving`
- `storage`
- `write_energy`
- `privacy_risk`
- `deletion_risk`
- `total_utility`
- `gain_per_storage`
- `saving_per_energy`

### 7.4 预期观察

Resource-Aware 写入更少，但单位存储收益和单位能耗收益更高，同时隐私和删除风险更低。

## 8. 真实数据集扩展计划

当前 synthetic facts 的优点是可控，缺点是还不是真实 QA。后续可以逐步接入：

1. **Natural Questions / TriviaQA 子集**
   - Common facts：base train 中出现的问题。
   - Tail facts：只在 memory train 中出现的问题。
   - General：简单分类或开放域非知识密集问题。

2. **HotpotQA**
   - 用于多跳知识和组合事实。

3. **LoCoMo / 长程对话记忆**
   - 用于模拟端侧长期个人记忆。

真实数据集阶段的重点不是一开始追求 SOTA，而是验证同样的模式是否存在：

```text
Base 缺 tail facts
Dense Memory 过度干扰
Conditional Memory 低激活率补充 tail facts
```

## 9. 推荐实验执行顺序

1. 主实验：Qwen hidden-state，`num_facts=60`。
2. Threshold 消融：确认 0.3-0.7 区间稳定。
3. 容量实验：`num_facts=60/100/160/240`。
4. 专家结构消融：`num_experts` 与 `top_k_experts`。
5. Router 噪声鲁棒性。
6. 写入策略仿真。
7. 接入真实 QA 子集。

## 10. 当前可以写入论文的初步结论

目前已有结果可以支持以下阶段性结论：

> 在冻结 Qwen2.5-0.5B hidden state 上，base head 对 tail facts 准确率为 0，而 dense memory 虽将 tail facts 提升到较高水平，但严重破坏 common/general；按需读取机制在约三分之一的 memory 激活率下同时保持 common、tail 和 general 三类性能，说明该方法在真实 LLM 表征上具有可行性。

