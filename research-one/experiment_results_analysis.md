# 实验结果分析：真实 LLM Hidden-State 按需参数记忆

## 1. 总体结论

本批实验基于冻结的 `Qwen2.5-0.5B-Instruct` hidden state，验证了按需读取参数记忆机制在真实 LLM 表征上的有效性。整体结果显示：

- Base head 对 `tail_fact` 基本没有能力，`acc_tail_fact=0`，说明长尾知识缺口明确存在。
- Dense Memory 能显著提升 `tail_fact`，但由于始终激活，会严重破坏 `common_fact` 和 `general`。
- Conditional Memory 能以约 34%-36% 的 memory 激活率补充长尾知识，同时保留 common/general 能力。
- 随着 `num_facts` 增大，tail fact 记忆能力逐渐下降，体现出 memory capacity 限制。
- 增加专家数和 Top-K 激活能提升 tail fact 存储能力，但会增加专家计算成本。
- Router 在一定监督噪声下仍然稳定，说明按需读取策略具备初步鲁棒性。

## 2. 容量实验：num_facts 扩展

固定 `num_experts=8, top_k=2`，改变可记忆事实数量。

| num_facts | best threshold | Base Acc | Dense Acc | Conditional Acc | Common Acc | Tail Acc | Activation Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.70 | 0.6467 | 0.3975 | 0.9725 | 0.9718 | 0.9467 | 0.3442 |
| 100 | 0.70 | 0.6258 | 0.4125 | 0.9608 | 0.8786 | 1.0000 | 0.3500 |
| 160 | 0.50 | 0.5875 | 0.3417 | 0.9150 | 0.8180 | 0.9313 | 0.3575 |
| 240 | 0.70 | 0.5517 | 0.3617 | 0.8500 | 0.6692 | 0.8765 | 0.3583 |

### 观察

1. Base 在所有设置下 `tail_fact=0`，说明这些 tail facts 确实没有被 base head 学到。
2. Conditional Memory 在 `num_facts=60/100` 时表现很强，整体准确率分别为 `0.9725` 和 `0.9608`。
3. 当 `num_facts` 增大到 `160/240`，整体准确率下降到 `0.9150/0.8500`，说明当前 memory bank 容量开始不足。
4. 激活率始终维持在约 `0.34-0.36`，说明 router 没有因为事实数量增加而盲目激活更多 memory。

### 可写结论

容量实验表明，按需读取机制在事实规模增加时仍能保持稳定的激活率，但固定规模 memory experts 的存储能力会随事实数量增加而下降。这说明后续需要研究可扩展 memory bank 或动态写入机制。

## 3. 专家结构消融

固定 `num_facts=60`，改变专家数量与 Top-K 激活数。

| num_experts | top_k | Best Threshold | Conditional Acc | Common Acc | Tail Acc | Activation Rate | Expert Cost = act * top_k |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.70 | 0.9767 | 0.9718 | 0.9588 | 0.3442 | 0.3442 |
| 8 | 1 | 0.50 | 0.8908 | 0.9718 | 0.7094 | 0.3442 | 0.3442 |
| 8 | 2 | 0.70 | 0.9725 | 0.9718 | 0.9467 | 0.3442 | 0.6883 |
| 16 | 2 | 0.30 | 0.9858 | 0.9769 | 0.9806 | 0.3492 | 0.6983 |
| 16 | 4 | 0.30 | 0.9892 | 0.9744 | 0.9927 | 0.3483 | 1.3933 |

### 观察

1. `16 experts + top_k=4` 取得最高 tail accuracy `0.9927` 和整体 accuracy `0.9892`，但专家计算成本最高。
2. `16 experts + top_k=2` 已经达到 `0.9858` overall 和 `0.9806` tail，成本约为 `0.6983`，比 top_k=4 更经济。
3. `8 experts + top_k=1` 明显较差，tail accuracy 只有 `0.7094`，说明 Top-K 激活数过小可能限制组合表达能力。
4. `4 experts + top_k=1` 反而较好，可能是该设置下专家划分更稳定，但需要多随机种子验证。

### 可写结论

专家结构消融表明，增加 memory experts 和 Top-K 激活数可以提升长尾知识存储能力，但会提高专家计算成本。当前结果中 `16 experts + top_k=2` 是较好的准确率/开销折中。

## 4. Router 噪声鲁棒性

固定 `num_facts=60, num_experts=8, top_k=2`，在训练 router 时加入标签噪声，评估标签仍保持干净。

| router_label_noise | Best Threshold | Conditional Acc | Common Acc | Tail Acc | Activation Rate | Router F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.70 | 0.9725 | 0.9718 | 0.9467 | 0.3442 | 1.0000 |
| 0.1 | 0.30 | 0.9850 | 0.9718 | 0.9831 | 0.3542 | 0.9857 |
| 0.2 | 0.50 | 0.9875 | 0.9718 | 0.9903 | 0.3592 | 0.9787 |
| 0.3 | 0.50 | 0.9717 | 0.9718 | 0.9443 | 0.3342 | 0.9803 |

### 观察

1. 在 `0.1-0.2` 噪声下，性能没有下降，甚至略高，可能是噪声带来了一定正则化效果。
2. `0.3` 噪声下整体准确率仍有 `0.9717`，tail accuracy 仍有 `0.9443`。
3. Router F1 在高噪声下仍接近 `0.98`，说明 router 对标签噪声较稳健。

### 可写结论

Router 噪声实验表明，即使训练阶段的读取监督存在一定错误，按需读取机制仍能保持较高准确率和稳定激活率，说明该机制对不完美 teacher 信号具有初步鲁棒性。

## 5. 可用于汇报的核心表述

可以在组会中这样总结：

> 在冻结 Qwen2.5-0.5B hidden state 的实验中，Base 对 tail facts 的准确率始终为 0，说明长尾知识缺口明确存在；Dense Memory 虽然能提升 tail facts，但会严重破坏 common/general；Conditional Memory 通过 router 按需激活 memory，在约 34%-36% 的激活率下显著提升 tail facts，并保持 common/general 能力。容量实验进一步表明，固定 memory bank 在事实数量增加时出现性能下降；专家结构消融说明更多专家和更高 Top-K 可以提升存储能力，但会增加计算成本；噪声实验表明 router 对不完美监督具有一定鲁棒性。

## 6. 后续建议

1. 对关键设置增加 3 个随机种子，验证稳定性。
2. 对容量实验补充 `num_facts=320/480`，观察性能下降趋势是否平滑。
3. 在专家结构实验中报告计算代理指标：`activation_rate * top_k`。
4. 接入真实 QA 子集，验证同样模式是否存在于自然语言问答。
5. 将当前 synthetic fact 构造从固定模板扩展为多模板、多 paraphrase，以降低模板记忆影响。

