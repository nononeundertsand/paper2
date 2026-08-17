# 端到端实验系统设计

## 1. 目标

当前研究问题是：

> 在端侧资源约束下，外部参数记忆是否应该始终激活？如果不应该，能否通过轻量路由器按需读取参数记忆，在补充 tail knowledge 的同时避免破坏 base model 已有能力？

端到端实验系统围绕这条证据链设计：

1. Base 对 tail knowledge 不足；
2. Dense Memory 能补 tail，但会破坏 common/general；
3. Conditional Memory 能低激活率补 tail，同时保持 common/general；
4. 该现象在真实 QA + 真实 general 数据上成立；
5. 容量、专家结构、router 噪声、写入策略实验进一步说明方法边界和扩展方向。

## 2. 数据与模型

默认正式设置：

- QA 数据：SQuAD
- General 数据：AG News
- Backbone：`Qwen2.5-0.5B-Instruct`
- Feature：冻结 LLM 最后一层 hidden state
- Memory：MLP memory experts
- Router：Read Router
- QA 指标：候选答案排序 EM/F1、MRR、Hits@5
- Efficiency proxy：activation rate、expert cost

## 3. 实验组件

### 3.1 主实验

比较：

- Base
- Dense Memory
- Conditional Memory

输出：

- overall accuracy
- common/tail/general accuracy
- QA EM/F1/MRR/Hits@5
- common/tail EM/F1
- activation rate
- router precision/recall/F1

### 3.2 多随机种子

默认 seed：

```text
42, 43, 44
```

用于生成 mean/std，避免单次结果偶然。

### 3.3 容量实验

默认：

```text
num_facts = 60, 120, 240, 360
```

用于观察 memory capacity。

### 3.4 专家结构消融

默认：

```text
(num_experts=4, top_k=1)
(num_experts=8, top_k=1)
(num_experts=8, top_k=2)
(num_experts=16, top_k=2)
(num_experts=16, top_k=4)
```

用于分析准确率和专家激活成本的权衡。

### 3.5 Router 噪声鲁棒性

默认：

```text
router_label_noise = 0.0, 0.1, 0.2, 0.3
```

用于模拟不完美 teacher / routing label。

### 3.6 资源感知写入

比较：

- all_write
- frequency_only
- resource_aware

用于支撑“写什么”的第二个 idea。

## 4. 一键脚本

### 4.1 Smoke Test

用于快速确认环境、数据、模型路径和代码都能跑通。

```powershell
.\scripts\run_e2e_smoke.bat D:\models\Qwen2.5-0.5B-Instruct
```

输出：

```text
outputs\e2e_smoke\summary\smoke_summary.csv
```

### 4.2 Main Experiment

跑主实验的 3 个随机种子 + 写入策略 + 汇总。

```powershell
.\scripts\run_e2e_main.bat D:\models\Qwen2.5-0.5B-Instruct
```

输出：

```text
outputs\e2e\summary\e2e_main_summary.csv
outputs\e2e\summary\e2e_main_aggregate.csv
```

### 4.3 Full Experiment

跑完整实验包：

- 3 seeds main experiment
- capacity sweep
- expert sweep
- router noise sweep
- write scheduler
- summary
- aggregate

```powershell
.\scripts\run_e2e_full.bat D:\models\Qwen2.5-0.5B-Instruct
```

输出：

```text
outputs\e2e\summary\e2e_full_summary.csv
outputs\e2e\summary\e2e_full_aggregate.csv
```

## 5. 推荐执行顺序

1. 先跑 smoke：

```powershell
.\scripts\run_e2e_smoke.bat D:\models\Qwen2.5-0.5B-Instruct
```

2. smoke 成功后跑 main：

```powershell
.\scripts\run_e2e_main.bat D:\models\Qwen2.5-0.5B-Instruct
```

3. main 结果稳定后跑 full：

```powershell
.\scripts\run_e2e_full.bat D:\models\Qwen2.5-0.5B-Instruct
```

## 6. 论文中推荐使用的核心表格

### 表 1：主实验

Base / Dense Memory / Conditional Memory 在真实 QA + AG News 上的对比。

核心列：

- Accuracy
- Common Acc
- Tail Acc
- General Acc
- QA EM
- QA F1
- QA MRR
- Hits@5
- Activation Rate

### 表 2：多 seed 稳定性

报告 mean ± std。

核心列：

- Conditional QA EM
- Conditional Tail EM
- Conditional General Acc
- Activation Rate

### 表 3：容量实验

横向比较 `num_facts`。

### 表 4：专家结构消融

横向比较 `num_experts` 和 `top_k`。

### 表 5：资源感知写入

比较 all-write、frequency-only、resource-aware。

## 7. 仍未覆盖的高阶实验

当前端到端系统仍是候选答案排序，不是自由生成式 QA。后续冲 A 类会议建议继续补：

1. Natural Questions / TriviaQA / HotpotQA；
2. 生成式 EM/F1；
3. 真实端侧延迟、显存、能耗；
4. memory 删除、冲突、过期；
5. micro-LoRA / MLP Memory / memory atoms 对比；
6. 与 Selective RAG、Static LoRA、Adaptive Retrieval 的强 baseline 对比。

