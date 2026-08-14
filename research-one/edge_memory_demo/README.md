# Edge Parametric Memory Demo

这是一个用于初步验证“面向边缘大模型的按需读写参数化记忆机制”的轻量实验原型。当前版本不依赖 HuggingFace 模型和外部数据集，只依赖 PyTorch，通过合成事实问答任务验证两个核心想法：

1. **按需读取**：Read Router 判断当前样本是否需要激活 memory experts，只在长尾事实问题上读取参数记忆。
2. **资源感知写入**：Write Scheduler 根据知识访问频率、准确率收益、能耗、存储、隐私和删除风险，决定哪些知识值得从情景记忆固化为参数记忆。

## 目录结构

```text
edge_memory_demo/
├── README.md
├── requirements.txt
├── run_scheduler.py
├── run_synthetic.py
├── run_llm_synthetic.py
├── run_threshold_sweep.py
├── collect_results.py
├── scripts/
│   ├── collect_results.bat
│   ├── run_llm_capacity_sweep.bat
│   ├── run_llm_expert_sweep.bat
│   ├── run_llm_synthetic.bat
│   ├── run_llm_synthetic.sh
│   ├── run_llm_router_noise_sweep.bat
│   ├── run_scheduler.bat
│   ├── run_scheduler.sh
│   ├── run_synthetic.bat
│   ├── run_synthetic.sh
│   ├── run_threshold_sweep.bat
│   ├── run_threshold_sweep_hard.bat
│   ├── run_threshold_sweep_hard.sh
│   └── run_threshold_sweep.sh
└── src/
    └── edge_memory/
        ├── __init__.py
        ├── collect_results.py
        ├── data.py
        ├── llm_features.py
        ├── model.py
        ├── scheduler.py
        ├── threshold_sweep.py
        └── train.py
```

## 环境安装

Windows 服务器：

```bat
cd research-one\edge_memory_demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS：

```bash
cd research-one/edge_memory_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果服务器离线，可以先在联网机器下载 wheel：

```bash
pip download -r requirements.txt -d wheels
```

然后将 `wheels/` 拷贝到服务器，在服务器执行：

```bat
pip install --no-index --find-links wheels -r requirements.txt
```

如果只跑 toy synthetic 实验，理论上只需要 `torch`；如果跑真实 LLM hidden-state 实验，需要额外安装 `transformers`。当前 `requirements.txt` 已包含两者。

## 实验一：按需读取参数记忆

直接运行：

```bat
python run_synthetic.py --device auto
```

或使用脚本：

```bat
scripts\run_synthetic.bat
```

该实验会自动生成三类样本：

- `common_fact`：base 模型训练阶段见过的常见事实，不需要 memory。
- `tail_fact`：base 模型没学过的长尾事实，需要 memory。
- `general`：普通模式任务，不需要 memory。

训练流程：

1. 训练一个小型 base classifier，只学习 common facts 和 general samples。
2. 冻结 base。
3. 训练 Conditional Memory，包括 Read Router 和多个 memory experts。
4. 对比 base、dense memory、conditional memory。

重点关注输出指标：

- `acc_tail_fact`：长尾事实准确率，代表 memory 是否补上 base 的知识缺口。
- `activation_rate`：硬路由下的记忆激活率，越低说明端侧开销越低。
- `router_f1`：Read Router 是否能识别哪些样本需要 memory。
- `dense_memory` vs `conditional_memory`：前者总是激活 memory，后者按需激活。

输出文件：

```text
outputs/synthetic_run/
├── base_model.pt
├── conditional_memory.pt
└── metrics.json
```

一个理想的初步现象是：

- `base` 在 `tail_fact` 上准确率较低；
- `conditional_memory` 显著提升 `tail_fact` 准确率；
- `conditional_memory` 的 `activation_rate` 明显低于 `dense_memory`；
- `common_fact` 和 `general` 不因 memory 引入明显退化。

## 实验二：资源感知写入仿真

直接运行：

```bat
python run_scheduler.py
```

或使用脚本：

```bat
scripts\run_scheduler.bat
```

该实验会模拟一批候选知识项，每个知识项包含：

- 访问频率；
- 预期准确率收益；
- 知识稳定性；
- 检索成本；
- 写入能耗；
- 参数存储成本；
- 隐私风险；
- 删除风险。

对比三种策略：

- `all_write`：只要预算允许就写入。
- `frequency_only`：只根据访问频率写入。
- `resource_aware`：综合收益、能耗、存储、隐私和删除风险写入。

输出文件：

```text
outputs/scheduler_run/
└── scheduler_metrics.json
```

重点关注：

- `expected_accuracy_gain`
- `retrieval_saving`
- `storage`
- `write_energy`
- `privacy_risk`
- `deletion_risk`
- `mean_utility`

如果 `resource_aware` 能在接近收益下减少写入数量、隐私风险和删除风险，就说明 Idea 2 有进一步实验价值。

## 实验三：Read Router Threshold 消融

直接运行：

```bat
python run_threshold_sweep.py --device auto
```

或使用脚本：

```bat
scripts\run_threshold_sweep.bat
```

该实验只训练一次 base 和 conditional memory，然后用不同 threshold 评估硬路由效果。默认扫描：

```text
0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
```

也可以手动指定：

```bat
python run_threshold_sweep.py --device auto --thresholds 0.05,0.10,0.20,0.30,0.50,0.70,0.90
```

如果默认任务过于干净，可能所有 threshold 得到相同结果。这表示 router 已经把需要 memory 的样本和不需要 memory 的样本完全分开。为了观察更真实的 trade-off，可以运行 harder ablation：

```bat
scripts\run_threshold_sweep_hard.bat
```

或手动运行：

```bat
python run_threshold_sweep.py --device auto ^
  --output-dir outputs\threshold_sweep_hard ^
  --thresholds 0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95 ^
  --router-label-noise 0.15 ^
  --router-loss-weight 0.25 ^
  --sparsity-weight 0.08
```

其中 `--router-label-noise` 只扰动训练阶段的 router 监督标签，评估标签仍然保持干净，用来模拟真实场景中 teacher 或路由监督不完美的情况。

输出文件：

```text
outputs/threshold_sweep/
├── base_model.pt
├── conditional_memory.pt
├── threshold_sweep.csv
└── threshold_sweep.json
```

重点关注：

- `threshold` 越低，`activation_rate` 通常越高，memory 使用更多；
- `threshold` 越高，`activation_rate` 通常越低，但可能漏掉需要 memory 的 tail facts；
- 如果在较宽 threshold 范围内 `accuracy` 和 `acc_tail_fact` 都稳定，说明 router 的可分性较好；
- 如果高 threshold 下 `router_recall` 降低，说明过于保守，会漏读 memory。

## 实验四：真实 LLM Hidden-State 验证

前面三个实验是合成 toy 验证，encoder 是轻量 Bag-of-Words。为了进一步验证方法能否迁移到真实 LLM 表征，可以运行真实 LLM hidden-state 实验。该实验会：

1. 加载一个 HuggingFace CausalLM，例如 Qwen2.5-0.5B-Instruct。
2. 冻结 LLM，不更新任何 LLM 参数。
3. 将 synthetic prompts 输入 LLM，抽取最后一层 hidden states。
4. 在这些真实 LLM features 上训练 base classifier、Read Router 和 memory experts。
5. 评估 dense memory 与 conditional memory 在不同 threshold 下的表现。

推荐先用较小模型：

```bat
python run_llm_synthetic.py --model-name-or-path Qwen/Qwen2.5-0.5B-Instruct --device auto --fp16
```

如果服务器不能联网下载模型，可以先把模型下载到本地目录，然后传本地路径：

```bat
python run_llm_synthetic.py --model-name-or-path D:\models\Qwen2.5-0.5B-Instruct --device auto --fp16
```

也可以使用脚本：

```bat
scripts\run_llm_synthetic.bat Qwen/Qwen2.5-0.5B-Instruct
```

如果模型需要自定义代码，追加：

```bat
python run_llm_synthetic.py --model-name-or-path D:\models\your_model --device auto --fp16 --trust-remote-code
```

输出文件：

```text
outputs/llm_synthetic/
├── llm_feature_base.pt
├── llm_feature_conditional_memory.pt
├── llm_synthetic_metrics.json
└── llm_threshold_sweep.csv
```

重点关注：

- `llm_feature_base` 的 `acc_tail_fact` 是否明显低于 common/general；
- `llm_feature_dense_memory` 是否能补 tail facts，但可能干扰 common facts；
- `llm_feature_conditional_threshold_*` 是否能在较低 `activation_rate` 下保持较高准确率；
- 如果真实 LLM features 上仍能复现 toy 实验趋势，说明方法从合成 encoder 迁移到真实 LLM 表征是可行的。

注意：该实验仍然使用合成 fact 数据，只是 encoder 换成真实 LLM。它是从 toy feasibility 到真实 QA 实验之间的中间验证。下一步才是接入 NQ、TriviaQA、HotpotQA 等真实数据集。

## 专业实验批处理脚本

如果主实验已经跑通，可以继续跑以下批量消融。命令中的模型路径可以替换为 HuggingFace 名称或本地模型目录。

### 容量实验：num_facts 扩展

```bat
scripts\run_llm_capacity_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
```

默认测试：

```text
num_facts = 60, 100, 160, 240
```

用于观察事实数量增加后，`acc_tail_fact` 和 `activation_rate` 的变化。

### 专家结构消融：num_experts 与 top_k

```bat
scripts\run_llm_expert_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
```

默认测试：

```text
(num_experts=4, top_k=1)
(num_experts=8, top_k=1)
(num_experts=8, top_k=2)
(num_experts=16, top_k=2)
(num_experts=16, top_k=4)
```

用于观察专家数量和每次激活专家数对准确率与开销的影响。

### Router 噪声鲁棒性

```bat
scripts\run_llm_router_noise_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
```

默认测试：

```text
router_label_noise = 0.0, 0.1, 0.2, 0.3
```

用于模拟真实场景中 router teacher 不完美的情况。

### 汇总结果

所有 LLM synthetic 实验跑完后，可以合并为一个 CSV：

```bat
scripts\collect_results.bat
```

输出：

```text
outputs\summary\llm_results_summary.csv
```

该 CSV 可直接用于画图或整理表格。

## 后续扩展方向

当前代码是最小可行验证，后续可以逐步替换为真实大模型实验：

1. 将 `BagOfWordsEncoder` 替换为 Qwen2.5-0.5B/1.5B 的 hidden state。
2. 将合成事实数据替换为 NQ、TriviaQA、HotpotQA 子集。
3. 将 memory expert 替换为 micro-LoRA 或 MLP Memory 形式。
4. 将 scheduler 的合成知识项替换为真实流式 QA 或个人文档访问轨迹。
5. 在 Windows 服务器上记录 GPU 显存、tokens/s、推理延迟和能耗代理指标。
