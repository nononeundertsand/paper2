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
├── scripts/
│   ├── run_scheduler.bat
│   ├── run_scheduler.sh
│   ├── run_synthetic.bat
│   └── run_synthetic.sh
└── src/
    └── edge_memory/
        ├── __init__.py
        ├── data.py
        ├── model.py
        ├── scheduler.py
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

## 后续扩展方向

当前代码是最小可行验证，后续可以逐步替换为真实大模型实验：

1. 将 `BagOfWordsEncoder` 替换为 Qwen2.5-0.5B/1.5B 的 hidden state。
2. 将合成事实数据替换为 NQ、TriviaQA、HotpotQA 子集。
3. 将 memory expert 替换为 micro-LoRA 或 MLP Memory 形式。
4. 将 scheduler 的合成知识项替换为真实流式 QA 或个人文档访问轨迹。
5. 在 Windows 服务器上记录 GPU 显存、tokens/s、推理延迟和能耗代理指标。

