# 科研诚信与相关工作重合性核查

## 1. 核查结论

当前方案不能简单声称“首次提出按需参数记忆”或“首次提出 memory routing”。已有大量工作分别研究了：

- 参数化记忆；
- 稀疏/条件激活；
- RAG 是否需要检索；
- LoRA/adapter 作为外部参数记忆；
- 情景记忆到参数记忆固化；
- 面向端侧的 RAG 或 memory 系统优化。

但是，当前尚未发现与我们完全相同的问题设定：

> **在边缘/端侧场景下，将外部参数记忆作为可按需读取的 memory bank，显式比较 Base、Dense Memory 和 Conditional Memory 的负迁移，并进一步把资源感知写入作为后续闭环问题。**

因此，当前工作更适合定位为：

> **面向边缘大模型的按需读写参数化记忆框架，重点解决参数记忆始终激活导致的负迁移和端侧资源浪费问题。**

而不应表述为：

> “首次提出路由记忆”“首次提出参数记忆”“首次提出 memory experts”。

## 2. 与现有工作的相似点和差异

| 工作方向 | 代表工作 | 相似点 | 关键差异 |
|---|---|---|---|
| 检索蒸馏式参数记忆 | MLP Memory | 都将外部知识能力转化为参数模块 | MLP Memory 主要是 retriever-pretrained memory，通常稠密/静态使用；我们强调按需读取、避免 dense memory 干扰、端侧激活率 |
| 大规模稀疏记忆层 | Memory Layers at Scale | 都利用稀疏记忆容量补充事实知识 | Memory Layers 是模型架构/预训练层级的 memory；我们是 frozen LLM 上的外挂 memory bank 和读取控制 |
| 文档到参数 | Doc-to-LoRA, SHINE | 都把外部知识转成 LoRA/参数模块 | 这些工作关注如何生成 adapter；我们关注何时读取、何时写入、如何避免对已有能力的负迁移 |
| 外部参数记忆/LoRA routing | PMD, LoRA routing, MoRAM | 都涉及多个参数记忆模块和路由 | 这些多面向 LoRA 模块选择或 continual learning；我们当前实验重点是 Base/Dense/Conditional 对比和端侧激活率/负迁移 |
| 自适应 RAG | RAGRouter, Skill-RAG, ReaLM-Retrieve | 都研究什么时候检索或选择哪个 retriever | 它们路由的是检索器/文档/技能；我们路由的是参数记忆模块，不拼接外部文档 |
| 长上下文条件记忆访问 | L2A | 都有“何时访问长期记忆”的条件机制 | L2A 访问的是全局 attention/长上下文；我们访问的是外部参数化 memory experts |
| 测试时/长期神经记忆 | Titans, Atlas, Locas | 都认为不是所有信息都应被同等记忆或访问 | 它们重点是长上下文/测试时写入机制；我们当前重点是端侧按需读取与参数记忆负迁移 |
| 情景到参数固化 | UniMem, RecMem, MemVerse | 都有“新知识先外部保存，再选择性固化”的思想 | 它们更关注 agent memory 管理或任务流；我们显式引入端侧资源、activation rate 和可逆写入作为研究主线 |
| 端侧 RAG/移动 RAG | MobileRAG, PerCache | 都关注端侧资源、延迟、内存和能耗 | 它们优化检索/缓存链路；我们试图减少检索，把高价值知识转成可按需读取的参数记忆 |

## 3. 需要避免的过度声明

以下说法不建议使用：

1. “首次提出按需记忆访问。”
2. “首次提出 memory experts。”
3. “首次提出参数记忆路由。”
4. “首次提出情景到参数固化。”
5. “完全替代 RAG。”

这些方向已有相邻工作，直接这样写容易造成 novelty 争议。

## 4. 可以相对稳妥声称的贡献

更稳妥的贡献表述应围绕我们实际解决的问题：

### 贡献 1：指出并验证 dense parametric memory 的负迁移问题

我们通过 Base / Dense Memory / Conditional Memory 三组对比，明确展示：

- Base 缺 tail knowledge；
- Dense Memory 能补 tail，但严重破坏 common/general；
- Conditional Memory 能同时补 tail 并保留 common/general。

这比单纯说明 memory 有用更具体，强调的是 **参数记忆何时不该被使用**。

### 贡献 2：提出面向端侧的按需参数记忆读取框架

我们的 router 根据 hidden state、base uncertainty 等信号决定是否激活 memory experts，并以 activation rate 作为核心指标之一。这个定位应写为：

> Resource-aware / edge-oriented conditional access to external parametric memory.

而不是简单 MoE。

### 贡献 3：将读取决策与写入决策放到统一问题中

Idea 1 解决“什么时候读”；Idea 2 解决“什么知识值得写”。这一读写闭环是本文可以强调的系统性贡献。

### 贡献 4：在真实数据上验证负迁移和按需读取收益

当前实验已经从 synthetic facts 扩展到：

- SQuAD 真实 QA；
- AG News 真实 general；
- 冻结 Qwen2.5-0.5B hidden state；
- 候选答案排序 EM/F1、MRR、Hits@K。

这能支撑方法有效性，但仍需进一步多 seed 和更难数据集。

## 5. 当前实验对科研诚信的边界说明

当前实验已经比 toy synthetic 更强，但仍需明确边界：

1. **不是自由生成式 QA**  
   当前是候选答案排序和分类式验证，不是模型自由生成答案后的标准 EM/F1。

2. **不是完整端侧部署实验**  
   当前 activation rate 是端侧计算代理指标，尚未实测手机/Jetson 上的延迟、能耗和内存。

3. **Write Scheduler 仍是仿真**  
   资源感知写入尚未和真实训练闭环完全打通。

4. **Memory 形式仍较简单**  
   当前 memory experts 是 MLP；未来可比较 micro-LoRA、MLP Memory、memory atoms。

5. **还需要更多数据集和随机种子**  
   当前 SQuAD + AG News 结果很好，但需要 NQ、TriviaQA、HotpotQA、多 seed 支撑泛化性。

## 6. 建议在论文 Related Work 中必须覆盖的类别

为了科研诚信，Related Work 至少应覆盖：

1. **Retrieval-augmented generation and adaptive retrieval**
   - RAG
   - RAGRouter
   - Skill-RAG
   - ReaLM-Retrieve

2. **Parametric memory and retriever distillation**
   - MLP Memory
   - Memory Layers at Scale
   - LoRA as Knowledge Memory

3. **External parametric memory and adapter routing**
   - Doc-to-LoRA / SHINE
   - PMD / LoRA routing
   - MoRAM

4. **Test-time and long-term neural memory**
   - Titans
   - Atlas
   - Locas
   - TransMem

5. **Episodic-to-parametric consolidation**
   - UniMem
   - MemVerse
   - RecMem

6. **On-device / edge memory systems**
   - MobileRAG
   - PerCache
   - DuoMem

## 7. 建议的论文定位

建议把论文定位为：

> Existing parametric memory methods show that knowledge can be stored in external parameters, but dense or unconditional memory activation can cause negative transfer and unnecessary edge-side overhead. We study conditional access to external parametric memory for edge LLMs, showing that a lightweight read router can preserve base-model capabilities while selectively activating memory for tail knowledge. We further frame this as a read-write memory lifecycle problem, where resource-aware consolidation decides which memories should become parametric.

中文表述：

> 现有参数化记忆方法证明知识可以被写入外部参数模块，但密集或无条件激活会带来负迁移和端侧资源浪费。我们研究面向边缘大模型的外部参数记忆按需读取问题，证明轻量级路由器能够在保留基础模型已有能力的同时，仅对长尾知识激活参数记忆；进一步地，我们将其扩展为读写一体的记忆生命周期问题，由资源感知固化机制决定哪些知识值得参数化。

## 8. 最终判断

当前方法与现有工作有明显相邻关系，但暂未发现完全相同方案。为了科研诚信，论文中必须充分承认：

- 条件访问不是新概念；
- 参数记忆不是新概念；
- adapter/LoRA routing 不是新概念；
- 情景到参数固化不是新概念。

我们的创新应落在更具体的问题组合上：

> **端侧资源约束下，外部参数记忆的按需读取与资源感知写入，并通过真实 QA + general 数据验证 dense memory 负迁移和 conditional memory 的优势。**

