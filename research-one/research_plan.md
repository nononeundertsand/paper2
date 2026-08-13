# 面向边缘大模型的按需读写参数化记忆研究计划

## 1. 研究定位

本文计划围绕一个核心问题展开：

> 在边缘设备上，如何让小型大语言模型获得接近 RAG 的长期知识增强能力，同时避免传统 RAG 的高延迟、持续检索开销，以及静态参数记忆的常驻计算和难以更新问题？

现有方法已经分别证明了几件事：MLP Memory 证明检索器的知识分布可以被蒸馏进参数模块；Memory Layers at Scale 证明稀疏激活的记忆层可以高效扩展事实知识容量；Doc-to-LoRA 证明长文档可以被快速编译成 LoRA 参数；Titans 证明模型可以根据 surprise 信号在测试时选择性记忆重要信息；UniMem 证明情景记忆和参数记忆可以互补协作；MobileRAG 证明端侧 RAG 需要严肃考虑内存、能耗和延迟。

但是，这些工作尚未系统回答一个边缘智能场景中的关键问题：

> 边缘设备不应该一直检索，也不应该一直激活参数记忆，而应该学会什么时候读记忆、什么时候写记忆。

因此，本文拟提出一个统一框架：

**On-Demand Read and Resource-Aware Write for Edge Parametric Memory**

中文可命名为：

**面向边缘大模型的按需读写参数化记忆机制**

该框架包含两个相辅相成的 idea：

- **Idea 1：按需读取记忆**，解决推理阶段“什么时候激活参数记忆、激活多少”的问题。
- **Idea 2：资源感知写入记忆**，解决运行阶段“哪些知识值得从情景记忆固化为参数记忆”的问题。

二者共同组成一个端侧长期记忆闭环：Idea 2 负责产生高价值、低冗余、可管理的参数记忆；Idea 1 负责在推理时只在真正需要知识增强的位置读取这些记忆。

## 2. 核心参考论文与可借鉴点

| 论文 | 主要贡献 | 可借鉴内容 | 现有不足 | 本文切入点 |
|---|---|---|---|---|
| MLP Memory | 将 kNN 检索器的输出分布蒸馏为 MLP 参数记忆 | 参数化检索蒸馏、LLM 与 Memory 输出插值 | 静态、稠密、始终激活，端侧开销高，难更新 | 改为端侧按需激活、可动态写入的小型 memory experts |
| Memory Layers at Scale | 通过稀疏 key-value memory layer 增加事实知识容量 | 稀疏激活、记忆专家、事实任务收益 | 偏预训练架构，不关注端侧资源和动态知识 | 将稀疏思想用于已有小模型的外挂参数记忆 |
| Doc-to-LoRA | 用 hypernetwork 将文档一次前向转成 LoRA | 文档/知识到参数模块的快速固化 | 不考虑何时固化，也不考虑设备状态 | 将文档参数化纳入资源感知决策 |
| Titans | 测试时神经记忆，使用 surprise 信号选择性记忆 | 基于不确定性/惊讶度判断信息价值 | 更偏长上下文架构，测试时更新成本可能较高 | 用 surprise 作为读写记忆的轻量触发信号 |
| UniMem | 情景记忆到参数记忆的互补固化 | 新知识先进入 episodic memory，高频可靠知识再参数化 | 主要面向任务流，未建模端侧电量、温度、网络等资源 | 加入资源预算和端侧状态，形成可控固化策略 |
| MobileRAG | 端侧 RAG 的内存、延迟和能耗优化 | 端侧实验指标、RAG baseline、内存/磁盘分层思想 | 仍以检索为主，没有解决知识参数化 | 与本文方法形成直接对比：端侧 RAG vs 端侧参数记忆 |

## 3. 统一系统框架

系统由四个部分组成：

1. **Base LLM**：端侧可运行的小模型，例如 Qwen2.5-0.5B/1.5B、Llama-3.2-1B、Gemma-2B，默认冻结。
2. **Episodic Memory**：本地向量库，保存新进入的文档、交互、个人知识和临时事实，负责快速写入和可追溯删除。
3. **Parametric Memory Bank**：由多个 micro-MLP 或 micro-LoRA memory atoms 组成，每个模块存储一组稳定、高频、低风险知识。
4. **Memory Controller**：统一控制读写，包括推理时的 read router 和空闲阶段的 write scheduler。

整体流程如下：

1. 新知识先进入本地向量库，不立即参数化。
2. 系统统计知识被访问的频率、回答收益、稳定性、隐私等级和资源代价。
3. 当某些知识值得长期保留且参数化收益高时，在设备空闲或充电阶段固化为 memory atom。
4. 推理时，read router 逐 token 或逐 span 判断是否需要读取参数记忆。
5. 如果基础模型已经高置信，则跳过 memory bank；如果出现知识密集或高不确定位置，则激活 Top-K memory atoms。
6. 当知识过期、冲突或用户删除时，卸载对应 atom，并回退到 episodic memory 或重新固化。

## 4. Idea 1：按需读取的稀疏参数记忆

### 4.1 要解决的问题

原始 MLP Memory 在每个 token 上都运行完整记忆模块，但在实际生成中，大量 token 不需要额外知识。例如功能词、常见表达和简单推理可由基础模型完成。真正需要记忆介入的位置通常是实体、时间、地点、长尾事实、专业术语，或基础模型分布不确定的位置。

在边缘设备上，常驻运行参数记忆会带来三个问题：

- **计算冗余**：每个 token 都激活记忆，但只有少数 token 需要事实知识。
- **内存带宽压力**：端侧推理常受内存读取限制，额外 memory parameters 会增加功耗。
- **延迟不可控**：如果每步生成都访问记忆模块，TPOT 难以下降。

### 4.2 技术路线

设计一个轻量级 **Read Router**，在每个生成位置判断是否激活参数记忆。

输入特征包括：

- 当前 token 的 hidden state；
- 基础模型输出分布的 entropy；
- Top-1 与 Top-2 logit gap；
- 当前 span 是否包含实体或数字；
- 最近若干 token 的平均不确定性；
- episodic retrieval score，可选，用于判断是否存在相关外部知识。

Read Router 输出三个量：

- `p_read`：是否读取参数记忆；
- `top-k memory atoms`：需要激活哪些 memory modules；
- `alpha`：基础模型输出和记忆输出的动态插值权重。

推理时：

```text
if p_read < threshold:
    P_final = P_base
else:
    activate Top-K memory atoms
    P_mem = weighted_sum(memory_atom_outputs)
    P_final = alpha * P_base + (1 - alpha) * P_mem
```

参数记忆模块有两种可选实现：

- **micro-MLP memory**：延续 MLP Memory 的形式，输入 hidden state，输出 Top-K token logits 或低秩词表投影。
- **micro-LoRA memory**：将知识固化成可插拔 LoRA，只在路由命中时临时加载或合并。

为了控制开销，优先采用稀疏输出：

- 不预测完整词表分布；
- 只蒸馏 kNN 或 RAG teacher 的 Top-50/Top-100 token 分布；
- 最终 logits 通过 sparse scatter 或小型 candidate vocabulary 合并到基础模型输出。

### 4.3 训练目标

训练阶段冻结 Base LLM，仅训练 Read Router 和 memory atoms。

Teacher 可以来自两类：

- kNN-LM 或 MLP Memory 的检索分布；
- 标准 RAG 在给定证据时的输出分布。

总损失：

```text
L = L_distill + lambda_1 * L_sparsity + lambda_2 * L_load_balance + lambda_3 * L_task
```

其中：

- `L_distill`：记忆输出逼近 teacher 的 Top-K token 分布；
- `L_sparsity`：惩罚过高激活率，鼓励少读；
- `L_load_balance`：避免所有查询都落到同一个 memory atom；
- `L_task`：QA 或语言建模任务损失，保证最终输出有效。

一个关键设计是 **benefit-aware routing label**：

如果 teacher 明显优于 base model，则该位置应读记忆；如果 base model 已经预测正确或 teacher 帮助有限，则不读记忆。这样 router 学到的不是普通 MoE 路由，而是“何时需要知识增强”。

### 4.4 预期创新点

Idea 1 的创新不在于简单加入 MoE，而在于：

- 将 MLP Memory 的稠密常驻访问改为端侧按需访问；
- 将 token 级不确定性、事实性和资源开销结合为记忆读取策略；
- 用稀疏蒸馏降低 memory output 的词表计算；
- 将 memory activation rate 作为与准确率同等重要的核心指标。

## 5. Idea 2：资源感知的记忆写入与固化

### 5.1 要解决的问题

边缘设备会不断产生新知识，例如个人文档、聊天记录、位置相关信息、应用操作经验和近期任务。纯 RAG 可以快速写入，但检索库持续增长后会带来延迟、存储和隐私问题。纯参数化可以降低检索开销，但每条知识都固化会导致训练成本高、过期知识难删除、隐私风险高。

因此，端侧记忆写入需要回答：

- 哪些知识值得固化为参数？
- 什么时候固化？
- 固化成多大的模块？
- 固化后如果过期或删除，如何回滚？

### 5.2 技术路线

设计一个 **Write Scheduler**，将新知识从 episodic memory 逐步固化为 parametric memory atom。

每条知识或知识簇维护元数据：

- `freq`：访问频率；
- `recency`：最近访问时间；
- `stability`：知识是否稳定，是否经常冲突；
- `gain`：使用该知识后相对 base model 的准确率收益；
- `retrieval_cost`：每次检索该知识的延迟和能耗；
- `privacy_level`：隐私敏感等级；
- `delete_risk`：未来被删除或过期的概率；
- `device_state`：电量、温度、是否充电、是否空闲、网络状态。

定义固化效用：

```text
U(write) =
    a * expected_accuracy_gain
  + b * expected_retrieval_cost_saving
  + c * expected_offline_availability_gain
  - d * write_energy
  - e * storage_cost
  - f * privacy_risk
  - g * deletion_cost
```

当 `U(write)` 超过阈值，且设备处于合适状态时，触发固化：

- 低资源版本：对知识簇训练一个小型 micro-LoRA；
- 更快版本：参考 Doc-to-LoRA/SHINE，用 hypernetwork 直接生成 LoRA；
- MLP Memory 版本：对该知识簇蒸馏一个小型 micro-MLP。

固化后的 memory atom 存储以下信息：

- atom id；
- 来源文档或知识簇 id；
- 时间戳和版本；
- 适用主题 embedding；
- 参数文件路径；
- 隐私等级；
- 可删除标记。

### 5.3 可逆生命周期

本文不把参数记忆视为永久写入，而是设计可逆生命周期：

1. **Observe**：新知识进入 episodic memory。
2. **Accumulate**：统计访问频率、收益和稳定性。
3. **Consolidate**：在资源允许时固化为 memory atom。
4. **Read**：推理时由 Read Router 按需激活。
5. **Validate**：持续检测该 atom 是否造成过期或错误回答。
6. **Retire**：如果知识过期、冲突或用户删除，则卸载 atom。
7. **Fallback**：回退到 episodic memory 或重新固化新版本。

### 5.4 与 Idea 1 的配合

Idea 2 决定 memory bank 里放什么；Idea 1 决定生成时什么时候用。二者可以互相反馈：

- 如果某个 atom 长期不被 Idea 1 激活，说明它价值低，可降级或删除。
- 如果某类查询频繁触发本地 RAG，但还没有对应 atom，Idea 2 可以优先固化。
- 如果 Idea 1 激活某 atom 后准确率下降，Idea 2 将其标记为冲突或过期。

这样形成一个闭环优化目标：

```text
minimize: task_error + latency + energy + storage + privacy_risk
subject to: memory_budget, battery_budget, deletion_constraint
```

## 6. 实验设计

### 6.1 总体研究问题

实验围绕五个 Research Questions 设计：

- **RQ1：准确率收益**  
  按需读写参数记忆能否在知识密集任务上接近或超过 RAG、MLP Memory 和 LoRA？

- **RQ2：端侧效率**  
  相比始终激活的参数记忆和本地 RAG，是否显著降低 TTFT、TPOT、内存占用和能耗？

- **RQ3：路由行为**  
  Read Router 是否真的在实体、长尾事实和高不确定位置激活记忆，而在普通 token 上跳过？

- **RQ4：动态更新能力**  
  Write Scheduler 能否只固化高频、稳定、高收益知识，并减少过期知识错误？

- **RQ5：读写协同效果**  
  Idea 1 与 Idea 2 结合后，是否优于单独使用按需读取或单独使用记忆固化？

### 6.2 数据集设计

使用三类数据，控制实验成本：

1. **静态知识问答数据**
   - Natural Questions
   - TriviaQA
   - HotpotQA
   - 用于验证知识增强能力。

2. **语言建模与通用能力数据**
   - WikiText-103
   - C4 子集
   - MMLU 小规模子集，可选
   - 用于检测参数记忆是否破坏通用能力。

3. **流式端侧记忆数据**
   - 从 QA 数据中构造时间流：
     - 稳定知识：长期不变；
     - 高频知识：多次被问到；
     - 低频知识：偶尔出现；
     - 过期知识：答案在某个时间点变化；
     - 错误知识：模拟噪声或用户误输入；
     - 隐私知识：只能本地保存，不允许上传。
   - 可补充 LoCoMo 或合成个人助手数据，用于长程交互记忆。

### 6.3 模型设置

优先选择小模型，控制成本：

- Base LLM：
  - Qwen2.5-0.5B-Instruct；
  - Qwen2.5-1.5B-Instruct；
  - 可选 Gemma-2B 或 Llama-3.2-1B。

- Memory modules：
  - micro-MLP：总参数 20M、50M、100M；
  - micro-LoRA：rank 4/8/16；
  - memory atoms 数量：8、16、32。

- 训练策略：
  - 冻结 Base LLM；
  - 只训练 router 和 memory atoms；
  - 先做单 GPU 可运行版本；
  - 不复现 1B MLP Memory，避免过高开销。

### 6.4 Baseline 设计

需要覆盖模型、检索和参数记忆三类方法：

1. **Base LLM**  
   无任何外部记忆。

2. **Local RAG / MobileRAG-style RAG**  
   本地向量库检索 Top-3/Top-5 文档后拼接输入。

3. **Dense Parametric Memory**  
   始终激活一个同等参数量的 MLP Memory，用于验证按需读取的价值。

4. **Static LoRA Memory**  
   将所有训练知识统一固化到一个 LoRA，不做动态路由。

5. **Episodic-to-Parametric without Resource Awareness**  
   类似 UniMem 的固化思路，但不考虑电量、能耗、隐私和删除成本。

6. **Ours-Read Only**  
   只使用 Idea 1，记忆模块预先固定。

7. **Ours-Write Only**  
   只使用 Idea 2，但推理时始终激活或简单检索。

8. **Ours-Full**  
   按需读取 + 资源感知写入。

### 6.5 指标设计

任务质量：

- EM / F1；
- Perplexity；
- HaluEval 或自建幻觉率；
- 过期知识错误率；
- 冲突知识处理准确率。

系统效率：

- TTFT；
- TPOT；
- tokens/s；
- 峰值内存；
- 参数记忆激活率；
- 平均激活 atom 数；
- 本地 RAG 调用次数；
- 检索次数下降比例；
- 写入次数；
- 固化训练耗时。

端侧资源：

- energy per query；
- energy per write；
- 存储占用；
- 下载/上传流量，可选；
- 不同电量/温度/空闲状态下的写入策略表现。

可维护性：

- 删除延迟；
- 删除后残留命中率；
- atom 冲突率；
- atom 淘汰率；
- memory bank 增长速度。

### 6.6 关键消融实验

Idea 1 消融：

- 不使用 router，始终激活；
- 只用 entropy 触发；
- 只用实体/数字触发；
- 使用 entropy + logit gap + entity 的组合触发；
- Top-1 atom vs Top-2 atoms vs Top-4 atoms；
- 完整词表蒸馏 vs Top-K 稀疏蒸馏；
- 固定插值 alpha vs 动态 alpha。

Idea 2 消融：

- 所有知识都固化；
- 只按访问频率固化；
- 频率 + 准确率收益固化；
- 频率 + 收益 + 设备状态固化；
- 加入隐私风险和删除成本；
- 不允许卸载 vs 支持可逆卸载。

协同消融：

- 固定 memory bank + 按需读取；
- 动态 memory bank + 始终读取；
- 动态 memory bank + 按需读取；
- 不同 memory budget 下的效果；
- 不同 stream 变化速度下的效果。

### 6.7 图表设计

建议论文中重点展示以下图表：

1. **准确率-能耗 Pareto 曲线**  
   横轴 energy/query 或 FLOPs/token，纵轴 QA F1。目标是证明 Ours-Full 位于更优 Pareto 区域。

2. **激活热力图**  
   展示生成句子中每个 token 的 `p_read`。预期实体、数字、长尾事实高亮，功能词低亮。

3. **流式知识更新曲线**  
   横轴时间步，纵轴准确率和过期知识错误率。目标是证明资源感知固化比静态 LoRA 更稳。

4. **memory bank 增长曲线**  
   展示不同固化策略下 atom 数量和存储占用增长速度。

5. **读写协同柱状图**  
   比较 Base、RAG、Dense Memory、Read Only、Write Only、Ours-Full。

## 7. 可控实验实施路线

### 阶段一：验证 Idea 1

目标：证明按需读取可以在保留准确率收益的同时显著降低激活率和推理开销。

具体步骤：

1. 选择 Qwen2.5-0.5B 或 1.5B。
2. 用 NQ/TriviaQA 子集构造训练样本。
3. 用 Local RAG 或 kNN teacher 生成 Top-K token 分布。
4. 训练 20M/50M micro-MLP memory。
5. 训练 Read Router。
6. 对比 Base、Local RAG、Dense Memory、Ours-Read。

预期最小可发表结果：

- 只激活 20%-40% token；
- 达到 Dense Memory 80%-95% 的准确率收益；
- TPOT 或能耗显著低于 Dense Memory 和 RAG。

### 阶段二：验证 Idea 2

目标：证明资源感知写入可以减少无效固化和过期知识风险。

具体步骤：

1. 构造流式 QA 数据。
2. 所有新知识先进入 FAISS/HNSW 向量库。
3. 统计知识簇访问频率和收益。
4. 用 micro-LoRA 或 micro-MLP 固化高效用知识。
5. 模拟设备状态：高电量/低电量、充电/非充电、空闲/忙碌。
6. 比较固定周期固化、按频率固化、资源感知固化。

预期结果：

- 在相近准确率下，写入次数更少；
- 过期知识错误率更低；
- 存储增长更慢；
- 删除或回滚成本更低。

### 阶段三：端到端整合

目标：验证两个 idea 的协同收益。

具体步骤：

1. 用 Idea 2 动态维护 memory bank。
2. 用 Idea 1 在推理时按需读取 memory bank。
3. 在静态 QA 和流式 QA 上统一评测。
4. 增加系统测量：延迟、内存、能耗、激活率。

预期结论：

- Ours-Full 在准确率接近 RAG 的同时，明显减少检索调用。
- Ours-Full 相比 Dense Memory，显著降低 memory activation rate。
- Ours-Full 相比 Static LoRA，更能处理知识更新、过期和删除。

## 8. 论文贡献点预期

可以将贡献凝练为三点：

1. **提出面向边缘大模型的按需读写参数化记忆问题**  
   不再单独优化 RAG、LoRA 或 MLP Memory，而是研究端侧长期记忆中的读写决策。

2. **提出按需读取的稀疏参数记忆机制**  
   通过 token/span 级 router 和稀疏 memory atoms，在知识密集位置激活记忆，在普通位置跳过计算。

3. **提出资源感知且可逆的记忆固化机制**  
   将访问频率、准确率收益、能耗、存储、隐私和删除成本纳入固化决策，使参数记忆适应边缘设备动态资源状态。

## 9. 风险与备选方案

风险一：micro-MLP 训练效果不稳定。  
备选：优先做 micro-LoRA memory，因为训练和加载工具链更成熟。

风险二：kNN teacher 构建成本较高。  
备选：用 RAG teacher 输出或直接用 evidence-conditioned teacher logits，避免构建大规模 datastore。

风险三：真实手机能耗测量复杂。  
备选：先用 Jetson、Mac 或 GPU 上的功耗估计；同时报告 FLOPs、激活率、延迟和内存作为代理指标。

风险四：动态写入实验过于系统化。  
备选：先做 trace-driven simulation，设备状态用合成轨迹模拟，后续再上真实设备。

风险五：两个 idea 组合后系统复杂。  
备选：论文主实验先证明 Read Router，Write Scheduler 作为第二贡献用流式仿真验证，避免工程实现过重。

## 10. 推荐推进顺序

1. 先实现 Local RAG baseline 和 Base LLM 评测。
2. 实现 Dense micro-MLP / micro-LoRA memory。
3. 加入 Read Router，完成 Idea 1 的主实验。
4. 构造流式 QA 数据，加入 knowledge metadata。
5. 实现 Write Scheduler，完成 Idea 2 的策略实验。
6. 整合两个 idea，完成端到端实验。
7. 补充消融、图表和案例分析。

如果时间有限，优先保证 Idea 1 的实验完整，因为它与 MLP Memory 联系最直接，也最容易形成清晰结果；Idea 2 可以先以轻量仿真和 micro-LoRA 固化实验支撑文章的第二贡献。

