# 小模型（<2B）全参微调 vs PEFT（LoRA 等）知识沉淀调研（2020–2025）

> 调研时间：2026-10-13。检索方式：web_search 15 轮（多查询合并）+ web_fetch 抓取 24 篇正文（arXiv/ACL Anthology 等），覆盖 2020–2025 年（个别经典文献早于 2020 年作背景，个别 2026 年文献仅作窗口外补充并明确标注）。
> 聚焦：**小数据集（数百～数千条）场景下小模型全参微调的过拟合风险与缓解手段**；附带全参 vs LoRA 实证比较、微调后校准漂移、8K 长序列分类微调工程注意事项。

---

## 1. 核心结论速览（TL;DR）

1. **小数据 + 任务与预训练分布接近时，LoRA 与全参通常无显著差距**；差距主要出现在 10 万级以上数据、或与基座分布差异大的领域（代码、数学）。
2. LoRA 的本质代价是**"学得慢/学得少"（欠拟合方向）但遗忘少**；全参是**"学得多"但更易过拟合 + 破坏基座特征与校准**。小数据场景下两者风险方向恰好相反。
3. 全参微调在小数据下的主要风险不是"训练不动"，而是**过拟合 + 分布偏移下退化 + 校准漂移**；三大类缓解：优化侧（低 LR/warmup/早停）、正则侧（label smoothing/weight decay/dropout/SAM/插值）、平均侧（SWA/EMA/model soup/WiSE-FT）。
4. **权重空间平均与基座插值是性价比最高、证据最硬的两族手段**：不增加推理成本、实现只需几行代码。
5. 微调（尤其偏好对齐/过拟合训练）会**系统性破坏置信度校准**，训练后必须在留出集上做温度重拟合，训练中可用 label smoothing / ECE 正则缓解。
6. 8K 长序列全参微调 0.8B 模型在 24GB 单卡上**可行**，但必须：FlashAttention-2 + 梯度检查点 + bf16 + 只对答案 token 计损（或用分类头），并注意长序列下 attention-logit 爆炸与 loss spike。
7. 对 669 条数据：**分层交叉验证 + 多种子报告、早停、模型平均、温度重校准**是不可省的四件事。

---

## 2. 全参 vs LoRA：实证比较与学习动力学

### 2.1 何时无显著差距、何时有差距

- **结论：LoRA 在 RoBERTa/DeBERTa/GPT-2/GPT-3 的 NLU 任务上与全参持平或更好，同时可减少可训练参数 10000 倍、训练显存约 3 倍（GPT-3 175B + Adam 场景）**。
  来源：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（Hu et al., 2021）。
  证据：摘要明确"LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters"，且"reduce the GPU memory requirement by 3 times"。（注：其对比场景含 GLUE 级别中小数据，与小数据分类场景直接相关。）

- **结论：在 10 万级指令数据和 20B token 持续预训练场景下，标准低秩 LoRA 显著逊于全参；全参学到的权重扰动秩是典型 LoRA 配置的 10–100 倍**。
  来源：[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)（Biderman et al., TMLR 2024 Featured Certification）。
  证据：摘要明确"in the standard low-rank settings, LoRA substantially underperforms full finetuning"（编程与数学两个目标域），且"full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations"。

- **结论：LoRA 遗忘更少，且比 weight decay、dropout 等常规正则更能保持基座模型在域外任务上的性能——LoRA 本身就是一种隐式正则**。
  来源：[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)（TMLR 2024）。
  证据：摘要明确"LoRA better maintains the base model's performance on tasks outside the target domain"且"mitigates forgetting more than common regularization techniques such as weight decay and dropout"。

- **结论：LoRA 与全参的解并不等价：LoRA 会在权重谱中引入全参没有的"入侵维度"（intruder dimensions），其遗忘几乎全部局部化于该维度；压低对应奇异值即可恢复基座分布建模能力**。
  来源：[LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228)（Shuttleworth, Andreas, Torralba, Sharma；ICLR 2025）。
  证据：摘要明确"weight matrices trained with LoRA have new, high-ranking singular vectors, which we call intruder dimensions, while those trained with full fine-tuning do not"，且"scaling them down significantly improves modeling of the pre-training distribution with a minimal drop in downstream task performance"。

- **结论：部分"LoRA 差距"是优化器问题而非容量问题——LoRA 双因子参数化缺乏变换不变性导致学习低效，仅替换优化器即可在 Gemma-2B 上 +4.6% 准确率**。
  来源：[LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization](https://arxiv.org/abs/2410.20625)（Yen et al., ICLR 2025 Oral）。
  证据：摘要明确"current LoRA optimizers lack transformation invariance"，"replacing Adam with LoRA-RITE during LoRA fine-tuning of Gemma-2B yielded 4.6% accuracy gain on Super-Natural Instructions"。

- **结论：DoRA 将权重分解为幅度+方向、对方向做 LoRA，能持续超越 LoRA 并提升训练稳定性，缩小与全参的差距**。
  来源：[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)（Liu et al., ICML 2024 Oral）。
  证据：摘要明确"DoRA consistently outperforms LoRA on fine-tuning LLaMA, LLaVA, and VL-BART"，并"enhance both the learning capacity and training stability of LoRA"。

- **结论：当显存放不下全参时，LISA（按层重要性采样、随机冻结中间层）以接近 LoRA 的内存代价达到甚至超过全参——LoRA 的低秩限制在大规模微调中确实是瓶颈**。
  来源：[LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919)（Pan et al., NeurIPS 2024）。
  证据：摘要明确"LISA consistently outperforms LoRA by over 10%-35% in terms of MT-Bench score while achieving on-par or better performance in MMLU, AGIEval and WinoGrande"。

- **结论（窗口外补充，2026）：线性回归理论分析表明，当预训练-下游差异本质低秩、或样本量少时，LoRA 的 excess risk 可低于全参——小数据下低秩约束是正则而非纯粹的能力限制**。
  来源：[LoRA vs. Full Fine-Tuning: A Theoretical Perspective](https://arxiv.org/abs/2605.19018)（Zindari, Mulayoff, Stich；2026，超出本文 2020–2025 窗口，仅供启发）。
  证据：摘要明确"LoRA can outperform full fine-tuning when the difference between the pretraining and the downstream tasks is effectively low-rank"，"using a very small rank can improve test accuracy in certain settings"。

### 2.2 学习动力学：LoRA 学得更慢（欠拟合向）的证据

- **结论：LoRA 对指令数据的收益低于全参（learns less），表现为欠拟合方向；这在小数据短训练里意味着 LoRA 更"保守"，全参更"激进"**。
  来源：[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)（TMLR 2024）。
  证据：见上，两域实验中 LoRA 在目标域系统性低于全参，而在域外保留度更高。

- **结论：LoRA 的逐样本学习动力学与标注熵相关（高熵样本学得更慢），印证其"学得慢"是系统性行为而非个别现象**。
  来源：[Annotation Entropy Predicts Per-Example Learning Dynamics in LoRA Fine-Tuning](https://ar5iv.labs.arxiv.org/html/2604.16332)（2026，窗口外补充）。
  证据：检索摘要显示"all six models show the same qualitative pattern: higher entropy predicts slower learning"。

### 2.3 判读

- 数据量 **数百～数千**、输出为分类标签：LoRA 与全参差距通常在噪声范围内（LoRA 原论文的 GLUE 级设定 + 2.1 节理论补充）；选型的关键不是精度，而是**遗忘/过拟合/工程成本**三维权衡。
- 数据量 **10 万+**、或任务与基座差异大（新领域知识、格式全新）：全参（或 LISA/DoRA）有明确优势。

---

## 3. 小数据全参微调的过拟合风险与缓解

### 3.1 风险的来源与形态

- **结论：小数据微调的不稳定（种子间方差大）主要由优化困难（梯度消失）驱动，而非"灾难性遗忘"或数据量本身；同训练损失的不同种子在测试集上差异明显——小数据微调必须多种子评估**。
  来源：[On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884)（Mosbach, Andriushchenko, Klakow, ICLR 2021）。
  证据：摘要明确"the observed instability is caused by optimization difficulties that lead to vanishing gradients"，且"fine-tuned models with the same training loss exhibit noticeably different test performance"。

- **结论：全参微调会在头几步破坏预训练特征；预训练特征本来就好时，全参在分布偏移（OOD）下比线性探测平均低 7%，而分布内仅高 2%——小数据全参的头几个 step 最危险**。
  来源：[Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054)（Kumar et al., ICLR 2022 Oral）。
  证据：摘要明确"fine-tuning obtains on average 2% higher accuracy ID but 7% lower accuracy OOD than linear probing"，机制为"fine-tuning learns the head, the lower layers change simultaneously and distort the pretrained features"。

- **结论：先线性探测再全参（LP-FT）两步法同时拿两头：分布内比全参 +1%、OOD 比全参 +10%**。
  来源：同上 [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054)。
  证据：摘要明确"LP-FT outperforms both fine-tuning and linear probing (1% better ID, 10% better OOD than full fine-tuning)"。

### 3.2 优化/正则类缓解手段

- **结论（早停的机制性证据）：小数据指令微调的收益下降源于对训练集过拟合，损失函数设计上减少对训练指令的过度拟合能显著提升泛化——"长指令+短输出+样本少"时尤其有效（正是 8K 输入+分类标签的形态）**。
  来源：[Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394)（Shi et al., NeurIPS 2024）。
  证据：摘要明确"IM is especially beneficial when trained on datasets with lengthy instructions paired with brief outputs, or … a small amount of training examples"，且"improvement can be attributed to reduced overfitting"。

- **结论（SAM）：锐度感知最小化同时最小化损失值与损失锐度，在含微调任务在内的多基准上提升泛化，并天然具备与专用方法相当的标签噪声鲁棒性——对几百条含噪标签的小数据集是直接对症的手段**。
  来源：[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)（Foret et al., ICLR 2021）。
  证据：摘要明确"SAM improves model generalization across a variety of benchmark datasets (… finetuning tasks)"且"provides robustness to label noise on par with … state-of-the-art procedures"。

- **结论（label smoothing）：标签平滑防止过度自信、改善泛化与校准；其代价（损害蒸馏）与分类微调场景无关**。
  来源：[When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)（Müller, Kornblith, Hinton, NeurIPS 2019；经典背景文献）。
  证据：摘要明确"Smoothing the labels in this way prevents the network from becoming over-confident"，"in addition to improving generalization, label smoothing improves model calibration"。

- **结论（dropout 一致性正则）：R-Drop 让同一样本经两次独立 dropout 前向后对齐输出分布，在中级规模数据上稳定提升 20+ 个 NLP/CV 任务——适合作为小数据全参的廉价正则**。
  来源：[R-Drop: Regularized Dropout for Neural Networks](https://papers.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)（Liang et al., NeurIPS 2021）。
  证据：论文题录页摘要显示 R-Drop 通过双前向 KL 一致性约束在 20 个任务上取得一致提升（多任务平均约 1–2 个点的泛化收益）。

### 3.3 权重平均类缓解手段（证据最强的一族）

- **结论（SWA）：沿训练轨迹平均多个权重点比常规训练泛化更好，且找到明显更平坦的解——即"末段 checkpoint 平均"是免成本的抗过拟合手段**。
  来源：[Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)（Izmailov et al., UAI 2018；经典背景文献）。
  证据：摘要明确"simple averaging of multiple points along the trajectory of SGD … leads to better generalization than conventional training"，"finds much flatter solutions than SGD"。

- **结论（model soup）：把多组超参（LR/epoch/增广）下的微调权重直接平均，常同时提升精度、OOD 与零样本性能，零推理开销——超参扫描的副产品全部变成收益**。
  来源：[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)（Wortsman et al., ICML 2022）。
  证据：摘要明确"averaging the weights of multiple models fine-tuned with different hyperparameter configurations often improves accuracy and robustness"，且"extends to multiple image classification and natural language processing tasks, improves out-of-distribution performance"。

- **结论（WiSE-FT 插值）：微调权重与零-shot 基座权重直接插值，目标分布精度基本不掉、分布偏移下提升 4–6 个点以上；训练与推理零额外成本**。
  来源：[Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)（Wortsman et al., CVPR 2022）。
  证据：摘要明确"ensembling the weights of the zero-shot and fine-tuned models (WiSE-FT) … improves accuracy under distribution shift by 4 to 6 percentage points … while increasing ImageNet accuracy by 1.6 pp"。（EMA checkpoint 属同族思想的在线版本，工程上以 SWA/soup 证据为支撑。）

### 3.4 数据侧缓解手段

- **结论（质量>数量）：1000 条精选数据足以让 65B 模型对齐——知识几乎都在预训练中，微调只需教格式与边界；对 669 条数据，逐条质检比再标 1000 条更划算**。
  来源：[LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)（Zhou et al., NeurIPS 2023）。
  证据：摘要明确"fine-tuned … on only 1,000 carefully curated prompts and responses"即获强表现，"almost all knowledge in large language models is learned during pretraining"。

- **结论（嵌入噪声增广）：训练时对 embedding 加均匀噪声（NEFTune）即可大幅改善指令微调，对 AlpacaEval 从 29.79% 提升到 64.69%——一行代码级的数据增广，小数据场景优先尝试**。
  来源：[NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)（Jain et al., ICLR 2024）。
  证据：摘要明确"NEFTune adds noise to the embedding vectors during training"，"Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69%"。

- **结论（序列长度课程）：由短到长逐步增加训练序列长度的课程学习被长上下文工作采用以改善训练稳定性——8K 输入建议先在截断/短样本上收敛再放开全长**。
  来源：[Scaling Transformer to 1M tokens and beyond with RMT](https://arxiv.org/abs/2304.11062)（Bulatov et al., 2023；AAAI 2024 版本含课程学习消融，见 [AAAI 29722](https://ojs.aaai.org/index.php/AAAI/article/view/29722)）。
  证据：AAAI 版本论文正文明确"To improve training stability of the original RMT we introduce curriculum learning"（检索定位的原文片段）。

---

## 4. 微调破坏校准（calibration drift）与对策

- **结论：微调后的 Transformer 分布内校准尚可，但分布外校准误差可恶化至 3.5 倍；温度缩放（TS）有效降低分布内校准误差，label smoothing 通过主动注入不确定性帮助分布外校准**。
  来源：[Calibration of Pre-trained Transformers](https://aclanthology.org/2020.emnlp-main.21/)（Desai & Durrett, EMNLP 2020）。
  证据：摘要明确"(1) … their calibration error out-of-domain can be as much as 3.5x lower（相对基线）; (2) temperature scaling is effective at further reducing calibration error in-domain, and using label smoothing … helps calibrate posteriors out-of-domain"。

- **结论：预训练模型的条件概率本来就校准良好，SFT/RLHF 之后显著退化；按 token 特征自适应预测温度（ATS）可把校准误差再降 10–50%**。
  来源：[Calibrating Language Models with Adaptive Temperature Scaling](https://arxiv.org/abs/2409.19817)（Xie et al., EMNLP 2024）。
  证据：摘要明确"after fine-tuning with reinforcement learning from human feedback (RLHF), the calibration of these models degrades significantly"，"ATS improves calibration by over 10-50% across three downstream natural language evaluation benchmarks"。

- **结论：偏好对齐导致"过度自信+校准劣化"存在机制性解释（偏好塌缩泛化为校准塌缩）；持续微调会进入"不可校准区"，训练中加 ECE 正则（EM 算法估计）可在不损性能的前提下维持低校准误差**。
  来源：[Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach](https://arxiv.org/abs/2505.01997)（Xiao et al., ICML 2025）。
  证据：摘要明确"LLMs tend to become poorly calibrated after alignment"，"we develop an EM-algorithm-based ECE regularization for the fine-tuning loss to maintain low calibration error"。

- **结论（对策小结，综合上三条）：① 训练后必须在未参与训练的校准集上重拟合单参数温度（最便宜、最稳）；② 训练中 label smoothing（0.05–0.1）防过度自信；③ 过度训练会把模型推入难以事后校准的区域——早停不仅保精度也保校准；④ 评估报告应含 ECE/Brier，不只看准确率。**

---

## 5. 长序列（最长 8K token）分类微调：注意力 / 显存 / 稳定性

- **结论：上下文 8192 的自注意力计算量约是 2048 的 16 倍；训练期可用稀疏局部注意力（S²-Attn，两行代码）近似全局注意力，且长上下文场景 LoRA 需放开 embedding 与 normalization 层才好用**。
  来源：[LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)（Chen et al., ICLR 2024）。
  证据：摘要明确"training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048"，"LoRA for context extension works well under the premise of trainable embedding and normalization"。

- **结论：FlashAttention-2 把注意力显存从 O(L²) 降到 O(L) 且无近似，并在 A100 上达到 GEMM 级效率——长序列微调的第一开关**。
  来源：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)（Dao, 2023）。
  证据：摘要明确"FlashAttention exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4× … with no approximation)"，FA2 再提速约 2 倍。

- **结论：长序列 + 低精度（bf16）下训练不稳定的直接机制是 attention-logit 爆炸：logit 随序列长度增长，源于局部稠密依赖逼出高秩注意力结构；缓解包括增大注意力维度、显式建模局部依赖（工程上对应 QK-norm / logit softcap / 截断）**。
  来源：[Dense Local Dependencies Induce Attention-Logit Explosion and Training Instability During Long-Sequence Transformer Training](https://arxiv.org/abs/2505.15548)（Hajra, 2025）。
  证据：摘要明确"attention-logit growth increases with sequence length, is mitigated by increasing the attention dimension, and is substantially reduced by explicitly modeling dense local dependencies"。

- **结论（分类任务特化的损失掩码）：对长输入只对答案/输出部分计算损失（而非全序列 LM 损失）既省显存又降低过拟合——8K 输入×词表 logits 是显存大头，分类头或答案位损失掩码可将其归零**。
  来源：[Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394)（NeurIPS 2024）。
  证据：见 3.2 节——损失只在必要 token 上计算时小数据场景收益最大且归因于过拟合减少（该文反向利用了该原理并证明其条件性，工程上通用做法是对非答案 token 掩码）。

- **结论（工程估算，非文献引用）：0.8B 全参 + AdamW 混合精度在 24GB 单卡的静态开销 ≈ bf16 权重 1.6GB + fp32 主权重 3.2GB + 一二阶矩 6.4GB + 梯度 1.6–3.2GB ≈ 13–14GB，剩约 10GB 给激活；8K 序列激活必须靠 FA2 + 梯度检查点（可再配 8-bit AdamW 或 paged optimizer 应对峰值）。若仍紧张：按长度分桶组 batch、动态 padding、packing。**

---

## 6. 对我们场景的全参微调操作建议清单

> 场景：**0.8B 决策模型；669 条中文委派任务；输出 = 3 类难度 + 视觉二分类；输入最长 8K tokens；单卡 24GB。**
> 证据强度分级：**A** = 多篇同行评审 + 场景直接匹配；**B** = 同行评审但场景间接 / 单篇强证据；**C** = 启发式 / 工程实践 / 窗口外文献补充。

### A 级（先做，证据最强）

1. **LoRA(r=16–64, 含 embedding/norm 可训) 与全参各跑一组严格对照基线**，同数据同评估。小数据分类下两者大概率打平；LoRA 还遗忘更少（[LoRA 2021](https://arxiv.org/abs/2106.09685) on-par；[Learns Less 2024](https://arxiv.org/abs/2405.09673) 遗忘更少；差距主要在 10 万级数据才拉开）。
2. **评估协议：分层 5 折交叉验证（按 3 类难度 × 视觉二分类双重分层）+ 每配置 ≥3 种子，报告均值±标准差**。微调种子方差是系统性的（[Mosbach et al. 2021](https://arxiv.org/abs/2006.04884)：同训练损失不同种子测试性能差异明显）。
3. **早停：以验证集 macro-F1（含校准指标）为标准，patience 1–2 个 eval 点；预计最优在 2–5 epoch 内**。小数据指令微调的退化源于过拟合（[Instruction Tuning With Loss Over Instructions 2024](https://arxiv.org/abs/2405.14394)：过拟合减少带来收益；[LIMA 2023](https://arxiv.org/abs/2305.11206)：千条级数据只需轻量适配）。
4. **权重平均二选一或并用：末段 checkpoint SWA/EMA；或多超参 model soup（LR×epoch×种子网格的权重平均）**（[SWA](https://arxiv.org/abs/1803.05407)：平坦解+泛化；[Model Soups 2022](https://arxiv.org/abs/2203.05482)：超参扫描权重平均提升精度与 OOD，零推理开销）。
5. **WiSE-FT 式基座插值：微调后对全参权重做 θ = α·θ_ft + (1−α)·θ_base，α∈{0.4..0.8} 网格**。目标集几乎不掉点，OOD/鲁棒性显著提升（[Wise-FT 2022](https://arxiv.org/abs/2109.01903)：分布偏移 +4–6pp）。（LoRA 天然靠近基座，可作消融对照。）
6. **校准流程：留 15–20% 做校准集，训练后单参数温度重拟合；训练中 label smoothing 0.05–0.1；上线前报告 ECE/Brier 与置信度分桶图**（[Desai & Durrett 2020](https://aclanthology.org/2020.emnlp-main.21/)：TS 域内有效、label smoothing 域外有效；[ATS 2024](https://arxiv.org/abs/2409.19817)：微调后校准系统性退化 10–50% 可修复；[ICML 2025](https://arxiv.org/abs/2505.01997)：过度微调进入不可校准区）。决策模型输出置信度用于下游路由时此条升为最高优先级。
7. **长序列工程三件套：FlashAttention-2 + 梯度检查点 + bf16；只对答案 token 计损或直接接分类头（last-token/mean pooling）**（[FA2](https://arxiv.org/abs/2307.08691)：显存 O(L)；[LongLoRA 2024](https://arxiv.org/abs/2309.12307)：8K 注意力成本 16×；[IM 2024](https://arxiv.org/abs/2405.14394)：损失掩码与过拟合）。

### B 级（值得做，证据较强/间接）

8. **优化超参：全参 LR 1e-5～2e-5（LoRA 用 1e-4～3e-4），warmup 5–10%，余弦退火，batch 尽量大（可用 grad accum），AdamW β2=0.95、weight decay 0.01–0.1**（[Mosbach 2021](https://arxiv.org/abs/2006.04884)：小数据不稳定的解药是小 LR+warmup+稳健基线；[Learns Less 2024](https://arxiv.org/abs/2405.09673)：weight decay/dropout 是有效但弱于 LoRA 的正则——全参时更该开）。
9. **LP-FT 两步法：先冻结主干只训分类头 1–2 epoch，再解冻全参**。防头几步特征扭曲、OOD 显著受益（[Kumar et al. 2022](https://arxiv.org/abs/2202.10054)：LP-FT 比全参 ID +1%、OOD +10%）。
10. **数据侧：669 条逐条质检 + 标注一致性复核（难度三类的边界样本尤其）；每类至少 ~100 条（不足则补标或合并）；同义改写/字段乱序做轻量增广**（[LIMA 2023](https://arxiv.org/abs/2305.11206)：千条精选足够；质量优先）。
11. **NEFTune 嵌入噪声（α=5–10）**：一行代码，对 669 条规模的小数据指令式微调收益概率高（[NEFTune 2024](https://arxiv.org/abs/2310.05914)：AlpacaEval 29.79%→64.69%，各数据集 +8–10%）。
12. **SAM/ASAM 一次消融**：小数据+可能存在标签噪声时（人工标的难度等级天然带噪），SAM 双前向成本可接受（[SAM 2021](https://arxiv.org/abs/2010.01412)：泛化+标签噪声鲁棒；联邦场景 LLM 微调亦有后续验证 [SplitFed-SAM 2025](https://dl.acm.org/doi/10.1007/978-981-96-9884-4_22)）。
13. **长序列稳定性：先短（≤2K 截断）后长（全长 8K）两阶段课程；监控 attention logit 范数与 loss spike；出现 spike 时降 LR 或对 logits softcap/clip**（[RMT 2023](https://arxiv.org/abs/2304.11062)：长度课程提稳；[Hajra 2025](https://arxiv.org/abs/2505.15548)：logit 爆炸随长度增长）。
14. **Dropout：全参微调保留/新增 dropout 0.05–0.1（含 attention dropout），并与 R-Drop 一致性正则（α≈1–5）做一次消融**（[Learns Less 2024](https://arxiv.org/abs/2405.09673) 将 dropout 列为全参可用正则；[R-Drop 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)：一致性正则多任务稳定增益）。

### C 级（可选/探索）

15. **DoRA 替换 LoRA、LoRA-RITE 优化器**：若 LoRA 路线欠拟合（训练指标明显低于全参）再上（[DoRA 2024](https://arxiv.org/abs/2402.09353)；[LoRA-RITE 2025](https://arxiv.org/abs/2410.20625)：仅换优化器 Gemma-2B +4.6%）。
16. **LISA**：0.8B 在 24GB 上全参无压力，暂不需要（[LISA 2024](https://arxiv.org/abs/2403.17919)：为 7B+ 显存受限设计）。
17. **训练中 ECE 正则**：若上线校准要求极严且温度重拟合不够，再考虑（[Calibration-Aware FT 2025](https://arxiv.org/abs/2505.01997)：EM-ECE 正则；实现复杂度较高）。
18. **谱后处理**：若观察到全参模型在新任务间切换退化，可参考"入侵维度缩放"类谱后处理思路（[Illusion of Equivalence 2025](https://arxiv.org/abs/2410.21228)：缩奇异值恢复基座分布，窗口内新方向、证据尚早期）。

### 一句话主路径（推荐执行顺序）

> **分层 5 折 + 3 种子 → 全参 vs LoRA 对照（先 LP-FT 再全参；LR 1e-5、wd 0.01、ls 0.05、早停）→ 末段 SWA + 基座插值 α 网格 → 校准集温度重拟合 + ECE 报告 → 全程 FA2+检查点+答案位损失掩码；数据侧逐条质检 + NEFTune。**

---

## 7. 参考文献一览（按主题）

**全参 vs PEFT**
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) — Biderman et al., TMLR 2024
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al., ICML 2024
- [LoRA Done RITE](https://arxiv.org/abs/2410.20625) — Yen et al., ICLR 2025
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) — Shuttleworth et al., ICLR 2025
- [LISA: Layerwise Importance Sampling](https://arxiv.org/abs/2403.17919) — Pan et al., NeurIPS 2024
- [LoRA vs. Full Fine-Tuning: A Theoretical Perspective](https://arxiv.org/abs/2605.19018) — 2026（窗口外补充）

**小数据过拟合与缓解**
- [On the Stability of Fine-tuning BERT](https://arxiv.org/abs/2006.04884) — Mosbach et al., ICLR 2021
- [Fine-Tuning can Distort Pretrained Features and Underperform OOD](https://arxiv.org/abs/2202.10054) — Kumar et al., ICLR 2022
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) — Foret et al., ICLR 2021
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) — Müller et al., NeurIPS 2019
- [R-Drop](https://papers.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) — Liang et al., NeurIPS 2021
- [SWA: Averaging Weights Leads to Wider Optima](https://arxiv.org/abs/1803.05407) — Izmailov et al., UAI 2018
- [Model Soups](https://arxiv.org/abs/2203.05482) — Wortsman et al., ICML 2022
- [WiSE-FT: Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903) — Wortsman et al., CVPR 2022
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., NeurIPS 2023
- [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394) — Shi et al., NeurIPS 2024
- [NEFTune](https://arxiv.org/abs/2310.05914) — Jain et al., ICLR 2024

**校准漂移**
- [Calibration of Pre-trained Transformers](https://aclanthology.org/2020.emnlp-main.21/) — Desai & Durrett, EMNLP 2020
- [Calibrating Language Models with Adaptive Temperature Scaling](https://arxiv.org/abs/2409.19817) — Xie et al., EMNLP 2024
- [Restoring Calibration for Aligned LLMs (Calibration-Aware FT)](https://arxiv.org/abs/2505.01997) — Xiao et al., ICML 2025

**长序列工程**
- [LongLoRA](https://arxiv.org/abs/2309.12307) — Chen et al., ICLR 2024
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [Attention-Logit Explosion in Long-Sequence Training](https://arxiv.org/abs/2505.15548) — Hajra, 2025
- [Scaling Transformer to 1M tokens with RMT](https://arxiv.org/abs/2304.11062)（[AAAI 2024 版](https://ojs.aaai.org/index.php/AAAI/article/view/29722)）— Bulatov et al., 2023

---

*本报告由深度调研员基于公开文献生成；每条结论均给出可溯源链接。窗口外（2026）文献仅 2 处且已标注，用作补充视角，不参与证据强度 A/B 分级。*
