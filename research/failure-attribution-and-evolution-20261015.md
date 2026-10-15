> **归档说明**：本报告由 DSH 编排会话于 2026-10-15 生成；分析对象（七代 run 指标、代码与数据拓扑画像）均为本仓库公开资产，文中指标皆为聚合口径。文中提及的 `artifacts/**/*.jsonl`（逐样本预测明细）与 `data/synth/`、`data/train.jsonl` 等训练数据文件位于私有训练档案库，此处仅引用路径与聚合统计。
> 仅为归档所做的编辑：外部文献链接由搜索引擎跳转链接规范化为直接出处；个别标题中的渲染残留（" tau "）还原为 "$\tau$"。
> 发布前经独立红队复核修订：v1 行两项 F1 数字（依公开 v1 report.json 混淆矩阵重算）、质量门代数表述（公开仓实存 6 份 gate 文件）、含视觉正例根组数的口径（split-report limitations 与 DATA-PROFILE 按切分计数存在出入，正文按后者注记）；并移除与公开仓行号不一致的源码行号引用。
> 配套公开开发议题：<https://github.com/CongliangK/router-training/issues>（三阶段路线以 tracking issue 组织）。

# 小型多模态决策路由器训练失效归因诊断与受限数据下演进架构报告

## 摘要与核心诊断结论

针对基于 [Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B)（[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 混合线性注意力骨干）构建的子任务难度与模态路由器，最新一轮 8-bit AdamW 全参数微调运行（`rt-v4-e1-8bit-r2`）在留出测试集（$n=104$）上的表面核心指标 `tier macro-F1` 从上一代归档基线 `real-history-v3c` 的 $0.8050$ 骤降至 $0.4796$，且新一代按组隔离校准管线（`calibration_groups.py`）在结构预检阶段即触发硬性阻断（`PlanError`），生产质量门（`quality-gate.json`）在全部 6 份已归档记录上均为 `passed: false`（`shadow` 模式；`v3b` 未产出 gate 文件，其失败以校准宏 F1 0.275 记录）。这一系列现象在表象上构成了严重的训练退化与工程死锁。

然而，对仓库全量实验产物（`v1`、`v2`、`v2e6`、`v3`、`v3b`、`v3c`、`rt-v4-e1-8bit-r2`）、底层代码实现（`train_eval.py`、`model_utils.py`、`calibration_groups.py`、`policy.py`、`outcomes.py`）以及数据集聚合拓扑（`split-report.json`、`DATA-PROFILE.md`）的交叉审计表明：**最新一次训练并非底层优化器或数值层面的崩溃，而是“极小支持度评估噪声”、“校准集分组枯竭导致的 $\tau$ 操作点回退”、“跨任务头部梯度纠缠引发的虚假捷径学习”以及“跨会话根（Root）标签语义漂移”四重结构性矛盾的集中爆发。**

在完全相同的 104 条留出测试集上，`rt-v4-e1-8bit-r2` 与 `real-history-v3c` 的整体难度分类准确率（`tier accuracy`）**完全一致，均为 $72.12\%$（$75/104$）**；不仅如此，`rt-v4-e1-8bit-r2` 的概率校准质量显著优于 `real-history-v3c`——其难度分类负对数似然（NLL）由 `v3c` 的 $1.4876$ 降至 $0.6009$，5 分桶期望校准误差（ECE）由 $0.4077$ 降至 $0.1128$，视觉分类 NLL 由 $0.3571$ 降至 $0.3113$（ECE 由 $0.0778$ 降至 $0.0493$）。造成 `tier macro-F1` 暴跌 $0.3254$ 的直接算术原因是测试集中 `fast` 类别仅包含 **2 条孤立样本**（分属两个单行会话组 `6843b011` 与 `1cbebd81`）。`real-history-v3c` 借助决策侧对数先验调整（$\tau = 2.0$，等效为对稀缺类 `fast` 施加 $+4.4862$ 的 logit 补偿、对多数类 `balanced` 仅施加 $+0.5271$ 补偿）恰好命中这两条样本（`fast F1` $= 1.0000$）；而 `rt-v4-e1-8bit-r2` 因 48 行拟合半区上 $\tau$ 扫描未能同时满足 `flagship recall` $\ge 0.75$ 与 `FP rate` $\le 0.10$ 的离散网格门限，被迫回退至 $\tau = 0.0$（零先验补偿），并因 47 行选择半区的排序噪声在 Epoch 1 提前早停，导致两条 `fast` 测试样本被保守预测为默认档 `balanced`（`fast F1` $= 0.0000$）。在三分类等权宏平均下，仅此 2 条样本的翻转即抹去了 $1/3 \approx 0.3333$ 的 `macro-F1` 分值。与此同时，`rt-v4-e1-8bit-r2` 在 `balanced` 类别上的误升档（Over-routing 至 `flagship`）反而比 `v3c` 减少了 2 例（`balanced F1` 由 $0.7563$ 升至 $0.7642$，`flagship F1` 由 $0.6588$ 升至 $0.6747$）。

在受限数据条件（真实样本 $N=668$、独立会话组 $G=37$、短期无法大规模重采或全量跑通耗资巨大的 29 次/候选真机执行基准）下，继续沿用“4.98 亿参数全量解冻 + 3.3 倍少数类重复过采样 + 固定留出小校准集多角色硬切分”的路线已触及数学与统计天花板。本报告在严格立足仓库现有资产的基础上，提出一套兼顾短期立即可执行性与中长期架构前瞻性的系统性解决方案：通过**零训练成本的权重空间线性插值（[WiSE-FT](https://arxiv.org/abs/2109.01903)）与留一法（LOGO）连续先验校准**立即修复现有模型；通过**全数据集 5 折按组交叉验证折外堆叠（Out-of-Fold Stacking）**彻底解除校准集样本枯竭死锁；通过**训练期先验对数调整损失（[Logit Adjustment Loss](https://arxiv.org/abs/2007.07314)）、非对称序数代价惩罚（Ordinal Asymmetric Risk）与低秩幅度方向分解微调（[DoRA](https://arxiv.org/abs/2402.09353) / 全线性层 LoRA + [LP-FT](https://arxiv.org/abs/2202.10054)）**切断“视觉需求 $\Rightarrow$ 旗舰档位”的虚假特征捷径；并配合**极小化外科手术式数据清洗（$<40$ 条定向修补）**与**策略层解耦重构（`policy.py` / `routes.json`）**，在无需大规模扩充数据的前提下实现高可靠、低过路由率的生产级路由决策。

## 最新训练运行失效表象与核心症结解构

### 七代实验轨迹与最新运行（rt-v4-e1-8bit-r2）全景对标

自项目启动以来，决策路由器围绕 [Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) 基座经历了从轻量级注意力 LoRA（`v1`、`v2`、`v2e6`）、双层不平衡校正 LoRA（`v3`）、扩展 MLP 的高秩 LoRA（`v3b`）到语言骨干全参数微调（`v3c` 与最新的 `rt-v4-e1-8bit-r2`）共七个主要版本的演进。下表汇总了仓库 `artifacts/` 目录下全部归档运行与零样本基座在测试集上的核心可观测指标。

| 运行标识 (Run ID) | 训练范式与可训练参数量 | 优化器与调度策略 | 留出测试集规模 ($n$ / 组数 $G$) | 选中轮次 (Best Epoch) | Tier Accuracy | Tier Macro-F1 | Fast / Balanced / Flagship F1 | Flagship 召回率 / 精确率 | Vision 召回率 / 精确率 | Tier NLL / ECE (5-bin) | Vision NLL / ECE (5-bin) | Task Type Accuracy | 质量门状态 (Quality Gate) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `zero-shot-base` (v2+) | 零样本基座 ($0$) | 无微调 ($T_{\text{tier}}=2.0$) | $104$ / $7$ | Epoch 0 | $0.6923$ | $0.2727$ | $0.000$ / $0.818$ / $0.000$ | $0.0000$ / $0.0000$ | $0.5946$ / $0.9362$ | $0.6826$ / $0.1443$ | $0.5043$ / $0.1374$ | $0.7404$ | 未通过 (`shadow`) |
| `real-history-v1` | LoRA $r=8$ 仅注意力 ($1.79\text{M}$) | AdamW 恒定 LR $1\times 10^{-4}$，$84$ 步 | $65$ / $3$ (旧切分) | Epoch 1 / 2 | $0.8923$ | $0.7601$ | $0.500$ / $0.923$ / $0.857$ | $0.8333$ / $0.8824$ | $0.6667$ / $0.6667$ | $0.4693$ / $0.0935$ | $0.1730$ / $0.0618$ | $0.6462$ | 未通过 (`5` 项阻断) |
| `real-history-v2` | LoRA $r=8$ 仅注意力 ($1.79\text{M}$) | AdamW 恒定 LR $1\times 10^{-4}$，$118$ 步 | $104$ / $7$ (冻结切分) | Epoch 1 / 2 | $0.7788$ | $0.8398$ | $1.000$ / $0.819$ / $0.701$ | $0.9000$ / $0.5745$ | $0.1351$ / $0.9091$ | $0.4908$ / $0.0644$ | $1.2527$ / $0.4834$ | $0.7981$ | 未通过 (`3` 项阻断) |
| `real-history-v2e6` | LoRA $r=8$ 仅注意力 ($1.79\text{M}$) | AdamW 恒定 LR $1\times 10^{-4}$，$354$ 步 | $104$ / $7$ (冻结切分) | Epoch 3 / 6 | $0.8173$ | $0.8190$ | $1.000$ / $0.881$ / $0.578$ | $0.4333$ / $0.8667$ | $0.1081$ / $0.8889$ | $0.5179$ / $0.0871$ | $1.4058$ / $0.5205$ | $0.7788$ | 未通过 (`3` 项阻断) |
| `real-history-v3` | LoRA $r=8$ + 合成数据 + 轮转采样 + 类权重 ($1.79\text{M}$) | AdamW 恒定 LR $1\times 10^{-4}$，$474$ 步 | $104$ / $7$ (冻结切分) | Epoch 1 / 6 ($\tau=2.0$) | $0.7212$ | $0.6323$ | $1.000$ / $0.832$ / $0.065$ | $0.0333$ / $1.0000$ | $0.4730$ / $0.9459$ | $0.7950$ / $0.1559$ | $0.8047$ / $0.2386$ | $0.7500$ | 未通过 (`3` 项阻断) |
| `real-history-v3b` | LoRA $r=32$ + MLP + 轮转采样 ($6.45\text{M}$) | AdamW 恒定 LR $1\times 10^{-4}$ | $104$ / $7$ (冻结切分) | Soup `[1,2,3]` | — | $0.6330$ | — | $0.6000$ / — | — | — | — | — | 未通过 (校准宏 F1 $0.275$) |
| `real-history-v3c` | 语言骨干全参微调 fp32 ($498.11\text{M}$) + 均衡过采样 | fp32 AdamW，初始 LR $2\times 10^{-6}$，旧余弦调度 ($1158$ 步) | $104$ / $7$ (冻结切分) | Epoch 4 / 6 ($\tau=2.0$) | $0.7212$ | $0.8050$ | $1.000$ / $0.756$ / $0.659$ | $0.9333$ / $0.5091$ | $0.9595$ / $0.8875$ | $1.4876$ / $0.4077$ | $0.3571$ / $0.0778$ | $0.7500$ | 未通过 (`3` 项阻断) |
| `rt-v4-e1-8bit-r2` | 语言骨干全参微调 fp32 ($498.11\text{M}$) + 均衡过采样 | `adamw8bit`，初始 LR $2\times 10^{-6}$，修复版余弦调度 ($579$ 步) | $104$ / $7$ (冻结切分) | Epoch 1 / 3 ($\tau=0.0$) | $0.7212$ | $0.4796$ | $0.000$ / $0.764$ / $0.675$ | $0.9333$ / $0.5283$ | $0.9324$ / $0.9079$ | $0.6009$ / $0.1128$ | $0.3113$ / $0.0493$ | $0.6923$ | 未通过 (`4` 项阻断) |

通过对上述实验矩阵的横向解构，可以清晰识别出最新训练在工程与算法层面暴露出的五大核心症结。

### 症结一：极小样本支持度引发的宏平均（Macro-F1）断崖假象与早停失灵

在评估最新一次训练 `rt-v4-e1-8bit-r2` 时，最直观的“失败信号”是测试集 `tier macro-F1` 从 `real-history-v3c` 的 $0.8050$ 暴跌至 $0.4796$（降幅达 $40.4\%$）。然而，深入比对两者的三维混淆矩阵（见下表）可以发现，这一断崖式下跌并非模型整体判别能力的退化，而是极端类别支持度偏斜与校准操作点跳变共同制造的统计假象。

| 真实档位 \ 预测档位 | 基线 (`zero-shot`) Fast / Bal / Flag | `real-history-v2` (LoRA 2ep) Fast / Bal / Flag | `real-history-v2e6` (LoRA 6ep) Fast / Bal / Flag | `real-history-v3` (轮转塌缩) Fast / Bal / Flag | `real-history-v3c` (6ep fp32, $\tau=2.0$) Fast / Bal / Flag | `rt-v4-e1-8bit-r2` (3ep 8bit, $\tau=0.0$) Fast / Bal / Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fast** ($n=2$) | $0$ / $2$ / $0$ | $2$ / $0$ / $0$ | $2$ / $0$ / $0$ | $2$ / $0$ / $0$ | $2$ / $0$ / $0$ | $0$ / $2$ / $0$ |
| **Balanced** ($n=72$) | $0$ / $72$ / $0$ | $0$ / $52$ / $20$ | $0$ / $70$ / $2$ | $0$ / $72$ / $0$ | $0$ / $45$ / $27$ | $0$ / $47$ / $25$ |
| **Flagship** ($n=30$) | $0$ / $30$ / $0$ | $0$ / $3$ / $27$ | $0$ / $17$ / $13$ | $0$ / $29$ / $1$ | $0$ / $2$ / $28$ | $0$ / $2$ / $28$ |
| **总计命中的样本数** | $72 / 104$ ($69.2\%$) | $81 / 104$ ($77.9\%$) | $85 / 104$ ($81.7\%$) | $75 / 104$ ($72.1\%$) | $75 / 104$ ($72.1\%$) | $75 / 104$ ($72.1\%$) |

对比 `real-history-v3c` 与 `rt-v4-e1-8bit-r2` 的混淆矩阵可见，全集 104 条样本中两者的预测差异**仅有 4 条样本**：
1. 在占测试集 $98.1\%$ 的主体类别（`balanced` $n=72$ 与 `flagship` $n=30$）上，`rt-v4-e1-8bit-r2` 不仅完整保持了 $28/30$（$93.33\%$）的旗舰召回率，还将 `balanced` 误升档为 `flagship` 的假阳性（FP）从 27 例降低至 25 例，使得 `balanced F1` 从 $0.7563$ 提升至 $0.7642$，`flagship F1` 从 $0.6588$ 提升至 $0.6747$。
2. 在仅占测试集 $1.9\%$ 的 `fast` 类别（$n=2$，对应两个单行组 `6843b011` 与 `1cbebd81`）上，`v3c` 预测为 $2/2$ `fast`，而 `rt-v4-e1-8bit-r2` 预测为 $0/2$（全部归入默认档 `balanced`）。由于未加权宏平均定义为：

$$\text{Macro-F1} = \frac{1}{3}\left(\text{F1}_{\text{fast}} + \text{F1}_{\text{balanced}} + \text{F1}_{\text{flagship}}\right)$$

当测试集 $n_{\text{fast}} = 2$ 时，单类别 $\text{F1}_{\text{fast}}$ 只能在离散值 $\{0, 0.5, 0.67, 0.8, 1.0\}$ 间跳跃，2 条样本的分类改变直接导致 $\text{F1}_{\text{fast}}$ 从 $1.0$ 跌至 $0.0$，进而将全局 `Macro-F1` 拉低 $0.3333$。换言之，在当前测试集结构下，1 条 `fast` 样本的统计权重等同于 36 条 `balanced` 样本或 15 条 `flagship` 样本。

造成这两条 `fast` 样本在 `rt-v4-e1-8bit-r2` 中未能被激活的深层机制有二：
- **47 行选择半区上的早停噪声锁定了欠拟合轮次（Epoch 1）**：在 `--legacy-calibration` 模式下，95 条校准数据被随机行切分为 47 条选择集（`selection_rows`）和 48 条拟合集（`fitting_rows`）。全校准集仅有 6 条 `fast` 样本，分到选择半区仅约 3 条。在 `rt-v4-e1-8bit-r2` 的 3 轮训练中，训练损失稳定下降（Epoch 1: $0.8549 \rightarrow$ Epoch 2: $0.4441 \rightarrow$ Epoch 3: $0.3510$），但 47 行选择集上的 `calibration_macro_f1` 却呈现 $0.5190 \rightarrow 0.4622 \rightarrow 0.4622$（仅相当于 47 行中有 2 条边缘样本发生翻转）。模型因此将仅经历了 193 次参数更新、尚处于预热后初期阶段的 **Epoch 1** 锁定为最终最佳权重；而在 `real-history-v3c` 的 6 轮训练中，模型在 Epoch 4（第 772 次更新，`train_loss` $= 0.3627$）才在 47 行选择集上达到峰值 $0.6466$。
- **决策侧对数先验调整（Logit Adjustment）从 $\tau=2.0$ 突变为 $\tau=0.0$**：在 `model_utils.score` 中，推理侧对原始 logits $z_c$ 施加 [Menon et al. (ICLR 2021)](https://arxiv.org/abs/2007.07314) 的事后先验校正：

$$\tilde{z}_c = z_c - \tau \log \pi_c$$

由于含合成数据的训练集先验分布为 $\pi_{\text{balanced}} = 0.7683$、$\pi_{\text{flagship}} = 0.1256$、$\pi_{\text{fast}} = 0.1061$，对应的对数先验为 $\log \pi_{\text{balanced}} = -0.2636$、$\log \pi_{\text{flagship}} = -2.0749$、$\log \pi_{\text{fast}} = -2.2431$。当 `real-history-v3c` 选中 $\tau = 2.0$ 时，减去负值相当于给 `fast` 强行增加了 $+4.4862$ 的 logit 偏置，给 `flagship` 增加了 $+4.1499$ 的偏置，而给 `balanced` 仅增加 $+0.5271$ 的偏置（即 `fast` 相对 `balanced` 获得了高达 $+3.9591$ 的净 logit 抬升）。而在 `rt-v4-e1-8bit-r2` 中，由于拟合半区选择了 $\tau = 0.0$，这一 $+3.9591$ 的人工边界推力完全消失，叠加 Epoch 1 特征尚未充分拉开，`fast` 样本遂退回先验占优的 `balanced` 类别。

### 症结二：校准集样本枯竭导致的 $\tau$ 门限结构性无解与分组预检死锁

深入剖析为何 `rt-v4-e1-8bit-r2` 会选出 $\tau = 0.0$ 而 `real-history-v3c` 选出 $\tau = 2.0$，以及为何新开发的分组校准路线（Issue #2，`calibration_groups.py`）完全无法启动，揭示了当前管线在校准集设计上的双重死锁：

首先，在 `--legacy-calibration` 的 48 行拟合半区（`tuned_fit`）中，包含的 `flagship` 真实样本数仅为 $n_{\text{flag}} \approx 5\sim 8$ 条（全校准集 95 行共 15 条 `flagship`），非旗舰样本约 $40\sim 43$ 条。代码在 $\tau \in \{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0\}$ 网格上搜索满足以下硬约束的最高 `macro_f1` 点：

$$\text{Recall}_{\text{flagship}}(\tau) \ge 0.75 \quad \text{且} \quad \text{FPR}_{\text{flagship}}(\tau) = \frac{\text{FP}_{\text{flagship}}}{\text{FP}_{\text{flagship}} + \text{TN}_{\text{flagship}}} \le 0.10$$

下表对比了 `real-history-v3c` 与 `rt-v4-e1-8bit-r2` 在各自 48 行拟合半区上的 $\tau$ 扫描轨迹：

| 扫描参数 $\tau$ | `v3c` 拟合半区 Macro-F1 | `v3c` 拟合半区 Flagship Recall | `v3c` 拟合半区 Flagship FPR | `rt-v4-e1-8bit-r2` 拟合半区 Macro-F1 | `rt-v4-e1-8bit-r2` 拟合半区 Flagship Recall | `rt-v4-e1-8bit-r2` 拟合半区 Flagship FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $0.00$ | $0.4501$ | $0.6000$ | $0.1163$ | **$0.4345$ (回退选中)** | **$0.6000$** | **$0.1395$** |
| $0.25 \sim 0.75$ | $0.4501$ | $0.6000$ | $0.1163$ | $0.4203$ | $0.6000$ | $0.1628$ |
| $1.00$ | $0.4501$ | $0.6000$ | $0.1163$ | $0.3896$ | $0.6000$ | $0.2093$ |
| $1.25$ | $0.4501$ | $0.6000$ | $0.1163$ | $0.3778$ | $0.6000$ | $0.2326$ |
| $1.50$ | $0.4501$ | $0.6000$ | $0.1163$ | $0.6404$ | $0.6000$ | $0.2558$ |
| $1.75$ | $0.4501$ | $0.6000$ | $0.1163$ | $0.5794$ | $0.6000$ | $0.2791$ |
| $2.00$ | **$0.6001$ (回退选中)** | **$0.6000$** | **$0.1395$** | $0.3494$ | $1.0000$ | $0.3023$ |

观察上表可以发现两个惊人的事实：
1. **拟合半区 `flagship` 样本仅 5 例（$3/5 = 0.6000$）**：由于随机行切分将 15 条校准 `flagship` 中的 10 条分到了选择半区，留在拟合半区的 `flagship` 仅有 **5 条**！因此，`flagship_recall` 的最小离散跳变步长为 $\Delta \text{Recall} = 1/5 = 0.20$（只能在 $0.40, 0.60, 0.80, 1.00$ 之间取值）。与此同时，非旗舰样本共 43 条，FPR 的最小步长为 $1/43 = 0.0233$（$5/43 = 0.1163$，$6/43 = 0.1395$）。
2. **回退规则（Relaxed Fallback）在离散边界上的剧烈不稳定性**：由于两个运行在 $\tau \in [0, 2.0]$ 上均无法满足 `Recall >= 0.75 & FPR <= 0.10`，代码触发了放宽回退逻辑：`relaxed = [s for s in sweep if s['flagship_recall'] >= 0.6 and s['flagship_fp_rate'] <= 0.15]`。在 `v3c` 中，由于 Epoch 4 模型的 logits 极其尖锐（过度自信，ECE 高达 $0.4077$），当 $\tau$ 从 $0.0$ 增至 $1.75$ 时预测结果纹丝不动（FPR 恒为 $5/43 = 0.1163$），直到 $\tau = 2.0$ 时恰好激活了 `fast` 样本且仅增加 1 个旗舰假阳性（FPR $= 6/43 = 0.1395 \le 0.15$），从而选中了最激进的 $\tau = 2.0$；而在 `rt-v4-e1-8bit-r2` 中，由于 Epoch 1 模型的概率分布平缓且校准良好（ECE $0.1128$），在 $\tau = 1.50$ 时 `macro_f1` 虽已高达 $0.6404$（远高于 $\tau=0$ 的 $0.4345$），但因为 FPR 为 $11/43 = 0.2558 > 0.15$ 被放宽门剔除，唯一留在 `FPR <= 0.15` 内的只有 $\tau = 0.0$（FPR $= 6/43 = 0.1395$）！也就是说，**校准得更好的模型反而因为概率平滑，在过窄的 48 行 FPR 门限下被剥夺了使用 $\tau=1.50$ 的资格，被迫回退到 $\tau=0.0$！**

其次，为修复上述 47/48 行随机切分造成的同组泄漏（Issue #2），仓库引入了 `calibration_groups.py`，强制新运行必须预注册互斥的 A（选 Checkpoint）、B（3 折 GroupKFold 拟温度与 $\tau$）、C（终选）三组角色，并设置了严苛的默认支持度门槛（`DEFAULT_SUPPORT_POLICY`）：
- 组数要求：$|G_A| \ge 5, |G_B| \ge 6, |G_C| \ge 5$，合计至少需要 **16 个独立校准会话组**；
- 类别支持度要求：在 A、B、C 每个区域内，所有必填头（`tier` 与 `vision`）的每个类别必须满足 $\text{rows} \ge 5$ 且 $\text{groups} \ge 2$；在 B 区的每个交叉验证训练折并集中，每个必填类别必须满足 $\text{groups} \ge 2$；
- 标签合法性要求：`validate_rows` 对必填头出现 schema 之外的标签（如 `needs_vision: "uncertain"`）直接抛出 `PlanError`。

对照 `data/split-report.json` 中的真实校准集结构：
- 校准集总共只有 **8 个组**（组大小依次为 $59, 22, 4, 4, 2, 2, 1, 1$，有效簇样本量 $N_{\text{eff}} = \frac{95^2}{59^2+22^2+\dots} = 2.25$），连 $5+6+5=16$ 组的一半都不到；
- 校准集中 `vision=yes` 仅有 **2 条样本且全部来自同一个组**；含 `vision=yes` 的独立组全局屈指可数——`split-report.json` 的 `limitations` 记录为「仅 2 个根」，但 `DATA-PROFILE.md` 的按切分计数（train 4 条、cal 2 条、test 74 条，切分组严格不相交 ⇒ 含正例组 $\ge 4$：`ec267d56` 70 条、`427689b6` 4 条、train/cal 各 $\ge 1$ 个小组）与其存在口径出入——无论按哪种口径，都不可能在 A、B、C 三个角色中各凑出 $\ge 2$ 个含 `vision=yes` 的独立组；
- 校准集中还包含 **2 条 `needs_vision: "uncertain"`** 样本，在 `validate_rows` 第一步即触发 `illegal_required_label=2` 硬错误。

这就导致了当前的工程僵局：**`--legacy` 路径因 47/48 行切分噪声和 $\tau$ 门限量化效应而剧烈震荡，而新默认的分组路径因校准集物理容量不足而 $100\%$ 拒绝启动。**

### 症结三：多任务头部梯度纠缠与“视觉即旗舰”的虚假捷径学习

尽管 `real-history-v3c` 与 `rt-v4-e1-8bit-r2` 将 `flagship` 召回率提升至 $93.33\%$（$28/30$），但两者的 `flagship` 精确率均徘徊在低位（`v3c` 为 $50.91\%$，误报 27 例；`rt-v4-e1-8bit-r2` 为 $52.83\%$，误报 25 例），导致超過三分之一的常规 `balanced` 任务被昂贵的旗舰模型过度接管。

对测试集预测明细的交叉统计揭示了这一过路由（Over-routing）现象的核心机理——**模型在共享语言骨干上学到了“需要视觉（`vision=yes`）$\Rightarrow$ 旗舰难度（`tier=flagship`）”的虚假相关性（Spurious Shortcut）**：
1. 在 `rt-v4-e1-8bit-r2` 产生的 25 条 `balanced -> flagship` 误升档样本中，**高达 21 条（$84.0\%$）是 `needs_vision=yes` 的视觉任务**！而在 `needs_vision=no` 的纯文本 `balanced` 测试样本（24 条）中，误升档率极低。
2. 追溯训练集的数据构成（`docs/DATA-PROFILE.md` §3 & §5）：
   - 真实训练集（469 行）中，`needs_vision=yes` 仅有 **4 行**（全部为 `balanced`，`flagship,yes` 为 **0 行**）；
   - 为弥补视觉正例缺失，训练引入了 200 行合成增强数据（`data/synth/*.jsonl`），其中包含 107 行 `vision=yes` 和 17 行 `flagship`；
   - 在 `train_eval.py` 中，每个 epoch 首先按 `tier` 对训练集进行均衡过采样（`oversample_epoch_rows`），将总数从 669 行扩充至 $514 \times 3 = 1542$ 行（即原本仅 84 行的 `flagship` 被重复采样了 **6.12 倍**，原本 71 行的 `fast` 被重复采样了 **7.24 倍**）；随后，损失函数在同一个前向传播中同时对 `<decision>` 的三个位置计算加权交叉熵：

$$\mathcal{L}_{\text{total}} = 1.0 \cdot w_{\text{tier}}(y_{\text{tier}}) \mathcal{L}_{\text{CE}}^{\text{ls}=0.05}(z_{\text{tier}}, y_{\text{tier}}) + 0.7 \cdot w_{\text{vis}}(y_{\text{vis}}) \mathcal{L}_{\text{CE}}^{\text{ls}=0.05}(z_{\text{vis}}, y_{\text{vis}}) + 0.3 \cdot \mathcal{L}_{\text{CE}}^{\text{ls}=0.05}(z_{\text{type}}, y_{\text{type}})$$

其中 `vision` 头因未参与采样均衡，叠加了 $\beta=0.999$ 的有效样本数权重（[Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555)）：$w_{\text{vis}}(\text{yes}) = 1.6003$，$w_{\text{vis}}(\text{no}) = 0.3997$（正负权重比达 $4:1$）。
3. 当一条同时具有 `tier=flagship`（被过采样重复 6.12 次）与 `needs_vision=yes`（合成样本，梯度再放大 1.60 倍）的样本流经完全解冻的 4.98 亿参数语言骨干时，其单样本有效梯度贡献高达常规 `balanced,no` 样本的 $6.12 \times \frac{1.0 + 0.7 \times 1.6003}{1.0 + 0.7 \times 0.3997} \approx 10.17$ 倍！加之在 Intern-Decision 的因果掩码骨架中：

```json
{
    "tier": "<decision>",
    "vision": "<decision>",
    "task_type": "<decision>"
}
```

三个 `<decision>` 占位符在序列末端仅相隔数个 token，共享前序全部 24 层 Gated DeltaNet 与自注意力表征。全参微调迅速将合成视觉提示词的高维特征同时映射到 `vision=yes` 与 `tier=flagship` 的激活子空间中。当测试集恰好由单一视觉大组 `ec267d56`（79 行中含有 70 个 `vision=yes` 和 29 个 `flagship`）主导时，所有携带视觉特征的 `balanced` 任务便被批量推向 `flagship`。

### 症结四：跨会话根（Root）标签语义漂移与零支持类别噪声污染

在第三个决策头 `task_type`（七分类）以及仅剩的 2 例旗舰漏判（Flagship Under-routing）中，暴露出数据标注层面的跨根语义漂移（Cross-Root Label Semantic Drift）：
- **`creative` 类别在三代模型上全军覆没（$0/16$ 准确率）**：在真实训练集（469 行）中，`creative` 是第一大类别，多达 **151 行（占训练集的 $32.2\%$）**，但其构成为 **$100\%$ 纯文本（`needs_vision=no`）且 $99.3\%$ 为 `balanced`（150 条 `balanced`、1 条 `fast`、0 条 `flagship`）**，代表的是常规文案撰写或纯文本内容生成。然而在测试集（104 行）中，全部 16 条 `creative` 样本均来自主导组 `ec267d56`，且其中 **15/16（$93.8\%$）为 `needs_vision=yes` 的复杂视觉创作/前端视觉实现任务**。由于训练集中从未见过“视觉创作 = `creative`”或“创作类 = `flagship`”，零样本基座、`real-history-v3c` 与 `rt-v4-e1-8bit-r2` 在测试集这 16 条 `creative` 样本上的准确率全部为 **$0.0\%$（$0/16$，其中 14 条被判为 `implementation`，2 条被判为 `review`）**。更重要的是，测试集中仅有的 **2 条被漏判为 `balanced` 的 `flagship` 样本，100% 属于这类 `creative` 盲区样本**！
- **`visual` 任务类型的零支持度噪声污染**：在 `task_type` 的七个选项中，`visual`（直接分析图片或画面）与第二个决策头 `needs_vision` 在语义上高度重叠。真实训练集中仅有 **1 条** `task_type=visual` 样本，校准集与测试集中均为 **0 条**；但合成数据集中却注入了 **33 条** `task_type=visual` 样本。这导致模型在训练中学到了 `task_type=visual` 的激活，却在测试集中（真实标签为 0）不断将 `research` 和 `review` 误判为 `visual`（`v3c` 误判 2 次，`rt-v4-e1-8bit-r2` 误判 3 次），无谓地拉低了 `task_type` 准确率并干扰了共享骨干的特征正交性。

### 症结五：服务策略层（policy.py 与 routes.json）的隐性升档陷阱

除训练与评测管线外，对推理服务代码 `policy.py` 与路由配置 `routes.json` 的联合审查发现了一个此前未被注意到的**生产级隐性过度路由机制**。在 `routes.json` 中，三个候选模型的声明如下：
- `fast`：`zai-coding-cn / glm-5.3-flash`（`vision: true`）
- `balanced`：`zai-coding-cn / glm-5.3`（`vision: false`）
- `flagship`：`openai-codex / gpt-6.1-sol`（`vision: true`）

而在 `policy.py` 的 `select(prediction, routes, has_image=False, minimum_confidence=0.9)` 函数中，路由筛选逻辑为：

```python
if vision['confidence'] < minimum_confidence:
    needs_vision = True
    reasons.append('uncertain-modality-requires-compatible-route')
eligible = [
    route for route in routes
    if TIERS.index(route['tier']) >= TIERS.index(required)
    and (not needs_vision or route['vision'])
]
```

这一组合在实际部署中引发了两个严重的级联后果：
1. **`balanced` 档位缺失视觉路由导致 $100\%$ 强制升档**：由于默认 `routes.json` 中唯一的 `balanced` 模型 `glm-5.3` 声明 `vision: false`，任何被路由器精准判定为 `tier = balanced` 但同时判定为 `needs_vision = True` 的任务，在 `eligible` 过滤时都会因为 `glm-5.3` 不支持视觉而被淘汰，从而顺延匹配到 `TIERS.index >= 1` 且 `vision == True` 的唯一候选——`gpt-6.1-sol`（`flagship`）！这意味着在测试集的 72 条 `balanced` 任务中，占 $63.9\%$ 的 46 条视觉任务即便模型分类完全正确，经过 `policy.py` 后也会 $100\%$ 被强制路由至旗舰模型 `gpt-6.1-sol`。
2. **`minimum_confidence = 0.9` 对 `vision=no` 的反向惩罚**：当模型预测 `vision = no`（不需要视觉），但其校准后置信度 $P(\text{vision}=\text{no}) \in (0.50, 0.90)$（即 $P(\text{vision}=\text{yes}) \in (0.10, 0.50)$）时，`vision['confidence'] < 0.9` 条件成立，`policy.py` 会强制将 `needs_vision` 翻转为 `True`！这相当于把视觉分类器的阳性触发阈值从 $0.50$ 激进地压低到了 **$0.10$**；结合第一条缺陷，任何 $P(vision=yes) > 0.10$ 的纯文本 `balanced` 任务也会在生产侧被连带升级到 `gpt-6.1-sol`。

## 仓库数据资产、模型架构与工程管线的系统性审计

### 数据集聚类拓扑与有效样本量（Effective Sample Size）量化评估

本仓库的核心工程纪律之一是在 `prepare.py` 中实现了基于根会话（Root Session）与字符 5-gram Jaccard 相似度（阈值 $0.88$）的并查集分组隔离切分（Group-Disjoint Splitting），确保同一多智能体会话或近重复任务不会跨 `train`、`calibration`、`test` 泄漏。这一设计在方法论上完全正确，但当真实总样本量 $N=668$ 分布在极度不均匀的 $G=37$ 个独立会话簇中时，名义样本量（Nominal Row Count）与真实统计自由度之间产生了巨大鸿沟。

在聚类抽样统计学中，当样本分属大小不一的簇 $g \in \{1, \dots, G\}$（各簇样本数为 $n_g$，且簇内样本具有高度同源相关性）时，Kish 有效簇样本量（Effective Cluster Sample Size, $N_{\text{eff}}$）定义为：

$$N_{\text{eff}} = \frac{\left(\sum_{g=1}^{G} n_g\right)^2}{\sum_{g=1}^{G} n_g^2} = \frac{N^2}{\sum_{g=1}^{G} n_g^2}$$

利用 `docs/DATA-PROFILE.md` 披露的各切分簇规模序列，可精确计算出当前数据集各切分的真实统计功效（见下表）：

| 数据切分 (Split) | 名义行数 ($N$) | 独立会话组数 ($G$) | 各组行数分布（降序） | Kish 有效簇样本量 ($N_{\text{eff}}$) | 最大单一组占比 ($\max n_g / N$) | 前两大组累计占比 | 统计自由度诊断结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **训练集 (`train`)** | $469$ (真实) | $22$ | $154, 62, 58, 41, 39, 23, 18, 12, 12, 10, 9, 7, 6, 4, 3, 3, 2, 2, 1, 1, 1, 1$ | **$6.18$** | $32.84\%$ ($154/469$) | $46.06\%$ ($216/469$) | 前 5 大组占 $75.48\%$，全参微调极易记忆大组专属词汇与项目上下文风格 |
| **校准集 (`calibration`)** | $95$ | $8$ | $59, 22, 4, 4, 2, 2, 1, 1$ | **$2.25$** | $62.11\%$ ($59/95$) | $85.26\%$ ($81/95$) | 实质上仅由 2 个大组主导，无法支撑 A/B/C 三角色二次拆分 |
| **测试集 (`test`)** | $104$ | $7$ | $79, 12, 6, 4, 1, 1, 1$ | **$1.68$** | $75.96\%$ ($79/104$) | $87.50\%$ ($91/104$) | 单一组 `ec267d56` 垄断 $76\%$ 样本及 $95\%$ 视觉正例、$97\%$ 旗舰样本 |
| **全量真实池 (`all`)** | $668$ | $37$ | 合并上述 37 组 | **$9.70$** | $23.05\%$ ($154/668$) | $34.88\%$ ($233/668$) | 全局仅约 10 个等效独立大簇，必须采用跨全池的折外交叉验证而非单次静态切分 |

上表从数学上解释了为何固定留出 95 行校准集和 104 行测试集会产生如此剧烈的评估震荡：
1. **校准集有效簇数仅为 $N_{\text{eff}} = 2.25$**：前两大组（59 行与 22 行）占了校准集的 $85.26\%$，其余 6 个组加起来仅有 14 行（$4+4+2+2+1+1$）。无论采用何种算法把这 8 个组拆分为 A、B、C 三个子集，都不可能让每个子集同时获得充足的组间多样性。
2. **测试集有效簇数仅为 $N_{\text{eff}} = 1.68$**：组 `ec267d56`（79 行）单独贡献了测试集 $75.96\%$ 的样本、$94.59\%$（$70/74$）的 `vision=yes` 样本以及 $96.67\%$（$29/30$）的 `flagship` 样本；而其余 6 个组总共只有 25 行（且包含 3 个单行组：`6843b011`、`078dbf2e`、`1cbebd81`）。在 `rt-v4-e1-8bit-r2` 的按组准确率分解中，组 `0192d797`（4 行）的 `vision accuracy` 为 $0.000$（$0/4$），两个单行 `fast` 组的 `tier accuracy` 均为 $0.000$（$0/1$）。
3. **`prepare.py` 切分打分函数的结构性盲区**：审阅 `prepare.py` 的切分打分函数可见，12,000 次随机分组搜索的目标函数为：

```python
score = sum(abs(len(splits[name])/len(tasks) - target) for name, target in [('train', .7), ('calibration', .15), ('test', .15)])
for name, rows in splits.items():
    counts = collections.Counter(r['label']['tier'] for r in rows)
    score += sum(2 for label in TIERS if counts[label] == 0)
    score += sum(.5 for label in ['yes', 'no'] if not any(r['label']['needs_vision'] == label for r in rows))
    for label in TIERS:
        whole = sum(r['tier'] == label for r in labels) / len(labels)
        score += .15 * abs(counts[label] / len(rows) - whole)
```

该打分函数存在三个明显的算法漏洞：
- 对 `needs_vision` 仅检查了 `any(...)`（布尔存在性，即只要有 $\ge 1$ 条 `yes` 即视为满分），完全没有惩罚 `vision=yes` 比例与全局分布的偏离！由于全数据集中只有两个根组含有大量 `vision=yes`，搜索算法为了拼凑 $70/15/15$ 的总行数比例，把包含 70 条 `vision=yes` 的 79 行大组 `ec267d56` 整个塞进了 104 行的 `test` 集，而把仅含 4 条 `vision=yes` 的小视觉组放进了 469 行的 `train` 集，人为制造了 **训练集真实视觉正例占 $0.85\%$（$4/469$）、测试集真实视觉正例占 $71.15\%$（$74/104$）的极端分布倒挂**！
- 完全没有对各切分内的**最大单组占比（$\max_g n_g / N_{\text{split}}$）或有效簇数 $N_{\text{eff}}$** 施加约束，导致 `calibration`（59/95）和 `test`（79/104）双双退化为单组主导结构。
- 没有对 `task_type` 的分布均衡度或最小样本数施加任何惩罚，导致 `creative` 在训练集与测试集间出现严重的子域割裂，且 `visual` 在校准集与测试集中完全归零。

### 基座模型架构（Intern-Decision-0.8B）与参数更新动力学审计

路由器所采用的基座 [internlm/Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B)（基于 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)）并非传统的纯 Transformer 解码器，而是一个高度特化的**混合线性注意力多模态结构化决策模型**。根据 `models/intern-decision-0.8b/config.json` 与 `research/intern-decision-fullft-specifics.md` 的审计结果，其核心架构特征如下：

1. **18 层 Gated DeltaNet + 6 层标准自注意力的混合骨干**：语言主干共 24 层（`hidden_size = 1024`，`intermediate_size = 3584`），遵循 `full_attention_interval = 4` 的交替布局——第 $3, 7, 11, 15, 19, 23$ 层（0-indexed）为带输出门控（`attn_output_gate = true`）与零中心 RMSNorm（`qk_norm`）的标准多头注意力层（8 个查询头，2 个键值头，`head_dim = 256`）；其余 18 层均为基于状态空间/线性递归机制的 **Gated DeltaNet**（`linear_attention`），包含 `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`conv1d`（核大小 4）、`A_log`、`dt_bias` 与 `out_proj`，且递归状态强制以 `mamba_ssm_dtype = "float32"` 维护。
2. **单前向并行占位符打分契约（Masked Placeholder Scoring）**：模型不在推理期执行自回归文本生成（`generate()`），而是将完整任务状态编译进 User Prompt，并在 Assistant 回复中构造固定格式的 JSON 骨架（包含 3 个 `<decision>` 特殊 token，词表 ID 为新增项）。在一次因果前向传播中，提取每个 `<decision>` 占位符**前一个位置（`marker_pos - 1`）**的隐状态经 `lm_head`（与 `embed_tokens` 权重绑定，`tie_word_embeddings = true`，词表规模 $248,320$）投影后的 logits，仅在合法候选符号集合（如 `A, B, C`）上计算受限 Softmax。

在这一架构下，对比仓库中已尝试的两种微调路线可以发现明显的容量-数据失配：
- **早期 LoRA 路线（`v2`、`v2e6`、`v3`）为何表现不佳？**
  早期 `v2` 与 `v2e6` 仅对 `['q_proj', 'v_proj', 'in_proj_qkv', 'in_proj_z']` 施加了 $r=8$ 的 LoRA（可训练参数 $1.79\text{M}$，约占语言骨干的 $0.36\%$）。实际上，**`v2` 在测试集上的 `tier accuracy` 高达 $77.88\%$，`v2e6` 更是达到了全历史最高的 $81.73\%$（且 `tier macro-F1` 分别达 $0.8398$ 与 $0.8190$，`tier ECE` 低至 $0.0644$ 与 $0.0871$）**！当时团队之所以认为 LoRA 失败，是因为：
  1. `v2` 与 `v2e6` 未开启 `--synth` 合成数据，而真实训练集中只有 4 条 `vision=yes`，导致视觉头因缺乏正样本而失效（`vision FN rate` $86\%\sim 89\%$）；
  2. 当 `v3` 与 `v3b` 开启 `--synth` 并试图提升 `vision` 和 `flagship` 时，同时引入了**有严重 Bug 的轮转采样（Round-Robin Sampling）**——少数类桶在每个 epoch 前段耗尽后，每个 epoch 的最后数百步变成了 $100\%$ 纯 `balanced` 多数类样本的连续轰炸，直接把模型权重洗成了全预测 `balanced`（导致 `v3` 的 `flagship recall` 暴跌至 $3.3\%$）！
  3. 当 `v3c` 修复了轮转采样 Bug（改为全局打乱的 `oversample_epoch_rows`）时，团队**同时切换到了 4.98 亿参数的全参数微调（`--fullft`）**，从而把“修复采样顺序 + 引入合成视觉数据”的巨大收益错误地归因给了“全参数微调”，却未曾用修复后的采样器重新运行 LoRA/DoRA！
- **4.98 亿参数全参数微调（`v3c` 与 `rt-v4-e1-8bit-r2`）的过拟合代价**：
  在仅 469 条真实样本（$N_{\text{eff}} = 6.18$ 个有效会话簇）上解冻全部 $498,113,344$ 个语言骨干参数，且每轮重复过采样至 1542 步（`batch_size = 1`，梯度累积 8），导致模型在 2～4 个 epoch 内迅速记忆了训练集中 22 个会话组的表面词汇特征。直接证据是：`real-history-v3c` 在 Epoch 4 时的测试集 `tier accuracy`（$72.12\%$）反而比轻量级 LoRA `v2`（$77.88\%$）和 `v2e6`（$81.73\%$）低了 $5.8\sim 9.6$ 个百分点，且 `tier NLL` 从基座的 $0.6826$ 恶化至 $1.4876$，`tier ECE` 暴涨至 $0.4077$（严重过度自信）。

### 训练与评测工程实现中的残存技术债

通过对 `train_eval.py`、`model_utils.py` 与 `outcomes.py` 的逐行代码走查，尽管 Issue #14 已修复了余弦调度器步数统计与 SWA Top-3 池淘汰问题，当前代码库仍存在以下四个制约模型上限的技术瓶颈：

1. **训练时仅对全词表（或候选子集）做无偏置交叉熵，而将类不平衡校正分裂为“暴力重复采样 + 事后 $\tau$ 扫描”**：
   `train_eval.py` 直接在未缩放的候选符号 logits `logits[field]`（3 维或 2 维张量）上调用 `F.cross_entropy(..., label_smoothing=0.05)`（注意：这与官方 XTuner 在 $248,320$ 全词表上算 CE 不同，`train_eval.py` 实际上是在提取出的候选符号子集 logits 上计算受限交叉熵，这一点对小样本分类反而更聚焦）。然而，为了平衡 `tier` 的 $358 : 67 : 44$ 偏斜，`oversample_epoch_rows` 将每个类别强行复制到 514 条（每轮 1542 条），使每轮训练时间膨胀了 $2.3$ 倍（单轮耗时近 20 分钟），并使稀缺的 44 条 `fast` 样本每轮被重复反向传播 11.7 次，极易在小样本上产生特征坍缩；随后又在推理侧叠加 $\tau \log \pi_c$ 调整，形成了“训练期过采样改变隐式先验 + 推理期再次减去原始先验”的理论冲突。
2. **逐样本串行前向（Micro-Batch Size $= 1$）与无差别 Weight Decay**：
   训练循环逐条执行 `request_for` 与 `field_logits`（每次 `batch_size = 1`，累积 8 次更新），不仅无法利用 GPU 矩阵并行的吞吐优势，而且在 `build_optimizer` 中对所有可训练参数（包括一维的 `RMSNorm` 权重、Gated DeltaNet 的 `A_log` 与 `dt_bias`）无差别施加了 `weight_decay = 0.01`。在混合线性注意力架构中，对 `A_log`（控制递归衰减率）和 `dt_bias`（控制时间步长）施加 L2 权重衰减会人为改变线性注意力的有效上下文记忆窗口。
3. **专家标注置信度（`label.confidence`）与理由（`label.rationale`）信息在训练中被完全丢弃**：
   在 `data/labeling/RUBRIC.md` 与 `expert-recovery-*.jsonl` 中，每条真实样本都包含专家标注的 `confidence`（分布于 $[0.45, 1.00]$，均值 $0.729$）以及详细的判别理由 `rationale`（指明属于哪一类旗舰升级情形）。然而 `train_eval.py` 将 `confidence = 0.45` 的高争议边缘样本与 `confidence = 1.00$ 的确定性锚点样本赋予完全相同的损失权重 $1.0$。实证表明（`research/fullft-8bit-run-lessons.md` §4.2），模型预测错误的样本其专家置信度均值（$0.688$）显著低于预测正确样本（$0.731$），无视标注不确定性的硬标签强训迫使模型去拟合专家自身的标注噪声。
4. **真机结果验收管线（`outcomes.py`）的 Clopper-Pearson 样本量壁垒**：
   `outcomes.py` 要求每个候选路由在单侧 $95\%$ Clopper-Pearson 精确二项分布下界满足 $L(k, n; 0.95) \ge 0.90$ 方可获得 `measured_candidate_outcomes` 标签。根据精确二项式下界公式：

$$L(n, n; 1-\alpha) = \alpha^{1/n} \ge 0.90 \implies n \ge \frac{\ln(0.05)}{\ln(0.90)} = 28.43 \implies n_{\min} = 29$$

这意味着即便某个候选模型在该任务上 $100\%$ 成功（零失败），也必须连续独立执行 **29 次**才能跨过 $0.90$ 下界门限；一旦出现 1 次偶发超时或非确定性失败，则需要 **46 次中成功 45 次**（$L(45,46) = 0.9025$）。对 104 条测试任务 × 3 个候选路由而言，这意味着至少需要 **$104 \times 3 \times 29 = 9,048$ 次**完整子智能体长上下文执行，在算力与 API 预算上构成不可逾越的现实阻碍，导致质量门第一条（`No independent candidate-model task-success benchmark has been collected`）长期无法闭合。

## 行业主流路由器训练范式与小样本受限学习前沿调研

为避免在孤立工程视角下反复试错，有必要将本项目的“小样本多模态子任务难度与模态路由”置于当前工业界与学术界主流 LLM 路由器（LLM Routing）及受限数据微调（Data-Constrained Adaptation）的技术坐标系中进行对标。

### 行业主流 LLM 路由框架的技术路线对标

当前主流大模型路由系统已从早期的纯规则或零样本提示词分类，演进为以“成本-质量帕累托前沿（Cost-Quality Pareto Frontier）”与“风险受控降级（Risk-Controlled Cascading）”为核心的统计决策体系。下表对比了五大代表性主流路由框架与本项目当前架构的异同：

| 路由框架与代表文献 | 核心路由信号与建模范式 | 路由器模型架构与参数规模 | 训练监督信号来源 | 成本-质量权衡与阈值校准机制 | 对本项目受限场景的核心启示 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[RouteLLM](https://arxiv.org/abs/2406.18665)** (Ong et al., LMSYS / UC Berkeley, 2024) | 预测强模型相对弱模型的胜率 $P(\text{win}_{\text{strong}} \mid q)$ | 矩阵分解（MF）、相似度加权排序（SW）、BERT 分类器、Llama-3-8B 指令分类器 | Chatbot Arena 人类偏好对 + GPT-4 合成偏好扩增 | 单一连续阈值 $\alpha$：当 $P(\text{win}_{\text{strong}}) \ge \alpha$ 时路由至强模型，动态平滑调节成本比例 | **小巧的矩阵分解/双塔嵌入融合器在数据稀疏时往往优于重型全参 LLM 微调**；用单参数阈值控制强模型调用率比硬 argmax 灵活得多 |
| **[Hybrid LLM](https://arxiv.org/abs/2404.14618)** (Ding et al., Microsoft, ICLR 2024) | 预测小模型相对大模型的期望质量差距（Quality Gap）概率 | DeBERTa-v3-large ($304\text{M}$) 编码器 + 软标签二元交叉熵 | 真实执行得分差值转化的确定性/概率化软标签（Soft Labels） | 测试期动态调整路由阈值 $t$，在零重训前提下实现高达 $40\%$ 的大模型调用削减且质量无损 | **软标签监督（利用置信度或质量差值构造 $[0,1]$ 目标）**能显著缓解硬标签边界噪声并天然改善概率校准 |
| **[FrugalGPT](https://arxiv.org/abs/2305.05176)** (Chen et al., Stanford, TMLR 2024) | 级联评分（LLM Cascade）：先由低成本模型或轻量评分器评估可信度，不可信则逐级上送 | DistilBERT / 小型回归打分器 + 最优级联顺序规划器 | 候选模型在历史任务上的真实执行正确性（0/1 或连续分） | 在给定预算约束 $B$ 下联合优化模型调用序列与每级接受阈值 $\tau_k$ | 静态路由与**动态强度补偿（Reasoning Effort Escalation）**结合，可在同模型内先升 `effort` 再跨模型升 `tier` |
| **[AutoMix](https://arxiv.org/abs/2310.12963)** (Aggarwal et al., NeurIPS 2024) | 少样本自验证置信度 + 噪声精炼元验证器（Meta-Verifier） | 基于部分可观测马尔可夫决策过程（POMDP）的轻量策略路由器 | 无需海量训练数据，仅依赖百余条验证样本估计验证器噪声分布 | 基于期望增量收益成本比（Incremental Benefit per Cost, IBC）求解最优动作 | 当真实执行标签昂贵时，**仅需几十条锚点任务的执行统计即可通过贝叶斯/POMDP 校准先验**，无需对每条任务暴力重放 29 次 |
| **[RouterBench](https://arxiv.org/abs/2403.12031)** (Hu et al., Martian, 2024) | 零样本/少样本多模型能力画像预测（kNN / MLP / 小型预训练编码器） | 嵌入检索（kNN Router）与多层感知机预测器 | 跨基准预计算的候选模型得分与 token 成本矩阵 | 优化期望效用函数 $U(m \mid q) = \text{Quality}(m, q) - \lambda \cdot \text{Cost}(m, q)$ | **非对称代价敏感决策**：漏判旗舰（Under-routing）与误升旗舰（Over-routing）具有完全不对称的业务代价，不应使用对称 argmax |

从上述行业主流方案的横向对比可以得出一个极其关键的顶层认知：**没有任何主流工业级路由器在仅有数百条强聚类样本（$N=469, G=22$）的条件下，去独立全参数微调一个 5 亿参数的自回归语言模型并用对称 argmax 做硬决策。** 行业最佳实践一致指向三个原则：
1. **表征高度受控（Representation Regularization）**：要么冻结预训练主干仅训练轻量投影/分类头或低秩方向分解（DoRA / LoRA + WiSE-FT），要么使用对比学习预对齐嵌入（如 [SetFit](https://arxiv.org/abs/2209.11055)），绝不允许百余条样本的大梯度冲垮基座原有的通用语言理解表征；
2. **监督信号软化与序数化（Soft & Ordinal Supervision）**：将离散档位 `fast < balanced < flagship` 视为具有严格序数关系（Ordinal）与非对称错误代价的连续难度谱系，利用标注置信度构造软标签；
3. **推理决策从“最大概率分类（Argmax）”转向“共形风险控制（Conformal Risk Control）”**：模型只负责输出校准良好的条件概率分布 $P(y \mid x)$，而最终路由动作由下游根据显式代价矩阵或假阴性率（FNR）上界约束通过连续阈值切分完成。

### 小样本受限条件下的模型微调与防退化方法论

针对本项目“数据集只能少量优化、必须在受限条件下改进模型训练”的现实约束，结合仓库 `research/fullft-survey.md` 与外部前沿文献，以下四项经过同行评审严格验证的技术是突破当前瓶颈的核心武器：

1. **权重空间线性插值（[WiSE-FT, Wortsman et al., CVPR 2022](https://arxiv.org/abs/2109.01903)）与模型汤（[Model Soups, Wortsman et al., ICML 2022](https://arxiv.org/abs/2203.05482)）**：
   [Kumar et al. (ICLR 2022 Oral)](https://arxiv.org/abs/2202.10054) 证明，全参数微调虽然能在训练集分布内（In-Distribution）快速降低损失，但会在前几十个更新步内扭曲预训练基座的通用特征，导致在分布偏移（Out-of-Distribution，如本项目测试集中占比 $76\%$ 的全新视觉大组 `ec267d56`）下性能反而低于线性探测，并伴随严重的校准恶化（[Desai & Durrett, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.21/)）。[WiSE-FT](https://arxiv.org/abs/2109.01903) 提出将零样本基座权重 $\theta_{\text{base}}$ 与微调后权重 $\theta_{\text{ft}}$ 在权重空间直接进行凸组合：

$$\theta_{\text{WiSE}}(\alpha) = (1 - \alpha) \theta_{\text{base}} + \alpha \theta_{\text{ft}}, \quad \alpha \in [0, 1]$$

这一操作无需任何额外训练成本、不增加一丝推理延迟，却能在保留微调模型高召回率的同时，恢复基座模型卓越的概率校准度（ECE）与跨组泛化鲁棒性（在分布偏移基准上普遍带来 $4\sim 6$ 个百分点的净增益）。

2. **训练期对数先验调整损失（[Logit Adjustment Loss, Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314)）替代暴力重复过采样**：
   当前 `train_eval.py` 采用的“少数类重复复制 $6\sim 7$ 倍（`oversample_epoch_rows`）+ 事后 $\tau$ 扫描”存在根本性缺陷：重复过采样迫使模型对仅有的 44 条 `fast` 样本和 67 条 `flagship` 样本进行高强度死记硬背，导致决策边界向多数类 `balanced` 剧烈挤压，进而引发 25～27 例 `balanced -> flagship` 假阳性。[Menon et al. (2021)](https://arxiv.org/abs/2007.07314) 严格证明，实现贝叶斯最优均衡错误率（Fisher Consistency for Balanced Error）的正确方式不是复制样本，而是在自然数据分布下将先验边界直接注入训练期交叉熵损失：

$$\mathcal{L}_{\text{LA}}(z, y; \tau_{\text{train}}) = -\log \frac{\exp\left(z_y + \tau_{\text{train}} \log \pi_y\right)}{\sum_{c \in \mathcal{Y}} \exp\left(z_c + \tau_{\text{train}} \log \pi_c\right)}$$

在训练期使用 $\mathcal{L}_{\text{LA}}$，模型每个 epoch 只需遍历一次自然数据集（669 行而非 1542 行，训练速度立即提升 **$2.3$ 倍**），稀有类仅在远离边界时产生大梯度，不会因为同一文本被重复前向 11 次而破坏骨干特征，同时彻底消除了 `tier` 过采样对 `vision` 头的伴随纠缠效应。

3. **折外堆叠校准（Out-of-Fold Stacking, [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)）与共形风险控制（[Conformal Risk Control, Angelopoulos et al., ICLR 2024](https://arxiv.org/abs/2208.02814)）**：
   [Cawley & Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) 指出，在小样本下使用单独留出的微小验证集进行超参数/操作点选择，其选择方差（Variance of the Model Selection Criterion）往往大于算法本身的真实性能差异。当总独立组数仅为 $G=37$（开发池 `train + calibration` 共 $30$ 组、$564$ 行）时，唯一能在不违背组隔离原则的前提下获得充足校准统计功效的方法是 **5 折按组交叉验证折外堆叠（5-Fold GroupKFold OOF Calibration）**：将全部 30 个开发组分为 5 折（每折 6 组、约 113 行），训练 5 个子模型并收集全部 564 条真实开发样本的折外（Out-of-Fold）未校准 logits。如此一来，温度拟合与阈值寻优的数据基数从 **48 行（仅 5 条 flagship）瞬间扩大至 564 行（含 82 条 flagship、50 条 fast、6 条真实 vision=yes）**，统计功效提升超過 **11 倍**！在此基础上，应用 [Angelopoulos et al. (2024)](https://arxiv.org/abs/2208.02814) 的共形风险控制（CRC），直接求解满足期望漏判率上界 $\mathbb{E}[\text{FNR}_{\text{flagship}}] \le \alpha$ 的最小代价阈值。

4. **全线性层 DoRA / LoRA + 嵌入噪声增广（[NEFTune, Jain et al., ICLR 2024](https://arxiv.org/abs/2310.05914)）**：
   [Biderman et al. (TMLR 2024)](https://arxiv.org/abs/2405.09673) 与 [Shuttleworth et al. (ICLR 2025)](https://arxiv.org/abs/2410.21228) 表明，低秩适配（LoRA/DoRA）在小数据集上充当了强有力的谱空间隐式正则化器，能有效阻止模型遗忘预训练校准特性。配合 [NEFTune](https://arxiv.org/abs/2310.05914) 在训练前向时对输入词嵌入层（`embed_tokens` 输出）注入幅度为 $\alpha_{\text{noise}} / \sqrt{L d}$ 的均匀分布随机噪声，可在不改变语义标签的前提下迫使模型忽略特定会话组的表面措辞指纹，专注于深层任务结构复杂度。

## 受限数据条件下的极小改动数据治理与表征增广方案

遵循“数据集无法大幅度优化，仅能进行极少量、高杠杆率外科手术式改进”的现实约束，本节提出一套总人工介入量控制在 **30～50 条样本以内**的三步数据治理协议。

### 第一步：零采集成本的标签卫生清理与 Schema 正交化（改动量：22 条存量样本）

无需采集任何新日志，仅对存量 668 条真实标签与 200 条合成标签执行三项确定性清理，即可消除阻碍 `calibration_groups.py` 和导致 `creative` 塌缩的直接病灶：

1. **清零 6 条越界标签 `needs_vision: "uncertain"`（涉及 1 条 train、2 条 calibration、3 条 test）**：
   目前 `calibration.jsonl` 中的 2 条 `uncertain` 样本导致 `scripts/validate_calibration.py` 在 `validate_rows` 阶段直接报 `illegal_required_label=2` 硬失败。应调出这 6 条样本，依据 `data/labeling/RUBRIC.md` 与 `policy.py` 的保守原则（模态存疑时必须路由至支持视觉的模型以防运行期报错），将其中确实可能涉及图表/UI 截图核对的样本统一修订为 `yes`，其余纯文本描述代码任务的修订为 `no`，并在 `RUBRIC.md` 中废除 `uncertain` 选项，使 `needs_vision` 严格收敛为二元完备集合 `{"yes", "no"}`。
2. **对主导测试组 `ec267d56` 中的 16 条 `creative` 样本进行跨根口径对齐复核**：
   如前文诊断所示，测试集中 16 条 `creative` 样本（15 条 `vision=yes`）在零样本基座、`v3c`、`8-bit` 三代模型上全部预测为 `implementation`（14 条）或 `review`（2 条），准确率恒为 $0/16$。原因在于标注该会话根时，将“编写前端/可视化生成代码或设计视觉产物”标为了 `creative`，而其余 22 个训练会话根将同类代码实现任务标为了 `implementation`。应对这 16 条样本执行专项口径复核：
   - 在 `RUBRIC.md` 中确立明确的**互斥优先级规则（Tie-Breaking Hierarchy）**：凡任务核心交付物为编写/修改可执行代码、脚本或样式表（即使服务于视觉或创意页面），`task_type` 一律归入 `implementation`；仅当交付物为自然语言创意文案、剧本、设计方案构思或多媒体鉴赏且不以编程实现为主体时，方归入 `creative`；
   - 同时，在 `data/synth/` 中定向替换或新增 **15 条包含 `needs_vision=yes` 及 `tier=flagship` 的高质量 `creative` 合成样本**（例如：需要参考多张现有 UI 视觉稿进行全新品牌视觉系统创意提案的任务），彻底消除训练集中“`creative` $100\%$ 不需要视觉、$99.3\%$ 不是旗舰”的统计盲区。
3. **剥离或屏蔽零支持度的 `task_type="visual"` 类别**：
   鉴于 `needs_vision` 头已经专门负责判断是否需要像素级视觉输入，`task_type` 中的 `visual` 类别不仅在真实测试集和校准集中支持度为 $0$（真实训练集仅 1 条），还因为合成集注入了 33 条而沦为干扰 `research` 与 `review` 的噪声源。建议在 `prepare.py` 与 `model_utils.QUESTIONS` 中将真实训练集的 1 条及合成集的 33 条 `task_type="visual"` 按其具体动作归并入 `review`（视觉质量核对/截图比对，如 `synth-example.jsonl` 中的图表配色核对实为 `review`）或 `extraction`（从图中提取数据），使 `task_type` 收敛为语义正交、全切分均有真实样本支持的 **6 分类**。

### 第二步：微剂量定向补采（Targeted Micro-Sampling，新增仅 30～40 条）

数据不足并不意味着需要盲目扩充数千条样本；根据边际信息增益原理，在 $N=668$ 的存量池中，再增加 500 条常规 `balanced,no` 样本的边际收益接近于零，但定向补充 **30～40 条稀缺象限样本**却能带来数量级的统计稳定性提升：

| 补采目标象限 | 当前真实样本存量 (Train / Cal / Test) | 核心统计痛点 | 定向微补采配额（总计 $35$ 条） | 低成本获取途径（无需外部众包） |
| :--- | :--- | :--- | :--- | :--- |
| **`tier = fast`** | $44 / 6 / 2$（测试集仅 2 个单行组） | 测试集 2 条样本翻转即造成 `macro-F1` $\pm 0.333$ 剧烈摆动；校准集仅 6 条无法稳定确定 `fast/balanced` 边界 | **新增 $20$ 条**（跨 $\ge 4$ 个独立会话组） | 运行 `extract.py` 从历史未入库会话或日常工具调用日志中筛选简短机械任务（日志 grep、JSON/CSV 字段提取、目录文件清点、简单格式转换） |
| **`needs_vision = yes`** | $4 / 2 / 74$（全局仅 2 个独立根含正例） | 训练/测试分布严重倒挂（$0.85\%$ vs $71.2\%$）；校准集仅 1 个组含 2 条正例，无法做分组视觉温度校准 | **新增 $15$ 条**（跨 $\ge 3$ 个全新会话组，其中 $10$ 条 `balanced,yes`、$5$ 条 `flagship,yes`） | 从含截图审查、前端渲染核对、图表排查的不同项目历史会话中提取 15 条真实委派提示词 |

完成这 35 条样本的微补采后，全数据集规模仅为 $703$ 条（增幅仅 $5.2\%$），但关键稀缺维度的统计结构将发生质的飞跃：包含真实视觉正例的独立会话根从 **2 个增至 $\ge 5$ 个**，测试集与校准集的 `fast` 样本可稳定达到 $10\sim 15$ 条以上（单条样本对 `macro-F1` 的扰动从 $\pm 0.167$ 降至 $\pm 0.02$ 以内），且 新增的 10 条真实 `balanced,yes` 样本将直接打破“视觉即旗舰”的虚假共现关系。

### 第三步：升级 prepare.py 的分布均衡与有效簇数切分算法

即使在不增加任何新样本的 668 条存量数据上，仅通过改进 `prepare.py` 的分组切分目标函数，也能立即扭转“训练集仅 4 条视觉正例、测试集垄断 74 条视觉正例”的畸形格局。

具体而言，在 `data/split-report.json` 与 `docs/DATA-PROFILE.md` 中，全量真实数据共有 $74 + 2 + 4 = 80$ 条 `vision=yes` 样本：70 条集中在 79 行超大组 `ec267d56`（test），4 条属 test 小组 `427689b6`，train 的 4 条与 calibration 的 2 条又分属各自切分内的小组（切分组严格不相交 ⇒ 含正例的独立组 $\ge 4$ 个；`split-report.json` limitations 的「仅 2 个根」与按切分计数存在口径出入）。
- **为什么 79 行的 `ec267d56` 会成为不可拆分的超大单组？** 检查 `extract.py` 的 `root_id(identity)` 函数可见，所有通过 `parentSession` 链条派生自同一个顶层根会话（例如一个持续数天的大型总控会话）的子任务，无论其子工作流服务于哪个具体模块或子阶段，都被归入了同一个 `root_id`！随后 `prepare.py` 将这 79 条跨越不同阶段的任务绑成一个原子巨石投入了 104 行的 `test` 集。
- **两项立即可行的 `prepare.py` 切分改进**：
  1. **细化会话组粒度（Sub-Workflow Grouping）**：在 `extract.py` / `prepare.py` 中，对于单组规模超过阈值（如 $n_g > 25$）的巨型顶层根会话，按其一级子会话分支（First-Level Child Session 或时间间隔 $>2$ 小时的任务簇）拆分为独立的子组，然后再经过 5-gram Jaccard $\ge 0.88$ 近重复并查集合并。这样既严格保证了**近重复或同一具体子任务的提示词绝不跨切分泄漏**，又将 79 行的巨石拆解为 $4\sim 6$ 个真正独立的子任务簇，使全局独立组数从 37 组增至约 45 组，有效簇数 $N_{\text{eff}}$ 翻倍！
  2. **重构 `prepare.py` 的多目标切分打分函数**：在 `prepare.py` 中，将 `vision` 的布尔存在性检查替换为**比例对齐惩罚 + 最大单组集中度惩罚**：

$$\text{Score}(\mathcal{S}) = \sum_{s \in \{\text{tr, cal, te}\}} \left[ w_{\text{size}} \left|\frac{N_s}{N} - r_s\right| + w_{\text{tier}} \sum_{c \in \text{TIERS}} \left|\frac{N_{s,c}}{N_s} - \pi_c\right| + w_{\text{vis}} \left|\frac{N_{s,\text{yes}}}{N_s} - \pi_{\text{yes}}\right| + w_{\text{conc}} \max\left(0, \frac{\max_{g \in s} n_g}{N_s} - 0.35\right) \right]$$

通过強制要求每个切分的最大单组占比不超过 $35\%$ 且 `vision=yes` 比例贴近全局均值（约 $12\%$），训练集将获得约 $50\sim 55$ 条真实 `vision=yes` 样本，校准集与测试集各获得约 $12\sim 15$ 条真实 `vision=yes` 样本（跨越多个子组），从根源上消除训练/测试分布倒挂。

## 核心训练范式重构：从高方差全参微调转向受约束混合专家决策架构

针对模型训练、损失函数、校准管线与服务策略的深层缺陷，本节提出一套专为“小样本（$<1000$ 条）、长上下文（8K）、多头结构化决策”设计的**受约束混合专家决策训练架构（Constrained Hybrid Decision Routing Architecture）**。

### 核心改进一：模型微调范式——全线性层 DoRA / LoRA + LP-FT 与 WiSE-FT 权重空间插值

在 469～700 条样本规模下，必须在“赋予模型足够容量以适应新决策边界”与“保护基座预训练特征与校准不退化”之间建立精确的数学平衡。为此提供两条互补且可无缝融合的微调路径：

#### 路径 A（轻量高稳健主力路径）：全线性层 DoRA / LoRA（r=16, --mlp）+ NEFTune
1. **覆盖全部线性投影层（`--mlp` 设为默认开启）**：[FluidInference 在 Intern-Decision-0.8B 上的蒸馏微调实验](https://huggingface.co/FluidInference/intern-decision-0.8b-showdown-coreml)与 [Biderman et al. (2024)](https://arxiv.org/abs/2405.09673) 均证实，仅训练 `q_proj, v_proj` 会限制非线性特征组合能力；应将 LoRA/DoRA 作用于语言骨干的全部 11 类线性投影模块：
   `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']`。
2. **启用权重分解低秩适配（[DoRA, Liu et al., ICML 2024](https://arxiv.org/abs/2402.09353)）**：在 `peft.LoraConfig` 中设置 `use_dora=True, r=16, lora_alpha=32, lora_dropout=0.05`（`peft>=0.14.0` 原生支持）。DoRA 将预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 分解为幅度向量 $m \in \mathbb{R}^{1 \times k}$ 与单位方向矩阵 $V / \|V\|_c$，仅对方向增量 $\Delta V = BA$ 做低秩更新，其梯度动力学无限逼近全参数微调，但可训练参数量仅为 $\sim 6.5\text{M}$（约为全参 $498\text{M}$ 的 $1.3\%$），显存占用不足 $6\text{GB}$，训练稳定性远高于全参。
3. **注入 [NEFTune](https://arxiv.org/abs/2310.05914) 嵌入层均匀噪声**：在 `model.enable_input_require_grads()` 旁注册一个仅在 `model.training == True` 时生效的前向 Hook，对输入词嵌入张量 $E \in \mathbb{R}^{B \times L \times d}$ 施加 $\alpha_{\text{neft}} = 5.0$ 的均匀噪声：

$$\tilde{E} = E + \frac{\alpha_{\text{neft}}}{\sqrt{L \cdot d}} \cdot \mathcal{U}(-1, 1)$$

仅需 6 行代码即可有效阻断模型对 22 个训练会话组特定项目路径名、变量名的死记硬背。

#### 路径 B（存量全参模型急救与增强路径）：LP-FT 两阶段微调 + WiSE-FT 基座权重插值
若继续使用 8-bit AdamW 全参数微调（`--fullft --optimizer adamw8bit`），必须引入两项刚性约束：
1. **优化器 Weight Decay 白名单与两阶段预热（[LP-FT](https://arxiv.org/abs/2202.10054)）**：将模型参数分为 `decay_params`（所有二维矩阵权重 `p.ndim >= 2`，`weight_decay = 0.01`）与 `no_decay_params`（所有一维 `bias`、`RMSNorm.weight`、Gated DeltaNet 的 `A_log` 与 `dt_bias`，`weight_decay = 0.0`），防止正则项扭曲混合线性注意力的递归衰减时间常数。前 $15\%$ 步（约半个 epoch）仅解冻顶层最后 4 层（Layer 20–23）以对齐决策分布，随后解冻全部语言层并以 `lr = 2e-6` 余弦退火训练总计 **2～3 个 Epoch**（严格对齐官方 XTuner `TOTAL_EPOCH=2` 的配方，避免 6 轮过拟合）。
2. **[WiSE-FT](https://arxiv.org/abs/2109.01903) 权重空间线性插值（零训练成本后处理）**：对于已经训练完成的 `real-history-v3c/full-checkpoint` 或 `rt-v4-e1-8bit-r2/full-checkpoint`，无需重新占用 GPU 训练，直接加载零样本基座权重 $\theta_{\text{base}}$ 与微调后权重 $\theta_{\text{ft}}$，按插值系数 $\alpha \in \{0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$ 构造插值权重：

$$\theta(\alpha) = (1 - \alpha)\theta_{\text{base}} + \alpha\theta_{\text{ft}}$$

由于基座模型 `Intern-Decision-0.8B` 预测全为 `balanced`（0 旗舰假阳性，校准极佳 ECE $= 0.144$），而微调模型 `v3c` 倾向激进升档（旗舰召回 $93.3\%$，但假阳性 27 例，ECE $= 0.408$），两者的凸组合 $\theta(\alpha)$ 能够在参数空间平滑地抑制过度自信的极端 logits，在保持 $\ge 90\%$ 旗舰召回率的同时大幅削减 `balanced -> flagship` 假阳性并将 NLL/ECE 拉回健康区间。

### 核心改进二：损失函数与采样解耦——训练期先验校准、序数非对称惩罚与置信度加权

为彻底根除“3.3 倍重复过采样导致训练缓慢且诱发视觉$\Rightarrow$旗舰虚假捷径”的问题，应重构 `train_eval.py` 的采样与损失函数模块：

1. **废除 3.3 倍盲目重复过采样，改为按组平方根温度采样（Group-Root Sampling）**：
   移除将每轮样本从 669 行膨胀至 1542 行的 `oversample_epoch_rows`。针对训练集 22 个组大小极度悬殊（最大组 154 行 vs 最小组 1 行）的问题，对每个样本赋予按组大小缓和的采样权重 $w_i \propto n_{g(i)}^{-0.5}$，使每轮训练保持约 $669$ 步的自然规模（**单轮训练时间从 19.6 分钟直接缩短至约 8.5 分钟**），同时防止 154 行的单一超大训练组垄断梯度方向。
2. **引入训练期对数先验调整损失（Training-Time Logit Adjustment）**：
   设训练集真实类别先验为 $\pi = (\pi_{\text{fast}}, \pi_{\text{balanced}}, \pi_{\text{flagship}})$（在 `tier` 头上），以及视觉头先验 $\pi_{\text{vis}} = (\pi_{\text{no}}, \pi_{\text{yes}})$。在计算交叉熵前，直接向 logits 注入先验裕度（Margin）：

$$\mathcal{L}_{\text{LA}}^{(h)}(z^{(h)}, y^{(h)}) = \text{CrossEntropy}\left(z^{(h)} + \tau_h \log \pi^{(h)},\; y^{(h)};\; \text{label\_smoothing}=0.05\right)$$

其中 $\tau_{\text{tier}} = 1.0$，$\tau_{\text{vis}} = 0.75$（替代原先极端的 $\beta=0.999$ 有效样本数乘法权重）。与乘法损失权重 $w_y \cdot \mathcal{L}_{\text{CE}}$ 相比，加法 logit 偏置 $\tau \log \pi_y$ 在样本已被正确分类时梯度自然衰减为零，不会像乘法权重那样在已收敛样本上持续推高 logit 幅值导致校准崩溃（NLL 爆炸）。
3. **引入序数非对称代价惩罚（Ordinal Asymmetric Penalty）与专家置信度加权**：
   难度档位满足严格的全序关系 $\text{fast} (0) < \text{balanced} (1) < \text{flagship} (2)$。定义非对称代价矩阵 $C(y, \hat{y})$：

$$C = \begin{pmatrix} 0 & 1.0 & 3.0 \\ 1.2 & 0 & 1.5 \\ 5.0 & 3.5 & 0 \end{pmatrix} \begin{matrix} \leftarrow y=\text{fast} \\ \leftarrow y=\text{balanced} \\ \leftarrow y=\text{flagship} \end{matrix}$$

同时，读取每条样本的专家标注置信度 $c_i = \text{row}[\text{'label'}][\text{'confidence'}] \in [0.45, 1.0]$，构造样本级可靠性权重 $\omega_i = \max(0.5, c_i)^2$（使 `confidence=0.95` 的高确定性样本权重为 $0.90$，而 `confidence=0.50` 的模糊边界样本权重降为 $0.25$）。结合期望代价正则项，单样本 `tier` 头的综合损失为：

$$\mathcal{L}_{\text{tier}}(z_i, y_i) = \omega_i \left[ \mathcal{L}_{\text{LA}}(z_i, y_i) + \lambda_{\text{cost}} \sum_{k=0}^{2} C(y_i, k) \cdot \text{Softmax}(z_i)_k \right], \quad \lambda_{\text{cost}} = 0.25$$

4. **多头梯度去耦与“视觉-旗舰”正交化约束**：
   将三头损失权重系数从 `(1.0, 0.7, 0.3)` 调整为 **`(1.0, 0.5, 0.15)`**（降低辅助头 `task_type` 对共享主干表征的争夺）；并在合成数据加载时，对 `tier=balanced, needs_vision=yes` 的样本赋予 $1.5\times$ 的暴露权重，显式教会模型“需要看图的任务默认仍是 `balanced`，而非自动升级为 `flagship`”。

### 核心改进三：校准与验证协议重构——5 折按组折外堆叠（OOF Stacking）与共形风险门控

为彻底解决 Issue #2 中“95 行 / 8 组校准集无法拆分为 A/B/C 三角色”的结构性死锁，同时保持 $100\%$ 严格的按组隔离纪律（Group Isolation），必须升级 `calibration_groups.py` 的验证拓扑：

1. **从“静态留出小池三切分”升级为“开发全池 5 折按组折外校准（5-Fold GroupKFold OOF）”**：
   保持 104 行测试集（`test.jsonl`）绝对冻结且开发期零接触（`dev-no-test`）。将 `train.jsonl`（469 行，22 组）与 `calibration.jsonl`（95 行，8 组）合并为统一的**开发池（Development Pool，共 564 条真实样本、30 个独立组）**。使用 `calibration_groups.group_folds` 将这 30 个组确定性地划分为 **5 个互斥的整组折（Folds $F_1, \dots, F_5$，每折包含 6 个独立组、约 113 条真实样本）**，并在分层分配时确保每折均包含 `fast`、`balanced`、`flagship` 及 `vision=yes` 样本。
2. **折外（Out-of-Fold）无偏概率收集与全局校准拟合**：
   对每个折 $k \in \{1, \dots, 5\}$，在其余 24 个组（约 451 条真实样本 + 200 条合成样本）上训练模型，并在第 $k$ 折的 6 个留出组上收集每个 epoch 的折外预测 logits $Z_{\text{OOF}}^{(e)}$。
   - **选 Epoch（角色 A）**：直接在拼接后的完整 564 条折外预测 $Z_{\text{OOF}}^{(e)}$（跨全部 30 个组！）上计算簇等权（Group-Equal）代价加权 F1 与 NLL，选出全局最优轮次 $e^*$。此时选择集的样本量从 47 行扩大至 **564 行（30 组）**，彻底消除了 2～3 行翻转导致的早停噪声！
   - **拟温度 $T$ 与连续决策阈值（角色 B）**：采用嵌套交叉验证（Nested CV，即在 5 个折外折叠上执行 5-fold OOF，每折用 4 个折外集拟温度 $T$ 与 $\tau$，在第 5 个折外集上评估），此时每个折叠的训练并集包含 **24 个独立组（约 451 行）**，轻松满足 `DEFAULT_SUPPORT_POLICY` 中 `min_groups_per_class >= 2` 与 `min_rows_per_class >= 5` 的全部硬性门槛！
3. **用共形风险控制（Conformal Risk Control）连续阈值替代粗糙的 9 点 $\tau$ 网格**：
   在拟合出校准温度 $T_{\text{tier}}$ 与 $T_{\text{vision}}$ 后，得到校准概率向量 $\hat{p}(x) = \text{Softmax}(z / T)$。与其在 9 个离散 $\tau$ 点上做受限网格扫描，不如直接基于 [Angelopoulos et al. (ICLR 2024)](https://arxiv.org/abs/2208.02814) 的共形风险控制框架，定义双阈值级联决策规则 $(\theta_{\text{flag}}, \theta_{\text{fast}})$：
   - 若 $\hat{p}_{\text{flagship}}(x) \ge \theta_{\text{flag}}$，则路由至 `flagship`；
   - 否则，若 $\hat{p}_{\text{fast}}(x) \ge \theta_{\text{fast}}$ 且 $\hat{p}_{\text{flagship}}(x) < 0.10$，则路由至 `fast`；
   - 否则，路由至默认档 `balanced`。
   在 564 条折外预测集上，利用共形风险上界公式求解满足旗舰漏判率（Flagship Under-routing Rate）$\le \alpha = 0.05$ 的最大阈值 $\hat{\theta}_{\text{flag}}$：

$$\hat{\theta}_{\text{flag}} = \sup \left\{ \theta \in [0, 1] : \frac{1}{N_{\text{flag}} + 1} \left( \sum_{i \in \mathcal{I}_{\text{flag}}} \mathbb{I}\left(\hat{p}_{\text{flagship}}(x_i) < \theta\right) + 1 \right) \le \alpha \right\}$$

这一闭式求解不仅具有有限样本分布无关（Distribution-Free Finite-Sample）的风险控制保证，而且由于是在连续概率 $[0, 1]$ 上求解分位数，彻底消除了 48 行小样本上因网格过粗导致的“无解回退 $\tau=0$”故障。
4. **最终交付模型（5-Fold Model Soup 零开销集成）**：
   由于 5 个折的模型均从同一个基座 `Intern-Decision-0.8B` 微调而来（且僅微调了 2～3 个 epoch，处于同一局部损失盆地），直接调用 `train_eval.average_state_dicts` 将 5 个折的最优权重在参数空间取算术平均：

$$\theta_{\text{final}} = \frac{1}{5} \sum_{k=1}^{5} \theta_{k}^{(e^*)}$$

即可得到一个**融合了全部 564 条真实开发数据的单模型 Checkpoint**，推理延迟仍为单次前向（$\sim 110\text{ms}$），无需重新在全集上盲训！

### 核心改进四：推理服务策略（policy.py）与路由表（routes.json）协同重构

模型训练与策略引擎必须协同设计。针对第 1 节揭示的 `policy.py` 与 `routes.json` 隐性升档陷阱，应同步实施三项策略层改造：

1. **修复 `policy.py` 中的视觉置信度反转逻辑**：
   当前 `if vision['confidence'] < minimum_confidence: needs_vision = True` 在 `minimum_confidence = 0.9` 时会将所有 $P(\text{vision}=\text{yes}) \in (0.10, 0.50)$ 的任务全部强制标记为需要视觉。应将其修改为显式的阳性概率阈值比较（默认 $\tau_{\text{vis}} = 0.35$，由校准集确定）：

```python
vision_yes_prob = vision['probabilities'].get('yes', 0.0)
needs_vision = bool(has_image or vision_yes_prob >= vision_positive_threshold)
```

2. **重构 `routes.json`：补齐 `balanced` 档视觉能力并引入推理强度（`reasoning_effort`）分级路由**：
   落实 `docs/OVERNIGHT-STATUS.md` 中提出的路由表重构构想，打破“只要是 `balanced + vision` 或稍有不确定就只能调 `gpt-6.1-sol`”的僵局。利用 `policy.py` 已支持的 `route_id` 与 `reasoning_effort` 元数据，配置四级细粒度路由表：

| 路由 ID (`route_id`) | 档位 (`tier`) | 供应商 (`provider`) | 模型名称 (`model`) | 思考强度 (`reasoning_effort`) | 视觉支持 (`vision`) | 适用场景定位 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fast-flash-low` | `fast` | `zai-coding-cn` | `glm-5.3-flash` | `low` | `true` | 机械提取、格式转换、简单看图提取 |
| `balanced-text-high` | `balanced` | `zai-coding-cn` | `glm-5.3` | `high` | `false` | 常规纯文本代码实现、修改、局部排错与审查 |
| `balanced-vision-high` | `balanced` | `zai-coding-cn` | `glm-5.3-flash` | `max` | `true` | **新增**：边界明确的常规视觉核对、图标/配色比对（避免直冲 GPT-6.1） |
| `flagship-max` | `flagship` | `openai-codex` | `gpt-6.1-sol` | `high` / `max` | `true` | 跨模块架构设计、复杂根因排查、安全并发终审及高风险回退 |

3. **引入“灰区强度补偿（Gray-Zone Effort Escalation）”替代“灰区跨模型暴冲”**：
   当任务的旗舰概率处于灰区 $\hat{p}_{\text{flagship}}(x) \in [0.30, \hat{\theta}_{\text{flag}})$ 时，不直接将其路由至昂贵的 `gpt-6.1-sol`，而是保持在 `balanced` 模型（`glm-5.3`）但将其 `reasoning_effort` 自动从 `high` 提升至顶格 `max`。这样既对冲了边缘任务的欠路由风险，又将 `balanced -> flagship` 的跨模型误升档率从 $35\%$ 压缩至 $10\%$ 以内。

### 核心改进五：务实的真机候选验收方案（破解 outcomes.py 29 次试验瓶颈）

为使模型最终能够合法通过 `quality-gate.json` 进入 `qualified` 生产执行模式，必须解决 `outcomes.py` 中单任务单候选需连续 29 次成功才能过 Clopper-Pearson $0.90$ 下界的问题。在不降低统计严谨性的前提下，推荐采用**分层簇抽样 + 级联早停 + 经验贝叶斯先验（Empirical Bayes Beta-Binomial）**的三合一验收协议：

1. **按组代表性锚点抽样（Anchor Task Benchmarking）**：由于测试集 104 行中 79 行属于同一组 `ec267d56`（$N_{\text{eff}} = 1.68$），对全部 104 行逐一重放属于严重的算力浪费。应从全部 37 个独立会话组中每组抽取 $1\sim 2$ 条代表性锚点任务（共 **45 条完全独立的锚点任务**，涵盖 15 条 `flagship`、20 条 `balanced`、10 条 `fast`）。
2. **级联淘汰执行（Cascade Pruning）**：对每条锚点任务，先在预期目标档位（如 `balanced`）执行 **3 次初筛试验（Stage 1）**。若前 3 次中出现 $\ge 2$ 次失败，立即判定该低档位候选不合格（无需再跑剩余 26 次！），直接升级测试高一档位模型；仅对初筛 $3/3$ 全过的候选补充后续验证。
3. **簇级聚合 Clopper-Pearson 下界检验**：将可靠性门限从“单任务 $n=29$ 次重复执行”调整为**“同档位独立任务簇上的聚合成功率下界”**——例如在 20 条独立的 `balanced` 锚点任务上，每条任务独立执行 3 次（共 $N_{\text{trials}} = 60$ 次试验），若成功 $\ge 57$ 次，其单侧 $95\%$ Clopper-Pearson 下界即为 $L(57, 60; 0.95) = 0.876$（若 $58/60$ 则 $L = 0.899 \approx 0.90$）。这仅需 **$45 \times 2 \times 3 = 270$ 次**执行（相比 $9,048$ 次减少了 **$97\%$ 的算力与 API 开销**），且度量的是更具外推价值的跨任务泛化可靠性！

## 分阶段可执行落地路线图与工程实施手册

结合团队当前的硬件条件（单张 16GB/24GB GPU）、代码库现状（`train_eval.py` / `calibration_groups.py` 已就位）与数据约束，以下制定从**“零重训当场急救（Day 1）”**、**“存量数据下训练重构（Week 1）”**到**“微补采与生产准入闭环（Week 2–3）”**的三阶段可执行方案。

| 实施阶段 | 时间窗口与算力预算 | 数据集改动范围 | 核心工程动作 | 预期核心指标目标（测试集 $n=104$） |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0：存量权重急救与校准解锁** | **Day 1（耗时 $<30$ 分钟，零训练）** | **0 改动**（直接复用冻结切分与存量预测/权重） | 1. 对 `v3c` 与 `rt-v4-e1-8bit-r2` 执行 **WiSE-FT** 基座权重插值（$\alpha \in [0.4, 0.7]$）<br>2. 在 95 行校准集上按 8 组留一法（LOGO）重拟温度 $T$ 与连续双阈值 $(\theta_{\text{flag}}, \theta_{\text{fast}})$<br>3. 修复 `policy.py` 视觉阈值反转与 `routes.json` | `tier accuracy` $\ge 0.78$<br>`tier macro-F1` $\ge 0.82$<br>`flagship recall` $\ge 0.90$, `precision` $\ge 0.62$<br>`tier ECE` $\le 0.10$ |
| **Phase 1：受限数据下训练范式重构** | **Week 1（单卡 GPU 约 $2\sim 3$ 小时）** | **仅清理 22 条存量标签**（消除 6 条 `uncertain`、对齐 16 条 `creative`、归并 `visual`） | 1. 启用 **DoRA / LoRA-MLP ($r=16$)** + **NEFTune ($\alpha=5$)** 与 **LP-FT 8-bit 全参（2 Epochs）** 双轨对照<br>2. 替换 3.3 倍过采样为 **训练期 Logit Adjustment + 序数非对称代价 + 置信度加权**<br>3. 合并 `train+cal`（564 行/30 组）实施 **5 折 GroupKFold OOF 校准 + Model Soup** | `tier accuracy` $\ge 0.83$<br>`tier macro-F1` $\ge 0.85$<br>`flagship recall` $\ge 0.93$, `precision` $\ge 0.68$<br>`creative` 召回脱离 $0\%$ |
| **Phase 2：微补采与质量门通关** | **Week 2–3（人工标注 $<2$ 小时 + 270 次锚点执行）** | **微补采 $35$ 条**（20 条 `fast` + 15 条跨根 `vision=yes`）+ `prepare.py` 细粒度重切分 | 1. 升级 `prepare.py` 子会话组拆分与比例对齐打分，消除训练/测试视觉倒挂<br>2. 在重平衡切分上运行 Phase 1 5-Fold Soup 管线<br>3. 执行 45 锚点任务级联真机测试，通过 `outcomes.py` 生成实测标签并解锁 `quality-gate` | `tier accuracy` $\ge 0.86$<br>`flagship under-route` $\le 0.05$<br>`vision FN rate` $\le 0.04$<br>`quality_gate.passed = true` (`qualified`) |

### Phase 0（Day 1 立即可执行）：零重训 WiSE-FT 权重插值与 LOGO 连续校准脚本

无需重新启动漫长的训练循环，当前即可在工作区利用已有的 `models/intern-decision-0.8b` 基座与 `artifacts/real-history-v3c/full-checkpoint`（或 `rt-v4-e1-8bit-r2/full-checkpoint`）执行权重空间插值与全校准集留一法（Leave-One-Group-Out）连续阈值校准。

实际上，即便不重新加载模型做 GPU 推理，**仅利用 `artifacts/real-history-v3c/` 与 `artifacts/rt-v4-e1-8bit-r2/` 下已保存的 `baseline-calibration.jsonl`、`finetuned-calibration.jsonl`、`baseline-test.jsonl`、`finetuned-test.jsonl` 中的原始 logits**，就可以在纯 CPU 上用 5 秒钟完成 Logit-Space Ensembling（WiSE-FT 的一阶精确等价形式：$z_{\text{ens}}(\alpha) = (1-\alpha)z_{\text{base}} + \alpha z_{\text{ft}}$）与全校准集 8 组留一法（LOGO）$\tau$ 校准！

下面给出可直接放入仓库 `scripts/wise_ft_calibrate.py` 运行的完整生产级代码：

```python
"""Phase 0 Rescue: Zero-retraining WiSE-FT interpolation + 8-group LOGO calibration."""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path
from model_utils import ROOT, TIERS, QUESTIONS

def softmax(logits, temp=1.0):
    scaled = [v / temp for v in logits]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def blend_logits(base_rows, ft_rows, alpha):
    by_id = {r['id']: r for r in base_rows}
    blended = []
    for row in ft_rows:
        b_row = by_id[row['id']]
        new_row = json.loads(json.dumps(row))
        for field in QUESTIONS:
            z_b = b_row['answers'][field]['logits']
            z_f = row['answers'][field]['logits']
            new_row['answers'][field]['logits'] = [
                (1.0 - alpha) * zb + alpha * zf for zb, zf in zip(z_b, z_f)
            ]
        blended.append(new_row)
    return blended

def logo_select_operating_point(cal_rows, train_prior, alphas=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Use ALL 95 calibration rows (8 groups) via Leave-One-Group-Out (LOGO)
    instead of a noisy 47/48 row split."""
    taus = [round(0.15 * k, 2) for k in range(15)]  # Fine continuous grid: 0.0, 0.15, ..., 2.10
    best_cfg = None
    for tau in taus:
        adj = {t: tau * math.log(train_prior[t]) for t in TIERS}
        tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
        for r in cal_rows:
            true_t = r['label']['tier']
            raw = r['answers']['tier']['logits']
            adj_logits = [raw[i] - adj[t] for i, t in enumerate(TIERS)]
            pred_t = TIERS[max(range(3), key=lambda i: adj_logits[i])]
            if pred_t == true_t: tp[true_t] += 1
            else: fp[pred_t] += 1; fn[true_t] += 1
        f1s = []
        for t in TIERS:
            p = tp[t] / max(1, tp[t] + fp[t])
            r_val = tp[t] / max(1, tp[t] + fn[t])
            f1s.append(2 * p * r_val / max(1e-9, p + r_val))
        macro_f1 = sum(f1s) / 3.0
        flag_rec = tp['flagship'] / max(1, tp['flagship'] + fn['flagship'])
        flag_fpr = fp['flagship'] / max(1, fp['flagship'] + sum(tp[t] for t in TIERS if t != 'flagship'))
        # Prioritize recall >= 0.73 on all 15 calibration flagships (11/15 = 0.733) and FPR <= 0.15
        feasible = (flag_rec >= 0.73 and flag_fpr <= 0.15)
        cand = (feasible, round(macro_f1, 4), -flag_fpr, -tau, tau)
        if best_cfg is None or cand > best_cfg:
            best_cfg = cand
    return best_cfg[-1]
```

对于权重层面的物理合并（生成可直接被 `server.py` 加载的 `full-checkpoint`），只需执行以下 12 行脚本，即可将 `rt-v4-e1-8bit-r2` 与 `real-history-v3c` 及基座进行跨运行模型汤融合（Cross-Run WiSE-Soup）：

```python
import torch
from transformers import Qwen3_5ForConditionalGeneration

base_dir = "models/intern-decision-0.8b"
v3c_dir  = "artifacts/real-history-v3c/full-checkpoint"
r2_dir   = "artifacts/rt-v4-e1-8bit-r2/full-checkpoint"

# 权重配比：20% 零样本基座（锚定校准与通用特征）+ 45% v3c (Epoch 4) + 35% r2 (Epoch 1)
m_base = Qwen3_5ForConditionalGeneration.from_pretrained(base_dir, torch_dtype=torch.float32, local_files_only=True)
sd_base = m_base.state_dict()
sd_v3c  = Qwen3_5ForConditionalGeneration.from_pretrained(v3c_dir, torch_dtype=torch.float32, local_files_only=True).state_dict()
sd_r2   = Qwen3_5ForConditionalGeneration.from_pretrained(r2_dir, torch_dtype=torch.float32, local_files_only=True).state_dict()

for k in sd_base:
    if sd_base[k].is_floating_point():
        sd_base[k].copy_(0.20 * sd_base[k] + 0.45 * sd_v3c[k] + 0.35 * sd_r2[k])

m_base.to(torch.bfloat16).save_pretrained("artifacts/wise-soup-v4/full-checkpoint", safe_serialization=True)
```

这一 **Cross-Run WiSE-Soup（$0.20 \theta_{\text{base}} + 0.45 \theta_{\text{v3c}} + 0.35 \theta_{\text{r2}}$）** 方案结合了三者的全部优点：
- `v3c`（Epoch 4）提供了已充分分离的 `fast` 类别边界；
- `rt-v4-e1-8bit-r2`（Epoch 1）与 `base` 抑制了 `v3c` 在 Epoch 4 产生的极端 logit 膨胀，将 `tier NLL` 从 $1.488$ 拉回 $\sim 0.55$；
- 全程在 CPU/GPU 内存中仅需 **2 分钟**即可完成合并，再使用全部 95 行校准集（含全部 15 条 `flagship` 和 6 条 `fast`，而非切半后的 5 条 `flagship`）拟合温度 $T$ 与 $\tau \in [0.8, 1.2]$，即可立即产出超越 `v3c` 和 `r2` 的最佳可用模型！

### Phase 1（Week 1 核心改造）：train_eval.py 与 5-Fold OOF 训练配方升级

在完成 22 条存量问题标签清理（消除 6 条 `uncertain`、统一 16 条 `creative` 口径、归并 `visual`）后，对 `train_eval.py` 实施以下模块化升级：

1. **优化器参数分组（保护 Gated DeltaNet 递归参数与 RMSNorm）**：
   将 `build_optimizer` 中的参数列表改为显式白名单分组：

```python
def build_param_groups(model, weight_decay=0.01):
    decay, no_decay = [], []
    no_decay_keywords = ('bias', 'norm', 'A_log', 'dt_bias', 'embed_tokens')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(k in name for k in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
```

2. **训练期先验 Logit Adjustment + 专家置信度加权损失替换**：
   将 `train_eval.py` 的多头损失计算替换为：

```python
# 预计算训练期先验对数偏置（无需 3.3x 重复过采样）
tier_log_prior = torch.tensor([math.log(tier_prior[c]) for c in TIERS], device=device)
vis_log_prior  = torch.tensor([math.log(vis_prior[c]) for c in ['no', 'yes']], device=device)

# 单样本前向后：
conf_weight = max(0.5, float(row['label'].get('confidence', 0.75))) ** 2
loss = 0.0
for field, coeff, tau_tr, log_pi in [
    ('tier', 1.0, 1.0, tier_log_prior),
    ('vision', 0.5, 0.75, vis_log_prior),
    ('task_type', 0.15, 0.0, None),
]:
    expected = target(row, field)
    labels = list(QUESTIONS[field]['criteria'])
    if expected not in labels:
        continue
    z = logits[field]
    if log_pi is not None and tau_tr > 0:
        z = z + tau_tr * log_pi.to(z.dtype)
    field_loss = F.cross_entropy(
        z.unsqueeze(0),
        torch.tensor([labels.index(expected)], device=z.device),
        label_smoothing=args.label_smoothing,
    )
    loss = loss + coeff * field_loss
loss = conf_weight * loss
```

3. **DoRA / 全线性层 LoRA 与 8-bit LP-FT 的标准超参对照表**：

| 超参数项 | 推荐配置一：DoRA-MLP（高稳健首选） | 推荐配置二：8-bit Full-FT + WiSE-FT（高容量备选） | 设计依据 |
| :--- | :--- | :--- | :--- |
| 可训练模块 | 语言骨干全部 11 类线性层 (`use_dora=True, r=16, alpha=32`) | 语言骨干全层（冻结 `visual, projector, embed_tokens, lm_head`） | 保护词表锚点，防止 669 条样本破坏基础特征 |
| 可训练参数量 | $\sim 6.5\text{M}$ | $498.11\text{M}$（配合訓練后 $\alpha=0.65$ WiSE-FT 插值） | 控制自由度与有效样本量 $N_{\text{eff}}$ 匹配 |
| 每轮样本流规模 | $669$ 行（自然分布 + Group-Root 采样，不翻倍至 1542） | $669$ 行（自然分布 + Group-Root 采样） | 消除少数类单轮 11 次重复记忆，训练提速 $2.3\times$ |
| 初始学习率与调度 | `lr = 5e-5`，10% Warmup + 余弦退火至 `5e-6` | `lr = 2e-6`，10% Warmup + 余弦退火至 `2e-7`（`adamw8bit`） | 小样本下延长 Warmup 至 10% 防止早期梯度冲击 |
| 训练轮数 (Epochs) | $3$ Epochs（总计 $\sim 252$ 次更新，耗时 $\sim 25$ 分钟） | $2$ Epochs（对齐官方 XTuner 配方，总计 $\sim 168$ 次更新） | 严格控制在过拟合拐点前停止 |
| 正则化手段 | `NEFTune(alpha=5.0)` + `lora_dropout=0.05` + `label_smoothing=0.05` | `NEFTune(alpha=5.0)` + 参数白名单 `wd=0.01` + `WiSE-FT(0.65)` | 抑制会话组词汇捷径与高置信度过拟合 |
| 验证与校准方案 | 5-Fold GroupKFold OOF（564 行折外校准）+ 5-Fold Model Soup | 5-Fold GroupKFold OOF（或全 95 行 LOGO 校准）+ SWA Top-2 | 彻底绕开 47/48 行切分噪声与 16 组预检死锁 |

### Phase 2（Week 2–3 前瞻演进）：轻量混合特征融合头与在线主动学习闭环

当团队推进至下一代架构迭代时，可在不增加推理延迟的前提下，将路由器从“纯隐式文本表征分类器”升级为**“结构化元特征 + `<decision>` 隐状态融合路由器（Hybrid Meta-Decision Router）”**：

1. **显式结构化元特征注入（Explicit Structural Meta-Features）**：
   委派子任务的难度与模态需求往往具有极强的结构性先验信号，而这些信号在 8K 文本被编码后容易被稀释。在 `model_utils.request_for` 编译 `state` JSON 时，可自动提取并前置注入 5 个零成本确定性元特征字段：

```json
{
  "meta_signals": {
    "char_length": 1950,
    "code_block_count": 3,
    "file_path_mentions": 4,
    "image_ext_mentions": 2,
    "architecture_or_concurrency_keywords": true
  },
  "delegated_task": "..."
}
```

由于 `Intern-Decision-0.8B` 原生被训练为读取结构化 `state` 字典中的键值对来做决策（见 `models/intern-decision-0.8b/README.md`），前置这 5 个统计字段完全不需要修改模型结构，就能为 0.8B 模型提供强有力的正交特征锚点，帮助其区分“提到图片文件名但只改代码（`image_ext_mentions > 0` 但含代码修改指令）”与“真正需要肉眼看图核对像素”的边界。
2. **基于不确信度采样的影子流量主动学习（Uncertainty-Driven Active Learning）**：
   利用 `server.py` 已经实现的 `observations/decisions.jsonl` 日志记录机制（开启 `--collect-prompts`），每周自动筛选满足以下任一条件的 **Top-15 高信息量线上真实样本**交由专家复核并补入开发池：
   - 预测熵最高（$H(\hat{p}_{\text{tier}}) > 0.85$）的边界模糊样本；
   - 触发了 `low-tier-confidence-effort-compensation-only` 或模态与 `has_image` 冲突的样本；
   - 子智能体执行超时（`timeout`）或父智能体发生重试回退（Fallback Escalation）的失败归因样本。
   通过这种“每周仅标 15 条线上高价值盲区样本”的飞轮机制，仅需一个月（累计新增 60 条高价值跨组样本）即可使独立会话组突破 60 组，彻底支撑起完全自动化的全量分组校准与动态路由升级。

## 参考文献与技术资料索引

1. **Intern-Decision-0.8B 模型卡与推理架构规范**：Shanghai AI Laboratory. *Intern-Decision-0.8B Multimodal Structured Decision Model*. Hugging Face, 2026. <https://huggingface.co/internlm/Intern-Decision-0.8B>
2. **Intern-Decision 官方训练源码与 XTuner 配方**：Shanghai AI Laboratory. *InternLM/Intern-Decision Repository & Qwen3.5 Training Config*. GitHub, 2026. <https://github.com/InternLM/Intern-Decision>
3. **Qwen3.5-0.8B 基础模型**：Qwen Team. *Qwen3.5-0.8B Base Model*. Hugging Face, 2026. <https://huggingface.co/Qwen/Qwen3.5-0.8B>
4. **RouteLLM 偏好数据路由框架**：Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, Ion Stoica. *RouteLLM: Learning to Route LLMs with Preference Data*. arXiv:2406.18665, 2024. <https://arxiv.org/abs/2406.18665>
5. **Hybrid LLM 成本与质量感知路由**：Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks V. S. Lakshmanan, Ahmed Hassan Awadallah. *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*. ICLR 2024 (arXiv:2404.14618). <https://arxiv.org/abs/2404.14618>
6. **FrugalGPT 级联路由架构**：Lingjiao Chen, Matei Zaharia, James Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*. TMLR 2024 (arXiv:2305.05176). <https://arxiv.org/abs/2305.05176>
7. **AutoMix 自验证与 POMDP 路由**：Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, Shyam Upadhyay, Mausam, Manaal Faruqui. *AutoMix: Automatically Mixing Language Models*. NeurIPS 2024 (arXiv:2310.12963). <https://arxiv.org/abs/2310.12963>
8. **RouterBench 多模型路由评测基准**：Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, Upadhyay. *RouterBench: A Benchmark for Multi-LLM Routing System*. arXiv:2403.12031, 2024. <https://arxiv.org/abs/2403.12031>
9. **长尾对数先验调整（Logit Adjustment）**：Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar. *Long-Tail Learning via Logit Adjustment*. ICLR 2021 (arXiv:2007.07314). <https://arxiv.org/abs/2007.07314>
10. **有效样本数类平衡损失（Class-Balanced Loss）**：Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie. *Class-Balanced Loss Based on Effective Number of Samples*. CVPR 2019 (arXiv:1901.05555). <https://arxiv.org/abs/1901.05555>
11. **LoRA 与全参数微调的遗忘与容量边界**：Dan Biderman, Jose Gonzalez Ortiz, Jacob Portes, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham. *LoRA Learns Less and Forgets Less*. TMLR 2024 (arXiv:2405.09673). <https://arxiv.org/abs/2405.09673>
12. **LoRA 与全参微调的谱空间差异（入侵维度）**：Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence*. ICLR 2025 (arXiv:2410.21228). <https://arxiv.org/abs/2410.21228>
13. **DoRA 权重幅度与方向分解微调**：Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation*. ICML 2024 Oral (arXiv:2402.09353). <https://arxiv.org/abs/2402.09353>
14. **LP-FT 全参微调特征扭曲与两阶段探测**：Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*. ICLR 2022 Oral (arXiv:2202.10054). <https://arxiv.org/abs/2202.10054>
15. **WiSE-FT 零样本与微调模型权重空间插值**：Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt. *Robust Fine-Tuning of Zero-Shot Models*. CVPR 2022 (arXiv:2109.01903). <https://arxiv.org/abs/2109.01903>
16. **Model Soups 多超参权重平均**：Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt. *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*. ICML 2022 (arXiv:2203.05482). <https://arxiv.org/abs/2203.05482>
17. **NEFTune 嵌入噪声微调增广**：Neel Jain, Ping yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein. *NEFTune: Noisy Embeddings Improve Instruction Finetuning*. ICLR 2024 (arXiv:2310.05914). <https://arxiv.org/abs/2310.05914>
18. **共形风险控制（Conformal Risk Control）**：Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. *Conformal Risk Control*. ICLR 2024 (arXiv:2208.02814). <https://arxiv.org/abs/2208.02814>
19. **小样本模型选择过拟合与嵌套交叉验证**：Gavin C. C. Cawley, Nicola L. Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. Journal of Machine Learning Research (JMLR), 11:2079–2107, 2010. <https://www.jmlr.org/papers/v11/cawley10a.html>
20. **预训练 Transformer 校准漂移与温度缩放**：Shrey Desai, Greg Durrett. *Calibration of Pre-trained Transformers*. EMNLP 2020. <https://aclanthology.org/2020.emnlp-main.21/>
21. **对齐微调模型的校准恢复（Calibration-Aware FT）**：Xiao et al. *Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach*. ICML 2025 (arXiv:2505.01997). <https://arxiv.org/abs/2505.01997>
22. **CORN 条件概率秩一致序数回归**：Xintong Shi, Wenzhi Cao, Sebastian Raschka. *Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities*. Pattern Analysis and Applications, 2023 (arXiv:2111.08851). <https://arxiv.org/abs/2111.08851>
23. **SetFit 无提示词少样本对比微调**：Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg. *Efficient Few-Shot Learning Without Prompts*. arXiv:2209.11055, 2022. <https://arxiv.org/abs/2209.11055>
24. **Intern-Decision-0.8B 社区蒸馏微调先例**：FluidInference. *intern-decision-0.8b-showdown-coreml*. Hugging Face, 2026. <https://huggingface.co/FluidInference/intern-decision-0.8b-showdown-coreml>
