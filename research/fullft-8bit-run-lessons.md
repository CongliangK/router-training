# 8-bit AdamW 全参微调：首次真实运行的工程验证与评估实证（2026-10-14）

> **公开边界声明**：本文提炼自一次私有数据上的验证运行，只含聚合指标、
> 计数与方法论结论，不含任何原始 prompt、标签值、预测行、行/组标识或
> 机器路径。运行代码为当时 main（含 issue #14 的可复现性修复与
> `--recipe {"optimizer":"adamw8bit"}` 支持，PR #17）；配置：3 epochs、
> `--legacy` 校准路线、与归档 v3c 相同的 seed 与 669 行训练数据
> （469 真实 + 200 合成，tier 均衡过采样至 1542 行/epoch）。
> 评估基准与 v3c 相同的冻结测试集（104 行、7 个独立组）。

## 1. 工程结论：8-bit 路径端到端可用（对 README 估计的实证）

| 维度 | 实测（3 epochs, adamw8bit） | 参照（v3c, 6 epochs, fp32 AdamW） |
|---|---|---|
| 数值稳定性 | 全程 loss 有限（0.855 → 0.444 → 0.351），无 NaN/OOM | 正常 |
| 吞吐 | 1.31 行/s（19.6 min/epoch，含期中校准评估） | 1.35 行/s（19.1 min/epoch） |
| 显存 | **在空闲 11.5GiB 的 16GB 卡上完整跑通**（同时有他人进程占用 4.5GiB） | fp32 状态路径需 ~15GiB，同窗口不可行 |

- README 中 "~9.7GiB 峰值 / 省 ~2.8GiB" 的估计（PR #17）与实测上限
  （≤11.5GiB 完整跑通）相容。**8-bit 把"与他人共享 GPU 的窗口"从不可行
  变为可行**——这是该特性最主要的运营价值，吞吐代价为零（−2.4%，噪声级）。
- 基座为 Gated DeltaNet 架构（473 个权重张量）；PR #17 担心的
  "8-bit × 该架构未实测"风险未出现。`min_8bit_size=16384` 默认下小张量
  走 fp32 路径，进一步缩小暴露面。
- **最便宜的早期验证点是首个 optimizer update 后 loss 有限**；过了这一步，
  本次再未见任何优化器侧异常。

### bitsandbytes 集成 gotcha（下次直接复用）

1. **GPU 可见时 `import bitsandbytes` 本身就会初始化 CUDA context**
   （实测 `torch.cuda.is_initialized()==True`，数百 MiB 量级）。监督器、
   预检脚本不要提前 import bnb；运行进程内无碍（本就要初始化）。
   CPU-only 环境（`CUDA_VISIBLE_DEVICES=''`）import 安全。
2. `pip install --no-deps` 即可（torch/numpy/packaging 均已满足），
   零新增依赖，`pip check` 干净；bnb 刻意不进 `requirements.txt` 与 CI
   （无 bnb 环境测试保持全绿）。

## 2. 评估方法论实证：红队预测 → 运行证据

`research/fullft-redteam.md` 记录的若干风险，本次运行提供了量化实例。

### 2.1 macro-F1 被稀有类噪声主导（issue #3 的 Exhibit）

同测试集上两种配方（6ep fp32 vs 3ep 8-bit）的 tier accuracy **完全相同**
（75/104 = 0.721），macro-F1 却是 0.805 vs 0.480。逐格对比混淆矩阵，
差异**恰好只有 4 行**：

- fast 类 2 行：参照全对（2/2），本次全错（均预测 balanced）→
  fast F1 1.0→0.0，macro-F1 直接被砸掉 0.33；
- balanced 类 2 行：参照过路由到 flagship，本次判对 →
  balanced F1 反而略升（0.764 vs 0.756）。

即除 fast 外两模型的 tier 行为等价甚至本次略好。**测试集 fast 仅 2 例时，
其 F1 在 0/1 间的一次摆动 = macro-F1 ±0.33**；以单一 macro-F1 选模型
等价于掷硬币。这正是 issue #3 要求"每类指标标注支持量、n<10 不参与
宏平均"的实证依据。

### 2.2 tau 操作点门在小编号集上结构性不可满足（issue #1）

legacy 拟合半区（48 行）中 flagship 仅 ~15 例：tau∈[0, 1.75] 时 sweep 的
flagship recall 恒为 0.6（召回粒度 1/15 ≈ 6.7pp，大于门的间距），FP rate
随 tau 单调升。**recall≥0.75 & FP≤0.10 在所有 tau 下无解**，放宽规则
（recall≥0.6 & FP≤0.15）只剩 tau=0 入选 → 最终不做 logit 调整。
门的设计粒度必须匹配校准集支持量，否则该门永远处于"放宽回退"状态，
失去把关意义（v3c 归档 run 亦如此，见其 quality-gate）。

### 2.3 epoch 选择被选择集噪声主导（issue #2 的同源问题）

选择半区仅 47 行：epoch 1 与 epoch 2/3 的选择 macro-F1 差 0.057 ≈
2–3 行翻转，本次 3-epoch 运行"选中"epoch 1；而 6-epoch 参照的最佳在
epoch 4（选择 macro-F1 0.647）。两点含义：

- 47 行上的 epoch 排序接近噪声，**"epoch 1 最佳"这类结论必须附带
  选择不确定度**（或按组 k-fold，呼应 #2 的组隔离要求）；
- 训练 loss 持续下降（0.855→0.351）而选择指标不升反降（0.519→0.462），
  提示校准分布与训练分布的差异在早期 epoch 后主导了选择指标。

### 2.4 稀有类在校准集中不可校准

95 行校准集中 vision 正例仅 2 例（同组）。后果：vision 温度实质未校准
（T=1.0），每 epoch 的 calibration_vision_recall 恒 0.0 无统计意义。
凡校准集支持量不足以约束某头的温度，应如实上报"未校准"而非给出
形式上的数值。

## 3. creative 塌缩根因：跨 root 标签语义漂移（issue #5 的诊断证据）

issue #5 待办中的"排查 creative 塌缩原因"，本次给出了根因定位：

| | 训练集（真实） | 测试集 |
|---|---|---|
| creative 行数 | **151/469（最大类）** | 16 |
| vision 构成 | **100% no** | **15/16 yes** |
| tier 构成 | 150 balanced + 1 fast | 集中于主导组 |

- **不是样本量或过采样问题**（它是最大真实类），也与 loss 系数无关；
- 三代模型——零样本基线、v3c（6ep fp32）、本次（3ep 8-bit）——在测试
  creative 上**全部 0/16**（14→implementation、2→review），说明
  **与训练配方无关，是数据/标注 schema 层面的问题**；
- 机制：训练分布里 "creative" 指纯文字、平衡级任务；测试的 creative
  全部来自占测试集 76% 的主导组，其中 15/16 是需要视觉的创作类任务。
  同一标签 token 在不同任务 root 间语义漂移，group-disjoint 切分把漂移
  固化为不可训练修复的分布偏移。
- **对 #5 的含义**：清单中的消融实验（tier-only / 独立头 / 系数扫描）
  不会修复此类塌缩；需要 root 级标注口径对齐（每 root codebook、跨
  root 标注一致性抽检），或对主导组的 creative 重新标注后再评估。

伴随的捷径证据（同一次预测明细的聚合计数）：25 条 balanced→flagship
过路由中 21 条 vision=yes，而测试 flagship 30 条中 28 条 vision=yes ——
tier 头在主导组内学到了"需要视觉 ⇒ 旗舰"的相关性；2 条 flagship 欠路由
则均为 creative（"创作类不可能是旗舰"盲区）。

## 4. 其他实测教训

1. **欠训练模型的概率可能更校准**：本次 epoch-1 单模 tier NLL 0.60 /
   ECE 0.11，对比参照 epoch-4 单模 NLL 1.49 / ECE 0.41（两者 accuracy
   相同）。若下游按阈值消费置信度，校准质量与准确率同等重要——
   issue #3 立场的直接实证。
2. **高置信错误常态化**：29 条 tier 错误中 17 条模型置信 ≥0.8（最高
   0.97）。置信度阈值自检无法兜底；同时错误行的专家标注置信度也偏低
   （均值 0.688 vs 正确行 0.731，全部为 provisional prompt-only 标签），
   提示相当一部分"错误"是与边缘标注的分歧。
3. **短程训练下 task_type 的 impl↔debugging/review 边界更模糊**：
   implementation 22/34（参照 31/34），新增 impl→debugging 6、
   impl→review 6；另有 research→visual 3 次（visual 类仅 1 条真实训练行，
   属无支持类的噪声预测，建议在报告中对零/一支持类直接标记而非计入混淆）。
4. **运行协议卫生（过程教训）**：
   - run.log / supervisor.log 未随产物归档，导致第一次尝试（死于训练
     中段、非窗口截止）死因不可判定——**日志是 artifacts 的一部分**，
     与 `report.json` 同级保存；
   - 协议与执行的偏离（epochs 2→3、窗口超时、校准路线选择）未留变更
     记录，只能靠文件 mtime + `report.seconds` 逆向还原时间线——
     运行 README 应有追加式变更记录，任何偏离在发生时即落笔。

## 5. 与 issue 的对应关系（本文只登记实证，不新增路线图）

| 发现 | 支撑的 issue |
|---|---|
| macro-F1 噪声主导的 4 行案例 | #3 |
| tau 门粒度 < 校准集支持量 → 结构性不可满足 | #1 |
| 47 行选择集上的 epoch 排序噪声 | #2 |
| 测试集主导组对指标的影响实证（配合 #4 的分组报告） | #4 |
| creative 塌缩根因 = 跨 root 标签语义漂移（消融不可修复） | #5 |
| 8-bit 优化器全链路验证（稳定性/吞吐/显存/gotcha） | #14（已关闭）的运行侧续证 |
