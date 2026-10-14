# 冻结数据集与运行结果聚合画像（供外部 / LLM 分析使用）

> **English abstract**: The real dataset behind this repository (668 historical
> subtask prompts + expert labels) cannot be published — even after automated
> redaction the prompts describe identifiable personal projects. This profile
> publishes **aggregate statistics only**, so that external analysts or LLMs can
> reason about the frozen dataset and the archived runs without access to raw
> data. Machine-readable run data lives in
> [`artifacts/real-history-v3c/report.json`](../artifacts/real-history-v3c/report.json)
> and [`artifacts/rt-v4-e1-8bit-r2/report.json`](../artifacts/rt-v4-e1-8bit-r2/report.json).
> 逐行预测导出（`*-test.jsonl` 等）按公开边界**不发布**——本文与 report.json
> 提供的是可公开的聚合层。

生成时间：2026-10-14。数据集与两个公开 run 的指标均可在上述 report.json
的 `split_report` 字段交叉核对（含数据集 sha256、组清单与全部类分布）。

---

## 1. 数据集总体

| 项 | 值 |
|---|---|
| 真实行数 | **668**（train 469 / calibration 95 / test 104） |
| 合成增强 | 200 行（**仅 train**，4 文件 × 50） |
| 切分方式 | 按任务组不相交切分（group-disjoint），`independent_groups: 37` |
| 组数 | train 22 / calibration 8 / test 7 |
| 近重复处理 | 阈值 0.88；跨 root 近重复边 0 |
| seed | 20261010 |
| 标签来源 | 全部 `expert_provisional`、`prompt_only` 证据（**无候选模型执行证据**） |

## 2. 标签 schema（三个固定选项头）

| 头 | 选项 | 说明 |
|---|---|---|
| `tier` | fast / balanced / flagship | 任务难度档位（路由到不同成本档的模型） |
| `needs_vision` | yes / no | 是否需要看图/截图；另有 **6 行 out-of-schema `uncertain`**（train 1 / cal 2 / test 3，评估时按缺失跳过） |
| `task_type` | extraction / research / implementation / debugging / review / visual / creative | 任务类型 |

专家标注置信度（`label.confidence`，0–1）聚合：

| 切分 | n | mean | median | min | max |
|---|---|---|---|---|---|
| train | 469 | 0.729 | 0.750 | 0.45 | 1.00 |
| calibration | 95 | 0.734 | 0.750 | 0.50 | 0.95 |
| test | 104 | 0.719 | 0.725 | 0.50 | 1.00 |

## 3. 逐切分类分布（真实行）

### tier

| 切分 | fast | balanced | flagship |
|---|---|---|---|
| train | 44 | 358 | 67 |
| calibration | 6 | 74 | 15 |
| test | **2** | 72 | **30** |

### needs_vision

| 切分 | no | yes | uncertain |
|---|---|---|---|
| train | 464 | **4** | 1 |
| calibration | 91 | **2** | 2 |
| test | 27 | **74** | 3 |

### task_type

| 切分 | extraction | research | implementation | debugging | review | visual | creative |
|---|---|---|---|---|---|---|---|
| train | 51 | 26 | 95 | 12 | 133 | **1** | 151 |
| calibration | 6 | 22 | 34 | 1 | 23 | 0 | 9 |
| test | 2 | 11 | 34 | 5 | 36 | **0** | 16 |

### tier × needs_vision 交叉（关键结构）

| 切分 | balanced,no | balanced,yes | flagship,no | flagship,yes | fast,no | 含 uncertain |
|---|---|---|---|---|---|---|
| train | 354 | 4 | 66 | 0 | 44 | b+1 |
| calibration | 71 | 1 | 14 | 1 | 6 | b+2 |
| test | 24 | 46 | 1 | **28** | 2 | b+2, f+1 |

## 4. 组结构（聚合，无组内容）

组规模（行数，降序）：

- train（22 组）：154, 62, 58, 41, 39, 23, 18, 12, 12, 10, 9, 7, 6, 4, 3, 3, 2, 2, 1, 1, 1, 1
- calibration（8 组）：59, 22, 4, 4, 2, 2, 1, 1
- test（7 组）：**79**, 12, 6, 4, 1, 1, 1

测试集按组构成（组哈希与 report.json `split_report` 一致，可连接）：

| 组 | 行数 | vision 构成 | tier 构成 |
|---|---|---|---|
| `ec267d56…a7722` | 79 | 70 yes / 7 no / 2 unc | 50 balanced / 29 flagship |
| `c023e8c9…4b52` | 12 | 12 no | 12 balanced |
| `427689b6…d784` | 6 | 4 yes / 1 no / 1 unc | 6 balanced |
| `0192d797…3fd35` | 4 | 4 no | 4 balanced |
| `6843b011…4f1a` | 1 | no | fast |
| `078dbf2e…2835` | 1 | no | flagship |
| `1cbebd81…7b08` | 1 | no | fast |

## 5. 合成增强聚合（仅进入训练）

200 行 = 4 × 50。合计构成：

- tier：balanced 156 / fast 27 / flagship 17
- vision：yes 107 / no 82 / uncertain 11
- task_type：review 71 / implementation 43 / visual 33 / extraction 21 / debugging 19 / creative 8 / research 5

训练视图（469 真实 + 200 合成 = 669，tier 均衡过采样至 1542 行/epoch）：
fast 71 / flagship 84 / balanced 514。vision 有效数权重 no 0.3997 / yes 1.6003。

## 6. prompt 聚合特征

| 切分 | 字符数 mean / median / max | 备注 |
|---|---|---|
| train | 1608 / 1519 / 17172 | 中文为主 |
| test | 1977 / 2100 / 3319 | 104 行中 102 行前 200 字符含 CJK |

评估侧 tokenizer 计量（8-bit run，test 104 行）：`input_tokens` mean 1580 /
median 1694 / max 2324；`context_truncated` = 0。

## 7. 已知结构性偏斜（分析结论时必须考虑）

1. **vision 分布倒挂**：真实 train yes 仅 4/469（0.85%），test yes 74/104
   （71%）——yes 侧几乎完全由合成数据教学，test 的 vision 指标度量的是
   "合成 → 单一 root 迁移"。
2. **测试集 76% 来自单一组**（79/104，含 70 个 yes、29 个 flagship）；
   行级 bootstrap CI 反保守（组内相关被忽略）。
3. **fast 仅 2 例**（两个单行组）——其 F1 在 0/1 间的一次摆动 =
   tier macro-F1 ±0.33。
4. **creative 语义漂移**：train creative 是最大真实类（151 行，100%
   no-vision、150/151 balanced），test creative 16 行全部来自主导组且
   15/16 为 vision-yes 的视觉创作任务；零样本基线、v3c、8-bit run
   三代模型全部 0/16。
5. **校准集稀有类不可校准**：vision yes 仅 2 行（同组）；task_type 缺
   visual 类（温度沿用 checkpoint）。2 行 out-of-schema `uncertain` 是
   grouped 校准路线预检失败的直接原因。
6. **visual 类无测试支持**（test 0 行；train 真实仅 1 行），其"指标"
   无意义，且会以噪声预测形式污染其他类的混淆矩阵。

## 8. 运行清单与公开指标

| run | 公开 artifact | 配置一句话 | 结果一句话 |
|---|---|---|---|
| `real-history-v3c` | `artifacts/real-history-v3c/`（4 文件） | 6 epochs，fp32 AdamW，fullft，legacy 校准 | best epoch 4；质量门 3 项未过（shadow） |
| `rt-v4-e1-8bit-r2` | `artifacts/rt-v4-e1-8bit-r2/`（4 文件，本次新增） | 3 epochs，adamw8bit，fullft，--legacy，同 seed 同数据 | best epoch 1；质量门 4 项未过（shadow） |
| 8-bit smoke | 无（管线检查，未归档为公共 artifact） | 1 epoch、24 训练行、80 秒 | tier acc 0.385（欠训练，仅验证流程） |
| 8-bit 首次尝试 | 无（死于 epoch 2 训练中段，无 report） | 推测 epochs=2 | 仅完成 baseline + epoch 1（train_loss 0.820） |

### 核心指标对照（同一冻结测试集；基线 = 未微调基座零样本）

| 指标 | 基线 | rt-v4-e1-8bit-r2 | real-history-v3c |
|---|---|---|---|
| tier accuracy | 0.692 | 0.721 | 0.721 |
| tier macro-F1 | 0.273 | 0.480 | 0.805 |
| tier fast / balanced / flagship F1 | 0 / 0.818 / 0 | 0 / 0.764 / 0.675 | 1.00 / 0.756 / 0.659 |
| flagship 召回 / 精确 | 0 / – | 0.933 / 0.528 | 0.933 / 0.509 |
| flagship 欠路由率 | 1.00 | 0.067 | 0.067 |
| vision accuracy | 0.673 | 0.881 | 0.881 |
| vision macro-F1 | 0.660 | 0.845 | 0.836 |
| vision 假阴性率 | 0.405 | 0.068 | 0.041 |
| task_type accuracy | 0.740 | 0.692 | 0.750 |
| tier NLL / ECE(5bin) | 0.683 / 0.144 | 0.601 / 0.113 | 1.488 / 0.408 |
| vision NLL / ECE | 0.504 / 0.137 | 0.311 / 0.049 | 0.357 / 0.078 |
| task_type NLL / ECE | 1.324 / 0.282 | 1.214 / 0.262 | 1.105 / 0.277 |
| 平均延迟 ms | 106 | 162 | 158 |

逐 epoch 选择历史、tau 扫描、温度、bootstrap CI：见两个 report.json 的
`training.history` / `logit_adjustment.sweep` / `temperatures` /
`{baseline,finetuned}.bootstrap` 字段。

### 按组测试集准确率（rt-v4-e1-8bit-r2 预测的聚合；vision n 剔除 uncertain）

| 组 | n(tier) | tier acc | vision acc (n) | task_type acc |
|---|---|---|---|---|
| `ec267d56…` | 79 | 0.709 | 0.909 (77) | 0.696 |
| `c023e8c9…` | 12 | 0.667 | 1.000 (12) | 0.583 |
| `427689b6…` | 6 | 1.000 | 0.800 (5) | 0.500 |
| `0192d797…` | 4 | 1.000 | 0.000 (4) | 1.000 |
| `6843b011…` | 1 | 0.000 | 1.000 (1) | 1.000 |
| `078dbf2e…` | 1 | 1.000 | 1.000 (1) | 1.000 |
| `1cbebd81…` | 1 | 0.000 | 1.000 (1) | 1.000 |

### tier 混淆矩阵（行 = 真实，列 = 预测）

基线：fast 0/0/2，balanced 0/72/0，flagship 0/30/0（几乎全预测 balanced）
r2：fast 0/2/0，balanced 0/47/25，flagship 0/2/28
v3c：fast 2/0/0，balanced 0/45/27，flagship 0/2/28

（列序 fast/balanced/flagship；完整含 NLL/Brier/每类 TP/FP/FN 的版本在
各 report.json。）

## 9. 引用与解读注意

1. 评估基准是**与 provisional 专家 prompt-only 标签的一致性**，
   不是候选模型任务成功率；质量门因此保持 shadow。
2. 错误行的专家置信度显著偏低（8-bit run：错误行均值 0.688 vs 正确行
   0.731），部分"错误"实为与边缘标注的分歧。
3. 本画像的所有比例受 §7 偏斜支配，**不得**作为生产流量分布估计；
   外推结论请附组结构上下文。
4. 画像数字由冻结数据与公开 report.json 计算；数据集或切分若变更，
   以 `split_report.dataset_sha256` 为准重新生成本文件。
