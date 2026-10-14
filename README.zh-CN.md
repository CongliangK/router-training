# router-training（中文文档）

[English](README.md)

用一个小型视觉语言模型（0.8B）做**子任务难度路由器**的完整训练流水线：读任务 prompt，输出 fast / balanced / flagship 三档能力档位、是否需要视觉（needs_vision）与任务类型（task_type）三个固定选项决策，附校准、诚实评测与生产质量门。基座为
[`internlm/Intern-Decision-0.8B`](https://huggingface.co/internlm/Intern-Decision-0.8B)
（固定 revision `85a0cc5a99d67ea8d56dfe98115689212867171d`，Apache-2.0）。

路由器只打分、不生成任务答案：固定选项 logits 决策，训练与服务共用同一条确定性截断链路（8K 上下文，超长走头/尾保留 + `[ROUTING_CONTEXT_TRUNCATED]` 标记）。

## 为什么值得看

模型本身不稀奇，稀奇的是周围的工程：

- **专家模仿训练 + 诚实评测**：指标全部标注为「与专家标签的一致性」，不是真实任务成功率；`quality_gate` 如实保留未通过状态。
- **校准作为一等公民**：留出集温度拟合 + 决策侧 logit 调整扫参（`tau*log(prior)`，受 flagship 召回/误报约束）。
- **红队加固过的采样与统计**：类均衡过采样（修掉了多数类塌缩）、近重复任务按 split_group 隔离、簇级 bootstrap 置信区间、校准集再分 selection/fitting 两半防复用。
- **完整失败史**：v2→v3→v3b→v3c 迭代记录（tier 塌缩、bf16 更新下溢、混合精度内核崩溃、16GB 显存工程）保留在 `research/` 与 `docs/`。

## 最终结果（v3c，留出测试集 n=104，专家标签一致性）

| 指标 | v3c |
|---|---:|
| tier macro-F1 | 0.805 |
| flagship 召回 / 精确 | 0.933 / 0.509 |
| vision 召回 / 精确 | 0.959 / 0.887 |
| test 上下文截断 | 0 |

质量门**未通过**（缺独立真实任务成功 benchmark；flagship 过升仍在）。这是有意保留的诚实状态，详见 `artifacts/real-history-v3c/quality-gate.json`（本地完整检出）。

## 目录结构

| 路径 | 用途 |
|---|---|
| `train_eval.py` | LoRA / 语言骨干全参训练、校准、评测、质量门 |
| `calibration_groups.py` | 纯 stdlib 的按组隔离校准规划与 OOF tau/T 选择（issue #2） |
| `model_utils.py` | 固定选项决策 logits、8K 上下文、训练/服务一致截断 |
| `prepare.py`、`extract.py` | 只读历史提取、脱敏及分组切分 |
| `server.py`、`policy.py`、`routes.json` | 可选独立推理 API（FastAPI） |
| `scripts/train.py` | 安全启动器：显式 `--legacy`（v3c 归档配方）或预注册分组的分组模式 |
| `scripts/validate_calibration.py` | 纯 CPU 匿名结构校验（绝不读取 test 数据） |
| `configs/v3c.json` | 完整 run 的精确 CLI 配方 |
| `tests/` | CPU 单元测试（无需 GPU） |
| `data/labeling/RUBRIC.md` | 档位/视觉/任务类型标注标准 |
| `research/`、`docs/` | 全参调研、红队记录、迭代史 |
| `docs/DATA.md` | 数据 schema、合成样例、如何构造自己的数据集 |

## 快速开始

需要 Python ≥ 3.11。训练需要 CUDA GPU（全参训练峰值约 15GB）与匹配驱动的 PyTorch CUDA 构建；CPU 单元测试不需要 GPU 或模型权重。

```sh
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

### 公开 checkout 验证（发布标准）

全新公开 clone 必须能安装原样的 `requirements.txt` 并运行
`python -m unittest discover -s tests -v` 全绿，无需私有文件、GPU 或权重。
GitHub Actions 在 push 和 pull request 时执行这一 CPU 检查；CPU runner
也保留固定的 CUDA 构建依赖，不替换为 CPU 版本。

公开测试在临时目录生成明确标注的合成数据，经 `prepare.py` 验证切分计数、
完整 ID 覆盖、ID/组互斥，以及与合成补充样本的分离。fixture 的计数
（120 条主样本、200 条补充样本）不是历史 668 条数据集。
缺少私有资产时，另三项归档验证会明确 **skip**：真实切分计数/互斥、
真实与合成样本 prompt 分离，以及 `snapshot-manifest.json` 中的源码校验和。
资产存在时仍执行原断言；skip 不代表已验证私有归档完整性。
测试不要求或分发真实 prompt、标签或私有指纹。

先下载固定 revision 的基座模型放入 `models/intern-decision-0.8b/`（权重不入 Git），然后（行级二分校准已改为**显式 `--legacy`**，仅用于原样复现 v3c，不再是新 run 的默认）：

```sh
.venv/bin/python scripts/train.py --run v3c-reproduction --legacy --dry-run   # 只看命令
.venv/bin/python scripts/train.py --run v3c-reproduction --legacy             # 完整六轮
.venv/bin/python scripts/train.py --run v3c-smoke --legacy --smoke            # GPU 冒烟
```

### 新 run 的按组隔离校准（issue #2）

外部审查 P1-3 / [issue #2](https://github.com/CongliangK/router-training/issues/2)：归档配方把校准集**按行**打乱二分为 selection/fitting 两半。任务 id 唯一，但同源的近重复任务共享 `split_group`，行级二分让相关行同时落在两侧；校准集只有约 8 个记录组，换任何 seed 重新按行切分都修不好复用问题。

因此新 run 默认走 `calibration_groups.py`（纯 stdlib，无 torch 也可导入）的严格预注册分组：

- **A/B/C 角色预注册**在 roles JSON（`role_groups` + 可选 `support_policy`/`protocol`）。A=checkpoint 选择行，B=温度/tau 拟合行，C=保留作最终配置选择。组与 id 跨角色互斥且必须覆盖全部校准行；缺角色即硬失败，**绝无按 seed 重切的回退**。
- **支持门 fail-closed**。示例门（provisional，*未经用户冻结*）：A/B/C ≥5/6/5 组、每区每 required 头每类 ≥5 行且 ≥2 组、B 每折训练并集每类 ≥2 组。vision 保持 required——preflight 绝不默默降级成 tier-only。`task_type` 可选：支持不足时温度固定 T=1 并如实标注未校准（绝不沿用 legacy 的 2.747 兜底）。业务 schema 只有一个来源：`model_utils.QUESTIONS`，由单一组装函数装配；stdlib 预检通过解析同一文件获得（不导入 torch）。roles 配置不得覆盖 schema、未知键一律拒绝——预检、launcher dry-run 与实际 run 校验同一事实。
- **tau 由 B 上确定性三折 GroupKFold 选出**（组按行数降序、sha256 打平、整组不拆；每条 B 行恰为一次 holdout）。每个 tau 先对各折训练行施加 `z − tau·log(prior)`，**再**拟合该折温度，holdout 用同公式评分；全程组等权。holdout 折缺类时该头记 null（绝不伪造 0 分）；无可行 tau 就不选择。选定后只在全 B 上重拟合并绑定 plan/prior/schema/protocol/checkpoint 哈希。这是开发估计，不是无偏评测。
- **prior 只来自真实唯一 train 行**（排除合成增广与重复 id；任一类无支持即硬错误）。
- **selected 校准产物可直接被服务消费**。selected 结果写出的 `calibration.json` 携带 `temperatures` 与按类的 `tier_adjust = tau·log(prior)`（另含 `tier_prior`、tau 与绑定哈希）——正是 `server.py` 加载、`model_utils.score` 从原始 logits 减去的量，服务端因此执行与拟合完全相同的"先调整再除温度"公式。**insufficient** 结果**不写** `calibration.json`：服务拒绝加载该 run，绝不默默路由未调整的 logits；质量门保持未通过，execution 模式永不授权。
- **绑定锁定 checkpoint 权重字节**。校准绑定中的 `checkpoint_ref` 是对实际 checkpoint 文件（`adapter/` 或 `full-checkpoint/`，流式 SHA-256，相对文件名明确）的摘要，不是 `best-epoch.json` 文本；交付权重任一字节变化都会改变绑定，自由文本或被篡改的摘要一律拒绝。
- **旧行级二分仅保留为显式 legacy**（`--legacy-calibration`），逐字复现 v3c。分组开发 run 绝不打开 test；`--structural-dry-run`（或分组模式下 `scripts/train.py --dry-run`）执行真实 CPU 预检——无模型、无产物、不读 test：

```sh
.venv/bin/python scripts/train.py --run v4-dev --roles roles.json --dry-run
.venv/bin/python scripts/validate_calibration.py --roles roles.json --data-dir data
```

`--data-dir` **只**作用于该预检：实际分组启动永远读仓库 `data/` 目录（`train_eval.py` 没有 data-dir 参数），非默认 `--data-dir` 在启动时被拒绝、仅允许与 `--dry-run` 同用——run 绝不会预检一份数据却在另一份上训练。分组 `--smoke` 名副其实地短：24 行训练数据 + 按整组收缩的校准子池，并用**同一**支持策略重新校验（绝不为了 smoke 放宽门槛；legacy smoke 保持归档的 24 行行为）。

输出摘要只含计数（不含组名、id、标签或 prompt）。**现状**：归档校准池无法满足示例门（约 8 个组、部分类集中在极少数组——见 issue #2），现有数据无法实例化任何分组 run；门槛不为迁就数据放宽，需要补按组隔离的开发数据。训练的完整 dev-no-test 与获批数值协议尚未满足——本仓库不声称已具备训练前置条件。

独立推理服务：

```sh
.venv/bin/python server.py --run real-history-v3c --port 19949 --routes routes.json
curl http://127.0.0.1:19949/health
```

### 服务契约（schema version 2）

默认 `--mode diagnostic` 始终**仅供诊断**，即使模型已合格也不授予执行许可。
`/health` 与 `/route` 均包含 `mode`（`diagnostic` / `execution`）、
`quality_gate_passed`、`quality_gate_mode`（`shadow` / `qualified`）、
`quality_gate_status`、`execution_allowed` 与 `advisory_only`。非空
`recommendation` 内也包含后两个许可字段。建议存在不等于允许执行；调用方必须
检查路由响应及建议的 `execution_allowed: true`，不能仅凭模型名称采取行动。

只有显式 `--mode execution`、对应 run 的 gate 为布尔值 `passed: true` 且
`mode: qualified`、并存在授权路由时，建议才允许作为执行依据。未通过、缺失、
不可读、畸形或自相矛盾的 gate 在模型评分前返回 HTTP 503，`detail` 内带拒绝
执行的契约；无可用建议也返回 503。诊断模式在这些 gate 状态下仍可给出明确标记
`advisory_only: true` 的建议。健康响应的许可表示服务就绪，不保证每个请求都有
兼容路由。schema 2 不再用响应 `mode=shadow/qualified` 表示质量门，而是将服务
模式和 gate 状态分开；旧客户端必须更新，不能沿用旧字段推断执行权限。
v3c 归档模型是 **shadow，不是生产合格模型**；历史服务切换或 `ready: true`
均不等于质量门通过。本契约不修改模型、质量门归档或实际部署。

### 基座与权重

基座 `internlm/Intern-Decision-0.8B`，固定 revision，加载不启用 remote model code。`.safetensors` 等大权重被 Git 忽略、**不在本仓库分发**：请从上游 Hugging Face（Apache-2.0）下载固定 revision，或复制本地权重后训练。基座推理代码、tokenizer、配置及上游许可证保留在 Git 中。

### v3c 配方

seed 20261010、accumulation 8、label smoothing 0.05、beta 0.999、`--fullft --synth --swa`。全模型 fp32 保存主权重，forward 用 bf16 autocast；冻结 visual/projector/lm_head/embed_tokens，可训练参数 498,113,344。

## 数据：有意不随仓库发布

真实训练数据（668 条历史子任务 prompt + 专家标签）含可识别的个人项目细节，**不公开**。仓库提供：

- 完整**标注标准**（`data/labeling/RUBRIC.md`）；
- 精确 **JSONL schema** 与若干**明确标注为合成**的示例行（`data/synthetic-examples/`）；
- **自建数据集指南**（`docs/DATA.md`）：如何从自己的多智能体会话日志提取、脱敏、切分、标注。

数据符合 schema 后，全部流水线代码原样可用。

## 复现要点（保留产出该 checkpoint 的算法原样）

1. fullft 实际初始 lr 为 **2e-6**，不是 CLI 默认 `lr=1e-4`；调度为 3% warmup + cosine（乘数下限 1e-3），总步数估算用过采样前数据量，实际更新 1158 次。
2. SWA 注释写「三个最佳 epoch」，实际池未剔除旧入选 epoch；soup 未胜出，最终选择单独 **epoch 4**。
3. 校准集按随机行分为 selection/fitting 两半，未再按组隔离（组会跨半泄漏，issue #2）；bootstrap 是实现层面重采样，不是真实任务成功证据。该行为仅在显式 `--legacy-calibration` 下原样保留，新 run 默认按组隔离（见上节）。
4. 无 optimizer resume，中断需重跑；相同参数不保证跨 GPU/依赖版本逐位一致。

## 许可证

代码与文档：**Apache-2.0**（见 [LICENSE](LICENSE)、[NOTICE](NOTICE)）。基座模型上游 Apache-2.0；再分发微调权重须继承其义务。真实任务数据不随仓库分发，故不在本许可覆盖范围内。
