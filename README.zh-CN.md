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
| `model_utils.py` | 固定选项决策 logits、8K 上下文、训练/服务一致截断 |
| `prepare.py`、`extract.py` | 只读历史提取、脱敏及分组切分 |
| `server.py`、`policy.py`、`routes.json` | 可选独立推理 API（FastAPI） |
| `scripts/train.py` | v3c 配方安全入口（拒绝覆盖既有 run） |
| `configs/v3c.json` | 完整 run 的精确 CLI 配方 |
| `tests/` | CPU 单元测试（无需 GPU） |
| `data/labeling/RUBRIC.md` | 档位/视觉/任务类型标注标准 |
| `research/`、`docs/` | 全参调研、红队记录、迭代史 |
| `docs/DATA.md` | 数据 schema、合成样例、如何构造自己的数据集 |

## 快速开始

需要 Python ≥ 3.11、CUDA GPU（全参训练峰值约 15GB）与匹配驱动的 PyTorch CUDA 构建。

```sh
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

先下载固定 revision 的基座模型放入 `models/intern-decision-0.8b/`（权重不入 Git），然后：

```sh
.venv/bin/python scripts/train.py --run v3c-reproduction --dry-run   # 只看命令
.venv/bin/python scripts/train.py --run v3c-reproduction             # 完整六轮
.venv/bin/python scripts/train.py --run v3c-smoke --smoke            # GPU 冒烟
```

独立推理服务：

```sh
.venv/bin/python server.py --run real-history-v3c --port 19949 --routes routes.json
curl http://127.0.0.1:19949/health
```

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
3. 校准集按随机行分为 selection/fitting 两半，未再按组隔离；bootstrap 是实现层面重采样，不是真实任务成功证据。
4. 无 optimizer resume，中断需重跑；相同参数不保证跨 GPU/依赖版本逐位一致。

## 许可证

代码与文档：**Apache-2.0**（见 [LICENSE](LICENSE)、[NOTICE](NOTICE)）。基座模型上游 Apache-2.0；再分发微调权重须继承其义务。真实任务数据不随仓库分发，故不在本许可覆盖范围内。
