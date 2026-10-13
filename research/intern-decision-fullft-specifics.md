# Intern-Decision 系列全参微调一手调研

调研时间：2026-10-13。调研人：模型特定调研员（子智能体）。
调研对象：InternLM **Intern-Decision**（0.8B / 2B / 4B，Qwen3.5 骨干，混合线性注意力，`<decision>` 决策头）的**全参微调**一手信息。

信息源优先级与实际覆盖情况：

1. **官方 GitHub**（github.com/InternLM/Intern-Decision，main 分支）——✅ 已逐文件抓取 raw 源码：README.md、configs/training/qwen35.py、scripts/train.sh、scripts/training_config.py、scripts/verify_checkpoint.py（经 README）、docs/DATA.md、docs/EVALUATION.md、requirements-xtuner.txt、requirements-inference.txt、src/inputs/masked.py、src/inputs/schema.py、src/model/qwen.py、src/model/size_config.py、runtime/xtuner/sitecustomize.py、src/eval/{collect,fit,replay,calibration}.py、benchmarks/temperature-presets.json
2. **HuggingFace 模型页与讨论区**——✅ internlm/Intern-Decision-0.8B 模型卡全文、config.json、model.safetensors.index.json、discussions API；HF 全站衍生模型检索
3. **XTuner 文档**——⚠️ 公开 readthedocs 仍是 legacy 文档；Intern-Decision 实际用的是 InternLM/xtuner fork 的 **pinned commit `fb51baebdf91b03bd39255c4427edac9aca82865`** 的 `xtuner.v1.*` 新 API，配方全部在 Intern-Decision 仓库内自洽
4. **社区微调先例**——✅ 找到 2 个（FluidInference Showdown LoRA 蒸馏、PyTrio-Jev LoRA 复刻）
5. **技术博客**——✅ orcarouter 系列分析（第三方，仅作旁证）

---

## 0. 架构事实核对（调研前提）

以下全部来自 [0.8B checkpoint 的 config.json](https://huggingface.co/internlm/Intern-Decision-0.8B/resolve/main/config.json)（snapshot `85a0cc5a`）与 [model.safetensors.index.json](https://huggingface.co/internlm/Intern-Decision-0.8B/resolve/main/model.safetensors.index.json)：

- `model_type: "qwen3_5"`，`architectures: ["Qwen3_5ForConditionalGeneration"]`；HF 标签 `base_model:finetune:Qwen/Qwen3.5-0.8B`（[0.8B 模型页](https://huggingface.co/internlm/Intern-Decision-0.8B)，[4B](https://huggingface.co/internlm/Intern-Decision-4B) 同理）。**官方就是在 Qwen3.5 instruction checkpoint 上"decision tuning"得到的**，模型卡原话："Intern-Decision-0.8B is a multimodal structured decision model fine-tuned from Qwen/Qwen3.5-0.8B"。
- **24 层 = 18 层 `linear_attention` + 6 层 `full_attention`**，`full_attention_interval: 4`：完整自注意力在第 **3, 7, 11, 15, 19, 23** 层（safetensors 里这些层是 `self_attn.q_proj/k_proj/v_proj/o_proj/q_norm/k_norm`，其余层是 `linear_attn.*`）。
- 线性注意力 = **Gated DeltaNet**（`[src/model/size_config.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/model/size_config.py)` 中映射为 `GatedDeltaNetConfig`）。0.8B 参数：`linear_num_key_heads=16, linear_num_value_heads=16, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4`。
- **权重命名（全参微调加载时最关心）**：线性层为 `model.language_model.layers.N.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,conv1d}.weight` + `A_log`、`dt_bias`、`norm.weight`。任务描述里的 "in_proj_a/b/qkv/z" 与此完全对应。
- 完整注意力层：`num_attention_heads=8, num_key_value_heads=2, head_dim=256, attn_output_gate=true, qk_norm（zero-centered RMSNorm）`；RoPE `partial_rotary_factor=0.25, rope_theta=1e7, mrope_interleaved=true, mrope_section=[11,11,10]`。
- **`mamba_ssm_dtype: "float32"`**（checkpoint config.json 原字段）——SSM 状态以 fp32 维护，这是混合架构数值稳定的官方取向。
- `tie_word_embeddings: true`，`vocab_size: 248320`；`attention_dropout: 0`、`attention_bias: false`。
- [size_config.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/model/size_config.py) 加载时会**断言校验**骨干结构，微调自己的 checkpoint 时这些断言必须继续成立（原文：`assert t.get("full_attention_interval", 4) == 4`、`assert t["layer_types"] == expected`、`assert not t["attention_bias"] and t["attention_dropout"] == 0 and t["attn_output_gate"]`）。**即官方只支持 interval=4 的 Qwen3.5 dense 布局**。

官方 README 明确冻结视觉部分、只训语言骨干：*"It fine-tunes the language backbone of Qwen3.5 while freezing the vision tower and projector."*（[README](https://github.com/InternLM/Intern-Decision)）

---

## 1. 官方 XTuner 训练配方细节

来源：[configs/training/qwen35.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/configs/training/qwen35.py)、[scripts/train.sh](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/scripts/train.sh)、[README](https://github.com/InternLM/Intern-Decision)、[requirements-xtuner.txt](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/requirements-xtuner.txt)。**qwen35.py、train.sh、docs/DATA.md 三个目标文件全部抓到原文**，以下是逐项摘录（均为源码原文，非转述）。

### 1.1 优化器 / 学习率 / 轮数（`trainer = TrainerConfig(...)`）

| 项 | 值 | 源码原文 |
|---|---|---|
| 优化器 | AdamW | `optim_cfg=AdamWConfig(lr=float(os.environ.get("LR", "2e-6")), weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.01")), foreach=False)` |
| **learning rate** | **`LR=2e-6`**（默认，env 可覆盖） | 同上 |
| weight decay | `0.01` | 同上 |
| `foreach` | `False`（显式关掉 foreach 优化器实现） | 同上 |
| scheduler | cosine | `lr_cfg=LRConfig(lr_type="cosine", warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.03")), lr_min=float(os.environ.get("LR_MIN", "2e-7")))` |
| warmup | `warmup_ratio=0.03` | 同上 |
| lr 下限 | `LR_MIN=2e-7` | 同上 |
| **训练轮数** | **`TOTAL_EPOCH=2`** | `total_epoch=int(os.environ.get("TOTAL_EPOCH", "2"))` |
| **全局 batch** | **`GLOBAL_BATCH_SIZE=16`** | `global_batch_size=int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))` |
| 序列并行 | `SP_SIZE=1` | `sp_size=int(os.environ.get("SP_SIZE", "1"))` |
| 种子 | `trainer.seed = 42` | 源码硬编码 |

### 1.2 Loss

```python
loss_cfg=CELossConfig(
    mode=os.environ.get("LOSS_MODE", "chunk"),          # 分块交叉熵
    chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")),
    loss_reduction=os.environ.get("LOSS_REDUCTION", "token"),  # 按 token 平均
),
```

README（Method 节）对 loss 语义的关键句：*"Training cross-entropy uses the full vocabulary."* —— 训练时是**全词表 CE**（只落在 marker 位置），推理时才做受限 softmax。

### 1.3 序列长度 / packing

| 项 | 默认值 | 说明 |
|---|---|---|
| `SAMPLE_MAX_LENGTH` | 8192 | 单样本上限；**超长直接报错拒绝，禁止截断**（masked.py：`raise ValueError(f"Example has {item['num_tokens']} tokens, above {limit}; truncation is forbidden")`） |
| `PACK_MAX_LENGTH` | = SAMPLE_MAX_LENGTH (8192) | packing 上限 |
| `pack_level` | `"soft"` | 软 packing |
| `pack_to_max_length` | `True` | pack 到定长（配合 `cu_seq_lens` 做变长注意力） |
| `pack_chunk_size` | 10000 | |
| `global_pack` | `True` | 跨 rank 全局 packing |
| `group_by_length` | `True` | 长度分组降低 padding 浪费 |
| `pack_extra_buffer_size` | 20 | |
| `pack_workers` / `DATALOADER_NUM_WORKERS` | 4 / 4 | |

多模态 packing 权重：`llm_pack_weight=1.0`、`visual_pack_weight=0.0`（视觉 token 不计 pack 权重）；`max_pixels=16384*32*32`、`rand_video_max_frames=24`。

### 1.4 bf16 / gradient checkpointing / FSDP

- **FSDP 全参分片**（XTuner v1 的 `FSDPConfig`），`RECOMPUTE_RATIO=1.0` —— **重计算比例 1.0，即全量 activation checkpointing**；视觉部分 `VISION_RECOMPUTE_RATIO=0.0`（冻结模块不做重计算）。
- `checkpoint_preserve_rng_state=False`。
- `MODEL_COMPILE` 默认 `False`（`torch_compile` 关）。
- bf16：README 延迟表注明 BF16；XTuner 后端 *"uses CUDA/BF16"*。
- 冻结策略：`freeze_vision=True, freeze_projector=True, freeze_language=False`；qwen.py 构造函数里还有硬校验，原文：`if any(p.requires_grad for p in self.vision_tower.parameters()): raise ValueError("Vision tower must remain frozen")`。
- 保存：`HFSaveCfg(fp32_keys_pattern=None)`——注释原文：*"Torch 2.6 removed composable FSDP's ignored_params argument. Keep fp32-only norm parameters in the normal sharding path for compatibility."*
- 激活检查点实现替换：`DecisionQwenConfig.build()` 把 XTuner 的 reentrant wrapper 换成 **non-reentrant**（`CheckpointImpl.NO_REENTRANT`），注释：*"Project-local adaptation of XTuner's fixed reentrant wrapper. The frozen visual modules do not need activation checkpointing."*（[src/model/qwen.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/model/qwen.py)）
- **只保存最终 HF checkpoint，无自动续训**：`trainer.checkpoint_interval = None; trainer.snapshot_interval = None; trainer.hf_interval = None; trainer.auto_resume = False`（README：*"saves only the final HF checkpoint. It does not automatically resume a prior run."*）。`CHECKPOINT_INTERVAL`/`HF_INTERVAL` 的 env 默认值 500 会被这三行覆盖为 None。

### 1.5 启动脚本 scripts/train.sh（环境变量全集）

```bash
export XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0            # 用 HF 实现、显式关 FA3
export XTUNER_ACTIVATION_OFFLOAD=0 XTUNER_GC_ENABLE=1
export PYTHONPATH="$PWD/runtime/xtuner:$PWD:$XTUNER_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export WEIGHT_AUDIT="$WORK_DIR/weight-audit"          # 训练前权重审计
export DS_BUILD_OPS=0 DS_BUILD_FP_QUANTIZER=0
python -m torch.distributed.run --nproc-per-node="${NPROC_PER_NODE:-4}" ... \
  -m xtuner.v1.train.cli.sft --config configs/training/qwen35.py
```

约束：`WORK_DIR` **必须不存在**（已存在则 exit 2）；训练前先跑 `python -m scripts.training_config` 把解析后的配置 + 数据/源码 SHA256 写进 `training-config.json`（[training_config.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/scripts/training_config.py)）；训练后用 `python -m scripts.verify_checkpoint` 对比基座做分片级权重审计。

### 1.6 版本锁定（复现训练环境的硬要求）

README "XTuner training and inference" 原文命令：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/InternLM/xtuner.git third_party/xtuner
git -C third_party/xtuner checkout fb51baebdf91b03bd39255c4427edac9aca82865
python -m pip install --no-deps -e third_party/xtuner
# Install a flash-attn 2.8.3 wheel/build matching your Torch/CUDA toolchain.
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

[requirements-xtuner.txt](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/requirements-xtuner.txt) 关键钉版：`transformers==4.57.0`、`flash-linear-attention==0.4.2`、`deepspeed==0.17.5`、`peft==0.14.0`、`mmengine==0.11.0rc2`。
README 原句：*"The explicit dependency override preserves Transformers 4.57.0 for this XTuner path."*、*"Keep HF and XTuner in separate virtual environments: their tested Transformers versions differ."*（HF 推理环境是 `torch==2.9.1 + transformers==5.14.1`，见 [requirements-inference.txt](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/requirements-inference.txt)，两个环境严禁混用。）

**通用性**：README 原句：*"`configs/training/qwen35.py` reads architecture dimensions from the base checkpoint and supports the 0.8B, 2B, 4B and 9B dense Qwen3.5 variants."* —— 一份配置通吃四个尺寸。

---

## 2. 混合注意力架构在全参微调中的已知问题

### 2.1 flash-attn 版本要求

- **训练（XTuner 路径）：`flash-attn==2.8.3`，Torch 2.6.0 + cu126**，且要"matching your Torch/CUDA toolchain"的 wheel/自编译；**`XTUNER_USE_FA3=0` 显式关闭 FlashAttention-3**（train.sh 与推理文档都设了）。README 原句：*"Install a flash-attn 2.8.3 wheel/build matching your Torch/CUDA toolchain."*
- 注意：完整注意力层（6 层）与视觉塔走 flash-attn/SDPA；线性注意力层不走 flash-attn，走下面的 fla + causal-conv1d。

### 2.2 fla（flash-linear-attention）依赖

- **训练与 XTuner 推理环境钉死 `flash-linear-attention==0.4.2`**；HF 推理延迟表也注明 *"Transformers 5.14.1 and flash-linear-attention 0.4.2"*。
- [runtime/xtuner/sitecustomize.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/runtime/xtuner/sitecustomize.py) 会**先探测真实 fla**：`from fla.modules import FusedRMSNormGated`，导入成功用真库；失败才注入纯 PyTorch 的 CPU fallback（注释：*"Prefer the real FLA modules when the environment has a working GPU/Triton runtime. CPU-only import checks may not have a Triton driver, so retain a small fallback."*）。**结论：GPU 全参训练必须装好 0.4.2，fallback 只为 CPU 冒烟测试**。
- causal-conv1d 同理：环境没有 `causal_conv1d` CUDA 扩展时注入可微分的 depthwise conv 替身；但**训练路径上官方在 `DecisionQwen.__init__` 里无条件替换了 `causal_conv1d_fn`**（见 2.3）。

### 2.3 packed 训练下 causal conv 的跨样本污染问题（官方最重要的架构级补丁）

[src/model/qwen.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/model/qwen.py) 中 `packed_causal_conv1d` 的 docstring 原文：

> "Channel-last depthwise convolution, with zero history at every row boundary. Accumulate the short (four-tap) kernel in FP32, matching the CUDA contract. **Masking each tap preserves gradients while preventing cross-example inputs.**"

背景：pack 多条样本进一个序列后，4-tap 因果卷积天然会把上一条样本的末尾 token 当作下一条样本开头的历史。官方实现：①FP32 累加（数值稳定，"matching the CUDA contract"）；②用 `seq_idx` 屏蔽跨样本 tap。构造函数里对 language_model 所有带 `causal_conv1d_fn` 的模块做替换：`for module in self.language_model.modules(): if hasattr(module, "causal_conv1d_fn"): module.causal_conv1d_fn = packed_causal_conv1d`。**自己写微调管线时这是最容易踩的坑：不做 per-sequence masking，线性注意力层会在 packed 序列边界学到脏历史。**

### 2.4 Torch 2.6 × FSDP 兼容（sitecustomize.py 全部补丁清单）

sitecustomize.py 逐条注释的补丁（XTuner fork 是按老 Torch 写的，Torch 2.6 全靠这层垫片）：

1. `torch.distributed.checkpoint` 的 `_async_process_executor` 被删 → 用 `get_free_port` 顶替；
2. fla `FusedRMSNormGated` 缺失时的 CPU fallback；
3. `causal_conv1d` CUDA 扩展缺失时的纯 PyTorch 替身；
4. **`fully_shard(ignored_params=...)` 被 Torch 2.6 移除** → wrapper 弹掉该 kwarg（注释：*"Torch 2.6 removed the ignored_params keyword accepted by the pinned XTuner FSDP call."*）；同时 `distribute_tensor` 对 Replicate-only placement 直接返回本地张量（*"leave replicate-only parameters local"*）；
5. `Shard._local_shard_size_and_offset` 改名补齐；
6. `DefaultSavePlanner` 第 5 个构造参数（plan caching）补齐；
7. `torch.distributed.checkpoint._extension.StreamTransformExtension` 空壳补齐。

**含义：换 Torch 版本（>2.6 或 <2.6）这层垫片都要重新核对；官方推荐照抄 torch==2.6.0+cu126。**

### 2.5 纯文本 batch 与冻结视觉 FSDP 的集体通信问题

qwen.py `_participate_empty_vision` docstring：*"Match visual FSDP collectives without adding image evidence to text rows."* —— 纯文本 batch 也要用零像素假图参与冻结视觉塔的 FSDP collective（输出丢弃），否则分布式训练 hang/报错。日志原句：*"Text-only batch participated in frozen visual FSDP collectives; outputs discarded"*。`only_llm_forward=not IS_MULTIMODAL`（纯文本时不走视觉前向，但仍要匹配 collective）。

### 2.6 梯度不稳定的公开报告

- **GitHub issues 目前共 4 条**（[#1 license](https://github.com/InternLM/Intern-Decision/issues/1)、[#2 Browser use 细节（已关）](https://github.com/InternLM/Intern-Decision/issues/2)、[#3 请求 LoRA 微调](https://github.com/InternLM/Intern-Decision/issues/3)、[#4 贡献导航方向](https://github.com/InternLM/Intern-Decision/issues/4)），**没有任何梯度爆炸/NaN/训练崩溃报告**。
- HF 0.8B 讨论区 3 条（[API](https://huggingface.co/api/models/internlm/Intern-Decision-0.8B/discussions)：打 tag 的 PR、MLX 移植 `pip install intern-decision-mlx`、Spaces demo 请求），同样无训练问题。
- 官方代码里能看到的**稳定性取向**（无公开故障报告，但全是防御性设计）：`mamba_ssm_dtype: float32`；conv FP32 累加；CE loss 分块（chunk 1024）计算；`recompute_ratio=1.0`；`foreach=False`；`WARMUP_RATIO=0.03`；微小的 `LR=2e-6`；activation checkpointing 换 non-reentrant 且不保 RNG state。
- README 另一条数值纪律：*"Different kernels and BF16 rounding can produce probability differences between backends; do not claim bitwise backend equivalence."*（HF vs XTuner 后端概率有差异，属预期。）

---

## 3. `<decision>` 决策头在微调中的注意点

来源：[src/inputs/schema.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/inputs/schema.py)、[src/inputs/masked.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/inputs/masked.py)、[0.8B 模型卡](https://huggingface.co/internlm/Intern-Decision-0.8B/raw/main/README.md)。

### 3.1 标记 token 的词表处理

- `DECISION_TOKEN = "<decision>"`（schema.py 硬编码常量）。
- **训练时动态加为特殊 token**：两个 TokenizeFunction 的 `__init__` 都执行 `tokenizer.add_special_tokens({"additional_special_tokens": [DECISION_TOKEN]})` 再取 `decision_token_id`。**从 Qwen3.5 基座起训时这是新增词表行（0.8B 词表 248320，tie_word_embeddings=true，新 token 的 embedding 是新参数）；官方发布的 checkpoint tokenizer 已含该 token**（CoreML 移植卡原句：*"The checkpoint's tokenizer (Qwen3.5 plus the `<decision>` token)"*）。
- 若从 Intern-Decision checkpoint 继续微调，tokenizer 已带 `<decision>`，重复 add_special_tokens 是幂等的；但要确保 `convert_tokens_to_ids` 拿到的是单 token id。
- 用户证据里出现保留标记会直接报错：`if DECISION_TOKEN in user_text: raise ValueError("Reserved decision marker appears in input evidence")`。

### 3.2 标记位对齐的机制（"标记前一位置"契约）

- **编译**：assistant 内容是完整 JSON 骨架，每个字段一个 `<decision>` 占位：`skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), indent=4)`；真实答案**只进 labels 不进 input**（README：*"Ground-truth answer symbols appear only in labels, never in the input."*）。
- **写 label**：`_mask_labels` 在 input_ids 中找**所有** marker 位置，labels 全 -100，然后 `masked[position] = target_id`（target 是该字段答案符号的 token id）。硬校验：marker 数必须等于字段数，否则 `raise ValueError(f"decision marker count ... does not match target count ...")`；没有 marker 也报错。
- **causal 移位发生在 collator**：`_masked_collate_one` 里 `input_ids = input_ids[:, :-1]; labels = labels[:, 1:]`，docstring 原文：*"The causal shift below places each target at the preceding input position, **matching the logits used by inference** and Jev constrained decoding."* —— 即**监督落在 marker 前一个 token 位置**，推理读的也是 marker 前位置的 logits（模型卡：*"read logits at the position immediately before each placeholder"*）。**自己写训练脚本时最容易错的就是这一步：label 写在 marker 位，shift 之后才对齐到"前一位"；如果你的框架自己再做一次 shift 就会错位一位。**
- packing 时 shift 与 `cu_seq_lens` 同步裁剪（`num_tokens[-1] -= 1`），多模态 position_ids 也做 `[..., :-1]`。
- **候选符号**：`ANSWER_SYMBOLS = "A-Z a-z 0-9"` 共 62 个，字段选项按顺序映射 A,B,…；每个字段所有候选必须**恰好单 token**（`_target_ids` 校验：`raise ValueError("Every candidate symbol must encode to exactly one token")`）。`noul` 固定 no 在前 yes 在后；choice 按 criteria 插入顺序；score 列表转 "0","1",…。每字段最多 62 个选项。
- 问题与选项顺序必须保持原序（README/Data.md 反复强调），字段答案互不可见、一次前向全部监督。

### 3.3 社区对照（证明机制可迁移）

- PyTrio-Jev 复刻（见第 4 节）**没有用特殊 token**，直接用词表里普通 token `decision`（id 61781）做 marker，训练/推理同样只监督 marker 前一位置——证明 marker 用普通单 token 也能跑通；他们的冒烟检查项就是"loss 下降、marker 对齐正常"。
- Showdown 蒸馏微调在 marker 位置上做 **KL(teacher‖student)（temperature-scaled 受限 softmax）** 并**每样本打乱选项顺序**做数据增强——说明 loss 形式可以换成蒸馏，对齐契约不变。

---

## 4. 社区经验（GitHub issues / HF discussions / 微调衍生模型）

### 4.1 GitHub issues（全部 4 条，经 API 抓取）

| # | 标题 | 状态 | 与微调的关系 |
|---|---|---|---|
| [#1](https://github.com/InternLM/Intern-Decision/issues/1) | Could you add a project license? | open | 无（后来仓库已有 LICENSE Apache-2.0，HF 权重同样 apache-2.0） |
| [#2](https://github.com/InternLM/Intern-Decision/issues/2) | Need more detail about Browser use implementation | closed | 推理/应用，非训练 |
| [#3](https://github.com/InternLM/Intern-Decision/issues/3) | **"Please add lora finetuning features！"**（VoyageWang，2026-10-07） | open，0 回复 | **官方仓库目前只提供语言骨干全参微调路径，LoRA 需求未被满足**；想 LoRA 得自己基于 peft==0.14.0 或参考社区先例 |
| [#4](https://github.com/InternLM/Intern-Decision/issues/4) | 贡献请求（Embodied Navigation 方向"InternDecision-Nav"） | open | 社区想扩领域，暂无落地 |

**结论：没有官方回复过的微调踩坑记录。** 训练相关的一手坑全部要从官方源码里读（第 2、3 节已覆盖）。

### 4.2 HF 讨论区（0.8B 3 条）

[discussions API](https://huggingface.co/api/models/internlm/Intern-Decision-0.8B/discussions)：① HF 员工提 PR 加 `decision-model` tag；② 社区发布 Apple Silicon MLX 移植（`pip install intern-decision-mlx`，含图像支持）；③ Spaces demo 请求。**无微调踩坑**。

### 4.3 HF 衍生模型检索（search=intern-decision，17 个结果逐一核对）

- **量化/移植类占绝大多数**：`bombdefuser-124` 的 0.8B/2B/4B GGUF、`prithivMLmods/Intern-Decision-4B-GGUF`、`yeaay/Intern-Decision-4B-GGUF`、`FluidInference` 的 0.8B CoreML（[base 移植](https://huggingface.co/FluidInference/intern-decision-0.8b-coreml)）、`novelaide` 的 MLX-5bit/GGUF-Q5、镜像 `AIArchiveInfo/Intern-Decision-0.8B`。
- **唯一的真·微调衍生模型：[FluidInference/intern-decision-0.8b-showdown-coreml](https://huggingface.co/FluidInference/intern-decision-0.8b-showdown-coreml)** —— "Intern-Decision-0.8B, fine-tuned for Pokémon Showdown"。训练配方（模型卡原文摘录）：
  - Teacher：Intern-Decision-4B 自对弈 220 场 `gen9randombattle`，逐 turn 记录 request + 概率向量（6,974 decisions）；
  - Student：**LoRA r=32 打在语言模型所有 linear 层**，1.5 epochs，**KL(teacher‖student) on the temperature-scaled restricted softmax at the `<decision>` marker**，**option order shuffled per sample**，MacBook Pro M5 Pro (MPS) 200 分钟，导出前 merge；
  - 效果：与 4B teacher 的 held-out top-choice 一致率 35% → 80%；战绩 vs random 9-1。
  - 训练/导出代码在 [mobius](https://github.com/FluidInference/FluidInference) 仓库 `models/computer-use/intern-decision-0.8b/showdown/`。
  - 该作者的[基础移植卡](https://huggingface.co/FluidInference/intern-decision-0.8b-coreml)还确认了移植事实：Gated DeltaNet 骨干无法跑 Neural Engine（"the Gated DeltaNet backbone does not run on the Neural Engine"），fp16 下与 fp32 参考概率最大偏差 ~0.006 且 top答案零翻转。
- **没有发现任何 0.8B/4B 的全参微调社区复现**——全参配方目前只有官方一条路径。

### 4.4 同配方社区复刻（LoRA、非 Intern-Decision 权重，但方法/数据/评测完全对齐）

[PyTrio-Jev 决策模型实战（Agentic RL Lab，2026-09-28）](https://kmno4-zx.github.io/agentic-rl-lab/experiments/pytrio-jev/)：

- 直接以 [Intern-Decision 仓库](https://github.com/InternLM/Intern-Decision)为方法参考，在 **Qwen3.5-4B base** 上 LoRA 复刻（原文明示："官方是全参数微调，我们是 LoRA"）；
- 配置：LoRA rank 32（mlp/attn/unembed）、2160 条 × 2 epoch、batch 16、**lr 1e-4**、270 步、CE 权重只落 marker 前一位置；**自定义干净 chat template**（`templates/qwen3_5_jev.jinja`）而非模型自带模板，以去除系统提示词噪声；marker 用普通 token `decision`（id 61781）；
- 校准：120 案例/600 题 calibration 划分上逆温度二分搜 NLL，**T=1.13**，Hard 档 ECE 0.2311→0.1884；
- **最重要的踩坑结论**：Jevbench-Hard 从 base 55.86% 掉到 47.75%——*"训练数据分布和 Hard 分布的不匹配"*（训练集 state 中位 727 字符 vs Hard 1611 字符的長文档多跳推理）；说明决策微调的数据分布比超参更影响泛化；
- 防污染纪律与官方一致：测试集行一律不进训练、温度只在 calibration 划分拟合、跨划分 state 哈希查重。

### 4.5 第三方分析（旁证）

- [orcarouter: Intern-Decision-2B Quiet Drop](https://www.orcarouter.ai/blog/intern-decision-2b-quiet-release)、[Decision 3.0 vs Intern-Decision-4B](https://www.orcarouter.ai/blog/decision-3-0-vs-intern-decision-4b)：社区对发布节奏与同基座竞品的对比分析（营销站内容为主，仅作存在性参考）。

---

## 5. 温度校准纪律（官方 fit/replay 工具用法，可照搬）

来源：README "Selecting temperature" 节、[docs/EVALUATION.md](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/docs/EVALUATION.md) §3/§4/§7、[src/eval/collect.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/eval/collect.py)、[src/eval/fit.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/eval/fit.py)、[src/eval/calibration.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/eval/calibration.py)、[src/eval/replay.py](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/eval/replay.py)、[benchmarks/temperature-presets.json](https://raw.githubusercontent.com/InternLM/Intern-Decision/main/benchmarks/temperature-presets.json)。

### 5.1 三步流水线（README 原命令）

```bash
# ① 收集：T=1.0、单 worker、与评测同后端
python -m src.eval.collect --backend hf --checkpoint "$MODEL_CHECKPOINT" \
  --data /path/to/your/validation.jsonl --output outputs/validation-predictions.jsonl
# ② 拟合：只在 calibration 行上最小化 NLL
python -m src.eval.fit --checkpoint "$MODEL_CHECKPOINT" \
  --predictions outputs/validation-predictions.jsonl --output outputs/calibration
# ③ 重放：对已保存的 T=1 基准概率离线施加温度
python -m src.eval.replay --checkpoint "$MODEL_CHECKPOINT" \
  --baseline outputs/eval-t1 --calibration outputs/calibration/calibration.json \
  --output outputs/eval-calibrated
export CALIBRATION_PATH="$PWD/outputs/calibration/calibration.json"
```

### 5.2 fit 的数学与防护栏（calibration.py 源码）

- **目标**：*"NLL is convex in inverse temperature; bisect its monotone derivative."* —— 对逆温度 β=1/T 做 **100 步二分**，搜索区间硬编码 **T ∈ [0.01, 100]**；断言拟合后 NLL 不得差于 T=1（`raise AssertionError("Fitted calibration NLL is worse than T=1")`）。
- **数据要求**：JSONL 每行带 `validation_role: "calibration"` 或 `"selection"`；**两类行都必须非空且互斥**（fit.py：`raise ValueError("Supply disjoint calibration and selection rows")`）；ID 必须唯一；收集时温度必须为 1.0（`raise ValueError("Collect unscaled (T=1) probabilities")`）；概率必须为正有限值（`raise ValueError("Exact positive probabilities required; collect raw logits if probabilities underflow")`）。
- **工件绑定 checkpoint**：`calibration.json` 记录 checkpoint 内所有 `.json/.safetensors/.jinja` 文件的 SHA256；replay 时逐一复核（`raise ValueError("Checkpoint differs from calibration")`）。README：*"Calibration artifacts are bound to checkpoint metadata hashes; regenerate one after changing weights."*
- 可选 `--expected-fit-rows` / `--expected-validation-rows` 校验行数。
- 产出三个文件：`calibration.json`（工件）、`search-curve.json`（201 点几何网格搜索曲线）、`validation.json`（baseline vs calibrated 的 acc/NLL/Brier/ECE）。

### 5.3 replay 的纪律（replay.py 源码）

- 公式：`softmax(log(p) / T)`，只作用于**已保存的候选概率**，CPU 离线执行；
- **零决策改变校验**：逐字段比对 argmax，`raise ValueError("Calibration changed a benchmark decision")`；聚合准确率也必须不变；产出 `verification.json` 要求 `passed: true, changed_predictions: 0`；
- 校准与基准评测**后端必须一致**（`raise ValueError("Calibration and benchmark backend differ")`）；baseline 必须是同一数据的 T=1 评测（哈希核对）。

### 5.4 官方红线（README/EVALUATION 原句）

- *"Do not reuse a temperature from another checkpoint or treat a sampling temperature as calibration."*
- *"Fit one positive scalar per checkpoint by NLL on calibration data only."* / *"Only `calibration` rows fit T; `selection` rows report validation metrics and never affect fitting."*
- *"Never feed either bundled test set to the fitting command."* / presets JSON 第一句：*"Previously fitted evaluation temperatures; do not fit on bundled test sets."*
- *"NLL-optimal temperature need not minimize ECE on another dataset."*
- *"It does not optimize benchmark ECE or select checkpoints."*（replay 不选 checkpoint）
- 训练数据、私有校准/验证划分不发布（DATA.md：*"本文不描述模型训练数据的来源、组成、数量或配比"*）。

### 5.5 官方预设温度（照搬时的参照值）

| 模型 | T | 拟合规模 | 后端 |
|---|---:|---|---|
| Intern-Decision-0.8B | **2.747760550702957** | 1,728 校准案例 + 1,693 验证（[模型卡](https://huggingface.co/internlm/Intern-Decision-0.8B/raw/main/README.md)：*"fitted separately for this checkpoint by NLL minimization on 1,728 designated calibration cases, with 1,693 separate validation cases"*） | xtuner |
| Intern-Decision-2B | 2.100509348277736 | 未披露 | xtuner |
| Intern-Decision-4B | 1.9924182353655278 | 未披露（*"a separate calibration split"*） | xtuner |

用 `python -m src.eval.use_temperature --model intern-decision-4b --checkpoint ... --output .../calibration.json` 可把预设按哈希绑定到本地 checkpoint（EVALUATION.md §3：*"The command checks inference metadata and weight-shard hashes, then writes a calibration artifact bound to your local path. Hash mismatches are errors."*）。模型越小 T 越大（0.8B 概率最"过自信"），自训模型应预期 T 在 1~3 区间。

---

## 6. 可直接抄进训练脚本的配置项清单

> 全部来自 configs/training/qwen35.py + scripts/train.sh 原文；`[]` 内为环境变量可覆盖项。

**优化**
- optimizer: AdamW（`foreach=False`）
- lr: `2e-6`（env `LR`）；weight_decay: `0.01`（`WEIGHT_DECAY`）
- scheduler: cosine，warmup_ratio `0.03`（`WARMUP_RATIO`），lr_min `2e-7`（`LR_MIN`）
- epochs: `2`（`TOTAL_EPOCH`）；global_batch_size: `16`（`GLOBAL_BATCH_SIZE`）；sp_size 1；seed 42

**Loss**
- 全词表 CE，仅 marker 前一位置监督（label 写 marker 位、collator 右移一位）
- chunk 模式：`LOSS_MODE=chunk`，`LOSS_CHUNK_SIZE=1024`，`LOSS_REDUCTION=token`

**长度与 packing**
- `SAMPLE_MAX_LENGTH=8192`（超长报错不截断）；`PACK_MAX_LENGTH=8192`
- pack_level `soft`；pack_to_max_length True；global_pack True；group_by_length True；pack_chunk_size 10000；pack_extra_buffer_size 20；pack_workers 4；dataloader workers 4
- 多模态：llm_pack_weight 1.0、visual_pack_weight 0.0、max_pixels 16384×32×32、视频帧 24

**模型与显存**
- freeze: vision ✻、projector ✻、language ✤（全参）；`only_llm_forward=无 MEDIA_ROOT 时`
- FSDP：`RECOMPUTE_RATIO=1.0`（全量重计算）、`VISION_RECOMPUTE_RATIO=0.0`、preserve_rng_state False
- bf16；torch_compile 关（`MODEL_COMPILE=0`）
- `HFSaveCfg(fp32_keys_pattern=None)`（fp32 norm 参数留在普通分片路径）
- non-reentrant activation checkpointing（替换 XTuner 默认 reentrant）
- 保存策略：只存最终 HF checkpoint（interval 全 None、auto_resume=False）
- packed causal conv：FP32 累加 + seq_idx 跨样本屏蔽（必须移植）
- 纯文本 batch 参与冻结视觉 FSDP collective（假像素、输出丢弃）

**版本锁定（训练环境）**
- torch 2.6.0+cu126、torchvision 0.21.0
- XTuner：InternLM/xtuner @ `fb51baebdf91b03bd39255c4427edac9aca82865`（--no-deps -e 安装）
- flash-attn 2.8.3（匹配 torch/CUDA 工具链）、flash-linear-attention 0.4.2
- transformers 4.57.0、deepspeed 0.17.5、peft 0.14.0、mmengine 0.11.0rc2
- PYTHONPATH 注入 `runtime/xtuner`（Torch2.6 兼容垫片）

**启动环境变量（train.sh 全集）**
- `XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0`（**FA3 必须关**）
- `XTUNER_ACTIVATION_OFFLOAD=0 XTUNER_GC_ENABLE=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline`
- `DS_BUILD_OPS=0 DS_BUILD_FP_QUANTIZER=0`
- `WEIGHT_AUDIT=$WORK_DIR/weight-audit`；`NPROC_PER_NODE`（默认 4）
- WORK_DIR 必须是新目录；训练前后跑 `scripts.training_config`（记录配置+哈希）与 `scripts.verify_checkpoint`（权重审计）

**数据契约（docs/DATA.md）**
- JSONL：`id/state/questions/images/targets`；type ∈ choice|score|noul；选项按插入顺序映射 A-Z a-z 0-9（≤62，均需单 token）；`targets[f].label` 只是标签不进 prompt；校准行加 `validation_role`
- `<decision>` 训练时 `tokenizer.add_special_tokens` 注入；骨架 `json.dumps(..., indent=4)`；证据文本含 `<decision>` 即报错

**温度校准（照搬 collect→fit→replay）**
- collect：T=1.0、workers=1、与评测同后端
- fit：逆温度二分 100 步，T∈[0.01,100]，calibration/selection 行都非空且互斥，工件绑定 checkpoint 文件哈希
- replay：`softmax(log(p)/T)`，验证 changed_predictions==0；参考预设 T：0.8B 2.7478 / 2B 2.1005 / 4B 1.9924

---

## 附：本报告引用的全部一手链接

- 仓库与 README：<https://github.com/InternLM/Intern-Decision>
- 训练配置：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/configs/training/qwen35.py>
- 启动脚本：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/scripts/train.sh>
- 配置记录：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/scripts/training_config.py>
- 数据契约：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/docs/DATA.md>
- 评测复现：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/docs/EVALUATION.md>
- 依赖：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/requirements-xtuner.txt> / <https://raw.githubusercontent.com/InternLM/Intern-Decision/main/requirements-inference.txt>
- 掩码编译：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/inputs/masked.py> / <https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/inputs/schema.py>
- 模型适配：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/model/qwen.py> / <https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/model/size_config.py>
- Torch 兼容垫片：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/runtime/xtuner/sitecustomize.py>
- 校准工具：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/src/eval/calibration.py> / `fit.py` / `collect.py` / `replay.py`
- 温度预设：<https://raw.githubusercontent.com/InternLM/Intern-Decision/main/benchmarks/temperature-presets.json>
- HF 0.8B：<https://huggingface.co/internlm/Intern-Decision-0.8B>（模型卡 / config.json / safetensors index / discussions）
- HF 4B / 合集：<https://huggingface.co/internlm/Intern-Decision-4B> / <https://huggingface.co/collections/internlm/intern-decision>
- GitHub issues API（全 4 条）：<https://api.github.com/repos/InternLM/Intern-Decision/issues?state=all&per_page=50>
- 社区微调先例：<https://huggingface.co/FluidInference/intern-decision-0.8b-showdown-coreml> / <https://huggingface.co/FluidInference/intern-decision-0.8b-coreml> / <https://kmno4-zx.github.io/agentic-rl-lab/experiments/pytrio-jev/>
- 第三方分析：<https://www.orcarouter.ai/blog/intern-decision-2b-quiet-release> / <https://www.orcarouter.ai/blog/decision-3-0-vs-intern-decision-4b>
