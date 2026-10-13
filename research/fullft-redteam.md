# 全参微调计划红队审查

审查对象：`train_eval.py --fullft` 路径 + 评估管线 + 部署路径（server.py / model_utils.py）。
所有数字均已在本仓库实测验证（模型 473 个权重张量、200 条合成语料、test/calibration 切分、正在运行的 v3 日志）。
按严重度排序：**P0 = 会直接导致训练失败或得出误导性结论；P1 = 显著削弱结论可信度；P2 = 值得修但非致命。**

## 结论摘要（前五大风险）

| # | 严重度 | 攻击点 | 一句话后果 |
|---|--------|--------|-----------|
| 1 | P0 | 纯 bf16 全参训练 + AdamW 无 fp32 master weights，lr 2e-5 的单步更新量小于 bf16 量化步长 | 大部分权重更新被舍入归零，全参微调"假性无效"，得出"全参不如 LoRA rank 8"的错误结论 |
| 2 | P0 | 校准集被三重自适应复用（选 epoch / 拟温度 / τ 扫描）且 --synth 后 40 条合成语料混入，vision 正例中 91% 是合成 | 温度与 τ 操作点拟合在合成捷径上，test 指标不可归因，部署操作点错位 |
| 3 | P0 | server.py 的 load_engine 只走 PeftModel(adapter) 分支；full-checkpoint 目录不含 tokenizer/processor | 全参产物根本无法部署，服务启动即崩，或任何"部署验证"静默跑在旧 LoRA 上 |
| 4 | P1 | test 104 行仅 7 个独立 group，单一 group 占 76%（含 95% 的 vision-yes、97% 的 flagship），行级 bootstrap 忽略聚类 | CI 严重反保守；fast 类仅 2 例，macro-F1 被噪声主导 |
| 5 | P1 | LoRA vs 全参对照不可归因：lr 未按方法扫描、单 seed、数据混合不同步、epoch 选择规则不对称 | 无法把差异归因于"容量"，只能归因于"整套超参" |

---

## P0-1 纯 bf16 全参训练：更新下溢，制造"全参无效"的假阴性

**失败场景。** `HFBackend.__init__` 以 `dtype=bfloat16` 载入模型（`models/intern-decision-0.8b/inference.py` L202），`train_eval.py` L143 直接对该 bf16 模型用 `torch.optim.AdamW` 训练——没有 autocast、没有 fp32 master weights、没有 stochastic rounding。AdamW 的 `exp_avg`/`exp_avg_sq` 与参数同 dtype（bf16，8 位尾数）。Adam 单步更新量 ≈ `lr × m/√v ≈ lr = 2e-5`；而典型 transformer 权重 |w|≈0.01–0.03，bf16 相对量化步长 2⁻⁸ ≈ 0.4%，即绝对步长 ≈ 8e-5–1.2e-4。**更新量比量化步长小 3–6 倍，多数权重更新直接舍入为零。** 这正是混合精度训练要求 fp32 master weights 的原因（[HF Transformers 混合精度文档](https://huggingface.co/docs/transformers/v5.10.1/en/mixed_precision_training)、[NVIDIA NeMo mixed-precision 说明](https://docs.nvidia.com/nemo/automodel/nightly/development/mixed-precision-training)）。更阴险的是：LoRA 对此免疫——B 矩阵初始化为 0，任何更新相对 0 都是"大"的，所以 LoRA 在同一套 bf16 管线里 loss 正常下降（v2: 0.81→0.48；v2e6: →0.15）。**结果就是：全参跑完 loss 几乎不动或动得很慢，团队得出"0.8B 全参打不过 rank-8 LoRA"的结论，而真实原因是数值机制不同。** 旁证：v2e6 的 calibration NLL 在 epoch 3 后剧烈非单调（0.47→0.98→0.88），说明该管线在低精度下本就不稳。

**检测信号。**
- 训练前后 `sum((w_after-w_before)**2)`：按张量对比 bf16 应有的"格子化"差异；若 90%+ 可训练张量逐比特不变，实锤下溢。
- 每 epoch 记录 `||Δw||₁/||w||₁` 与"更新被舍入为零的比例"（`w + u == w` 的元素占比，训练中直接统计）。
- 消融对照：同 seed 同数据跑 200 步 fp32 模型（0.8B fp32 ≈ 3.2GB 显存，RTX 4080 16GB 可行），比较 loss 曲线斜率。
- 训练中打印 symbol logits 的原始幅值（不是 softmax 后）：若幅值漂移但精度不动，是表征在漂而权重被锁死的信号。

**预防与修复。**
- 首选：fp32 master weights + bf16 前向，即模型以 fp32 载入、`torch.autocast('cuda', bfloat16)` 包住 `field_logits` 的 forward（bf16 无需 GradScaler）；或参数 fp32、优化器状态 fp32，只在前向 cast。
- 备选（显存不够时）：AdamW 换 bitsandbytes 8-bit Adam（状态量化但 master 仍 fp32），或降低要求用 fp16+GradScaler。
- 最廉价兜底：保持 bf16 但把 fullft lr 提到 1e-4 量级并加 warmup，使单步更新 ≥ 量化步长——不推荐作为正式方案，只用于快速验证"数值假设"是否成立。
- 在 report.json 里记录 `dtype_policy` 字段，防止未来对比时忘记两种方法跑在不同数值机制下。

---

## P0-2 校准集三重自适应复用 + 合成语料污染温度/τ 拟合

**失败场景。** 同一批 95 行真实校准数据（`--synth` 后 = 95 真实 + 40 合成 = 135 行）被串行用于三次自适应选择：① `run()` L184-193 按校准集 tier macro-F1 选 best epoch；② `fit_temperatures` 在同一批上网格搜 10 个温度（NLL 最小）；③ τ logit 调整扫描（L214-222）再在同一批上按 macro-F1 + 约束选 τ。三重 winner's curse 叠加在 135 行上，选出的"操作点"（epoch, T, τ）是对这 135 行的过拟合，部署后必然退化。合成污染的实测数字更糟：
- 40 条合成校准行里 vision=yes 有 21 条，而真实校准集 vision=yes 只有 2 条——**vision 温度在 23 个正例中 91% 拟合于合成样本**。合成样本的标签规则带词汇捷径（"截图""PNG""肉眼比对""禁止 DOM diff"），拟合出的温度对真实流量必然错位。
- 真实校准集 task_type 缺 visual 类（v2 报告显示"missing classes; retained checkpoint temperature"）；混入合成后 7 类凑齐（visual 6 条），task_type 温度从"保留基线 2.748"切换为"本地拟合"，而拟合样本以合成口味为主——**一个超参的语义在 --synth 开关前后 silently 改变**。
- τ 用的 tier prior 取自含合成的训练集（balanced 0.768 / flagship 0.130 / fast 0.102，与纯真实的 0.763/0.143/0.094 不同），τ·log(prior) 的校正量随数据混合漂移。
- τ 约束"flagship recall ≥ 0.75 / FP rate ≤ 0.10"在仅 ~17 个校准 flagship 上判定（真实 15 + 合成 2），recall 粒度 1/17 ≈ 5.9pp，约束是否满足基本是掷硬币；且 L220-221 在无可行解时**静默回退到无约束 argmax**，不报警。

**检测信号。**
- 消融：只用 95 行真实校准重拟 T 和 τ，与混合版对比——若 τ 或 T_vision 相差一档以上、或 test 指标差 >2pp，污染实锤。
- 报告"约束在 test 上的违反率"：chosen τ 在 test 上重算 recall/FP rate 是否仍满足 0.75/0.10。
- 校准集上留一法（k-fold）选 τ，看选中 τ 的方差：方差大 = 操作点未定。

**预防与修复。**
- 数据三分：train / early-stop（选 epoch）/ calibration（T 与 τ），禁止同一批数据承担两次以上选择职责；行数不够就 k-fold 交叉拟合并取中位数 τ。
- `--synth` 时把合成行挡在"选 epoch + 拟 T/τ"之外（或单独报告两套操作点：real-only 与 mixed），合成只进 train。
- τ 扫描无可行解时显式 fail（写入 report 并置 quality_gate 理由），不要静默回退。
- 报告里区分"合成参与的指标"与"纯真实指标"，并把 `temperatures`/`tier_adjust` 的拟合来源（行数、正例数、合成占比）写进 calibration.json。

---

## P0-3 部署断层：server.py 无法加载全参产物

**失败场景。** `server.py` L25 写死 `load_engine(artifacts/'adapter')`；`model_utils.load_engine`（L31-38）只有 `PeftModel.from_pretrained` 一条适配器分支。全参 run 的产物目录只有 `full-checkpoint/`，没有 `adapter/`——**服务 lifespan 启动即抛异常，全参模型完全无法上线**。即便手工把 run 指到 full-checkpoint，还有第二层坑：`save_pretrained` 只写权重 + config，**不写 tokenizer.json / chat_template / preprocessor**，而 `HFBackend` 要求从 checkpoint 目录取 tokenizer 且必须含 `<decision>` 标记（L196-198）——直接指过去会在 `AutoTokenizer` 或 marker 检查处报错。第三层：训练结束的"恢复 best"路径（train_eval.py L199-201）在 CPU 上再实例化一整份模型取 state_dict，峰值 RAM 翻倍（~3.4GB），且 strict `load_state_dict` 依赖 tie_word_embeddings=True 的重绑行为，属于脆弱但当前可用的链路。另外运维现实：**v3 训练正在占用 10.4/16GB 显存**，此刻并发启动全参会 OOM。

**检测信号。**
- 冒烟测试：对每个 run 目录断言 `adapter/` 或 `full-checkpoint/` 与 server 期望一致；server 启动后 `/health` 必须返回该 run 的 checkpoint 类型。
- 一条集成测试：从 `full-checkpoint` 冷启动 server → 打一条 `/route` → 比对 logits 与训练侧 finetuned-test.jsonl 一致（同一 prompt 同一温度/τ 下 choice 必须相同）。

**预防与修复。**
- `load_engine` 增加 `kind` 参数：`adapter` 走 PeftModel；`full` 走 `Qwen3_5ForConditionalGeneration.from_pretrained(full_ckpt)` + tokenizer 固定从 `models/intern-decision-0.8b` 读取（或在保存时把 tokenizer/processor 文件一并拷入 full-checkpoint，使其自包含）。
- server.py 从 run 目录自动探测产物类型，并在 calibration.json / quality-gate.json 与 checkpoint 类型不一致时拒绝启动。
- 训练末尾的 best 恢复改为增量：只保存可训练参数的 state_dict（语言侧 ~1.4GB bf16），或直接在内存中保留 best 权重的 CPU 副本，避免二次 from_pretrained。
- 全参训练排队等 v3 结束再启动（显存争用实测 10.4/16GB 已被占）。

---

## P1-4 test 集统计功效：7 个独立组，单组占 76%，行级 bootstrap 反保守

**失败场景。** test 104 行来自 7 个 group（split-report），实测组成：group `ec267d56…` 一个组就有 79 行（76%），包含 74 个 vision-yes 中的 70 个、30 个 flagship 中的 29 个；`fast` 类全局只有 2 行且分属两个单人组。后果：① 行级 bootstrap（`bootstrap_ci` L64-70）假设行间可交换，但同组行共享同一根任务，真实独立单元 ≈ 7，**现有 CI 至少乐观 2–4 倍**；② macro-F1 中 fast 类的 F1 由 2 个样本决定，单个 FP/FN 摆动 macro-F1 ±0.15；③ "vision recall" 实际是"在 ec267d56 这一个根任务上的 recall"，报告却写成泛化指标。flagship n=30 时 recall=0.9 的 Wilson 95% CI 约 ±11pp——任何"全参比 LoRA flagship recall 高 8pp"式的结论都在噪声内。

**检测信号。**
- 用组级（cluster）bootstrap 重算现有 CI：按 7 个 group 重采样，观察 CI 宽度膨胀多少倍。
- 报告 per-group 指标表：若去掉 ec267d56 后结论翻转，结论即绑定于单根任务。
- per-class n 打进报告（现在 confusion 里有，但摘要没引用）。

**预防与修复。**
- 所有 CI 换成按 `split_group` 重采样；macro-F1 加 CI。
- 摘要指标区分"主组指标"与"去主组指标"；fast 类单独标注 n=2、不参与显著性结论。
- 若要真正回答视觉路由，需要第三个含 vision 正例的独立根（split-report limitations 已自认只有两个根），或明确把 vision 结论降级为"单根任务上的观察"。
- 方法间比较用配对检验（同一 test 行上的 McNemar / 配对 bootstrap），而不是两个独立 CI 是否重叠。

---

## P1-5 LoRA rank 8 vs 全参：对照实验不可归因

**失败场景。** 现有基线（v2: 2 epochs、v2e6: 6 epochs，rank 8，attention-only，lr 1e-4，**纯真实数据**）与计划中的全参（lr 2e-5，--synth 与否未定）之间至少有五个未受控变量：① lr 未按方法扫描——2e-5 对 0.8B 全参偏保守、1e-4 对 rank-8 LoRA 是常规值，若全参在 5e-6–1e-4 区间有更优点，单点比较会误判；② 数值机制不同（见 P0-1，bf16 舍入对 LoRA 免疫、对全参致命）；③ 数据混合不同——v3 已带 --synth 在跑，若全参也带 --synth 而对照对象是 v2/v2e6，"容量收益"与"合成数据收益"完全混同；④ best-epoch 规则相同但选择噪声不同（全参每个 epoch 波动更大），best-vs-best 比较放大双方选择偏差；⑤ 单 seed（20261010），629 行/epoch × 6 epoch 的全参训练方差未被测量。结论"全参优于/劣于 LoRA rank 8"在这套设计下**不可归因到容量**。

**检测信号。**
- 每种方法至少 lr 3 点 × seed 2–3 个；若两种方法的最优 lr 区间不同但各自只跑了一个点，即不可比。
- 训练方差：同配置换 seed 重跑，看 test macro-F1 极差；若极差 > 方法间差，比较无意义。
- 报告必须列出：trainable params（LoRA 1,794,048 vs 全参 ~7.7 亿语言侧）、数据混合、epoch 选择结果、数值策略，缺一不可。

**预防与修复。**
- 固定一个"数据混合协议"（建议：LoRA 与全参各跑 --synth 与否两版，4 个 cell），比较只在同列内做。
- lr 网格 {5e-6, 1e-5, 2e-5, 5e-5}（全参）/ {5e-5, 1e-4, 2e-4}（LoRA），先小 epoch（2）筛再全量跑。
- 结论措辞降级为"在该 lr、该数据混合、该 seed 下"，除非做了网格 + 多 seed。
- 若资源不够，至少保证：同数据、同 epoch 数、同选择规则、同 seed，加上 P0-1 的数值修复——这是"容量归因"的最低门槛。

---

## P1-6 双重类校正叠加：round-robin 采样 × effective-number 权重 → 过路由与视觉头崩坏

**失败场景。** 训练同时开了两层不平衡校正：round-robin 按 tier 均匀出样（L160-167）**加上** effective-number 损失权重（β=0.999）。实测 --synth 后权重为 balanced≈0.25 / flagship≈1.25 / fast≈1.50——采样已经把频率拉平，权重再把梯度质量比拉开 ~5–6 倍，双重校正等效于把稀有类信号放大一个数量级。证据已在基线里：v2 出现 balanced→flagship 20 个 FP（过度路由）。更严重的是 vision 权重：真实 train vision 计数 464 no / 4 yes，effective-number 归一后 **w(no)≈2×10⁻⁵，w(yes)≈2.0**——"no"类监督几乎被清零，而"yes"只有 4 个真实样本（+合成 ~85 个），结果 v2/v2e6 的 vision FN rate 86–89%（视觉头灾难性崩坏，且这两个 run 都没开 --synth）。全参容量更大，对这种失衡目标的"忠实执行"只会更极端。

**检测信号。**
- 训练日志已打印权重（L154）——把 w(no)≈2e-5 这种极端值本身当告警：<1e-3 的权重意味着该类监督事实消失。
- 每 epoch 结束在校准集上打混淆矩阵：flagship FP 数与 vision FN 数同时监控（现在只进 report 不进决策）。
- 消融：round-robin 与权重二选一，跑 2 epoch 对比。

**预防与修复。**
- 两层校正只留一层：用 round-robin 采样时把权重退回温和版本（如 sqrt 逆频或 β=0.9），或用权重时改随机采样。
- vision 权重设下限（如 clamp 到 ≥0.1），避免"no"类监督归零；'uncertain' 行（真实 1+2+3 行、合成 11 行）目前直接跳过损失，保持跳过但需在报告写明。
- 权重基于的计数来源（真实 vs 真实+合成）要固定并记录，否则跨 run 不可比。

---

## P1-7 lr 2e-5 恒定无 warmup/decay，总更新步数 ~470

**失败场景。** 全参 lr 硬编码 2e-5（L143），无 warmup、无 decay、无 schedule。总步数：629 行/epoch ÷ accumulation 8 ≈ 79 步/epoch，6 epochs ≈ 474 步——而 AdamW 默认 β₂=0.999 的二阶矩记忆窗口是 1000 步，**整个训练都处在 v 估计未收敛区**，早期步的 update 噪声大；配合无 warmup，第一个 epoch 的类加权大梯度（flagship 权重 1.25×早期高 CE）直接全参更新主干。对 0.8B 这不算必死，但与 P0-1 的舍入问题、P1-6 的失衡目标相乘后，早期几步就可能把决策头附近表征推歪再难回来（v2e6 的 NLL 尖峰是同一不稳性的 LoRA 版本）。warmup/decay 是全参 SFT 的标准配置（参见 [SFT warmup/checkpoint 课程](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning/blob/main/lessons/mod-103-supervised-fine-tuning/03-learning-rate-warmup-and-checkpoint-cadence.md)）。

**检测信号。** 记录每步梯范数（clip 前后）与 update/param 比值曲线；若前 20 步梯范数比均值高 5×以上，说明缺 warmup 正在放大早期噪声。

**预防与修复。** linear warmup 10–20 步（约 1/4 epoch）+ cosine 衰减到 10%；或最低限度把前 10 步 lr 线性从 0 升到 2e-5。若做了 P0-1 的 fp32 修复，此条从"致命组合因子"降为"稳健性改进"。

---

## P2-8 "决策头格式遗忘 / <decision> 标记位漂移"的真实机制（一半被证伪）

**失败场景。** 先证伪一半：`<decision>` 标记位由 tokenizer 在输入中的位置决定（`encode` 里 `marker_id` 的 nonzero 索引减一，inference.py L232-234），训练与推理走同一条 encode 路径，且从不自回归生成——**标记位在结构上不可能漂移，"格式崩坏"这一担忧不成立**。真实的风险是软性的：全参把 `embed_tokens`（含 `<decision>`、符号 A/B/C… 的 embedding）一并解冻训练，而 tie_word_embeddings=True 意味着 lm_head 与 embed 共享同一张量——代码注释声称"keep lm_head frozen"，但 `named_parameters()` 去重后 `lm_head.weight` 根本不在列表里，**输出头实际上随 embed 一起被训练**（与注释相反，对本任务反而有利，但属于未声明的行为）。损失只施加在 marker-1 位置的符号 logits 上，其余位置的表征无任何正则（无 KL-to-base 锚），全参可能在非监督位置发生功能性漂移——当前评估路径不经过那些位置，漂移不会被指标发现，直到上游改 prompt 模板/系统提示才暴露。

**检测信号。**
- 每 epoch 计算各 field 在 marker-1 位置的"符号集概率质量"（正确符号 A/B/C 之外全词表的概率和）：从 ~0.99 掉到 <0.9 即表征漂移。
- 训练前后对比：同一批固定 probe prompt 的全词表 top-10（不只符号位），看非监督位置输出是否剧变。
- 检查 `<decision>` embedding 与训练前的余弦距离随 epoch 的变化。

**预防与修复。**
- 冻结 `embed_tokens`（或单独给 embedding 一个 10× 更小的 lr），既防漂移又省显存；符号语义由训练数据已经覆盖。
- 若坚持全解冻：加轻量 KL-to-base 正则（对 marker-1 位置的符号分布）或 EWC；至少保留 symbol-mass 监控。
- 修正代码注释（lm_head 实为 tied 可训练），避免误导后续维护。

---

## P2-9 best-epoch 全量 1.6–1.7GB checkpoint 的磁盘与 IO 问题

**失败场景。** 每个"新最优"epoch 都 `save_pretrained(output/'full-checkpoint')`（L191）全量覆盖写：473 个张量、~1.7GB，**其中 ~0.9GB 是被冻结的视觉塔与 projector——纯属重复搬运**。风险面：① 同目录覆盖写非原子，进程在写分片途中被杀会留下损坏的 best checkpoint，且 run 末尾的恢复逻辑会 from_pretrained 它（虽会响亮报错，但整个 run 作废）；② 每个 improving epoch 一次 10–30 秒级串行写停顿，6 epochs 最坏 ~10GB IO；③ 多 run 累积（当前磁盘 578GB 空闲，短期无虞，长期无清理策略）；④ run 末尾再在 CPU 实例化第二份完整模型取 state_dict（见 P0-3）。对照：LoRA adapter 只有 ~7MB，同样的代码路径对 LoRA 无感，对全参每个改善 epoch 都是一次全量序列化。

**检测信号。** 监控 run 目录大小与单次 save 耗时；`best-epoch.json` 的 mtime 与 report 的 `seconds` 字段差值暴露 IO 占比。

**预防与修复。**
- 只保存语言侧可训练参数（`state_dict={k:v for k,v in model.state_dict().items() if requires_grad}`，恢复时 load 进已载好基座的模型），体积从 1.7GB 降到 ~1.4GB 且避免复制冻结权重；或保存与基座的 diff。
- save 用 tmp 目录 + 原子 rename；run 结束时校验 checkpoint 可加载后再删 tmp。
- 加 `--keep-last-k` 或定期清理脚本；report 里记录 checkpoint 字节数。

---

## P2-10 零散但真实的缺陷

- **τ 扫描静默违反约束**：`feasible` 为空时直接取全局 argmax（L220-221），选出的 τ 可能在校准集上就违反 recall≥0.75 / FPR≤0.10，但 report 只记 sweep 不记违反。修复：无可行解时置 quality_gate 失败理由。
- **`field_logits(..., train=True)` 是死参数**：函数体从不使用 `train`，暗示曾经计划 autocast/训练态分支但没实现——与 P0-1 是同一个缺口的两面。
- **截断路径非 token 幂等**：`request_for` 超长时 decode→重 encode（model_utils L48-49），往返不保序，且 `[ROUTING_CONTEXT_TRUNCATED]` 是杜撰串会被普通分词。当前数据 0 截断（max 3284 < 7700）处于休眠状态，但任何未来长上下文评估的行数统计会失真。
- **weight_decay=0.01 无差别施加**于所有可训练参数（含 embedding、LayerNorm），全参下这比 LoRA 下影响更大；标准做法是 decay 白名单。
- **单条 batch=1 训练**：629 步/epoch 的序列化 forward（无 padding 批处理）使全参 6 epochs 预计数小时——不是错误，但意味着 lr 网格 × 多 seed 的修复方案需要按此估时。

---

## 已验证的非问题（避免误伤）

1. **子串冻结名单审计通过**：`frozen={'visual','projector','lm_head'}` 的子串匹配经对 473 个张量全量核对，恰好覆盖全部 153 个视觉/投影张量，无误伤语言侧参数——`vision 参数名不含 'visual' 导致误训`这一担忧不成立。
2. **标记位结构稳定性**：如 P2-8 所述，位置由 tokenizer 决定，训练不改变分词，不存在"标记位漂移"的结构性路径。
3. **save→reload 的 strict load_state_dict**：因 tie_word_embeddings=True，from_pretrained 重绑后 state_dict 键完整，当前实现大概率可用（仍建议冒烟测试一次）。
4. **梯度检查点 + 累积交互**：`use_reentrant=False` 与逐样本 backward/按 8 步累积的组合正确，epoch 尾部残余样本也被 `i+1==len(rows)` 兜住；未发现正确性缺陷。

## 建议的最小行动集（按序）

1. 修 P0-1（fp32 master weights / autocast）——否则全参实验无论结果如何都不可信。
2. 修 P0-3 的 load_engine 分支 + 全参 run 的服务冒烟测试。
3. 校准协议三分 + 合成不进 T/τ 拟合（P0-2）。
4. 组级 bootstrap + per-class n 进报告摘要（P1-4）。
5. lr 网格 + 双 seed 的 LoRA/全参同数据对照（P1-5），结论措辞按实际控制变量降级。
6. 等 v3 训练结束再启动全参（显存 10.4/16GB 已被占用）。
