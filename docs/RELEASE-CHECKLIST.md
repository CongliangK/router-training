# 开源发布清单（Release Checklist）

状态：准备工作已完成（2026-10-14）。按下列步骤执行**实际发布**。

## 已完成的准备

- [x] `LICENSE`：Apache-2.0（附录署名 router-training contributors）
- [x] `NOTICE`：上游 Intern-Decision-0.8B（Apache-2.0, Shanghai AI Laboratory）署名与数据不发布声明
- [x] `README.md`（英文公开版）+ `README.zh-CN.md`（中文公开版），原私有 README 已被替换（原版在 git 历史与 `docs/LEGACY-README.md` 中）
- [x] `AGENTS.md`：面向 AI 编码代理的仓库工作规范
- [x] `docs/DATA.md`：数据 schema、合成样例说明、自建数据集指南
- [x] `data/synthetic-examples/`：三条任务示例 + 标签示例 + 合成增广示例（全部虚构）
- [x] `scripts/export_public.sh`：清洁导出脚本（白名单复制 + 泄露守卫）
- [x] `.gitignore`：权重/日志/预测文件忽略规则

## 关键约束：git 历史含隐私数据

**本仓库（私有）的 git 历史包含真实任务 prompt 与标签。** 绝不能直接
`git push` 本仓库、任何分支或 `git archive`。公开仓库必须用
`scripts/export_public.sh` 从工作区文件重建全新 git 历史：

```sh
bash scripts/export_public.sh            # 生成 ../router-training-public
cd ../router-training-public
# 人工抽查内容（重点：data/、docs/、models/ 许可证文件齐全）
git init && git add -A && git commit -m "Initial public release"
```

## 发布前人工复核（必做）

- [ ] 通读导出目录全部 `.md`，确认无个人路径/项目名/内部服务名残留（导出脚本会 grep 提醒，但需人工确认）
- [ ] 确认导出目录中 **没有** `data/*.jsonl`（真实数据）、`data/labeling/batch-*`、`data/labeling/labels-*`、`data/synth/`（真实合成集）、`artifacts/**/*.jsonl`（预测文件含 prompt）
- [ ] `artifacts/**/report.json`、`calibration.json`、`quality-gate.json`、`data/split-report.json` 仅含聚合指标与哈希，可发布（已复核其结构；仍建议再扫一眼）
- [ ] `models/intern-decision-0.8b/` 中无 `.safetensors` 等权重，且 LICENSE / LICENSE-QWEN / README 在位
- [ ] 公开仓库跑 `python -m unittest discover -s tests -v`：实测 16/19 通过；3 个失败均为**预期**（依赖被刻意排除的私有文件 `data/train.jsonl` 与 `snapshot-manifest.json`）。发布时可接受，后续可将这些用例改为无文件时 skip
- [ ] `environment.json`、`snapshot-manifest.json` 未导出（含本机路径与源位置）

## 待定决策（发布时或发布后）

- [ ] GitHub 仓库名定为 `router-training`，填写 About/Topics（llm-routing, fine-tuning, Intern-Decision, calibration）
- [ ] 是否附 v3c checkpoint 发布（Hugging Face，权重须继承上游 Apache-2.0 义务；当前脚本默认不含权重）
- [ ] 引用方式（CITATION.cff）是否需要
- [ ] 路由表 `routes.json` 中 provider 名（zai-coding-cn / openai-codex）是否保留或匿名化为 provider-a/b
