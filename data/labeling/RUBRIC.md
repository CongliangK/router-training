# 委派任务标注标准：router-label-v2（#18，公开 CPU 子集）

本标准只适用于**显式选择** `router-label-v2` 的新标签版本。旧
`router-label-v1` / frozen-v1 的七分类标签、问题模板、checkpoint 和指标不改写；
没有版本声明的历史产物只能按 v1 解读，不能自动迁移。

只依据任务及已获授权的上下文证据，不看路由器预测、具体模型名或历史选模。
真实复核必须另行授权；本文与合成例子不证明真实标注已复核。

## tier：不对称默认

- **balanced 默认**：按明确要求实现、修改、局部排错、常规审查、研究综合、
  内容创作的常规执行。称职工程师按步骤即可完成，不需先发明方案。
- **fast 只给机械操作**：提取、格式转换、枚举、简单摘要，几乎不需判断。
- 写代码、执行命令、提示词很长、需要视觉，**本身均不是升级理由**。
- **flagship 仅限以下五项，rationale 必须点明一项与具体依据**：
  1. 跨模块/系统架构设计，多个合理方案须权衡；
  2. 无现成路径或参考答案的原创方法/创造性方案；
  3. 症状与病因跨模块分离，含并发/时序/生命周期的复杂根因；
  4. 安全/并发正确性存疑，常规测试难发现的漏洞、竞争或不可逆损坏；
  5. 对高影响产出的对抗性终审，主动寻找作者盲点。

## needs_vision：仅 yes / no（不含 uncertain）

- **yes**：交付物必须读取、比较或理解实际图片像素、截图、图表、渲染画面、视频帧。
- **no**：文本、代码、音频、配置已足够。代码谈到 UI/图像/视频不等于需要视觉。
- 不确定图片是否存在、上下文缺失、动作不明：**不得猜 yes/no**。
  写 `label_status: "adjudication"`，未知字段用 `null`，记录
  `adjudication_reasons` 与缺失证据；或明确 `excluded` 并写理由。
  待裁定/排除记录放在独立账本，绝不进入 train/calibration/test 监督或支持计数。
- 旧 `uncertain` 只有在独立复核记录明确 yes/no 和证据时才转换。
  低 confidence 不能代替待裁定，预检失败也不是强制二值化的理由。

## task_type：六选一，按交付动作互斥

`extraction / research / implementation / debugging / review / creative`

1. **主要交付为编写/修改可执行代码、脚本、样式表、测试 → implementation**。
   即使主题是艺术、创意页面、视觉化或视频，也优先此规则。
   例：按截图实现 CSS 页面是 implementation + yes；仅按文字需求实现是
   implementation + no。审核代码不是“编写代码”，仍归 review。
2. **非编程创作/鉴赏 → creative**：自然语言文案、剧本、设计构思、音乐、视频
   创作方案及多媒体鉴赏；需读实际画面时可同时 needs_vision=yes，亦可旗舰。
3. 其余动作：机械提取统计转换 → extraction；调查综合 → research；
   诊断错误根因 → debugging；审核规范/方案/画面质量 → review。
4. **废除 visual 类型**，但不是把所有视觉任务自动改成 review：
   - 图片读字、抄图表数值、提取可见属性 → extraction；
   - 按标准检查/比较画面、审核质量 → review；
   - 旧记录仅写“分析图片”，无法判断提取还是审核 → 待裁定，须获取动作证据。
   - 可确认的编程/创作主交付仍按前两项优先；不能只靠含“图”关键词迁移。

纯 stdlib 的 `label_schema.reviewed_task_type` 执行上述确定动作映射；
`review_legacy_label` 复制旧标签并记录证据或待裁定，不修改输入，不猜测提示词。
代码不保证人工提交的证据真实，也不替代独立审查。

## 输出契约与盲审

- 监督行顶层 `label_schema_version: "router-label-v2"`；嵌入 `label` 内
  `label_status: "resolved"`，tier、needs_vision、task_type 均为合法值。
  原始 recovery 标签自身声明版本，prepare 将其传递到最终行。
- `confidence` 是 0–1 的 **tier 专家判断确定度**，不是成功率、难度或统计区间置信度。
  `rationale` 说明档位理由；不输出模型名/provider/选模建议。
- 所有真实修订写入新数据/标签版本、保留原标签和审查来源。先预注册规则、
  对预测盲化，再复核；不能用测试预测决定如何改标签。
- #18 的真实六条 uncertain 证据复核、主导测试组 **16 条 creative 盲审**仍未进行；
  不发布真实行、不覆盖 frozen-v1，不宣称两个版本指标可直接比较。
- schema 合法性与独立组支持、各切分缺类分开报告。A/B/C 门槛保持原样；
  合法标签不保证校准可行，合成例子不能补真实独立组。

## 合成补充边界

`data/synthetic-examples/labels-v2-creative-vision.jsonl` 的 15 行均明确虚构，
creative+yes，其中 4 行 flagship。只有文字格式/规则例子，没有真实图片或真实任务。
标记 `synthetic: true`、`supplemental_only: true`，无 split，统一合成来源桶；
不自动分配训练/校准/测试，不代表真实支持或生产改善。
显式 v2 `--supplemental` 可仅加入训练；不得加入校准、测试、真实 prior 或支持门。

## English summary

V2 has six action-based task types and binary vision labels. Writing executable
code/scripts/stylesheets takes precedence over a creative subject; nonprogramming
creation/appreciation is creative. Map legacy visual to extraction or review only
with explicit action evidence (unless a known primary programming/creative action
applies). Ambiguity remains adjudication/excluded, never forced binary. Keep v1
questions/checkpoints and frozen-v1 labels untouched. Real six-uncertain review,
16-creative blinded review, group support and no-leak evidence remain outstanding;
15 synthetic supplemental examples cannot establish these acceptance criteria.
