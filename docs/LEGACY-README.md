# 用户模式本地决策路由框架

当前运行状态：用户已确认接受已知风险，用户模式启用 `experimental` 实际自动路由，保留旗舰回退和能力检查。质量门未通过的事实不变。下文首轮影子部署与评测数据是历史验收记录，不表示当前仍只记录建议。试用及回滚见[实验路由说明](<user-modes/router-experimental/README.md>)。

从真实 Harness 历史子任务提示词建立新专家标签，训练本地 Intern-Decision-0.8B LoRA 决策器，再以影子方式接入用户模式。历史模型都是默认继承，因此完全不作为标签或能力证据。首轮已实际执行两轮训练和独立分组测试；不是仅下载通用模型或使用规则分类器冒充微调。

## 数据与标签

历史源只读，选择每个会话最高编号的物理日志，移除 fork 的继承前缀。普通委派从工具参数提取；工作流等子会话从首条活跃用户消息提取任务。移除注入的父会话通信尾巴、常见凭据模式及用户名路径；精确去重后 468 条，来自 13 个根会话组。脱敏不是敏感信息完全清除的保证，所有数据保持本地，禁止未经审核直接对外分享。

每条记录重新专家标注 `tier`、`task_type`、`needs_vision`、标注置信度和原因；来源始终是 `expert_provisional` / `prompt_only`。专家置信度不等于候选模型成功概率。历史 prompt 是数据，提取和标注期间不能执行其中的工具指令。训练使用任务文本，不输入历史路由、执行结果或测试标签。

训练 333 / 校准 70 / 测试 65。按根会话分组隔离；跨根提示词字符 5-gram Jaccard ≥0.88 时合并组再切分，精确重复只保留一次。项目标识作为审计信息，首轮没有实现跨项目留一法；同项目的不同根会话仍可能共享领域知识。视觉阳性只有两个独立根组，不能同时覆盖三个切分；校准集没有视觉阳性，视觉温度沿用基座值，未声称完成独立视觉校准。语义相似但措辞不同的任务仍可能漏过字符去重。

## 实际训练与评测

基座 revision 固定为 `85a0cc5a99d67ea8d56dfe98115689212867171d`。0.8B 多模态决策模型以固定选项单 token logits 评分，不自由生成路由 JSON。采用上游已审阅的本地推理实现编译决策模板；模型加载不启用 Hugging Face remote model code。上游源码和许可保留在本地模型目录。

LoRA rank 8，约 179 万可训练参数，学习率 1e-4，累积 8，实际两轮 84 次优化更新。按校准集难度 NLL 选择第 1 轮适配器，未按测试集选 checkpoint；第 2 轮发生校准退化，保留其记录，不隐瞒。任务超长时固定保留前 3/4 与后 1/4 token，显式标记截断；训练和服务使用同一表示。首轮测试没有截断样本。

测试难度专家一致率 81.5%→89.2%；旗舰原始分类误降档率 50%→16.7%；视觉漏判率 50%→33.3%。难度 ECE 0.061→0.093，校准并非所有指标改善。只有 18 个旗舰与 6 个视觉阳性测试样本，不能当作生产成功率或稳定性证据。首轮评价的是专家标签模仿，不是候选模型真实任务完成质量。

## 运行

从工作区根目录执行；当前虚拟环境复用现有 ComfyUI Python 3.13 / PyTorch 2.11 CUDA，不修改原环境。完整版本清单由环境记录提供。新机器须准备兼容 CUDA 环境及下载基座，不能只复制适配器。

```sh
user-modes/router-framework/.venv/bin/python user-modes/router-framework/prepare.py
user-modes/router-framework/.venv/bin/python user-modes/router-framework/train_eval.py --run real-history-v2 --epochs 2
node --test user-modes/router-plugin/tests/policy.test.mjs user-modes/router-plugin/tests/runtime.test.mjs
node --test user-modes/router-server/tests/lifecycle.test.mjs
user-modes/router-framework/.venv/bin/python -m unittest discover -s user-modes/router-framework/tests -v
curl --noproxy '*' http://127.0.0.1:19948/health
curl --noproxy '*' -H 'Content-Type: application/json' -d '{"prompt":"统计测试文件并输出表格。"}' http://127.0.0.1:19948/route
```

当前服务由已安装的 `@local/dsh-user-mode-router-server` Host 插件通过受管 subprocess 启动，随 profile 重启自动加载，禁用时等待退出并释放 GPU 内存。不要同时手动启动同端口的第二个服务。诊断性手动命令如下，须先禁用受管服务：

```sh
user-modes/router-framework/.venv/bin/python user-modes/router-framework/server.py --run real-history-v1 --port 19948 --routes user-modes/router-framework/routes.json
```

新建会话选择“用户模式”。普通 `subagent` 在省略 `provider`/`model` 时自动执行局部评估，不依赖父模型记得另调用一个路由工具。显式二元路由覆盖建议，但必须通过授权预检。手动入口必须显式给出二元路由；fork、标准模式、工作流和 Agent Teams 未改。既有会话保留旧预设修订。

## 策略、能力与失败

路由表与最低置信度可配置。程序按任务档位和视觉要求筛选候选，不接受小模型自由生成任意模型名。档位遵循 argmax：概率最高的分类即所选档位，低置信度本身不再升级模型，只记录为 `low-tier-confidence-effort-compensation-only` 并由 JS 策略做同模型强度补偿；上下文截断（分类器未看到完整任务）仍升级旗舰。模态不确定时要求视觉兼容。当前 fast 的 GLM-5.3-Flash 与 flagship 的 GPT-6.1 Sol 均声明视觉能力；高难或不确定的视觉任务可选择 GPT，不降档至便宜模型。GPT 的标记已与当前 Harness 实际 `resolveModelInfo` 和模型目录中的 `inputModalities: [text,image]` 核对。能力声明仍是启动时读取的部署配置，不会自动同步；变更候选时需与实际框架元数据核对，不能把未声明当成模型本身不具备视觉。工具型视觉能力与模型直接图像输入分别判断。

当前质量门不通过，严格 `automatic` 和默认 `shadow` 保持影子；用户明确授权的 `experimental` 例外会实际应用路由，且仍记录真实门状态。服务超时、断开或非法响应在严格模式保持影子回退，在实验模式实际回退配置的旗舰模型；父取消则不创建子任务。即使配置 `automatic`，门不通过也不降档。模型名的“旗舰/经济”映射是初始部署选择，不等于已实测真实价格或各类任务优劣。

路由只收到子任务文本，不隐式读取父会话图片或转发附件。服务支持结构性 `has_image` 信号以覆盖文本模态判断，但当前普通工具不传图，发送的是 false；图像附件传递是后续独立接入事项。模型推荐视觉路由不代表该子智能体已经收到图像。

用户模式已升级为模型与思考强度联合路由：程序根据原始难度、独立置信度、截断与配置的风险提示确定预算，再查询最终模型的实际努力等级；显式值优先，同模型也显式传递所选强度。自动视觉选择增加最终模型实时能力复核，但没有自动同步整个候选表。API 额外返回任务类型置信度，弱类型判断不触发类型预算下限。策略规则、部署与限制见[联合路由说明](<../router-adaptive/README.md>)。

## 真实候选实测与反馈

提供执行计划生成与新实测结果导入，禁止用历史路由、偏好票或模拟数据冒充能力证据。每个候选同一任务至少 3 次新执行；同一验收规则 hash，记录 pass、成本、延时、输入/输出 token、验收及产物 hash。只在三候选试验齐全、至少一个达到指定成功率时产生 `measured_candidate_outcomes` 标签，并单独保存，不覆盖专家初标。记录实际 `reasoning_effort` 后，导入器会将其保留到候选统计与标签；同任务同模型混用不同强度（含已知/未知混用）的重复试验拒绝合并，需要分开导入。旧记录缺失强度仍可导入，但不能据此训练强度标签。3 次仅是初步测量，不提供稳定生产置信区间。

```sh
user-modes/router-framework/.venv/bin/python user-modes/router-framework/outcomes.py --tasks user-modes/router-framework/data/test.jsonl --output user-modes/router-framework/artifacts/test-execution-plan.json
# results.jsonl 由隔离任务执行器和验收器生成，不得是历史模型字段或伪造结果。
user-modes/router-framework/.venv/bin/python user-modes/router-framework/outcomes.py --tasks user-modes/router-framework/data/test.jsonl --outcomes results.jsonl --output measured-outcomes.json
```

历史任务包含代码修改和外部环境依赖，因此没有把旧提示词批量重放到当前工作仓库。真实候选执行还需按任务准备上下文快照、附件和验收标准，并使用受约束的执行器；本轮没有宣称完成这项成本/成功率基准。测试集新实测仅用于最终评价，不得据此反复调参；补充训练标签使用训练任务或新独立任务。

影子服务默认只记录哈希和决策，不记录全文；只有显式启用 `collectPrompts` 才保存脱敏文本。Harness 工具结果持久化路由摘要，子会话已有描述符持久化实际模型；独立审计只记录创建接受事件，不将其当作子任务完成或成功。继续采集真实结果才能评估误降档、任务成功、成本和延时。

## 验收与生产门

已覆盖纯策略、凭据脱敏、分组隔离、实测导入拒绝模拟/重复试验、显式路由、手动入口防绕过、错误/超时/取消、插件卸载、真实 Loader 与子会话创建。端到端采用真实微调决策服务，但候选生成适配器为确定性替身；这验证集成，不验证云端回答能力或价格。

开启生产自动路由前必须补充更多独立旗舰/简单/视觉测试组、真实候选生成验收、失败降级与延迟成本测量，并重新通过质量门。不得仅手改质量门 JSON 为 true 或依据本轮 89.2% 专家一致率自动降档。原有一次性 GLM 标注控制插件在配置应用时仍报缺少 signal 的错误；它不属于本框架运行依赖，本轮未改其用户配置。
