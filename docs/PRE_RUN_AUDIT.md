# 主采样前审查：PLAN v0.3 与当前 data

后续状态（2026-09-14）：PrOntoQA已换成作者原版v1归档的1/3/5跳题，原题/原标签保真、逻辑一致性和最短证明跳数均已通过检查；主/预实验每跳100/10题，标签平衡。下文关于扩词表版本的发现是历史审查，其旧数据与split保存在 `data/legacy/prontoqa_extended/`。comparison随后已修复为v2，见 [修复记录](COMPARISON_REPAIR.md)；下文comparison问题也为历史发现。采样记录契约仍待实现验证。作者数值比较仓库的可用内容见 [来源状态](LEHMANN_SOURCE_STATUS.md)。下文原路径/行号对应审查时的v0.3版本。

审查日期：2026-09-14。结论：**目前不应启动正式预实验或主实验。数据有可复现的构建错误；第 5 节信号框架基本覆盖研究目标，但字段与事件定义尚不足以保证不用重采主轨迹。**

本次完成代码/实际数据审查、独立逻辑验证、tokenizer 核验及官方接口文档核验，未改动原始计划、构建代码或数据。下一步实现要求见 [SAMPLING_CONTRACT.md](SAMPLING_CONTRACT.md)。以下问题仍待修复。

## 1. 数据侧必须先修的事项

### P1：PrOntoQA 生成了矛盾及证明捷径

`src/lra/data/prontoqa.py:98–100` 在每次 `generate_theory` 时追加扩展词族。官方生成器在同一道题构建主本体和干扰本体时两次调用该函数，并共享被逐步消费的词族列表（`third_party/prontoqa/run_experiment.py:463,504`）。第二次追加让已经消费的词族重新出现。

独立解析题目英文规则，并与原始 41,632 条公式逐句核对一致，再作 Horn 闭包与最短路径检查，发现：

| 问题 | 全量 | main | pilot |
|---|---:|---:|---:|
| 能推导出同一对象同时具备 p 与非 p | 50 / 1200 | 12 / 300 | 3 / 30 |
| query 本身正反都可证明 | 1 | 0 | 0 |
| query 的最短证明距离不等于标称 hops | 1 | 0 | 0 |

例如 `prontoqa:h8:0051`（`data/tasks/prontoqa.jsonl:852`），Sally 是 stirpor，stirpor 直接推出 hollow；另一条八步链又推出 not hollow，而问题问 not hollow。`prontoqa:h3:0087`（第 488 行）标称三跳，但 “Stella is a timpus” 和 “Timpuses are not quiet” 一步就回答了问题。

应让扩词族在每题初始化时只发生一次，生成后独立检验全局一致性、query 的唯一可判定性及最短证明距离，再按原有分层规模重建。只修代码而沿用旧 JSONL 不够。50 题都应排查，而不只剔除 query 正反都可证的那一道。

### P1：Wikidata 的 PreferredRank 规则失效

`src/lra/data/comparison.py:135` 用只按 `/` 截断的 `_qid` 解析 rank；实际缓存为 `ontology#PreferredRank`、`ontology#NormalRank`。第 186 行却判断是否等于 `PreferredRank`，因此优先级筛选一直不起作用。

在临时副本中**只修正 rank 字符串**，其余选值规则不变，得到：

| 范围 | 数值变化题 | 标签翻转 | 新增并列 | 难度档变化 |
|---|---:|---:|---:|---:|
| 全量 6000 题 | 602 | 28 | 0 | 164 |
| main 300 题 | 31 | 1 | 0 | 6 |
| pilot 30 题 | 2 | 0 | 0 | 1 |

258 个实体的选中数值变化；164 个难度档变化题中有 24 题跑出配置允许区间。上表按题计数，包含镜像两题。

main 的 `comparison:countries:0093:ab`（`data/tasks/comparison.jsonl:2187`）比较 Denmark 和 Lebanon 的更小人口，按承诺的选值规则修正后，标签从 B 变 A。这里指的是**构建规则一致性**，不意味着 PreferredRank 自动代表现实最新真值。

修复需兼容已有缓存中的 fragment 写法，并重新选值、分箱、抽 pair、渲染 prompts 和构建 splits；只改标签会保留错误的难度分布。

### P1：数值比较的预实验没有覆盖六种属性

`src/lra/data/splits.py:41–55` 按排序后的联合 strata 做 round-robin；不足整轮时，前面的属性优先取到余数。

| 属性 | main | pilot |
|---|---:|---:|
| buildings | 60 | 20 |
| cities | 60 | 10 |
| countries | 60 | 0 |
| mountains | 40 | 0 |
| rivers | 40 | 0 |
| stadiums | 40 | 0 |

pilot 难度档计数为 8/8/6/4/4；PrOntoQA pilot 的跳数计数也偏为 6/6/6/4/4/4。

建议 comparison 先约束属性×难度：pilot 六属性×五档各 1 题，main 每格 10 题，再协调全局标签、问法及每 pair 只取一种顺序。PrOntoQA pilot 可每跳 5 题，标签在全局平衡。只打乱 strata 顺序不能保证这些边际约束。

### P1：人口题的年代与实体范围还没有可辩护的统一口径

`comparison.py:181–183` 把不同年代的人口一起按中位数过滤，`188–189` 再在剩余记录中选日期；题面却没有年份或人口口径。此外，选值代码只要有 dated statement 就优先于所有 undated statement。实际数据包含：

- Yekaterinburg 的选中数值为 **1926 年 140,000 人**（`data/tasks/comparison.jsonl:1047`）。
- main 包含 **Polish–Lithuanian Commonwealth 的 1771 年人口**与 Somaliland 的比较（第 2771 行）。
- main 的 120 道人口题有 9 道双方取值年份相差至少十年；pilot 的 10 道城市题中有 3 道。

展开 Yekaterinburg 缓存后确认，原始数据其实包含2009年1,401,729、2017年1,455,904、2018年1,468,833（PreferredRank）。合理区间过滤后，跨年代记录的中位数为91,400，五倍上限为457,000，上述三条较新记录全部被当成离群值剔除；剩下的最新年份才变成1926。**修正 rank 字符串也救不回已被前一步过滤掉的值。** `comparison:cities:0265:ab`（`data/tasks/comparison.jsonl:1531`，当前未进入 main/pilot）问 Kielce 与 Yekaterinburg 谁人口更小：按缓存的2021年186,894与2018年1,468,833应选A，当前却按1926年140,000标为B。

正式采样前要固定时间范围、现存实体范围及城市人口边界，并过滤明显不匹配或排序不稳定的 pairs。人口、建筑高度、河长等还可能有不同统计口径；接近的 pair 尤其需要检查。不要把由统计口径引起的错误当作模型推理噪声。

`resolve_value` 已产生的 `value_point_in_time`、`value_rank` 在第 313–314 行写入 task 时被丢弃，必须保留。原始查询还应保存 statement ID、来源/适用范围等必要限定信息。现有缓存只含部分限定字段，不能据此宣称已完成逐实体现实真值核查。

### P2：manifest 不能证明当前配置真的生成了当前数据

`src/lra/data/build.py:38` 可以直接复用旧 items，后面仍将当前配置 hash 写入 manifest。配置变了但未重建时，会出现“新配置 hash + 旧数据”。正式采样应锁定每个产物文件 SHA-256、数据版本、builder 代码状态和 split hash，并检验依赖是否过期。

另外 `--force` 会触发 Wikidata 重新抓取。应区分“从冻结缓存重建”和“刷新外部来源”，避免修构建代码时把数据来源也改变。

## 2. 已通过与未验证的检查

- 现有测试：`.venv/bin/pytest -q`，**9 passed**。它们主要验证 schema、计数、镜像和标签平衡，没有覆盖上述语义问题。
- StrategyQA：2290 条与本地官方 train 缓存逐条对齐，题文和标签一致；有一组同标签重复问题，当前 main/pilot 均未选中。
- 28,470 条 prompts 的 question、label、options、prompt_hash 与当前 canonical tasks 一致。这里证明的是传递一致，不能证明上游标签正确。
- PrOntoQA 没有“看到 not 就直接知道标签”的完全泄漏；按问句极性猜答案约 54.7%，可作为协变量。跳数与上下文长度相关，分析中仍需区分。
- Qwen3-8B tokenizer 本地缓存 revision `b968826d9c46dd6066d109eabc6255188de91218`：`Answer:` 后的 ` Yes/ No/ True/ False/ A/ B` 均为单 token，IDs 分别为 **7414/2308/3007/3557/362/425**；`</think>` 为 **151668**，EOS 为 **151645**。这些是该版本的核验结果，其他模型必须另查。
- 现有 `check_tokens.py` 只做词级检查。当前环境缺少 **jinja2**，实际 `apply_chat_template` 会报错；也未安装 **torch/vLLM**。所以这次没有验证 GPU 生成、完整模板渲染、HF-vLLM 概率一致性、峰值显存或吞吐。
- 已实际复现字符串重分词改变句尾 token 的问题：拼接 readout suffix 时必须保留原 token 前缀，见记录契约。

审计证据保存在 [data_audit.json](audit/data_audit.json)、[输入快照哈希](audit/baseline.json)。[复现脚本](audit/data_audit.py) 可从仓库根目录用 `.venv/bin/python docs/audit/data_audit.py` 运行，读取已归档的扩词表版PrOntoQA、comparison v1及其构建器/配置，以及未变化的StrategyQA数据，写 `/tmp/lra_data_audit.json`；它复现历史审查，不是当前原版PrOntoQA的CI测试。当前原版检查运行 `.venv/bin/pytest -q`。

## 3. 第 5 节有哪些关键缺口

| 现有描述 | 问题 | 需要补的记录/定义 |
|---|---|---|
| 完整 prompt + 输出 tokens | prompt 文本不能唯一复原实际模型输入 | 实际 prompt_token_ids、模板参数/内容、模型与 tokenizer revision、全部生效采样参数 |
| top-5 + sampled logprob | 候选外概率未知，不能恢复全分布熵或两个答案差 | logprobs_mode、候选 IDs/ranks、回放 raw/policy 两套量；现场精确概率另定义采样器导出 |
| `P(</think>)` 就是停止风险 | raw softmax 与经过 temperature/top-k/top-p 的行为策略不同 | 同时区分 raw 模型倾向和 policy hazard，正确使用停止前的 logits |
| 是否发出 close 就是删失标志 | 无 close 可能是失败/格式错误；有 close 也可能答案截断 | 思考停止、回答完成、停止原因、预算与删失时点分开 |
| 每句读数最多 60 点 | 起点、无句号终点、自然答案位置和固定绝对时间没有保证 | 必存 t=0、close 前、自然答案前，额外固定网格/风险集抽样说明 |
| 只存 X(t) | 看不到两个答案词在全词表上的总概率质量 | 两候选各自 logits/logprobs、IDs、总质量、二选一归一化概率 |
| 4 层隐藏状态 | 层号/hook/归一化位置、边界语义不明确 | 精确提取位置、dtype、坐标、子集清单、模型实现版本 |
| trace_id 断点续跑 | 换模板/阶段后可能覆盖；半文件可能误判完成 | stage/config hash、分片 checksum、原子提交、期望 trace 清单 |

vLLM 官方明确 raw 与 processed 概率的区别，且指定候选 logprob 与 top-N 的接口语义需要按版本核验；不能凭“vLLM 会给”省略接口验收。[ModelConfig](https://docs.vllm.ai/en/latest/api/vllm/config/model/)、[SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)

精确 token 轨迹、模型快照和配置保存完整后，大多数漏掉的**派生信号**可以对同一轨迹追加回放，不必重新采主轨迹。生成引擎当时的确切数值、时序与错误事件不应事后假定能逐位重现。没跑过的条件、被截断的自然续写、分支和干预也不属于可离线恢复的记录。

## 4. PLAN 需要修正的研究解释

1. **温度不是已知的 DDM 扩散系数。** temperature 改变 token 策略，也会改变内容、漂移和停止概率。零漂移、边界 ±a 的 DDM 有 `E[T]=a²/σ²`、`Var[T]=2a⁴/(3σ⁴)`，增大 σ 时均值和方差都下降，已是“必然分布变宽”的反例。应比较约束参数映射，而非把该映射当前提。基本固定漂移、对称起点 DDM 也不自动预测错误更慢/更快，需区分扩展变异模型。[Ratcliff & McKoon](https://pmc.ncbi.nlm.nih.gov/articles/PMC2474742/)
2. **模型对照名称不准确。** Qwen3-1.7B/4B/8B/14B 与 32B 并非全用同一后训练流程；前者采用蒸馏路线。Qwen3 关 thinking 是同权重模式对照，不是非 RL 模型；DeepSeek-R1-0528-Qwen3-8B 是基于 Qwen3-8B Base 的蒸馏版本，不能解释为“只改变 RL 配方”。规模关联、模式效应和后训练来源可以研究，但不能声称分别隔离 scale/RL 因果。[Qwen3 技术报告 §4.5](https://arxiv.org/html/2505.09388v1)、[DeepSeek 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
3. **强制读数不自动等于潜在决策状态。** 后缀会干预条件分布；末尾一致不证明中途忠实。直接按 DDM 路径拟合前，要建测量映射并做包含测量误差、非等距采样和观测间越界的参数恢复。分支 16 次的最坏 SE=0.125，只能称 Monte Carlo 估计；真实条件概率是鞅不意味着估计值或其 log-odds 是鞅。
4. **停止分析必须使用仍在思考的风险集。** 绝对 token 时间可用于预测；相对位置 t/T 借用了未来 T，只能事后描述。实际发出 stop 的位置其 P(stop) 高存在事件选择效应，单凭与随机位置的比值不能证明边界停止。删失轨迹须进入适当似然或单独统计。[PyDDM 的 undecided 数据说明](https://pyddm.readthedocs.io/en/latest/cookbook/loss.html)
5. **预实验表不一致。** 最低阈值和目标阈值应分别命名：读数95/98%、解析98/99%、close90/95%。C0/C1 pilot 无法验证 C2。第199行的“早期AUC≈0.5直到末尾，支持先决定后写理由”解释方向不对；这种说法需要早期已含充分决定信息、之后仍继续写。分析要与 prompt-only 和题内预测基线比较，按 item 分组验证。
6. **K=100 不是单题拟合通用下限。** 例如准确率98%的题只有约2条错误，无法可靠计算多个错误长度分位数；应由 pilot 和参数恢复决定层级共享与样本量。pilot 调过提示、题目选择或预算后，也不能无标记地并入确认性主实验。

这些定义多数可先修文字和分析实现；若要换真正的训练对照、加 C2/温度条件或改变预算，则必须在主生成前决定，旧轨迹不能代替新条件。

## 5. 算力与存储需要用实测替换保证式估计

以下是直接按 PLAN 的规模计算，单位为十进制 GB，不是实测吞吐，也未计压缩、元数据和重试。

- 270k×1500 = **405M** 主生成 token。5400 条 pilot 若也按1500计，则是 **8.1M**，不是5M。
- token IDs int32、sampled logprob float32、top-5 的 IDs+logprob：约 **19.44GB**；改存 top-20 为 **68.04GB**。六个逐步 float32 回放量另约 **9.72GB**。
- Qwen3-8B hidden_size=4096。探针子集54k条×平均40边界×4层×4096×2字节≈**70.8GB**；平均60边界则≈**106.2GB**。70GB依赖平均边界数假设，不是上限；原计划没有对全部隐藏状态句边界设60点上限。
- 分支子集2700条父轨迹×10点×16续写=**432,000条新续写**，还未计首次越界点。若平均主思考1500 token、位置0.1到1.0且平均剩余比例0.45，剩余思考量约 **291.6M token**，另加答案输出。该阶段与主生成同量级，不能理解为免费后处理。
- 全量60个读数点意味着最多 **16.2M 次读数请求**（另计强制端点）。前缀缓存依赖调度和命中；若改用 prompt_logprobs 打分，官方 V1 文档说明会重新计算完整 prefill，因此不能直接沿用原先缓存成本估算。[vLLM V1](https://docs.vllm.ai/en/v0.25.1/usage/v1_guide/)

建议顺序：**修数据和分层 → 冻结记录契约/依赖 → generate+segment+score+readout+pilot 分析的最小闭环 → 18–36条 GPU 冒烟 → 5400条 pilot → 根据实测冻结主实验**。不能只写 generate 和行为统计脚本就开始全量采样，因为预实验的读数效度、停止指标和回放一致性还无从验收。
