# 采样与分析记录契约（审查建议，待实现）

2026-09-14。对应 PLAN.md 第 5 节和 Q1–Q5。本文件是下一步脚本的验收要求，不表示这些字段已经实现或通过 GPU 验证。

“一次生成”指不因漏存数据而重新生成已完成的主轨迹。HF 回放、强制读数仍要做模型计算；分支和干预仍要生成新续写。保存原始轨迹能支持这些工作，但不能恢复没有运行过的实验条件，也不能把被 max_tokens 截断的轨迹当作自然停止。

## 1. 必须在主生成时冻结的输入与记录

下面字段可以按 run / prompt / trace 去重存储，但必须有稳定外键、完整内容和校验和。只有 hash 而没有其对应内容不够。

| 层级 | 必存内容 | 目的 |
|---|---|---|
| run | schema_version、run_id、实验配置全文及 SHA-256、数据/tasks/prompts/split 的快照和文件 SHA-256、代码 commit 与未提交改动快照 | 当前仓库很多文件尚未提交，单独记 git commit 无法复原代码和数据 |
| model | HF repo、不可变 model/tokenizer revision、权重/adapter 身份、config、generation_config、tokenizer 文件与 chat_template 的内容/哈希 | 模型名和 vLLM 版本不足以回放 |
| engine | vLLM/transformers/torch/tokenizers/CUDA/driver 版本、GPU 型号、dtype、量化、KV dtype、TP/PP、attention/sampling backend、RoPE/context 配置、前缀缓存/调度/推测解码设置 | 解释回放差异及性能；不承诺跨硬件逐位复现 |
| policy | 最终生效的 temperature、top_p、top_k、min_p、所有 penalty、seed、min/max_tokens、stop 字符串和 token、EOS 集合、ignore_eos、logit processors/bias/allowed tokens、结构化输出、thinking budget、generation_config 合并结果 | 不依赖未记录的默认值 |
| prompt | messages、apply_chat_template 参数（包括 enable_thinking、add_generation_prompt）、实际渲染文本、**实际送入模型的 prompt_token_ids**、token 数及 hash | 避免模板漂移、重复 BOS、截断、分词变化 |
| item | task/item_id、question、options、label、difficulty、meta、数据版本/哈希 | 防止数据更新后重新评分串版本；显式规定 options[0] 为 X 的正方向 |
| trace | trace_id、prompt_id/hash、condition、sample_idx、有效 request seed、request_id、attempt_id、run_id、request/finish 时间、耗时、完成状态/错误信息 | 幂等、排错、成本测量 |
| output | **完整 generated_token_ids**，包含收到的 think/EOS/终止 token；保留特殊 token 的原始解码文本；API 原始 text 可另存 | token 序列是唯一权威输入，显示文本不能替代它 |
| termination | finish_reason、stop_reason、实际终止 token/字符串、requested/effective_max_new_tokens、max_model_len、是否输入截断、输出是否完整 | 区分自然结束、长度限制、失败与被强制结束 |
| token signal | 采样 token 的 logprob、候选 token_ids/logprobs/ranks、请求的 top-N、**logprobs_mode**、缺失状态和数组偏移 | 缺失候选不代表概率为零；候选数可能是 N+1 |

每条完成轨迹必须有唯一实验身份：数据/模型/模板/策略版本、item、condition、sample_idx。run_id 可另作执行批次。推荐按稳定身份派生每条 trace 的 seed，并用 n=1 请求避免无法解释批量 n 的子样本 seed；保留重试记录，完成轨迹不再采样。跨条件是否共享随机数要显式定义。

主实验建议首先使用单一 Qwen3-8B thinking adapter，显式设置计划里的 0.6/0.95/20/min_p=0。不要自动添加长度强制、重复检测终止、grammar 限制或其它改变停止机制的处理。所有这些开关在配置中显式记录；确需使用时作为独立 policy 版本。

不要把 `</think>` 设为整条 completion 的 stop，否则无法取得自然最终答案及 Confidence。不要只保存 reasoning parser 的拆分结果；解析前 token 流也要落盘。

vLLM 的返回模式区分 raw 与 processed，后者包括 temperature 和 top-k/top-p；prompt_logprobs 不经过这些采样处理。应锁定实际安装版本，并在运行时核验其 API。[vLLM ModelConfig](https://docs.vllm.ai/en/latest/api/vllm/config/model/)

## 2. 原始概率与实际采样概率分开保存

定义 `p_raw(v | prefix)` 为模型原始 softmax；`q_policy(v | prefix)` 为执行全部采样处理后的分布。二者不同，禁止共用一个没有语义标签的 `logprob` 字段。

| 信号 | 建议生成/计算位置 | 能否凭同一原始轨迹后补 |
|---|---|---|
| 生成引擎当时返回的 sampled logprob 与 top-N | 生成时 | 不保证 HF 数值与当时的引擎完全相同，现场记录最可靠 |
| 每位置 raw sampled logprob、raw 全词表熵、raw log P(think close)、raw log P(EOS) | HF/同引擎回放 | 能，必须冻结模型、精确 token 与配置；注明 replay 来源 |
| 每位置 policy sampled logprob、policy 熵、policy log P(think close)、policy log P(EOS) | 从完整回放 logits 按同一策略重建；与生成记录核验 | 能重算该前缀的策略分布，但不能未经核验便称为原引擎的精确现场概率 |
| 精确的生成时 policy stop 概率、熵、支持集 | 若研究要求逐步精确现场值，需要在采样器内导出相应标量/支持集 | 标准 top-5 返回不保证包含这些量；应在小样本中验证导出路径 |
| 自然最终答案位置的两个候选 raw logits/logprobs | 回放该实际前缀 | 能，不能依赖另一答案恰好进 top-5 |
| 句子/标记/试探性答案、长度与口头 Confidence | CPU 后处理 | 能 |
| 强制读数、隐藏状态 | 从相同 token 前缀回放/打分 | 能，需要额外计算 |
| 分支/干预结果 | 新续写 | 不能从已有主轨迹凭空算出 |

top-5 不能恢复全分布熵、被排除的 stop 概率或两个答案的差值。不要用 top-5 重新归一化来冒充全分布。也不必把整张 `[time, vocabulary]` logits 永久存盘：回放时逐块计算目标量后即可释放。

当前 top_k=20 下可考虑把生成保存量升级为 raw top-20 加 sampled token，保留更多实际策略支持信息；但这不是通用无损方案。k 阈值并列、mask/penalty、不同 kernel 的边界行为都要检查；精确复原还需要完整保留实际支持集。普通 top-N 与 `logprob_token_ids` 不应被误当成自动取并集：当前文档中后者指定固定候选集合，版本可能还要求 N 等于该集合长度。强制读数可单独请求两个指定 token；不支持时用 HF 打分。[SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)、[采样器实现](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/)

PLAN 中“logprobs 成本为零”应改为“无需第二次自回归生成，但有概率计算、GPU→CPU 传输及存储开销”。

## 3. token 位置、停止事件与删失

统一采用 0-based、左闭右开区间。保存 `think_start`、`think_end`、`first_think_close_index`、`final_answer_span`、`confidence_span`、`eos_index`，注明坐标相对 generated tokens 还是 prompt+generated。原始所有标记位置也保留，解析器版本单独记录。

若 prompt 长 P、输出为 g[0:L]：

- 输出 g[j] 的概率来自 HF logits[P+j-1]。
- 给前缀 g[:b] 读“下一个 token”的概率来自 logits[P+b-1]。
- 若首个自然 close token 是 g[e]，停止前读数必须使用 g[:e]，不能把 close、最终答案或 Confidence 纳入前缀。
- 明确 `<think>` 是模板注入还是实际生成；推理计时从定义好的思考开始处计数。

至少分别存 `thinking_stop_observed`、`response_complete`、`censor_reason`、`censor_at_token`、`parse_status`。`</think>` 缺失不自动意味着右删失：也可能是非 thinking 模式、EOS 提前结束或格式异常。已产生 close、但最终答案被长度上限截断时，思考结束已观测而回答不完整。

Q4 的事件定义为“下一 token 发出首次 think close”，风险集只包含仍在思考的前缀。在这个定义和当前单 token close 下，`q_policy(close | prefix)` 才是行为策略的下一步停止风险。raw 概率是模型倾向，应另作分析。无 close 的 EOS 可作为竞争事件/异常单独报告。

保存 raw/policy 的 log 概率，避免低概率下溢；policy 的 -inf 可以表示真实被屏蔽，缺失用 null 加状态，二者不能混用。

## 4. 分段、读数与隐藏状态

保存全部句子边界、字符/字节到 token 的映射、分段器与 marker 正则版本；标记保留词、位置、所属段及句子。只在思考段统计 Wait/verify 等标记，注意 C1 本身会诱发这些词，不能直接等同心理状态改变。

读数位置必须包含：

1. prompt-only / 思考起点 t=0（明确是否已包含实际生成的 `<think>`）。
2. 计划的句子边界，超过上限时保存完整候选边界、选择规则及被选下标。
3. 首个自然 close **之前**的终点，即使末句没有句号。
4. 自然最终答案 token 之前的实际前缀，用于与强制终点读数比较。
5. 用于停止分析的固定绝对 token 网格，或明确风险集抽样概率及权重。

第 1/3/4 类必需点不应被“最多 60 句”删掉。固定网格可限定在预先选择的子集以控制成本。`t/T` 只用于事后轨迹可视化；在线停止预测不能借用未来终止时间 T。

**按 token 拼接前缀与后缀**：`prompt_token_ids + generated_token_ids[:b] + suffix_token_ids`。不要 decode 后拼文本再 encode，也不要把一段未完成 assistant 思考重新包装成新对话。

本地 Qwen3 tokenizer 已复现：`This is done.` 的句点是 token 13，但字符串拼上 `\n</think>...` 后再编码会合并句点和换行，原 token 前缀不再一致。无修改分支必须严格重用原始 token 前缀。

每个读数保存 `readout_id`、trace_id、prefix_token_count/hash、原始绝对位置、sentence_idx、selection_kind、suffix 内容/IDs/hash、候选答案词/IDs、两个 raw logits 或 raw logprobs、差值 X、候选概率总质量、二选一归一化概率、计算版本与状态。不要只存 X：两候选在整个词表上的质量可能都很小。

自然答案的 token 必须在其实际上下文中定位；`" Yes"`、`"Yes"`、换行后的词不能默认同一个 token。口头 Confidence 保存原字符串、数值、合法性及解析状态，不做静默截断或修补。

隐藏状态存原始前缀的状态，注明层号、hook 位置（block 输出/末层归一化前后）、token 位置、dtype、形状和子集选择版本。Qwen3-8B 为 36 层，可将 9/18/27/36 作为“block 后”语义的四层，但必须核实模型实现末层归一化定义，不能直接猜 `hidden_states` tuple 下标。探针训练/测试按 item 分组；comparison 另注意 pair/entity 重用。

强制读数是加后缀后的测量，不能仅凭末尾 argmax 一致率宣称中途忠实。预实验应比较 prompt-only 基线、模板敏感性和分支概率校准；自然答案有采样噪声，95%/98% 一致率不是理论必然。

## 5. 分支与干预记录

每条分支保留 `branch_id`、parent_trace_id、prefix_token_count/hash、选择规则、branch sample_idx/seed、策略版本、剩余预算、完整续写 IDs/text、停止/解析状态和最终答案。分支复用主轨迹的记录结构。

同一过程的条件续写必须保留原前缀、原采样策略及原**剩余总 token 预算**。从半途分支时重置为完整 max_tokens 会改变所估概率。默认不插入读数后缀、不额外调用 chat template。100% 位置需明确是 close 前还是 close 后；两者对应不同问题。

每个分支点保存正/负/无效/删失数与实际分母、置信区间；16 次续写只是 Monte Carlo 估计，最坏标准误为 0.125，不能称真值。`P(final=option0 | prefix)` 的理论鞅性质也不自动适用于有限次估计或其 log-odds。

干预另存 intervention_id、原 parent/prefix、插入 token、方向、强度、层/位置、对照类型及其配对关系。它是单独阶段的新生成。

## 6. 落盘与完成性检查

- parquet 存元数据、标量与文本；zarr 或其它明确支持变长数组的格式存 token/逐步量。变长数据采用连续值数组 + offsets/lengths，保存 trace 外键及数组 schema。
- 概率/熵建议 float32，token IDs int32；隐藏状态 fp16/bf16 并记录真实 dtype。
- 单写者或独立 worker 分片，避免多进程同时追加同一个 zarr chunk。先写临时分片，关闭并校验后原子提交。
- manifest 记录每分片 checksum、记录数、字节数、完成状态和应有的 trace 身份。不能仅凭文件存在判断阶段完成。
- 按 `(trace_id, stage, stage_config_hash)` 断点续跑；换读数模板只生成新读数版本，不能覆盖旧结果或重跑主生成。
- 每完成一小批即校验 token 数组长度、sampled token 在返回候选中的对应概率、token/text 一致性、seed/identity 唯一性、异常率与磁盘余量。故障后先修复不完整分片；不得丢弃难题/长轨迹后继续当作完整样本。

vLLM 官方仅在特定相同硬件/版本和确定性设置下提供复现条件。seed 是记录的一部分，不是跨环境重现的保证。[Reproducibility](https://docs.vllm.ai/en/v0.14.0/usage/reproducibility/)

## 7. 开始 5400 条预实验前的最小闭环

先完成 18–36 条、覆盖三任务与 C0/C1 的 GPU 冒烟测试；这是接口验收，不用于判断准确率分布。用单独测试请求覆盖一次长度截断和一次断点续跑。

验收必须同时通过：完整原始记录落盘并重读；think/EOS/answer 的 token 对齐；两个候选概率均可得；生成 logprob 与回放对应语义的误差报告；起点/中点/真实终点读数；隐藏状态子集格式；至少一个分支点无修改续写；异常/删失分类；续跑不产生重复完成 trace。报告吞吐、显存、token 长度分布、缓存命中和每条记录字节数。

然后运行分层修正后的 5400 条 pilot，产生：每题每条件 accuracy/长度/CV/信心；解析和删失率；按 item 配对的 C1/C0 效应与区间；读数效度和 prompt-only 基线；按 item 分组的早期预测增益；各项最低线/目标线及失败原因。只有 C0/C1 的 pilot 不检验 C2。

主实验配置在 pilot 后冻结。改过 prompt、模型、budget 或 policy 的 pilot 轨迹属于旧实验版本。即使配置未改，若题目或分析选择依赖 pilot 结果，确认性检验也应明确探索/验证样本边界。
