# Reasoning datasets and prompts

本仓库仅保留当前数据、提示、分层抽样，以及生成和校验它们的代码。

## 目录

```text
configs/          数据构建参数、当前 prompt 模板
src/lra/data/     数据导入、筛选、配对、prompt 渲染、抽样和校验
src/lra/paths.py  仓库路径
data/
  raw/            重建当前数据所需的冻结来源
  tasks/          三个任务的有效题目 JSONL
  prompts/        每题 × 三种条件的 messages 与 prompt_hash
  splits/         main、pilot 的题目 ID；pilot 是 main 的子集
  manifest.json   当前配置、来源、构建代码、产物哈希与抽样依赖
```

## 使用

```bash
python -m pip install -e .
python -m lra.data.build
python -m lra.data.validate
```

只重建 comparison：`python -m lra.data.build --task comparison`。普通构建（包括兼容参数 `--force`）使用冻结缓存；依赖过期的任务及下游产物会一起重建。只有显式 `--refresh-sources` 才重新抓取 comparison 数值和人口元数据，并在下载完成后替换缓存。人口元数据补查入口为 `python -m lra.data.comparison_metadata`。

校验覆盖来源和产物哈希、题目及答案格式、PrOntoQA 独立逻辑证明、comparison 选值/分箱/镜像、全部 prompts 和分层抽样。

## 数据与提示

- **StrategyQA**：作者公开的 2,290 道带标签 train 题，答案为 Yes/No。仅保留本项目使用的原始 train JSON。
- **PrOntoQA**：直接导入[作者原版归档](https://github.com/asaparov/prontoqa/blob/0a6412b6fddf46324a1cb96e066dd7b3d89b87d6/model_outputs_v1.zip)的 1/3/5 跳题，每跳 400 题，共 1,200 题；不扩词、不生成新题。原题、作者 Expected answer、文件与试次编号可追溯。归档与三个成员文件均固定 SHA-256。八个示例和模型预测不进入当前 prompts；本项目的提示协议不同于原论文的八样本协议。
- **Comparison v2**：参照 [Lehmann 等人的实体比较方法](https://github.com/HeLehm/facts-vs-shortcuts)自建的 Wikidata 适配版，共 6,000 题，答案为 A/B。使用 sitelinks 知名度代理、五档 log10 差距，并保留两个呈现顺序；不声称使用论文原始数据或直接运行作者代码。

人口取值限定在 2015 年至 2026-09-14，题面标明年份，配对年份最多相差 3 年。补充元数据必须匹配冻结数值、单位、日期和 rank；只接纳配置允许的限定字段，并排除已解散实体。取消跨年代中位数过滤，优先 PreferredRank，再选最新日期。人口候选范围包括选中年份前 3 年至截止日的合格值；其余属性采用优先 rank 池的合格值。只保留候选范围严格不重叠的配对。

人口仍依具体 Wikidata 实体定义，未逐城市人工核验边界；其他四类属性沿用原数值缓存，缺少完整范围限定，保留 `scope_check` 标记。自动校验不等于逐条外部事实核验。

主实验每任务 300 题、预实验每任务 30 题。PrOntoQA 按跳数和标签平衡；comparison 主/预实验在六种属性 × 五档难度中每格 10/1 题，标签与问法平衡，每对仅选一种呈现顺序。

`tasks/*.jsonl` 的共同字段为 `item_id`、`task`、`question`、`options`、`label`、`difficulty`、`meta`。`prompts/*.jsonl` 包含 `prompt_id`、`item_id`、`task`、`condition`、`messages`、`options`、`label`、`prompt_hash`。三个条件为基线、谨慎、快速；同一道题的用户消息完全相同，仅系统指令变化。准确措辞以 `configs/prompts.yaml` 为准。

## 当前构建

<!-- summary:start -->
Built 2026-09-14T22:07:23+00:00 (data config `e6cc16cb0716`, prompts config `dbc7f2b86fca`).

| task | items | labels | breakdown |
|---|---|---|---|
| strategyqa | 2290 | Yes 1071, No 1219 | by_decomposition_steps: {1: 18, 2: 626, 3: 1219, 4: 342, 5: 85} |
| prontoqa | 1200 | False 580, True 620 | by_hops: {1: 400, 3: 400, 5: 400} |
| comparison | 6000 | B 3000, A 3000 | by_attribute: {'buildings': 1000, 'cities': 1000, 'countries': 1000, 'mountains': 1000, 'rivers': 1000, 'stadiums': 1000}; by_ratio_bin: {0: 1200, 1: 1200, 2: 1200, 3: 1200, 4: 1200}; n_pairs: 3000 |

Split `main`: comparison 300, prontoqa 300, strategyqa 300

Split `pilot`: comparison 30, prontoqa 30, strategyqa 30
<!-- summary:end -->
