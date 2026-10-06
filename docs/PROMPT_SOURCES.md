# Prompt sources

Every wording in `data/prompts/medxpertqa.jsonl` and in the readout prompts, with where it comes from.

## Question text

- Stem: the MedXpertQA question verbatim (Zuo et al. 2025, ICML; Hugging Face `TsinghuaC3I/MedXpertQA`, config Text, split test, revision pinned in `data/raw/medxpertqa/source_manifest.json`), minus the dataset's own `Answer Choices: (A) ... (J)` line. Stems already end with the question ("Which of the following is the most likely diagnosis?" or a variant containing "most likely diagnosis").
- Options: two of the source's ten option texts, verbatim, on their own lines as `A. ...` and `B. ...`. Same layout as the comparison task.
- Two options instead of ten: the correct diagnosis paired with each distractor. Precedent for removing distractors from medical MCQs: Wang et al., *Beyond the Answers* (arXiv:2402.01349 v1 §3.3 / v2 §2.2.1; MMLU and MedMCQA reduced to 3 and 2 options by randomly eliminating distracting options; the experiment is absent from the COLING 2025 version, cite the arXiv versions). Changing the option count is also what the dataset authors did (options augmented to 10) and what the MedQA authors did (5 -> 4 options). Pairing every distractor and asking both orders (ab/ba) is the two-alternative forced-choice layout of the comparison task.

## Instruction lines

- `C0_baseline`: no instruction.
- `C1_caution_think`: "Think as carefully as possible before answering, even if this takes longer."
- `C2_speed_think`: "Think as briefly as possible before answering, even if this leads to more errors."
- Answer line, all conditions: "Answer with only A or B. No other words." (chosen in the 2026-09-15 format test: answer-first-word rate 100% on every task; the shortest of the forms that reached >= 98%).
- `C3_no_think`: no instruction sentence; the user message ends with " /no_think". This is Qwen3's documented soft switch (model card: "/think" and "/no_think" added to the user prompt switch the thinking mode turn by turn); the model then emits an empty thinking block. Used as the zero-reasoning anchor of the instruction pilot; cf. Ma et al. 2025, "Reasoning Models Can Be Effective Without Thinking" (arXiv:2504.09858).
- `C4_draft`: "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most." This is the Chain-of-Draft instruction verbatim (Xu et al. 2025, arXiv:2502.18600); their "Return the answer at the end after ####." is replaced by the answer line above so the answer format is the same in every condition.

C0-C2 are the comparison task's sentences, carried over unchanged so the two tasks differ only in the question. C1/C2 are instruction-only speed-accuracy emphasis in the sense of Forstmann et al. 2008 (one-word cues) and Zhang & Rowe 2014 ("Be fast this time" / "Be accurate this time"), rephrased for a model that thinks before answering.

## Readout prompts (measurements, not conditions)

- Forced answers inside a trace (`src/readout.py`): the thinking is cut and closed with `\n</think>\n\n`, the way Qwen3 closes it; the start readout uses Qwen3's empty-thinking block `<think>\n\n</think>\n\n`.
- Ten-option prior (`src/pair_distance.py prior`): the stem, the ten source options as `A. ...` to `J. ...`, and the answer line extended to the ten letters: "Answer with only A, B, C, D, E, F, G, H, I, or J. No other words." Read with the empty-thinking block; used only to rank distractors by the model's prior.
