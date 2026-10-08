"""Dump everything recorded about the study pairs (data/screen/medxpertqa_study20.csv) for inspection.

    python src/study_dump.py       # -> data/study20/

Writes, for the 20 pairs x 2 orders:
  index.md               one table row per pair with the screening numbers
  pairs/<pair_id>.md     the question, both option orders, the no-thinking answer with its
                         probabilities, a table of the 30 thinking samples per order, and the full
                         text of every sample (thinking and answer)
  study20.html           the same as one page with collapsible samples
  nothink.jsonl          the 40 no-thinking records as stored by src/generate.py
  traces.jsonl           the 1,200 thinking records as stored (token ids and log probabilities included)
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT, iter_traces, load_tasks  # noqa: E402

SCREEN = ROOT / "data/screen"
OUT = ROOT / "data/study20"
RUNS = {"nothink": [ROOT / "runs/medxpertqa/nothink-qwen3-8b-ab", ROOT / "runs/medxpertqa/nothink-qwen3-8b-ba"],
        "think": [ROOT / "runs/medxpertqa/think-qwen3-8b-ab", ROOT / "runs/medxpertqa/think-qwen3-8b-ba"]}
ORDER_LABEL = {"ab": "correct option shown as A", "ba": "correct option shown as B"}


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> None:
    study = pd.read_csv(SCREEN / "medxpertqa_study20.csv")
    prompts = pd.read_csv(SCREEN / "medxpertqa_prompts.csv").set_index("item_id")
    pair_ids = list(study.pair_id)
    tasks = {it["item_id"]: it for it in load_tasks() if it["pair_id"] in set(pair_ids)}
    item_ids = set(tasks)
    nothink = {r["item_id"]: r for run in RUNS["nothink"] for r in iter_traces(run) if r["item_id"] in item_ids}
    think: dict[str, list[dict]] = {iid: [] for iid in item_ids}
    for run in RUNS["think"]:
        for r in iter_traces(run):
            if r["item_id"] in item_ids:
                think[r["item_id"]].append(r)
    for iid in think:
        think[iid].sort(key=lambda r: r["sample"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs").mkdir(exist_ok=True)
    with (OUT / "nothink.jsonl").open("w", encoding="utf-8") as f:
        for iid in sorted(nothink):
            f.write(json.dumps(nothink[iid], ensure_ascii=False) + "\n")
    with (OUT / "traces.jsonl").open("w", encoding="utf-8") as f:
        for iid in sorted(think):
            for r in think[iid]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- index
    cols = ["group", "pair_id", "source_id", "body_system", "correct_text", "distractor_text", "share_correct_ab", "share_correct_ba",
            "nothink_gap_ab", "nothink_gap_ba", "think_gap_median_ab", "think_gap_median_ba", "tok_p50_ab", "tok_p50_ba"]
    lines = ["# Study set: 20 pairs", "", "Columns: thinking accuracy, no-thinking gap and thinking gap (median of 30) per order "
             "(ab = correct option shown as A, ba = shown as B), median thinking tokens. One file per pair in pairs/.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in study.iterrows():
        lines.append("| " + " | ".join(f"[{r[c]}](pairs/{r[c]}.md)" if c == "pair_id" else fmt(r[c]) for c in cols) + " |")
    (OUT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- per pair markdown + html
    html_parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Study set 20 pairs</title><style>"
                  "body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:13px;max-width:1100px;margin:20px auto;padding:0 16px;color:#111}"
                  "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:2px 6px;font-size:12px;text-align:right}"
                  "th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap;background:#f6f6f4;padding:8px;border:1px solid #ddd;font-size:12px}"
                  "details{margin:4px 0}summary{cursor:pointer}.wrong{background:#f3dede}.right{background:#e3efe1}"
                  "nav a{margin-right:10px}h2{border-top:2px solid #333;padding-top:10px}</style></head><body>",
                  "<h1>Study set: 20 pairs</h1><nav>" + " ".join(f"<a href='#{html.escape(p)}'>{html.escape(p)}</a>" for p in pair_ids) + "</nav>"]
    for _, srow in study.iterrows():
        pid = srow.pair_id
        items = {o: tasks[f"medxpertqa:{pid}:{o}"] for o in ("ab", "ba")}
        md = [f"# {pid}", "", f"Group {srow.group}: {srow.group_label}. Source question {srow.source_id} ({srow.body_system}, {srow.question_type}).",
              f"Correct option: **{srow.correct_text}**. Distractor: **{srow.distractor_text}**.", ""]
        hp = [f"<h2 id='{html.escape(pid)}'>{html.escape(pid)}</h2><p>Group {srow.group}: {html.escape(srow.group_label)}. Source {srow.source_id} "
              f"({html.escape(str(srow.body_system))}, {html.escape(str(srow.question_type))}).<br>Correct option: <b>{html.escape(srow.correct_text)}</b>. "
              f"Distractor: <b>{html.escape(srow.distractor_text)}</b>.</p>"]
        stem = items["ab"]["messages"][0]["content"]
        stem_only = stem.rsplit("\nA. ", 1)[0]
        md += ["## Question", "", "```", stem_only, "```", ""]
        hp.append(f"<details><summary>Question stem</summary><pre>{html.escape(stem_only)}</pre></details>")
        for o in ("ab", "ba"):
            it, iid = items[o], f"medxpertqa:{pid}:{o}"
            nt, ths, scr = nothink[iid], think[iid], prompts.loc[iid]
            opts = stem[len(stem_only) + 1:]
            md += [f"## Order {o}: {ORDER_LABEL[o]} (label {it['label']})", "", "```", opts, "```", "",
                   f"**No thinking (greedy):** answer {nt['answer']} ({'correct' if nt['correct'] else 'wrong'}), "
                   f"p_A {fmt(nt['p_A'], 4)}, p_B {fmt(nt['p_B'], 4)}, logp_A {fmt(nt['logp_A'])}, logp_B {fmt(nt['logp_B'])}, "
                   f"gap {fmt(nt['gap'])}; text: `{nt['text']!r}`", "",
                   f"**Thinking (30 samples):** accuracy {fmt(scr.share_correct, 3)}, class {scr.cls} ({scr.reason if isinstance(scr.reason, str) else ''}), "
                   f"gap median {fmt(scr.think_gap_median)} (min {fmt(scr.think_gap_min)}, max {fmt(scr.think_gap_max)}), "
                   f"share gap up {fmt(scr.share_gap_up)}, down {fmt(scr.share_gap_down)}, think tokens median {fmt(scr.think_tokens_median, 0)}, "
                   f"truncated {scr.n_truncated}", "",
                   "| sample | answer | correct | gap | logp_A | logp_B | think tokens | total tokens | finish | format |", "|---|---|---|---|---|---|---|---|---|---|"]
            hp.append(f"<h3>Order {o}: {ORDER_LABEL[o]} (label {it['label']})</h3><pre>{html.escape(opts)}</pre>"
                      f"<p><b>No thinking (greedy):</b> answer {nt['answer']} ({'correct' if nt['correct'] else 'wrong'}), p_A {fmt(nt['p_A'], 4)}, "
                      f"p_B {fmt(nt['p_B'], 4)}, logp_A {fmt(nt['logp_A'])}, logp_B {fmt(nt['logp_B'])}, gap {fmt(nt['gap'])}; text <code>{html.escape(repr(nt['text']))}</code></p>"
                      f"<p><b>Thinking (30 samples):</b> accuracy {fmt(scr.share_correct, 3)}, class {scr.cls} ({html.escape(scr.reason) if isinstance(scr.reason, str) else ''}), "
                      f"gap median {fmt(scr.think_gap_median)} (min {fmt(scr.think_gap_min)}, max {fmt(scr.think_gap_max)}), share gap up {fmt(scr.share_gap_up)}, "
                      f"down {fmt(scr.share_gap_down)}, think tokens median {fmt(scr.think_tokens_median, 0)}, truncated {scr.n_truncated}</p>"
                      "<table><tr><th>sample</th><th>answer</th><th>correct</th><th>gap</th><th>logp_A</th><th>logp_B</th><th>think tokens</th><th>total</th><th>finish</th><th>format</th></tr>")
            for r in ths:
                md.append(f"| {r['sample']} | {r['answer']} | {r['correct']} | {fmt(r['gap'])} | {fmt(r['logp_A'])} | {fmt(r['logp_B'])} | "
                          f"{r['think_tokens']} | {r['n_tokens']} | {r['finish_reason']} | {r['answer_format']} |")
                cls = "right" if r["correct"] else ("wrong" if r["correct"] is False else "")
                hp.append(f"<tr class='{cls}'><td>{r['sample']}</td><td>{r['answer']}</td><td>{r['correct']}</td><td>{fmt(r['gap'])}</td><td>{fmt(r['logp_A'])}</td>"
                          f"<td>{fmt(r['logp_B'])}</td><td>{r['think_tokens']}</td><td>{r['n_tokens']}</td><td>{r['finish_reason']}</td><td>{r['answer_format']}</td></tr>")
            hp.append("</table>")
            md += ["", f"### Full samples, order {o}", ""]
            for r in ths:
                tag = f"sample {r['sample']}: answer {r['answer']} ({'correct' if r['correct'] else 'wrong'}), gap {fmt(r['gap'])}, {r['think_tokens']} thinking tokens"
                md += [f"#### {tag}", "", "```", r["text"], "```", ""]
                cls = "right" if r["correct"] else ("wrong" if r["correct"] is False else "")
                hp.append(f"<details class='{cls}'><summary>{html.escape(tag)}</summary><pre>{html.escape(r['text'])}</pre></details>")
        (OUT / "pairs" / f"{pid}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        html_parts.extend(hp)
    html_parts.append("</body></html>")
    (OUT / "study20.html").write_text("".join(html_parts), encoding="utf-8")
    n_tr = sum(len(v) for v in think.values())
    print(f"{len(pair_ids)} pairs, {len(nothink)} no-thinking records, {n_tr} thinking records -> {OUT}/ (index.md, pairs/*.md, study20.html, nothink.jsonl, traces.jsonl)")


if __name__ == "__main__":
    main()
