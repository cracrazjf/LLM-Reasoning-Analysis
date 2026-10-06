"""Give every MedXpertQA pair (correct diagnosis vs one distractor) a distance, from three sources.

    python src/pair_distance.py icd                                   # local + NLM lookups (cached)
    python src/pair_distance.py prior --plan-only                      # offline: render the 10-option prompts
    python src/pair_distance.py prior --out runs/medxpertqa/prior-qwen3-8b          # GPU
    python src/pair_distance.py summarise --screen runs/medxpertqa/screen-qwen3-8b \\
        --prior runs/medxpertqa/prior-qwen3-8b                        # local

Sources
  screen  accuracy of the model's own answers on each pair, both orders, from a screening
          run of src/generate.py over data/selections/medxpertqa_screen.json (a few samples
          per prompt): the behavioural difficulty and the basis for choosing pairs.
  prior   the model's next-token distribution over the ten letters of the ORIGINAL
          10-option question read with an empty thinking block ("<think>\\n\\n</think>\\n\\n",
          the start readout of src/readout.py): one forward pass per question. The gap
          log p(distractor) - log p(correct) says how close the model itself finds the pair
          before any reasoning.
  icd     ICD-10-CM codes of both diagnoses (NLM Clinical Tables icd10cm and conditions
          tables; every lookup cached in data/raw/medxpertqa/icd10_cache.json so a rerun is
          offline; manual codes in data/raw/medxpertqa/icd10_overrides.json win) and the
          deepest level the two codes share: 0 different chapter, 1 same chapter, 2 same
          3-character category, 3 same 4-character subcategory, 4 identical code.
          Model-independent.

Outputs: data/pairs/medxpertqa_icd10.json (icd); <out>/prior.jsonl + manifest.json (prior);
data/pairs/medxpertqa_distance.csv, summary.json and fig_*.png (summarise). Pairs are nested
in questions, so the summary compares the sources WITHIN question (Spearman per question,
median over questions) and reports pooled numbers only as a side figure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from generate import ANSWER_TOP_LOGPROBS, MODEL, Tokens, chat_ids, iter_traces, load_llm, parse_answer  # noqa: E402
from medxpertqa_dataset import (  # noqa: E402
    RAW_DIR, ROOT, SOURCE_LETTERS, TASKS_PATH, load_source, read_jsonl, split_stem, stable_hash,
)

PAIRS_DIR = ROOT / "data" / "pairs"
ICD_PATH = PAIRS_DIR / "medxpertqa_icd10.json"
ICD_CACHE_PATH = RAW_DIR / "icd10_cache.json"
ICD_OVERRIDES_PATH = RAW_DIR / "icd10_overrides.json"
DISTANCE_CSV = PAIRS_DIR / "medxpertqa_distance.csv"
SUMMARY_PATH = PAIRS_DIR / "summary.json"

# The comparison task's answer line extended to the ten source letters (docs/PROMPT_SOURCES.md).
TEN_ANSWER_INSTRUCTION = "Answer with only A, B, C, D, E, F, G, H, I, or J. No other words."
PRIOR_TOP_LOGPROBS = 50  # the ten letters must all be among the returned candidates


# ------------------------------------------------------------------ pairs


def load_pairs() -> dict[str, dict[str, Any]]:
    """pair_id -> question and option fields (from the ab item of each pair)."""
    pairs = {}
    for it in read_jsonl(TASKS_PATH):
        m = it["meta"]
        if m["order"] != "ab":
            continue
        pairs[m["pair_id"]] = {
            "pair_id": m["pair_id"], "source_id": m["source_id"], "body_system": m["body_system"],
            "question_type": m["question_type"], "correct": m["correct"]["letter"],
            "distractor": m["distractor"]["letter"], "correct_text": m["correct"]["text"],
            "distractor_text": m["distractor"]["text"],
        }
    return pairs


# -------------------------------------------------------------------- icd

NLM = "https://clinicaltables.nlm.nih.gov/api"
# ICD-10-CM chapters by code range (inclusive, compared as strings on the 3-character category).
CHAPTERS = [
    ("A00", "B99", "I"), ("C00", "D49", "II"), ("D50", "D89", "III"), ("E00", "E89", "IV"), ("F01", "F99", "V"),
    ("G00", "G99", "VI"), ("H00", "H59", "VII"), ("H60", "H95", "VIII"), ("I00", "I99", "IX"), ("J00", "J99", "X"),
    ("K00", "K95", "XI"), ("L00", "L99", "XII"), ("M00", "M99", "XIII"), ("N00", "N99", "XIV"), ("O00", "O9A", "XV"),
    ("P00", "P96", "XVI"), ("Q00", "Q99", "XVII"), ("R00", "R99", "XVIII"), ("S00", "T88", "XIX"),
    ("U00", "U85", "XXII"), ("V00", "Y99", "XX"), ("Z00", "Z99", "XXI"),
]
GENERIC_WORDS = {"syndrome", "disease", "disorder", "infection", "deficiency", "type", "primary", "secondary",
                 "idiopathic", "the", "of", "with", "and", "due", "to", "in", "a", "an", "or"}
HIT_NOISE = GENERIC_WORDS | {"unspecified", "other", "nos", "specified", "without", "not", "elsewhere",
                             "classified", "site", "unilateral", "bilateral", "initial", "encounter", "subsequent"}
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu"}
LEVEL_NAMES = {0: "different chapter", 1: "same chapter", 2: "same category", 3: "same subcategory", 4: "same code"}
ADMIN_WORDS = {"screening", "encounter", "history", "status", "carrier", "contact", "exposure", "examination"}
ADMIN_NAME = re.compile(r"^(family|personal) history|^encounter for|^contact with|^carrier of|screening", re.I)


def normalise(text: str) -> str:
    """Query form of a diagnosis name: Greek letters spelled out, possessives and parentheticals dropped."""
    t = "".join(GREEK.get(ch, ch) for ch in text).replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"\s*\([^)]*\)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def words(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", normalise(text))]


def key_words(text: str, noise: set[str] = GENERIC_WORDS) -> list[str]:
    """Words that carry the diagnosis: no generic words; single characters only when they are
    a capital letter or a digit in the original ("hepatitis B", "Fragile X", "C4")."""
    out = []
    for t in re.findall(r"[A-Za-z0-9]+", normalise(text)):
        w = t.lower()
        keep = (t.isupper() or t.isdigit()) if len(w) == 1 else w not in noise
        if keep:
            out.append(w)
    return out


def _has(word: str, pool: set[str]) -> bool:
    return any(x == word or (len(word) > 4 and len(x) > 4 and (x.startswith(word) or word.startswith(x))) for x in pool)


def match_hit(query: str, hit_name: str) -> dict[str, Any]:
    """How well an ICD entry's name matches a diagnosis name."""
    q = key_words(query)
    pool = set(words(hit_name))
    if not q:
        return {"coverage": 0.0, "exact": False, "head": False, "ok": False}
    cov = sum(_has(w, pool) for w in q) / len(q)
    exact = set(key_words(hit_name, HIT_NOISE)) == set(q)
    head, first = _has(q[-1], pool), _has(q[0], pool)
    return {"coverage": cov, "exact": exact, "head": head,
            "ok": exact or cov == 1.0 or (cov >= 0.5 and head and first)}


def pick_code(query: str, hits: list[list[str]]) -> dict[str, Any] | None:
    """Code for a query from search hits [[code(s), name], ...], or None if no hit matches well enough.

    Hits are scored by word coverage of the diagnosis name. Exact matches win; otherwise
    every acceptable hit counts and the code is their common prefix (e.g. the C92 category
    for the chronic myeloid leukemia entries). Hits spread over categories give the majority
    category or nothing: a miss is visible, a wrong code is not.
    """
    scored = []
    admin = bool(ADMIN_WORDS & set(words(query)))  # the name itself is about screening, history, encounters
    for h in hits:
        codes = [c.strip() for c in (h[0] or "").split(",") if c.strip()]
        if not codes:
            continue
        if not admin and (all(c.upper().startswith("Z") for c in codes) or ADMIN_NAME.match(h[1])):
            continue  # "Encounter for screening for ...", "Family history of ...": not a diagnosis
        m = match_hit(query, h[1])
        if m["ok"]:
            scored.append((m, codes, h[1]))
    if not scored:
        return None
    if any(m["exact"] for m, _, _ in scored):
        scored = [s for s in scored if s[0]["exact"]]
    codes = [c for _, cs, _ in scored for c in cs]
    code = common_prefix(codes)
    if len(code) < 3:  # hits spread over categories: take a category only if two thirds of the hits share it
        cats = Counter(c.replace(".", "")[:3] for c in codes)
        cat, n = cats.most_common(1)[0]
        if n < 2 * len(codes) / 3:
            return None
        code = common_prefix([c for c in codes if c.replace(".", "")[:3] == cat])
    if len(code) < 3:
        return None
    return {"code": code, "hits": [[cs, n, round(m["coverage"], 2)] for m, cs, n in scored[:5]]}


def chapter(code: str) -> str | None:
    cat = code.replace(".", "")[:3].upper()
    for lo, hi, name in CHAPTERS:
        if lo <= cat <= hi:
            return name
    return None


def shared_level(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    x, y = a.replace(".", "").upper(), b.replace(".", "").upper()
    if chapter(x) != chapter(y) or chapter(x) is None:
        return 0
    common = 0
    for p, q in zip(x, y):
        if p != q:
            break
        common += 1
    if x == y and len(x) >= 4:
        return 4
    if common >= 4:
        return 3
    if common >= 3:
        return 2
    return 1


def common_prefix(codes: list[str]) -> str:
    if not codes:
        return ""
    first = codes[0]
    n = 0
    while n < len(first) and all(len(c) > n and c[n] == first[n] for c in codes):
        n += 1
    return first[:n].rstrip(".")


class IcdLookup:
    """NLM Clinical Tables lookups with an on-disk cache."""

    def __init__(self, cache_path: Path, offline: bool = False) -> None:
        self.cache_path = cache_path
        self.cache: dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.offline = offline
        self.calls = 0

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False, sort_keys=True) + "\n")

    def query(self, table: str, terms: str) -> list[list[str]]:
        key = f"{table}|{terms}"
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            raise SystemExit(f"not cached and --offline: {key}")
        import requests

        params = {"terms": terms, "maxList": 7}
        if table == "icd10cm":
            params.update({"sf": "code,name", "df": "code,name"})
        else:  # conditions: consumer health names with ICD-10-CM mappings
            params.update({"sf": "primary_name,consumer_name,synonyms", "df": "primary_name,icd10cm_codes"})
        for attempt in range(4):
            try:
                r = requests.get(f"{NLM}/{table}/v3/search", params=params, timeout=30)
                r.raise_for_status()
                hits = r.json()[3]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2.0 * (attempt + 1))
                print(f"  retry {terms!r}: {e}", file=sys.stderr)
        self.calls += 1
        time.sleep(0.15)
        self.cache[key] = hits
        if self.calls % 50 == 0:
            self.save()
        return hits


def shortened(name: str) -> list[str]:
    """Shorter search terms to try when the full name finds nothing acceptable."""
    base = normalise(name)
    kept = key_words(name)
    out = []
    if base.lower() != name.lower():
        out.append(base)
    if kept and " ".join(kept).lower() != base.lower():
        out.append(" ".join(kept))
    seen, uniq = set(), []
    for t in out:
        if t.lower() not in seen and t.lower() != name.lower():
            seen.add(t.lower())
            uniq.append(t)
    return uniq


def resolve_code(name: str, lookup: IcdLookup, overrides: dict[str, str]) -> dict[str, Any]:
    """ICD-10-CM code for a diagnosis name, with the search that produced it."""
    if name in overrides:
        return {"code": overrides[name], "method": "override", "terms": name, "hits": []}
    query = normalise(name)
    picked = pick_code(name, lookup.query("icd10cm", query))
    if picked:
        return {"method": "icd10cm", "terms": query, **picked}
    picked = pick_code(name, [[h[1], h[0]] for h in lookup.query("conditions", query)])
    if picked:
        return {"method": "conditions", "terms": query, **picked}
    for terms in shortened(name):
        picked = pick_code(name, lookup.query("icd10cm", terms))
        if picked:
            return {"method": "icd10cm_short", "terms": terms, **picked}
    return {"code": None, "method": "none", "terms": query, "hits": []}


def cmd_icd(args: argparse.Namespace) -> None:
    pairs = load_pairs()
    names = sorted({t for p in pairs.values() for t in (p["correct_text"], p["distractor_text"])})
    overrides = json.loads(ICD_OVERRIDES_PATH.read_text()) if ICD_OVERRIDES_PATH.exists() else {}
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_") and v}  # null = still to fill
    lookup = IcdLookup(ICD_CACHE_PATH, offline=args.offline)
    codes: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(names, 1):
        codes[name] = resolve_code(name, lookup, overrides)
        if i % 100 == 0:
            print(f"  {i}/{len(names)} names, {lookup.calls} API calls", flush=True)
    lookup.save()
    rows = {}
    for pid, p in pairs.items():
        a, b = codes[p["correct_text"]]["code"], codes[p["distractor_text"]]["code"]
        rows[pid] = {"code_correct": a, "code_distractor": b, "level": shared_level(a, b)}
    by_method = defaultdict(int)
    for c in codes.values():
        by_method[c["method"]] += 1
    levels = defaultdict(int)
    for r in rows.values():
        levels[LEVEL_NAMES.get(r["level"], "unknown")] += 1
    out = {"source": "NLM Clinical Tables icd10cm + conditions (cached), manual overrides",
           "n_names": len(names), "by_method": dict(by_method), "pairs_by_level": dict(levels),
           "names": codes, "pairs": rows}
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    ICD_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    missing = [n for n, c in codes.items() if c["code"] is None]
    print(f"{len(names)} names: {dict(by_method)}; pairs by level {dict(levels)} -> {ICD_PATH.relative_to(ROOT)}")
    print(f"{len(missing)} names without a code (add them to {ICD_OVERRIDES_PATH.relative_to(ROOT)}):")
    for n in missing:
        print(f"  {n}")


# ------------------------------------------------------------------ prior


def ten_option_prompts() -> list[dict[str, Any]]:
    """One 10-option prompt per kept source question."""
    kept = sorted({p["source_id"] for p in load_pairs().values()}, key=lambda s: int(s.split("-")[1]))
    source = {row["id"]: row for row in load_source()}
    out = []
    for sid in kept:
        row = source[sid]
        stem = split_stem(row["question"])
        lines = [stem] + [f"{L}. {row['options'][L].strip()}" for L in SOURCE_LETTERS]
        content = "\n".join(lines) + "\n\n" + TEN_ANSWER_INSTRUCTION
        messages = [{"role": "user", "content": content}]
        out.append({"source_id": sid, "label": row["label"], "messages": messages, "prompt_hash": stable_hash(messages)})
    return out


def cmd_prior(args: argparse.Namespace) -> None:
    prompts = ten_option_prompts()
    print(f"{len(prompts)} ten-option prompts; example:\n\n{prompts[0]['messages'][0]['content']}\n")
    if args.plan_only:
        return

    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    llm, model_path = load_llm(args.max_num_seqs, max_logprobs=PRIOR_TOP_LOGPROBS)
    engine = llm.llm_engine
    tok = llm.get_tokenizer()
    tokens = Tokens(tok, tuple(SOURCE_LETTERS))
    empty_think = tok("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=PRIOR_TOP_LOGPROBS,
                            output_kind=RequestOutputKind.FINAL_ONLY)
    for p in prompts:
        engine.add_request(p["source_id"], {"prompt_token_ids": chat_ids(tok, p["messages"]) + empty_think}, params)
    got: dict[str, dict[int, Any]] = {}
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                got[o.request_id] = o.outputs[0].logprobs[0]
    rows = []
    for p in prompts:
        top = {k: v.logprob for k, v in got[p["source_id"]].items()}
        floor = min(top.values())
        logp = {L: top.get(tokens.options[L]) for L in SOURCE_LETTERS}
        missing = [L for L, v in logp.items() if v is None]
        filled = {L: (v if v is not None else floor) for L, v in logp.items()}  # bound for letters outside the top list
        import math
        z = math.log(sum(math.exp(v) for v in filled.values()))
        correct = p["label"]
        rows.append({
            "source_id": p["source_id"], "prompt_hash": p["prompt_hash"], "correct": correct,
            "logp": {L: round(v, 4) for L, v in filled.items()}, "missing": missing,
            "p_letters": round(sum(math.exp(v) for v in filled.values()), 6),  # mass on the ten letters
            "p": {L: round(math.exp(v - z), 6) for L, v in filled.items()},
            "gap": {L: round(filled[L] - filled[correct], 4) for L in SOURCE_LETTERS if L != correct},
            "rank": {L: 1 + sum(filled[M] > filled[L] for M in SOURCE_LETTERS) for L in SOURCE_LETTERS},
        })
    (out / "prior.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "manifest.json").write_text(json.dumps({
        "model": MODEL, "model_revision": Path(model_path).name, "n_questions": len(rows),
        "readout": "empty thinking block, greedy next token, top-%d logprobs" % PRIOR_TOP_LOGPROBS,
        "instruction": TEN_ANSWER_INSTRUCTION, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1) + "\n")
    top1 = sum(r["rank"][r["correct"]] == 1 for r in rows)
    print(f"done: {len(rows)} questions, correct letter ranked first in {top1}; "
          f"{sum(bool(r['missing']) for r in rows)} with letters outside the top list -> {out}")


# -------------------------------------------------------------- summarise


def spearman(x: list[float], y: list[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).correlation)


def within_question(df: Any, x: str, y: str) -> dict[str, Any]:
    """Spearman(x, y) per question, summarised over questions."""
    import numpy as np

    rhos = []
    for _, g in df.dropna(subset=[x, y]).groupby("source_id"):
        r = spearman(g[x].tolist(), g[y].tolist())
        if r is not None and not np.isnan(r):
            rhos.append(r)
    if not rhos:
        return {"n_questions": 0}
    return {"n_questions": len(rhos), "median": float(np.median(rhos)),
            "q25": float(np.percentile(rhos, 25)), "q75": float(np.percentile(rhos, 75)),
            "share_positive": float(np.mean([r > 0 for r in rhos]))}


def cmd_summarise(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    pairs = load_pairs()
    stats: dict[str, dict[str, Any]] = {pid: {"n": 0, "n_answered": 0, "n_correct": 0, "think": [],
                                              "ab": [0, 0], "ba": [0, 0]} for pid in pairs}
    n_traces = 0
    formats: dict[str, int] = defaultdict(int)
    for r in iter_traces(args.screen, ("pair_id", "order", "label", "answer_text", "think_tokens", "finish_reason")):
        s = stats.get(r["pair_id"])
        if s is None:
            continue
        n_traces += 1
        s["n"] += 1
        answer, fmt = parse_answer(r["answer_text"], ("A", "B"))  # re-derived: same rule for every stored run
        formats[fmt or "none"] += 1
        if answer is not None:
            correct = answer == r["label"]
            s["n_answered"] += 1
            s["n_correct"] += correct
            s[r["order"]][0] += correct
            s[r["order"]][1] += 1
        if r["think_tokens"] is not None:
            s["think"].append(r["think_tokens"])
    if n_traces == 0:
        raise SystemExit(f"no traces of these pairs in {args.screen}")

    prior: dict[tuple[str, str], dict[str, float]] = {}
    if args.prior:
        for r in read_jsonl(Path(args.prior) / "prior.jsonl"):
            for L, gap in r["gap"].items():
                prior[(r["source_id"], L)] = {"prior_gap": gap, "prior_p_distractor": r["p"][L],
                                              "prior_rank_distractor": r["rank"][L], "prior_p_correct": r["p"][r["correct"]]}
    icd = json.loads(ICD_PATH.read_text())["pairs"] if ICD_PATH.exists() else {}

    rows = []
    for pid, p in pairs.items():
        s = stats[pid]
        row = dict(p)
        row.update({
            "n_traces": s["n"], "n_answered": s["n_answered"],
            "acc": s["n_correct"] / s["n_answered"] if s["n_answered"] else None,
            "acc_ab": s["ab"][0] / s["ab"][1] if s["ab"][1] else None,
            "acc_ba": s["ba"][0] / s["ba"][1] if s["ba"][1] else None,
            "think_tokens_mean": float(np.mean(s["think"])) if s["think"] else None,
        })
        row.update(prior.get((p["source_id"], p["distractor"]), {}))
        c = icd.get(pid, {})
        row.update({"icd_correct": c.get("code_correct"), "icd_distractor": c.get("code_distractor"),
                    "icd_level": c.get("level")})
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = args.out or PAIRS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, summary_path = out_dir / DISTANCE_CSV.name, out_dir / SUMMARY_PATH.name
    df.to_csv(csv_path, index=False)

    answered = df.dropna(subset=["acc"])
    summary: dict[str, Any] = {
        "screen_run": str(args.screen), "n_traces": n_traces, "n_pairs": len(df),
        "n_pairs_with_answers": int(len(answered)), "n_questions": int(df["source_id"].nunique()),
        "traces_per_pair": {"min": int(df["n_traces"].min()), "max": int(df["n_traces"].max())},
        "answer_formats": dict(formats),
        "accuracy": {
            "mean": float(answered["acc"].mean()),
            "share_at_1": float((answered["acc"] == 1).mean()),
            "share_in_[0.2,0.8]": float(answered["acc"].between(0.2, 0.8).mean()),
            "share_below_0.5": float((answered["acc"] < 0.5).mean()),
            "order_effect_mean_abs(acc_ab-acc_ba)": float((answered["acc_ab"] - answered["acc_ba"]).abs().mean()),
        },
        "within_question_spread": {
            "mean_acc_range_over_9_pairs": float(answered.groupby("source_id")["acc"].agg(lambda g: g.max() - g.min()).mean()),
            "questions_with_a_pair_at_1_and_a_pair_below_0.5": int(
                answered.groupby("source_id")["acc"].agg(lambda g: (g.max() == 1) and (g.min() < 0.5)).sum()),
        },
        "think_tokens_mean": float(df["think_tokens_mean"].dropna().mean()),
    }
    if "prior_gap" in df:
        summary["prior_vs_accuracy"] = {
            "within_question_spearman": within_question(df, "prior_gap", "acc"),
            "pooled_spearman": spearman(answered.dropna(subset=["prior_gap"])["prior_gap"].tolist(),
                                        answered.dropna(subset=["prior_gap"])["acc"].tolist()),
        }
    if df["icd_level"].notna().any():
        by_level = answered.dropna(subset=["icd_level"]).groupby("icd_level")["acc"].agg(["mean", "count"])
        summary["icd_vs_accuracy"] = {
            "pairs_with_both_codes": int(df["icd_level"].notna().sum()),
            "accuracy_by_level": {LEVEL_NAMES[int(k)]: {"mean": float(v["mean"]), "n": int(v["count"])}
                                  for k, v in by_level.iterrows()},
            "within_question_spearman": within_question(df, "icd_level", "acc"),
        }
        if "prior_gap" in df:
            summary["icd_vs_prior"] = {"within_question_spearman": within_question(df, "icd_level", "prior_gap")}
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    make_figures(df, answered, out_dir)
    print(json.dumps(summary, indent=1))
    print(f"-> {csv_path}, {summary_path}, {out_dir}/fig_*.png")


def make_figures(df: Any, answered: Any, out_dir: Path) -> None:
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(answered["acc"], bins=np.linspace(0, 1, 11), color="#4C72B0")
    ax.set_xlabel("pair accuracy (screening)")
    ax.set_ylabel("pairs")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_accuracy_hist.png", dpi=150)
    plt.close(fig)

    if "prior_gap" in df and df["prior_gap"].notna().any():
        d = answered.dropna(subset=["prior_gap"])
        fig, ax = plt.subplots(figsize=(5, 3.6))
        colours = pd.to_numeric(d["icd_level"], errors="coerce").fillna(-1)
        sc = ax.scatter(d["prior_gap"], d["acc"], c=colours, cmap="viridis", s=14, alpha=0.7)
        ax.set_xlabel("prior gap: log p(distractor) - log p(correct), 10-option empty-thinking readout")
        ax.set_ylabel("pair accuracy (screening)")
        fig.colorbar(sc, ax=ax, label="ICD-10 shared level (-1 unknown)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_prior_vs_accuracy.png", dpi=150)
        plt.close(fig)

    if df["icd_level"].notna().any():
        d = answered.dropna(subset=["icd_level"])
        levels = sorted(d["icd_level"].unique())
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.boxplot([d[d["icd_level"] == lv]["acc"] for lv in levels], tick_labels=[LEVEL_NAMES[int(lv)] for lv in levels])
        ax.set_ylabel("pair accuracy (screening)")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_icd_vs_accuracy.png", dpi=150)
        plt.close(fig)



# ----------------------------------------------------------------- choose

MAIN_SEED = "20261001:medxpertqa_main"
HARD_MAX, MID_LO, MID_HI, PERFECT = 0.3, 0.3, 0.7, 1.0
BOTH_LO, BOTH_HI = 0.25, 0.75  # per-order accuracy range (8 samples) that counts as varying within the order
PILOT_CONDITIONS = ["C0_baseline", "C1_caution_think", "C2_speed_think", "C3_no_think", "C4_draft"]


def cmd_choose(args: argparse.Namespace) -> None:
    """Main-run selection from the screening accuracies: per question one hard pair (accuracy
    <= 0.3), one middle pair (0.3 < accuracy <= 0.7) and one perfect pair (accuracy 1.0), each
    drawn at random (seeded) among the question's pairs in that band. Questions missing any
    band are skipped. Both orders of every chosen pair, one condition."""
    import random

    import pandas as pd

    from medxpertqa_dataset import PROMPTS_PATH, SELECTIONS_DIR, sha256_file

    df = pd.read_csv(DISTANCE_CSV)
    items = {it["meta"]["pair_id"] + ":" + it["meta"]["order"]: it for it in read_jsonl(TASKS_PATH)}
    prompts = {p["item_id"]: p for p in read_jsonl(PROMPTS_PATH) if p["condition"] == args.condition}
    chosen, skipped, mid_kind = [], [], {}
    for sid, g in df.groupby("source_id", sort=True):
        rng = random.Random(f"{MAIN_SEED}:{sid}")
        hard = sorted(g[g["acc"] <= HARD_MAX]["pair_id"])
        mid = sorted(g[(g["acc"] > MID_LO) & (g["acc"] <= MID_HI)]["pair_id"])
        # prefer mid pairs whose answers vary within BOTH orders (not a letter preference averaged out)
        mid_both = sorted(g[(g["acc"] > MID_LO) & (g["acc"] <= MID_HI) & g["acc_ab"].between(BOTH_LO, BOTH_HI)
                            & g["acc_ba"].between(BOTH_LO, BOTH_HI)]["pair_id"])
        perfect = sorted(g[g["acc"] >= PERFECT]["pair_id"])
        if not hard or not mid or not perfect:
            skipped.append(sid)
            continue
        h, m, e = rng.choice(hard), rng.choice(mid_both or mid), rng.choice(perfect)
        mid_kind[sid] = "both_orders" if mid_both else "pooled"
        for band, pid in (("hard", h), ("mid", m), ("perfect", e)):
            row = g[g["pair_id"] == pid].iloc[0]
            chosen.append((band, pid, float(row["acc"]), float(row["prior_gap"]) if pd.notna(row["prior_gap"]) else None))
    rows = []
    for band, pid, acc, gap in chosen:
        for order in ("ab", "ba"):
            it = items[f"{pid}:{order}"]
            m, p = it["meta"], prompts[it["item_id"]]
            rows.append({
                "prompt_id": p["prompt_id"], "item_id": it["item_id"], "pair_id": pid, "source_id": m["source_id"],
                "order": order, "band": band, "mid_kind": mid_kind[m["source_id"]] if band == "mid" else None,
                "screen_acc": acc, "prior_gap": gap,
                "correct": m["correct"]["letter"], "distractor": m["distractor"]["letter"],
                "correct_text": m["correct"]["text"], "distractor_text": m["distractor"]["text"],
                "body_system": m["body_system"], "question_type": m["question_type"],
                "options": p["options"], "label": p["label"], "messages": p["messages"],
            })
    out = {
        "task": "medxpertqa", "purpose": "main", "condition": args.condition, "select_seed": MAIN_SEED,
        "rule": {"hard": f"random pair with screening accuracy <= {HARD_MAX}",
                 "mid": f"random pair with accuracy in ({MID_LO}, {MID_HI}], preferring pairs with both orders in [{BOTH_LO}, {BOTH_HI}]",
                 "perfect": "random pair with accuracy 1.0"},
        "mid_kind_counts": {k: sum(1 for v in mid_kind.values() if v == k) for k in ("both_orders", "pooled")},
        "screen_run": str(args.screen), "n_questions": len(chosen) // 3, "skipped_questions": skipped,
        "source": {"tasks_sha256": sha256_file(TASKS_PATH), "prompts_sha256": sha256_file(PROMPTS_PATH),
                   "distance_csv_sha256": sha256_file(DISTANCE_CSV)},
        "prompts": rows,
    }
    path = SELECTIONS_DIR / "medxpertqa_main.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    import statistics as st
    for band in ("hard", "mid", "perfect"):
        accs = [a for b, _, a, _ in chosen if b == band]
        print(f"  {band:<8} {len(accs)} pairs, screening accuracy median {st.median(accs):.2f} "
              f"min {min(accs):.2f} max {max(accs):.2f}")
    print(f"{len(chosen) // 3} questions, {len(chosen)} pairs, {len(rows)} prompts; mid pairs varying within both orders: "
          f"{out['mid_kind_counts']['both_orders']}, pooled: {out['mid_kind_counts']['pooled']}; "
          f"skipped {len(skipped)} questions without all three bands -> {path.relative_to(ROOT)}")


def cmd_pilot(args: argparse.Namespace) -> None:
    """Instruction pilot: the main selection's hard and mid pairs of the questions whose mid pair varies
    within both orders, both orders, under PILOT_CONDITIONS. Rows are ordered pair, order, condition so a
    smoke test over the first prompts covers every condition."""
    from medxpertqa_dataset import PROMPTS_PATH, SELECTIONS_DIR, sha256_file

    main = json.loads((SELECTIONS_DIR / "medxpertqa_main.json").read_text())
    prompts = {(p["item_id"], p["condition"]): p for p in read_jsonl(PROMPTS_PATH)}
    both = {r["source_id"] for r in main["prompts"] if r["band"] == "mid" and r["mid_kind"] == "both_orders"}
    rows = []
    for r in main["prompts"]:
        if r["source_id"] not in both or r["band"] not in ("hard", "mid"):
            continue
        for cond in PILOT_CONDITIONS:
            p = prompts[(r["item_id"], cond)]
            row = {k: v for k, v in r.items() if k != "messages"}
            row.update({"prompt_id": p["prompt_id"], "condition": cond, "messages": p["messages"]})
            rows.append(row)
    out = {
        "task": "medxpertqa", "purpose": "pilot_instructions", "conditions": PILOT_CONDITIONS,
        "select_seed": "20261002:medxpertqa_pilot", "from": "data/selections/medxpertqa_main.json",
        "n_questions": len(both), "source": {"prompts_sha256": sha256_file(PROMPTS_PATH)},
        "prompts": rows,
    }
    path = SELECTIONS_DIR / "medxpertqa_pilot.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"pilot: {len(both)} questions, {len(rows) // len(PILOT_CONDITIONS)} prompts x {len(PILOT_CONDITIONS)} conditions "
          f"= {len(rows)} prompts -> {path.relative_to(ROOT)}")



# Pairs whose options are tests or management rather than diagnoses (the stem filter admits a few such
# questions); excluded from the helps pilot and the main run at the user's request (2026-10-03).
NON_DIAGNOSIS_PAIRS = {
    **{f"Text-164:G-{d}": "options are tests ('No additional tests are required' vs a test)" for d in "ABDEFHIJ"},
    "Text-500:F-J": "both options name the same diagnosis and differ only in the imaging test",
}


def class_pairs(band: str):
    """The pairs of one thinking-gain class in both orders (data/pairs/medxpertqa_gain.csv, src/thinking_gain.py),
    minus NON_DIAGNOSIS_PAIRS; returns the table and the excluded pairs with reasons."""
    import pandas as pd

    gain = pd.read_csv(PAIRS_DIR / "medxpertqa_gain.csv")
    chosen = gain[gain[f"{band}_both_orders"] == True].sort_values("pair_id")  # noqa: E712
    excluded = {pid: why for pid, why in NON_DIAGNOSIS_PAIRS.items() if pid in set(chosen.pair_id)}
    return chosen[~chosen.pair_id.isin(excluded)], excluded


def cmd_class_selection(args: argparse.Namespace) -> None:
    """Selection file for the pairs where reasoning helps (or hurts) in both orders: both orders x the given
    conditions; rows ordered pair, order, condition. `helps-pilot` wrote the main run's selection; `hurts` writes
    the mirror set."""
    from medxpertqa_dataset import PROMPTS_PATH, SELECTIONS_DIR, sha256_file

    chosen, excluded = class_pairs(args.band)
    items = {it["meta"]["pair_id"] + ":" + it["meta"]["order"]: it for it in read_jsonl(TASKS_PATH)}
    prompts = {(p["item_id"], p["condition"]): p for p in read_jsonl(PROMPTS_PATH)}
    conds = args.conditions.split(",")
    rows = []
    for r in chosen.itertuples():
        for order in ("ab", "ba"):
            it = items[f"{r.pair_id}:{order}"]
            m = it["meta"]
            for cond in conds:
                p = prompts[(it["item_id"], cond)]
                rows.append({
                    "prompt_id": p["prompt_id"], "item_id": it["item_id"], "pair_id": r.pair_id, "source_id": m["source_id"],
                    "order": order, "condition": cond, "band": args.band, "gain": float(r.gain),
                    "acc_think_screen": float(r.acc_think), "acc_nothink_screen": float(r.acc_nothink),
                    "correct": m["correct"]["letter"], "distractor": m["distractor"]["letter"],
                    "correct_text": m["correct"]["text"], "distractor_text": m["distractor"]["text"],
                    "body_system": m["body_system"], "question_type": m["question_type"],
                    "options": p["options"], "label": p["label"], "messages": p["messages"],
                })
    sign = ">= 0.2" if args.band == "helps" else "<= -0.2"
    out = {"task": "medxpertqa", "purpose": args.purpose, "conditions": conds, "select_seed": f"{args.seed_date}:{args.name}",
           "rule": f"pairs with {args.band}_both_orders in medxpertqa_gain.csv (thinking gain {sign} in each order), minus NON_DIAGNOSIS_PAIRS",
           "excluded": excluded,
           "n_pairs": int(len(chosen)), "n_questions": int(chosen.source_id.nunique()),
           "source": {"prompts_sha256": sha256_file(PROMPTS_PATH), "gain_csv_sha256": sha256_file(PAIRS_DIR / "medxpertqa_gain.csv")},
           "prompts": rows}
    path = SELECTIONS_DIR / f"{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(chosen)} pairs, {chosen.source_id.nunique()} questions, {len(rows)} prompts ({len(conds)} conditions) -> {path.relative_to(ROOT)}")


MAIN_CONDITIONS = ("C0_baseline", "C1_caution_think", "C2_speed_think")


def cmd_export_main(args: argparse.Namespace) -> None:
    """The main experiment's material as plain files:
      data/pairs/medxpertqa_helps70.csv, medxpertqa_hurts40.csv   one row per pair: options, screening numbers,
          and the accuracy per condition where a run exists (main run for helps, instruction pilot for hurts)
      data/prompts/conditions.json         the three conditions' wording and the prompt template
      data/prompts/medxpertqa_main.jsonl   the full prompts of both classes: pairs x 2 orders x 3 conditions
    """
    import pandas as pd

    from medxpertqa_dataset import ANSWER_INSTRUCTION, CONDITIONS, PROMPTS_PATH, user_message

    items = {it["meta"]["pair_id"] + ":" + it["meta"]["order"]: it for it in read_jsonl(TASKS_PATH)}
    prompts = {(p["item_id"], p["condition"]): p for p in read_jsonl(PROMPTS_PATH)}
    acc = {}
    for band, run in (("helps", ROOT / "runs/medxpertqa/main-qwen3-8b"), ("hurts", ROOT / "runs/medxpertqa/pilot-qwen3-8b")):
        f = run / "analysis" / ("per_prompt.csv" if band == "helps" else "per_prompt.csv")
        if f.exists():
            acc[band] = pd.read_csv(f)
    main_rows, counts = [], {}
    for band, name in (("helps", "medxpertqa_helps70"), ("hurts", "medxpertqa_hurts40")):
        chosen, _ = class_pairs(band)
        rows = []
        for r in chosen.itertuples():
            m = items[f"{r.pair_id}:ab"]["meta"]
            row = {"pair_id": r.pair_id, "source_id": r.source_id, "band": band, "body_system": m["body_system"], "question_type": m["question_type"],
                   "correct_letter": m["correct"]["letter"], "correct_text": m["correct"]["text"],
                   "distractor_letter": m["distractor"]["letter"], "distractor_text": m["distractor"]["text"],
                   "acc_think_screen": r.acc_think, "acc_think_screen_ab": r.acc_think_ab, "acc_think_screen_ba": r.acc_think_ba,
                   "acc_nothink_screen": r.acc_nothink, "gain": r.gain, "think_tokens_screen": r.think_tokens}
            if band in acc:
                a = acc[band]
                a = a[a.pair_id == r.pair_id] if "pair_id" in a else a[a.item_id.str.contains(r.pair_id + ":")]
                for cond in ("baseline", "careful", "speed"):
                    col = "cond" if "cond" in a else "condition"
                    g = a[a[col].astype(str).str.contains(cond)]
                    if len(g):
                        row[f"acc_{cond}_run"] = float((g.acc * g.n).sum() / g.n.sum()) if "n" in g else float(g.acc.mean())
                        row[f"think_{cond}_run"] = float(g.think.mean()) if "think" in g else None
            rows.append(row)
            for order in ("ab", "ba"):
                it = items[f"{r.pair_id}:{order}"]
                for cond in MAIN_CONDITIONS:
                    p = prompts[(it["item_id"], cond)]
                    main_rows.append({"prompt_id": p["prompt_id"], "item_id": it["item_id"], "pair_id": r.pair_id, "source_id": r.source_id,
                                      "band": band, "order": order, "condition": cond, "label": p["label"], "options": p["options"],
                                      "messages": p["messages"]})
        pd.DataFrame(rows).to_csv(PAIRS_DIR / f"{name}.csv", index=False)
        counts[band] = (len(rows), int(chosen.source_id.nunique()))
    example = items["Text-1074:A-B:ab"]["question"]
    conditions = {
        "answer_line": ANSWER_INSTRUCTION,
        "template": "question text, then the instruction sentence (none for the baseline), then the answer line, separated by blank lines; "
                    "sent as a single user message to Qwen3-8B in thinking mode (chat template with enable_thinking=True)",
        "conditions": {c: {"instruction": CONDITIONS[c]["instruction"], "example_user_message": user_message(example, c)} for c in MAIN_CONDITIONS},
        "sources": "docs/PROMPT_SOURCES.md",
    }
    (PROMPTS_PATH.parent / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (PROMPTS_PATH.parent / "medxpertqa_main.jsonl").open("w", encoding="utf-8") as fh:
        for row in main_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"helps {counts['helps'][0]} pairs / {counts['helps'][1]} questions, hurts {counts['hurts'][0]} pairs / {counts['hurts'][1]} questions; "
          f"{len(main_rows)} prompts -> data/pairs/medxpertqa_helps70.csv, medxpertqa_hurts40.csv, data/prompts/conditions.json, medxpertqa_main.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    i = sub.add_parser("icd", help="ICD-10-CM codes and shared level per pair (NLM lookups, cached)")
    i.add_argument("--offline", action="store_true", help="fail instead of querying for names not in the cache")
    i.set_defaults(fn=cmd_icd)
    p = sub.add_parser("prior", help="10-option empty-thinking readout per question (GPU)")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "medxpertqa" / "prior-qwen3-8b")
    p.add_argument("--plan-only", action="store_true", help="render the prompts and stop (no model)")
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.set_defaults(fn=cmd_prior)
    s = sub.add_parser("summarise", help="merge screening accuracy, prior and ICD into data/pairs/")
    s.add_argument("--screen", type=Path, required=True, help="screening run directory (src/generate.py)")
    s.add_argument("--prior", type=Path, default=None, help="prior run directory (prior.jsonl)")
    s.add_argument("--out", type=Path, default=None, help="output directory (default data/pairs)")
    s.set_defaults(fn=cmd_summarise)
    c = sub.add_parser("choose", help="main-run selection (hard / mid / perfect pair per question) from the screening")
    c.add_argument("--screen", type=Path, default=ROOT / "runs" / "medxpertqa" / "screen-qwen3-8b")
    c.add_argument("--condition", default="C0_baseline")
    c.set_defaults(fn=cmd_choose)
    pi = sub.add_parser("pilot", help="instruction-pilot selection from the main selection (hard + mid pairs, 5 conditions)")
    pi.set_defaults(fn=cmd_pilot)
    hp = sub.add_parser("helps", help="selection: the pairs where reasoning helps in both orders (the main run; written in 2026-10 as medxpertqa_helps_pilot.json, since renamed)")
    hp.add_argument("--conditions", default=",".join(MAIN_CONDITIONS))
    hp.add_argument("--name", default="medxpertqa_helps")
    hp.set_defaults(fn=cmd_class_selection, band="helps", purpose="helps_pilot", seed_date="20261003")
    hu = sub.add_parser("hurts", help="selection: the pairs where reasoning hurts in both orders (mirror of the main run)")
    hu.add_argument("--conditions", default=",".join(MAIN_CONDITIONS))
    hu.add_argument("--name", default="medxpertqa_hurts")
    hu.set_defaults(fn=cmd_class_selection, band="hurts", purpose="hurts_main", seed_date="20261005")
    ex = sub.add_parser("export-main", help="pair lists, condition wording and full prompts of the main experiment as plain files")
    ex.set_defaults(fn=cmd_export_main)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
