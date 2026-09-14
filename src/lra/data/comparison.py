"""Pairwise magnitude comparison built from Wikidata.

Design follows Lehmann et al. (EACL 2026, "Knowing the Facts but Choosing the
Shortcut"): entities of one class with one numerical attribute, top-N by
popularity. This is a custom reconstruction, not an execution of the authors'
pipeline. Differences serving our experiment include:

* pairs are binned by |log10(value_a / value_b)| so difficulty is controlled
  and the drift rate can be predicted from the ratio;
* every pair is emitted in both presentation orders (A/B swapped) and with a
  balanced "larger"/"smaller" question direction, so position bias and the
  "fame implies bigger" heuristic can be estimated as starting-point / drift
  biases instead of confounding accuracy.

Popularity is the number of Wikipedia sitelinks (a QRank-free proxy). Raw
SPARQL results are cached under data/raw/comparison/ and committed, so the
build is reproducible even as Wikidata changes.
Population qualifiers are supplemented separately without replacing cached
values. Recent dates are explicit in prompts; competing-value ranges must
preserve the ordering. Policy and scope limitations are described in README.md.
"""

from __future__ import annotations

import json
import datetime as dt
import hashlib
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from lra.data.schema import Item

log = logging.getLogger(__name__)

ENDPOINT = "https://query.wikidata.org/sparql"
OPTIONS = ["A", "B"]
DATA_VERSION = "v2"

# unit QID -> metres
LENGTH_TO_M = {"Q3710": 0.3048, "Q11573": 1.0, "Q828224": 1000.0, "Q253276": 1609.344}
TARGET_UNIT_IN_M = {"m": 1.0, "km": 1000.0}

ENTITY_QUERY = """
SELECT ?entity ?entityLabel ?sitelinks WHERE {
  VALUES ?class { %(classes)s }
  ?entity wdt:P31 ?class ;
          wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= %(min_sitelinks)d)
  FILTER EXISTS { ?entity wdt:%(prop)s ?anyValue . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?sitelinks)
LIMIT %(limit)d
"""

STATEMENT_QUERY = """
SELECT ?entity ?value ?unit ?rank ?pit ?countryLabel ?adminLabel WHERE {
  VALUES ?entity { %(qids)s }
  ?entity p:%(prop)s ?st .
  ?st wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  ?st ps:%(prop)s ?value .
  OPTIONAL { ?st psv:%(prop)s ?vn . ?vn wikibase:quantityUnit ?unit . }
  OPTIONAL { ?st pq:P585 ?pit . }
  OPTIONAL { ?entity wdt:P17 ?country . }
  OPTIONAL { ?entity wdt:P131 ?admin . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def _sparql(query: str, user_agent: str, retries: int = 5, backoff: float = 5.0) -> list[dict[str, Any]]:
    headers = {"User-Agent": user_agent, "Accept": "application/sparql-results+json"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, data={"query": query}, headers=headers, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            last = e
            wait = backoff * (attempt + 1)
            log.warning("SPARQL attempt %d failed (%s); retrying in %.0fs", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"SPARQL query failed after {retries} attempts: {last}")


def _v(b: dict[str, Any], key: str) -> str | None:
    return b[key]["value"] if key in b else None


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value.lstrip("+")[:10])
    except (ValueError, TypeError):
        return None


def source_parameters(acfg: dict, ccfg: dict) -> dict:
    return {"classes": acfg["classes"], "property": acfg["property"],
            "min_sitelinks": acfg["min_sitelinks"],
            "candidate_limit": 2 * int(ccfg["max_entities_per_attribute"])}


def write_source_manifest(raw_dir: Path, ccfg: dict) -> None:
    records = {}
    for attr, acfg in ccfg["attributes"].items():
        path = raw_dir / f"{attr}.entities.json"
        records[attr] = {"query_parameters": source_parameters(acfg, ccfg),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (raw_dir / "source_manifest.json").write_text(json.dumps(records, indent=2) + "\n")


def fetch_entities(attr: str, acfg: dict[str, Any], ccfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one record per candidate entity with its raw statements, popularity and place names.

    No value is resolved here: `resolve_value` does that at build time, so the
    resolution rules can change without touching Wikidata again."""
    ua = ccfg["user_agent"]
    classes = " ".join(f"wd:{c}" for c in acfg["classes"])
    limit = int(ccfg["max_entities_per_attribute"])
    rows = _sparql(
        ENTITY_QUERY % {"classes": classes, "min_sitelinks": int(acfg["min_sitelinks"]), "prop": acfg["property"], "limit": limit * 2},
        ua,
    )
    ents: dict[str, dict[str, Any]] = {}
    for b in rows:
        qid = _qid(b["entity"]["value"])
        label = _v(b, "entityLabel") or qid
        if label == qid or qid in ents:
            continue  # no English label, or duplicate row
        ents[qid] = {"qid": qid, "label": label, "sitelinks": int(float(b["sitelinks"]["value"]))}
    log.info("%s: %d candidate entities with an English label", attr, len(ents))

    qids = sorted(ents, key=lambda q: -ents[q]["sitelinks"])[: limit * 2]
    statements: dict[str, list[dict[str, Any]]] = defaultdict(list)
    countries: dict[str, set[str]] = defaultdict(set)
    admins: dict[str, set[str]] = defaultdict(set)
    batch = 150
    for i in range(0, len(qids), batch):
        chunk = qids[i : i + batch]
        rows = _sparql(STATEMENT_QUERY % {"qids": " ".join(f"wd:{q}" for q in chunk), "prop": acfg["property"]}, ua)
        for b in rows:
            qid = _qid(b["entity"]["value"])
            try:
                value = float(b["value"]["value"])
            except (KeyError, ValueError):
                continue
            unit = _qid(_v(b, "unit")) if _v(b, "unit") else None
            st = {"value": value, "unit": unit, "rank": _qid(b["rank"]["value"]), "pit": _v(b, "pit")}
            if st not in statements[qid]:
                statements[qid].append(st)
            for key, store in (("countryLabel", countries), ("adminLabel", admins)):
                lab = _v(b, key)
                if lab and not (lab[0] == "Q" and lab[1:].isdigit()):
                    store[qid].add(lab)
        time.sleep(1.0)  # be polite to the public endpoint

    records = []
    for qid in qids:
        if not statements.get(qid):
            continue
        records.append(
            {
                **ents[qid],
                "country": " / ".join(sorted(countries.get(qid, []))[:2]) or None,
                "admin": sorted(admins.get(qid, []))[0] if admins.get(qid) else None,
                "statements": statements[qid],
            }
        )
    log.info("%s: %d entities with at least one statement", attr, len(records))
    return records


def resolve_value(rec: dict[str, Any], acfg: dict[str, Any], metadata: dict | None = None) -> dict[str, Any] | None:
    """Pick one value per entity from its raw statements.

    1. convert to the attribute's unit (drop unknown units);
    2. drop values outside the attribute's plausible range (catches unit and
       data-entry errors on Wikidata);
    3. for population, require recent dated whole-entity statements whose
       qualifiers were checked against supplemental metadata;
    4. prefer PreferredRank, then the latest dated statement, else the max.

    Keep the range of competing values for rejecting ambiguous pairs. Never
    discard recent population values using a median over historical values.
    """
    target = acfg.get("unit")
    population = acfg.get("population_policy")
    if population:
        if metadata is None:
            return None
        as_of = dt.date.fromisoformat(population["as_of"])
        if any((_date(d) or dt.date.min) <= as_of for d in metadata["dissolved"]):
            return None
    usable = []
    for raw_st in rec["statements"]:
        st = {**raw_st, "rank": _qid(raw_st["rank"])}
        if st["rank"] not in ("PreferredRank", "NormalRank") or not math.isfinite(st["value"]):
            continue
        date = _date(st["pit"])
        if st["pit"] and date is None:
            continue
        if population:
            if date is None or not population["min_year"] <= date.year or date > as_of:
                continue
            matches = [s for s in metadata["statements"]
                       if s["value"] == st["value"] and s["unit"] == st["unit"]
                       and _date(s["pit"]) == date and _qid(s["rank"]) == st["rank"]
                       and set(s["qualifiers"]) <= set(population["allowed_qualifiers"])
                       and len(s["qualifiers"].get("P585", [])) == 1]
            if not matches:
                continue
            st["source_statements"] = sorted(matches, key=lambda s: s["statement_id"])
        if target is None:
            if st["unit"] in (None, "Q199"):  # dimensionless / "1"
                usable.append({**st, "value_conv": st["value"]})
        elif st["unit"] in LENGTH_TO_M:
            usable.append({**st, "value_conv": st["value"] * LENGTH_TO_M[st["unit"]] / TARGET_UNIT_IN_M[target]})
    usable = [u for u in usable if u["value_conv"] > 0]
    lo, hi = acfg.get("plausible_range") or (0, math.inf)
    usable = [u for u in usable if lo <= u["value_conv"] <= hi]
    if not usable:
        return None
    preferred = [u for u in usable if u["rank"] == "PreferredRank"]
    pool = preferred or usable
    dated = [u for u in pool if u["pit"]]
    chosen = max(dated, key=lambda u: (_date(u["pit"]), u["value_conv"])) if dated else max(pool, key=lambda u: u["value_conv"])
    alternatives = pool
    if population:
        alternatives = [u for u in usable if _date(u["pit"]).year >= _date(chosen["pit"]).year - population["stability_years"]]
    values = [u["value_conv"] for u in alternatives]
    return {
        "qid": rec["qid"],
        "label": rec["label"],
        "sitelinks": rec["sitelinks"],
        "country": rec.get("country"),
        "admin": rec.get("admin"),
        "value": chosen["value_conv"],
        "unit": target,
        "value_point_in_time": chosen["pit"],
        "value_rank": chosen["rank"],
        "value_min": min(values),
        "value_max": max(values),
        "source_statements": chosen.get("source_statements", []),
        "scope_check": "whole_entity_qualifiers" if population else "legacy_cache_no_scope_qualifiers",
        "metadata_fetched_at": metadata["fetched_at"] if population else None,
        "n_statements": len(usable),
        "value_spread_log10": math.log10(max(values) / min(values)),
    }


def load_or_fetch_entities(attr: str, acfg: dict[str, Any], ccfg: dict[str, Any], raw_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    """Cached raw records (with statements), resolved to one value each and cut to the top N by popularity."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{attr}.entities.json"
    if path.exists() and not force:
        source_path = raw_dir / "source_manifest.json"
        if not source_path.exists():
            raise ValueError("Missing source_manifest.json for frozen comparison caches")
        origin = json.loads(source_path.read_text())[attr]
        if (origin["query_parameters"] != source_parameters(acfg, ccfg)
                or origin["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("Frozen comparison cache differs from source parameters/hash; explicitly refresh sources")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw and "statements" not in raw[0]:
            raise RuntimeError(f"{path} is from an older cache format without statements; rebuild with --force")
    else:
        raw = fetch_entities(attr, acfg, ccfg)
        path.write_text(json.dumps(raw, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {}
    if acfg.get("population_policy"):
        meta_path = raw_dir / "population_metadata.json"
        if not meta_path.exists():
            raise ValueError("Run python -m lra.data.comparison_metadata to supplement population qualifiers")
        snapshot = json.loads(meta_path.read_text())
        if snapshot["input_sha256"][path.name] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Population metadata does not match frozen entity cache")
        metadata = snapshot["entities"]
        if any(rec["qid"] not in metadata for rec in raw):
            raise ValueError("Population metadata is incomplete; resume the metadata fetch")
    resolved = [r for r in (resolve_value(rec, acfg, metadata.get(rec["qid"])) for rec in raw) if r is not None]
    resolved.sort(key=lambda r: (-r["sitelinks"], r["qid"]))
    limit = int(ccfg["max_entities_per_attribute"])
    log.info("%s: %d/%d entities with a plausible value (kept top %d by sitelinks)", attr, len(resolved), len(raw), min(limit, len(resolved)))
    return resolved[:limit]


def _display(rec: dict[str, Any], display_with: str | None) -> str:
    place = rec.get(display_with) if display_with in ("country", "admin") else None
    if place and place != rec["label"]:
        return f"{rec['label']} ({place})"
    return rec["label"]


def pair_is_eligible(a: dict, b: dict, population_policy: dict | None = None) -> bool:
    if a["label"].casefold() == b["label"].casefold():
        return False
    # Every retained competing value must give the same strict ordering.
    if not (a["value_min"] > b["value_max"] or b["value_min"] > a["value_max"]):
        return False
    if population_policy:
        ay, by = _date(a["value_point_in_time"]), _date(b["value_point_in_time"])
        if ay is None or by is None or abs(ay.year - by.year) > population_policy["max_pair_year_gap"]:
            return False
    return True


def sample_pairs(records: list[dict[str, Any]], bins: list[list[float]], pairs_per_bin: int, max_uses: int, rng: random.Random, population_policy: dict | None = None) -> list[tuple[int, int, int]]:
    """Return (i, j, bin_index) triples. Bin 0 is the hardest (smallest ratio)."""
    lo, hi = bins[0][0], bins[-1][1]
    logs = [math.log10(r["value"]) for r in records]
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            if not pair_is_eligible(records[i], records[j], population_policy):
                continue
            d = abs(logs[i] - logs[j])
            if d < lo or d > hi:
                continue
            for k, (a, b) in enumerate(bins):
                if a <= d < b or (k == len(bins) - 1 and d == b):
                    candidates[k].append((i, j))
                    break
    uses: dict[int, int] = defaultdict(int)
    chosen: list[tuple[int, int, int]] = []
    for k in range(len(bins)):
        cands = candidates[k]
        rng.shuffle(cands)
        got = 0
        for i, j in cands:
            if got >= pairs_per_bin:
                break
            if uses[i] >= max_uses or uses[j] >= max_uses:
                continue
            uses[i] += 1
            uses[j] += 1
            chosen.append((i, j, k))
            got += 1
        if got < pairs_per_bin:
            raise ValueError(f"bin {k}: only {got}/{pairs_per_bin} eligible pairs under max_uses={max_uses}")
    return chosen


def build(ccfg: dict[str, Any], raw_dir: Path, seed: int, force: bool = False) -> list[Item]:
    bins = [[float(a), float(b)] for a, b in ccfg["ratio_bins"]]
    items: list[Item] = []
    for attr, acfg in ccfg["attributes"].items():
        rng = random.Random(f"{seed}:comparison:{attr}")
        records = load_or_fetch_entities(attr, acfg, ccfg, raw_dir, force=force)
        max_uses = int(acfg.get("max_uses_per_entity", ccfg["max_uses_per_entity"]))
        pairs = sample_pairs(records, bins, int(ccfg["pairs_per_bin"]), max_uses, rng, acfg.get("population_policy"))
        # balance question direction within each bin
        by_bin: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for p in pairs:
            by_bin[p[2]].append(p)
        pair_idx = 0
        for k in sorted(by_bin):
            for t, (i, j, _) in enumerate(by_bin[k]):
                direction = "larger" if t % 2 == 0 else "smaller"
                first, second = (i, j) if rng.random() < 0.5 else (j, i)
                pair_id = f"{DATA_VERSION}:{attr}:{pair_idx:04d}"
                for order, (ia, ib) in (("ab", (first, second)), ("ba", (second, first))):
                    ra, rb = records[ia], records[ib]
                    larger_is_a = ra["value"] > rb["value"]
                    label = ("A" if larger_is_a else "B") if direction == "larger" else ("B" if larger_is_a else "A")
                    correct = ra if label == "A" else rb
                    other = rb if label == "A" else ra
                    item_id = f"comparison:{pair_id}:{order}"
                    mirror = f"comparison:{pair_id}:{'ba' if order == 'ab' else 'ab'}"
                    displays = [_display(r, acfg.get('display_with')) for r in (ra, rb)]
                    if acfg.get("population_policy"):
                        displays = [f"{d} (population in {_date(r['value_point_in_time']).year})" for d, r in zip(displays, (ra, rb))]
                    question = f"{acfg[direction]}\nA. {displays[0]}\nB. {displays[1]}"
                    items.append(
                        Item(
                            item_id=item_id,
                            task="comparison",
                            question=question,
                            options=list(OPTIONS),
                            label=label,
                            difficulty={
                                "ratio_bin": k,
                                "abs_log10_ratio": abs(math.log10(ra["value"] / rb["value"])),
                                "log10_ratio_a_over_b": math.log10(ra["value"] / rb["value"]),
                            },
                            meta={
                                "data_version": DATA_VERSION,
                                "attribute": attr,
                                "property": acfg["property"],
                                "unit": acfg.get("unit"),
                                "direction": direction,
                                "pair_id": pair_id,
                                "order": order,
                                "mirror_of": mirror,
                                "entity_a": dict(ra),
                                "entity_b": dict(rb),
                                "famous_is_correct": correct["sitelinks"] > other["sitelinks"],
                                "sitelinks_log10_ratio_a_over_b": math.log10(ra["sitelinks"] / rb["sitelinks"]),
                            },
                        )
                    )
                pair_idx += 1
        log.info("comparison/%s: %d pairs -> %d items", attr, pair_idx, 2 * pair_idx)
    return items
