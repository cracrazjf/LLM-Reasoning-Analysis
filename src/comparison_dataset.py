"""Build and verify the pairwise magnitude comparison dataset (v3).

    python src/comparison_dataset.py build          # frozen sources -> tasks, prompts, verification report
    python src/comparison_dataset.py verify         # offline checks of every item and prompt
    python src/comparison_dataset.py fetch --force  # re-query Wikidata (produces a different dataset)

Design follows Lehmann et al. (EACL 2026, "Knowing the Facts but Choosing the
Shortcut"; question wording from their templates): entities of one class with
one numerical attribute, top-N by popularity (number of Wikipedia sitelinks).
Pairs are binned by |log10(value_a / value_b)| so difficulty is controlled,
every pair appears in both presentation orders, and "larger"/"smaller"
questions alternate within each bin.

Values
  rivers, mountains, buildings, stadiums: Wikidata (frozen caches in
      data/raw/comparison/, query parameters and sha256 pinned in
      source_manifest.json). Per entity: convert units, drop implausible
      values, prefer PreferredRank, then the latest dated statement, else the
      maximum. A pair is eligible only if every competing value of one entity
      is above every competing value of the other.
  countries: World Bank total population (SP.POP.TOTL) for POPULATION_YEAR,
      one source and one year for every country.

Verification
  Every Wikidata-valued pair is compared with the entity's English Wikipedia
  infobox, a source independent of Wikidata. A pair is kept only if the
  infobox values give the same answer and neither option's text names more
  than one entity. Dropped pairs are not replaced, so bins hold between
  MIN_PAIRS_PER_BIN and PAIRS_PER_BIN pairs. The status of every sampled pair
  is written to data/verification/comparison.jsonl.

External sources are fetched once and frozen next to the Wikidata caches
(enwiki_infobox.json, worldbank_population.json); `build` fetches only what is
missing from them, so rebuilding is offline and deterministic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "comparison"
TASKS_PATH = ROOT / "data" / "tasks" / "comparison.jsonl"
PROMPTS_PATH = ROOT / "data" / "prompts" / "comparison.jsonl"
REPORT_PATH = ROOT / "data" / "verification" / "comparison.jsonl"
INFOBOX_PATH = RAW_DIR / "enwiki_infobox.json"
WORLD_BANK_PATH = RAW_DIR / "worldbank_population.json"

# ---------------------------------------------------------------- parameters

SEED = 20260914
DATA_VERSION = "v3"
# Wikimedia rate-limits clients without contact details in the User-Agent to 10 requests/min.
USER_AGENT = f"LLM-Reasoning-Analysis/0.1 (https://github.com/cracrazjf/LLM-Reasoning-Analysis) python-requests/{requests.__version__}"
MAX_ENTITIES = 1000                    # per attribute, top by sitelinks
RATIO_BINS = [[0.02, 0.05], [0.05, 0.10], [0.10, 0.20], [0.20, 0.40], [0.40, 1.00]]
PAIRS_PER_BIN = 100                    # sampled per attribute per bin
MIN_PAIRS_PER_BIN = 50                 # kept after verification, per attribute per bin
MAX_USES = 3                           # per entity within an attribute
OPTIONS = ["A", "B"]
POPULATION_YEAR = 2020

ATTRIBUTES: dict[str, dict[str, Any]] = {
    "rivers": {
        "classes": ["Q4022"], "property": "P2043", "unit": "km", "min_sitelinks": 20,
        "value_source": "wikidata", "display_with": "country", "plausible_range": [5, 7500],
        "larger": "Which river is longer?",
        "smaller": "Which river is shorter?",
    },
    "countries": {
        "classes": ["Q6256", "Q3624078"], "property": "P1082", "unit": None, "min_sitelinks": 30,
        "max_uses_per_entity": 8,  # only ~200 countries exist
        "value_source": "worldbank", "display_with": None, "plausible_range": [500, 2000000000],
        # Lehmann et al.'s question with the year added, as in their population number prompts.
        "larger": f"Which country had a larger population in {POPULATION_YEAR}?",
        "smaller": f"Which country had a smaller population in {POPULATION_YEAR}?",
    },
    "mountains": {
        "classes": ["Q8502"], "property": "P2044", "unit": "m", "min_sitelinks": 20,
        "value_source": "wikidata", "display_with": "country", "plausible_range": [100, 9000],
        "larger": "Which mountain is higher?",
        "smaller": "Which mountain is lower?",
    },
    "buildings": {
        "classes": ["Q11303", "Q12518"], "property": "P2048", "unit": "m", "min_sitelinks": 10,
        "value_source": "wikidata", "display_with": "admin", "plausible_range": [30, 1000],
        "larger": "Which building is taller?",
        "smaller": "Which building is shorter?",
    },
    "stadiums": {
        "classes": ["Q483110", "Q1154710"], "property": "P1083", "unit": None, "min_sitelinks": 15,
        "value_source": "wikidata", "display_with": "admin", "plausible_range": [2000, 200000],
        "larger": "Which stadium can accommodate more spectators?",
        "smaller": "Which stadium can accommodate fewer spectators?",
    },
}

# The question text is identical across conditions; only the instruction changes.
CONDITIONS = {
    "C0_baseline": "",
    "C1_caution_think": "Think as carefully as possible before answering, even if this takes longer.",
    "C2_speed_think": "Think as briefly as possible before answering, even if this leads to more errors.",
}
ANSWER_INSTRUCTION = "Answer with only A or B. No other words."

# unit QID -> metres
LENGTH_TO_M = {"Q3710": 0.3048, "Q11573": 1.0, "Q828224": 1000.0, "Q253276": 1609.344}
TARGET_UNIT_IN_M = {"m": 1.0, "km": 1000.0}

# ------------------------------------------------------------------ helpers


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


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stable_hash(obj: Any, n: int = 12) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def _jsonl_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(r, ensure_ascii=False) + "\n" for r in rows]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_jsonl_lines(rows)), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------ Wikidata fetch

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

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


def _sparql(query: str, retries: int = 5, backoff: float = 5.0) -> list[dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(SPARQL_ENDPOINT, data={"query": query}, headers=headers, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"SPARQL attempt {attempt + 1} failed ({e}); retrying", file=sys.stderr)
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"SPARQL query failed after {retries} attempts: {last}")


def source_parameters(acfg: dict[str, Any]) -> dict[str, Any]:
    return {"classes": acfg["classes"], "property": acfg["property"],
            "min_sitelinks": acfg["min_sitelinks"], "candidate_limit": 2 * MAX_ENTITIES}


def fetch_entities(acfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per candidate entity with its raw statements, sitelinks and place names."""
    classes = " ".join(f"wd:{c}" for c in acfg["classes"])
    rows = _sparql(ENTITY_QUERY % {"classes": classes, "min_sitelinks": int(acfg["min_sitelinks"]),
                                   "prop": acfg["property"], "limit": 2 * MAX_ENTITIES})
    ents: dict[str, dict[str, Any]] = {}
    for b in rows:
        qid = _qid(b["entity"]["value"])
        label = _v(b, "entityLabel") or qid
        if label == qid or qid in ents:
            continue  # no English label, or duplicate row
        ents[qid] = {"qid": qid, "label": label, "sitelinks": int(float(b["sitelinks"]["value"]))}
    qids = sorted(ents, key=lambda q: -ents[q]["sitelinks"])[: 2 * MAX_ENTITIES]
    statements: dict[str, list[dict[str, Any]]] = defaultdict(list)
    countries: dict[str, set[str]] = defaultdict(set)
    admins: dict[str, set[str]] = defaultdict(set)
    for i in range(0, len(qids), 150):
        chunk = " ".join(f"wd:{q}" for q in qids[i: i + 150])
        for b in _sparql(STATEMENT_QUERY % {"qids": chunk, "prop": acfg["property"]}):
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
    return [{**ents[q],
             "country": " / ".join(sorted(countries.get(q, []))[:2]) or None,
             "admin": sorted(admins.get(q, []))[0] if admins.get(q) else None,
             "statements": statements[q]}
            for q in qids if statements.get(q)]


def cmd_fetch(args: argparse.Namespace) -> None:
    caches = [RAW_DIR / f"{attr}.entities.json" for attr in ATTRIBUTES]
    if any(p.exists() for p in caches) and not args.force:
        sys.exit(f"{RAW_DIR} already holds frozen caches; pass --force to replace them "
                 "(the rebuilt dataset will differ from the current one)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for attr, acfg in ATTRIBUTES.items():
        path = RAW_DIR / f"{attr}.entities.json"
        path.write_text(json.dumps(fetch_entities(acfg), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest[attr] = {"query_parameters": source_parameters(acfg), "sha256": _sha256(path)}
        print(f"{attr}: wrote {path}", flush=True)
    (RAW_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ------------------------------------------------- World Bank populations

WORLD_BANK_URL = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
ISO_QUERY = """
SELECT ?entity ?iso2 ?iso3 WHERE {
  VALUES ?entity { %(qids)s }
  OPTIONAL { ?entity wdt:P297 ?iso2 . }
  OPTIONAL { ?entity wdt:P298 ?iso3 . }
}
"""
# Wikidata puts NLD on the Kingdom of the Netherlands (with Aruba, Curaçao, Sint
# Maarten); the World Bank's NLD is the country Netherlands.
ISO_OVERRIDES = {"Q55": "NLD", "Q29999": None}


def load_or_fetch_worldbank(qids: list[str]) -> dict[str, dict[str, Any]]:
    """World Bank SP.POP.TOTL series per entity, matched by ISO 3166 code (Wikidata P298, else P297)."""
    if WORLD_BANK_PATH.exists():
        snapshot = json.loads(WORLD_BANK_PATH.read_text())
        if set(qids) <= set(snapshot["entities"]):
            return snapshot["entities"]
    codes: dict[str, list[str]] = {q: [] for q in qids}
    for b in _sparql(ISO_QUERY % {"qids": " ".join(f"wd:{q}" for q in qids)}):
        q = _qid(b["entity"]["value"])
        codes[q] += [c for c in (_v(b, "iso3"), _v(b, "iso2")) if c and c not in codes[q]]
    for q, code in ISO_OVERRIDES.items():
        if q in codes:
            codes[q] = [code] if code else []
    r = requests.get(WORLD_BANK_URL, params={"date": "2010:2026", "format": "json", "per_page": 20000},
                     headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    meta, data = r.json()
    series: dict[str, dict[str, float]] = defaultdict(dict)
    for row in data:
        if row["value"] is not None:
            for code in (row["countryiso3code"], row["country"]["id"]):
                if code:  # regional aggregates have no ISO3 code
                    series[code][row["date"]] = float(row["value"])
    entities = {}
    for q in qids:
        code = next((c for c in codes[q] if c in series), None)
        entities[q] = {"code": code, "population": series[code] if code else {}}
    _save_json(WORLD_BANK_PATH, {"source": WORLD_BANK_URL, "lastupdated": meta.get("lastupdated"),
                                 "fetched_at": _now(), "entities": entities})
    return entities


# --------------------------------------------------------- entity values


def resolve_value(rec: dict[str, Any], acfg: dict[str, Any]) -> dict[str, Any] | None:
    """One Wikidata value per entity, plus the range of competing values."""
    target = acfg.get("unit")
    usable = []
    for raw_st in rec["statements"]:
        st = {**raw_st, "rank": _qid(raw_st["rank"])}
        if st["rank"] not in ("PreferredRank", "NormalRank") or not math.isfinite(st["value"]):
            continue
        if st["pit"] and _date(st["pit"]) is None:
            continue
        if target is None:
            if st["unit"] in (None, "Q199"):  # dimensionless / "1"
                usable.append({**st, "value_conv": st["value"]})
        elif st["unit"] in LENGTH_TO_M:
            usable.append({**st, "value_conv": st["value"] * LENGTH_TO_M[st["unit"]] / TARGET_UNIT_IN_M[target]})
    lo, hi = acfg["plausible_range"]
    usable = [u for u in usable if u["value_conv"] > 0 and lo <= u["value_conv"] <= hi]
    if not usable:
        return None
    pool = [u for u in usable if u["rank"] == "PreferredRank"] or usable
    dated = [u for u in pool if u["pit"]]
    chosen = max(dated, key=lambda u: (_date(u["pit"]), u["value_conv"])) if dated else max(pool, key=lambda u: u["value_conv"])
    values = [u["value_conv"] for u in pool]
    return {
        "qid": rec["qid"],
        "label": rec["label"],
        "sitelinks": rec["sitelinks"],
        "country": rec.get("country"),
        "admin": rec.get("admin"),
        "value": chosen["value_conv"],
        "unit": target,
        "value_source": "wikidata",
        "value_point_in_time": chosen["pit"],
        "value_rank": chosen["rank"],
        "value_min": min(values),
        "value_max": max(values),
        "n_statements": len(usable),
        "value_spread_log10": math.log10(max(values) / min(values)),
    }


def worldbank_value(rec: dict[str, Any], wb: dict[str, Any] | None) -> dict[str, Any] | None:
    value = ((wb or {}).get("population") or {}).get(str(POPULATION_YEAR))
    if not value:
        return None
    return {
        "qid": rec["qid"],
        "label": rec["label"],
        "sitelinks": rec["sitelinks"],
        "country": rec.get("country"),
        "admin": rec.get("admin"),
        "value": value,
        "unit": None,
        "value_source": f"World Bank SP.POP.TOTL {POPULATION_YEAR} ({wb['code']})",
        "value_min": value,
        "value_max": value,
    }


def load_entities(attr: str, acfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Frozen Wikidata records with one value each, top MAX_ENTITIES by sitelinks."""
    path = RAW_DIR / f"{attr}.entities.json"
    origin = json.loads((RAW_DIR / "source_manifest.json").read_text())[attr]
    if origin["query_parameters"] != source_parameters(acfg) or origin["sha256"] != _sha256(path):
        raise ValueError(f"{path.name} differs from source_manifest.json (parameters or sha256)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if acfg["value_source"] == "worldbank":
        wb = load_or_fetch_worldbank([rec["qid"] for rec in raw])
        resolved = [worldbank_value(rec, wb.get(rec["qid"])) for rec in raw]
    else:
        resolved = [resolve_value(rec, acfg) for rec in raw]
    resolved = sorted((r for r in resolved if r is not None), key=lambda r: (-r["sitelinks"], r["qid"]))
    return resolved[:MAX_ENTITIES]


def display_name(rec: dict[str, Any], acfg: dict[str, Any]) -> str:
    place = rec.get(acfg["display_with"]) if acfg["display_with"] else None
    return f"{rec['label']} ({place})" if place and place != rec["label"] else rec["label"]


# ------------------------------------------------ English Wikipedia infoboxes

WIKI_API = "https://en.wikipedia.org/w/api.php"
ENWIKI_QUERY = """
SELECT ?entity ?title WHERE {
  VALUES ?entity { %(qids)s }
  ?article schema:about ?entity ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title .
}
"""
VOID_TAGS = {"br", "img", "link", "meta", "wbr", "hr", "input", "col", "source", "area"}


class InfoboxRows(HTMLParser):
    """(group, label, text) rows of the first infobox table in rendered HTML.

    `group` is the latest section header or top-level label, so bullet rows
    such as "• Total" keep their context. Footnote markers, styles and hidden
    elements are dropped.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0      # 1 = infobox table, >1 = a table nested inside it
        self.done = False
        self.skip = 0       # open tags inside a skipped element
        self.row: list[tuple[str, str, str]] | None = None
        self.cell: list[Any] | None = None
        self.group = ""
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.done:
            return
        if self.skip:
            self.skip += tag not in VOID_TAGS
            return
        a = dict(attrs)
        if tag == "table":
            if self.depth or "infobox" in (a.get("class") or "").split():
                self.depth += 1
            return
        if not self.depth:
            return
        if tag in ("sup", "style", "script") or "display:none" in (a.get("style") or "").replace(" ", ""):
            self.skip = int(tag not in VOID_TAGS)
        elif tag == "tr" and self.depth == 1:
            self.row = []
        elif tag in ("th", "td") and self.depth == 1 and self.row is not None:
            self.cell = [tag, a.get("class") or "", []]
        elif tag == "br" and self.cell:
            self.cell[2].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.done or not self.depth:
            return
        if self.skip:
            self.skip -= tag not in VOID_TAGS
            return
        if tag == "table":
            self.depth -= 1
            self.done = self.depth == 0
        elif tag in ("th", "td") and self.depth == 1 and self.cell:
            text = "".join(self.cell[2]).replace("\xa0", " ").replace(" ", " ").replace("﻿", "")
            text = "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()
            self.row.append((self.cell[0], self.cell[1], text))
            self.cell = None
        elif tag == "tr" and self.depth == 1 and self.row is not None:
            ths = [c for c in self.row if c[0] == "th"]
            tds = [c for c in self.row if c[0] == "td"]
            self.row = None
            if ths and not tds and "infobox-header" in ths[0][1]:
                self.group = ths[0][2]
            elif ths and tds:
                label = ths[0][2]
                if not label.startswith("•"):
                    self.group = label
                self.rows.append([self.group, label.lstrip("• ").strip(), tds[0][2]])

    def handle_data(self, data: str) -> None:
        if self.depth and not self.skip and self.cell:
            self.cell[2].append(data)


class Throttle:
    """Shared spacing between API requests; widens after every HTTP 429."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        time.sleep(start - now)

    def backoff(self, seconds: float) -> None:
        with self.lock:
            self.interval = min(2 * self.interval, 5.0)
            self.next_at = max(self.next_at, time.monotonic() + seconds)


def _get(session: requests.Session, throttle: Throttle, params: dict[str, Any], retries: int = 8) -> dict[str, Any]:
    for attempt in range(retries):
        throttle.wait()
        try:
            r = session.get(WIKI_API, params={**params, "format": "json", "formatversion": 2}, timeout=60)
            if r.status_code == 429:
                throttle.backoff(float(r.headers.get("Retry-After") or 10 * (attempt + 1)))
                raise requests.HTTPError("HTTP 429")
            if r.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {params.get('page')}: {e}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def enwiki_titles(qids: list[str]) -> dict[str, str | None]:
    titles: dict[str, str | None] = dict.fromkeys(qids)
    for i in range(0, len(qids), 400):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 400])
        for b in _sparql(ENWIKI_QUERY % {"qids": values}):
            titles[_qid(b["entity"]["value"])] = b["title"]["value"]
    return titles


def fetch_infobox(session: requests.Session, throttle: Throttle, title: str) -> dict[str, Any]:
    data = _get(session, throttle, {"action": "parse", "page": title, "prop": "text|revid",
                                    "section": 0, "redirects": 1, "disabletoc": 1})
    if "error" in data:
        return {"title": title, "fetched_at": _now(), "error": data["error"].get("code")}
    parser = InfoboxRows()
    parser.feed(data["parse"]["text"])
    return {"title": data["parse"]["title"], "revid": data["parse"]["revid"], "fetched_at": _now(), "rows": parser.rows}


def load_or_fetch_infoboxes(qids: list[str], workers: int = 2) -> dict[str, dict[str, Any]]:
    """Infobox rows per entity from the frozen snapshot, fetching only entities not in it."""
    snapshot = json.loads(INFOBOX_PATH.read_text()) if INFOBOX_PATH.exists() else {
        "schema_version": 1, "source": f"{WIKI_API} action=parse section=0 (rendered infobox rows)",
        "user_agent": USER_AGENT, "entities": {}}
    missing = [q for q in qids if q not in snapshot["entities"]]
    if not missing:
        return snapshot["entities"]
    print(f"fetching English Wikipedia infoboxes for {len(missing)} entities", flush=True)
    titles = enwiki_titles(missing)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    throttle = Throttle(0.35)  # ~170 requests/min, under the 200/min limit for identified clients

    def one(q: str) -> tuple[str, dict[str, Any]]:
        if titles[q] is None:
            return q, {"title": None, "error": "no_enwiki_article"}
        return q, fetch_infobox(session, throttle, titles[q])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (q, rec) in enumerate(pool.map(one, missing), 1):
            snapshot["entities"][q] = rec
            if n % 200 == 0 or n == len(missing):
                _save_json(INFOBOX_PATH, snapshot)  # checkpoint: an interrupted fetch resumes where it stopped
                print(f"  infoboxes {n}/{len(missing)}", flush=True)
    return snapshot["entities"]


# Only metric figures are read, because the imperial figure in "{{convert}}"
# output is a rounded conversion.
NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
UNITS = {"km": r"(?:km|kilomet(?:re|er)s?)", "m": r"(?:m|met(?:re|er)s?)"}
# Infobox labels read for each attribute, in order of preference.
WIKI_LABELS = {
    "rivers": ["length"],
    "mountains": ["elevation", "highest elevation", "highest point"],
    "buildings": ["architectural", "height", "roof", "tip", "antenna spire"],
    "stadiums": ["capacity"],
}


def _metric(text: str, unit: str) -> list[float]:
    out = []
    for m in re.finditer(rf"({NUM})(?:\s*(?:–|-|to)\s*({NUM}))?\s*{UNITS[unit]}(?![\w/²])", text):
        out.extend(float(g.replace(",", "")) for g in m.groups() if g)
    return out


def _counts(text: str) -> list[float]:
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)  # drop "(2019)", "[a]"
    return [float(m.replace(",", "")) for m in re.findall(NUM, text)]


def wiki_value(attr: str, box: dict[str, Any]) -> dict[str, Any] | None:
    """The first plausible value in the most preferred infobox row, with that row's label."""
    acfg = ATTRIBUTES[attr]
    lo, hi = acfg["plausible_range"]
    labels = WIKI_LABELS[attr]
    found = []
    for _, label, text in box.get("rows") or []:
        if label.lower() not in labels:
            continue
        vals = _metric(text, acfg["unit"]) if acfg["unit"] else _counts(text)
        vals = [v for v in vals if lo <= v <= hi]
        if vals:
            found.append((labels.index(label.lower()), label, vals[0]))
    if not found:
        return None
    _, row, value = min(found, key=lambda f: f[0])
    return {"value": value, "row": row, "title": box["title"], "revid": box.get("revid")}


# ------------------------------------------------------------------- build


def pair_is_eligible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["label"].casefold() == b["label"].casefold():
        return False
    # Every retained competing value must give the same strict ordering.
    return a["value_min"] > b["value_max"] or b["value_min"] > a["value_max"]


def sample_pairs(records: list[dict[str, Any]], max_uses: int, rng: random.Random) -> list[tuple[int, int, int]]:
    """(i, j, bin) triples, PAIRS_PER_BIN per bin. Bin 0 is the hardest (smallest ratio)."""
    lo, hi = RATIO_BINS[0][0], RATIO_BINS[-1][1]
    logs = [math.log10(r["value"]) for r in records]
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if not pair_is_eligible(records[i], records[j]):
                continue
            d = abs(logs[i] - logs[j])
            if d < lo or d > hi:
                continue
            for k, (a, b) in enumerate(RATIO_BINS):
                if a <= d < b or (k == len(RATIO_BINS) - 1 and d == b):
                    candidates[k].append((i, j))
                    break
    uses: dict[int, int] = defaultdict(int)
    chosen: list[tuple[int, int, int]] = []
    for k in range(len(RATIO_BINS)):
        cands = candidates[k]
        rng.shuffle(cands)
        got = 0
        for i, j in cands:
            if got >= PAIRS_PER_BIN:
                break
            if uses[i] >= max_uses or uses[j] >= max_uses:
                continue
            uses[i] += 1
            uses[j] += 1
            chosen.append((i, j, k))
            got += 1
        if got < PAIRS_PER_BIN:
            raise ValueError(f"bin {k}: only {got}/{PAIRS_PER_BIN} eligible pairs under max_uses={max_uses}")
    return chosen


def expected_label(a: float, b: float, direction: str) -> str:
    return "A" if (a > b) == (direction == "larger") else "B"


def pair_status(ra: dict[str, Any], rb: dict[str, Any], wa: dict[str, Any] | None, wb: dict[str, Any] | None,
                acfg: dict[str, Any], shared_names: set[str]) -> str:
    """confirmed / contradicted / unchecked / ambiguous_name / reference_source.

    ambiguous_name: an option's text also names another entity of the class.
    reference_source: the value itself comes from a single reference source.
    """
    if any(display_name(r, acfg) in shared_names for r in (ra, rb)):
        return "ambiguous_name"
    if acfg["value_source"] != "wikidata":
        return "reference_source"
    if wa is None or wb is None:
        return "unchecked"
    same = (wa["value"] > wb["value"]) == (ra["value"] > rb["value"]) and wa["value"] != wb["value"]
    return "confirmed" if same else "contradicted"


KEEP = ("confirmed", "reference_source")


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(items, report): items for the kept pairs, one report row per sampled pair."""
    items: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for attr, acfg in ATTRIBUTES.items():
        rng = random.Random(f"{SEED}:comparison:{attr}")
        records = load_entities(attr, acfg)
        names = Counter(display_name(r, acfg) for r in records)
        shared_names = {n for n, c in names.items() if c > 1}
        pairs = sample_pairs(records, int(acfg.get("max_uses_per_entity", MAX_USES)), rng)
        boxes: dict[str, Any] = {}
        if acfg["value_source"] == "wikidata":
            boxes = load_or_fetch_infoboxes(sorted({records[x]["qid"] for p in pairs for x in p[:2]}))
        by_bin: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for p in pairs:
            by_bin[p[2]].append(p)
        pair_idx = 0
        for k in sorted(by_bin):
            for t, (i, j, _) in enumerate(by_bin[k]):
                direction = "larger" if t % 2 == 0 else "smaller"
                first, second = (i, j) if rng.random() < 0.5 else (j, i)
                pair_id = f"{DATA_VERSION}:{attr}:{pair_idx:04d}"
                pair_idx += 1
                ra, rb = records[first], records[second]
                ca, cb = (wiki_value(attr, boxes.get(r["qid"], {})) if boxes else None for r in (ra, rb))
                status = pair_status(ra, rb, ca, cb, acfg, shared_names)
                report.append({
                    "pair_id": pair_id, "attribute": attr, "ratio_bin": k, "direction": direction,
                    "status": status, "kept": status in KEEP,
                    "entities": [{"qid": r["qid"], "name": display_name(r, acfg), "value": r["value"],
                                  "value_source": r["value_source"], "crosscheck": w}
                                 for r, w in ((ra, ca), (rb, cb))],
                })
                if status not in KEEP:
                    continue
                for order, (xa, xb, wa, wb) in (("ab", (ra, rb, ca, cb)), ("ba", (rb, ra, cb, ca))):
                    label = expected_label(xa["value"], xb["value"], direction)
                    correct, other = (xa, xb) if label == "A" else (xb, xa)
                    items.append({
                        "item_id": f"comparison:{pair_id}:{order}",
                        "task": "comparison",
                        "question": f"{acfg[direction]}\nA. {display_name(xa, acfg)}\nB. {display_name(xb, acfg)}",
                        "options": list(OPTIONS),
                        "label": label,
                        "difficulty": {
                            "ratio_bin": k,
                            "abs_log10_ratio": abs(math.log10(xa["value"] / xb["value"])),
                            "log10_ratio_a_over_b": math.log10(xa["value"] / xb["value"]),
                        },
                        "meta": {
                            "data_version": DATA_VERSION,
                            "attribute": attr,
                            "property": acfg["property"],
                            "unit": acfg["unit"],
                            "direction": direction,
                            "pair_id": pair_id,
                            "order": order,
                            "mirror_of": f"comparison:{pair_id}:{'ba' if order == 'ab' else 'ab'}",
                            "entity_a": dict(xa),
                            "entity_b": dict(xb),
                            # English Wikipedia infobox values that confirm the answer (None: reference source)
                            "crosscheck": {"entity_a": wa, "entity_b": wb} if wa else None,
                            "famous_is_correct": correct["sitelinks"] > other["sitelinks"],
                            "sitelinks_log10_ratio_a_over_b": math.log10(xa["sitelinks"] / xb["sitelinks"]),
                        },
                    })
    return items, report


def user_message(question: str, condition: str) -> str:
    parts = [question, CONDITIONS[condition], ANSWER_INSTRUCTION]
    return "\n\n".join(p for p in parts if p)


def build_prompts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for condition in CONDITIONS:
            messages = [{"role": "user", "content": user_message(item["question"], condition)}]
            rows.append({
                "prompt_id": f"{item['item_id']}|{condition}",
                "item_id": item["item_id"],
                "task": item["task"],
                "condition": condition,
                "messages": messages,
                "options": item["options"],
                "label": item["label"],
                "prompt_hash": stable_hash(messages),
            })
    return rows


def print_counts(items: list[dict[str, Any]], report: list[dict[str, Any]]) -> None:
    print("\nSampled pairs by status (confirmed / reference_source / contradicted / unchecked / ambiguous_name):")
    statuses = ["confirmed", "reference_source", "contradicted", "unchecked", "ambiguous_name"]
    for attr in ATTRIBUTES:
        c = Counter(r["status"] for r in report if r["attribute"] == attr)
        print(f"  {attr:<10} " + " / ".join(str(c[s]) for s in statuses))
    print("\nKept pairs per bin (larger + smaller):")
    print(f"  {'':<10} " + " ".join(f"{'bin ' + str(k):>12}" for k in range(len(RATIO_BINS))) + f" {'all':>7}")
    for attr in ATTRIBUTES:
        cells, total = [], 0
        for k in range(len(RATIO_BINS)):
            d = Counter(it["meta"]["direction"] for it in items
                        if it["meta"]["attribute"] == attr and it["difficulty"]["ratio_bin"] == k and it["meta"]["order"] == "ab")
            cells.append(f"{d['larger'] + d['smaller']} ({d['larger']}+{d['smaller']})".rjust(12))
            total += d["larger"] + d["smaller"]
        print(f"  {attr:<10} " + " ".join(cells) + f" {total:>7}")


def cmd_build(args: argparse.Namespace) -> None:
    items, report = build()
    prompts = build_prompts(items)
    write_jsonl(TASKS_PATH, items)
    write_jsonl(PROMPTS_PATH, prompts)
    write_jsonl(REPORT_PATH, report)
    for path, rows in ((TASKS_PATH, items), (PROMPTS_PATH, prompts), (REPORT_PATH, report)):
        print(f"wrote {len(rows)} rows -> {path.relative_to(ROOT)}")
    print_counts(items, report)


# ------------------------------------------------------------------ verify


class Checks:
    """Named checks; each failure keeps a few examples."""

    def __init__(self) -> None:
        self.counts: dict[str, list[int]] = {}
        self.examples: dict[str, list[str]] = defaultdict(list)

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        n = self.counts.setdefault(name, [0, 0])
        n[0] += 1
        if not ok:
            n[1] += 1
            if len(self.examples[name]) < 3:
                self.examples[name].append(detail)

    def report(self) -> bool:
        width = max(len(k) for k in self.counts)
        for name, (total, failed) in self.counts.items():
            print(f"  {'FAIL' if failed else 'ok  '}  {name:<{width}}  {total - failed}/{total}")
            for ex in self.examples[name]:
                print(f"          e.g. {ex}")
        return not any(f for _, f in self.counts.values())


def check_items(items: list[dict[str, Any]], check: Checks) -> None:
    by_id = {it["item_id"]: it for it in items}
    check("unique item_id", len(by_id) == len(items), f"{len(items) - len(by_id)} duplicates")
    names: dict[tuple[str, str], set[str]] = defaultdict(set)
    uses: Counter = Counter()
    cells: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for it in items:
        iid, m, d = it["item_id"], it["meta"], it["difficulty"]
        attr, direction = m["attribute"], m["direction"]
        acfg = ATTRIBUTES[attr]
        a, b = m["entity_a"], m["entity_b"]
        check("options are [A, B]", it["options"] == OPTIONS, iid)
        check("label follows from the values", a["value"] != b["value"]
              and it["label"] == expected_label(a["value"], b["value"], direction), iid)
        check("every competing value gives the same answer",
              a["value_min"] <= a["value"] <= a["value_max"] and b["value_min"] <= b["value"] <= b["value_max"]
              and (a["value_min"] > b["value_max"] or b["value_min"] > a["value_max"]), iid)
        if acfg["value_source"] == "wikidata":
            cc = m["crosscheck"]
            check("English Wikipedia gives the same answer", cc is not None
                  and it["label"] == expected_label(cc["entity_a"]["value"], cc["entity_b"]["value"], direction), iid)
        else:
            check(f"value from World Bank {POPULATION_YEAR}", all(
                e["value_source"].startswith(f"World Bank SP.POP.TOTL {POPULATION_YEAR}") for e in (a, b)), iid)
        lines = it["question"].split("\n")
        check("question text matches the entities",
              lines == [acfg[direction], f"A. {display_name(a, acfg)}", f"B. {display_name(b, acfg)}"], iid)
        ratio = math.log10(a["value"] / b["value"])
        lo, hi = RATIO_BINS[d["ratio_bin"]]
        check("difficulty matches the values", abs(d["log10_ratio_a_over_b"] - ratio) < 1e-12
              and lo <= abs(ratio) <= hi, iid)
        mirror = by_id.get(m["mirror_of"])
        check("mirror item swaps A/B and flips the label", mirror is not None
              and mirror["meta"]["entity_a"] == b and mirror["meta"]["entity_b"] == a
              and mirror["meta"]["direction"] == direction and mirror["label"] != it["label"], iid)
        for e in (a, b):
            names[(attr, display_name(e, acfg))].add(e["qid"])
        if m["order"] == "ab":
            uses.update([(attr, a["qid"]), (attr, b["qid"])])
            cells[(attr, d["ratio_bin"])]["pairs"] += 1
        cells[(attr, d["ratio_bin"])][it["label"]] += 1
    for (attr, name), qids in names.items():
        check("each option text names a single entity", len(qids) == 1, f"{attr}: {name!r} -> {sorted(qids)}")
    for (attr, qid), n in uses.items():
        limit = ATTRIBUTES[attr].get("max_uses_per_entity", MAX_USES)
        check("entity reuse within limit", n <= limit, f"{attr} {qid} used {n}x")
    for attr in ATTRIBUTES:
        for k in range(len(RATIO_BINS)):
            c = cells[(attr, k)]
            check(f"at least {MIN_PAIRS_PER_BIN} pairs per attribute x bin", c["pairs"] >= MIN_PAIRS_PER_BIN,
                  f"{attr} bin {k}: {c['pairs']}")
            check("A/B labels balanced per attribute x bin", c["A"] == c["B"], f"{attr} bin {k}: {dict(c)}")


def check_prompts(items: list[dict[str, Any]], prompts: list[dict[str, Any]], check: Checks) -> None:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in prompts:
        by_item[p["item_id"]].append(p)
    check("every prompt belongs to an item", set(by_item) == {it["item_id"] for it in items})
    for it in items:
        rows = by_item.get(it["item_id"], [])
        check("one prompt per condition", [p["condition"] for p in rows] == list(CONDITIONS), it["item_id"])
        for p in rows:
            msgs = p["messages"]
            ok = (p["prompt_id"] == f"{it['item_id']}|{p['condition']}" and p["label"] == it["label"]
                  and p["options"] == it["options"] and p["prompt_hash"] == stable_hash(msgs)
                  and msgs == [{"role": "user", "content": user_message(it["question"], p["condition"])}])
            check("prompt carries the item's question, label and condition text", ok, p["prompt_id"])


def cmd_verify(args: argparse.Namespace) -> None:
    on_disk = {path: path.read_text(encoding="utf-8").splitlines(keepends=True)
               for path in (TASKS_PATH, PROMPTS_PATH, REPORT_PATH)}
    items = [json.loads(line) for line in on_disk[TASKS_PATH]]
    prompts = [json.loads(line) for line in on_disk[PROMPTS_PATH]]
    check = Checks()
    built_items, built_report = build()
    for path, rows in ((TASKS_PATH, built_items), (PROMPTS_PATH, build_prompts(built_items)), (REPORT_PATH, built_report)):
        lines = _jsonl_lines(rows)
        diff = [i for i, (x, y) in enumerate(zip(lines, on_disk[path])) if x != y]
        check(f"{path.name} in {path.parent.name}/ equals a rebuild from the frozen sources",
              len(lines) == len(on_disk[path]) and not diff,
              f"{len(lines)} vs {len(on_disk[path])} lines, first differing line {diff[:1]}")
    check_items(items, check)
    check_prompts(items, prompts, check)
    print(f"Checks on {len(items)} items and {len(prompts)} prompts:")
    ok = check.report()
    print_counts(items, built_report)
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="frozen sources -> data/tasks, data/prompts, data/verification").set_defaults(fn=cmd_build)
    sub.add_parser("verify", help="check every item and prompt against a rebuild and the sources").set_defaults(fn=cmd_verify)
    f = sub.add_parser("fetch", help="re-query Wikidata into data/raw/comparison (changes the dataset)")
    f.add_argument("--force", action="store_true", help="replace the frozen caches")
    f.set_defaults(fn=cmd_fetch)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
