"""Supplement frozen population values with statement IDs and qualifiers.

This reads Wikidata but never overwrites the original entity/value caches.
Only statements matching a frozen value and date are used by the builder.
Run explicitly: python -m lra.data.comparison_metadata
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

from lra.data.comparison import _sparql, _qid, _v
from lra.paths import RAW_DIR


QUERY = """
SELECT ?entity ?dissolved ?statement ?value ?unit ?rank ?pit ?qualifier ?qvalue ?reference WHERE {
  VALUES ?entity { %(entities)s }
  OPTIONAL { ?entity wdt:P576 ?dissolved . }
  OPTIONAL {
    ?entity p:P1082 ?statement .
    ?statement ps:P1082 ?value ; wikibase:rank ?rank ; pq:P585 ?pit .
    FILTER(?pit >= "2015-01-01T00:00:00Z"^^xsd:dateTime)
    OPTIONAL { ?statement psv:P1082 ?vn . ?vn wikibase:quantityUnit ?unit . }
    OPTIONAL {
      ?statement ?qualifier ?qvalue .
      FILTER(REGEX(STR(?qualifier), "^http://www.wikidata.org/prop/qualifier/P[0-9]+$"))
    }
    OPTIONAL { ?statement prov:wasDerivedFrom ?reference . }
  }
}
"""


def fetch(raw_dir: Path = RAW_DIR / "comparison") -> None:
    path = raw_dir / "population_metadata.json"
    raw_paths = [raw_dir / f"{a}.entities.json" for a in ("cities", "countries")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths}
    snapshot = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "endpoint": "https://query.wikidata.org/sparql",
        "query": QUERY, "input_sha256": hashes, "entities": {},
    }
    if snapshot["input_sha256"] != hashes or snapshot["query"] != QUERY:
        raise ValueError("Metadata source changed; archive the old metadata before refreshing")
    qids = sorted({r["qid"] for p in raw_paths for r in json.loads(p.read_text())})
    missing = [q for q in qids if q not in snapshot["entities"]]
    for offset in range(0, len(missing), 50):
        batch = missing[offset:offset + 50]
        rows = _sparql(QUERY % {"entities": " ".join(f"wd:{q}" for q in batch)},
                       "LLM-Reasoning-Analysis/0.1 research", retries=3, backoff=2)
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        records = {q: {"fetched_at": now, "dissolved": [], "statements": {}} for q in batch}
        for row in rows:
            record = records[_qid(row["entity"]["value"])]
            if date := _v(row, "dissolved"):
                if date not in record["dissolved"]:
                    record["dissolved"].append(date)
            if not (sid := _v(row, "statement")):
                continue
            st = record["statements"].setdefault(sid, {
                "statement_id": sid, "value": float(row["value"]["value"]),
                "unit": _qid(_v(row, "unit")) if _v(row, "unit") else None,
                "rank": _qid(row["rank"]["value"]), "pit": _v(row, "pit"),
                "qualifiers": {}, "references": [],
            })
            if prop := _v(row, "qualifier"):
                values = st["qualifiers"].setdefault(_qid(prop), [])
                if row["qvalue"]["value"] not in values:
                    values.append(row["qvalue"]["value"])
            if ref := _v(row, "reference"):
                if ref not in st["references"]:
                    st["references"].append(ref)
        for record in records.values():
            record["statements"] = sorted(record["statements"].values(), key=lambda s: s["statement_id"])
        snapshot["entities"].update(records)
        # Atomic checkpoint permits resume after a transient endpoint failure.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n")
        temporary.replace(path)
        print(f"population metadata: {len(snapshot['entities'])}/{len(qids)} entities", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    fetch()
