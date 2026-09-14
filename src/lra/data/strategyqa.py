"""StrategyQA (Geva et al., 2021): implicit multi-step yes/no questions.

We use the original release's *train* file, the only split with public labels
(2,290 questions). Every question is kept; subsets are chosen later by
`lra.data.splits`, so nothing has to be rebuilt when the sample size changes.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import requests

from lra.data.schema import Item

log = logging.getLogger(__name__)

URL = "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip"
TRAIN_FILE = "strategyqa_train.json"
OPTIONS = ["Yes", "No"]


def download(raw_dir: Path, force: bool = False) -> Path:
    """Download and extract the official zip; return the path of the train file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "strategyqa_dataset.zip"
    extracted = raw_dir / "extracted"
    train_path = extracted / TRAIN_FILE
    if train_path.exists() and not force:
        return train_path
    if not zip_path.exists() or force:
        log.info("downloading %s", URL)
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extracted)
    if not train_path.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found after extracting {zip_path}")
    return train_path


def build(raw_dir: Path) -> list[Item]:
    train_path = download(raw_dir)
    records = json.loads(train_path.read_text(encoding="utf-8"))
    items: list[Item] = []
    for rec in records:
        question = rec["question"].strip()
        label = "Yes" if bool(rec["answer"]) else "No"
        decomposition = rec.get("decomposition") or []
        facts = rec.get("facts") or []
        items.append(
            Item(
                item_id=f"strategyqa:{rec['qid']}",
                task="strategyqa",
                question=question,
                options=list(OPTIONS),
                label=label,
                difficulty={
                    "n_decomposition_steps": len(decomposition),
                    "n_facts": len(facts),
                },
                meta={
                    "qid": rec["qid"],
                    "term": rec.get("term"),
                    "description": rec.get("description"),
                    "decomposition": decomposition,
                    "facts": facts,
                },
            )
        )
    log.info("strategyqa: %d items (%d Yes / %d No)", len(items),
             sum(i.label == "Yes" for i in items), sum(i.label == "No" for i in items))
    return items
