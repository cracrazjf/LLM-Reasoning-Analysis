"""Import the authors' original PrOntoQA v1 questions, without generation.

The published model_outputs_v1.zip contains eight demonstrations and one test
question per trial. Only the test question and the author's Expected answer
are imported; model predictions and demonstrations never become task items.
"""

from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

from lra.data.schema import Item, write_jsonl
from lra.paths import REPO_ROOT

log = logging.getLogger(__name__)
OPTIONS = ["True", "False"]
SOURCE_RELEASE = "original_v1"
SOURCE_REPOSITORY = "https://github.com/asaparov/prontoqa"
SOURCE_ARCHIVE_URL = (
    "https://github.com/asaparov/prontoqa/blob/"
    "0a6412b6fddf46324a1cb96e066dd7b3d89b87d6/model_outputs_v1.zip"
)
EXPECTED_ANSWER = re.compile(r"^Expected answer: ([^\r\n]+)\r?$", re.MULTILINE)
QUESTION = re.compile(r"^Q: ([^\r\n]+)\r?$", re.MULTILINE)


def parse_log(
    text: str,
    hops: int,
    source_member: str,
    source_archive_sha256: str,
    source_member_sha256: str,
) -> list[Item]:
    """Extract each trial's last Q and gold answer, preserving question text."""
    items: list[Item] = []
    previous_end = 0
    for expected in EXPECTED_ANSWER.finditer(text):
        block = text[previous_end : expected.start()]
        questions = list(QUESTION.finditer(block))
        if not questions:
            raise ValueError(f"{source_member}: missing test question before gold answer")
        last_question = questions[-1]
        # The test prompt ends in a blank A:, unlike the answered demonstrations.
        if not re.match(r"\r?\nA:[ \t]*\r?\n", block[last_question.end() :]):
            raise ValueError(f"{source_member}: final question has no empty A: prompt")
        question = last_question.group(1)
        context, separator, query = question.partition(" True or false: ")
        if not separator or not query.endswith("."):
            raise ValueError(f"{source_member}: malformed binary test question")
        answer = expected.group(1)
        gold = re.fullmatch(r"(.+\.) (True|False)", answer)
        if gold is None:
            raise ValueError(f"{source_member}: malformed Expected answer")
        summary = re.match(r"\r?\nn: (\d+),", text[expected.end() :])
        if summary is None or int(summary.group(1)) != len(items) + 1:
            raise ValueError(f"{source_member}: missing or nonsequential trial index")
        trial_index = int(summary.group(1))
        proof = [sentence + "." for sentence in gold.group(1)[:-1].split(". ")]
        item = Item(
            item_id=f"prontoqa:v1:h{hops}:t{trial_index:04d}",
            task="prontoqa",
            question=question,
            options=list(OPTIONS),
            label=gold.group(2),
            difficulty={
                "hops": hops,
                "n_context_sentences": context.count(". ") + 1,
                "n_proof_steps": len(proof),
            },
            meta={
                "source_release": SOURCE_RELEASE,
                "source_repository": SOURCE_REPOSITORY,
                "source_archive_url": SOURCE_ARCHIVE_URL,
                "source_archive_sha256": source_archive_sha256,
                "source_member": source_member,
                "source_member_sha256": source_member_sha256,
                "source_trial_index": trial_index,
                "source_question": question,
                "expected_answer": answer,
                "n_source_demonstrations": len(questions) - 1,
                "chain_of_thought": proof,
                "ontology": "fictional",
                "formula_ordering": "postorder",
            },
        )
        item.validate()
        items.append(item)
        previous_end = expected.end()
    if not items:
        raise ValueError(f"{source_member}: no author-labelled trials")
    return items


def source_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_release": SOURCE_RELEASE,
        "archive": cfg["archive"],
        "archive_sha256": cfg["archive_sha256"],
        "members": cfg["members"],
        "hops": cfg["hops"],
        "source_archive_url": SOURCE_ARCHIVE_URL,
    }


def build(cfg: dict[str, Any], raw_dir: Path) -> list[Item]:
    """Import the pinned source archive and keep every original test trial."""
    if cfg.get("source") != "official_v1_archive":
        raise ValueError("PrOntoQA requires source=official_v1_archive; generation is disabled")
    if cfg["hops"] != [1, 3, 5]:
        raise ValueError("The selected original release contains hops [1, 3, 5]")
    archive = REPO_ROOT / cfg["archive"]
    if not archive.is_file():
        raise FileNotFoundError(f"Missing official archive {archive}; restore the pinned official archive under data/raw/prontoqa")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha256 != cfg["archive_sha256"]:
        raise ValueError("PrOntoQA archive checksum mismatch; refusing an unpinned source")
    items: list[Item] = []
    with zipfile.ZipFile(archive) as source:
        for hops in cfg["hops"]:
            member = cfg["members"][hops]
            required_name = f"gpt_textdavinci002_{hops}hop.log"
            if member["name"] != required_name:
                raise ValueError(f"Expected original postorder member {required_name}")
            payload = source.read(member["name"])
            member_sha256 = hashlib.sha256(payload).hexdigest()
            if member_sha256 != member["sha256"]:
                raise ValueError(f"{member['name']}: source member checksum mismatch")
            imported = parse_log(
                payload.decode("utf-8"), hops, member["name"], archive_sha256, member_sha256
            )
            if len(imported) != cfg["expected_trials_per_hop"]:
                raise ValueError(f"{member['name']}: unexpected number of original trials")
            if any(item.meta["n_source_demonstrations"] != 8 for item in imported):
                raise ValueError(f"{member['name']}: unexpected original demonstration count")
            items.extend(imported)
            log.info("prontoqa: imported %d original %d-hop trials", len(imported), hops)
    if len({item.question for item in items}) != len(items):
        raise ValueError("Duplicate questions in selected original PrOntoQA trials")
    # This is an extract from the original archive, not generated data.
    write_jsonl(
        raw_dir / "original_v1.jsonl",
        ({"item_id": item.item_id, "hops": item.difficulty["hops"], **item.meta} for item in items),
    )
    return items
