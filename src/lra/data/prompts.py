"""Render chat messages for every item x condition.

The rendered file is what the generation stage consumes, so a prompt is fixed
once and hashed; nothing about wording is decided at generation time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from lra.data.schema import Item, stable_hash, write_jsonl
from lra.paths import PROMPTS_CONFIG


def load_prompt_config(path: Path = PROMPTS_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_system(cfg: dict[str, Any], task: str, condition: str) -> str:
    instruction = (cfg["conditions"][condition] or "").strip()
    text = cfg["system_template"].format(
        answer_spec=cfg["answer_specs"][task],
        condition_instruction=instruction,
    )
    return text.strip() + "\n" if instruction else text.rstrip() + "\n"


def render_messages(cfg: dict[str, Any], item: Item, condition: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": render_system(cfg, item.task, condition)},
        {"role": "user", "content": cfg["user_template"].format(question=item.question)},
    ]


def render_rows(cfg: dict[str, Any], items: Iterable[Item]) -> Iterable[dict[str, Any]]:
    for item in items:
        for condition in cfg["conditions"]:
            messages = render_messages(cfg, item, condition)
            yield {
                "prompt_id": f"{item.item_id}|{condition}",
                "item_id": item.item_id,
                "task": item.task,
                "condition": condition,
                "messages": messages,
                "options": item.options,
                "label": item.label,
                "prompt_hash": stable_hash(messages),
            }


def build_prompts(cfg: dict[str, Any], items: list[Item], out_path: Path) -> int:
    return write_jsonl(out_path, render_rows(cfg, items))
