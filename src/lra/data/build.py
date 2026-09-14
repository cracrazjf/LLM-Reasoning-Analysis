"""Build every dataset artefact under data/.

    python -m lra.data.build                 # everything
    python -m lra.data.build --task prontoqa # one task, all stages
    python -m lra.data.build --stage prompts # refresh prompts and dependent artifacts
    python -m lra.data.build --force         # rebuild from frozen sources
    python -m lra.data.build --task comparison --refresh-sources # explicit network refresh

Stages: items -> prompts -> splits -> manifest. Legacy --stage and --force
flags are accepted; requested tasks always rebuild all stages from sources.
Untargeted tasks also rebuild when their recorded dependencies are stale.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from lra import __version__
from lra.data import comparison, prompts, prontoqa, splits, strategyqa
from lra.data.schema import TASKS, Item, read_items, stable_hash, write_items
from lra.paths import DATA_CONFIG, DATA_DIR, PROMPTS_DIR, RAW_DIR, SPLITS_DIR, TASKS_DIR, ensure_dirs

log = logging.getLogger("lra.data.build")


def load_data_config(path: Path = DATA_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_items(task: str, cfg: dict[str, Any], force: bool, refresh_sources: bool = False) -> list[Item]:
    out = TASKS_DIR / f"{task}.jsonl"
    # The small, pinned archive is cheap to import. Always reimport it so old
    # generated questions cannot silently survive a source/config migration.
    if out.exists() and not force and task == "strategyqa":
        items = read_items(out)
        log.info("%s: loaded %d existing items from %s", task, len(items), out.relative_to(DATA_DIR.parent))
        return items
    if task == "strategyqa":
        if cfg[task]["url"] != strategyqa.URL:
            raise ValueError("StrategyQA importer only supports its pinned official source URL")
        items = strategyqa.build(RAW_DIR / "strategyqa")
    elif task == "prontoqa":
        items = prontoqa.build(cfg["prontoqa"], RAW_DIR / "prontoqa")
    elif task == "comparison":
        items = comparison.build(cfg["comparison"], RAW_DIR / "comparison", seed=int(cfg["seed"]), force=refresh_sources)
    else:
        raise ValueError(task)
    n = write_items(out, items)
    log.info("%s: wrote %d items to %s", task, n, out.relative_to(DATA_DIR.parent))
    return items


def refresh_comparison_sources(cfg: dict) -> None:
    """Explicit refresh: fetch a complete new snapshot before replacing caches."""
    import tempfile
    from lra.data.comparison_metadata import fetch as fetch_metadata

    raw_dir = RAW_DIR / "comparison"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RAW_DIR, prefix="comparison-refresh-") as temporary:
        staging = Path(temporary)
        for attr, acfg in cfg["attributes"].items():
            records = comparison.fetch_entities(attr, acfg, cfg)
            (staging / f"{attr}.entities.json").write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
        comparison.write_source_manifest(staging, cfg)
        fetch_metadata(staging)
        for path in staging.glob("*.json"):
            path.replace(raw_dir / path.name)


def task_dependencies(task: str, cfg: dict, pcfg: dict) -> dict:
    """Inputs actually used to build a task, independent of the requested stage."""
    source_files = {
        "strategyqa": [RAW_DIR / "strategyqa/extracted/strategyqa_train.json"],
        "prontoqa": [DATA_DIR.parent / cfg["prontoqa"]["archive"]],
        "comparison": sorted((RAW_DIR / "comparison").glob("*.json")),
    }[task]
    module_dir = Path(__file__).parent
    code_files = [module_dir / name for name in (f"{task}.py", "build.py", "schema.py", "prompts.py")]
    if task == "comparison":
        code_files.append(module_dir / "comparison_metadata.py")
    paths = [*source_files, *code_files]
    return json.loads(json.dumps({
        "task_config": cfg[task], "seed": cfg["seed"], "prompt_config": pcfg,
        "files": {str(p.relative_to(DATA_DIR.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in paths if p.exists()},
    }))


def summarize(items: list[Item]) -> dict[str, Any]:
    from collections import Counter

    s: dict[str, Any] = {"n_items": len(items), "labels": dict(Counter(i.label for i in items))}
    task = items[0].task if items else None
    if task == "prontoqa":
        s["by_hops"] = dict(sorted(Counter(i.difficulty["hops"] for i in items).items()))
    if task == "comparison":
        s["by_attribute"] = dict(sorted(Counter(i.meta["attribute"] for i in items).items()))
        s["by_ratio_bin"] = dict(sorted(Counter(i.difficulty["ratio_bin"] for i in items).items()))
        s["n_pairs"] = len({i.meta["pair_id"] for i in items})
    if task == "strategyqa":
        s["by_decomposition_steps"] = dict(sorted(Counter(i.difficulty["n_decomposition_steps"] for i in items).items()))
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("all", *TASKS), default="all")
    p.add_argument("--stage", choices=("all", "items", "prompts", "splits"), default="all", help="compatibility flag; dependencies and downstream artifacts are rebuilt together")
    p.add_argument("--force", action="store_true", help="compatibility flag; requested tasks always rebuild from frozen sources")
    p.add_argument("--refresh-sources", action="store_true", help="explicitly replace comparison values and metadata with a newly fetched snapshot")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    ensure_dirs()
    cfg = load_data_config()
    pcfg = prompts.load_prompt_config()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    if args.refresh_sources:
        if "comparison" not in tasks:
            p.error("--refresh-sources requires --task comparison or --task all")
        refresh_comparison_sources(cfg["comparison"])
    previous_path = DATA_DIR / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    old_builds = previous.get("task_builds", {})
    # Rebuild stale dependencies even on a task-specific or stage-specific call.
    # The cheap downstream steps always run together, preventing mixed versions.
    for task in TASKS:
        old = old_builds.get(task, {})
        out = TASKS_DIR / f"{task}.jsonl"
        pout = PROMPTS_DIR / f"{task}.jsonl"
        fresh = (old.get("dependencies") == task_dependencies(task, cfg, pcfg)
                 and out.exists() and pout.exists()
                 and old.get("items_sha256") == hashlib.sha256(out.read_bytes()).hexdigest()
                 and old.get("prompts_sha256") == hashlib.sha256(pout.read_bytes()).hexdigest())
        if not fresh and task not in tasks:
            tasks.append(task)
            log.info("%s: stale or missing dependencies; rebuilding", task)

    items_by_task: dict[str, list[Item]] = {}
    for task in tasks:
        items_by_task[task] = build_items(task, cfg, force=True)

    if items_by_task:
        for task, items in items_by_task.items():
            n = prompts.build_prompts(pcfg, items, PROMPTS_DIR / f"{task}.jsonl")
            log.info("%s: rendered %d prompts (%d conditions)", task, n, len(pcfg["conditions"]))

    if items_by_task:
        if set(items_by_task) != set(TASKS):
            # splits are defined over all tasks; load the others from disk
            for task in TASKS:
                if task not in items_by_task and (TASKS_DIR / f"{task}.jsonl").exists():
                    items_by_task[task] = read_items(TASKS_DIR / f"{task}.jsonl")
        available = {t: v for t, v in items_by_task.items() if v}
        result = splits.build_splits(cfg["splits"], available, SPLITS_DIR, seed=int(cfg["seed"]))
        for name, per_task in result.items():
            log.info("split %s: %s", name, {t: len(v) for t, v in per_task.items()})

    # A task-specific build must still describe all the artifacts on disk.
    for task in TASKS:
        if task not in items_by_task and (TASKS_DIR / f"{task}.jsonl").exists():
            items_by_task[task] = read_items(TASKS_DIR / f"{task}.jsonl")
    artifact_paths = sorted(
        [*TASKS_DIR.glob("*.jsonl"), *PROMPTS_DIR.glob("*.jsonl"), *SPLITS_DIR.glob("*.json")]
    )
    import scipy
    manifest = {
        "lra_version": __version__,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data_config_hash": stable_hash(cfg),
        "prompts_config_hash": stable_hash(pcfg),
        "conditions": list(pcfg["conditions"]),
        "sources": {"prontoqa": prontoqa.source_manifest(cfg["prontoqa"]),
                    "comparison": {"data_version": comparison.DATA_VERSION,
                                   "method": "custom_reconstruction_after_Lehmann_et_al",
                                   "popularity": "Wikipedia_sitelinks"}},
        "task_builds": {
            task: {"dependencies": task_dependencies(task, cfg, pcfg),
                   "items_sha256": hashlib.sha256((TASKS_DIR / f"{task}.jsonl").read_bytes()).hexdigest(),
                   "prompts_sha256": hashlib.sha256((PROMPTS_DIR / f"{task}.jsonl").read_bytes()).hexdigest()}
            for task in items_by_task
        },
        "split_build": {"config": cfg["splits"], "seed": cfg["seed"], "scipy_version": scipy.__version__,
                        "builder_sha256": hashlib.sha256(Path(splits.__file__).read_bytes()).hexdigest()},
        "artifacts": {
            str(path.relative_to(DATA_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in artifact_paths
        },
        "tasks": {t: summarize(v) for t, v in items_by_task.items() if v},
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote data/manifest.json")
    update_data_card(manifest)
    return 0


def update_data_card(manifest: dict[str, Any]) -> None:
    """Refresh the summary table between the markers in README.md."""
    card = DATA_DIR.parent / "README.md"
    if not card.exists():
        return
    lines = [f"Built {manifest['built_at']} (data config `{manifest['data_config_hash']}`, prompts config `{manifest['prompts_config_hash']}`).", ""]
    lines += ["| task | items | labels | breakdown |", "|---|---|---|---|"]
    for task, s in manifest["tasks"].items():
        labels = ", ".join(f"{k} {v}" for k, v in s["labels"].items())
        extra = {k: v for k, v in s.items() if k not in ("n_items", "labels")}
        breakdown = "; ".join(f"{k}: {v}" for k, v in extra.items())
        lines.append(f"| {task} | {s['n_items']} | {labels} | {breakdown} |")
    splits_path = SPLITS_DIR
    for name in ("main", "pilot"):
        f = splits_path / f"{name}.json"
        if f.exists():
            sizes = json.loads(f.read_text(encoding="utf-8"))["sizes"]
            lines.append(f"\nSplit `{name}`: " + ", ".join(f"{t} {n}" for t, n in sizes.items()))
    text = card.read_text(encoding="utf-8")
    start, end = "<!-- summary:start -->", "<!-- summary:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    card.write_text(text[:i] + "\n" + "\n".join(lines) + "\n" + text[j:], encoding="utf-8")
    log.info("updated README.md summary")


if __name__ == "__main__":
    sys.exit(main())
