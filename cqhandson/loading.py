"""Loading benchmark files into plain Python objects and pandas frames."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import paths

#: Directory stem -> how the author is named in tables and figures.
AUTHOR_LABELS = {
    "human":   "Human",
    "chatgpt": "ChatGPT",
    "dsc":     "DeepSeek-Coder",
    "qwen":    "Qwen2.5-Coder",
    "claude":  "Claude Opus 4.8",
}
#: Display order: human reference first, then models oldest-to-newest.
AUTHOR_ORDER = ["human", "chatgpt", "dsc", "qwen", "claude"]
AUTHORS = AUTHOR_ORDER


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tasks(path: str | Path | None = None) -> dict[str, dict]:
    """task_id -> task record (prompt, signature, stratum, difficulty)."""
    return {r["task_id"]: r for r in read_jsonl(path or paths.TASKS)}


def load_references(path: str | Path | None = None) -> dict[str, dict]:
    """task_id -> human structural + complexity reference."""
    return {r["task_id"]: r for r in read_jsonl(path or paths.REFERENCES)}


def load_predictions(author: str, directory: str | Path | None = None) -> dict[str, str]:
    """task_id -> the code this author produced for it."""
    directory = Path(directory or paths.PREDICTIONS)
    return {r["task_id"]: r["code"] for r in read_jsonl(directory / f"{author}.jsonl")}


def load_results(author: str, directory: str | Path | None = None) -> list[dict]:
    directory = Path(directory) if directory is not None else paths.results_dir()
    return read_jsonl(Path(directory) / f"{author}.jsonl")


def results_frame(
    authors: Iterable[str] = AUTHOR_ORDER,
    directory: str | Path | None = None,
) -> pd.DataFrame:
    """One tidy frame: every author's per-task evaluation, stacked.

    Adds `author` (stem), `author_label` (display name) and the three derived
    booleans the paper reports: `defective`, `vulnerable`, `high_severity`,
    plus the headline `clean_strict`.
    """
    frames = []
    for author in authors:
        frame = pd.json_normalize(load_results(author, directory))
        frame.insert(0, "author", author)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["author"] = pd.Categorical(frame["author"], categories=list(authors), ordered=True)
    frame["author_label"] = frame["author"].map(AUTHOR_LABELS).astype(str)
    frame["defective"] = frame["defects_total"] > 0
    frame["vulnerable"] = frame["vulns_total"] > 0
    frame["high_severity"] = frame["vulns_high_sev"] > 0
    frame["clean_strict"] = (
        frame["strict_nontrivial"] & frame["defects_total"].eq(0) & frame["vulns_total"].eq(0)
    )
    return frame


def show_code(author: str, task_id: str, predictions_dir=None) -> None:
    """Pretty-print one author's answer to one task (syntax highlighted in Jupyter)."""
    code = load_predictions(author, predictions_dir)[task_id]
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"**{AUTHOR_LABELS.get(author, author)}** — `{task_id}`\n\n"
                         f"```python\n{code}\n```"))
    except ImportError:
        print(f"# {author} / {task_id}\n{code}")
