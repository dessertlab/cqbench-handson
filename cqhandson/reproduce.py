"""Checking our run against the study's own published results.

Two questions live here, both of which a notebook should ask in one line:

* do we reproduce the study's per-task results? (`agreement_table`)
* what did the de-duplication fix actually move? (`dedup_effect`)
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import subprocess
import tempfile

import pandas as pd

from . import paths
from .loading import AUTHOR_LABELS, load_predictions, load_results, read_jsonl

#: our stem -> the stem the study used ("openai" is ChatGPT for Python)
FROZEN_NAMES = {"human": "human", "chatgpt": "openai", "dsc": "dsc", "qwen": "qwen"}


def agreement_table(authors=tuple(FROZEN_NAMES)) -> pd.DataFrame:
    """Per-author agreement between our run and the study's frozen results."""
    rows = []
    for author in authors:
        mine = pd.json_normalize(load_results(author)).set_index("task_id").sort_index()
        theirs = pd.json_normalize(
            read_jsonl(paths.FROZEN / f"{FROZEN_NAMES[author]}.jsonl")
        ).set_index("task_id").sort_index()
        clean = lambda d: (d["strict_nontrivial"] & d["defects_total"].eq(0)
                           & d["vulns_total"].eq(0)).mean()
        rows.append({
            "author": AUTHOR_LABELS[author],
            "structural agree": (mine["strict_nontrivial"] == theirs["strict_nontrivial"]).mean(),
            "defect count agree": (mine["defects_total"] == theirs["defects_total"]).mean(),
            "vuln count agree": (mine["vulns_total"] == theirs["vulns_total"]).mean(),
            "clean@1 ours": 100 * clean(mine),
            "clean@1 theirs": 100 * clean(theirs),
            "total vulns ours": int(mine["vulns_total"].sum()),
            "total vulns theirs": int(theirs["vulns_total"].sum()),
        })
    return pd.DataFrame(rows).set_index("author")


def matched_text_values() -> pd.Series:
    """What `extra["lines"]` — the third part of the de-duplication key — holds."""
    from .loading import AUTHOR_ORDER
    counter = collections.Counter(
        finding["extra"]["lines"].strip()[:40]
        for author in AUTHOR_ORDER
        for row in load_results(author)
        for finding in row["vulnerability_findings"])
    return pd.Series(counter, name="findings").sort_values(ascending=False)


def dedup_effect(authors=None) -> pd.DataFrame:
    """What the de-duplication key changes, and what it leaves alone.

    Every stored finding carries both the matched source text and its position,
    so both keys can be counted from results already on disk -- no re-scan.

    The released v1 key was `(class, severity, matched source text)`. Semgrep
    redacts that third field to the literal ``"requires login"`` for
    registry-sourced rules whenever the CLI is unauthenticated, so on a fresh
    install it is a constant and stops discriminating: the key silently
    degrades to `(class, severity)` and distinct findings collapse. Keying on
    the finding's start position instead is what this repository ships.

    The fixed key refines the released one, so counting released keys over the
    surviving findings reproduces exactly what the released evaluator reported.
    """
    import sys
    if str(paths.VENDOR) not in sys.path:
        sys.path.insert(0, str(paths.VENDOR))
    from support.rq4_build_table import normalized_cwes

    from .loading import AUTHOR_ORDER
    authors = tuple(authors or AUTHOR_ORDER)

    rows = []
    for author in authors:
        results = load_results(author)
        released = fixed = 0
        for row in results:
            findings = row["vulnerability_findings"]
            released += len({(normalized_cwes(f["extra"]["metadata"]["cwe"]),
                              str(f["extra"]["severity"]).upper(),
                              str(f["extra"]["lines"]).strip())
                             for f in findings})
            fixed += len(findings)
        flagged = sum(bool(row["vulnerability_findings"]) for row in results)
        rows.append({
            "author": AUTHOR_LABELS[author],
            "vulnerabilities, released key": released,
            "vulnerabilities, fixed key": fixed,
            "tasks with \u22651 finding": flagged,
            "% of tasks flagged": 100 * flagged / len(results),
        })
    return pd.DataFrame(rows).set_index("author")


def matched_text_is_constant() -> bool:
    """True when every stored finding carries the same redacted source text."""
    return len(matched_text_values()) == 1
