"""Checking our run against the study's own published results.

Two questions live here, both of which a notebook should ask in one line:

* do we reproduce the study's per-task results? (`agreement_table`)
* where we don't, why not? (`dedup_experiment`)
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


def dedup_experiment(authors=tuple(FROZEN_NAMES)) -> pd.DataFrame:
    """Re-scan each corpus once, then count under two de-duplication keys.

    The released key is `(CWE, severity, matched text)`. When Semgrep redacts
    the matched text — which it does for registry rules unless the CLI is
    logged in — that third part is a constant and stops discriminating. The
    alternative keys on the finding's source position, which is always present.
    """
    import sys
    if str(paths.VENDOR) not in sys.path:
        sys.path.insert(0, str(paths.VENDOR))
    from support.rq4_build_table import normalized_cwes
    from .runner import _env, _write_corpus

    rows = []
    for author in authors:
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            _write_corpus(load_predictions(author), directory)
            completed = subprocess.run(
                ["semgrep", "scan", "--config", str(paths.SEMGREP_RULES), "--json",
                 "--metrics", "off", "--no-git-ignore", "--disable-version-check",
                 "--max-target-bytes", "1000000", str(directory)],
                capture_output=True, text=True, env=_env())
            report = json.loads(completed.stdout)

        released, by_position = collections.defaultdict(set), collections.defaultdict(set)
        for finding in report.get("results", []):
            extra = finding["extra"]
            cwes = extra.get("metadata", {}).get("cwe")
            if not cwes:
                continue
            name = os.path.basename(finding["path"])
            classes, severity = normalized_cwes(cwes), str(extra["severity"]).upper()
            released[name].add((classes, severity, extra["lines"].strip()))
            by_position[name].add((classes, severity,
                                   finding["start"]["line"], finding["start"]["col"]))

        errored = {os.path.basename(str(e.get("path", "")))
                   for e in report.get("errors", [])
                   if not str(e.get("path", "")).startswith("https:/semgrep.dev/...")}
        rows.append({
            "author": AUTHOR_LABELS[author],
            "as released": sum(len(v) for k, v in released.items() if k not in errored),
            "keyed on source position":
                sum(len(v) for k, v in by_position.items() if k not in errored),
            "the study's number": sum(
                r["vulns_total"] for r in read_jsonl(
                    paths.FROZEN / f"{FROZEN_NAMES[author]}.jsonl")),
        })
    return pd.DataFrame(rows).set_index("author")
