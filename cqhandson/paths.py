"""Every path the hands-on needs, resolved from this file's location."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "cqbench-v1"

DATA = REPO / "data"
TASKS = DATA / "tasks.jsonl"
REFERENCES = DATA / "references.jsonl"
PREDICTIONS = DATA / "predictions"

RESULTS = REPO / "results"
LIVE = RESULTS / "live"              # what you produce during the session
PRECOMPUTED = RESULTS / "precomputed"  # shipped fallback, identical pipeline
FIGURES = REPO / "figures"

SEMGREP_RULES = VENDOR / "cqbench" / "rules" / "semgrep.json"
PYLINT_ODC_MAP = VENDOR / "mappings" / "python" / "pylint_odc.xlsx"


def results_dir(prefer_live: bool = True):
    """Live results if the session produced them, otherwise the shipped fallback."""
    if prefer_live and any(LIVE.glob("*.jsonl")):
        return LIVE
    return PRECOMPUTED
