"""Helpers for the CQBench hands-on.

Importing this package puts the vendored CQBench v1 evaluator on ``sys.path``,
so ``from cqbench.structural import analyze_structure`` works from any notebook
without installing anything extra.
"""
from __future__ import annotations

import sys

from .paths import REPO, VENDOR

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from .loading import (  # noqa: E402
    AUTHORS,
    AUTHOR_LABELS,
    AUTHOR_ORDER,
    AUTHOR_ROLES,
    BASELINES,
    ROLE_LABELS,
    ROLE_SHORT,
    SUBMISSION,
    load_predictions,
    load_references,
    load_results,
    load_tasks,
    read_jsonl,
    results_source,
    results_frame,
    show_code,
)
from .metrics import (  # noqa: E402
    ODC_COLUMNS,
    ODC_LABELS,
    headline_table,
    paired_bootstrap_ci,
    rate,
)
from .viz import (  # noqa: E402
    AUTHOR_COLORS,
    PALETTE,
    ROLE_COLORS,
    author_color,
    style,
)
from . import figures, whatif  # noqa: E402,F401

__all__ = [
    "REPO", "VENDOR",
    "AUTHORS", "AUTHOR_LABELS", "AUTHOR_ORDER", "AUTHOR_ROLES",
    "ROLE_LABELS", "ROLE_SHORT", "SUBMISSION", "BASELINES",
    "load_tasks", "load_references", "load_predictions", "load_results",
    "results_frame", "show_code", "read_jsonl", "results_source",
    "ODC_COLUMNS", "ODC_LABELS", "headline_table", "paired_bootstrap_ci", "rate",
    "PALETTE", "AUTHOR_COLORS", "ROLE_COLORS", "author_color", "style",
    "figures", "whatif",
]
