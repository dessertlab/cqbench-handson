"""The benchmark's scoring vocabulary, in one place.

Everything here mirrors `cqbench/report.py` and `cqbench/evaluate.py`; it is
re-expressed on tidy pandas frames so it can be inspected and modified during
the session.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

#: The six Orthogonal Defect Classification buckets Pylint findings map to.
ODC_COLUMNS = (
    "def_assignment", "def_algorithm", "def_interface",
    "def_checking", "def_timing", "def_function_class_object",
)
ODC_LABELS = {
    "def_assignment": "Assignment",
    "def_algorithm": "Algorithm",
    "def_interface": "Interface",
    "def_checking": "Checking",
    "def_timing": "Timing/Serialization",
    "def_function_class_object": "Function/Class/Object",
}

SEED = 2025
BOOTSTRAP_RESAMPLES = 10_000

#: name -> callable(frame) -> boolean Series. These are the reported rates.
RATE_METRICS = {
    "parseable_rate":          lambda f: f["parseable"],
    "target_present_rate":     lambda f: f["target_present"],
    "arity_ok_rate":           lambda f: f["target_matches_arity"],
    "nonstub_rate":            lambda f: f["nonstub"],
    "strict_nontrivial_rate":  lambda f: f["strict_nontrivial"],
    "defect_free_rate":        lambda f: f["defects_total"].eq(0),
    "vulnerability_free_rate": lambda f: f["vulns_total"].eq(0),
    "high_severity_free_rate": lambda f: f["vulns_high_sev"].eq(0),
    "clean_strict_at_1":       lambda f: (
        f["strict_nontrivial"] & f["defects_total"].eq(0) & f["vulns_total"].eq(0)
    ),
}


def rate(frame: pd.DataFrame, metric: str) -> float:
    """The value of one named rate metric over a frame of per-task results."""
    return float(RATE_METRICS[metric](frame).astype(float).mean())


def headline_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Table 5 of the paper, per author: the numbers that get quoted.

    `frame` is the tidy multi-author frame from `results_frame()`.
    """
    rows = []
    for author, group in frame.groupby("author", observed=True, sort=False):
        rows.append({
            "Author":        group["author_label"].iloc[0],
            "N":             len(group),
            "Defective %":   100 * group["defective"].mean(),
            "Vulnerable %":  100 * group["vulnerable"].mean(),
            "High sev. %":   100 * group["high_severity"].mean(),
            "Clean %":       100 * group["clean_strict"].mean(),
            "Total defects": int(group["defects_total"].sum()),
            "Total vulns":   int(group["vulns_total"].sum()),
        })
    return pd.DataFrame(rows).set_index("Author")


def _seed(label: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def paired_bootstrap_ci(
    a: pd.Series | np.ndarray,
    b: pd.Series | np.ndarray,
    *,
    label: str = "",
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """95% CI for the paired difference in rate between two authors.

    `a` and `b` must be boolean/0-1 vectors **aligned task by task** (same task
    order, same length). Pairing is what makes the interval tight: both authors
    faced exactly the same tasks, so per-task differences cancel task difficulty.

    Reproduces the seeding scheme of `cqbench compare`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired vectors must have the same length"
    differences = a - b
    rng = np.random.default_rng(_seed(label))

    unique, counts = np.unique(differences, return_counts=True)
    if len(unique) <= 20:  # fast exact-multinomial path (rates are 0/±1 valued)
        draws = rng.multinomial(len(differences), counts / len(differences), size=resamples)
        boot = draws @ unique / len(differences)
    else:
        boot = np.empty(resamples)
        for start in range(0, resamples, 250):
            stop = min(start + 250, resamples)
            idx = rng.integers(0, len(differences), size=(stop - start, len(differences)))
            boot[start:stop] = differences[idx].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    return {
        "a": float(a.mean()), "b": float(b.mean()), "delta": float(differences.mean()),
        "ci_lo": lo, "ci_hi": hi, "significant": bool(lo > 0 or hi < 0), "n": len(differences),
    }


def compare_authors(frame: pd.DataFrame, metric: str, reference: str = "human") -> pd.DataFrame:
    """Every author against `reference` on one metric, task-paired, with CIs."""
    wide = (
        frame.assign(value=RATE_METRICS[metric](frame).astype(float))
             .pivot_table(index="task_id", columns="author", values="value", observed=True)
    )
    rows = []
    for author in wide.columns:
        if author == reference:
            continue
        stats = paired_bootstrap_ci(
            wide[author], wide[reference], label=f"{metric}:{author}:{reference}"
        )
        rows.append({
            "author": author, "metric": metric,
            "author_value": stats["a"], "reference_value": stats["b"],
            "delta": stats["delta"], "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
            "significant": stats["significant"], "n": stats["n"],
        })
    return pd.DataFrame(rows)


def odc_profile(frame: pd.DataFrame, normalize: str = "incidence") -> pd.DataFrame:
    """Per-author ODC profile.

    normalize="incidence" -> share of tasks with >=1 finding of that type
    normalize="share"     -> share of that author's findings falling in that type
    normalize="count"     -> raw finding counts
    """
    rows = {}
    for author, group in frame.groupby("author", observed=True, sort=False):
        label = group["author_label"].iloc[0]
        if normalize == "incidence":
            rows[label] = {ODC_LABELS[c]: (group[c] > 0).mean() for c in ODC_COLUMNS}
        elif normalize == "count":
            rows[label] = {ODC_LABELS[c]: int(group[c].sum()) for c in ODC_COLUMNS}
        elif normalize == "share":
            total = sum(group[c].sum() for c in ODC_COLUMNS) or 1
            rows[label] = {ODC_LABELS[c]: group[c].sum() / total for c in ODC_COLUMNS}
        else:
            raise ValueError(normalize)
    return pd.DataFrame(rows).T


def cwe_profile(frame: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Per-author CWE finding counts (a task can contribute several CWEs)."""
    records = []
    for _, row in frame.iterrows():
        for cwe in row["cwes"] or []:
            records.append({"author_label": row["author_label"], "cwe": cwe})
    if not records:
        return pd.DataFrame()
    table = (pd.DataFrame(records)
               .value_counts(["cwe", "author_label"]).unstack(fill_value=0))
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    return table.head(top) if top else table
