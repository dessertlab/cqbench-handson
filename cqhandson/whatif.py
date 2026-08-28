"""Change one measurement decision and re-score, without re-running anything.

Every benchmark number is the output of choices someone made: where to put the
complexity floor, which linter messages to ignore, which security rules to
count. This module lets you move one of those and see the whole picture move,
in about a second, because the raw findings are already stored per task.

Each function returns a frame with the same shape as `results_frame()`, so the
plotting functions take it unchanged:

    from cqhandson import figures, whatif
    figures.plot_scoreboard(whatif.with_complexity_gate(results, 0.30))
"""
from __future__ import annotations

import pandas as pd

from .metrics import ODC_COLUMNS

#: Pylint symbol -> ODC column, mirroring the study's mapping.
_ODC_COLUMN = {
    "Assignment": "def_assignment", "Algorithm": "def_algorithm",
    "Algorithm/Method": "def_algorithm", "Interface": "def_interface",
    "Checking": "def_checking", "Timing": "def_timing",
    "Timing/Serialization": "def_timing",
    "Function/Class/Object": "def_function_class_object",
}

#: Findings that arguably say more about the evaluation format than the author:
#: `self` is unused because the method was scored outside its class; the
#: parameter-count checks fire on the *requested* signature, so every author who
#: obeys the prompt inherits them; the class wrapper is scaffolding the model
#: emitted to look complete.
FORMAT_ARTIFACTS = {
    "unused-argument",
    "too-few-public-methods",
    "too-many-arguments",
    "too-many-positional-arguments",
}


def _rederive(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute the derived booleans after any column changed."""
    frame["defective"] = frame["defects_total"] > 0
    frame["vulnerable"] = frame["vulns_total"] > 0
    frame["high_severity"] = frame["vulns_high_sev"] > 0
    frame["clean_strict"] = (frame["strict_nontrivial"]
                             & frame["defects_total"].eq(0)
                             & frame["vulns_total"].eq(0))
    return frame


def with_complexity_gate(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Move the non-degeneracy floor. The shipped benchmark uses 0.10.

    An answer qualifies when its size reaches `threshold` × the human
    implementation, on lines of code **or** on Halstead volume.
    """
    frame = results.copy()
    passes = (frame["complexity_nloc_ratio"].fillna(-1).ge(threshold)
              | frame["complexity_halstead_volume_ratio"].fillna(-1).ge(threshold))
    frame["complexity_non_degenerate"] = passes
    frame["strict_nontrivial"] = frame["structural_strict_nontrivial"] & passes
    frame["status"] = frame["status"].where(
        ~(frame["structural_strict_nontrivial"] & ~passes), "complexity_degenerate")
    return _rederive(frame)


def without_symbols(results: pd.DataFrame, symbols=FORMAT_ARTIFACTS) -> pd.DataFrame:
    """Drop Pylint checks from the defect count, and re-derive the ODC columns."""
    symbols = set(symbols)
    frame = results.copy()

    kept = frame["defect_findings"].map(
        lambda findings: [f for f in (findings or []) if f["symbol"] not in symbols])
    frame["defect_findings"] = kept
    frame["defects_total"] = kept.map(len)
    for column in ODC_COLUMNS:
        frame[column] = kept.map(
            lambda findings: sum(_ODC_COLUMN[f["odc_category"]] == column
                                 for f in findings))
    return _rederive(frame)


def without_rules(results: pd.DataFrame, rules=("B404",)) -> pd.DataFrame:
    """Drop Semgrep rules from the security count.

    Match is by prefix, so `"B404"` also removes `B404-1` and friends. Removing
    findings can only lower a count, so no re-scan is needed.
    """
    rules = tuple(rules)
    frame = results.copy()

    def keep(findings):
        return [f for f in (findings or [])
                if not f["check_id"].split(".")[-1].startswith(rules)]

    kept = frame["vulnerability_findings"].map(keep)
    frame["vulnerability_findings"] = kept
    frame["vulns_total"] = kept.map(len)
    frame["vulns_high_sev"] = kept.map(
        lambda fs: sum(str(f["extra"]["severity"]).upper() in {"CRITICAL", "ERROR"}
                       for f in fs))
    frame["cwes"] = kept.map(lambda fs: sorted({
        c.split(":")[0].strip()
        for f in fs
        for c in ([f["extra"]["metadata"]["cwe"]]
                  if isinstance(f["extra"]["metadata"]["cwe"], str)
                  else f["extra"]["metadata"]["cwe"])}))
    return _rederive(frame)
