from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from support.complexity_metrics_all_functions import (
    aggregate_functions,
    analyze_all_functions,
)

from .analyzers import analyze_defects, analyze_semgrep
from .config import BENCHMARK_VERSION, ODC_COLUMNS, ROOT
from .io import load_index, read_jsonl, write_jsonl_atomic
from .structural import Signature, analyze_structure


COMPLEXITY_RATIO_THRESHOLD = 0.10


def _complexity_gate(
    complexity: dict[str, Any],
    human_complexity: dict[str, Any] | None,
) -> dict[str, Any]:
    if not human_complexity:
        return {
            "complexity_nloc_ratio": None,
            "complexity_halstead_volume_ratio": None,
            "complexity_non_degenerate": True,
        }
    generated_nloc = complexity.get("nloc_mean")
    generated_volume = complexity.get("halstead_volume_mean")
    human_nloc = human_complexity.get("nloc")
    human_volume = human_complexity.get("halstead_v")
    if not all(
        isinstance(value, (int, float)) and float(value) > 0
        for value in (generated_nloc, generated_volume, human_nloc, human_volume)
    ):
        return {
            "complexity_nloc_ratio": None,
            "complexity_halstead_volume_ratio": None,
            "complexity_non_degenerate": False,
        }
    nloc_ratio = float(generated_nloc) / float(human_nloc)
    volume_ratio = float(generated_volume) / float(human_volume)
    return {
        "complexity_nloc_ratio": nloc_ratio,
        "complexity_halstead_volume_ratio": volume_ratio,
        "complexity_non_degenerate": bool(
            nloc_ratio >= COMPLEXITY_RATIO_THRESHOLD
            or volume_ratio >= COMPLEXITY_RATIO_THRESHOLD
        ),
    }


def validate_submission(
    tasks_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    tasks = load_index(tasks_path, "task_id")
    predictions: dict[str, str] = {}
    for row in read_jsonl(predictions_path):
        task_id = row.get("task_id")
        code = row.get("code")
        assert isinstance(task_id, str) and task_id, "Invalid prediction task_id"
        assert task_id in tasks, f"Unknown prediction task_id: {task_id}"
        assert task_id not in predictions, f"Duplicate prediction: {task_id}"
        assert isinstance(code, str), f"Prediction code must be a string: {task_id}"
        predictions[task_id] = code
    return tasks, predictions


def _complexity(code: str, language: str, task_id: str) -> dict[str, Any]:
    if not code.strip():
        return {
            "analysis_status": "empty_code",
            "function_count_detected": 0,
            "function_count_computable": 0,
        }
    try:
        functions = analyze_all_functions(code, language, task_id, "submission_code", False)
        status = "ok" if functions else "no_functions_detected"
        aggregate = aggregate_functions(task_id, "submission_code", language, functions, status)
        aggregate["function_name_lengths"] = [
            len(str(row["function_name"])) for row in functions
        ]
        return aggregate
    except Exception as exc:
        return {
            "analysis_status": "analysis_error",
            "analysis_error": str(exc),
            "function_count_detected": 0,
            "function_count_computable": 0,
        }


def _lexical_tokens(code: str) -> list[str]:
    return sorted(
        set(
            re.findall(
                r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||->|[^\s]",
                code,
            )
        )
    )


def evaluate(
    tasks_path: Path,
    references_path: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    rules_path: Path | None = None,
    structural_only: bool = False,
    overwrite: bool = False,
) -> int:
    tasks, predictions = validate_submission(tasks_path, predictions_path)
    references = load_index(references_path, "task_id")
    assert set(tasks) == set(references), "Task/reference keys differ"
    rules = rules_path or ROOT / "cqbench/rules/semgrep.json"
    rows = []
    for task_id, task in tasks.items():
        signature = Signature(**task["signature"])
        human = references[task_id]["human_metrics"]
        code = predictions.get(task_id, "")
        structure = analyze_structure(
            code,
            task["language"],
            signature,
            human_token_count=int(human["token_count"]),
            human_ast_count=int(human["ast_node_count"]),
        )
        complexity = _complexity(code, task["language"], task_id)
        gate = _complexity_gate(
            complexity,
            references[task_id].get("human_complexity"),
        )
        structure_values = structure.to_dict()
        structure_values["structural_strict_nontrivial"] = structure.strict_nontrivial
        structure_values["strict_nontrivial"] = bool(
            structure.strict_nontrivial and gate["complexity_non_degenerate"]
        )
        if structure.strict_nontrivial and not gate["complexity_non_degenerate"]:
            structure_values["status"] = "complexity_degenerate"
        row: dict[str, Any] = {
            "benchmark_version": BENCHMARK_VERSION,
            "task_id": task_id,
            "language": task["language"],
            "stratum": task["stratum"],
            "submitted": task_id in predictions,
            **structure_values,
            **gate,
            "complexity": complexity,
            "lexical_tokens": _lexical_tokens(code),
        }
        if structural_only:
            row.update(
                {
                    "defects_total": None,
                    **{column: None for column in ODC_COLUMNS},
                    "vulns_total": None,
                    "vulns_high_sev": None,
                    "cwes": [],
                    "static_analysis_status": "skipped",
                }
            )
        else:
            defects = analyze_defects(task["language"], code)
            vulnerabilities = analyze_semgrep(code, task["language"], rules)
            row.update(defects)
            row.update(vulnerabilities)
            row["static_analysis_status"] = "ok"
        rows.append(row)
    assert len(rows) == len(tasks)
    return write_jsonl_atomic(output_path, rows, overwrite=overwrite)


def summarize(results_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = list(read_jsonl(results_path))
    frame = pd.json_normalize(rows)
    assert not frame.empty
    summary: dict[str, Any] = {"n": len(frame), "languages": {}}
    for language, group in frame.groupby("language", sort=True):
        values: dict[str, Any] = {"n": len(group)}
        for column in (
            "submitted",
            "parseable",
            "target_present",
            "nonstub",
            "strict_nontrivial",
        ):
            values[f"{column}_rate"] = float(group[column].mean())
        if group["defects_total"].notna().all():
            values.update(
                {
                    "defect_incidence_all": float((group["defects_total"] > 0).mean()),
                    "vulnerability_incidence_all": float((group["vulns_total"] > 0).mean()),
                    "high_severity_incidence_all": float(
                        (group["vulns_high_sev"] > 0).mean()
                    ),
                    "clean_nonstub_at_1": float(
                        (
                            group["nonstub"]
                            & group["defects_total"].eq(0)
                            & group["vulns_total"].eq(0)
                        ).mean()
                    ),
                    "clean_strict_at_1": float(
                        (
                            group["strict_nontrivial"]
                            & group["defects_total"].eq(0)
                            & group["vulns_total"].eq(0)
                        ).mean()
                    ),
                }
            )
            values["odc_incidence"] = {
                column: float((group[column] > 0).mean()) for column in ODC_COLUMNS
            }
        complexity_columns = [
            column
            for column in group.columns
            if column.startswith("complexity.")
            and column.endswith(("_mean", "_sum"))
            and pd.api.types.is_numeric_dtype(group[column])
        ]
        values["complexity_means_strict"] = {
            column.removeprefix("complexity."): float(
                group.loc[group["strict_nontrivial"], column].dropna().mean()
            )
            for column in complexity_columns
            if group.loc[group["strict_nontrivial"], column].notna().any()
        }
        strict_group = group.loc[group["strict_nontrivial"]]
        function_name_lengths = [
            int(length)
            for lengths in strict_group.get(
                "complexity.function_name_lengths", pd.Series(dtype=object)
            )
            for length in (lengths if isinstance(lengths, list) else [])
        ]
        if function_name_lengths:
            values["complexity_means_strict"]["function_name_length_mean"] = float(
                np.mean(function_name_lengths)
            )
        if not strict_group.empty:
            values["complexity_means_strict"]["target_token_count_mean"] = float(
                strict_group["token_count"].mean()
            )
        values["unique_tokens_corpus"] = len(
            {
                token
                for tokens in group["lexical_tokens"]
                for token in (tokens if isinstance(tokens, list) else [])
            }
        )
        summary["languages"][language] = values
    return frame, summary
