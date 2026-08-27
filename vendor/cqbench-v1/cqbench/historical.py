from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import BENCHMARK_VERSION, LANGUAGES, ODC_COLUMNS
from .evaluate import _complexity, _complexity_gate, _lexical_tokens
from .io import load_index, read_jsonl, write_jsonl_atomic
from .structural import Signature, analyze_structure


def subset_results(
    tasks_path: Path,
    results_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    task_ids = [row["task_id"] for row in read_jsonl(tasks_path)]
    assert len(task_ids) == len(set(task_ids)), "Duplicate task_id in tasks"
    results = {}
    for row in read_jsonl(results_path):
        task_id = row.get("task_id")
        assert isinstance(task_id, str) and task_id
        assert task_id not in results, f"Duplicate result for {task_id}"
        results[task_id] = row
    missing = set(task_ids) - set(results)
    assert not missing, f"Missing results for {sorted(missing)[:10]}"
    rows = [results[task_id] for task_id in task_ids]
    assert all(row["task_id"] == task_id for row, task_id in zip(rows, task_ids))
    return write_jsonl_atomic(output_path, rows, overwrite=overwrite)


def export_historical_results(
    candidates_path: Path,
    model: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    assert model in {"human", "openai", "dsc", "qwen"}
    candidates = list(read_jsonl(candidates_path))
    tables = {}
    for language, specification in LANGUAGES.items():
        table = pd.read_parquet(
            specification.table,
            columns=[
                "hm_index",
                "author",
                "defects_total",
                *ODC_COLUMNS,
                "vulns_total",
                "vulns_high_sev",
            ],
        )
        tables[language] = table.set_index(["hm_index", "author"])

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        language = candidate["language"]
        author = (
            ("gptoss" if language == "c" else "chatgpt")
            if model == "openai"
            else model
        )
        source_id = candidate["source_id"]
        table_row = tables[language].loc[(source_id, author)]
        code = candidate["codes"][author]
        structure = candidate["structures"][author]
        row = {
            "benchmark_version": BENCHMARK_VERSION,
            "task_id": candidate["task_id"],
            "language": language,
            "stratum": candidate["stratum"],
            "submitted": True,
            **structure,
            "complexity": _complexity(code, language, candidate["task_id"]),
            "lexical_tokens": _lexical_tokens(code),
            "defects_total": int(table_row["defects_total"]),
            **{column: int(table_row[column]) for column in ODC_COLUMNS},
            "vulns_total": int(table_row["vulns_total"]),
            "vulns_high_sev": int(table_row["vulns_high_sev"]),
            "cwes": candidate["selection"]["cwe_by_author"][author],
            "static_analysis_status": "precomputed_frozen_study_results",
        }
        rows.append(row)
    assert len(rows) == len(candidates)
    return write_jsonl_atomic(output_path, rows, overwrite=overwrite)


def export_large_historical_results(
    benchmark_dir: Path,
    model: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    assert model in {"human", "openai", "dsc", "qwen"}
    tasks = list(read_jsonl(benchmark_dir / "tasks.jsonl"))
    references = load_index(benchmark_dir / "references.jsonl", "task_id")
    predictions = load_index(
        benchmark_dir / "baselines" / f"{model}.jsonl",
        "task_id",
    )
    assert {row["task_id"] for row in tasks} == set(references) == set(predictions)
    metric_columns = [
        "function_count_computable",
        "nloc",
        "ccn",
        "parameter_count",
        "max_nesting_depth",
        "distinct_operators",
        "total_operators",
        "distinct_operands",
        "total_operands",
        "halstead_v",
        "halstead_difficulty",
        "halstead_effort",
        "mi",
    ]
    outcome_columns = [
        "defects_total",
        *ODC_COLUMNS,
        "vulns_total",
        "vulns_high_sev",
    ]
    tables = {}
    for language, specification in LANGUAGES.items():
        frame = pd.read_parquet(
            specification.table,
            columns=["hm_index", "author", *metric_columns, *outcome_columns],
        )
        assert not frame.duplicated(["hm_index", "author"]).any()
        tables[language] = frame.set_index(["hm_index", "author"])

    metric_names = {
        "nloc": "nloc_mean",
        "ccn": "ccn_mean",
        "parameter_count": "parameter_count_mean",
        "max_nesting_depth": "max_nesting_depth_mean",
        "distinct_operators": "distinct_operators_mean",
        "total_operators": "total_operators_mean",
        "distinct_operands": "distinct_operands_mean",
        "total_operands": "total_operands_mean",
        "halstead_v": "halstead_volume_mean",
        "halstead_difficulty": "halstead_difficulty_mean",
        "halstead_effort": "halstead_effort_mean",
        "mi": "maintainability_index_mean",
    }
    rows = []
    for task in tasks:
        task_id = task["task_id"]
        language = task["language"]
        author = (
            ("gptoss" if language == "c" else "chatgpt")
            if model == "openai"
            else model
        )
        values = tables[language].loc[(task["source_id"], author)]
        code = predictions[task_id]["code"]
        structure = analyze_structure(
            code,
            language,
            Signature(**task["signature"]),
            human_token_count=int(references[task_id]["human_metrics"]["token_count"]),
            human_ast_count=int(
                references[task_id]["human_metrics"]["ast_node_count"]
            ),
        )
        complexity = {
            "analysis_status": "ok",
            "function_count_computable": int(values["function_count_computable"]),
            **{
                target: float(values[source])
                for source, target in metric_names.items()
            },
        }
        gate = _complexity_gate(
            complexity,
            references[task_id].get("human_complexity"),
        )
        structure_values = structure.to_dict()
        structure_values["structural_strict_nontrivial"] = (
            structure.strict_nontrivial
        )
        structure_values["strict_nontrivial"] = bool(
            structure.strict_nontrivial
            and gate["complexity_non_degenerate"]
        )
        if structure.strict_nontrivial and not gate["complexity_non_degenerate"]:
            structure_values["status"] = "complexity_degenerate"
        rows.append(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "task_id": task_id,
                "language": language,
                "stratum": task["stratum"],
                "submitted": True,
                **structure_values,
                **gate,
                "complexity": complexity,
                "lexical_tokens": _lexical_tokens(code),
                "defects_total": int(values["defects_total"]),
                **{
                    column: int(values[column])
                    for column in ODC_COLUMNS
                },
                "vulns_total": int(values["vulns_total"]),
                "vulns_high_sev": int(values["vulns_high_sev"]),
                "cwes": [],
                "static_analysis_status": "precomputed_frozen_study_results",
            }
        )
    assert len(rows) == len(tasks)
    return write_jsonl_atomic(output_path, rows, overwrite=overwrite)
