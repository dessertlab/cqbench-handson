"""
Compute auditable complexity metrics for every function in every code sample.

This script reuses the metric definitions in ``complexity_metrics_extended.py``
but does not assume that Lizard function index 0 is the requested or primary
function. It can write two JSONL outputs:

1. A function table containing every function detected by Lizard, including
   its name, qualified name, source span, nesting relationship, source-order
   index, computability status, and the original complexity metrics.
2. A sample/author table containing counts and explicit sum/mean/median/min/max
   summaries over every computable function in that sample.

The aggregate suffixes are deliberate. For example, ``ccn_sum`` is the sum of
the per-function CCN values; it is not silently presented as the CCN of one
arbitrarily selected function. Nested function spans can overlap their parent
spans, so sums of NLOC or Halstead values are sums of function measurements,
not necessarily unique physical source lines or tokens.

Functions with non-positive NLOC or degenerate Halstead vocabulary are emitted
with ``metrics_computable=false`` and a drop reason. They are excluded from the
sample metric summaries, never imputed, and remain visible for auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import lizard
import numpy as np
from tqdm import tqdm

from support.complexity_metrics_extended import (
    LANGUAGE_CONFIG,
    METRIC_NAMES,
    compute_halstead_metrics,
    compute_maintainability_index,
    compute_max_nesting_depth,
    parse_field_spec,
    strip_python_docstrings,
)


AUTHOR_ALIASES = {
    "human_code": "human",
    "chatgpt_code": "chatgpt",
    "openAI_code": "chatgpt",
    "dsc_code": "dsc",
    "qwen_code": "qwen",
    "gptoss_code": "gptoss",
}

AGGREGATIONS = ("sum", "mean", "median", "min", "max")
SCHEMA_VERSION = 1


def normalize_author(field: str) -> str:
    if field in AUTHOR_ALIASES:
        return AUTHOR_ALIASES[field]
    return field[:-5] if field.endswith("_code") else field


def short_function_name(name: str) -> str:
    """Return the innermost function/method name from Lizard's qualified name."""
    return name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]


def function_container(name: str) -> str | None:
    """Return Lizard's lexical/type container, if one is encoded in the name."""
    if "::" in name:
        return name.rsplit("::", 1)[0]
    if "." in name:
        return name.rsplit(".", 1)[0]
    return None


def parent_function_indices(functions: Sequence[Any]) -> list[int | None]:
    """
    Infer lexical function nesting from source-span containment.

    Java class membership does not count as function nesting. Anonymous-class
    methods and Python nested functions do count when their complete span is
    inside another function's span.
    """
    parents: list[int | None] = []
    for child_index, child in enumerate(functions):
        candidates: list[tuple[int, int]] = []
        child_span = (int(child.start_line), int(child.end_line))
        for candidate_index, candidate in enumerate(functions):
            if candidate_index == child_index:
                continue
            candidate_span = (int(candidate.start_line), int(candidate.end_line))
            strictly_contains = (
                candidate_span[0] <= child_span[0]
                and child_span[1] <= candidate_span[1]
                and candidate_span != child_span
            )
            if strictly_contains:
                span_size = candidate_span[1] - candidate_span[0]
                candidates.append((span_size, candidate_index))
        parents.append(min(candidates)[1] if candidates else None)
    return parents


def source_order_indices(functions: Sequence[Any]) -> dict[int, int]:
    ordered = sorted(
        range(len(functions)),
        key=lambda index: (
            int(functions[index].start_line),
            int(functions[index].end_line),
            index,
        ),
    )
    return {function_index: source_index for source_index, function_index in enumerate(ordered)}


def metric_values(func: Any, snippet: str, language: str) -> tuple[dict[str, float], str | None]:
    """
    Reuse the base script's formulas and return (metrics, drop_reason).

    A non-null drop reason means the function remains in the detailed output
    but is excluded from aggregate summaries.
    """
    analysis_snippet = strip_python_docstrings(snippet) if language == "python" else snippet
    halstead = compute_halstead_metrics(analysis_snippet, language)
    nloc = int(func.nloc)
    ccn = int(func.cyclomatic_complexity)

    if nloc <= 0:
        drop_reason = "nonpositive_nloc"
    elif halstead["distinct_operators"] == 0:
        drop_reason = "no_operators"
    elif halstead["distinct_operands"] == 0:
        drop_reason = "no_operands"
    else:
        drop_reason = None

    values: dict[str, float] = {
        "nloc": nloc,
        "ccn": ccn,
        "parameter_count": int(func.parameter_count),
        "max_nesting_depth": compute_max_nesting_depth(analysis_snippet, language),
        **halstead,
        "maintainability_index": compute_maintainability_index(
            halstead["halstead_volume"],
            ccn,
            nloc,
        ),
    }
    assert set(values) == set(METRIC_NAMES)
    assert all(math.isfinite(float(value)) for value in values.values())
    assert all(float(value) >= 0.0 for value in values.values())
    return values, drop_reason


def analyze_all_functions(
    code: str,
    language: str,
    row_id: str,
    field: str,
    include_source: bool,
) -> list[dict[str, Any]]:
    """Return one auditable record for every Lizard-detected function."""
    config = LANGUAGE_CONFIG[language]
    analysis = lizard.analyze_file.analyze_source_code(config["filename"], code)
    functions = list(analysis.function_list)
    parents = parent_function_indices(functions)
    source_orders = source_order_indices(functions)
    lines = code.splitlines()
    author = normalize_author(field)

    preliminary: list[dict[str, Any]] = []
    for index, func in enumerate(functions):
        start_line = int(func.start_line)
        end_line = int(func.end_line)
        span_valid = 1 <= start_line <= end_line <= len(lines)
        if span_valid:
            snippet = "\n".join(lines[start_line - 1 : end_line])
            values, drop_reason = metric_values(func, snippet, language)
            source_sha256 = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
        else:
            # Lizard can expose partial function objects with line 0 when it
            # recovers from malformed source. Preserve their identity for the
            # audit trail, but never fabricate a snippet or complexity value.
            snippet = None
            values = {metric: None for metric in METRIC_NAMES}
            drop_reason = "invalid_source_span"
            source_sha256 = None
        qualified_name = str(func.name)
        function_uid = f"{row_id}:{field}:{index}"
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "field": field,
            "author": author,
            "language": language,
            "row_id": row_id,
            "function_uid": function_uid,
            # This is Lizard output order, not a primary-function indicator.
            "function_index_in_row": index,
            "source_order_index": source_orders[index],
            "function_name": short_function_name(qualified_name),
            "qualified_name": qualified_name,
            "long_name": str(func.long_name),
            "container_name": function_container(qualified_name),
            "start_line": start_line,
            "end_line": end_line,
            "parent_function_index": parents[index],
            "is_nested_function": parents[index] is not None,
            "parameter_names": [str(value) for value in getattr(func, "parameters", [])],
            "source_sha256": source_sha256,
            "metrics_computable": drop_reason is None,
            "drop_reason": drop_reason,
            **values,
        }
        if include_source:
            record["source"] = snippet
        preliminary.append(record)

    computable_count = sum(bool(record["metrics_computable"]) for record in preliminary)
    for record in preliminary:
        parent_index = record.pop("parent_function_index")
        record["parent_function_uid"] = (
            preliminary[parent_index]["function_uid"] if parent_index is not None else None
        )
        record["function_count_detected_in_sample"] = len(preliminary)
        record["function_count_computable_in_sample"] = computable_count

    assert len({record["function_uid"] for record in preliminary}) == len(preliminary)
    assert sorted(record["source_order_index"] for record in preliminary) == list(
        range(len(preliminary))
    )
    return preliminary


def aggregate_functions(
    row_id: str,
    field: str,
    language: str,
    records: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    """Create explicit summaries over every computable function in one sample."""
    computable = [record for record in records if record["metrics_computable"]]
    aggregate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "field": field,
        "author": normalize_author(field),
        "language": language,
        "row_id": row_id,
        "analysis_status": status,
        "function_count_detected": len(records),
        "function_count_computable": len(computable),
        "function_count_dropped": len(records) - len(computable),
        "function_count_nested": sum(bool(record["is_nested_function"]) for record in records),
        "function_count_outermost": sum(
            not bool(record["is_nested_function"]) for record in records
        ),
        "function_uids": [record["function_uid"] for record in records],
        "function_names": [record["qualified_name"] for record in records],
    }

    for metric in METRIC_NAMES:
        values = np.asarray([record[metric] for record in computable], dtype=float)
        if values.size == 0:
            for operation in AGGREGATIONS:
                aggregate[f"{metric}_{operation}"] = None
            continue
        aggregate[f"{metric}_sum"] = float(values.sum())
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_median"] = float(np.median(values))
        aggregate[f"{metric}_min"] = float(values.min())
        aggregate[f"{metric}_max"] = float(values.max())

    return aggregate


class AtomicJsonlWriter:
    """Write JSONL atomically so failed runs cannot masquerade as complete output."""

    def __init__(self, path: Path | None, overwrite: bool):
        self.path = path
        self.overwrite = overwrite
        self.temp_path: Path | None = None
        self.handle = None

    def __enter__(self):
        if self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {self.path}; pass --overwrite explicitly."
            )
        self.temp_path = self.path.with_name(f".{self.path.name}.tmp")
        self.handle = self.temp_path.open("w", encoding="utf-8")
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self.handle is not None:
            self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.close()
        if self.temp_path is None or self.path is None:
            return False
        if exc_type is None:
            os.replace(self.temp_path, self.path)
        elif self.temp_path.exists():
            self.temp_path.unlink()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--fields",
        nargs="+",
        required=True,
        help="Field specs such as human_code:python chatgpt_code:python.",
    )
    parser.add_argument("--id-field", default="hm_index")
    parser.add_argument(
        "--functions-output",
        type=Path,
        help="JSONL output with one record for every detected function.",
    )
    parser.add_argument(
        "--aggregates-output",
        type=Path,
        help="JSONL output with one all-functions summary per sample and field.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include each function's source text; hashes are always emitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many selected dataset rows (for validation only).",
    )
    parser.add_argument(
        "--only-id",
        action="append",
        default=[],
        help="Process only this row ID; repeat to select multiple IDs.",
    )
    parser.add_argument(
        "--continue-on-analysis-error",
        action="store_true",
        help="Log a field analysis error and continue; the default is fail-fast.",
    )
    args = parser.parse_args()
    if args.functions_output is None and args.aggregates_output is None:
        parser.error("At least one of --functions-output or --aggregates-output is required.")
    if (
        args.functions_output is not None
        and args.aggregates_output is not None
        and args.functions_output.resolve() == args.aggregates_output.resolve()
    ):
        parser.error("Function and aggregate outputs must be different paths.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def print_summary(
    dataset: Path,
    processed_rows: int,
    stats: Mapping[str, Counter],
) -> None:
    summary = {
        "dataset": str(dataset),
        "processed_dataset_rows": processed_rows,
        "fields": {field: dict(counter) for field, counter in stats.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    assert args.dataset.is_file(), f"Dataset does not exist: {args.dataset}"
    field_specs = [parse_field_spec(spec) for spec in args.fields]
    fields = [field for field, _ in field_specs]
    assert len(fields) == len(set(fields)), f"Duplicate field specifications: {fields}"
    selected_ids = set(args.only_id)

    seen_ids: set[str] = set()
    stats: dict[str, Counter] = defaultdict(Counter)
    processed_rows = 0

    with AtomicJsonlWriter(args.functions_output, args.overwrite) as function_writer, (
        AtomicJsonlWriter(args.aggregates_output, args.overwrite)
    ) as aggregate_writer:
        with args.dataset.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(tqdm(handle, desc="Processing"), start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_id = item.get(args.id_field)
                assert isinstance(raw_id, str) and raw_id, (
                    f"Missing/invalid {args.id_field!r} at line {line_number}"
                )
                row_id = raw_id
                if selected_ids and row_id not in selected_ids:
                    continue
                assert row_id not in seen_ids, f"Duplicate row ID: {row_id}"
                seen_ids.add(row_id)
                processed_rows += 1

                for field, language in field_specs:
                    stats[field]["samples_seen"] += 1
                    code = item.get(field, "")
                    assert code is None or isinstance(code, str), (
                        f"Non-string code at {(line_number, row_id, field)}"
                    )
                    code = code or ""
                    if not code.strip():
                        records: list[dict[str, Any]] = []
                        status = "empty_code"
                        stats[field]["samples_empty_code"] += 1
                    else:
                        try:
                            records = analyze_all_functions(
                                code,
                                language,
                                row_id,
                                field,
                                args.include_source,
                            )
                            status = "ok" if records else "no_functions_detected"
                        except Exception:
                            if not args.continue_on_analysis_error:
                                raise
                            records = []
                            status = "analysis_error"
                            stats[field]["samples_analysis_error"] += 1

                        if records:
                            stats[field]["samples_with_functions"] += 1
                        else:
                            stats[field]["samples_no_functions"] += 1

                    if len(records) > 1:
                        stats[field]["samples_with_multiple_functions"] += 1
                    stats[field]["functions_detected"] += len(records)
                    stats[field]["functions_computable"] += sum(
                        bool(record["metrics_computable"]) for record in records
                    )
                    stats[field]["functions_dropped"] += sum(
                        not bool(record["metrics_computable"]) for record in records
                    )

                    for record in records:
                        function_writer.write(record)
                    aggregate_writer.write(
                        aggregate_functions(row_id, field, language, records, status)
                    )

                if args.limit is not None and processed_rows >= args.limit:
                    break

    if selected_ids:
        missing = selected_ids - seen_ids
        assert not missing, f"Requested IDs not found: {sorted(missing)}"
    assert processed_rows > 0, "No dataset rows were selected."
    print_summary(args.dataset, processed_rows, stats)


if __name__ == "__main__":
    main()
