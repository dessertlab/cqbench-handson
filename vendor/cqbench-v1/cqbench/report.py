from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_RESAMPLES, SEED
from .evaluate import summarize
from .io import read_jsonl


RATE_METRICS = {
    "submitted_rate": "submitted",
    "parseable_rate": "parseable",
    "target_present_rate": "target_present",
    "nonstub_rate": "nonstub",
    "strict_nontrivial_rate": "strict_nontrivial",
    "defect_free_rate": lambda frame: frame["defects_total"].eq(0),
    "vulnerability_free_rate": lambda frame: frame["vulns_total"].eq(0),
    "high_severity_free_rate": lambda frame: frame["vulns_high_sev"].eq(0),
    "clean_nonstub_at_1": lambda frame: (
        frame["nonstub"] & frame["defects_total"].eq(0) & frame["vulns_total"].eq(0)
    ),
    "clean_strict_at_1": lambda frame: (
        frame["strict_nontrivial"]
        & frame["defects_total"].eq(0)
        & frame["vulns_total"].eq(0)
    ),
}


def _metric_values(frame: pd.DataFrame, metric: str) -> np.ndarray:
    definition = RATE_METRICS[metric]
    values = frame[definition] if isinstance(definition, str) else definition(frame)
    return values.astype(float).to_numpy()


def _comparison_seed(baseline: str, language: str, metric: str) -> int:
    """Deterministic per-comparison seed.

    Seeding each (baseline, language, metric) bootstrap independently makes the
    resulting interval a function of that comparison alone. It no longer depends
    on how many baselines are compared or the order they are passed, which a
    single shared RNG stream would otherwise leak into every interval.
    """
    digest = hashlib.sha256(
        f"{SEED}:{baseline}:{language}:{metric}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_mean_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    *,
    batch_size: int = 250,
) -> tuple[float, float]:
    unique, counts = np.unique(differences, return_counts=True)
    if len(unique) <= 20:
        draws = rng.multinomial(
            len(differences),
            counts / len(differences),
            size=BOOTSTRAP_RESAMPLES,
        )
        boot = draws @ unique / len(differences)
        return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    boot = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(
            0,
            len(differences),
            size=(stop - start, len(differences)),
        )
        boot[start:stop] = differences[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def compare_results(
    submission_path: Path,
    baseline_paths: list[Path],
    output_csv: Path,
) -> int:
    submission = pd.json_normalize(list(read_jsonl(submission_path)))
    rows = []
    for baseline_path in baseline_paths:
        baseline = pd.json_normalize(list(read_jsonl(baseline_path)))
        merged = submission.merge(
            baseline,
            on=["task_id", "language"],
            suffixes=("_submission", "_baseline"),
            validate="one_to_one",
        )
        assert len(merged) == len(submission) == len(baseline), (
            f"Result keys differ for {baseline_path}"
        )
        for language, group in merged.groupby("language", sort=True):
            left = pd.DataFrame(
                {
                    column.removesuffix("_submission"): group[column]
                    for column in group
                    if column.endswith("_submission")
                }
            )
            right = pd.DataFrame(
                {
                    column.removesuffix("_baseline"): group[column]
                    for column in group
                    if column.endswith("_baseline")
                }
            )
            for metric in RATE_METRICS:
                if metric.startswith(("defect_", "vulnerability_", "high_", "clean_")):
                    if left.get("defects_total", pd.Series(dtype=float)).isna().any():
                        continue
                a, b = _metric_values(left, metric), _metric_values(right, metric)
                differences = a - b
                rng = np.random.default_rng(
                    _comparison_seed(baseline_path.stem, language, metric)
                )
                ci_lo, ci_hi = _bootstrap_mean_ci(differences, rng)
                rows.append(
                    {
                        "baseline": baseline_path.stem,
                        "language": language,
                        "metric": metric,
                        "submission": float(a.mean()),
                        "baseline_value": float(b.mean()),
                        "delta": float(differences.mean()),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "n": len(differences),
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def write_report(results_path: Path, output_dir: Path, model_name: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Palette extracted from the paper's figures (teal -> blue for categoricals,
    # a Blues sequential colormap for heatmaps).
    PAPER_PALETTE = ["#008080", "#40e0d0", "#afeeee", "#4169e1", "#87cefa", "#4682b4"]
    HEATMAP_CMAP = "Blues"

    frame, summary = summarize(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    flat_rows = []
    for language, values in summary["languages"].items():
        flat_rows.append({"model": model_name, "language": language, **values})
    summary_csv = output_dir / "summary.csv"
    pd.DataFrame(flat_rows).to_csv(summary_csv, index=False)

    markdown = [
        f"# CQBench report: {model_name}",
        "",
        "This benchmark measures static-analysis findings and structural non-triviality; "
        "it does not establish functional correctness or actual exploitability.",
        "",
        "| Language | N | Non-stub | Strict non-trivial | Defect incidence | "
        "Vulnerability incidence | High-severity incidence | Clean strict@1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def percent(value: Any) -> str:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.1%}"

    for row in flat_rows:
        markdown.append(
            f"| {row['language']} | {row['n']} | {percent(row['nonstub_rate'])} | "
            f"{percent(row['strict_nontrivial_rate'])} | "
            f"{percent(row.get('defect_incidence_all'))} | "
            f"{percent(row.get('vulnerability_incidence_all'))} | "
            f"{percent(row.get('high_severity_incidence_all'))} | "
            f"{percent(row.get('clean_strict_at_1'))} |"
        )
    report_path = output_dir / "report.md"
    complexity_rows = []
    for language, values in summary["languages"].items():
        for metric, value in values.get("complexity_means_strict", {}).items():
            complexity_rows.append(
                {"model": model_name, "language": language, "metric": metric, "mean": value}
            )
    complexity_csv = output_dir / "complexity_summary.csv"
    pd.DataFrame(complexity_rows).to_csv(complexity_csv, index=False)
    if complexity_rows:
        selected_metrics = (
            "nloc_mean",
            "ccn_mean",
            "parameter_count_mean",
            "max_nesting_depth_mean",
            "distinct_operators_mean",
            "total_operators_mean",
            "distinct_operands_mean",
            "total_operands_mean",
            "halstead_volume_mean",
            "halstead_difficulty_mean",
            "halstead_effort_mean",
            "maintainability_index_mean",
            "function_name_length_mean",
            "target_token_count_mean",
        )
        lookup = {
            (row["language"], row["metric"]): row["mean"] for row in complexity_rows
        }
        markdown.extend(
            [
                "",
                "## Structural metrics among strict non-trivial outputs",
                "",
                "| Language | Metric | Mean |",
                "|---|---|---:|",
            ]
        )
        for language in sorted(summary["languages"]):
            for metric in selected_metrics:
                if (language, metric) in lookup:
                    markdown.append(
                        f"| {language} | {metric} | {lookup[(language, metric)]:.3f} |"
                    )
            markdown.append(
                f"| {language} | unique_tokens_corpus | "
                f"{summary['languages'][language]['unique_tokens_corpus']} |"
            )

    odc_rows = []
    for language, values in summary["languages"].items():
        for odc, value in values.get("odc_incidence", {}).items():
            odc_rows.append(
                {"model": model_name, "language": language, "odc": odc, "incidence": value}
            )
    odc_csv = output_dir / "odc_incidence.csv"
    pd.DataFrame(odc_rows).to_csv(odc_csv, index=False)
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    plot_columns = [
        ("nonstub", "Non-stub"),
        ("strict_nontrivial", "Strict non-trivial"),
    ]
    if frame["defects_total"].notna().all():
        frame["defect_free"] = frame["defects_total"].eq(0)
        frame["vulnerability_free"] = frame["vulns_total"].eq(0)
        frame["high_severity_free"] = frame["vulns_high_sev"].eq(0)
        plot_columns.extend(
            [
                ("defect_free", "Defect-free"),
                ("vulnerability_free", "SAST-alert-free"),
                ("high_severity_free", "High-severity-alert-free"),
            ]
        )
    plot = (
        frame.groupby("language")[[column for column, _ in plot_columns]]
        .mean()
        .rename(columns=dict(plot_columns))
    )
    axes = plot.plot(
        kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0,
        color=PAPER_PALETTE[: len(plot.columns)],
    )
    axes.set_ylabel("Proportion of benchmark tasks")
    axes.set_title(f"CQBench static quality profile — {model_name}")
    axes.legend(loc="lower right")
    plt.tight_layout()
    png = output_dir / "quality_rates.png"
    pdf = output_dir / "quality_rates.pdf"
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    plt.close()
    if odc_rows:
        heatmap = pd.DataFrame(odc_rows).pivot(
            index="odc", columns="language", values="incidence"
        )
        figure, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(heatmap.to_numpy(), cmap=HEATMAP_CMAP, vmin=0, vmax=1)
        axis.set_xticks(range(len(heatmap.columns)), heatmap.columns)
        axis.set_yticks(range(len(heatmap.index)), heatmap.index)
        for row_index in range(len(heatmap.index)):
            for column_index in range(len(heatmap.columns)):
                value = heatmap.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, label="Task incidence")
        axis.set_title(f"ODC incidence — {model_name}")
        figure.tight_layout()
        figure.savefig(output_dir / "odc_heatmap.png", dpi=200)
        figure.savefig(output_dir / "odc_heatmap.pdf")
        plt.close(figure)
    return {
        "summary_json": str(summary_path),
        "summary_csv": str(summary_csv),
        "report": str(report_path),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "complexity_csv": str(complexity_csv),
        "odc_csv": str(odc_csv),
    }
