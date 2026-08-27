from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, validate_submission
from .historical import (
    export_historical_results,
    export_large_historical_results,
    subset_results,
)
from .large import build_large_benchmark
from .report import compare_results, write_report
from .rules import vendor_rules


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cqbench")
    commands = result.add_subparsers(dest="command", required=True)

    vendor = commands.add_parser("vendor-rules")
    vendor.add_argument(
        "--output",
        type=Path,
        default=Path("cqbench/rules/semgrep.json"),
    )
    vendor.add_argument("--overwrite", action="store_true")

    historical = commands.add_parser("historical-results")
    historical.add_argument("--candidates", type=Path, required=True)
    historical.add_argument(
        "--model", choices=("human", "openai", "dsc", "qwen"), required=True
    )
    historical.add_argument("--output", type=Path, required=True)
    historical.add_argument("--overwrite", action="store_true")

    subset = commands.add_parser("subset-results")
    subset.add_argument("--tasks", type=Path, required=True)
    subset.add_argument("--results", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--overwrite", action="store_true")

    large_historical = commands.add_parser("large-historical-results")
    large_historical.add_argument("--benchmark-dir", type=Path, required=True)
    large_historical.add_argument(
        "--model", choices=("human", "openai", "dsc", "qwen"), required=True
    )
    large_historical.add_argument("--output", type=Path, required=True)
    large_historical.add_argument("--overwrite", action="store_true")

    large = commands.add_parser("build-large")
    large.add_argument("--output-dir", type=Path, required=True)
    large.add_argument("--difficulty-threshold", type=int, default=3)
    large.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser("validate-submission")
    validate.add_argument("--tasks", type=Path, required=True)
    validate.add_argument("--predictions", type=Path, required=True)

    run = commands.add_parser("evaluate")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--references", type=Path, required=True)
    run.add_argument("--predictions", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--rules", type=Path)
    run.add_argument("--structural-only", action="store_true")
    run.add_argument("--overwrite", action="store_true")

    compare = commands.add_parser("compare")
    compare.add_argument("--submission", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)

    report = commands.add_parser("report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--model-name", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "vendor-rules":
        print(json.dumps(vendor_rules(args.output, overwrite=args.overwrite), indent=2))
    elif args.command == "historical-results":
        print(
            json.dumps(
                {
                    "results": export_historical_results(
                        args.candidates,
                        args.model,
                        args.output,
                        overwrite=args.overwrite,
                    )
                }
            )
        )
    elif args.command == "subset-results":
        print(
            json.dumps(
                {
                    "results": subset_results(
                        args.tasks,
                        args.results,
                        args.output,
                        overwrite=args.overwrite,
                    )
                }
            )
        )
    elif args.command == "large-historical-results":
        print(
            json.dumps(
                {
                    "results": export_large_historical_results(
                        args.benchmark_dir,
                        args.model,
                        args.output,
                        overwrite=args.overwrite,
                    )
                }
            )
        )
    elif args.command == "build-large":
        print(
            json.dumps(
                build_large_benchmark(
                    args.output_dir,
                    threshold=args.difficulty_threshold,
                    overwrite=args.overwrite,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "validate-submission":
        tasks, predictions = validate_submission(args.tasks, args.predictions)
        print(json.dumps({"tasks": len(tasks), "predictions": len(predictions)}))
    elif args.command == "evaluate":
        print(
            json.dumps(
                {
                    "evaluated": evaluate(
                        args.tasks,
                        args.references,
                        args.predictions,
                        args.output,
                        rules_path=args.rules,
                        structural_only=args.structural_only,
                        overwrite=args.overwrite,
                    )
                }
            )
        )
    elif args.command == "compare":
        print(
            json.dumps(
                {
                    "comparisons": compare_results(
                        args.submission, args.baseline, args.output
                    )
                }
            )
        )
    elif args.command == "report":
        print(
            json.dumps(
                write_report(args.results, args.output_dir, args.model_name),
                indent=2,
            )
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
