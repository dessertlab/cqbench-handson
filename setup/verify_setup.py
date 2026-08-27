#!/usr/bin/env python3
"""Run this once after creating the conda environment.

    conda activate cqbench-handson
    python setup/verify_setup.py

It checks the things that actually break on the day: the two analyzer binaries,
their exact versions, the data files, and one end-to-end scoring call.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems: list[str] = []


def check(label: str, condition: bool, detail: str = "", fatal: bool = True) -> bool:
    print(f"[{OK if condition else (BAD if fatal else WARN)}] {label}"
          + (f"  ({detail})" if detail else ""))
    if not condition and fatal:
        problems.append(label)
    return condition


def main() -> int:
    print("CQBench hands-on — environment check\n" + "-" * 52)

    check("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0])

    try:
        import pandas, numpy, scipy, matplotlib, lizard, openpyxl  # noqa: F401
        check("analysis packages import", True,
              f"pandas {pandas.__version__}, numpy {numpy.__version__}")
    except ImportError as exc:
        check("analysis packages import", False, str(exc))

    try:
        import cqhandson
        from cqhandson import paths, runner
        check("cqhandson imports", True)
    except Exception as exc:  # noqa: BLE001
        check("cqhandson imports", False, str(exc))
        print("\nStopping: the helper package could not be imported.")
        return 1

    try:
        versions = runner.check_analyzers()
        check("pylint on PATH", True, versions["pylint"])
        check("semgrep on PATH", True, versions["semgrep"])
        check("pylint version is 3.3.6", "3.3.6" in versions["pylint"],
              "a different version changes the defect counts", fatal=False)
        check("semgrep version is 1.120.0", "1.120.0" in versions["semgrep"],
              "a different version changes the vulnerability counts", fatal=False)
    except RuntimeError as exc:
        check("analyzers on PATH", False, str(exc))

    for label, path in [
        ("tasks.jsonl", paths.TASKS),
        ("references.jsonl", paths.REFERENCES),
        ("frozen semgrep rules", paths.SEMGREP_RULES),
        ("pylint -> ODC mapping", paths.PYLINT_ODC_MAP),
    ]:
        check(f"data: {label}", path.exists(), str(path.relative_to(REPO)))

    missing = [a for a in cqhandson.AUTHOR_ORDER
               if not (paths.PREDICTIONS / f"{a}.jsonl").exists()]
    check("data: 5 prediction files", not missing, "missing: " + ", ".join(missing) if missing else "")

    precomputed = sorted(paths.PRECOMPUTED.glob("*.jsonl"))
    check("fallback results present", len(precomputed) == 5,
          f"{len(precomputed)} files", fatal=False)

    if not problems:
        print("-" * 52)
        print("Scoring 8 tasks end to end (this exercises both analyzers) ...")
        started = time.time()
        try:
            rows = runner.evaluate(
                REPO / "data/reference_check/human.jsonl",
                tasks=REPO / "data/reference_check/tasks.jsonl",
                references=REPO / "data/reference_check/references.jsonl",
                verbose=False,
            )
            elapsed = time.time() - started
            check("end-to-end scoring", len(rows) == 8, f"{elapsed:.1f}s for 8 tasks")
            print("\nAlmost all of that is one-off analyzer start-up (semgrep parses")
            print("1,847 rules once per run), so the cost barely grows with the task")
            print("count: a full 200-task author took 13-23 s on the reference machine,")
            print("about 2 minutes for all five authors. Notebook 02 does the real run.")
        except Exception as exc:  # noqa: BLE001
            check("end-to-end scoring", False, str(exc))

    print("-" * 52)
    if problems:
        print("NOT READY. Fix:")
        for item in problems:
            print(f"  - {item}")
        print("\nMost failures are a missing `conda activate cqbench-handson`.")
        return 1
    print("Ready. Open notebooks/00_setup.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
