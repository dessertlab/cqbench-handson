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

    # Every third-party module the session needs, with the pip spec that provides
    # it. Checked in one pass so a missing package is reported with its fix
    # instead of surfacing later as an opaque ImportError inside a notebook.
    REQUIRED = [
        ("pandas", "pandas>=2.0,<3"),
        ("numpy", "numpy>=1.24,<3"),
        ("scipy", "scipy>=1.10,<2"),
        ("matplotlib", "matplotlib>=3.7,<4"),
        ("lizard", "lizard==1.17.25"),
        ("openpyxl", "openpyxl==3.1.5"),
        ("tqdm", "tqdm>=4.65,<5"),
        ("pyarrow", "pyarrow>=15"),
        ("tree_sitter", "tree-sitter==0.23.2"),
        ("tree_sitter_language_pack", "tree-sitter-language-pack>=0.7,<1"),
    ]
    import importlib

    missing = []
    for module, spec in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(spec)

    if missing:
        check("required packages", False, f"{len(missing)} missing")
        print("\n" + "-" * 52)
        print("Missing packages. Install them into THIS interpreter with:\n")
        print(f"    {Path(sys.executable).name} -m pip install " +
              " ".join(f'"{spec}"' for spec in missing))
        print(f"\n(interpreter: {sys.executable})")
        print("\nUse `python -m pip`, not a bare `pip`: a bare `pip` can belong")
        print("to a different interpreter, in which case the install succeeds and")
        print("the import still fails.")
        print("-" * 52)
        return 1

    import numpy, pandas  # noqa: E402
    check("required packages", True,
          f"all {len(REQUIRED)} present · pandas {pandas.__version__}, numpy {numpy.__version__}")

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
        if "pkg_resources" in str(exc):
            print("\n" + "-" * 52)
            print("This one has a known cause and a one-line fix:\n")
            print(f"    {Path(sys.executable).name} -m pip install \"setuptools<82\"")
            print("\nsemgrep reaches pkg_resources through opentelemetry. "
                  "pkg_resources\nships inside setuptools, which conda does not "
                  "install by default, and\nwhich removed pkg_resources in "
                  "version 82 -- so plain `pip install\nsetuptools` does not fix "
                  "it. The pin is what matters.")
            print("-" * 52)

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
