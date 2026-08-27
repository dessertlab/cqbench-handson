"""A batched CQBench evaluator that finishes inside a coffee break.

Why this file exists
--------------------
`python -m cqbench evaluate` scores one task at a time. For every task it starts
a fresh `semgrep` process, which parses all 1,847 frozen rules from scratch
(~6 s) and — unless told not to — makes a network round trip to check for a new
release (~25 s). Add a fresh `pylint` process per task and the measured cost on
a 2-core machine is **~33 s per task**. Scoring 200 tasks for 5 authors that way
takes about nine hours.

This module produces *identical rows* (verified field by field against the
reference evaluator, see notebook 02) by moving the process boundary:

===========================  ============================  =================
step                         stock evaluator               here
===========================  ============================  =================
semgrep                      1 process per task            1 process per author
pylint                       1 process per task            1 process per 200 files
lizard / tree-sitter         serial                        process pool
semgrep version check        ~25 s per task                disabled
===========================  ============================  =================

Two correctness notes:

* Batching pylint is only sound with ``--disable=duplicate-code``. R0801 is the
  one check that compares *across* modules; it can never fire when files are
  linted one at a time, so disabling it in a batch reproduces per-file output
  exactly. Every other Pylint message is file-local.
* A semgrep run over a directory reports the originating ``path`` on every
  finding *and* every error, so per-file attribution — including the study's
  rule that a file which produced an error contributes zero findings — survives
  batching unchanged.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import paths

sys.path.insert(0, str(paths.VENDOR)) if str(paths.VENDOR) not in sys.path else None

import pandas as pd  # noqa: E402

from cqbench.analyzers import _empty_defects, _odc_column  # noqa: E402
from cqbench.config import BENCHMARK_VERSION  # noqa: E402
from cqbench.evaluate import _complexity, _complexity_gate, _lexical_tokens  # noqa: E402
from cqbench.io import read_jsonl, write_jsonl_atomic  # noqa: E402
from cqbench.structural import Signature, analyze_structure  # noqa: E402
from support.rq4_build_table import PYLINT_EXCLUDED_SYMBOLS, normalized_cwes  # noqa: E402

PYLINT_CHUNK = 200

_SCRATCH = Path(tempfile.gettempdir())
_ANALYZER_ENV = {
    # Keep analyzer state out of the user's home directory, so a shared machine
    # or a read-only home does not change what the benchmark measures ...
    "XDG_CONFIG_HOME": str(_SCRATCH / "cqbench-xdg-config"),
    "XDG_CACHE_HOME": str(_SCRATCH / "cqbench-xdg-cache"),
    "SEMGREP_SETTINGS_FILE": str(_SCRATCH / "cqbench-semgrep-settings.yml"),
    "SEMGREP_LOG_FILE": str(_SCRATCH / "cqbench-semgrep.log"),
    # ... and stop semgrep phoning home, which costs ~25 s per invocation.
    "SEMGREP_ENABLE_VERSION_CHECK": "0",
}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(_ANALYZER_ENV)
    return env


def check_analyzers() -> dict[str, str]:
    """Report the versions of the two external analyzers, or raise if missing."""
    found = {}
    for tool, args in (("pylint", ["--version"]), ("semgrep", ["--version"])):
        try:
            out = subprocess.run([tool, *args], capture_output=True, text=True,
                                 env=_env(), timeout=120)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{tool} is not on PATH. Activate the conda environment: "
                f"`conda activate cqbench-handson`."
            ) from exc
        found[tool] = (out.stdout or out.stderr).strip().splitlines()[0]
    return found


# --------------------------------------------------------------------------- #
# stage 1: write every submission to a temporary corpus directory
# --------------------------------------------------------------------------- #
def _write_corpus(codes: dict[str, str], directory: Path) -> dict[str, str]:
    names = {}
    for task_id, code in codes.items():
        filename = task_id.replace(":", "__").replace("/", "_") + ".py"
        (directory / filename).write_text(code, encoding="utf-8")
        names[filename] = task_id
    return names


# --------------------------------------------------------------------------- #
# stage 2: security findings — one semgrep process for the whole corpus
# --------------------------------------------------------------------------- #
def semgrep_corpus(directory: Path, names: dict[str, str], jobs: int) -> dict[str, dict]:
    proc = subprocess.run(
        ["semgrep", "scan",
         "--config", str(paths.SEMGREP_RULES),
         "--json", "--metrics", "off", "--no-git-ignore", "--disable-version-check",
         "--max-target-bytes", "1000000", "-j", str(jobs), str(directory)],
        capture_output=True, text=True, env=_env(), stdin=subprocess.DEVNULL,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(f"semgrep returned invalid JSON:\n{proc.stderr[-2000:]}") from exc

    findings: dict[str, dict] = collections.defaultdict(dict)
    for finding in report.get("results", []):
        extra = finding.get("extra", {})
        cwes = extra.get("metadata", {}).get("cwe")
        if not cwes:                       # no CWE metadata -> not a vulnerability finding
            continue
        key = (normalized_cwes(cwes),
               str(extra.get("severity", "")).upper(),
               str(extra.get("lines", "")).strip())
        findings[os.path.basename(finding["path"])].setdefault(key, finding)

    errors: dict[str, list] = collections.defaultdict(list)
    for error in report.get("errors", []):
        path = str(error.get("path", ""))
        if path.startswith("https:/semgrep.dev/..."):
            continue                        # known noise entry, not a scan error
        errors[os.path.basename(path)].append(error)

    out = {}
    for filename, task_id in names.items():
        unique = findings.get(filename, {})
        file_errors = errors.get(filename, [])
        if file_errors:                     # study convention: an errored file scores zero
            unique = {}
        out[task_id] = {
            "vulns_total": len(unique),
            "vulns_high_sev": sum(k[1] in {"CRITICAL", "ERROR"} for k in unique),
            "cwes": sorted({cwe for key in unique for cwe in key[0]}),
            "semgrep_error": bool(file_errors),
            "semgrep_errors": file_errors,
            "vulnerability_findings": list(unique.values()),
        }
    return out


# --------------------------------------------------------------------------- #
# stage 3: defects — pylint in chunks, mapped to ODC categories
# --------------------------------------------------------------------------- #
def pylint_corpus(directory: Path, names: dict[str, str], jobs: int) -> dict[str, dict]:
    mapping = pd.read_excel(paths.PYLINT_ODC_MAP, engine="openpyxl")
    odc_of = dict(zip(mapping["symbol"], mapping["odc_category"]))

    files = sorted(directory.glob("*.py"))
    messages: list[dict] = []
    for start in range(0, len(files), PYLINT_CHUNK):
        chunk = files[start:start + PYLINT_CHUNK]
        proc = subprocess.run(
            ["pylint", *map(str, chunk), "--output-format=json", "--score=no",
             f"-j={jobs}", "--disable=duplicate-code"],
            capture_output=True, text=True, env=_env(), stdin=subprocess.DEVNULL,
        )
        messages.extend(json.loads(proc.stdout or "[]"))

    per_file: dict[str, dict] = collections.defaultdict(dict)
    for message in messages:
        symbol = message.get("symbol")
        odc = odc_of.get(symbol, "--")
        # Two filters, both from the study: an explicit exclusion list of noisy
        # or environment-dependent symbols, and "no ODC category -> not a defect".
        if symbol in PYLINT_EXCLUDED_SYMBOLS or odc == "--":
            continue
        message = dict(message)
        message["odc_category"] = odc
        # De-duplicate on (symbol, category, line): one defect per site.
        per_file[os.path.basename(message["path"])].setdefault(
            (symbol, odc, message.get("line")), message)

    out = {}
    for filename, task_id in names.items():
        unique = per_file.get(filename, {})
        row = _empty_defects()
        for _, odc, _ in unique:
            row[_odc_column(str(odc))] += 1
        row["defects_total"] = len(unique)
        row["defect_findings"] = list(unique.values())
        out[task_id] = row
    return out


# --------------------------------------------------------------------------- #
# stage 4: structure + complexity (pure Python, parallelised)
# --------------------------------------------------------------------------- #
def _structural(payload):
    task_id, code, language, signature, human_metrics, human_complexity = payload
    structure = analyze_structure(
        code, language, Signature(**signature),
        human_token_count=int(human_metrics["token_count"]),
        human_ast_count=int(human_metrics["ast_node_count"]),
    )
    complexity = _complexity(code, language, task_id)
    gate = _complexity_gate(complexity, human_complexity)
    return task_id, structure.to_dict(), structure.strict_nontrivial, complexity, gate


# --------------------------------------------------------------------------- #
# the whole thing
# --------------------------------------------------------------------------- #
def evaluate(
    predictions,
    tasks=None,
    references=None,
    output=None,
    *,
    jobs: int | None = None,
    overwrite: bool = True,
    verbose: bool = True,
):
    """Score one author's submission file. Returns the list of result rows.

    Parameters
    ----------
    predictions : path to a JSONL of ``{"task_id": ..., "code": ...}``
    tasks, references : default to the hands-on subset in ``data/``
    output : where to write the result JSONL (optional)
    jobs : worker count; defaults to the machine's CPU count
    """
    jobs = jobs or (os.cpu_count() or 4)
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    started = time.time()

    task_rows = {r["task_id"]: r for r in read_jsonl(Path(tasks or paths.TASKS))}
    refs = {r["task_id"]: r for r in read_jsonl(Path(references or paths.REFERENCES))}
    preds = {r["task_id"]: r["code"] for r in read_jsonl(Path(predictions))}

    unknown = set(preds) - set(task_rows)
    assert not unknown, f"prediction file contains unknown task_ids: {sorted(unknown)[:5]}"
    # Missing predictions stay in the denominator and are scored as empty output.
    codes = {task_id: preds.get(task_id, "") for task_id in task_rows}

    with tempfile.TemporaryDirectory(prefix="cqhandson-") as tmp:
        corpus = Path(tmp)
        names = _write_corpus(codes, corpus)

        mark = time.time()
        security = semgrep_corpus(corpus, names, jobs)
        say(f"  semgrep    {len(names):4d} files   {time.time() - mark:6.1f}s")

        mark = time.time()
        defects = pylint_corpus(corpus, names, jobs)
        say(f"  pylint     {len(names):4d} files   {time.time() - mark:6.1f}s")

    mark = time.time()
    payloads = [(task_id, codes[task_id], task["language"], task["signature"],
                 refs[task_id]["human_metrics"], refs[task_id].get("human_complexity"))
                for task_id, task in task_rows.items()]
    structural = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for record in pool.map(_structural, payloads, chunksize=8):
            structural[record[0]] = record
    say(f"  structure  {len(names):4d} files   {time.time() - mark:6.1f}s")

    rows = []
    for task_id, task in task_rows.items():
        _, structure, strict, complexity, gate = structural[task_id]
        structure = dict(structure)
        structure["structural_strict_nontrivial"] = strict
        # A structurally fine answer that is 10x smaller than the human reference
        # is not credited: it is "degenerate", not "clean".
        structure["strict_nontrivial"] = bool(strict and gate["complexity_non_degenerate"])
        if strict and not gate["complexity_non_degenerate"]:
            structure["status"] = "complexity_degenerate"
        row = {
            "benchmark_version": BENCHMARK_VERSION,
            "task_id": task_id,
            "language": task["language"],
            "stratum": task["stratum"],
            "submitted": task_id in preds,
            **structure, **gate,
            "complexity": complexity,
            "lexical_tokens": _lexical_tokens(codes[task_id]),
        }
        row.update(defects[task_id])
        row.update(security[task_id])
        row["static_analysis_status"] = "ok"
        rows.append(row)

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(output, rows, overwrite=overwrite)
    say(f"  ------------------------------------  {time.time() - started:6.1f}s total")
    return rows


def evaluate_all(
    authors,
    predictions_dir=None,
    output_dir=None,
    *,
    jobs: int | None = None,
    verbose: bool = True,
):
    """Score several authors on the same tasks. Returns {author: n_rows}."""
    predictions_dir = Path(predictions_dir or paths.PREDICTIONS)
    output_dir = Path(output_dir or paths.LIVE)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    started = time.time()
    for author in authors:
        if verbose:
            print(f"[{author}]")
        rows = evaluate(predictions_dir / f"{author}.jsonl",
                        output=output_dir / f"{author}.jsonl",
                        jobs=jobs, verbose=verbose)
        counts[author] = len(rows)
    if verbose:
        print(f"\nDone: {sum(counts.values())} evaluations "
              f"in {time.time() - started:.0f}s -> {output_dir}")
    return counts


# --------------------------------------------------------------------------- #
# equivalence check against the reference evaluator
# --------------------------------------------------------------------------- #
#: Fields holding raw analyzer payloads. They carry absolute temp-file paths and
#: so differ between runs by construction; every *scored* field is compared.
_PAYLOAD_FIELDS = {"defect_findings", "vulnerability_findings", "semgrep_errors"}


def diff_results(rows_a, rows_b, *, ignore=_PAYLOAD_FIELDS) -> list[dict]:
    """Field-by-field comparison of two result sets. Empty list means identical.

    Both sides are passed through a JSON round trip first, so that a tuple held
    in memory and the list it was serialised as do not read as a difference.
    """
    def canonical(rows):
        return [json.loads(json.dumps(row, default=list)) for row in rows]

    rows_a, rows_b = canonical(rows_a), canonical(rows_b)
    index_b = {row["task_id"]: row for row in rows_b}
    differences = []
    for row in rows_a:
        other = index_b.get(row["task_id"])
        if other is None:
            differences.append({"task_id": row["task_id"], "field": "<missing>",
                                "a": "present", "b": None})
            continue
        for field, value in row.items():
            if field in ignore:
                continue
            if other.get(field) != value:
                differences.append({"task_id": row["task_id"], "field": field,
                                    "a": value, "b": other.get(field)})
    return differences
