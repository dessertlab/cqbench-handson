from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ODC_COLUMNS, ODC_LABELS, ROOT
from .legacy import rq4_module


class AnalyzerUnavailable(RuntimeError):
    pass


def _run(
    command: list[str],
    *,
    timeout: int = 120,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
):
    executable = shutil.which(command[0])
    if executable is None:
        raise AnalyzerUnavailable(f"Required executable is unavailable: {command[0]}")
    environment = dict(os.environ)
    environment["XDG_CONFIG_HOME"] = "/tmp/cqbench-xdg-config"
    environment["XDG_CACHE_HOME"] = "/tmp/cqbench-xdg-cache"
    environment["SEMGREP_SETTINGS_FILE"] = "/tmp/cqbench-semgrep-settings.yml"
    environment["SEMGREP_LOG_FILE"] = "/tmp/cqbench-semgrep.log"
    if env:
        environment.update(env)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return result


def _empty_defects() -> dict[str, Any]:
    return {
        "defects_total": 0,
        **{column: 0 for column in ODC_COLUMNS},
        "defect_findings": [],
    }


def _odc_column(label: str) -> str | None:
    module = rq4_module()
    return module.ODC_TO_COLUMN.get(label)


def analyze_pylint(code: str) -> dict[str, Any]:
    mapping = pd.read_excel(
        ROOT / "mappings/python/pylint_odc.xlsx", engine="openpyxl"
    )
    odc_map = dict(zip(mapping["symbol"], mapping["odc_category"]))
    module = rq4_module()
    with tempfile.TemporaryDirectory(prefix="cqbench-pylint-") as directory:
        path = Path(directory) / "submission.py"
        path.write_text(code, encoding="utf-8")
        result = _run(
            ["pylint", str(path), "--output-format=json", "--score=no", "-j=1"],
            timeout=30,
        )
        try:
            messages = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Pylint returned invalid JSON: {result.stderr}") from exc
    unique = {}
    for message in messages:
        symbol = message.get("symbol")
        odc = odc_map.get(symbol, "--")
        if symbol in module.PYLINT_EXCLUDED_SYMBOLS or odc == "--":
            continue
        column = _odc_column(str(odc))
        if column is None:
            raise AssertionError(f"Unknown Pylint ODC category: {odc}")
        message["odc_category"] = odc
        unique.setdefault((symbol, odc, message.get("line")), message)
    output = _empty_defects()
    for _, odc, _ in unique:
        output[_odc_column(str(odc))] += 1
    output["defects_total"] = len(unique)
    output["defect_findings"] = list(unique.values())
    return output


def _wrap_java(code: str) -> str:
    """Make a bare Java method parseable, mirroring the study's
    ``wrap_java_functions.py``:

    - lift ``package``/``import`` statements out of the body and re-attach them
      outside the synthetic class (otherwise they land inside it and the file
      fails to parse);
    - wrap when the snippet has orphan (top-level) methods OR has no top-level
      ``public class`` — even if some other type is present;
    - keep the code unchanged when it already declares a top-level public class
      and has no orphan methods.
    """
    module = rq4_module()
    package_match = re.search(r"^\s*package\s+[^\n;]+;", code, flags=re.M)
    package_stmt = package_match.group(0).strip() if package_match else ""
    code_no_pkg = re.sub(r"^\s*package\s+[^\n;]+;\n?", "", code, flags=re.M)
    imports = re.findall(r"^\s*import\s+[^\n;]+;", code_no_pkg, flags=re.M)
    imports_stmt = "\n".join(imports)
    cleaned = re.sub(r"^\s*import\s+[^\n;]+;\n?", "", code_no_pkg, flags=re.M).strip()

    top_level_public_class = re.search(r"^\s*public\s+class\s+\w+", cleaned, flags=re.M)
    should_wrap = module.java_has_orphan_methods(cleaned) or not top_level_public_class
    if not should_wrap and module.java_top_level_type(cleaned):
        body = cleaned
    else:
        indented = "\n".join("    " + line for line in cleaned.splitlines())
        body = f"public class CQBenchSubmission {{\n{indented}\n}}"
    parts = [part for part in (package_stmt, imports_stmt, body) if part]
    return "\n\n".join(parts).strip() + "\n"


def analyze_pmd(code: str) -> dict[str, Any]:
    mapping = pd.read_excel(ROOT / "mappings/java/pmd_odc.xlsx", engine="openpyxl")
    odc_map = dict(zip(mapping["Difetto PMD"], mapping["Classificazione ODC"]))
    module = rq4_module()
    with tempfile.TemporaryDirectory(prefix="cqbench-pmd-") as directory:
        path = Path(directory) / "CQBenchSubmission.java"
        report = Path(directory) / "report.json"
        path.write_text(_wrap_java(code), encoding="utf-8")
        rulesets = ",".join(
            f"category/java/{name}.xml"
            for name in ("bestpractices", "design", "errorprone", "multithreading", "performance")
        )
        result = _run(
            [
                "pmd",
                "check",
                "-d",
                str(path),
                "-R",
                rulesets,
                "-f",
                "json",
                "-r",
                str(report),
            ],
            timeout=120,
            # Matches run_PMD_analysis_NEW.sh: report violations for the
            # parseable regions of a partially-broken file.
            env={"PMD_JAVA_OPTS": "-Dpmd.error_recovery"},
        )
        if not report.exists():
            raise RuntimeError(f"PMD did not create a report: {result.stderr}")
        data = json.loads(report.read_text(encoding="utf-8"))
    unique = {}
    for file_entry in data.get("files", []):
        for violation in file_entry.get("violations", []):
            rule = violation.get("rule")
            odc = odc_map.get(rule, "--")
            if rule in module.PMD_EXCLUDED_RULES or odc == "--":
                continue
            column = _odc_column(str(odc))
            if column is None:
                raise AssertionError(f"Unknown PMD ODC category: {odc}")
            finding = dict(violation)
            finding["odc_category"] = odc
            unique.setdefault((rule, odc, violation.get("beginline")), finding)
    output = _empty_defects()
    for _, odc, _ in unique:
        output[_odc_column(str(odc))] += 1
    output["defects_total"] = len(unique)
    output["defect_findings"] = list(unique.values())
    return output


CLANG_LINE_RE = re.compile(
    r"^(?P<path>.*?):(?P<line>\d+):(?P<column>\d+):\s+"
    r"(?P<severity>\w+):\s+(?P<message>.*?)\s+\[(?P<check>[^\]]+)\]\s*$"
)


def analyze_clang(code: str) -> dict[str, Any]:
    mapping = pd.read_excel(
        ROOT / "mappings/c/clang_tidy_odc.xlsx", engine="openpyxl"
    )
    odc_map = {
        str(check).strip(): "--" if pd.isna(odc) else str(odc).strip()
        for check, odc in zip(mapping["Clang-tidy Check"], mapping["ODC Defect Type"])
        if not pd.isna(check)
    }
    module = rq4_module()
    with tempfile.TemporaryDirectory(prefix="cqbench-clang-") as directory:
        path = Path(directory) / "submission.c"
        path.write_text(code, encoding="utf-8")
        # Matches the study runner (C_defects/clang_ODC.py): enable ALL checks
        # with the compiler-default dialect (no -std, no compilation database).
        # Restricting to an allowlist and forcing -std=c17 shifts alias
        # attribution and parsing, so findings diverge from the C baselines.
        result = _run(
            [
                "clang-tidy",
                str(path),
                "--checks=*",
                "--extra-arg=-ferror-limit=0",
            ],
            timeout=120,
        )
    unique = {}
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        match = CLANG_LINE_RE.match(line)
        if not match:
            continue
        finding = match.groupdict()
        # clang-tidy can report a diagnostic under a comma-separated alias chain
        # (e.g. "[a,b]"); the study's converter maps the first alias that has an
        # ODC mapping. Reproduce that here rather than dropping the whole chain.
        check = odc = None
        for candidate in (part.strip() for part in finding["check"].split(",")):
            candidate_odc = odc_map.get(candidate, "--")
            if candidate_odc != "--":
                check, odc = candidate, candidate_odc
                break
        if check is None:
            continue
        if (
            module.clang_symbol_excluded(check)
            or odc in module.C_CLANG_EXCLUDED_ODC
        ):
            continue
        column = _odc_column(odc)
        if column is None:
            raise AssertionError(f"Unknown Clang ODC category: {odc}")
        finding["check"] = check
        finding["odc_category"] = odc
        finding["line"] = int(finding["line"])
        unique.setdefault((check, odc, finding["line"]), finding)
    output = _empty_defects()
    for _, odc, _ in unique:
        output[_odc_column(str(odc))] += 1
    output["defects_total"] = len(unique)
    output["defect_findings"] = list(unique.values())
    return output


def analyze_defects(language: str, code: str) -> dict[str, Any]:
    return {
        "python": analyze_pylint,
        "java": analyze_pmd,
        "c": analyze_clang,
    }[language](code)


def analyze_semgrep(code: str, language: str, rules: Path) -> dict[str, Any]:
    if not rules.is_file():
        raise AnalyzerUnavailable(f"Vendored Semgrep rules are missing: {rules}")
    suffix = {"python": ".py", "java": ".java", "c": ".c"}[language]
    # The study scanned pre-wrapped Java (wrap_java_functions.py); mirror that so
    # bare methods parse and match identically.
    scanned_code = _wrap_java(code) if language == "java" else code
    with tempfile.TemporaryDirectory(prefix="cqbench-semgrep-") as directory:
        target = Path(directory) / f"submission{suffix}"
        target.write_text(scanned_code, encoding="utf-8")
        result = _run(
            [
                "semgrep",
                "scan",
                "--config",
                str(rules),
                "--json",
                "--metrics",
                "off",
                "--no-git-ignore",
                # run_semgrep_*.py skipped files larger than 1 MB.
                "--max-target-bytes",
                "1000000",
                str(target),
            ],
            timeout=180,
        )
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Semgrep returned invalid JSON: {result.stderr}") from exc
    module = rq4_module()
    unique = {}
    for finding in report.get("results", []):
        extra = finding.get("extra", {})
        cwes = extra.get("metadata", {}).get("cwe")
        if not cwes:
            continue
        key = (
            module.normalized_cwes(cwes),
            str(extra.get("severity", "")).upper(),
            str(extra.get("lines", "")).strip(),
        )
        unique.setdefault(key, finding)
    errors = [
        error
        for error in report.get("errors", [])
        if not str(error.get("path", "")).startswith("https:/semgrep.dev/...")
    ]
    # process_semgrep_results_*.py discards every finding from a sample that
    # produced a (non-noise) Semgrep error; such a sample contributes zero.
    if errors:
        unique = {}
    cwes = sorted({cwe for key in unique for cwe in key[0]})
    high = sum(key[1] in {"CRITICAL", "ERROR"} for key in unique)
    return {
        "vulns_total": len(unique),
        "vulns_high_sev": high,
        "cwes": cwes,
        "semgrep_error": bool(errors),
        "semgrep_errors": errors,
        "vulnerability_findings": list(unique.values()),
    }
