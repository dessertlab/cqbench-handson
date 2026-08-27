#!/usr/bin/env python3
"""Build an asserted, per-sample RQ4 analysis table.

The implementation is intentionally key driven. Dataset positions are used only
to resolve source-native IDs (KenLM's zero-based sample_idx and Semgrep's
one-based filename suffix) to hm_index. Every subsequent merge uses hm_index.

Structural predictors are the means across every computable function detected
in a sample. ``sample_nloc`` is a separate physical nonblank, non-comment line
count for the complete code sample and is used only as the size control in
Stage 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MEAN_COMPLEXITY_PREDICTORS = (
    "nloc",
    "ccn",
    "parameter_count",
    "max_nesting_depth",
    "distinct_operators",
    "distinct_operands",
    "total_operators",
    "total_operands",
    "halstead_v",
    "halstead_difficulty",
    "halstead_effort",
    "mi",
)
COMPLEXITY_PREDICTORS = MEAN_COMPLEXITY_PREDICTORS
PREDICTORS = (*COMPLEXITY_PREDICTORS, "naturalness")
ODC_OUTPUTS = (
    "def_assignment",
    "def_algorithm",
    "def_interface",
    "def_checking",
    "def_timing",
    "def_function_class_object",
)
OUTCOMES = ("defects_total", *ODC_OUTPUTS, "vulns_total", "vulns_high_sev")

PYLINT_EXCLUDED_SYMBOLS = {
    "bad-indentation",
    "missing-module-docstring",
    "missing-function-docstring",
    "missing-final-newline",
    "bad-docstring-quotes",
    "consider-using-f-string",
    "undefined-variable",
    "import-error",
    "invalid-name",
    "redundant-u-string-prefix",
    "multiple-statements",
    "pointless-string-statement",
    "unnecessary-comprehension",
    "no-member",
    "syntax-error",
    "protected-access",
    "unnecessary-semicolon",
    "no-else-return",
}

PMD_EXCLUDED_RULES = {
    "AvoidDuplicateLiterals",
    "UseLocaleWithCaseConversions",
    "AbstractClassWithoutAbstractMethod",
    "AccessorClassGeneration",
    "AbstractClassWithoutAnyMethod",
    "ClassWithOnlyPrivateConstructorsShouldBeFinal",
    "DataClass",
    "GodClass",
    "CloneMethodReturnTypeMustMatchClassName",
    "MethodWithSameNameAsEnclosingClass",
    "MissingStaticMethodInNonInstantiatableClass",
    "UseUtilityClass",
    "LawOfDemeter",
    "UnusedPrivateMethod",
    "AvoidLiteralsInIfCondition",
}

C_CLANG_EXCLUDED_SYMBOLS = {
    "clang-diagnostic-error",
    "cppcoreguidelines-init-variables",
    "misc-include-cleaner",
    "google-readability-casting",
    "fuchsia-default-arguments-declarations",
    "clang-analyzer-cplusplus.ArrayDelete",
    "clang-analyzer-cplusplus.InnerPointer",
    "clang-analyzer-cplusplus.Move",
    "clang-analyzer-cplusplus.NewDelete",
    "clang-analyzer-cplusplus.NewDeleteLeaks",
    "clang-analyzer-cplusplus.PlacementNew",
    "clang-analyzer-cplusplus.SelfAssignment",
    "clang-analyzer-cplusplus.StringChecker",
    "clang-analyzer-deadcode.DeadStores",
    "clang-analyzer-fuchsia.HandleChecker",
    "clang-analyzer-optin.core.EnumCastOutOfRange",
    "clang-analyzer-optin.cplusplus.UninitializedObject",
    "clang-analyzer-optin.cplusplus.VirtualCall",
    "clang-analyzer-optin.mpi.MPI-Checker",
    "clang-analyzer-optin.osx.cocoa.localizability.EmptyLocalizationContextChecker",
    "clang-analyzer-optin.osx.cocoa.localizability.NonLocalizedStringChecker",
    "clang-analyzer-optin.performance.GCDAntipattern",
    "clang-analyzer-optin.performance.Padding",
    "clang-analyzer-optin.portability.UnixAPI",
    "clang-analyzer-optin.taint.TaintedAlloc",
    "clang-analyzer-osx.API",
    "clang-analyzer-osx.NumberObjectConversion",
    "clang-analyzer-osx.ObjCProperty",
    "clang-analyzer-osx.SecKeychainAPI",
    "clang-analyzer-osx.cocoa.AtSync",
    "clang-analyzer-osx.cocoa.AutoreleaseWrite",
    "clang-analyzer-osx.cocoa.ClassRelease",
    "clang-analyzer-osx.cocoa.Dealloc",
    "clang-analyzer-osx.cocoa.IncompatibleMethodTypes",
    "clang-analyzer-osx.cocoa.Loops",
    "clang-analyzer-osx.cocoa.MissingSuperCall",
    "clang-analyzer-osx.cocoa.NSAutoreleasePool",
    "clang-analyzer-osx.cocoa.NSError",
    "clang-analyzer-osx.cocoa.NilArg",
    "clang-analyzer-osx.cocoa.NonNilReturnValue",
    "clang-analyzer-osx.cocoa.ObjCGenerics",
    "clang-analyzer-osx.cocoa.RetainCount",
    "clang-analyzer-osx.cocoa.RunLoopAutoreleaseLeak",
    "clang-analyzer-osx.cocoa.SelfInit",
    "clang-analyzer-osx.cocoa.SuperDealloc",
    "clang-analyzer-osx.cocoa.UnusedIvars",
    "clang-analyzer-osx.cocoa.VariadicMethodTypes",
    "clang-analyzer-osx.coreFoundation.CFError",
    "clang-analyzer-osx.coreFoundation.CFNumber",
    "clang-analyzer-osx.coreFoundation.CFRetainRelease",
    "clang-analyzer-osx.coreFoundation.containers.OutOfBounds",
    "clang-analyzer-osx.coreFoundation.containers.PointerSizedValues",
    "clang-analyzer-security.FloatLoopCounter",
    "clang-analyzer-security.PutenvStackArray",
    "clang-analyzer-security.SetgidSetuidOrder",
    "clang-analyzer-security.cert.env.InvalidPtr",
    "clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling",
    "clang-analyzer-security.insecureAPI.UncheckedReturn",
    "clang-analyzer-security.insecureAPI.bcmp",
    "clang-analyzer-security.insecureAPI.bcopy",
    "clang-analyzer-security.insecureAPI.bzero",
    "clang-analyzer-security.insecureAPI.decodeValueOfObjCType",
    "clang-analyzer-security.insecureAPI.getpw",
    "clang-analyzer-security.insecureAPI.gets",
    "clang-analyzer-security.insecureAPI.mkstemp",
    "clang-analyzer-security.insecureAPI.mktemp",
    "clang-analyzer-security.insecureAPI.rand",
    "clang-analyzer-security.insecureAPI.strcpy",
    "clang-analyzer-security.insecureAPI.vfork",
    "clang-analyzer-unix.API",
    "clang-analyzer-unix.BlockInCriticalSection",
    "clang-analyzer-unix.Errno",
    "clang-analyzer-unix.Malloc",
    "clang-analyzer-unix.MallocSizeof",
    "clang-analyzer-unix.MismatchedDeallocator",
    "clang-analyzer-unix.StdCLibraryFunctions",
    "clang-analyzer-unix.Stream",
    "clang-analyzer-unix.Vfork",
    "clang-analyzer-unix.cstring.BadSizeArg",
    "clang-analyzer-unix.cstring.NullArg",
    "clang-analyzer-webkit.NoUncountedMemberChecker",
    "clang-analyzer-webkit.RefCntblBaseVirtualDtor",
    "clang-analyzer-webkit.UncountedLambdaCapturesChecker",
    "cppcoreguidelines-macro-to-enum",
    "modernize-macro-to-enum",
}
C_CLANG_EXCLUDED_PREFIXES = ("readability-", "altera-", "android-cloexec-")
C_CLANG_EXCLUDED_ODC = {"Documentation", "Build/Package/Merg"}

JAVA_PMD_DSC_FILENAME_OVERRIDES = {
    "TempClass126235.java": 126235,
    "TempClass146601.java": 146601,
    "TempClass16773.java": 16773,
    "TempClass192369.java": 192369,
    "TempClass199042.java": 199042,
    "TempClass216892.java": 216892,
    "TempClass35001.java": 35001,
    "TempClass39336.java": 39336,
    "TempClass80092.java": 80092,
    # Identified from the unique class declaration and the PMD violation line.
    "DatabaseHelper_2.java": 119922,
    "VersionProvider.java": 185887,
}

ODC_TO_COLUMN = {
    "Assignment": "def_assignment",
    "Algorithm": "def_algorithm",
    "Algorithm/Method": "def_algorithm",
    "Interface": "def_interface",
    "Checking": "def_checking",
    "Timing": "def_timing",
    "Timing/Serialization": "def_timing",
    "Function/Class/Object": "def_function_class_object",
}

PATH_SAMPLE_RE = re.compile(r"_(\d+)\.(?:py|java|c)$")
CWE_RE = re.compile(r"(CWE-\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    dataset_path: Path
    dataset_key: str
    metrics_path: Path
    authors: Tuple[str, ...]
    author_fields: Mapping[str, str]
    defect_kind: str
    defect_paths: Mapping[str, Path]
    odc_mapping_path: Path | None
    semgrep_dirs: Mapping[str, Path]
    semgrep_filename_mode: str
    semgrep_requires_full_dataset: bool
    entropy_paths: Mapping[str, Path]
    entropy_language: str


PYTHON_CONFIG = LanguageConfig(
    name="python",
    dataset_path=ROOT / "datasets/final_datasets/python_dataset_nodocs_dsc_qwen_FINAL.jsonl",
    dataset_key="hm_index",
    metrics_path=ROOT / "python_all_function_metrics_aggregates.jsonl",
    authors=("human", "chatgpt", "dsc", "qwen"),
    author_fields={
        "human": "human_code",
        "chatgpt": "chatgpt_code",
        "dsc": "dsc_code",
        "qwen": "qwen_code",
    },
    defect_kind="pylint",
    defect_paths={
        "human": ROOT / "risultati_python/report_pylint/pylint_output_human_with_odc.jsonl",
        "chatgpt": ROOT / "risultati_python/report_pylint/pylint_output_chatgpt_with_odc.jsonl",
        "dsc": ROOT / "risultati_python/report_pylint/pylint_output_DSC_with_odc.jsonl",
        "qwen": ROOT / "risultati_python/report_pylint/pylint_output_QWEN_with_odc.jsonl",
    },
    odc_mapping_path=ROOT / "mappings/python/pylint_odc.xlsx",
    semgrep_dirs={
        author: ROOT / f"risultati_python/report_semgrep/semgrep_batches_python_{author}"
        for author in ("human", "chatgpt", "dsc", "qwen")
    },
    semgrep_filename_mode="python_index",
    semgrep_requires_full_dataset=True,
    entropy_paths={
        author: ROOT
        / "EXTENDED/naturalness/RESULS_TREESITTER/python_norm_treesitter"
        / f"python_KenLM_{author}_code_6gram_treesitter_dataset_no_comments_normalized.csv"
        for author in ("human", "chatgpt", "dsc", "qwen")
    },
    entropy_language="python",
)

JAVA_CONFIG = LanguageConfig(
    name="java",
    dataset_path=ROOT / "datasets/final_datasets/java_dataset_dsc_qwen_FINAL.jsonl",
    dataset_key="hm_index",
    metrics_path=ROOT / "java_all_function_metrics_aggregates.jsonl",
    authors=("human", "chatgpt", "dsc", "qwen"),
    author_fields={
        "human": "human_code",
        "chatgpt": "chatgpt_code",
        "dsc": "dsc_code",
        "qwen": "qwen_code",
    },
    defect_kind="pmd",
    defect_paths={
        author: ROOT / f"risultati_java/report_PMD/reports_{author}/reports"
        for author in ("human", "chatgpt", "dsc", "qwen")
    },
    odc_mapping_path=ROOT / "mappings/java/pmd_odc.xlsx",
    semgrep_dirs={
        author: ROOT / f"risultati_java/report_semgrep/semgrep_batches_java_{author}"
        for author in ("human", "chatgpt", "dsc", "qwen")
    },
    semgrep_filename_mode="java_wrapped",
    semgrep_requires_full_dataset=False,
    entropy_paths={
        author: ROOT
        / "EXTENDED/naturalness/RESULS_TREESITTER/java_norm_treesitter"
        / f"java_KenLM_{author}_code_6gram_treesitter_dataset_no_comments_normalized.csv"
        for author in ("human", "chatgpt", "dsc", "qwen")
    },
    entropy_language="java",
)

C_CONFIG = LanguageConfig(
    name="c",
    dataset_path=ROOT / "c_dataset_final_corrected.jsonl",
    dataset_key="hexsha",
    metrics_path=ROOT / "c_all_function_metrics_aggregates.jsonl",
    authors=("human", "gptoss", "dsc", "qwen"),
    author_fields={
        "human": "human_code",
        "gptoss": "gptoss_code",
        "dsc": "dsc_code",
        "qwen": "qwen_code",
    },
    defect_kind="clang",
    defect_paths={
        author: ROOT
        / "C_defects/results"
        / author
        / f"clangt_output_{author}_with_odc_v2.jsonl"
        for author in ("human", "gptoss", "dsc", "qwen")
    },
    odc_mapping_path=ROOT / "mappings/c/clang_tidy_odc.xlsx",
    semgrep_dirs={
        author: ROOT / "C_security" / author
        for author in ("human", "gptoss", "dsc", "qwen")
    },
    semgrep_filename_mode="c_index",
    semgrep_requires_full_dataset=True,
    entropy_paths={
        author: ROOT
        / "EXTENDED/naturalness/RESULTS_REGEX"
        / "c_results_KenLM_10fold_regex_no_comments_final_normalized"
        / (
            f"c_KenLM_{author}_code_6gram_regex_"
            "dataset_final_no_comments_normalized.csv"
        )
        for author in ("human", "gptoss", "dsc", "qwen")
    },
    entropy_language="c",
)

LANGUAGE_CONFIGS = {"c": C_CONFIG, "java": JAVA_CONFIG, "python": PYTHON_CONFIG}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=sorted(LANGUAGE_CONFIGS), default="python")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "python_rq4_table.parquet",
        help="Parquet output path.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=ROOT / "python_rq4_validation.json",
        help="JSON validation report path.",
    )
    parser.add_argument(
        "--sample-per-author",
        type=int,
        default=0,
        help="Seeded reservoir sample size per author; 0 retains all functions.",
    )
    parser.add_argument("--seed", type=int, default=20250701)
    return parser.parse_args()


def require_files(config: LanguageConfig) -> None:
    paths = [
        config.dataset_path,
        config.metrics_path,
        *([config.odc_mapping_path] if config.odc_mapping_path else []),
        *config.entropy_paths.values(),
    ]
    if config.defect_kind in {"clang", "pylint"}:
        paths.extend(config.defect_paths.values())
    missing = [str(path) for path in paths if not path.is_file()]
    if config.defect_kind == "pmd":
        missing.extend(str(path) for path in config.defect_paths.values() if not path.is_dir())
    missing.extend(str(path) for path in config.semgrep_dirs.values() if not path.is_dir())
    assert not missing, f"Missing required inputs: {missing}"


def load_dataset_keys(config: LanguageConfig) -> Tuple[List[str], Dict[str, int]]:
    keys: List[str] = []
    key_to_idx: Dict[str, int] = {}
    with config.dataset_path.open(encoding="utf-8") as handle:
        for sample_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get(config.dataset_key)
            assert isinstance(key, str) and key, (
                f"Invalid {config.dataset_key} at dataset record {sample_idx}"
            )
            assert key not in key_to_idx, f"Duplicate dataset key: {key}"
            key_to_idx[key] = len(keys)
            keys.append(key)
    assert keys, "Dataset is empty"
    return keys, key_to_idx


def java_clean_for_wrapping(code: str) -> str:
    without_package = re.sub(
        r"^\s*package\s+[^\n;]+;\n?",
        "",
        code,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^\s*import\s+[^\n;]+;\n?",
        "",
        without_package,
        flags=re.MULTILINE,
    ).strip()


def java_has_orphan_methods(code: str) -> bool:
    pattern = re.compile(
        r"^\s*(public|protected|private)?\s+(static\s+)?"
        r"[\w<>\[\]]+\s+\w+\s*\([^;]*\)\s*"
        r"(throws\s+[\w, ]+)?\s*{",
        flags=re.MULTILINE,
    )
    return bool(pattern.search(code))


def java_top_level_type(code: str) -> str | None:
    match = re.search(r"public\s+(class|interface|enum)\s+(\w+)", code)
    return match.group(2) if match else None


def sanitize_java_filename(name: str) -> str:
    allowed = set(string.ascii_letters + string.digits + "_")
    return "".join(character if character in allowed else "_" for character in name)


def physical_sample_nloc(code: str, language: str) -> int:
    """Count nonblank physical lines after removing lexical comments.

    This scanner is deliberately parse-error tolerant. It preserves string and
    character literals, including Python triple-quoted strings, while removing
    Python ``#`` comments and Java ``//``/``/* ... */`` comments. Newlines are
    always preserved so the result remains a physical-line count.
    """
    output: List[str] = []
    index = 0
    state = "code"
    delimiter = ""
    escaped = False

    while index < len(code):
        character = code[index]
        following = code[index + 1] if index + 1 < len(code) else ""

        if state == "line_comment":
            if character == "\n":
                output.append(character)
                state = "code"
            else:
                output.append(" ")
            index += 1
            continue

        if state == "block_comment":
            if character == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "code"
            else:
                output.append("\n" if character == "\n" else " ")
                index += 1
            continue

        if state == "string":
            output.append(character)
            if escaped:
                escaped = False
                index += 1
                continue
            if character == "\\":
                escaped = True
                index += 1
                continue
            if delimiter in {"'''", '"""'}:
                if code.startswith(delimiter, index):
                    output.extend(delimiter[1:])
                    index += len(delimiter)
                    state = "code"
                else:
                    index += 1
            elif character == delimiter:
                index += 1
                state = "code"
            else:
                index += 1
            continue

        assert state == "code"
        if language == "python" and character == "#":
            output.append(" ")
            state = "line_comment"
            index += 1
        elif language in {"c", "java"} and character == "/" and following == "/":
            output.extend((" ", " "))
            state = "line_comment"
            index += 2
        elif language in {"c", "java"} and character == "/" and following == "*":
            output.extend((" ", " "))
            state = "block_comment"
            index += 2
        elif language == "python" and (
            code.startswith("'''", index) or code.startswith('"""', index)
        ):
            delimiter = code[index : index + 3]
            output.extend(delimiter)
            index += 3
            state = "string"
            escaped = False
        elif character in {"'", '"'}:
            delimiter = character
            output.append(character)
            index += 1
            state = "string"
            escaped = False
        else:
            output.append(character)
            index += 1

    value = sum(bool(line.strip()) for line in "".join(output).splitlines())
    assert value >= 0
    return value


def build_java_filename_maps(
    config: LanguageConfig,
    dataset_keys: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    assert config.name == "java"
    maps: Dict[str, Dict[str, str]] = {author: {} for author in config.authors}
    seen_names: Dict[str, Dict[str, int]] = {author: {} for author in config.authors}

    with config.dataset_path.open(encoding="utf-8") as handle:
        for dataset_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record[config.dataset_key])
            assert key == dataset_keys[dataset_idx]
            for author in config.authors:
                code = record.get(config.author_fields[author], "")
                if not code:
                    continue
                cleaned = java_clean_for_wrapping(str(code))
                top_level_class = re.search(
                    r"^\s*public\s+class\s+\w+",
                    cleaned,
                    flags=re.MULTILINE,
                )
                should_wrap = java_has_orphan_methods(cleaned) or not top_level_class
                type_name = java_top_level_type(cleaned)
                if not should_wrap and type_name:
                    occurrence = seen_names[author].get(type_name, 0) + 1
                    seen_names[author][type_name] = occurrence
                    final_name = type_name if occurrence == 1 else f"{type_name}_{occurrence}"
                    filename = f"{sanitize_java_filename(final_name)}.java"
                else:
                    filename = f"TempClass{dataset_idx}.java"
                assert filename not in maps[author], (
                    f"Duplicate generated Java filename: {(author, filename)}"
                )
                maps[author][filename] = key

    return maps


def metric_record(record: Mapping[str, Any], author: str) -> Dict[str, Any]:
    key = record.get("row_id")
    assert isinstance(key, str) and key, f"Invalid metric row_id: {record!r}"
    assert record.get("analysis_status") in {
        "ok",
        "empty_code",
        "no_functions_detected",
        "analysis_error",
    }
    assert int(record["function_count_computable"]) > 0
    result = {
        "hm_index": key,
        "author": author,
        "function_count_computable": int(record["function_count_computable"]),
        "nloc": float(record["nloc_mean"]),
        "ccn": float(record["ccn_mean"]),
        "parameter_count": float(record["parameter_count_mean"]),
        "max_nesting_depth": float(record["max_nesting_depth_mean"]),
        "distinct_operators": float(record["distinct_operators_mean"]),
        "distinct_operands": float(record["distinct_operands_mean"]),
        "total_operators": float(record["total_operators_mean"]),
        "total_operands": float(record["total_operands_mean"]),
        "halstead_v": float(record["halstead_volume_mean"]),
        "halstead_difficulty": float(record["halstead_difficulty_mean"]),
        "halstead_effort": float(record["halstead_effort_mean"]),
        "mi": float(record["maintainability_index_mean"]),
    }
    assert result["nloc"] > 0 and result["ccn"] > 0, f"Invalid complexity: {record!r}"
    assert all(float(result[name]) >= 0.0 for name in COMPLEXITY_PREDICTORS), (
        f"Negative complexity metric: {record!r}"
    )
    assert all(math.isfinite(float(result[name])) for name in COMPLEXITY_PREDICTORS)
    return result


def select_complexity_rows(
    config: LanguageConfig,
    dataset_keys: Mapping[str, int],
    sample_per_author: int,
    seed: int,
    capture_raw: bool,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, int],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, Dict[str, int]],
]:
    field_to_author = {field: author for author, field in config.author_fields.items()}
    retained: Dict[str, List[Dict[str, Any]]] = {author: [] for author in config.authors}
    retained_raw: Dict[str, List[Dict[str, Any]]] = {
        author: [] for author in config.authors
    }
    retained_counts = {author: 0 for author in config.authors}
    seen_keys = {author: set() for author in config.authors}
    seen_all_keys = {author: set() for author in config.authors}
    complexity_stats = {
        author: {
            "aggregate_rows": 0,
            "samples_without_computable_functions": 0,
            "functions_detected": 0,
            "functions_computable": 0,
            "functions_dropped": 0,
        }
        for author in config.authors
    }
    rngs = {
        author: random.Random(seed + 100_003 * index)
        for index, author in enumerate(config.authors)
    }

    with config.metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            field = raw.get("field")
            assert field in field_to_author, f"Unknown metric field at line {line_number}: {field}"
            author = field_to_author[field]
            key = raw.get("row_id")
            assert isinstance(key, str) and key in dataset_keys
            assert key not in seen_all_keys[author], (
                f"Duplicate sample aggregate key: {(author, key)}"
            )
            seen_all_keys[author].add(key)
            detected = int(raw["function_count_detected"])
            computable = int(raw["function_count_computable"])
            dropped = int(raw["function_count_dropped"])
            assert detected >= computable >= 0
            assert dropped == detected - computable
            stats = complexity_stats[author]
            stats["aggregate_rows"] += 1
            stats["functions_detected"] += detected
            stats["functions_computable"] += computable
            stats["functions_dropped"] += dropped
            if computable == 0:
                stats["samples_without_computable_functions"] += 1
                continue
            row = metric_record(raw, author)
            key = row["hm_index"]
            assert key in dataset_keys, f"Metric key absent from dataset: {(author, key)}"
            assert key not in seen_keys[author], f"Duplicate target metric key: {(author, key)}"
            seen_keys[author].add(key)
            retained_counts[author] += 1

            if sample_per_author <= 0:
                retained[author].append(row)
            elif len(retained[author]) < sample_per_author:
                retained[author].append(row)
                if capture_raw:
                    retained_raw[author].append(raw)
            else:
                replacement = rngs[author].randrange(retained_counts[author])
                if replacement < sample_per_author:
                    retained[author][replacement] = row
                    if capture_raw:
                        retained_raw[author][replacement] = raw

    for author in config.authors:
        assert len(seen_all_keys[author]) == len(dataset_keys), (
            f"Aggregate coverage mismatch for {author}: "
            f"{len(seen_all_keys[author])} != {len(dataset_keys)}"
        )
        assert complexity_stats[author]["aggregate_rows"] == len(dataset_keys)
        assert (
            complexity_stats[author]["samples_without_computable_functions"]
            == len(dataset_keys) - retained_counts[author]
        )
        assert retained_counts[author] == len(seen_keys[author])
        if sample_per_author > 0:
            assert len(retained[author]) == sample_per_author, (
                f"Not enough retained functions for {author}: "
                f"{len(retained[author])} < {sample_per_author}"
            )

    raw_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not capture_raw:
        return retained, retained_counts, raw_by_key, complexity_stats

    for author in config.authors:
        assert len(retained_raw[author]) == len(retained[author])
        for raw in retained_raw[author]:
            pair = (author, str(raw["row_id"]))
            assert pair not in raw_by_key
            raw_by_key[pair] = raw
    selected_keys = {
        (author, str(row["hm_index"]))
        for author, rows in retained.items()
        for row in rows
    }
    assert raw_by_key.keys() == selected_keys
    return retained, retained_counts, raw_by_key, complexity_stats


def attach_sample_nloc(
    config: LanguageConfig,
    selected: Mapping[str, Sequence[MutableMapping[str, Any]]],
) -> None:
    selected_by_author = {
        author: {str(row["hm_index"]): row for row in rows}
        for author, rows in selected.items()
    }
    attached = Counter()
    with config.dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record[config.dataset_key])
            for author in config.authors:
                row = selected_by_author[author].get(key)
                if row is None:
                    continue
                code = record.get(config.author_fields[author], "") or ""
                assert isinstance(code, str)
                value = physical_sample_nloc(code, config.name)
                assert value > 0, f"Nonpositive sample NLOC for {(author, key)}"
                row["sample_nloc"] = value
                attached[author] += 1

    for author in config.authors:
        assert attached[author] == len(selected[author]), (
            f"Missing sample NLOC values for {author}: "
            f"{attached[author]} != {len(selected[author])}"
        )


def aggregate_pylint(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    unique_findings: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for message in messages:
        symbol = message.get("symbol")
        odc = message.get("odc_category", "--")
        if symbol in PYLINT_EXCLUDED_SYMBOLS or odc == "--":
            continue
        assert odc in ODC_TO_COLUMN, f"Unknown ODC category: {odc!r}"
        finding_key = (symbol, odc, message.get("line"))
        unique_findings.setdefault(finding_key, dict(message))

    counts = {name: 0 for name in ODC_OUTPUTS}
    for _, odc, _ in unique_findings:
        counts[ODC_TO_COLUMN[odc]] += 1
    counts["defects_total"] = sum(counts.values())
    return counts, list(unique_findings.values())


def load_pylint_defects(
    config: LanguageConfig,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    capture_raw: bool,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, int]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, int],
]:
    results: Dict[Tuple[str, str], Dict[str, int]] = {}
    raw_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_counts: Dict[str, int] = {}

    for author in config.authors:
        targets = {str(row["hm_index"]) for row in selected[author]}
        seen: set[str] = set()
        with config.defect_paths[author].open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("hm_index")
                assert isinstance(key, str) and key, (
                    f"Invalid defect key in {config.defect_paths[author]}:{line_number}"
                )
                assert key not in seen, f"Duplicate defect source key: {(author, key)}"
                seen.add(key)
                if key in targets:
                    messages = record.get("pylint_output", [])
                    assert isinstance(messages, list)
                    counts, included = aggregate_pylint(messages)
                    results[(author, key)] = counts
                    if capture_raw:
                        raw_records[(author, key)] = {
                            "hm_index": key,
                            "pylint_output": messages,
                            "included_unique_findings": included,
                        }
        source_counts[author] = len(seen)
        missing = targets - seen
        assert not missing, f"Selected functions missing from defect source for {author}: {sorted(missing)[:5]}"

    return results, raw_records, source_counts


def load_pmd_defects(
    config: LanguageConfig,
    dataset_keys: Sequence[str],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    capture_raw: bool,
    java_filename_maps: Mapping[str, Mapping[str, str]],
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, int]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, int],
]:
    assert config.odc_mapping_path is not None
    mapping_frame = pd.read_excel(config.odc_mapping_path, engine="openpyxl")
    odc_map = dict(
        zip(
            mapping_frame["Difetto PMD"],
            mapping_frame["Classificazione ODC"],
        )
    )
    results: Dict[Tuple[str, str], Dict[str, int]] = {}
    raw_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_counts: Dict[str, int] = {}

    for author in config.authors:
        targets = {str(row["hm_index"]) for row in selected[author]}
        filename_map = dict(java_filename_maps[author])
        if author == "dsc":
            for filename, dataset_idx in JAVA_PMD_DSC_FILENAME_OVERRIDES.items():
                assert 0 <= dataset_idx < len(dataset_keys)
                filename_map[filename] = dataset_keys[dataset_idx]

        unique_findings: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {
            key: {} for key in targets
        }
        raw_entries: Dict[str, List[Dict[str, Any]]] = {
            key: [] for key in targets
        } if capture_raw else {}
        reported_filenames: set[str] = set()

        report_paths = sorted(config.defect_paths[author].glob("report_*.json"))
        assert report_paths, f"No PMD reports for {author}"
        for report_path in report_paths:
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            for file_entry in report.get("files", []):
                filename = Path(str(file_entry.get("filename", ""))).name
                assert filename in filename_map, (
                    f"Unresolved PMD filename for {author}: {filename}"
                )
                reported_filenames.add(filename)
                key = filename_map[filename]
                if key not in targets:
                    continue
                if capture_raw:
                    raw_entries[key].append(
                        {
                            "report": report_path.name,
                            "filename": file_entry.get("filename"),
                            "violations": file_entry.get("violations", []),
                        }
                    )
                for violation in file_entry.get("violations", []):
                    rule = violation.get("rule")
                    odc = odc_map.get(rule, "--")
                    if rule in PMD_EXCLUDED_RULES or odc == "--":
                        continue
                    assert odc in ODC_TO_COLUMN, f"Unknown PMD ODC category: {odc!r}"
                    finding_key = (rule, odc, violation.get("beginline"))
                    unique_findings[key].setdefault(finding_key, dict(violation))

        for key in targets:
            counts = {name: 0 for name in ODC_OUTPUTS}
            for _, odc, _ in unique_findings[key]:
                counts[ODC_TO_COLUMN[odc]] += 1
            counts["defects_total"] = sum(counts.values())
            results[(author, key)] = counts
            if capture_raw:
                raw_records[(author, key)] = {
                    "hm_index": key,
                    "pmd_file_entries": raw_entries[key],
                    "included_unique_findings": list(unique_findings[key].values()),
                }
        source_counts[author] = len(reported_filenames)

    return results, raw_records, source_counts


def clang_symbol_excluded(symbol: Any) -> bool:
    checks = [part.strip() for part in str(symbol or "").split(",") if part.strip()]
    return any(
        check in C_CLANG_EXCLUDED_SYMBOLS
        or check.startswith(C_CLANG_EXCLUDED_PREFIXES)
        for check in checks
    )


def load_clang_defects(
    config: LanguageConfig,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    capture_raw: bool,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, int]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, int],
]:
    assert config.odc_mapping_path is not None
    mapping_frame = pd.read_excel(config.odc_mapping_path, engine="openpyxl")
    odc_map = {
        str(check).strip(): (
            "--" if pd.isna(odc) else str(odc).strip()
        )
        for check, odc in zip(
            mapping_frame["Clang-tidy Check"],
            mapping_frame["ODC Defect Type"],
        )
        if not pd.isna(check)
    }
    results: Dict[Tuple[str, str], Dict[str, int]] = {}
    raw_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_counts: Dict[str, int] = {}

    for author in config.authors:
        targets = {str(row["hm_index"]) for row in selected[author]}
        seen: set[str] = set()
        with config.defect_paths[author].open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("hm_index")
                assert isinstance(key, str) and key, (
                    f"Invalid Clang key in {config.defect_paths[author]}:{line_number}"
                )
                assert key not in seen, f"Duplicate Clang source key: {(author, key)}"
                seen.add(key)
                if key not in targets:
                    continue

                messages = record.get("clangt_output", [])
                assert isinstance(messages, list)
                unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
                for message in messages:
                    symbol = str(message.get("check_name") or "").strip()
                    odc = message.get("odc_category", "--")
                    expected_odc = odc_map.get(symbol, "--")
                    assert odc == expected_odc, (
                        f"Stale C ODC mapping at {(author, key, symbol)}: "
                        f"{odc!r} != {expected_odc!r}"
                    )
                    if (
                        clang_symbol_excluded(symbol)
                        or odc == "--"
                        or odc in C_CLANG_EXCLUDED_ODC
                    ):
                        continue
                    assert odc in ODC_TO_COLUMN, f"Unknown Clang ODC category: {odc!r}"
                    location = message.get("location", {})
                    lines = location.get("lines", {}) if isinstance(location, dict) else {}
                    begin_line = lines.get("begin") if isinstance(lines, dict) else None
                    finding_key = (symbol, odc, begin_line)
                    unique.setdefault(finding_key, dict(message))

                counts = {name: 0 for name in ODC_OUTPUTS}
                for _, odc, _ in unique:
                    counts[ODC_TO_COLUMN[odc]] += 1
                counts["defects_total"] = sum(counts.values())
                results[(author, key)] = counts
                if capture_raw:
                    raw_records[(author, key)] = {
                        "hm_index": key,
                        "clangt_output": messages,
                        "included_unique_findings": list(unique.values()),
                    }

        source_counts[author] = len(seen)
        missing = targets - seen
        assert not missing, (
            f"Selected samples missing from Clang source for {author}: "
            f"{sorted(missing)[:5]}"
        )

    return results, raw_records, source_counts


def load_defects(
    config: LanguageConfig,
    dataset_keys: Sequence[str],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    capture_raw: bool,
    java_filename_maps: Mapping[str, Mapping[str, str]] | None,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, int]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, int],
]:
    if config.defect_kind == "pylint":
        return load_pylint_defects(config, selected, capture_raw)
    if config.defect_kind == "pmd":
        assert java_filename_maps is not None
        return load_pmd_defects(
            config,
            dataset_keys,
            selected,
            capture_raw,
            java_filename_maps,
        )
    if config.defect_kind == "clang":
        return load_clang_defects(config, selected, capture_raw)
    raise ValueError(f"Unsupported defect kind: {config.defect_kind}")


def path_sample_id(path_value: Any) -> int | None:
    if not isinstance(path_value, str):
        return None
    match = PATH_SAMPLE_RE.search(path_value)
    return int(match.group(1)) if match else None


def normalized_cwes(cwes: Any) -> Tuple[str, ...]:
    values = cwes if isinstance(cwes, list) else [cwes]
    normalized = []
    for value in values:
        text = str(value).strip().upper()
        match = CWE_RE.match(text)
        normalized.append(match.group(1).upper() if match else text)
    return tuple(sorted(normalized))


def batch_number(path: Path) -> int:
    match = re.search(r"_batch_(\d+)\.json$", path.name)
    assert match, f"Unexpected Semgrep batch filename: {path}"
    return int(match.group(1))


def load_vulnerabilities(
    config: LanguageConfig,
    dataset_keys: Sequence[str],
    key_to_idx: Mapping[str, int],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    capture_raw: bool,
    java_filename_maps: Mapping[str, Mapping[str, str]] | None,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, int]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, Dict[str, int]],
]:
    results: Dict[Tuple[str, str], Dict[str, int]] = {}
    raw_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_stats: Dict[str, Dict[str, int]] = {}
    dataset_size = len(dataset_keys)

    for author in config.authors:
        target_keys = {str(row["hm_index"]) for row in selected[author]}
        scanned_keys: set[str] = set()
        source_references: Dict[str, str] = {}
        target_errors: set[str] = set()
        target_findings: Dict[str, List[Dict[str, Any]]] = {}
        all_error_keys: set[str] = set()

        def resolve_path(path_value: Any) -> str | None:
            if not isinstance(path_value, str):
                return None
            if config.semgrep_filename_mode in {"c_index", "python_index"}:
                sample_id = path_sample_id(path_value)
                if sample_id is None or not (1 <= sample_id <= dataset_size):
                    return None
                return dataset_keys[sample_id - 1]
            if config.semgrep_filename_mode == "java_wrapped":
                assert java_filename_maps is not None
                return java_filename_maps[author].get(Path(path_value).name)
            raise ValueError(
                f"Unsupported Semgrep filename mode: {config.semgrep_filename_mode}"
            )

        batch_paths = sorted(
            (
                path
                for path in config.semgrep_dirs[author].glob("*.json")
                if re.search(r"_batch_\d+\.json$", path.name)
            ),
            key=batch_number,
        )
        assert batch_paths, f"No Semgrep batches for {author}"
        for batch_path in batch_paths:
            with batch_path.open(encoding="utf-8") as handle:
                batch = json.load(handle)

            for scanned_path in batch.get("paths", {}).get("scanned", []):
                key = resolve_path(scanned_path)
                assert key is not None, (
                    f"Invalid scanned path: {scanned_path!r}"
                )
                assert key not in scanned_keys, (
                    f"Duplicate Semgrep scanned function for {author}: {key}"
                )
                scanned_keys.add(key)
                source_references[key] = Path(str(scanned_path)).name

            for error in batch.get("errors", []):
                error_path = error.get("path")
                if isinstance(error_path, str) and error_path.startswith("https:/semgrep.dev/..."):
                    continue
                key = resolve_path(error_path)
                if key is not None:
                    all_error_keys.add(key)
                    if key in target_keys:
                        target_errors.add(key)

            for finding in batch.get("results", []):
                key = resolve_path(finding.get("path"))
                if key not in target_keys:
                    continue
                cwes = finding.get("extra", {}).get("metadata", {}).get("cwe")
                if cwes:
                    target_findings.setdefault(key, []).append(finding)

        assert target_keys <= scanned_keys, (
            f"Selected functions missing from Semgrep scan for {author}: "
            f"{sorted(target_keys - scanned_keys)[:5]}"
        )
        if config.semgrep_requires_full_dataset:
            assert scanned_keys == set(dataset_keys), (
                f"Semgrep does not cover the full dataset for {author}"
            )
        elif config.semgrep_filename_mode == "java_wrapped":
            assert java_filename_maps is not None
            assert scanned_keys == set(java_filename_maps[author].values()), (
                f"Semgrep coverage differs from wrapped Java inputs for {author}"
            )

        for key in target_keys:
            raw_findings = target_findings.get(key, [])
            # A Semgrep partial-parse error does not invalidate findings emitted
            # from the portions that were parsed successfully. Keep those
            # findings; zero-fill only when no CWE-bearing finding was returned.
            eligible = raw_findings
            unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
            for finding in eligible:
                extra = finding.get("extra", {})
                cwes = extra.get("metadata", {}).get("cwe")
                issue_key = (
                    normalized_cwes(cwes),
                    str(extra.get("severity", "")).upper(),
                    str(extra.get("lines", "")).strip(),
                )
                unique.setdefault(issue_key, finding)

            high = sum(
                1
                for _, severity, _ in unique
                if severity in {"CRITICAL", "ERROR"}
            )
            results[(author, key)] = {
                "vulns_total": len(unique),
                "vulns_high_sev": high,
            }
            if capture_raw:
                raw_records[(author, key)] = {
                    "source_file": source_references[key],
                    "semgrep_error": key in target_errors,
                    "raw_cwe_findings": raw_findings,
                    "included_unique_findings": list(unique.values()),
                }

        source_stats[author] = {
            "scanned": len(scanned_keys),
            "error_functions": len(all_error_keys),
            "sampled_error_functions": len(target_errors),
        }

    return results, raw_records, source_stats


def load_entropy(
    config: LanguageConfig,
    dataset_keys: Sequence[str],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    capture_raw: bool,
) -> Tuple[
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, int],
    Dict[str, int],
]:
    results: Dict[Tuple[str, str], float] = {}
    raw_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_counts: Dict[str, int] = {}
    missing_selected = {author: 0 for author in config.authors}
    missing_pairs: set[Tuple[str, str]] = set()
    dataset_size = len(dataset_keys)
    log2_10 = math.log2(10.0)

    for author in config.authors:
        targets = {str(row["hm_index"]) for row in selected[author]}
        seen = bytearray(dataset_size)
        own_count = 0
        expected_train = f"train{config.author_fields[author]}"
        with config.entropy_paths[author].open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                if record.get("source") != author:
                    continue
                model_type = record.get("model_type", "")
                assert expected_train in model_type, (
                    f"Wrong-author entropy model for {author}: {model_type}"
                )
                assert f"lang{config.entropy_language}" in model_type
                sample_idx = int(record["sample_idx"])
                assert 0 <= sample_idx < dataset_size
                assert seen[sample_idx] == 0, (
                    f"Duplicate own-author OOF entropy: {(author, sample_idx)}"
                )
                seen[sample_idx] = 1
                own_count += 1
                stored_value = float(record["cross_entropy_bits"])
                key = dataset_keys[sample_idx]
                if key in targets:
                    pair = (author, key)
                    if math.isfinite(stored_value):
                        results[pair] = stored_value * log2_10
                        if capture_raw:
                            raw_records[pair] = dict(record)
                    else:
                        missing_selected[author] += 1
                        missing_pairs.add(pair)

        assert own_count == dataset_size, (
            f"Own-author entropy coverage mismatch for {author}: "
            f"{own_count} != {dataset_size}"
        )
        assert all(seen), f"Missing own-author OOF entropy rows for {author}"
        source_counts[author] = own_count

    expected_pairs = {
        (author, str(row["hm_index"]))
        for author in config.authors
        for row in selected[author]
    }
    assert results.keys() | missing_pairs == expected_pairs, (
        "Selected own-author entropy records do not cover the selected cohort"
    )
    assert not (results.keys() & missing_pairs)
    return results, raw_records, source_counts, missing_selected


def build_rows(
    config: LanguageConfig,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    defects: Mapping[Tuple[str, str], Mapping[str, int]],
    vulnerabilities: Mapping[Tuple[str, str], Mapping[str, int]],
    entropy: Mapping[Tuple[str, str], float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for author in config.authors:
        for metric in selected[author]:
            key = str(metric["hm_index"])
            pair = (author, key)
            if pair not in entropy:
                continue
            row = dict(metric)
            row["naturalness"] = float(entropy[pair])
            row.update(defects[pair])
            row.update(vulnerabilities[pair])
            rows.append(row)

    rows.sort(key=lambda row: (config.authors.index(str(row["author"])), str(row["hm_index"])))
    expected = len(entropy)
    assert len(rows) == expected
    assert len({(row["hm_index"], row["author"]) for row in rows}) == expected

    for row in rows:
        assert isinstance(row["hm_index"], str) and row["hm_index"]
        assert row["author"] in config.authors
        assert all(math.isfinite(float(row[name])) for name in PREDICTORS)
        for name in OUTCOMES:
            assert isinstance(row[name], int) and row[name] >= 0, (
                f"Invalid outcome {name}: {row!r}"
            )
        assert row["defects_total"] == sum(row[name] for name in ODC_OUTPUTS)
        assert row["vulns_high_sev"] <= row["vulns_total"]
    return rows


def table_from_rows(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            ("hm_index", pa.string()),
            ("author", pa.string()),
            ("sample_nloc", pa.int64()),
            ("function_count_computable", pa.int64()),
            ("nloc", pa.float64()),
            ("ccn", pa.float64()),
            ("parameter_count", pa.float64()),
            ("max_nesting_depth", pa.float64()),
            ("distinct_operators", pa.float64()),
            ("distinct_operands", pa.float64()),
            ("total_operators", pa.float64()),
            ("total_operands", pa.float64()),
            ("halstead_v", pa.float64()),
            ("halstead_difficulty", pa.float64()),
            ("halstead_effort", pa.float64()),
            ("mi", pa.float64()),
            ("naturalness", pa.float64()),
            ("defects_total", pa.int64()),
            *[(name, pa.int64()) for name in ODC_OUTPUTS],
            ("vulns_total", pa.int64()),
            ("vulns_high_sev", pa.int64()),
        ]
    )
    columns = {
        field.name: [row[field.name] for row in rows]
        for field in schema
    }
    return pa.Table.from_pydict(columns, schema=schema)


def choose_worked_example(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    human = [row for row in rows if row["author"] == "human"]
    for row in human:
        if row["defects_total"] > 0 and row["vulns_total"] > 0:
            return row
    for row in human:
        if row["defects_total"] > 0 or row["vulns_total"] > 0:
            return row
    assert human
    return human[0]


def write_validation_report(
    args: argparse.Namespace,
    config: LanguageConfig,
    dataset_keys: Sequence[str],
    retained_counts: Mapping[str, int],
    complexity_stats: Mapping[str, Mapping[str, int]],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
    raw_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
    raw_defects: Mapping[Tuple[str, str], Mapping[str, Any]],
    raw_vulnerabilities: Mapping[Tuple[str, str], Mapping[str, Any]],
    raw_entropy: Mapping[Tuple[str, str], Mapping[str, Any]],
    defect_source_counts: Mapping[str, int],
    semgrep_stats: Mapping[str, Mapping[str, int]],
    entropy_source_counts: Mapping[str, int],
    entropy_missing_selected: Mapping[str, int],
    key_to_idx: Mapping[str, int],
) -> Dict[str, Any]:
    by_author = {author: [row for row in rows if row["author"] == author] for author in config.authors}
    report: Dict[str, Any] = {
        "language": config.name,
        "seed": args.seed,
        "sample_per_author": args.sample_per_author,
        "dataset_rows": len(dataset_keys),
        "authors": {},
        "assertions": {
            "unique_hm_index_author": True,
            "complexity_uses_all_computable_function_means": True,
            "sample_nloc_is_whole_sample_physical_nloc": True,
            "predictors_non_null_and_finite": True,
            "outcomes_integer_nonnegative": True,
            "own_author_oof_entropy_exactly_one": True,
            "semgrep_complete_scan_coverage": True,
            "all_joins_explicitly_resolved_to_hm_index": True,
        },
    }
    for author, author_rows in by_author.items():
        report["authors"][author] = {
            "complexity_retained_full": retained_counts[author],
            "complexity_dropped_full": len(dataset_keys) - retained_counts[author],
            "complexity_source": dict(complexity_stats[author]),
            "sample_rows": len(author_rows),
            "defect_source_rows": defect_source_counts[author],
            "entropy_own_author_rows": entropy_source_counts[author],
            "entropy_nonfinite_selected_dropped": entropy_missing_selected[author],
            "semgrep": dict(semgrep_stats[author]),
            "predictor_nulls": {
                name: sum(row[name] is None for row in author_rows)
                for name in ("sample_nloc", *PREDICTORS)
            },
            "outcome_zero_counts": {
                name: sum(row[name] == 0 for row in author_rows)
                for name in OUTCOMES
            },
        }

    if raw_metrics:
        example = dict(choose_worked_example(rows))
        pair = (str(example["author"]), str(example["hm_index"]))
        report["worked_example"] = {
            "dataset_key_bridge": {
                "sample_idx_zero_based": key_to_idx[pair[1]],
                "semgrep_source_reference": raw_vulnerabilities[pair].get(
                    "source_file",
                    key_to_idx[pair[1]] + 1,
                ),
                "hm_index": pair[1],
            },
            "complexity_raw": raw_metrics[pair],
            "defects_raw": raw_defects[pair],
            "semgrep_raw": raw_vulnerabilities[pair],
            "entropy_raw": raw_entropy[pair],
            "entropy_conversion": "naturalness = stored cross_entropy_bits * log2(10)",
            "final_joined_row": example,
        }

    args.validation_report.parent.mkdir(parents=True, exist_ok=True)
    with args.validation_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def print_report(report: Mapping[str, Any], output: Path, validation_report: Path) -> None:
    print(f"Wrote table: {output}")
    print(f"Wrote validation report: {validation_report}")
    print("Validation by author:")
    for author, values in report["authors"].items():
        print(
            f"  {author}: retained={values['complexity_retained_full']}, "
            f"dropped={values['complexity_dropped_full']}, "
            f"entropy_dropped={values['entropy_nonfinite_selected_dropped']}, "
            f"sample={values['sample_rows']}, "
            f"sampled_semgrep_errors={values['semgrep']['sampled_error_functions']}"
        )
        print(f"    complexity_source={values['complexity_source']}")
        print(f"    predictor_nulls={values['predictor_nulls']}")
        print(f"    outcome_zero_counts={values['outcome_zero_counts']}")
    if "worked_example" in report:
        print("Worked example:")
        print(json.dumps(report["worked_example"], indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    assert args.sample_per_author >= 0
    config = LANGUAGE_CONFIGS[args.language]
    require_files(config)

    print("Loading canonical dataset keys...", flush=True)
    dataset_keys, key_to_idx = load_dataset_keys(config)
    java_filename_maps = (
        build_java_filename_maps(config, dataset_keys)
        if config.name == "java"
        else None
    )
    capture_raw = args.sample_per_author > 0
    print("Selecting asserted all-function sample aggregates...", flush=True)
    selected, retained_counts, raw_metrics, complexity_stats = select_complexity_rows(
        config,
        key_to_idx,
        args.sample_per_author,
        args.seed,
        capture_raw,
    )
    print("Computing whole-sample physical NLOC controls...", flush=True)
    attach_sample_nloc(config, selected)
    print(f"Aggregating {config.defect_kind.upper()}/ODC outcomes...", flush=True)
    defects, raw_defects, defect_source_counts = load_defects(
        config,
        dataset_keys,
        selected,
        capture_raw,
        java_filename_maps,
    )
    print("Aggregating Semgrep/CWE outcomes...", flush=True)
    vulnerabilities, raw_vulnerabilities, semgrep_stats = load_vulnerabilities(
        config,
        dataset_keys,
        key_to_idx,
        selected,
        capture_raw,
        java_filename_maps,
    )
    print("Selecting own-author out-of-fold entropy...", flush=True)
    entropy, raw_entropy, entropy_source_counts, entropy_missing_selected = load_entropy(
        config,
        dataset_keys,
        selected,
        capture_raw,
    )
    print("Building and validating joined rows...", flush=True)
    rows = build_rows(config, selected, defects, vulnerabilities, entropy)
    table = table_from_rows(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    written = pq.read_table(args.output)
    assert written.num_rows == len(rows)
    assert written.schema == table.schema

    report = write_validation_report(
        args,
        config,
        dataset_keys,
        retained_counts,
        complexity_stats,
        selected,
        rows,
        raw_metrics,
        raw_defects,
        raw_vulnerabilities,
        raw_entropy,
        defect_source_counts,
        semgrep_stats,
        entropy_source_counts,
        entropy_missing_selected,
        key_to_idx,
    )
    print_report(report, args.output, args.validation_report)


if __name__ == "__main__":
    main()
