from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from .config import ROOT


@lru_cache(maxsize=1)
def rq4_module() -> ModuleType:
    path = ROOT / "support/rq4_build_table.py"
    specification = importlib.util.spec_from_file_location("cqbench_rq4_legacy", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def rq4_config(language: str):
    module = rq4_module()
    return module.LANGUAGE_CONFIGS[language]


def dataset_keys(language: str) -> list[str]:
    module = rq4_module()
    config = rq4_config(language)
    keys, _ = module.load_dataset_keys(config)
    return keys


def selected_rows(authors: Sequence[str], keys: Sequence[str]):
    return {
        author: [{"hm_index": key, "author": author} for key in keys]
        for author in authors
    }


def load_raw_findings(
    language: str,
    authors: Sequence[str],
    keys: Sequence[str],
) -> tuple[
    Mapping[tuple[str, str], Mapping[str, Any]],
    Mapping[tuple[str, str], Mapping[str, Any]],
]:
    module = rq4_module()
    config = rq4_config(language)
    all_keys = dataset_keys(language)
    key_to_idx = {key: index for index, key in enumerate(all_keys)}
    selected = selected_rows(authors, keys)
    java_maps = (
        module.build_java_filename_maps(config, all_keys)
        if language == "java"
        else None
    )
    _, raw_defects, _ = module.load_defects(
        config,
        all_keys,
        selected,
        True,
        java_maps,
    )
    _, raw_vulnerabilities, _ = module.load_vulnerabilities(
        config,
        all_keys,
        key_to_idx,
        selected,
        True,
        java_maps,
    )
    return raw_defects, raw_vulnerabilities


def load_raw_vulnerabilities(
    language: str,
    authors: Sequence[str],
    keys: Sequence[str],
) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    module = rq4_module()
    config = rq4_config(language)
    all_keys = dataset_keys(language)
    key_to_idx = {key: index for index, key in enumerate(all_keys)}
    selected = selected_rows(authors, keys)
    java_maps = (
        module.build_java_filename_maps(config, all_keys)
        if language == "java"
        else None
    )
    _, raw_vulnerabilities, _ = module.load_vulnerabilities(
        config,
        all_keys,
        key_to_idx,
        selected,
        True,
        java_maps,
    )
    return raw_vulnerabilities


def cwes_from_raw(record: Mapping[str, Any]) -> set[str]:
    module = rq4_module()
    values: set[str] = set()
    for finding in record.get("included_unique_findings", []):
        metadata = finding.get("extra", {}).get("metadata", {})
        cwes = metadata.get("cwe")
        if cwes:
            values.update(module.normalized_cwes(cwes))
    return values


def defect_evidence(record: Mapping[str, Any], odc_column: str) -> list[dict[str, Any]]:
    module = rq4_module()
    output = []
    for finding in record.get("included_unique_findings", []):
        odc = finding.get("odc_category")
        if odc is None and "rule" in finding:
            # PMD raw findings do not carry the mapped category. The selection
            # count remains authoritative; include the raw rule for review.
            output.append(dict(finding))
        elif module.ODC_TO_COLUMN.get(odc) == odc_column:
            output.append(dict(finding))
    return output
