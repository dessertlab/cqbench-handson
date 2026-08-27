from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
SEED = 2025
BOOTSTRAP_RESAMPLES = 10_000
BENCHMARK_VERSION = "cqbench-v1"

ODC_COLUMNS = (
    "def_assignment",
    "def_algorithm",
    "def_interface",
    "def_checking",
    "def_timing",
    "def_function_class_object",
)

ODC_LABELS = {
    "def_assignment": "Assignment",
    "def_algorithm": "Algorithm",
    "def_interface": "Interface",
    "def_checking": "Checking",
    "def_timing": "Timing/Serialization",
    "def_function_class_object": "Function/Class/Object",
}

@dataclass(frozen=True)
class LanguageSpec:
    name: str
    dataset: Path
    source_key: str
    table: Path
    human_field: str
    model_fields: Mapping[str, str]

    @property
    def authors(self) -> tuple[str, ...]:
        return ("human", *self.model_fields.keys())


LANGUAGES = {
    "python": LanguageSpec(
        name="python",
        dataset=ROOT / "datasets/final_datasets/python_dataset_nodocs_dsc_qwen_FINAL.jsonl",
        source_key="hm_index",
        table=ROOT / "python_rq4_table.parquet",
        human_field="human_code",
        model_fields={
            "chatgpt": "chatgpt_code",
            "dsc": "dsc_code",
            "qwen": "qwen_code",
        },
    ),
    "java": LanguageSpec(
        name="java",
        dataset=ROOT / "datasets/final_datasets/java_dataset_dsc_qwen_FINAL.jsonl",
        source_key="hm_index",
        table=ROOT / "java_rq4_table.parquet",
        human_field="human_code",
        model_fields={
            "chatgpt": "chatgpt_code",
            "dsc": "dsc_code",
            "qwen": "qwen_code",
        },
    ),
    "c": LanguageSpec(
        name="c",
        dataset=ROOT / "c_dataset_final_corrected.jsonl",
        source_key="hexsha",
        table=ROOT / "c_rq4_table.parquet",
        human_field="human_code",
        model_fields={
            "gptoss": "gptoss_code",
            "dsc": "dsc_code",
            "qwen": "qwen_code",
        },
    ),
}

