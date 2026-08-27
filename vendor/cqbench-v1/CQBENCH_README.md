# CQBench v1

CQBench is a 27,346-task static code-quality challenge benchmark for Python, Java, and C. It evaluates whether generated code is structurally complete, non-degenerate relative to a human reference, and free from defects and vulnerability patterns.

CQBench does not execute code and does not measure functional correctness or exploitability. It is a failure-derived stress benchmark, not a representative sample of all programming tasks.

## Contents

- `benchmark/tasks.jsonl`: 27,346 prompts and public task metadata.
- `benchmark/references.jsonl`: human structural and complexity references.
- `benchmark/baselines/`: baseline Human, OpenAI, DeepSeek, and Qwen code.
- `benchmark/results/`: keyed baseline evaluation results.
- `cqbench/`: evaluator, reporting, and audit implementation.
- `cqbench/rules/`: frozen Semgrep rules and rule manifest.
- `mappings/`: language-organized analyzer-to-ODC mappings.
- `replication/`: study scripts (naturalness) and result summaries for transparency; see `replication/README.md`.
- `experiments/`: worked example evaluations (e.g. `claude-opus-4-8/`).
- `data/DATA.md`: pointer to the full dataset (hosted on Zenodo).
- `Dockerfile`: pinned container environment.

## Package layout

```text
CQBench/
├── benchmark/
│   ├── tasks.jsonl
│   ├── references.jsonl
│   ├── baselines/
│   └── results/
├── cqbench/
│   └── rules/
├── mappings/
│   ├── python/pylint_odc.xlsx
│   ├── java/pmd_odc.xlsx
│   └── c/clang_tidy_odc.xlsx
├── data/
│   └── DATA.md
├── replication/
│   ├── scripts/       (naturalness)
│   ├── results/       (rq1–rq4)
│   └── calibration/   (§4.2 model consistency)
├── experiments/
│   └── claude-opus-4-8/   (worked example run)
├── support/
├── tests/
├── tools/
├── Dockerfile
└── pyproject.toml
```

`support/` contains compatibility modules because the evaluator imports the exact metric and exclusion definitions used in the study.

The unified `openai.jsonl` baseline uses ChatGPT for Python and Java and GPT-OSS for C, matching the study convention.

## Native setup

```bash
python -m pip install -e '.[analysis,test]'
python -m pytest -q
```

Full evaluation additionally requires PMD 7.11.0 and Clang-Tidy 18 on `PATH`.
The Docker image installs them.

## Docker setup

```bash
docker build -t cqbench:1.0 .
docker run --rm \
  -v "$PWD/benchmark:/workspace/benchmark:ro" \
  cqbench:1.0 validate-submission \
  --tasks benchmark/tasks.jsonl \
  --predictions benchmark/baselines/human.jsonl
```

To evaluate a host-side prediction file:

```bash
docker run --rm \
  -v "$PWD/predictions.jsonl:/data/predictions.jsonl:ro" \
  -v "$PWD/output:/data/output" \
  cqbench:1.0 evaluate \
  --tasks benchmark/tasks.jsonl \
  --references benchmark/references.jsonl \
  --predictions /data/predictions.jsonl \
  --output /data/output/results.jsonl
```

## Submission format

Provide one JSON object per task. `task_id` is the only join key.

```json
{"task_id":"python:gp000001","code":"def requested_function(...):\n    ..."}
```

Validate before analysis:

```bash
python -m cqbench validate-submission \
  --tasks benchmark/tasks.jsonl \
  --predictions predictions.jsonl
```

Missing predictions remain part of the denominator and are evaluated as empty outputs. Unknown and duplicate task IDs fail validation.

## Scores

The principal endpoint is `clean_strict_at_1`: the generated target must be parseable, present with the expected arity, non-stub, non-constant, structurally nontrivial, complexity-nondegenerate, and have zero included defect and vulnerability findings.

Supporting measures include submission, parseability, target-presence, non-stub and strict-nontrivial rates; defect-, vulnerability-, and high-severity-free rates; ODC incidence; and structural complexity summaries. `Critical` and `Error` Semgrep findings are high severity.

Generate a report:

```bash
python -m cqbench report \
  --results results.jsonl \
  --model-name my-model \
  --output-dir reports/my-model
```

Compare it with baselines using paired, seeded 10,000-resample bootstrap intervals:

```bash
python -m cqbench compare \
  --submission results.jsonl \
  --baseline benchmark/results/openai.jsonl \
  --baseline benchmark/results/dsc.jsonl \
  --baseline benchmark/results/qwen.jsonl \
  --output comparison.csv
```

## Dataset

The benchmark tasks, references, baselines, and results in `benchmark/` are self-contained. The full source dataset — the ⟨docstring, human-code, LLM-code⟩ tuples and per-function metric tables the tasks are derived from — is hosted on Zenodo (it is too large to ship in this repository). See [`data/DATA.md`](data/DATA.md) for the DOI, contents, checksums, and where to place each file.

## Interpretation constraint

Tasks were selected because at least two historical model outputs passed the complexity gate, had at least three included findings, and shared an ODC type or normalized CWE. This enriches failures by design. Results support claims about robustness on known issue-prone tasks, not population-wide model quality.
