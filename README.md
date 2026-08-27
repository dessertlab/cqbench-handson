# Benchmarking the Quality of AI Code Generators — hands-on

A three-hour, Python-only hands-on session for **LLMA4SE 2026**, built on
**CQBench v1** — the static code-quality challenge benchmark released with
*"[paper title — fill this in]"*.

You will score **200 benchmark tasks × 5 code authors = 1,000 evaluations**
live, on your own laptop, in about two minutes — then take the numbers apart.

| Author | What it is |
|---|---|
| **Human** | The original human implementation the task was mined from |
| **ChatGPT** | `gpt-3.5-turbo` |
| **DeepSeek-Coder** | `DeepSeek-Coder-33B-Instruct` |
| **Qwen2.5-Coder** | `Qwen2.5-Coder-32B-Instruct` |
| **Claude Opus 4.8** | A frontier model released *after* the benchmark was built |

No code is generated during the session: every submission already exists in
`data/predictions/`. The session is about **measurement**, not generation.

---

## What you will actually do

| # | Notebook | Time | What happens |
|---|---|---|---|
| 00 | `00_setup.ipynb` | 15 min | Check the environment, meet the data, read one task and its five answers |
| 01 | `01_measurement_pipeline.ipynb` | 35 min | Run Pylint, Semgrep and lizard **by hand** on a single function and watch a raw finding become a benchmark number |
| 02 | `02_run_the_benchmark.ipynb` | 30 min | Score all 5 authors on all 200 tasks; verify the fast runner against the reference evaluator; reproduce Table 5 of the paper |
| 03 | `03_rq1_structure_and_style.ipynb` | 35 min | **RQ1** — do models write structurally different code? NLOC, cyclomatic complexity, Halstead, maintainability, lexical diversity |
| 04 | `04_rq2_defects.ipynb` | 30 min | **RQ2** — defect types and frequencies, mapped to Orthogonal Defect Classification |
| 05 | `05_rq3_security.ipynb` | 30 min | **RQ3** — vulnerability classes and severity, mapped to CWE |

`slides/index.html` is a 14-slide opening deck (10–15 min) to project before
anyone opens a notebook. See `slides/README.md` for the controls.

Notebooks 00–02 are fully worked: run them and read. Notebooks 03–05 mix worked
analysis with `TODO` exercises; worked solutions are in `notebooks/solutions/`.

---

## Setup

**Before the session.** Setup takes about five minutes and needs ~2 GB of disk.

```bash
git clone <this repo>
cd cqbench-handson

conda env create -f environment.yml
conda activate cqbench-handson

python setup/verify_setup.py     # must print "Ready."
jupyter lab
```

`verify_setup.py` checks the two analyzer binaries and their exact versions,
then scores 8 tasks end to end. If it prints `Ready.` you are done.

> **Platform.** macOS and Linux work out of the box. Semgrep does not run
> natively on Windows — Windows users should work inside **WSL2** (Ubuntu),
> where the same instructions apply unchanged.

> **Version pins are part of the measurement.** `pylint==3.3.6` and
> `semgrep==1.120.0` are the versions the study used. A different version
> reports different findings and your numbers stop being comparable to the
> paper's. This is the first lesson of the session, and it is enforced in
> `environment.yml`.

**If setup fails on the day**, everything still works: `results/precomputed/`
ships the output of an identical run, and every analysis notebook falls back to
it automatically. You will lose notebook 02's live run and nothing else.

---

## Repository layout

```
cqbench-handson/
├── environment.yml            conda environment (pinned analyzers)
├── setup/verify_setup.py      pre-flight check
├── data/
│   ├── tasks.jsonl            200 Python tasks: prompt, signature, stratum
│   ├── references.jsonl       human structural + complexity reference per task
│   ├── predictions/*.jsonl    the five authors' submissions
│   ├── frozen/*.jsonl         the study's own frozen results, for cross-checking
│   └── reference_check/       8-task slice used for the equivalence check
├── results/
│   ├── precomputed/*.jsonl    shipped fallback (identical pipeline)
│   ├── reference_check/       output of the stock `cqbench evaluate`
│   └── live/                  your run lands here
├── cqhandson/                 helper package
│   ├── runner.py              batched evaluator — see "Why it is fast"
│   ├── metrics.py             the scoring vocabulary, on tidy frames
│   ├── loading.py             data loading
│   └── viz.py                 paper palette and figure defaults
├── notebooks/                 the session
├── slides/index.html          opening framing deck
└── vendor/cqbench-v1/         the unmodified CQBench evaluator (GPL-3.0)
```

## The 200 tasks

They are the Python half of the 600-task subset the paper used for its
frontier-model demonstration, so notebook 02 reproduces a published table
exactly. Sampled deterministically (seed 2025), proportional to the benchmark's
three selection strata:

| Stratum | Tasks | Selected because |
|---|---:|---|
| `defect_consensus` | 100 | ≥2 models produced ≥3 findings of a shared **ODC defect type** |
| `mixed_consensus` | 77 | both gates fired |
| `vulnerability_consensus` | 23 | ≥2 models produced ≥3 findings of a shared **CWE class** |

**This is a failure-derived challenge set, not a random sample of programming.**
Tasks were kept *because* the 2023–24 models struggled with them. Even the human
references trigger a defect on 62% of them. Rates measured here describe
robustness on known issue-prone code; they do not estimate a model's average
quality. Notebook 00 makes this concrete and every later notebook keeps it in view.

## What CQBench measures — and what it does not

It measures, without executing anything:

- **structural validity** — does the output parse, contain the requested
  function with the requested arity, and is it more than a stub?
- **non-degeneracy** — is it at least 10% of the human reference's size?
- **defects** — Pylint findings mapped to six ODC categories
- **vulnerabilities** — Semgrep findings mapped to CWE classes and severity

The headline endpoint `clean_strict@1` requires *all* of these at once.

It does **not** measure functional correctness, semantic equivalence, or actual
exploitability. Nothing is run against tests. A perfectly clean output can be
completely wrong.

## Why the runner is fast

`python -m cqbench evaluate` scores one task per process pair. Measured on a
2-core machine: **~33 s per task**, i.e. about nine hours for 200 tasks × 5
authors. Nearly all of it is overhead, not analysis — semgrep re-parses 1,847
rules every time and makes a ~25 s network version check.

`cqhandson.runner` moves the process boundary: one semgrep run per author, one
pylint run per 200 files, structural analysis in a process pool. Same rules,
same exclusions, same de-duplication — **1,000 evaluations in ~100 s**.

Notebook 02 proves the equivalence rather than asserting it: it diffs every
scored field of the fast runner against `results/reference_check/`, produced by
the stock evaluator. The expected answer is zero differences.

## What the session turns up

Not a scripted tour — these come out of the data, and the exercises are built so
participants find them rather than being told:

- **A composite metric can collapse for the wrong reason.** ChatGPT's
  `clean_strict@1` of 0.5% is mostly an *instruction-following* failure: on 86.5%
  of tasks it wrote a function with a different name. Decompose before ranking.
- **Not every counted defect was authored.** `too-many-arguments` fires
  identically for the human and Claude — both inherit it from the requested
  signature. `unused-argument` is inflated for everyone because methods are
  scored outside their class. Four such symbols account for 35–49% of every
  author's findings; removing them leaves the *comparisons* between authors
  intact.
- **Ratio of means ≠ mean of ratios.** Claude's size relative to the human is
  0.99 one way and 1.38 the other. The exercise resolves it: every model is
  verbose on short functions and compressed on long ones.
- **A headline difference can rest on one permissive rule.** Drop `B404` (which
  fires on `import subprocess`) and Claude's overall vulnerability gap against
  the human stops being significant — while the high-severity gap survives.
- **A pipeline can reproduce perfectly until it meets a dependency nobody
  pinned.** See `NOTES-FOR-THE-CQBENCH-AUTHORS.md`.

## Licence and attribution

`vendor/cqbench-v1/` is the CQBench v1 artifact, unmodified, under **GPL-3.0**.
This repository inherits that licence. The benchmark, dataset and paper are the
work of the CQBench authors; the full source dataset lives on Zenodo
([10.5281/zenodo.21282648](https://doi.org/10.5281/zenodo.21282648)).
