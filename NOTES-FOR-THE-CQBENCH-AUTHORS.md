# Two findings from building this hands-on

Both surfaced while preparing the session and both are reproducible from the
notebooks. Neither touches the results reported in the paper.

## 1. The released evaluator's Semgrep de-duplication key degrades silently

**Where.** `cqbench/analyzers.py :: analyze_semgrep`, the path taken by
`python -m cqbench evaluate` when it scores a new submission.

```python
key = (normalized_cwes(cwes), severity, extra["lines"].strip())
```

`extra["lines"]` holds the matched source text and is the only component of the
key that discriminates — many distinct findings share a class and a severity.
Semgrep redacts it to the literal string `"requires login"` for
registry-sourced rules when the CLI is unauthenticated, and the frozen ruleset
is registry-resolved, so on a fresh install **100% of findings carry that
constant** — checked across all five authors on the 200-task Python subset. The
key degrades to `(class, severity)` and distinct findings in the same file
collapse into one.

**This does not affect the paper.** The results the artifact ships in
`benchmark/results/*.jsonl` never pass through this function:
`cqbench/historical.py :: export_historical_results` reads `vulns_total` and
`vulns_high_sev` straight out of the study's parquet tables and stamps
`static_analysis_status: "precomputed_frozen_study_results"`. The published
numbers come from the study pipeline and are independent of the CLI's
authentication state.

**What it does affect** is the artifact's internal consistency: the released
evaluator disagrees with the artifact's own frozen results by 25–30% on counts.
Re-keying on the finding's source position closes most of that gap:

| author | released key | position key | `benchmark/results/` |
|---|---:|---:|---:|
| Human | 39 | 53 | 52 |
| ChatGPT | 124 | 180 | 177 |
| DeepSeek-Coder | 144 | 213 | 208 |
| Qwen2.5-Coder | 112 | 177 | 162 |

**What survives either key.** Every incidence rate, and `clean_strict@1`, are
identical to four decimal places for all five authors under both keys —
collapsing findings cannot turn a task that had findings into a task that had
none. Only unbounded counts move.

**Suggested fix.** One line:

```python
key = (normalized_cwes(cwes), severity, (start["line"], start["col"]))
```

`extra["fingerprint"]` works equally well. Neither depends on authentication
state. `benchmark/README.md` could also note that `semgrep login` changes
reported counts.

This repository vendors CQBench with that fix applied; see `vendor/PATCHES.md`.
Notebook `02_reproduce_the_study.ipynb` §5 walks through the diagnosis.

## 2. `python -m cqbench evaluate` pays a ~25 s network round trip per task

`_run` sets `SEMGREP_SETTINGS_FILE` and the XDG variables but not
`SEMGREP_ENABLE_VERSION_CHECK`, so every invocation checks for a newer Semgrep
release. Measured on a 2-core machine: **31.5 s per task with the check, 6.5 s
without** — and semgrep is invoked once per task.

Adding `"SEMGREP_ENABLE_VERSION_CHECK": "0"` to the environment dict in `_run`
(or `--disable-version-check` to the argument list) is a one-line change worth
about a 5× speed-up, with byte-identical output — verified field-by-field on an
8-task slice covering `nontrivial`, `target_missing`, `arity_mismatch` and
`parse_error`.

Batching the analyzers on top of that (one semgrep process per submission set,
one pylint process per 200 files with `--disable=duplicate-code`) takes the
full 200-task × 5-author run from an estimated ~9 hours to **~150 seconds**,
again with identical output. `cqhandson/runner.py` is the implementation, and
notebook 02 diffs it against `results/reference_check/`, produced by the
reference CLI.
