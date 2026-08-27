# Two findings from building this hands-on

Both surfaced while preparing the session, both are reproducible from the
notebooks. Neither affects the paper's reported *rates*; one affects its
reported *counts*.

## 1. The Semgrep de-duplication key degrades when the CLI is not logged in

`cqbench/analyzers.py :: analyze_semgrep` de-duplicates findings on

```python
key = (normalized_cwes(cwes), severity, extra["lines"].strip())
```

`extra["lines"]` is redacted by Semgrep to the literal string `"requires login"`
for registry-sourced rules when the CLI is unauthenticated. Since the frozen
ruleset is registry-resolved, **100% of findings carry that constant** on a fresh
install — we checked all 496 kept findings across five authors on the 200-task
Python subset.

The key therefore degrades to `(CWE class, severity)`, collapsing distinct
findings in the same file. Re-running the identical scan and de-duplicating on
the finding's source position instead:

| author | as released | by source position | study's frozen results |
|---|---:|---:|---:|
| Human | 39 | 53 | 52 |
| ChatGPT | 124 | 180 | 177 |
| DeepSeek-Coder | 144 | 213 | 208 |
| Qwen2.5-Coder | 112 | 177 | 162 |

Position-based de-duplication lands within a few percent of the published
numbers; the released key undercounts by 25–30%.

**What is unaffected:** every incidence rate, and `clean_strict@1`, are identical
to the frozen results to the decimal for all four authors — collapsing findings
cannot turn a task with findings into a task without any. Only unbounded counts
move.

**Suggested fix:** use `(normalized_cwes, severity, start.line, start.col)`, or
`extra["fingerprint"]`, neither of which depends on authentication state.
`benchmark/README.md` could also note that `semgrep login` changes reported
counts.

Notebook `02_run_the_benchmark.ipynb` §5 walks through the diagnosis and ships
the verification code.

## 2. `python -m cqbench evaluate` pays a ~25 s network round trip per task

`_run` sets `SEMGREP_SETTINGS_FILE` and the XDG variables but not
`SEMGREP_ENABLE_VERSION_CHECK`, so every invocation checks for a newer Semgrep
release. Measured on a 2-core machine: **31.5 s per task with the check, 6.5 s
without** — and semgrep is invoked once per task.

Adding `"SEMGREP_ENABLE_VERSION_CHECK": "0"` to the environment dict in `_run`
(or `--disable-version-check` to the argument list) is a one-line change worth
about a 5× speed-up, with byte-identical output — we verified field-by-field on
an 8-task slice covering `nontrivial`, `target_missing`, `arity_mismatch` and
`parse_error`.

Batching the analyzers on top of that (one semgrep process per submission set,
one pylint process per 200 files with `--disable=duplicate-code`) takes the
full 200-task × 5-author run from an estimated ~9 hours to **~100 seconds**,
again with identical output. `cqhandson/runner.py` is the implementation, and
notebook 02 diffs it against `results/reference_check/`, produced by the stock
CLI.
