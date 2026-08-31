# Patches applied to the vendored evaluator

`vendor/cqbench-v1/` is the CQBench evaluator, carrying one change from the
version published with the paper. It is listed here so that anyone diffing the
two can see what moved, and why.

## `cqbench/analyzers.py` — Semgrep de-duplication key

The evaluator decides when two Semgrep findings are the same finding before it
counts them. The third component of that key is the only one that discriminates,
since many distinct findings share a weakness class and a severity.

```python
# published version
key = (normalized_cwes(cwes), severity, extra["lines"].strip())

# here
key = (normalized_cwes(cwes), severity, (start["line"], start["col"]))
```

`extra["lines"]` holds the matched source text, and Semgrep redacts it to the
literal string `"requires login"` for registry-sourced rules whenever the CLI is
unauthenticated. The vendored ruleset is registry-resolved, so on a fresh
install that field is the same constant for every finding: the key degrades to
`(class, severity)` and distinct findings in the same file collapse into one.

The start position is always present and does not depend on authentication
state; `extra["fingerprint"]` would work equally well.

Counts rise by a quarter to a third. Nothing else moves: every incidence rate
and `clean_strict@1` are identical to four decimal places under both keys,
because collapsing findings cannot turn a task that had findings into a task
that had none.

`cqhandson/runner.py` carries the same key, so the two evaluators stay
comparable.
