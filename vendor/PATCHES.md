# Patches applied to the vendored CQBench v1

`vendor/cqbench-v1/` is the CQBench v1 artifact. It carries exactly one change
from the released version, listed here so anyone can diff it against upstream
and see what moved.

## 1. Semgrep de-duplication key — `cqbench/analyzers.py`

**What changed.** The third component of the key that decides when two Semgrep
findings are the same finding:

```python
# released v1
key = (normalized_cwes(cwes), severity, extra["lines"].strip())

# here
key = (normalized_cwes(cwes), severity, (start["line"], start["col"]))
```

**Why.** `extra["lines"]` holds the matched source text, and Semgrep redacts it
to the literal string `"requires login"` for registry-sourced rules whenever the
CLI is unauthenticated. The frozen ruleset is registry-resolved, so on a fresh
install that field is the same constant for **100% of findings** — verified
across all five authors on the 200-task Python subset. A constant cannot
discriminate, so the key silently degrades from a triple to the pair
`(class, severity)` and distinct findings in the same file collapse into one.

The start position is always present and does not depend on authentication
state. `extra["fingerprint"]` would work equally well.

**Effect.** The released key misses between a quarter and a third of the
findings. Corrected counts land close to the reference results the benchmark
ships in `data/frozen/`:

| author | released key | position key | `data/frozen/` |
|---|---:|---:|---:|
| Human | 39 | 53 | 52 |
| ChatGPT | 124 | 180 | 177 |
| DeepSeek-Coder | 144 | 213 | 208 |
| Qwen2.5-Coder | 112 | 177 | 162 |

**What is unaffected.** Every incidence rate and `clean_strict@1` are identical
to four decimal places, for every author, under both keys — collapsing findings
cannot turn a task that had findings into a task that had none. Only unbounded
counts move. `cqhandson/runner.py` carries the same key so the two evaluators
stay comparable, and `results/reference_check/` was regenerated with the patched
CLI.

Notebook `02_reproduce_the_study.ipynb` §5 walks through the diagnosis.
