# %% [markdown]
# # 02 — Reproducing the study
#
# **Time:** ~30 minutes · **You run cells and read charts. No code to write.**
#
# Before testing anything new, we check that our machine agrees with the
# published research. That is the whole job of this notebook, and it is not a
# formality: a re-implementation that has never been diffed against the original
# is a re-implementation you cannot cite.
#
# We score the **four authors the study itself measured** — the human reference
# and the three models that built the benchmark — and compare our per-task
# results with theirs.
#
# Claude does not appear here. It arrives in notebook 03, as a submission the
# benchmark has never seen.

# %%
import sys, pathlib, time, os, subprocess
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import cqhandson as ch
from cqhandson import paths, runner, figures

ch.style()
pd.set_option("display.width", 200)
print(runner.check_analyzers())

# %% [markdown]
# ## 1. Who is in this benchmark, and in what capacity
#
# **The five authors are not five peers.** CQBench kept a task only when at
# least two of ChatGPT, DeepSeek-Coder and Qwen2.5-Coder produced three or more
# findings of a shared class. Those three *defined* the selection: their failure
# rates here are inflated by construction. They are a ceiling.
#
# The human reference never entered the consensus gate, and Claude Opus 4.8 was
# released after the benchmark was built. Both are outside the construction —
# which is the only reason the benchmark can test anything.

# %%
display(pd.DataFrame({
    "Author": [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER],
    "Role": [ch.ROLE_LABELS[ch.AUTHOR_ROLES[a]] for a in ch.AUTHOR_ORDER],
}).set_index("Author"))

# %% [markdown]
# Every chart in this session colours by that role. Aqua is the reference, blue
# is "built the benchmark", orange is the model under test. You will not have to
# remember which is which.

# %% [markdown]
# ## 2. Why this used to take nine hours
#
# `python -m cqbench evaluate` starts a fresh `pylint` and a fresh `semgrep` for
# **every task**. Pylint is cheap. Semgrep is not: it re-parses all 1,847 frozen
# rules each time (~6 s) and, by default, makes a network round trip to check
# for a new release (~25 s). Measured: **~33 s per task**, of which ~32 s is
# overhead.
#
# `cqhandson.runner` moves the process boundary — one semgrep run per author,
# one pylint run per 200 files — and finishes in about 20 seconds per author.
#
# **We do not ask you to trust that.** `results/reference_check/` holds output
# from the stock CLI on an 8-task slice; we score the same slice and diff every
# field.

# %%
for author in ("human", "chatgpt"):
    fast = runner.evaluate(
        paths.DATA / f"reference_check/{author}.jsonl",
        tasks=paths.DATA / "reference_check/tasks.jsonl",
        references=paths.DATA / "reference_check/references.jsonl", verbose=False)
    official = ch.read_jsonl(paths.RESULTS / f"reference_check/{author}.jsonl")
    differences = runner.diff_results(fast, official)
    print(f"{author:8s}  {len(fast)} tasks  ->  {len(differences)} differing fields")
    assert not differences

# %% [markdown]
# Zero differing fields. The speed-up is free.

# %% [markdown]
# ## 3. Score the four authors the study measured
#
# About **80 seconds**.

# %%
started = time.time()
counts = runner.evaluate_all(ch.BASELINES)
print(f"\n{sum(counts.values())} evaluations in {time.time() - started:.0f}s")

# %% [markdown]
# > **If that failed**, nothing downstream breaks: every chart falls back to
# > `results/precomputed/`, produced by exactly this code.

# %%
results = ch.results_frame(ch.BASELINES)
display(ch.results_source(ch.BASELINES).to_frame())
display(ch.headline_table(results).round(1))

# %% [markdown]
# ## 4. Did we reproduce the study?
#
# `data/frozen/` holds the study's **published per-task results** for these same
# 200 tasks, produced by the original research pipeline rather than by the
# released evaluator. This is a real target, not a self-check.

# %%
agreement = ch.reproduce.agreement_table()
display(agreement.round(3))
figures.plot_reproduction(agreement);

# %% [markdown]
# Read it in two parts.
#
# **What reproduces perfectly.** Structural verdicts agree on 100% of tasks.
# Defect counts agree on 99–100%. And `clean_strict@1` — the metric the paper
# leads with — is **identical to the decimal** for all four authors. Pinned
# analyzer versions plus a frozen ruleset plus a fixed mapping do their job.
#
# **What does not.** Vulnerability *counts* agree on only 76–83% of tasks, and
# our totals run about 25% below the study's — same tasks, same weakness
# classes, fewer findings each time. Something is collapsing findings the study
# counted separately.

# %% [markdown]
# ## 5. Finding the discrepancy
#
# The released evaluator de-duplicates Semgrep findings on the triple
# `(weakness class, severity, matched source text)`. Look at what that third
# component actually contains in our results:

# %%
display(ch.reproduce.matched_text_values().to_frame())

# %% [markdown]
# **Every one of the 496 findings says `"requires login"`.**
#
# Semgrep redacts the matched source text for registry rules unless the CLI is
# authenticated. The frozen ruleset was resolved from the registry, so on any
# unauthenticated machine that field is a **constant** — and a constant cannot
# discriminate. The key silently degrades from a triple to a pair, and findings
# that share a class and a severity collapse into one.
#
# The test: re-scan and key on the finding's **source position** instead, which
# is always present.

# %%
display(ch.reproduce.dedup_experiment())

# %% [markdown]
# Confirmed. Keying on position lands within a few percent of the study's own
# numbers for every author; the released key undercounts by 25–30%.
#
# **What kind of bug this is.** Not sloppiness — the key is reasonable and works
# exactly as intended on the machine where the study ran. It is a **hidden
# environmental dependency**: the output depends on an authentication state that
# appears in no version pin, no checksum, no container image. Everything that
# *was* pinned reproduced perfectly. The one thing nobody thought to pin is the
# one that moved.
#
# **And what survived it.** Every incidence rate is unchanged, and so is
# `clean_strict@1`. Collapsing several findings into one cannot turn a task with
# findings into a task without any.
#
# > Report "model X produced N vulnerabilities" and you are reporting a number
# > that depends on a de-duplication key your reader never sees — and, as it
# > turns out, on whether you were logged in. Report "model X had at least one
# > finding on Y% of tasks" and you are not.
#
# The rest of the session uses the released key, unchanged, so our incidence
# rates stay directly comparable to the paper's.
#
# ---
# ## Takeaways
#
# 1. **Three of these five authors built the benchmark.** Their rates are a
#    ceiling, not a measurement.
# 2. **Diff your tooling against the reference implementation.** Ours matches on
#    every field; that is what makes the next notebook's numbers usable.
# 3. **Bounded rates reproduce; unbounded counts do not.** Prefer summaries that
#    survive a reasonable disagreement about how findings are counted.
#
# Next: `03_evaluate_a_new_model.ipynb` — a model the benchmark has never seen.
