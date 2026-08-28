# %% [markdown]
# # 02 — Getting a trustworthy baseline
#
# **Time:** ~25 minutes · **You run cells and read charts. No code to write.**
#
# Before measuring anything new, we make sure **this machine** — your laptop,
# your analyzer versions, your operating system — produces the same numbers as
# the reference pipeline. Everything in notebook 03 depends on that.
#
# This is not a formality. Static analysis is exquisitely sensitive to its
# environment, and a result you cannot trace back to a verified setup is a
# result you cannot report.
#
# We score the **four authors CQBench ships as baselines** — the human reference
# and the three models that built the benchmark — and check our per-task output
# against the reference results the benchmark distributes.
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
# from the reference CLI in `vendor/` on an 8-task slice; we score the same
# slice and diff every field.

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
# ## 4. Does this machine agree with the reference?
#
# `data/frozen/` holds the **reference per-task results** that ship with the
# benchmark for these same 200 tasks. They were produced by the study's own
# pipeline, so comparing against them answers a practical question — *is my
# environment sound?* — and it is the deepest check available, far beyond what
# `verify_setup.py` does.

# %%
agreement = ch.reproduce.agreement_table()
display(agreement.round(3))
figures.plot_reproduction(agreement);

# %% [markdown]
# **This machine is sound.** Structural verdicts agree on 100% of tasks. Defect
# counts agree on 99–100%. Vulnerability counts agree on 95% of tasks or more.
# And `clean_strict@1` — the headline metric — is **identical to the decimal**
# for all four authors. Pinned analyzer versions plus a frozen ruleset plus a
# fixed mapping do their job.
#
# Note what the residual is and is not. The reference numbers come from a
# *separate implementation* of the same measurement, so a few percent of
# disagreement on unbounded counts is what two independent pipelines agreeing
# looks like. Not one rate moves.
#
# That distinction — counts drift, rates hold — is not a coincidence, and the
# next section is about why.

# %% [markdown]
# ## 5. One line that moved a quarter of the counts
#
# The evaluator has to decide when two Semgrep findings are "the same finding"
# before it counts them. CQBench v1 as released keyed on the triple
# `(weakness class, severity, matched source text)`.
#
# Read that key as a reviewer would, and ask the only question that matters:
# **does every component actually discriminate?** The first two are coarse — many
# distinct findings share a class and a severity. So the third one is carrying
# the whole key. Here is what it holds:

# %%
display(ch.reproduce.matched_text_values().to_frame())

# %% [markdown]
# **One value. Every finding says `"requires login"`.**
#
# Semgrep redacts the matched source text for registry-sourced rules unless the
# CLI is authenticated, and the frozen ruleset is registry-resolved. On any
# fresh install that field is a **constant** — and a constant discriminates
# nothing. The key silently degrades from a triple to a pair, and every finding
# sharing a class and a severity in the same file collapses into one.
#
# This repository ships the one-line fix: key on the finding's **source
# position**, which is always present and never redacted. Both keys can be
# counted from results already on disk, so we can see exactly what the fix did.

# %%
effect = ch.reproduce.dedup_effect()
display(effect)
figures.plot_dedup_effect(effect);

# %% [markdown]
# **Look at the two panels together, because the pair is the point.**
#
# On the left, the fix moved every count: the released key was missing between
# a quarter and a third of the findings. On the right, it moved
# nothing at all — not one incidence rate, and not `clean_strict@1`. That is not
# luck. Collapsing three findings into one cannot turn a task that *had*
# findings into a task that had none, so anything phrased as *"on what fraction
# of tasks"* is immune to the entire question, while anything phrased as *"how
# many"* is not.
#
# > Report "model X produced N vulnerabilities" and you are reporting a number
# > that depends on a de-duplication key your reader never sees — and, as it
# > turns out, on whether you were logged in. Report "model X had at least one
# > finding on Y% of tasks" and you are not.
#
# **And the lesson generalises far past CQBench.** The released key is a
# perfectly reasonable design that behaves exactly as intended on an
# authenticated machine. What breaks it is a **hidden environmental
# dependency**: the output turns on a login state that appears in no version
# pin, no checksum, no container image. Everything that *was* pinned matched
# perfectly. The one thing nobody thinks to pin is the one that moved.
#
# Ask it of your own pipeline before you publish from it: *does any tool in here
# behave differently when logged in?* Reproducibility checklists ask about
# versions, seeds and data. They rarely ask that.
#
# ---
# ## Takeaways
#
# 1. **Three of these five authors built the benchmark.** Their rates are a
#    ceiling, not a measurement.
# 2. **Diff your tooling against a reference before you trust it.** Ours matches
#    on every field; that is what makes the next notebook's numbers usable.
# 3. **Bounded rates are robust; unbounded counts are not.** Prefer summaries
#    that survive a reasonable disagreement about how findings are counted.
# 4. **Pin more than versions.** A login state moved a number by a third.
#
# Next: `03_evaluate_a_new_model.ipynb` — a model the benchmark has never seen.
