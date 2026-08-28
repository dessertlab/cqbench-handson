# %% [markdown]
# # 00 — Setup, and what a benchmark task actually is
#
# **Session:** Benchmarking the Quality of AI Code Generators · LLMA4SE 2026
# **Time:** ~15 minutes
#
# By the end of this notebook you will have:
#
# 1. confirmed your environment can run both static analyzers,
# 2. looked at one benchmark task and the five answers to it,
# 3. understood *why these particular 200 tasks are in the benchmark* — which
#    turns out to constrain every claim you can make from the results.
#
# No code is generated during this session. Every answer already exists on disk.
# The subject of the session is **measurement**.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # so `import cqhandson` works from notebooks/

import pandas as pd
import cqhandson as ch
from cqhandson import paths, runner

ch.style()
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)

print(f"repository : {paths.REPO}")
print(f"analyzers  : {runner.check_analyzers()}")

# %% [markdown]
# If the line above raised, you have not activated the environment:
#
# ```bash
# conda activate cqbench-handson
# ```
#
# and then restart the kernel. If that still fails, run `python setup/verify_setup.py`
# from a terminal — it says exactly which piece is missing.
#
# > **Those two version numbers are data, not trivia.** `pylint 3.3.6` and
# > `semgrep 1.120.0` are the versions the study ran. Pylint adds and retires
# > checks between releases; Semgrep ships new rules. Run this benchmark with
# > different analyzer versions and you get different numbers for the same code.
# > A static-analysis benchmark is only reproducible if the analyzers are pinned —
# > which is why `environment.yml` pins them and why CQBench also freezes its
# > Semgrep ruleset into a file instead of pulling it from the registry.

# %% [markdown]
# ## 1. The tasks
#
# A CQBench task is a *code-generation request derived from real code*: someone's
# actual function was taken out of an open-source repository, its docstring
# became the specification, and its signature became the contract.

# %%
tasks = ch.load_tasks()
references = ch.load_references()

print(f"{len(tasks)} tasks, all Python\n")

TASK = "python:gp206544"          # our running example for this notebook and the next
task = tasks[TASK]

print("task_id   :", task["task_id"])
print("stratum   :", task["stratum"])
print("signature :", task["signature"]["text"])
print("arity     :", task["signature"]["arity"])
print("docstring :", task["docstring"])

# %% [markdown]
# The task is handed to a model as a single canonical prompt — the same prompt
# for every model, with no extra instructions, no examples, no retries:

# %%
print(task["prompt"])

# %% [markdown]
# ## 2. The five answers
#
# Five authors answered this prompt. One of them is the human who wrote the
# original code; the other four are models. Read them before you look at any
# number — the whole session is about whether a static analyzer's verdict
# matches yours.

# %%
for author in ch.AUTHOR_ORDER:
    ch.show_code(author, TASK)

# %% [markdown]
# ### What did you just read?
#
# Take thirty seconds and rank them yourself before scrolling on. Some things
# worth noticing:
#
# * **ChatGPT did not answer the question asked.** The prompt says
#   `def install_key(self, key_data)`; it produced
#   `install_untrusted_repo_signing_key(key_url)` — a different name, a different
#   parameter, and a different job (it downloads a key from a URL). It is not
#   *bad code*; it is *not the requested function*. Hold on to that distinction:
#   it will dominate one of the results later.
# * **Three of the five shell out to `apt-key`.** The human used a Python GPG
#   binding instead. That difference is exactly the kind of thing a security
#   analyzer notices.
# * **Claude wrote by far the most defensive version** — it validates the input,
#   handles both a path and raw key material, checks the return code, and raises
#   with the captured stderr. It is also four times longer than everything else.
#   Keep that in mind when we count defects in notebook 04.

# %% [markdown]
# ## 3. The human reference
#
# Every task ships with measurements of its *original human implementation*.
# They are not there for scoring the human — they are the denominator that makes
# "is this generated function suspiciously small?" a well-defined question.

# %%
reference = references[TASK]
print("structural reference:")
for key, value in reference["human_metrics"].items():
    print(f"    {key:24s} {value}")
print("\ncomplexity reference:", reference["human_complexity"])

# %% [markdown]
# ## 4. Why *these* 200 tasks — the most important slide of the session
#
# CQBench is a **failure-derived challenge set**. Tasks were not sampled at
# random from a corpus of programming problems. Out of ~256,000 candidate Python
# tasks, one was kept only if:
#
# 1. at least **two** of the three original models produced a
#    complexity-qualified answer (not empty, not a stub), **and**
# 2. each of those two answers had **≥ 3 analyzer findings**, **and**
# 3. those findings **shared a class** — the same ODC defect type, or the same
#    normalized CWE.
#
# That third condition is the interesting one. It does not select *hard* tasks;
# it selects tasks where independent models **fail in the same way**. The
# resulting strata are named after which gate fired:

# %%
strata = pd.Series([t["stratum"] for t in tasks.values()]).value_counts()
display(strata.rename("tasks").to_frame().assign(
    share=lambda d: (100 * d["tasks"] / d["tasks"].sum()).round(1)))

difficulty = pd.Series([t["difficulty"]["score"] for t in tasks.values()])
print("\nconsensus finding burden (the 'difficulty' score):")
print(difficulty.value_counts().sort_index().to_string())

# %% [markdown]
# ### And now the part that decides how you read every later number
#
# Those gates were computed from **three specific models**. So the five authors
# you met in section 2 are not five peers — they stand in three different
# relations to the benchmark:

# %%
roles = pd.DataFrame({
    "Author": [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER],
    "Role": [ch.ROLE_LABELS[ch.AUTHOR_ROLES[a]] for a in ch.AUTHOR_ORDER],
}).set_index("Author")
display(roles)

# %% [markdown]
# ChatGPT, DeepSeek and Qwen **defined** the selection: a task is here precisely
# because at least two of them failed it in a shared way. Their failure rates on
# this benchmark are inflated by construction — they are a ceiling, not a
# measurement.
#
# The human reference took no part in the consensus gate, and Claude Opus 4.8 was
# released *after* the benchmark was built. Those two are outside the
# construction, which is the only reason anything here can be tested.
#
# Notebook 02 reproduces the study with the four authors it measured; notebook 03
# then submits **Claude as a model under test**. Keep the consequence in view:
#
# > Beating the three models that built the benchmark is the weakest possible
# > result. Reaching the human reference is the one that means something.

# %% [markdown]
# **What follows from this, and what does not.**
#
# ✅ You *can* say: "on tasks where 2023–24 models were known to fail in a shared
# way, model X still fails on Y% of them." That is a robustness statement about
# known issue-prone code, and it is what the benchmark was built to support.
#
# ❌ You *cannot* say: "model X produces defective code Y% of the time." These
# tasks were chosen *because* they produce findings. The base rate here is
# inflated by construction.
#
# The clearest evidence for that: the **human** implementations — real code
# shipped in real repositories — trigger at least one defect on 62% of these 200
# tasks. That is not a claim about human programmers. It is a description of how
# the tasks were selected.
#
# > A benchmark's sampling frame determines which sentences about it are true.
# > Most benchmark misreporting in the literature is a sampling-frame error, not
# > an arithmetic one.

# %% [markdown]
# ## 5. A quick look around
#
# **Exercise (3 minutes).** Pick another task and read its five answers. Some
# suggestions with different flavours:
#
# * `python:gp247196` — a raw HTTP GET; watch what each author does about TLS
#   verification and error handling.
# * `python:gp329939` — reloading functions via `exec`; consensus CWEs are
#   CWE-78 and CWE-95.
# * `python:gp280603` — no security angle at all, purely a defect-consensus task.

# %%
OTHER = "python:gp247196"     # <- change this and re-run

print(tasks[OTHER]["signature"]["text"])
print(tasks[OTHER]["docstring"], "\n")
for author in ch.AUTHOR_ORDER:
    ch.show_code(author, OTHER)

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Pinned analyzer versions are part of the benchmark definition.** Change
#    them and the measurements change.
# 2. **A task is a signature + a specification.** Answering a *different*
#    question is a distinct failure mode from answering it badly — and a
#    benchmark has to decide how to treat that.
# 3. **CQBench is failure-derived.** Its rates describe robustness on known
#    issue-prone tasks. They are not population estimates.
#
# Next: `01_measurement_pipeline.ipynb` — what the four tools do, and how a raw
# analyzer message becomes a number in a table.
