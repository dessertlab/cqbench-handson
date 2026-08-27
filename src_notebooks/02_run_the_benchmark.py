# %% [markdown]
# # 02 — Submitting a model to CQBench
#
# **Time:** ~30 minutes
#
# This is the notebook where you *use* the benchmark, following the flow a real
# CQBench user follows:
#
# > **validate** the submission → **evaluate** it → **compare** it with the
# > shipped baselines.
#
# Our submission is **Claude Opus 4.8**. Its 200 answers already exist in
# `data/predictions/claude.jsonl`; treat them as the output of a model you just
# ran. The other four authors are the baselines you score it against.
#
# Along the way we will see why the stock evaluator would take about **nine
# hours** for this, verify that our faster runner is identical rather than
# assuming it, and reproduce a published table.

# %%
import sys, pathlib, time, json, os, subprocess, tempfile
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths, runner
from cqhandson.metrics import compare_submission

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print(runner.check_analyzers())
print("CPU cores available:", os.cpu_count())

# %% [markdown]
# ## 1. Who built this benchmark, and who is being tested
#
# **Read this before any number.** The five authors are not five peers.
#
# CQBench kept a task only when **at least two of ChatGPT, DeepSeek-Coder and
# Qwen2.5-Coder** produced three or more findings of a shared class. Those three
# models *defined* the selection. Their failure rates on this benchmark are
# inflated by construction and are not estimates of anything — they are the
# ceiling, and they are there to give the other two a scale.
#
# The other two took no part in it:
#
# * the **human reference** only had to parse and be non-trivial; its findings
#   never entered the consensus gate;
# * **Claude Opus 4.8** was released *after* the benchmark was built.
#
# That asymmetry is the whole reason the benchmark can test anything. It is
# recorded in the data, not just in this paragraph:

# %%
roles = pd.DataFrame({
    "Author": [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER],
    "Role": [ch.ROLE_LABELS[ch.AUTHOR_ROLES[a]] for a in ch.AUTHOR_ORDER],
}).set_index("Author")
display(roles)

print(f"submission : {ch.AUTHOR_LABELS[ch.SUBMISSION]}")
print(f"baselines  : {', '.join(ch.AUTHOR_LABELS[b] for b in ch.BASELINES)}")

# %% [markdown]
# Keep the consequence in mind for the rest of the session:
#
# > **Beating the three construction models is the weakest possible result.**
# > They defined the tasks. Reaching the *human reference* is the result that
# > means something.

# %% [markdown]
# ## 2. Why the reference evaluator is slow
#
# `python -m cqbench evaluate` scores **one task per process pair**. For each
# task it starts a fresh `pylint` and a fresh `semgrep`. Pylint is cheap
# (~0.5 s). Semgrep is not, for two reasons that have nothing to do with your
# code:
#
# * it parses all **1,847 frozen rules** from scratch on every invocation (~6 s), and
# * by default it makes a **network round trip** to check for a newer release
#   (~25 s on the reference machine).
#
# Measured cost: **~33 s per task**, of which ~32 s is overhead. Times 1,000
# evaluations, that is about nine hours — for maybe ten minutes of actual
# analysis.
#
# Time it yourself on a single file if you want to feel it:

# %%
code = ch.load_predictions("claude")["python:gp206544"]

with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / "one.py"
    target.write_text(code, encoding="utf-8")

    environment = dict(os.environ)          # note: version check NOT disabled
    environment.pop("SEMGREP_ENABLE_VERSION_CHECK", None)

    started = time.time()
    subprocess.run(["semgrep", "scan", "--config", str(paths.SEMGREP_RULES),
                    "--json", "--metrics", "off", str(target)],
                   capture_output=True, text=True, env=environment)
    print(f"semgrep, one file, defaults          : {time.time() - started:5.1f}s")

    started = time.time()
    subprocess.run(["semgrep", "scan", "--config", str(paths.SEMGREP_RULES),
                    "--json", "--metrics", "off", "--disable-version-check", str(target)],
                   capture_output=True, text=True, env=environment)
    print(f"semgrep, one file, no version check  : {time.time() - started:5.1f}s")

# %% [markdown]
# ## 3. Moving the process boundary — and proving it is safe
#
# `cqhandson.runner` does exactly the same analysis with a different process
# layout:
#
# | step | reference evaluator | here |
# |---|---|---|
# | semgrep | one process per task | **one process per author** (a whole directory) |
# | pylint | one process per task | **one process per 200 files** |
# | lizard / AST | serial | process pool |
# | version check | ~25 s per task | disabled |
#
# Two things have to be true for that to be legitimate, and both are checkable:
#
# * **Semgrep** reports the originating `path` on every finding *and* every
#   error, so per-file attribution survives batching — including the study's rule
#   that a file which produced a scan error contributes zero findings.
# * **Pylint** is batched with `--disable=duplicate-code`. R0801 is the only
#   check that compares *across* modules; it can never fire when files are linted
#   one at a time, so disabling it makes a batch identical to per-file runs.
#
# `results/reference_check/` holds the output of the **stock** `cqbench evaluate`
# on an 8-task slice (chosen to cover `nontrivial`, `target_missing`,
# `arity_mismatch` and `parse_error`). We score the same slice with the fast
# runner and diff **every scored field**.

# %%
differences = {}
for author in ("human", "chatgpt"):
    fast = runner.evaluate(
        paths.DATA / f"reference_check/{author}.jsonl",
        tasks=paths.DATA / "reference_check/tasks.jsonl",
        references=paths.DATA / "reference_check/references.jsonl",
        verbose=False,
    )
    official = ch.read_jsonl(paths.RESULTS / f"reference_check/{author}.jsonl")
    differences[author] = runner.diff_results(fast, official)
    print(f"{author:8s}  {len(fast)} tasks  ->  {len(differences[author])} differing fields")

assert not any(differences.values()), differences
print("\nIdentical. The speed-up is free.")

# %% [markdown]
# ## 4. Step one — validate the submission
#
# Before running any analyzer, CQBench checks the submission file itself: one
# JSON object per task, `task_id` as the only join key, no unknown ids, no
# duplicates, `code` always a string. This is the stock CLI, unchanged.

# %%
validate = subprocess.run(
    [sys.executable, "-m", "cqbench", "validate-submission",
     "--tasks", str(paths.TASKS),
     "--predictions", str(paths.PREDICTIONS / f"{ch.SUBMISSION}.jsonl")],
    cwd=paths.VENDOR, capture_output=True, text=True,
    env=dict(os.environ) | {"SEMGREP_ENABLE_VERSION_CHECK": "0"},
)
print(validate.stdout.strip() or validate.stderr.strip())

# %% [markdown]
# `{"tasks": 200, "predictions": 200}` — every task answered.
#
# Note what validation does **not** do: it never looks at the code. A file of 200
# empty strings validates fine. Missing predictions are also legal — they stay in
# the denominator and are scored as empty output. The benchmark refuses to let
# you quietly shrink your own test set.

# %% [markdown]
# ## 5. Step two — evaluate the submission
#
# 200 tasks, one author. Expect **20–30 seconds**.

# %%
started = time.time()
rows = runner.evaluate(
    paths.PREDICTIONS / f"{ch.SUBMISSION}.jsonl",
    output=paths.LIVE / f"{ch.SUBMISSION}.jsonl",
)
print(f"\n{len(rows)} tasks scored in {time.time() - started:.0f}s")

# %% [markdown]
# That is the whole submission flow. If you had generated 200 completions from
# your own model this morning, you would now be done.
#
# We score the four baselines too, because we want the comparison on *this*
# machine with *these* analyzer versions rather than trusting a shipped file.

# %%
started = time.time()
counts = runner.evaluate_all(ch.BASELINES, jobs=None)
elapsed = time.time() - started
total = sum(counts.values()) + len(rows)

print(f"\n{total} evaluations in total, {elapsed:.0f}s for the baselines")
print(f"the stock evaluator would have taken about {total * 33 / 3600:.0f} hours")

# %% [markdown]
# > **If the cells above failed** — no analyzers, no disk, no time — everything
# > downstream still works. `ch.results_frame()` falls back to
# > `results/precomputed/`, which was produced by exactly this code.

# %%
results = ch.results_frame()          # live results if present, else the fallback
print("source:", paths.results_dir().relative_to(paths.REPO))
print(f"{len(results)} rows = {results['task_id'].nunique()} tasks "
      f"× {results['author'].nunique()} authors")
results.head(3)[["author", "role", "task_id", "status", "defects_total", "vulns_total"]]

# %% [markdown]
# ## 6. The scoreboard
#
# * **Defective** — share of tasks with ≥1 Pylint finding that maps to ODC
# * **Vulnerable** — share with ≥1 CWE-carrying Semgrep finding
# * **High sev.** — share with ≥1 `CRITICAL`/`ERROR` finding
# * **Clean** — `clean_strict@1`: all four layers passed
#
# Read the `Role` column first. It is the most important column in the table.

# %%
headline = ch.headline_table(results)
display(headline.round(1))

# %% [markdown]
# The submission clears the clean-code bar on **27.5%** of these tasks. In
# isolation that number says nothing at all — it needs the scale the other rows
# provide:
#
# * against the three models that **built** the benchmark (0.5%, 2.5%, 5.5%) it
#   looks transformative, and that comparison is nearly worthless: those tasks
#   were selected *because* those models failed them;
# * against the **human reference** (31.5%) it is slightly behind, and that is
#   the comparison that carries information.
#
# Notice also that the submission is the only author with no structural failures
# at all, and that Qwen accumulates 649 findings against the human's 284 on the
# same 200 tasks.

# %% [markdown]
# ## 7. Step three — compare, with the shipped tool
#
# CQBench ships `cqbench compare` for exactly this. It merges the submission
# with each baseline **task by task**, then bootstraps the paired difference
# 10,000 times with a per-comparison seed.
#
# Pairing is the point. Every author answered the *same* 200 tasks, so we can
# subtract per task and average the differences, instead of comparing two
# independent percentages. That removes task difficulty from the comparison and
# gives far tighter intervals.

# %%
comparison_path = paths.RESULTS / "comparison.csv"
compare = subprocess.run(
    [sys.executable, "-m", "cqbench", "compare",
     "--submission", str(paths.results_dir() / f"{ch.SUBMISSION}.jsonl"),
     *sum([["--baseline", str(paths.results_dir() / f"{b}.jsonl")] for b in ch.BASELINES], []),
     "--output", str(comparison_path)],
    cwd=paths.VENDOR, capture_output=True, text=True,
    env=dict(os.environ) | {"SEMGREP_ENABLE_VERSION_CHECK": "0"},
)
print(compare.stdout.strip() or compare.stderr.strip()[-500:])

table = pd.read_csv(comparison_path)
table["role"] = table["baseline"].map(
    lambda b: ch.ROLE_SHORT[ch.AUTHOR_ROLES[b]])
table["baseline"] = table["baseline"].map(ch.AUTHOR_LABELS)
display(table[table["metric"] == "clean_strict_at_1"]
        [["baseline", "role", "submission", "baseline_value", "delta", "ci_lo", "ci_hi"]]
        .round(3))

# %% [markdown]
# **How to read a row.** `delta` is the submission's rate minus the baseline's,
# averaged over the 200 paired tasks. If the interval `[ci_lo, ci_hi]` excludes
# zero, the difference is not explained by which tasks happened to be sampled.
#
# Against the human reference: **−0.04, interval [−0.11, +0.03]**. It contains
# zero, so on the headline metric the submission is **statistically
# indistinguishable from the human reference** — the first model in this
# comparison for which that is true.

# %%
for metric in ("strict_nontrivial_rate", "defect_free_rate",
               "vulnerability_free_rate", "high_severity_free_rate"):
    result = compare_submission(results, metric, submission=ch.SUBMISSION)
    print(f"\n{metric}  ({ch.AUTHOR_LABELS[ch.SUBMISSION]} minus each baseline, paired)")
    print(result.round(3).to_string(index=False))

# %% [markdown]
# Ignore the three `built it` rows — the submission beats them everywhere, as it
# must. The `reference` row is the result:
#
# | | vs the human reference |
# |---|---|
# | structural validity | **level** (both 100%) |
# | defect-free | **level** — interval contains zero |
# | vulnerability-free | **worse**, significantly |
# | high-severity-free | **worse**, significantly |
#
# A frontier model has closed the structural and maintainability gap to human
# code on these tasks, and has **not** closed the security gap. That is the
# paper's headline, reproduced from raw code on your laptop in two minutes, and
# the next three notebooks ask why.

# %% [markdown]
# ## 8. Did we reproduce the paper?
#
# The paper reports, for Python on exactly these 200 tasks:
#
# | Author | Defective | Vulnerable | High sev. | Clean | Total defects |
# |---|---:|---:|---:|---:|---:|
# | Human | 62.0 | 15.5 | 1.5 | 31.5 | 284 |
# | Claude Opus 4.8 | 63.0 | 28.0 | 7.5 | 27.5 | 239 |

# %%
paper = pd.DataFrame(
    {"Defective %": [62.0, 63.0], "Vulnerable %": [15.5, 28.0],
     "High sev. %": [1.5, 7.5], "Clean %": [31.5, 27.5], "Total defects": [284, 239]},
    index=["Human", "Claude Opus 4.8"])

ours = headline.loc[paper.index, paper.columns]
display(pd.concat({"paper": paper, "ours": ours.round(1),
                   "difference": (ours - paper).round(1)}, axis=1)
          .swaplevel(axis=1).sort_index(axis=1))

# %% [markdown]
# ## 9. A second cross-check: the study's own frozen results
#
# `data/frozen/` contains the study's *published* per-task results for the four
# original authors — computed by the original research pipeline, not by the
# released evaluator. Comparing them with what we just measured tells us
# something the paper cannot tell us about itself.

# %%
pairs = {"human": "human", "chatgpt": "openai", "dsc": "dsc", "qwen": "qwen"}
rows = []
for ours_name, frozen_name in pairs.items():
    mine = pd.json_normalize(ch.load_results(ours_name)).set_index("task_id").sort_index()
    theirs = pd.json_normalize(
        ch.read_jsonl(paths.FROZEN / f"{frozen_name}.jsonl")).set_index("task_id").sort_index()
    clean = lambda d: (d["strict_nontrivial"] & d["defects_total"].eq(0)
                       & d["vulns_total"].eq(0)).mean()
    rows.append({
        "author": ch.AUTHOR_LABELS[ours_name],
        "structural agree": (mine["strict_nontrivial"] == theirs["strict_nontrivial"]).mean(),
        "defect count agree": (mine["defects_total"] == theirs["defects_total"]).mean(),
        "vuln count agree": (mine["vulns_total"] == theirs["vulns_total"]).mean(),
        "vulnerable % ours": 100 * (mine["vulns_total"] > 0).mean(),
        "vulnerable % theirs": 100 * (theirs["vulns_total"] > 0).mean(),
        "total vulns ours": int(mine["vulns_total"].sum()),
        "total vulns theirs": int(theirs["vulns_total"].sum()),
        "clean@1 ours": 100 * clean(mine),
        "clean@1 theirs": 100 * clean(theirs),
    })
display(pd.DataFrame(rows).set_index("author").round(2))

# %% [markdown]
# * **Structural verdicts: 100% agreement.** Deterministic parsing reproduces.
# * **Defect counts: 99–100% agreement.** Pinned Pylint + a frozen mapping
#   reproduces.
# * **`clean_strict@1`: identical to the decimal** for all four authors.
# * **Vulnerability *counts*: 76–83% agreement**, and our totals are ~25% lower
#   throughout — same tasks, same CWE classes, fewer findings each time.
#
# Same rules, same Semgrep version, same code. Something in the pipeline is
# collapsing findings that the study counted separately. Let's find it.

# %% [markdown]
# ### Diagnosis: a de-duplication key that depends on being logged in
#
# The released evaluator de-duplicates Semgrep findings on the triple
#
# ```python
# (normalized CWEs, severity, extra["lines"])
# ```
#
# where `extra["lines"]` is the source text Semgrep matched. Look at what that
# field actually contains in our results:

# %%
import collections
matched_text = collections.Counter(
    finding["extra"]["lines"].strip()[:40]
    for author in ch.AUTHOR_ORDER
    for row in ch.load_results(author)
    for finding in row["vulnerability_findings"])
print(f"{sum(matched_text.values())} findings kept across all five authors")
display(pd.Series(matched_text).rename("findings").to_frame())

# %% [markdown]
# **Every single one says `"requires login"`.**
#
# Semgrep redacts the matched source text for registry rules unless the CLI is
# authenticated (`semgrep login`). The frozen ruleset was resolved from the
# registry, so on an unauthenticated machine — yours, ours, and anyone who
# installs the artifact from `pip` — that field is a **constant**.
#
# A constant is not a discriminator. The de-duplication key silently degrades
# from `(CWE, severity, matched text)` to `(CWE, severity)`, and every finding on
# a file that shares a CWE class and severity collapses into one.
#
# Test the diagnosis: re-run the same scan and de-duplicate on the finding's
# **source position** instead — which is always present — and see whether the
# counts move back toward the study's.

# %%
sys.path.insert(0, str(paths.VENDOR))
from support.rq4_build_table import normalized_cwes
from cqhandson.runner import _env, _write_corpus

def recount(author: str) -> dict:
    """Scan one author's corpus once, then count under two de-duplication keys."""
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        names = _write_corpus(ch.load_predictions(author), directory)
        completed = subprocess.run(
            ["semgrep", "scan", "--config", str(paths.SEMGREP_RULES), "--json",
             "--metrics", "off", "--no-git-ignore", "--disable-version-check",
             "--max-target-bytes", "1000000", str(directory)],
            capture_output=True, text=True, env=_env())
        report = json.loads(completed.stdout)

    as_released, by_position = collections.defaultdict(set), collections.defaultdict(set)
    for finding in report.get("results", []):
        extra = finding["extra"]
        cwes = extra.get("metadata", {}).get("cwe")
        if not cwes:
            continue
        filename = os.path.basename(finding["path"])
        classes, severity = normalized_cwes(cwes), str(extra["severity"]).upper()
        as_released[filename].add((classes, severity, extra["lines"].strip()))
        by_position[filename].add((classes, severity,
                                   finding["start"]["line"], finding["start"]["col"]))

    errored = {os.path.basename(str(e.get("path", ""))) for e in report.get("errors", [])
               if not str(e.get("path", "")).startswith("https:/semgrep.dev/...")}
    return {
        "as released (key includes 'lines')":
            sum(len(v) for k, v in as_released.items() if k not in errored),
        "de-duplicated by source position":
            sum(len(v) for k, v in by_position.items() if k not in errored),
    }

rows = []
for ours, frozen in [("human", "human"), ("chatgpt", "openai"),
                     ("dsc", "dsc"), ("qwen", "qwen")]:
    counts = recount(ours)
    counts["study's frozen results"] = sum(
        r["vulns_total"] for r in ch.read_jsonl(paths.FROZEN / f"{frozen}.jsonl"))
    rows.append({"author": ch.AUTHOR_LABELS[ours], **counts})
display(pd.DataFrame(rows).set_index("author"))

# %% [markdown]
# That is the bug, confirmed. De-duplicating by source position lands within a
# few percent of the study's own numbers for every author; the released key
# undercounts by 25–30%.
#
# **What this is an example of.** Not sloppiness — the key is perfectly
# reasonable, and it works exactly as intended on the machine where the study was
# run. It is a **hidden environmental dependency**: the artifact's output depends
# on an authentication state that is nowhere in the environment specification, no
# version pin, no checksum, no Docker layer. Everything that *was* pinned
# reproduced perfectly. The one thing nobody thought to pin is the one that moved.
#
# Reproducibility checklists ask about versions, seeds and data. They rarely ask:
# *does any tool in this pipeline behave differently when logged in?*
#
# **And now the good news.** Look at what survived it:
#
# * every **incidence** rate — the share of tasks with at least one finding — is
#   unchanged;
# * `clean_strict@1` is identical to the decimal for all four authors;
# * only the unbounded **counts** moved.
#
# Collapsing several findings into one cannot turn a task with findings into a
# task without findings. Bounded, order-invariant summaries are robust to
# de-duplication; sums are not.
#
# > If you report "model X produced N vulnerabilities", you are reporting a
# > number that depends on a de-duplication key most readers never see — and, as
# > it turns out, on whether you were logged in. If you report "model X had ≥1
# > finding on Y% of tasks", you are not. This is why the paper leads with
# > incidence, and it is the single most transferable design lesson of the day.
#
# *(Everything downstream in this session uses the released key, unchanged. Our
# incidence rates are therefore directly comparable to the paper's; our raw
# finding totals run about 25% below the study's, for the reason above.)*

# %% [markdown]
# ## 10. The picture

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

bars = headline[["Defective %", "Vulnerable %", "High sev. %"]]
bars.plot(kind="bar", ax=axes[0], rot=20,
          color=[ch.PALETTE[0], ch.PALETTE[3], ch.PALETTE[5]])
axes[0].set_title("Share of tasks with at least one finding")
axes[0].set_ylabel("% of the 200 tasks")
axes[0].set_xlabel("")

clean = headline["Clean %"]
axes[1].bar(range(len(clean)), clean.values,
            color=[ch.AUTHOR_COLORS[a] for a in clean.index])
axes[1].set_xticks(range(len(clean)), clean.index, rotation=20, ha="right")
axes[1].set_title("clean_strict@1 — passed all four layers")
axes[1].set_ylabel("% of the 200 tasks")
axes[1].axhline(clean["Human"], color="#008080", ls="--", lw=1.2)
axes[1].annotate("human reference", (len(clean) - 0.4, clean["Human"] + 0.6),
                 ha="right", fontsize=8, color="#008080")
for x, value in enumerate(clean.values):
    axes[1].text(x, value + 0.7, f"{value:.1f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(paths.FIGURES / "02_headline.png")
plt.show()

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Who built the benchmark decides what its numbers mean.** Three of these
#    five authors defined the selection; beating them is not a result. The human
#    reference is the bar.
# 2. **Benchmark cost is usually tooling, not analysis.** 97% of the reference
#    evaluator's runtime was process start-up and a network call.
# 3. **Prove equivalence, don't assert it.** A field-by-field diff against the
#    reference implementation is the only thing that makes a re-implementation
#    citable.
# 4. **Incidence reproduces; counts do not.** Choose summaries that survive
#    reasonable disagreement about de-duplication.
# 5. **Pair your comparisons.** The same tasks for every author is a design
#    advantage; throwing it away in the analysis wastes it.
