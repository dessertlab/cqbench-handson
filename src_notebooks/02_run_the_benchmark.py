# %% [markdown]
# # 02 — Running the benchmark: 1,000 evaluations
#
# **Time:** ~30 minutes
#
# Five authors × 200 tasks. We will:
#
# 1. see why the stock evaluator would take about **nine hours** for this,
# 2. **verify** that our faster runner produces identical results — not assume it,
# 3. run the whole thing in ~2 minutes,
# 4. reproduce a published table, and cross-check against the study's own frozen
#    results.
#
# Step 2 is not ceremony. Benchmark tooling that is "obviously equivalent" is
# where reported numbers quietly go wrong.

# %%
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths, runner
from cqhandson.metrics import compare_authors

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print(runner.check_analyzers())
print("CPU cores available:", __import__("os").cpu_count())

# %% [markdown]
# ## 1. Why the reference evaluator is slow
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
import subprocess, tempfile, os
code = ch.load_predictions("human")["python:gp206544"]

with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / "one.py"
    target.write_text(code, encoding="utf-8")

    environment = dict(os.environ)          # note: version check NOT disabled
    for key in ("SEMGREP_ENABLE_VERSION_CHECK",):
        environment.pop(key, None)

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
# ## 2. Moving the process boundary
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
#   Every other Pylint message is file-local.
#
# ### Verify it
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
# > **Want to see the slow path yourself?** Uncomment the cell below — it runs
# > the stock CLI on those 8 tasks and takes about a minute even with the version
# > check disabled. That is the command a user of the released benchmark runs.

# %%
# import subprocess, os
# environment = dict(os.environ) | {"SEMGREP_ENABLE_VERSION_CHECK": "0"}
# started = time.time()
# subprocess.run(
#     [sys.executable, "-m", "cqbench", "evaluate",
#      "--tasks", str(paths.DATA / "reference_check/tasks.jsonl"),
#      "--references", str(paths.DATA / "reference_check/references.jsonl"),
#      "--predictions", str(paths.DATA / "reference_check/human.jsonl"),
#      "--output", "/tmp/official_rerun.jsonl", "--overwrite"],
#     cwd=paths.VENDOR, env=environment, check=True)
# print(f"stock evaluator, 8 tasks: {time.time() - started:.0f}s")

# %% [markdown]
# ## 3. The run
#
# Five authors, 200 tasks each. Expect roughly **2 minutes**; more cores, less
# time. Results land in `results/live/`.

# %%
started = time.time()
counts = runner.evaluate_all(ch.AUTHOR_ORDER, jobs=None)
elapsed = time.time() - started

print(f"\n{sum(counts.values())} evaluations in {elapsed:.0f}s "
      f"({elapsed / sum(counts.values()) * 1000:.0f} ms per evaluation)")
print(f"the stock evaluator would have taken about "
      f"{sum(counts.values()) * 33 / 3600:.0f} hours")

# %% [markdown]
# > **If the cell above failed** — no analyzers, no disk, no time — everything
# > downstream still works. `ch.results_frame()` falls back to
# > `results/precomputed/`, which was produced by exactly this code.

# %%
results = ch.results_frame()          # live results if present, else the fallback
print("source:", paths.results_dir().relative_to(paths.REPO))
print(f"{len(results)} rows = {results['task_id'].nunique()} tasks "
      f"× {results['author'].nunique()} authors")
results.head(3)[["author", "task_id", "status", "defects_total", "vulns_total", "cwes"]]

# %% [markdown]
# ## 4. The headline table
#
# This is Table 5 of the paper, Python rows, plus the three older baselines.
#
# * **Defective** — share of tasks with ≥1 Pylint finding that maps to ODC
# * **Vulnerable** — share with ≥1 CWE-carrying Semgrep finding
# * **High sev.** — share with ≥1 `CRITICAL`/`ERROR` finding
# * **Clean** — `clean_strict@1`: all four layers passed

# %%
headline = ch.headline_table(results)
display(headline.round(1))

# %% [markdown]
# ### Did we reproduce the paper?
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
# ## 5. A second cross-check: the study's own frozen results
#
# `data/frozen/` contains the study's *published* per-task results for the four
# original authors — computed by the original research pipeline, not by the
# released evaluator. Comparing them to what we just measured tells us something
# the paper cannot tell us about itself.

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
# Read that table carefully, because it contains the most useful methodological
# result of the session:
#
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
import subprocess, tempfile, os
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
# ## 6. The picture

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
for x, value in enumerate(clean.values):
    axes[1].text(x, value + 0.7, f"{value:.1f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(paths.FIGURES / "02_headline.png")
plt.show()

# %% [markdown]
# ## 7. Are the differences real?
#
# Every author answered the *same* 200 tasks, so comparisons should be
# **paired**: compare per task, then average the differences. Pairing removes
# task difficulty from the comparison and gives much tighter intervals than
# comparing two independent percentages.
#
# `paired_bootstrap_ci` resamples tasks 10,000 times with the seeding scheme of
# `cqbench compare`. An interval that excludes zero means the difference is not
# explained by which tasks happened to be sampled.

# %%
for metric in ("clean_strict_at_1", "defect_free_rate",
               "vulnerability_free_rate", "high_severity_free_rate"):
    table = compare_authors(results, metric, reference="human")
    table["author"] = table["author"].map(ch.AUTHOR_LABELS)
    print(f"\n{metric}  (each author minus Human, paired over 200 tasks)")
    print(table.round(3).to_string(index=False))

# %% [markdown]
# The pattern that matters:
#
# * **Claude is statistically indistinguishable from the human reference** on
#   `clean_strict@1` and on defect-freedom — the first model in this comparison
#   for which that is true.
# * **It is still significantly worse on security**, both for any vulnerability
#   finding and for high-severity ones.
# * The three 2023–24 baselines are significantly worse than the human on
#   everything.
#
# That is the paper's headline, reproduced from raw code on your laptop in two
# minutes. The next three notebooks ask *why* — RQ1 structure, RQ2 defects,
# RQ3 security.
#
# ---
# ## Takeaways
#
# 1. **Benchmark cost is usually tooling, not analysis.** 97% of the reference
#    evaluator's runtime was process start-up and a network call.
# 2. **Prove equivalence, don't assert it.** A field-by-field diff against the
#    reference implementation is cheap and is the only thing that makes a
#    re-implementation citable.
# 3. **Incidence reproduces; counts do not.** Choose summaries that survive
#    reasonable disagreement about de-duplication.
# 4. **Pair your comparisons.** Same tasks for every author is a design
#    advantage; throwing it away in the analysis wastes it.
