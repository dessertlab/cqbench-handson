# %% [markdown]
# # 05 — RQ3 exercises, worked

# %%
import sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths
from cqhandson.metrics import paired_bootstrap_ci

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

results = ch.results_frame()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]


def findings(author: str) -> dict[str, list]:
    return {row["task_id"]: row["vulnerability_findings"] for row in ch.load_results(author)}


def rule_of(finding: dict) -> str:
    return finding["check_id"].split(".")[-1]

# %% [markdown]
# ## Exercise 1 — Severity-weighted comparison

# %%
severity = pd.DataFrame({
    ch.AUTHOR_LABELS[a]: collections.Counter(
        f["extra"]["severity"] for fs in findings(a).values() for f in fs)
    for a in ch.AUTHOR_ORDER}).fillna(0).astype(int)
severity = severity.reindex(["INFO", "WARNING", "ERROR"]).reindex(columns=order)
display(severity)

incidence = pd.DataFrame({
    "≥1 finding": results.groupby("author_label", observed=True)["vulnerable"].mean() * 100,
    "≥1 high-severity": results.groupby("author_label", observed=True)["high_severity"].mean() * 100,
}).reindex(order)
display(incidence.round(1))

# %%
scored = {a: {r["task_id"]: r for r in ch.load_results(a)} for a in ch.AUTHOR_ORDER}
task_ids = sorted(scored["human"])

for author in ("chatgpt", "dsc", "qwen", "claude"):
    a = np.array([scored[author][t]["vulns_high_sev"] == 0 for t in task_ids], float)
    b = np.array([scored["human"][t]["vulns_high_sev"] == 0 for t in task_ids], float)
    stats = paired_bootstrap_ci(a, b, label=f"hs:{author}")
    print(f"{ch.AUTHOR_LABELS[author]:16s} high-severity-free {stats['a']:.3f} "
          f"vs human {stats['b']:.3f}  delta {stats['delta']:+.3f} "
          f"CI [{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}]  "
          f"significant={stats['significant']}")

# %% [markdown]
# **Answer — the ranking compresses but does not reorder.**
#
# | severity | Human | ChatGPT | DeepSeek | Qwen | Claude |
# |---|---:|---:|---:|---:|---:|
# | INFO | 10 | 11 | 21 | 13 | 4 |
# | WARNING | 26 | 90 | 102 | 76 | 58 |
# | ERROR | 3 | 23 | 21 | 23 | **15** |
#
# Restricting to high severity, the three 2023–24 models become
# indistinguishable from each other (21–23 findings, 10.5–11.5% incidence) — the
# spread that separated them on total findings was carried by WARNING-level
# rules. Claude keeps a real advantage over them (15 findings, 7.5%) and remains
# significantly worse than the human (3 findings, 1.5%), with a paired delta of
# −0.060 and a CI that excludes zero.
#
# The human's INFO count (10) is the highest relative to its total — mostly
# `B101` (`assert` outside tests) and `B311` (`random` for non-crypto use), both
# routinely fine in application code. **Severity filtering removes more noise
# from the human baseline than from the models**, which makes the high-severity
# comparison the more conservative one — and it is the one where the gap persists.

# %% [markdown]
# ## Exercise 2 — Is the security gap just `B404`?

# %%
def without(author: str, prefix: str) -> pd.Series:
    """Per-task 'no findings left' after dropping every rule starting with `prefix`."""
    return pd.Series({
        task: len([f for f in fs if not rule_of(f).startswith(prefix)]) == 0
        for task, fs in findings(author).items()}).sort_index()

table = pd.DataFrame({
    ch.AUTHOR_LABELS[a]: {
        "vulnerable %, official": 100 * results.loc[results["author"] == a, "vulnerable"].mean(),
        "vulnerable %, no B404": 100 * (1 - without(a, "B404").mean()),
        "findings, official": int(results.loc[results["author"] == a, "vulns_total"].sum()),
        "findings, no B404": sum(len([f for f in fs if not rule_of(f).startswith("B404")])
                                 for fs in findings(a).values()),
    } for a in ch.AUTHOR_ORDER}).T.reindex(order)
display(table.round(1))

# %%
free = {a: without(a, "B404").astype(float) for a in ch.AUTHOR_ORDER}
for author in ("chatgpt", "dsc", "qwen", "claude"):
    stats = paired_bootstrap_ci(free[author], free["human"], label=f"noB404:{author}")
    print(f"{ch.AUTHOR_LABELS[author]:16s} vuln-free without B404 {stats['a']:.3f} "
          f"vs human {stats['b']:.3f}  delta {stats['delta']:+.3f} "
          f"CI [{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}]  "
          f"significant={stats['significant']}")

# %%
MISUSE = ("B602", "B603", "B605", "B606", "B607")   # rules that need actual subprocess use

rows = []
for author in ch.AUTHOR_ORDER:
    with_b404 = co_occurring = 0
    for fs in findings(author).values():
        rules = {rule_of(f) for f in fs}
        if any(r.startswith("B404") for r in rules):
            with_b404 += 1
            co_occurring += any(r.startswith(MISUSE) for r in rules)
    rows.append({"author": ch.AUTHOR_LABELS[author],
                 "tasks with B404": with_b404,
                 "also a misuse rule": co_occurring,
                 "% co-occurring": 100 * co_occurring / with_b404 if with_b404 else np.nan})
display(pd.DataFrame(rows).set_index("author").round(1))

# %% [markdown]
# **Answer — this is the most consequential result in the exercises. Read it
# carefully.**
#
# **(1) Removing `B404` changes the headline.**
#
# | | vulnerable %, official | vulnerable %, no `B404` |
# |---|---:|---:|
# | Human | 15.5 | 15.0 |
# | ChatGPT | 48.5 | 38.0 |
# | DeepSeek-Coder | 54.0 | 42.0 |
# | Qwen2.5-Coder | 41.5 | 33.0 |
# | Claude Opus 4.8 | **28.0** | **19.0** |
#
# The human barely moves (one task). Every model drops 8–12 points. `B404` fires
# on `import subprocess` and models import subprocess far more than humans do.
#
# **(2) And it changes a conclusion.** With `B404` removed, the paired test on
# overall vulnerability incidence gives Claude a delta of **−0.040 with a CI of
# [−0.105, +0.025] — no longer significant.** The other three models stay
# significantly worse. So notebook 05's statement *"every model, including
# Claude, is significantly worse than the human on vulnerability incidence"* was
# **carried by a single rule that matches an import statement**.
#
# **(3) Is `B404` a fair proxy? Mostly not.** Among tasks where an author trips
# `B404`, a rule requiring *actual* subprocess misuse (`shell=True`, shell
# injection, partial-path execution) also fires on only **9–24%** of them —
# Claude lowest at 9%, Qwen highest at 24%. Four times out of five, importing
# subprocess is all that happened.
#
# **What to conclude.** Of the two readings in section 3, the data supports the
# critical one for *overall incidence*: `B404` is a weak proxy for
# command-injection risk, and Claude's overall security gap does not survive its
# removal. It does **not** support the critical reading for *high severity*: the
# `ERROR`-rated findings (`B410` lxml XXE, `B602` `shell=True`) exclude `B404`
# entirely, and Claude is still significantly worse there — 15 findings against
# the human's 3.
#
# **The defensible claim is therefore narrower than the paper's Python row, and
# sharper:** the frontier model's residual security gap is in *high-severity*
# patterns — unsafe XML parsing and shell execution — not in overall finding
# incidence. That is a better finding, and you got it by trying to break the
# original one.

# %% [markdown]
# ## Exercise 3 — The human baseline is not zero

# %%
human = results[results["author"] == "human"]
candidates = pd.concat([human[human["vulns_high_sev"] > 0],
                        human.nlargest(5, "vulns_total")]).drop_duplicates("task_id")

human_code = ch.load_predictions("human")
human_findings = findings("human")

for task_id in candidates["task_id"]:
    lines = human_code[task_id].splitlines()
    print("=" * 78)
    print(task_id)
    for finding in human_findings[task_id]:
        extra, line = finding["extra"], finding["start"]["line"]
        cwe = extra["metadata"].get("cwe")
        cwe = cwe[0] if isinstance(cwe, list) else cwe
        print(f"  {rule_of(finding):28s} {extra['severity']:8s} {str(cwe)[:52]}")
        print(f"      line {line}: {lines[line - 1].strip()[:84]}")

# %% [markdown]
# **Answer — a worked triage of the human high-severity findings.**
#
# Exactly three human answers carry an `ERROR`-rated finding, and they are three
# *different* weaknesses in real, shipped, human-written code:
#
# | task | rule | weakness | verdict |
# |---|---|---|---|
# | `gp132925` | `B410` | `lxml.etree` parsing untrusted XML — CWE-611 (XXE) | **(a) genuine** |
# | `gp222065` | `sqlalchemy-execute-raw-query` | raw SQL built by string formatting — CWE-89 | **(a) genuine** |
# | `gp227745` | `B602` | `subprocess.call(..., shell=True)` — CWE-78 | **(a) genuine** |
#
# All three are context-dependent — exploitability turns on whether the input is
# attacker-controlled, which the analyzer cannot see and the extracted function
# does not say — but none is a false positive. They are the same patterns the
# models get flagged for, in code that shipped.
#
# Among the highest-*count* human tasks the picture is different: `B113` (a
# `requests` call with no timeout — (a) genuine, low severity), `B101` (`assert`
# in application code — (b) fine in context, and the reason `INFO` is excluded
# from the high-severity endpoint), and `B311` (`random` where cryptographic
# randomness is not needed — (b)).
#
# **The question that decides how you read the whole notebook.** A rough triage
# puts this pipeline's false-positive-in-context rate somewhere around a third —
# high. But look at *what kind* of false positive it is: `B404` on an import,
# `B101` on an assert, `B311` on `random`. These fire on **constructs**, not on
# authorship. There is no mechanism by which the analyzer would be more
# forgiving of human code than of generated code.
#
# So the false positives are approximately a **constant offset applied to every
# author on every task** — and every author faced the same 200 tasks. That is
# precisely what a paired design cancels. A 33% false-positive rate would
# devastate an absolute claim ("15.5% of human code is vulnerable") and leaves a
# paired one ("models trip these rules 1.8× more often than humans on identical
# tasks") substantially intact.
#
# **This is the argument for paired benchmarking, and it is the one thing from
# today worth carrying into your own work:** build your benchmark so that the
# claims you care about live in *differences measured on shared items*, not in
# levels. Then imperfect instruments still yield valid comparisons.
