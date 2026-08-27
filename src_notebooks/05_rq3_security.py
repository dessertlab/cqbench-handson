# %% [markdown]
# # 05 — RQ3: Do security vulnerabilities differ?
#
# **Time:** ~30 minutes · **Format:** worked analysis + `TODO` exercises
#
# > **RQ3.** Do security vulnerabilities differ between human-written and
# > AI-generated code, in terms of type and severity?
#
# This is where the frontier model does **not** catch up. Claude matched the
# human reference on structure (notebook 03) and on defects (notebook 04). On
# security it is significantly worse — and the classes where it is worse are
# specific and explicable.
#
# First, the standing caveat, because it governs every sentence below:
#
# > Semgrep reports **risk-associated patterns**, not exploitable vulnerabilities.
# > `import subprocess` is a CWE-78 finding whether or not any command is ever
# > built from user input. Nothing here is executed and nothing is proven
# > reachable. "Vulnerable" in this notebook means "matched a security rule".

# %%
import sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths
from cqhandson.metrics import compare_authors, cwe_profile

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

results = ch.results_frame()
tasks = ch.load_tasks()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]

# %% [markdown]
# ## 1. Incidence and severity

# %%
security = results.groupby("author_label", observed=True).agg(
    vulnerable_pct=("vulnerable", lambda s: 100 * s.mean()),
    high_severity_pct=("high_severity", lambda s: 100 * s.mean()),
    total_findings=("vulns_total", "sum"),
    high_severity_findings=("vulns_high_sev", "sum"),
).reindex(order)
display(security.round(2))

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(order))
ax.bar(x - 0.19, security["vulnerable_pct"], 0.38, label="≥1 finding",
       color=ch.PALETTE[0])
ax.bar(x + 0.19, security["high_severity_pct"], 0.38, label="≥1 high-severity finding",
       color=ch.PALETTE[3])
ax.set_xticks(x, order, rotation=20, ha="right")
ax.set(ylabel="% of the 200 tasks", title="Security findings by author")
ax.legend()
fig.tight_layout()
fig.savefig(paths.FIGURES / "05_security_incidence.png")
plt.show()

# %% [markdown]
# The human reference trips a security rule on 15.5% of tasks; every model is
# above it. Claude is the best model at 28.0% — and still **1.8×** the human,
# with **5×** the human's high-severity rate. Compare that with notebook 04,
# where Claude's defect incidence (63.0%) was indistinguishable from the human's
# (62.0%).
#
# **Whatever closed the maintainability gap between 2023 and 2026 did not close
# the security gap.**

# %% [markdown]
# ## 2. Which weaknesses
#
# Findings are mapped to CWE classes. Counts, not incidence — one task can
# contribute several.

# %%
cwe = cwe_profile(results).reindex(columns=order).fillna(0).astype(int)
display(cwe.head(12))

CWE_NAMES = {
    "CWE-78":  "OS command injection",
    "CWE-400": "Uncontrolled resource consumption",
    "CWE-611": "XML external entity (XXE)",
    "CWE-319": "Cleartext transmission",
    "CWE-95":  "Eval injection",
    "CWE-330": "Insufficiently random values",
    "CWE-89":  "SQL injection",
    "CWE-939": "Improper URL authorization",
    "CWE-79":  "Cross-site scripting",
    "CWE-116": "Improper encoding/escaping",
    "CWE-754": "Improper check for unusual conditions",
    "CWE-502": "Deserialization of untrusted data",
}

fig, ax = plt.subplots(figsize=(10, 4.5))
top = cwe.head(8)
labels = [f"{c}\n{CWE_NAMES.get(c, '')}" for c in top.index]
bottom = np.zeros(len(top))
for author in order:
    ax.bar(range(len(top)), top[author], 0.7, bottom=bottom, label=author,
           color=ch.AUTHOR_COLORS[author])
    bottom += top[author].to_numpy()
ax.set_xticks(range(len(top)), labels, fontsize=8)
ax.set(ylabel="findings", title="Where the security findings are")
ax.legend(ncols=5, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(paths.FIGURES / "05_cwe_profile.png")
plt.show()

# %% [markdown]
# ### The story is two classes
#
# **CWE-78 — OS command injection.** Human 8, Claude 26, and 30–39 for the
# 2023–24 models. This is the surplus the paper highlights. The mechanism is
# visible in notebook 00's example: asked to install a signing key, the human
# used a Python GPG binding; the models shelled out to `apt-key`. Models reach
# for the subprocess where humans reach for a library.
#
# **CWE-400 — uncontrolled resource consumption.** Human 7, ChatGPT 26,
# DeepSeek 39, Qwen 27 — but **Claude 6**, *below* the human. Almost all of these
# are `requests.get(...)` with no timeout. It is the most mechanical mistake in
# the table and the frontier model has essentially stopped making it.
#
# So Claude's remaining security gap is not diffuse. It has cleaned up the
# forgetful category and kept the architectural one.

# %% [markdown]
# ## 3. Down to the rules
#
# Which Semgrep rules actually fire, and at what severity?

# %%
rules = {}
for author in ch.AUTHOR_ORDER:
    counter = collections.Counter(
        (finding["check_id"].split(".")[-1], finding["extra"]["severity"])
        for row in ch.load_results(author) for finding in row["vulnerability_findings"])
    rules[ch.AUTHOR_LABELS[author]] = counter

names = sorted({k for c in rules.values() for k in c},
               key=lambda k: -sum(c[k] for c in rules.values()))[:10]
display(pd.DataFrame({label: {f"{rule} ({severity})": counter[(rule, severity)]
                              for rule, severity in names}
                      for label, counter in rules.items()})[order])

# %% [markdown]
# | rule | what it matches | severity |
# |---|---|---|
# | `B404` | `import subprocess` — *the import itself* | WARNING |
# | `B113` | a `requests` call with no `timeout=` | WARNING |
# | `B410` / `B314` | XML parsed with `lxml` / `ElementTree` (XXE exposure) | ERROR / WARNING |
# | `B602` | `subprocess` with `shell=True` | ERROR |
# | `B101` | `assert` used outside tests | INFO |
#
# `B404` is the largest single contributor for every model, and it fires on an
# import. Both readings are available to you: *"the benchmark is counting an
# import as a vulnerability"*, and *"choosing to shell out is exactly the design
# decision that creates command-injection surface, and the models make it far
# more often than humans do."* Exercise 2 asks you to settle it with the data.
#
# The **high-severity** findings are a different set — `B410` (lxml XXE) and
# `B602` (`shell=True`) — and those are not stylistic. Claude carries 15 of them
# against the human's 3.

# %% [markdown]
# ## 4. Does the benchmark's selection generalise?
#
# The sharpest threat to a failure-derived benchmark: it was built from the
# failures of *these three models*, so of course they fail on it. Does it retain
# any force against a model that took no part in its construction?
#
# 100 of our 200 tasks carry a **consensus CWE** — a weakness class that two
# 2023–24 models both produced, fixed at selection time. Claude was released
# after that. Does it trip the same class?

# %%
consensus = {task_id: set(task["difficulty"]["consensus_cwes"])
             for task_id, task in tasks.items()}
gate_tasks = [t for t, classes in consensus.items() if classes]

rows = []
for author in ch.AUTHOR_ORDER:
    scored = {row["task_id"]: row for row in ch.load_results(author)}
    hits = sum(bool(set(scored[t]["cwes"]) & consensus[t]) for t in gate_tasks)
    any_finding = sum(bool(scored[t]["cwes"]) for t in gate_tasks)
    rows.append({
        "author": ch.AUTHOR_LABELS[author],
        "trips the task's own CWE class": 100 * hits / len(gate_tasks),
        "any finding at all": 100 * any_finding / len(gate_tasks),
        "share of its findings in-class": 100 * hits / any_finding if any_finding else np.nan,
    })
print(f"{len(gate_tasks)} tasks with a consensus CWE class\n")
display(pd.DataFrame(rows).set_index("author").round(1))

# %% [markdown]
# The three models used to build the benchmark trip its consensus classes on
# 79–94% of these tasks — as they must, they defined them. The **human**
# reference, which also took no part in selection, trips them on 22%.
#
# **Claude: 49%.** Less than the construction models, but more than double the
# human baseline on the same tasks. The tasks elicit not merely findings, but
# *the same class* of finding, from a model outside the selection. That is real
# evidence that CQBench is capturing something about the tasks and not only
# about its three original authors — and it is the strongest available answer to
# the "you only measured your own construction set" objection.

# %% [markdown]
# ## 5. Significance

# %%
for metric in ("vulnerability_free_rate", "high_severity_free_rate"):
    table = compare_authors(results, metric, reference="human")
    table["author"] = table["author"].map(ch.AUTHOR_LABELS)
    print(f"\n{metric}  (author minus Human, paired over 200 tasks)")
    print(table.round(3).to_string(index=False))

# %% [markdown]
# Every model, including Claude, is significantly worse than the human reference
# on both. This is the one result in the session where no model reaches the human
# baseline.

# %% [markdown]
# ---
# # Exercises
#
# Worked answers: `notebooks/solutions/05_rq3_solutions.ipynb`.

# %% [markdown]
# ### Exercise 1 — Severity-weighted comparison (10 min)
#
# Incidence treats an `INFO`-level `assert` and an `ERROR`-level `shell=True` as
# the same event.
#
# **TODO:** build a per-author severity breakdown from `vulnerability_findings`
# (`finding["extra"]["severity"]`), then compare authors on **high-severity
# findings only**. Recompute a `high_severity_free_rate` comparison against the
# human. Does the author ranking change when you drop `INFO` and `WARNING`?

# %%
# TODO
# severities = ...

# %% [markdown]
# ### Exercise 2 — Is the security gap just `B404`? (15 min)
#
# `B404` fires on `import subprocess` alone and is the largest single contributor
# for every model. Settle whether it is carrying the result.
#
# **TODO:**
# 1. Recompute `vulns_total` and the vulnerable-incidence rate per author with
#    `B404` findings removed.
# 2. Re-run `compare_authors` on the recomputed rate. Is Claude still
#    significantly worse than the human?
# 3. Then check the *other* side: among the tasks where an author trips `B404`,
#    how often do they *also* trip a rule that requires actual misuse (`B602`
#    `shell=True`, `B605` shell injection, `B603`)? If shelling out reliably
#    comes with real misuse, `B404` is a decent proxy after all.
#
# Write down which of the two readings from section 3 the data supports.

# %%
# TODO

# %% [markdown]
# ### Exercise 3 — The human baseline is not zero (10 min, discuss)
#
# Real, shipped, human-written code trips a security rule on 15.5% of these
# tasks and carries 3 high-severity findings.
#
# **TODO:** read five of them. Pull the human tasks with `vulns_high_sev > 0` and
# the tasks with the highest `vulns_total`, look at the code and the flagged
# line, and classify each as: (a) a genuine weakness, (b) a true pattern that is
# fine in context, (c) a false positive.
#
# *(Remember from notebook 02 that `extra["lines"]` is redacted — use
# `finding["start"]["line"]` and index into the source yourself. The helper below
# does it.)*
#
# Then answer the question that decides how you read this whole notebook: **what
# is the false-positive rate of this pipeline, and does it differ between human
# and model code?** If it does not, the *differences* between authors survive even
# a high false-positive rate — which is the argument for paired benchmarking.

# %%
human = results[results["author"] == "human"]
candidates = pd.concat([
    human[human["vulns_high_sev"] > 0],
    human.nlargest(5, "vulns_total"),
]).drop_duplicates("task_id")
display(candidates[["task_id", "vulns_total", "vulns_high_sev", "cwes"]])

human_code = ch.load_predictions("human")
human_findings = {row["task_id"]: row["vulnerability_findings"]
                  for row in ch.load_results("human")}

def inspect(task_id: str) -> None:
    """Print each finding on a task next to the line it actually flagged."""
    lines = human_code[task_id].splitlines()
    print(f"--- {task_id} " + "-" * 50)
    for finding in human_findings[task_id]:
        extra, line = finding["extra"], finding["start"]["line"]
        print(f"  {finding['check_id'].split('.')[-1]:28s} {extra['severity']:8s} "
              f"{extra['metadata'].get('cwe')}")
        print(f"      line {line}: {lines[line - 1].strip()[:84]}")

inspect(candidates["task_id"].iloc[0])

# TODO: read the rest, and classify each finding (a) / (b) / (c)

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Security is the gap that did not close.** Structure and defects converged
#    to the human profile in the frontier model; vulnerability incidence did not.
# 2. **The gap is concentrated, not diffuse** — command execution and XML
#    parsing, not everything at once. Concentrated gaps are actionable ones.
# 3. **SAST measures patterns.** Say so when you report it, and check whether one
#    permissive rule is carrying your headline before you publish it.
# 4. **The selection generalises.** A model built after the benchmark still trips
#    the tasks' pre-registered weakness classes at twice the human rate. That is
#    what makes a failure-derived benchmark worth keeping.
#
# ---
# ## End of the session
#
# What you did: took a published benchmark apart, ran it yourself on 1,000
# code samples, verified your tooling against the reference implementation,
# reproduced a published table, and then spent three notebooks finding out which
# of its numbers you would be willing to defend.
#
# That last part is the transferable skill.
