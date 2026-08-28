# %% [markdown]
# # 01 — From a raw analyzer message to a benchmark number
#
# **Time:** ~35 minutes
#
# In notebook 02 you will run the whole benchmark and get a table of percentages.
# This notebook is the one that decides whether you can *trust* that table,
# because here we take a single function and walk it through the pipeline by
# hand — running the analyzers ourselves and watching each transformation.
#
# CQBench scores an answer in four layers. An answer must survive all of them:
#
# | Layer | Question | Tool |
# |---|---|---|
# | 1 · Structural | Does it parse, and is it *the requested function*, and is it more than a stub? | Python `ast` |
# | 2 · Complexity gate | Is it plausibly a real implementation, or a token gesture? | `lizard` |
# | 3 · Defects | What maintainability/correctness issues does a linter find? | `pylint` → ODC |
# | 4 · Vulnerabilities | What security weaknesses does a SAST tool find? | `semgrep` → CWE |
#
# The headline metric, `clean_strict@1`, is the conjunction: parse **and** right
# signature **and** non-stub **and** non-degenerate **and** zero defects **and**
# zero vulnerabilities. Every layer can veto.

# %% [markdown]
# ## The four tools, and what each is for
#
# Before running anything, know what the instruments do — and what they cannot do.
#
# ### `lizard` — measures shape
# Parses a function and reports its size and control flow: lines of code,
# **cyclomatic complexity** (how many independent paths through the function, so
# roughly how many tests it would take to cover), nesting depth, and the
# **Halstead** family (volume ≈ how much information the code carries,
# difficulty ≈ how much effort it takes to read). It judges nothing. It is used
# here for one job: deciding whether a generated answer is substantial enough to
# be worth analysing at all.
#
# ### `pylint` — finds defects
# A linter. It reads Python without running it and reports things that are
# wrong, suspicious, or unmaintainable: a variable assigned and never used, a
# file opened without an encoding, an HTTP call with no timeout, a function with
# too many parameters. **None of these fail a test suite.** They are the class of
# problem that surfaces later — on someone else's machine, under load, during a
# review.
#
# ### `semgrep` — finds security patterns
# A static application security testing tool. It matches code against rules that
# describe *shapes* known to create risk: building a shell command from a
# variable, parsing XML with a library that resolves external entities, an HTTP
# call over plain http. It reports a **pattern**, not a proven exploit. Nothing
# is executed and nothing is shown to be reachable by an attacker.
#
# ### ODC and CWE — the translation layer
# This is the part people skip, and it is the reason the benchmark works at all.
#
# A count of Pylint warnings is not a scientific quantity: it depends entirely on
# which checks Pylint happens to implement this year, and it cannot be compared
# with a count of PMD warnings in Java or Clang-Tidy warnings in C. So every
# finding is mapped into a taxonomy:
#
# * **ODC** — *Orthogonal Defect Classification*, IBM, 1992. Classifies a defect
#   by the **kind of fix** it needs: `Assignment` (a wrong value), `Checking` (a
#   missing guard or validation), `Algorithm` (wrong logic in a right
#   structure), `Interface` (wrong interaction with callers),
#   `Function/Class/Object` (the unit itself is structured wrongly),
#   `Timing/Serialization` (ordering, concurrency, resource lifetime).
# * **CWE** — *Common Weakness Enumeration*, the industry catalogue of software
#   weakness types. CWE-78 is OS command injection, CWE-611 is XML external
#   entity, CWE-400 is uncontrolled resource consumption.
#
# Both mappings do double duty: they translate, **and they filter**. A Pylint
# symbol with no ODC category is not counted as a defect; a Semgrep finding with
# no CWE metadata is not counted as a vulnerability. Someone decided which is
# which, and you are about to see the spreadsheet where they wrote it down.

# %%
import sys, pathlib, json, subprocess, tempfile, os
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import cqhandson as ch
from cqhandson import paths
from cqhandson.runner import _env      # analyzer environment (temp dirs, no version check)

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)

tasks, references = ch.load_tasks(), ch.load_references()

TASK = "python:gp206544"
task, reference = tasks[TASK], references[TASK]
answers = {a: ch.load_predictions(a)[TASK] for a in ch.AUTHOR_ORDER}

print("required signature:", task["signature"]["text"])

# %% [markdown]
# ## Layer 1 — Structural validity
#
# Before any quality question, a mechanical one: *is this the thing we asked for?*
# `analyze_structure` parses the answer, looks for a function with the required
# **name**, checks its **arity**, and applies a set of stub detectors.

# %%
from cqbench.structural import Signature, analyze_structure

signature = Signature(**task["signature"])

rows = {}
for author, code in answers.items():
    result = analyze_structure(
        code, "python", signature,
        human_token_count=reference["human_metrics"]["token_count"],
        human_ast_count=reference["human_metrics"]["ast_node_count"],
    )
    rows[ch.AUTHOR_LABELS[author]] = result.to_dict()

structure = pd.DataFrame(rows).T
display(structure[["parseable", "target_present", "target_matches_arity",
                   "explicit_stub", "constant_noop_only", "undersized",
                   "nonstub", "strict_nontrivial", "status"]])

# %% [markdown]
# ChatGPT is `target_missing`: it wrote a syntactically perfect function that is
# simply **not the one requested**. This is worth sitting with. The benchmark
# treats "wrote a different function" and "wrote a broken function" as the same
# kind of failure — both score zero. That is a defensible choice (you asked for
# an API contract and did not get it) but it means the headline number mixes
# *instruction-following* with *code quality*. When you see a model's score
# collapse in notebook 02, this column is the first place to look.
#
# The possible `status` values, in the order they are checked:
#
# | status | meaning |
# |---|---|
# | `parse_error` | the answer is not valid Python |
# | `target_missing` | no function with the required name |
# | `arity_mismatch` | right name, wrong number of parameters |
# | `explicit_stub` | `pass`, `...`, `raise NotImplementedError`, or a TODO |
# | `constant_noop` | the body is a single `return <constant>` |
# | `undersized` | ≤2 statements *and* <10% of the human reference on both token and AST count |
# | `complexity_degenerate` | passed the above but failed layer 2 |
# | `nontrivial` | survived everything |

# %% [markdown]
# ## Layer 2 — The complexity gate
#
# Layer 1 can be gamed by a one-liner that technically has the right signature.
# Layer 2 asks whether the answer has *substance*, relative to the code the task
# was mined from. `lizard` extracts per-function metrics; the rule is deliberately
# permissive:
#
# > qualified if `generated_NLOC / human_NLOC ≥ 0.10` **or**
# > `generated_HalsteadVolume / human_HalsteadVolume ≥ 0.10`
#
# Ten percent. It is a floor against emptiness, not a demand to match human
# complexity — and `or` rather than `and`, so one metric is enough.

# %%
from cqbench.evaluate import _complexity, _complexity_gate

gate_rows = {}
for author, code in answers.items():
    complexity = _complexity(code, "python", TASK)
    gate = _complexity_gate(complexity, reference["human_complexity"])
    gate_rows[ch.AUTHOR_LABELS[author]] = {
        "NLOC": complexity.get("nloc_mean"),
        "CCN": complexity.get("ccn_mean"),
        "Halstead V": round(complexity.get("halstead_volume_mean", 0), 1),
        "NLOC ratio": round(gate["complexity_nloc_ratio"], 2),
        "Halstead ratio": round(gate["complexity_halstead_volume_ratio"], 2),
        "passes gate": gate["complexity_non_degenerate"],
    }
print(f"human reference: NLOC {reference['human_complexity']['nloc']}, "
      f"Halstead V {reference['human_complexity']['halstead_v']:.1f}\n")
display(pd.DataFrame(gate_rows).T)

# %% [markdown]
# Everyone clears 0.10 here. Note Claude at ~4× the human NLOC — the gate has no
# upper bound, so verbosity is never penalised by this layer. It will be
# penalised by the next one, for a reason worth arguing about.

# %% [markdown]
# ## Layer 3 — Defects: Pylint, then three filters
#
# Now we run the analyzer ourselves, exactly as the evaluator does. Let's use
# Claude's answer — the longest and most careful of the five.

# %%
CODE = answers["claude"]

with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / "submission.py"
    target.write_text(CODE, encoding="utf-8")
    completed = subprocess.run(
        ["pylint", str(target), "--output-format=json", "--score=no", "-j=1"],
        capture_output=True, text=True, env=_env(),
    )
    raw_messages = json.loads(completed.stdout or "[]")

raw = pd.DataFrame(raw_messages)[["symbol", "type", "line", "message"]]
print(f"Pylint emitted {len(raw)} messages:\n")
display(raw)

# %% [markdown]
# Pylint said a lot. Most of it is not what the benchmark means by "a defect".
# Three filters run, in order.
#
# ### Filter 1 — the exclusion list
#
# The study excludes symbols that are noisy, stylistic, or an artifact of
# analyzing a *fragment* rather than a program. `missing-module-docstring` fires
# on every single submission because a bare function is not a module.
# `undefined-variable` and `import-error` fire because the surrounding file, its
# imports and its class are not there.

# %%
from support.rq4_build_table import PYLINT_EXCLUDED_SYMBOLS

print(f"{len(PYLINT_EXCLUDED_SYMBOLS)} excluded symbols:\n")
print(", ".join(sorted(PYLINT_EXCLUDED_SYMBOLS)))

survivors = raw[~raw["symbol"].isin(PYLINT_EXCLUDED_SYMBOLS)]
print(f"\n{len(raw)} messages -> {len(survivors)} after the exclusion list")
display(survivors)

# %% [markdown]
# ### Filter 2 — the ODC mapping
#
# A count of linter warnings is not a scientific quantity: it depends on which
# checks a tool happens to implement. To compare *across languages and tools*,
# every surviving symbol is mapped to a category of **Orthogonal Defect
# Classification** (Chillarege et al., IBM, 1992) — a taxonomy of what *kind of
# mistake* a defect represents.
#
# A symbol with no ODC category is **not counted as a defect at all**. So the
# mapping is simultaneously a translation *and* a second exclusion list.

# %%
mapping = pd.read_excel(paths.PYLINT_ODC_MAP, engine="openpyxl")
print(f"{len(mapping)} Pylint symbols carry an ODC category:\n")
display(mapping["odc_category"].value_counts().rename("symbols").to_frame())

odc_of = dict(zip(mapping["symbol"], mapping["odc_category"]))
mapped = survivors.assign(odc=survivors["symbol"].map(odc_of).fillna("--"))
display(mapped)
print(f"{len(mapped)} -> {(mapped['odc'] != '--').sum()} messages carry an ODC category")

# %% [markdown]
# ### Filter 3 — de-duplication
#
# Finally, findings are de-duplicated on `(symbol, ODC category, line)`: one
# defect per site, however many times the tool reports it.

# %%
counted = mapped[mapped["odc"] != "--"].drop_duplicates(["symbol", "odc", "line"])
display(counted)
print(f"defects_total = {len(counted)}")
print("ODC breakdown:", counted["odc"].value_counts().to_dict())

# %% [markdown]
# ### Now look at what survived
#
# Claude's three defects are `consider-using-with`, `unused-argument 'self'` and
# `unused-variable 'out'`.
#
# `unused-argument 'self'` deserves attention. `self` is unused because the
# method was extracted from its class and is being analyzed as a free-standing
# function — the benchmark's own task format created that finding. It is a
# **measurement artifact**, and it is counted as an Assignment defect.
#
# Meanwhile Qwen's six-line answer, which writes key material to a
# world-readable path and shells out to `apt-key`, scores **zero defects**.
#
# Neither of those is a bug in the pipeline. They are consequences of what a
# linter can see. But they are exactly why you look at findings, not just counts,
# before drawing conclusions — and we will come back to this in notebook 04.

# %% [markdown]
# ## Layer 4 — Vulnerabilities: Semgrep, then CWE
#
# Same exercise for security. CQBench does **not** use the live Semgrep registry:
# it ships a frozen snapshot of 1,847 rules resolved from 16 rule packs
# (`p/trailofbits`, `p/cwe-top-25`, `p/owasp-top-ten`, `p/command-injection`, …),
# with a recorded SHA-256. A benchmark whose rules can change under it is not a
# benchmark.

# %%
manifest = json.loads((paths.VENDOR / "cqbench/rules/manifest.json").read_text())
print(json.dumps(manifest, indent=2))

# %%
with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / "submission.py"
    target.write_text(CODE, encoding="utf-8")
    completed = subprocess.run(
        ["semgrep", "scan", "--config", str(paths.SEMGREP_RULES),
         "--json", "--metrics", "off", "--no-git-ignore", "--disable-version-check",
         str(target)],
        capture_output=True, text=True, env=_env(),
    )
    report = json.loads(completed.stdout)

source_lines = CODE.splitlines()
print(f"{len(report['results'])} raw findings, {len(report.get('errors', []))} scan errors\n")
for finding in report["results"]:
    extra, line = finding["extra"], finding["start"]["line"]
    print(f"  rule       {finding['check_id'].split('.')[-1]}")
    print(f"  severity   {extra['severity']}")
    print(f"  cwe        {extra['metadata'].get('cwe')}")
    print(f"  at line {line}: {source_lines[line - 1].strip()}")
    print(f"  extra['lines'] == {extra['lines'].strip()!r}   <- note this")
    print(f"  message    {extra['message'].strip()[:110]}\n")

# %% [markdown]
# Two rules that matter here:
#
# * a finding **without CWE metadata is discarded** — the benchmark counts
#   weaknesses, not style hints;
# * survivors are de-duplicated on `(normalized CWEs, severity, matched source
#   text)`, and `CRITICAL`/`ERROR` severities are counted as **high severity**.
#
# > Did you notice `extra['lines']` above? It should hold the source text
# > Semgrep matched — and instead it says `'requires login'`. That is the third
# > component of the de-duplication key. Park the observation; notebook 02 shows
# > what it does to the published numbers.
#
# The single finding on Claude's answer is `B404`: *"Consider possible security
# implications associated with the subprocess module"*, CWE-78 (OS command
# injection), triggered by `import subprocess`.
#
# It is not wrong — shelling out is genuinely where command injection lives — but
# the rule fires on the **import**, not on any misuse. Claude passes a fixed
# argument list to `Popen` with no shell, which is the safe way to do it. The
# human answer avoids the finding by using a Python GPG binding instead. So on
# this task, "vulnerable" mostly means "chose to call an external program".
#
# **This is the honest reading of a SAST-based benchmark: it measures
# *risk-associated patterns*, not exploitable bugs.** The paper says exactly
# this. Whether the pattern is a fair proxy is a question you get to argue about.

# %%
from support.rq4_build_table import normalized_cwes

unique = {}
for finding in report["results"]:
    extra = finding["extra"]
    cwes = extra.get("metadata", {}).get("cwe")
    if not cwes:
        continue
    key = (normalized_cwes(cwes), str(extra["severity"]).upper(), extra["lines"].strip())
    unique.setdefault(key, finding)

print("vulns_total     =", len(unique))
print("vulns_high_sev  =", sum(k[1] in {"CRITICAL", "ERROR"} for k in unique))
print("cwes            =", sorted({c for k in unique for c in k[0]}))

# %% [markdown]
# ## Putting the layers together
#
# `clean_strict@1` — the number the benchmark leads with — is the conjunction of
# all four layers.

# %%
def clean_strict(author: str) -> dict:
    """The full scoring path for one answer, laid out layer by layer."""
    code = answers[author]
    structural = analyze_structure(
        code, "python", signature,
        human_token_count=reference["human_metrics"]["token_count"],
        human_ast_count=reference["human_metrics"]["ast_node_count"],
    )
    gate = _complexity_gate(_complexity(code, "python", TASK), reference["human_complexity"])
    scored = [r for r in ch.load_results(author) if r["task_id"] == TASK][0]
    strict = structural.strict_nontrivial and gate["complexity_non_degenerate"]
    return {
        "1 structural": structural.strict_nontrivial,
        "2 non-degenerate": gate["complexity_non_degenerate"],
        "3 zero defects": scored["defects_total"] == 0,
        "4 zero vulns": scored["vulns_total"] == 0,
        "clean_strict@1": bool(strict and scored["defects_total"] == 0
                               and scored["vulns_total"] == 0),
    }

display(pd.DataFrame({ch.AUTHOR_LABELS[a]: clean_strict(a) for a in ch.AUTHOR_ORDER}).T)

# %% [markdown]
# On this one task the benchmark's verdict is: **Human and Qwen clean, everyone
# else not.** Go back and re-read the five answers in notebook 00 and ask
# yourself whether you agree. Most people don't — Qwen's answer writes key
# material to a predictable path and never checks whether the import succeeded,
# and Claude's careful version is penalised for a `self` it cannot use and an
# `import subprocess` it uses safely.
#
# That disagreement is not a reason to discard the benchmark. Over 200 tasks
# these idiosyncrasies partly average out, and the *paired* design means every
# author meets the same artifacts. It is a reason to know what you are measuring
# before you rank anything.

# %% [markdown]
# ## Exercise — move a boundary, watch the number move (5 minutes)
#
# Every filter above is a decision someone made. Change one and see what happens
# to Claude's defect count.
#
# **TODO:** add `unused-argument` to the exclusion list — the argument being that
# scoring a bare method out of its class manufactures that finding — and
# recompute `defects_total`.

# %%
# TODO: build `my_exclusions` = the study's list plus "unused-argument",
#       re-apply filters 1-3 to `raw`, and print the new defects_total.
#
# my_exclusions = ...
# my_survivors  = raw[...]
# my_counted    = ...
# print("defects_total with my exclusion list:", len(my_counted))

# %% [markdown]
# **Discussion.** You just changed a published number by editing one line of a
# spreadsheet. Two questions worth arguing about over the next hour:
#
# 1. Is `unused-argument` on a bare method a *defect in the generated code*, or
#    an artifact of the task format? Whose responsibility is it to decide?
# 2. If a reviewer can move your headline metric by 20% with a defensible tweak
#    to an exclusion list, what should a paper report alongside the metric?
#
# ---
# ## Takeaways
#
# 1. A benchmark number is the **output of a pipeline of choices**: which
#    analyzer, which version, which rules, which exclusions, which taxonomy,
#    which de-duplication key, which conjunction.
# 2. `clean_strict@1` fails an answer for **failing to follow instructions** and
#    for **being insecure** with the same zero. Always decompose before ranking.
# 3. SAST findings are **risk patterns, not exploits**. `import subprocess` is a
#    finding; a shell injection is also a finding.
#
# Next: `02_reproduce_the_study.ipynb` — the same pipeline, 800 times, and a
# check that your machine agrees with the reference results.
