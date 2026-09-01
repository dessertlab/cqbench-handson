# %% [markdown]
# # 04 — Where the differences actually are
#
# **Charts to read, one parameter to change, and a verdict to argue with.**
#
# Notebook 03 gave a verdict: the frontier model matches the human reference on
# maintainability and not on security. This notebook asks *what the code looks
# like* underneath that verdict, along three questions:
#
# | | question |
# |---|---|
# | **RQ1** | Do models write structurally different code from humans? |
# | **RQ2** | Do they make different **kinds** of mistake? |
# | **RQ3** | Do they introduce different **weaknesses**, at different severity? |
#
# Every chart is one call. If you want to see how a number is made, open
# `cqhandson/figures.py` — the functions are short and that is what they are for.
#
# > **Colour is role.** Aqua = the human reference. Blue = the three models that
# > *built* the benchmark, so their rates are a ceiling. Orange = the model under
# > test. Where three shades of blue appear, they distinguish the three
# > construction models from one another.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import cqhandson as ch
from cqhandson import figures, whatif
from cqhandson.metrics import compare_submission

ch.style()
pd.set_option("display.width", 200)

results = ch.results_frame()
tasks = ch.load_tasks()
display(ch.results_source().to_frame())

# %% [markdown]
# ---
# # RQ1 · Structure and style

# %% [markdown]
# ## How big is the generated code, next to the human's?
#
# Both authors answered the same task, so we can divide. Left: every task, model
# size against human size. Right: the same ratios as a distribution, in log2, so
# "half" and "double" sit symmetrically either side of zero.

# %%
figures.plot_size_ratio(results);

# %% [markdown]
# DeepSeek sits mostly **below** the diagonal — it writes less code than the job
# needs. Claude straddles it. And the spread grows with task size: on short
# functions the models are verbose, on long ones they compress.
#
# That single fact resolves a contradiction you will meet in the literature.
# "LLMs are verbose" and "LLMs structurally compress" are usually reported as
# competing findings; they are **the same behaviour measured on different task
# mixes**. Models regress toward a preferred output length.

# %%
figures.plot_structural_profile(results);

# %% [markdown]
# Every metric, as a fraction of the human value on the tasks both completed.
# The aqua line is the human implementation.
#
# The 2023–24 models sit at 0.5–0.9 across the board. Claude sits at 0.86–1.12 —
# structurally, it writes human-scale code: on these tasks Claude reaches
# **98.7%** of the human implementations' lines of code.
#
# **Watch the `n` in the legend.** ChatGPT's dots are computed on **12
# surviving outputs**, because the rest never cleared the structural gate. They
# are shown for completeness and mean very little.

# %%
figures.plot_lexical(results);

# %% [markdown]
# Vocabulary. A model that writes less code necessarily uses fewer distinct
# tokens, so comparing totals would be unfair — read the chart **vertically at
# the dashed line**, where every author has written the same number of tokens.
#
# At matched volume, DeepSeek's apparent vocabulary poverty **disappears**
# (2,149 against the human's 2,151): it was never repetitive, it just wrote half
# as much. Qwen's does not (1,759): Qwen genuinely reuses the same constructs.
# One uncontrolled confounder would have turned a real finding about one model
# into a false finding about two.

# %% [markdown]
# ---
# # RQ2 · Defects
#
# Counting linter warnings is not a scientific quantity — it depends on which
# checks a tool happens to implement. So every finding is mapped to a category
# of **Orthogonal Defect Classification**, a taxonomy of *what kind of mistake*
# a defect is. A finding with no category is not counted at all.

# %%
figures.plot_defect_burden(results);

# %% [markdown]
# Human and Claude are nearly the same shape. Qwen and DeepSeek carry visibly
# more. ChatGPT looks better than it is — 86% of its answers were a *different*,
# usually shorter function, so there was less code to find defects in.
#
# **A benchmark cannot tell "clean" from "absent" by counting.**

# %%
figures.plot_odc_profile(results);

# %% [markdown]
# Now the kinds. Two columns carry the story.
#
# **Function/Class/Object**: human 2%, Claude 1% — but DeepSeek **26%** and Qwen
# **20%**. That column is the signature of *templated* output: asked for a
# method, these models wrap it in a synthetic `class MyClass:`, which trips
# `too-few-public-methods` every time.
#
# **Assignment**: Qwen 62%, DeepSeek 55%, against the human's 26% — parameters
# accepted and never read, values computed and never used. Code shaped like a
# solution without being one.

# %%
figures.plot_top_symbols(results);

# %% [markdown]
# The actual checks. Three things worth stopping for:
#
# **Some defects are inherited, not authored.** `too-many-arguments` and
# `too-many-positional-arguments`: human and Claude are *identical*, because
# they fire on the **requested signature**. Every author who follows the
# contract inherits them. The only way to avoid them is to disobey — which is
# what ChatGPT does.
#
# **Some are artifacts of the format.** `unused-argument` is the largest single
# category for Qwen and DeepSeek, and inflated for everyone because methods are
# scored *outside their class*, so `self` is always unused.
#
# **And some are exactly the real-world risk.** `unspecified-encoding` (a file
# opened without an encoding — breaks on someone else's machine) and
# `missing-timeout` (an HTTP call that can hang forever) are elevated for the
# models. No test suite catches either.

# %% [markdown]
# So which of this section's findings survive if we simply refuse to count the
# format artifacts? `whatif.without_symbols` drops the four checks above and
# re-derives every ODC column.

# %%
deartifacted = whatif.without_symbols(results, whatif.FORMAT_ARTIFACTS)

def incidence(frame, column):
    """% of the 200 tasks where this author has at least one such finding."""
    return 100 * frame.assign(hit=frame[column] > 0).groupby(
        "author_label", observed=True)["hit"].mean()

display(pd.DataFrame({
    "defective %": incidence(results, "defects_total"),
    "defective %, de-artifacted": incidence(deartifacted, "defects_total"),
    "Checking %": incidence(results, "def_checking"),
    "Checking %, de-artifacted": incidence(deartifacted, "def_checking"),
}).round(1))

# %% [markdown]
# Every absolute rate falls by twenty to twenty-five points — the human from 62
# to 41, DeepSeek from 89.5 to 64.5. A defensible edit to an exclusion list moved
# **every level in this notebook**. It barely moved the *order* between authors.
#
# And one column does not move at all: **Checking** stays at 16.0 / 32.5 / 39.5 /
# 27.0 / 19.5, because none of the four artifacts map there. It is a clean
# control column, and it carries the one claim from RQ2 that needs no caveat:
#
# > The 2023–24 models produce missing-validation and missing-guard defects
# > **twice as often as the human reference**, and that survives every filter we
# > can think to apply. Claude, at 19.5 against the human's 16.0, does not.

# %% [markdown]
# ---
# # RQ3 · Security
#
# One caveat governs everything below: **Semgrep reports risk-associated
# patterns, not exploitable vulnerabilities.** Nothing is executed and nothing
# is proven reachable. "Vulnerable" here means "matched a security rule".
#
# This is the gap that did not close. Notebook 03 put a number on it — the
# submission trips a security rule on **28%** of tasks against the human's
# **15.5%**, and a high-severity one on **7.5%** against **1.5%** — while its
# defect incidence, two blocks ago, was indistinguishable from the human's.
#
# So the question here is not *how big* the gap is. It is **what it is made
# of**.

# %%
figures.plot_cwe(results);

# %% [markdown]
# The gap is concentrated, not diffuse — which makes it actionable.
#
# **CWE-78, OS command injection**: human 8, Claude 26, the older models 30–39.
# The mechanism is visible in notebook 00's example: asked to install a signing
# key, the human used a Python GPG library; the models shelled out to `apt-key`.
# **Models reach for the subprocess where humans reach for a library.**
#
# **CWE-400, resource exhaustion**: human 7, the older models 26–39 — but Claude
# **6**, *below* the human. These are almost all HTTP calls with no timeout, the
# most mechanical mistake in the table, and the frontier model has essentially
# stopped making it.
#
# So Claude cleaned up the forgetful category and kept the architectural one.

# %% [markdown]
# ---
# # Exercise — move one decision and watch the picture move
#
# Every number in this notebook is the output of choices someone made. This one
# changes a single line and redraws. Worked answer and discussion:
# `notebooks/solutions/04_solutions.ipynb`.

# %% [markdown]
# ### Where should the "too small to count" floor be?
#
# An answer is credited only if it reaches **10%** of the human implementation's
# size. Ten percent is a floor against emptiness, not a demand to match human
# complexity. Someone chose that number.
#
# **TODO:** change `GATE` below and re-run. Try `0.30`, then `0.50`, then `0.02`.
# Watch the "Clean" panel and the structural validity chart. Which authors move,
# and — the more interesting question — who does *not*, and why?

# %%
GATE = 0.10        # ← the shipped value. Change it.

adjusted = whatif.with_complexity_gate(results, GATE)
figures.plot_scoreboard(adjusted, save=None)
figures.plot_validity(adjusted, save=None);

# %% [markdown]
# ---
# # Now try to break the result
#
# The last step of an evaluation is to attack your own headline. If it falls
# over, you want to be the one who found out — not a reviewer.
#
# `B404` fires on `import subprocess` — the import itself, not any misuse. It is
# the loudest rule against our submission (**22** of Claude's findings, against
# the human's **1**) and the most obviously permissive one in the set. It is
# exactly the rule you would remove if you wanted the security result to
# disappear. So let us remove it.
#
# **Before running the cell: does the gap survive?** Take a show of hands.

# %%
DROP_RULES = ("B404",)      # ← then try () to restore, or add "B113"

filtered = whatif.without_rules(results, DROP_RULES)
figures.plot_forest(compare_submission(filtered, "vulnerability_free_rate"),
                    f"Free of security findings — without {', '.join(DROP_RULES) or 'nothing'}",
                    save=None)
figures.plot_forest(compare_submission(filtered, "high_severity_free_rate"),
                    "Free of high-severity findings — same filter", save=None);

# %% [markdown]
# | rule removed | gap on incidence | verdict | high severity |
# |---|---|---|---|
# | none | −0.125 [−0.190; −0.060] | real | −0.060 |
# | `B404` | −0.080 [−0.140; −0.020] | real | −0.060 |
# | `B404` + `B113` | **−0.095** [−0.155; −0.040] | real | −0.060 |
#
# One third of the gap was that rule. Two thirds were not, and the interval
# never touches zero. High severity does not move by a thousandth, because
# `B404` is informational and never enters that count — those findings are
# `B410` (unsafe XML parsing) and `B602` (`shell=True`), rules that require
# genuine misuse.
#
# And piling on **hurts**: removing `B113` as well *widens* the gap to −0.095,
# because that rule fires on the human 7 times and on Claude 6 — so dropping it
# helps the human more. **No single rule is carrying the result.**
#
# You did not break the claim, and that is a result too. What you can state at
# the end is more *precise* than the headline:
#
# > The frontier model's security gap against human code is not an artifact of
# > one permissive import-level rule: it narrows by about a third when that rule
# > is removed and remains significant, and the high-severity gap — unsafe XML
# > parsing and shell execution — does not move at all.

# %% [markdown]
# ---
# # So — is the model any good?
#
# We have run a lot of code. Here is the answer in plain words, because a
# benchmark that never produces one is a benchmark nobody uses.
#
# **On maintainability: yes, as good as the human reference.** Not "close to" —
# indistinguishable on this sample. The clean rate is 27.5% against 31.5% and
# defect incidence 63% against 62%; both intervals cross zero, so the data does
# not separate them. And it is not the same number arrived at differently: the
# ODC profile has the *same shape* as the human's (Assignment 29.5 vs 25.5,
# Function/Class/Object 1.0 vs 1.5), the code is human-scale on all five
# structural metrics (0.86–1.12 of the human value), and at matched volume it
# uses the same vocabulary (2,211 vs 2,151). Three years ago the models in this
# benchmark were writing half the code with three times the defects.
#
# **On security: no, clearly worse.** It trips a security rule on 28% of tasks
# against the human's 15.5%, and a high-severity one on 7.5% against 1.5%.
# Neither interval touches zero, and the gap survives removing the noisiest
# rule. On the 100 tasks that carry a consensus weakness class, it trips *that
# class* on 49% against the human's 22%.
#
# **And the gap is one habit, not general sloppiness.** It is concentrated in
# command execution — CWE-78, 26 findings against the human's 8: the model
# reaches for a subprocess where the human reaches for a library. Meanwhile the
# forgetful category is *gone*: HTTP calls with no timeout, CWE-400, 6 findings
# against the human's 7 — below the human, where the 2023–24 models sat at
# 26–39.
#
# So the one-sentence answer:
#
# > On code-quality grounds this model's output is as maintainable as the human
# > original and about twice as likely to carry a security weakness, and that
# > weakness is one identifiable architectural habit rather than diffuse
# > carelessness.
#
# Which is a **useful** finding rather than a verdict: a habit is something a
# system prompt, a CI lint rule, or a review checklist can catch. "The model is
# bad at security" is not actionable. "The model shells out where you would
# import" is.
#
# **The two caveats that must travel with that sentence.** These are tasks
# selected because older models failed them alike, so none of these numbers
# estimate the model's average behaviour — they describe robustness on
# issue-prone code. And nothing was executed, so "good" here means well-formed
# and pattern-clean, never *correct*.

# %% [markdown]
# ---
# ## What to take away about measuring
#
# 1. **Verbosity and compression are the same behaviour** seen on different task
#    sizes. Always condition on task size before reporting either.
# 2. **Control for volume** before claiming anything about vocabulary or
#    density — one confounder invented a finding about DeepSeek that was not there.
# 3. **Not every counted defect was authored.** Some are inherited from the
#    requested signature; some are manufactured by the evaluation format.
# 4. **The last step is attacking your own headline.** A claim that survived a
#    serious attempt to break it is worth more than one nobody tested.
# 5. **Absolute rates are fragile; paired differences are robust.** Every
#    exercise moved the levels a lot and the author-to-author comparisons much
#    less, because every author meets the same artifacts on the same tasks.
#
# If you design a benchmark, design it that way: make the claims you care about
# live in **differences measured on shared items**, not in levels. Then even an
# imperfect instrument gives you valid comparisons.

# %% [markdown]
# ---
# # Discussion — where is this benchmark attackable?
#
# Everything above was built on decisions we have now seen the insides of. This
# is the part where you attack them. None of these questions has a settled
# answer, and the first three are the ones worth arguing about out loud.
#
# **1. The analyzer *is* the definition of quality.** Pylint decides what counts
# as a defect, and the ODC mapping is a second exclusion list on top: 352 symbols
# have a category and everything else counts as zero. Swap in Ruff or Flake8 and
# every number in this notebook moves.
# → *Should a quality benchmark ship one analyzer, or several, and report the
# spread between them as part of the result?*
#
# **2. How independent are the three "independent" models?** The gate needs two
# of three to agree. But ChatGPT only clears the finding threshold on 111 of our
# 200 tasks, which means that on **89 of 200 the consensus is DeepSeek + Qwen
# alone** — two code-specialised models released the same year, plausibly trained
# on overlapping data.
# → *How independent does "independent agreement" have to be before agreement
# means something? And would you rather have three diverse weak models or five
# similar strong ones?*
#
# **3. A pattern is not a risk.** `B404` fires on an import. So part of
# "vulnerable" here means "chose to call an external program", which is a design
# choice, not a bug.
# → *What would you have to add to turn "28% of tasks match a security rule"
# into an estimate of actual risk — and could you do it without executing
# anything?*
#
# **4. Two magic numbers hold up the dataset.** The 10% complexity floor and the
# ≥3 finding burden. 139 of our 200 tasks sit at exactly 3, and on the full
# Python set that is 69% — move the threshold to 4 and two thirds of the
# benchmark disappears.
# → *Should a benchmark ship one threshold, or a curve across thresholds?*
#
# **5. The evaluation format manufactures defects.** Methods are scored outside
# their class, so `self` is always unused; `too-many-arguments` fires on the
# requested signature, so every obedient author inherits it.
# → *Whose responsibility is it to correct for that — the benchmark's, by
# excluding them, or the reader's, by knowing?*
#
# **6. Contamination, from here on.** The tasks are mined from public
# repositories, and this benchmark is public. Any model trained after its release
# may have seen these functions *and* their human implementations.
# → *How long does a static, mined-from-public-code benchmark stay valid, and how
# would you detect that yours has gone stale?*
#
# The honest closing position: every one of these is a real weakness, and none of
# them is a reason to discard the benchmark. They are reasons to state precisely
# what it measured — which is the whole skill this session was about.
