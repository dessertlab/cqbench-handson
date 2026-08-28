# %% [markdown]
# # 04 — Where the differences actually are
#
# **Time:** ~45 minutes · **Charts to read, and three parameters to change.**
#
# Notebook 03 gave a verdict: the frontier model matches the human reference on
# maintainability and not on security. This notebook asks *what the code looks
# like* underneath that verdict, along the paper's three research questions:
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
# structurally, it writes human-scale code. This is the paper's finding that
# Claude reaches 98.7% of human lines of code, reproduced from raw code.
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
# ---
# # RQ3 · Security
#
# One caveat governs everything below: **Semgrep reports risk-associated
# patterns, not exploitable vulnerabilities.** Nothing is executed and nothing
# is proven reachable. "Vulnerable" here means "matched a security rule".

# %%
figures.plot_security(results);

# %% [markdown]
# This is the gap that did not close. Claude is the best model at 28%, and still
# **1.8×** the human reference, with **5×** its high-severity rate — while its
# defect incidence (RQ2) was indistinguishable from the human's.

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
# # Exercises — move one decision and watch the picture move
#
# Every number in this notebook is the output of choices someone made. Each
# exercise changes **one line** and redraws. Worked answers and discussion:
# `notebooks/solutions/04_solutions.ipynb`.

# %% [markdown]
# ### Exercise 1 — where should the "too small to count" floor be?
#
# An answer is credited only if it reaches **10%** of the human implementation's
# size. Ten percent is a floor against emptiness, not a demand to match human
# complexity. Someone chose that number.
#
# **TODO:** change `GATE` below and re-run. Try `0.30`, then `0.50`, then `0.02`.
# Watch the "Clean" panel and the structural validity chart. Which authors move,
# and why do they move by such different amounts?

# %%
GATE = 0.10        # ← the shipped value. Change it.

adjusted = whatif.with_complexity_gate(results, GATE)
figures.plot_scoreboard(adjusted, save=None)
figures.plot_validity(adjusted, save=None);

# %% [markdown]
# ### Exercise 2 — which findings did the author actually cause?
#
# Section RQ2 identified four checks that arguably say more about the evaluation
# format than about the author: `self` unused because the method was scored
# outside its class, the parameter-count checks inherited from the requested
# signature, and the synthetic class wrapper.
#
# **TODO:** run the cell as-is to remove all four, then try removing only
# `unused-argument`, then only the two `too-many-*`. Compare the ODC profile
# with the one further up. **Which conclusions from RQ2 survive, and which were
# the format talking?**

# %%
DROPPED = whatif.FORMAT_ARTIFACTS       # ← try {"unused-argument"} alone

deartifacted = whatif.without_symbols(results, DROPPED)
print("defect incidence, official vs de-artifacted:")
display(pd.DataFrame({
    "official": 100 * results.groupby("author_label", observed=True)["defective"].mean(),
    "de-artifacted": 100 * deartifacted.groupby("author_label", observed=True)["defective"].mean(),
}).round(1))
figures.plot_odc_profile(deartifacted, save=None);

# %% [markdown]
# ### Exercise 3 — is the security gap carried by one rule?
#
# `B404` fires on `import subprocess` — the import itself, not any misuse. It is
# the loudest rule against our submission (22 of Claude's findings) and the most
# obviously permissive one in the set. If the security result rests on it, that
# changes what you are allowed to claim.
#
# **TODO:** run as-is to drop `B404`, and compare the two forest plots below
# with the ones in notebook 03. How much of the submission's gap against the
# human reference was that one rule carrying? Does the gap still clear zero
# without it? Then try adding `"B113"` and see whether piling on more removals
# helps your case or hurts it.

# %%
DROP_RULES = ("B404",)      # ← try () to restore, or add "B113"

filtered = whatif.without_rules(results, DROP_RULES)
figures.plot_forest(compare_submission(filtered, "vulnerability_free_rate"),
                    f"Free of security findings — without {', '.join(DROP_RULES) or 'nothing'}",
                    save=None)
figures.plot_forest(compare_submission(filtered, "high_severity_free_rate"),
                    "Free of high-severity findings — same filter", save=None);

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Verbosity and compression are the same behaviour** seen on different task
#    sizes. Always condition on task size before reporting either.
# 2. **Control for volume** before claiming anything about vocabulary or
#    density — one confounder invented a finding about DeepSeek that was not there.
# 3. **Not every counted defect was authored.** Some are inherited from the
#    requested signature; some are manufactured by the evaluation format.
# 4. **The security gap is concentrated** in command execution and XML parsing —
#    and exercise 3 asks how much of it rests on a single permissive rule.
# 5. **Absolute rates are fragile; paired differences are robust.** Every
#    exercise moved the levels a lot and the author-to-author comparisons much
#    less, because every author meets the same artifacts on the same tasks.
