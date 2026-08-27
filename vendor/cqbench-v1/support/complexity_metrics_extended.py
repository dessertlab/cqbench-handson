"""
Function-level complexity metrics for C / Python / Java code corpora.

Fixes applied relative to the previous version:
  1. Filtering of degenerate functions (NLOC <= 0, no operators, no operands).
  2. Correct Cliff's delta computation; output is guaranteed in [-1, 1].
  3. MAX_NESTING_DEPTH: 'else' treated as continuation, not new level (C/Java).
  4. Python docstrings stripped before Halstead counting.
  5. DATA_COMPLEXITY composite removed (not methodologically sound as defined).
  6. MAINTAINABILITY_INDEX explicitly named as Microsoft-adjusted Oman-Hagemeister.
  7. Benjamini-Hochberg FDR correction applied to pairwise p-values.
  8. Function-level sampling when --sample-size is provided (coupling preserved).

Metric set (10 metrics, no invented composites):
  NLOC, CCN, PARAMETER_COUNT, MAX_NESTING_DEPTH,
  DISTINCT_OPERATORS, DISTINCT_OPERANDS, TOTAL_OPERATORS, TOTAL_OPERANDS,
  HALSTEAD_VOLUME, HALSTEAD_DIFFICULTY, HALSTEAD_EFFORT,
  MAINTAINABILITY_INDEX

The ten metrics reported in the paper summary: NLOC, CCN, PARAMETER_COUNT,
MAX_NESTING_DEPTH, DISTINCT_OPERANDS, TOTAL_OPERANDS, HALSTEAD_VOLUME,
HALSTEAD_DIFFICULTY, HALSTEAD_EFFORT, MAINTAINABILITY_INDEX.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import statistics
import tokenize
from collections import defaultdict

import lizard
import numpy as np
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from tqdm import tqdm
from scipy import stats as scipy_stats


LANGUAGE_CONFIG = {
    "c": {"filename": "temp.c", "lexer": "c"},
    "python": {"filename": "temp.py", "lexer": "python"},
    "java": {"filename": "Temp.java", "lexer": "java"},
}


# --------------------------------------------------------------------------
# Field-spec parsing
# --------------------------------------------------------------------------

def parse_field_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            f"Invalid field specification '{spec}'. Use the format field_name:language."
        )
    field_name, language = spec.split(":", 1)
    field_name = field_name.strip()
    language = language.strip().lower()
    if language not in LANGUAGE_CONFIG:
        raise ValueError(
            f"Unsupported language '{language}' in '{spec}'. "
            f"Supported: {', '.join(sorted(LANGUAGE_CONFIG))}."
        )
    return field_name, language


# --------------------------------------------------------------------------
# Function extraction
# --------------------------------------------------------------------------

def get_function_snippets(code: str, language: str):
    """Extract (lizard_function_object, source_snippet) pairs from a code string."""
    config = LANGUAGE_CONFIG[language]
    try:
        analysis = lizard.analyze_file.analyze_source_code(config["filename"], code)
    except Exception:
        return
    lines = code.splitlines()
    for func in analysis.function_list:
        snippet = "\n".join(lines[func.start_line - 1 : func.end_line])
        yield func, snippet


# --------------------------------------------------------------------------
# Docstring stripping (Python only; C/Java don't have docstrings)
# --------------------------------------------------------------------------

def strip_python_docstrings(snippet: str) -> str:
    """
    Remove docstrings from a Python function snippet before Halstead counting.

    Uses the ast module. If parsing fails (e.g. snippet is incomplete), returns
    the original snippet unchanged; the caller is expected to tolerate that.
    """
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return snippet

    # Walk top-level; for each function/class/module with a docstring, null it.
    class DocstringStripper(ast.NodeTransformer):
        def _strip_from(self, node):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                # Replace with `pass` so line numbers stay roughly aligned.
                node.body[0] = ast.Pass()
            return node

        def visit_FunctionDef(self, node):
            node = self._strip_from(node)
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node):
            node = self._strip_from(node)
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            node = self._strip_from(node)
            self.generic_visit(node)
            return node

        def visit_Module(self, node):
            node = self._strip_from(node)
            self.generic_visit(node)
            return node

    stripped = DocstringStripper().visit(tree)
    ast.fix_missing_locations(stripped)
    try:
        return ast.unparse(stripped)
    except Exception:
        return snippet


# --------------------------------------------------------------------------
# Token classification
# --------------------------------------------------------------------------

def classify_token(token_type, token_value: str) -> str | None:
    """
    Classify a Pygments token as 'operator' or 'operand' per the Halstead
    convention implemented here: keywords/operators/punctuation are operators;
    identifiers and literals are operands. Comments and whitespace are dropped.

    This is one of several defensible Halstead operationalizations (see
    Decomposed Halstead Metrics, Karim et al. 2023).
    """
    if not token_value.strip():
        return None
    if token_type in Token.Comment or token_type in Token.Text:
        return None
    if (
        token_type in Token.Keyword
        or token_type in Token.Operator
        or token_type in Token.Punctuation
    ):
        return "operator"
    if token_type in Token.Name or token_type in Token.Literal:
        return "operand"
    return None


# --------------------------------------------------------------------------
# Nesting depth (control-flow only; `else` treated as continuation)
# --------------------------------------------------------------------------

CONTROL_KEYWORDS_NEW_LEVEL = {
    # Only constructs that begin a NEW nested block are listed here.
    # `else` is explicitly excluded: it continues the `if` it follows.
    "c": {"if", "for", "while", "switch", "do", "try", "catch"},
    "java": {"if", "for", "while", "switch", "do", "try", "catch", "synchronized"},
    "python": {"if", "elif", "for", "while", "try", "except", "with", "match"},
    # Python `else` / `finally` / `case` are block headers but attach to an
    # outer construct's nesting level, so we exclude them.
}


def compute_max_nesting_depth(snippet: str, language: str) -> int:
    if language in ("c", "java"):
        lexer = get_lexer_by_name(LANGUAGE_CONFIG[language]["lexer"])
        pending_control = False
        # Track () nesting so semicolons inside `for (init; cond; step)` headers
        # do not falsely reset pending_control before the body's opening brace.
        paren_depth = 0
        # Stack stores bool: is this brace attached to a control construct?
        brace_stack: list[bool] = []
        max_depth = 0
        depth_of_control = 0

        for token_type, token_value in lexer.get_tokens(snippet):
            if (
                token_type in Token.Keyword
                and token_value in CONTROL_KEYWORDS_NEW_LEVEL[language]
            ):
                pending_control = True
            elif token_value == "(":
                paren_depth += 1
            elif token_value == ")":
                if paren_depth > 0:
                    paren_depth -= 1
            elif token_value == "{":
                brace_stack.append(pending_control)
                if pending_control:
                    depth_of_control += 1
                    max_depth = max(max_depth, depth_of_control)
                    pending_control = False
            elif token_value == "}":
                if brace_stack:
                    was_control = brace_stack.pop()
                    if was_control:
                        depth_of_control -= 1
            elif token_value == ";" and paren_depth == 0:
                # Only reset on top-level semicolons (statement terminators),
                # not on those inside `for (init; cond; step)` headers.
                pending_control = False

        return max_depth

    # Python path
    max_depth = 0
    current_depth = 0
    in_control_header = False
    control_header_completed = False

    try:
        tokens = tokenize.generate_tokens(io.StringIO(snippet).readline)
        for token in tokens:
            if (
                token.type == tokenize.NAME
                and token.string in CONTROL_KEYWORDS_NEW_LEVEL["python"]
            ):
                in_control_header = True
            elif (
                in_control_header
                and token.type == tokenize.OP
                and token.string == ":"
            ):
                control_header_completed = True
            elif token.type == tokenize.INDENT:
                if control_header_completed:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                in_control_header = False
                control_header_completed = False
            elif token.type == tokenize.DEDENT:
                current_depth = max(0, current_depth - 1)
            elif control_header_completed and token.type not in (
                tokenize.NEWLINE,
                tokenize.NL,
            ):
                # Same-line suite like "if x: return y" does not open a block.
                in_control_header = False
                control_header_completed = False
    except (tokenize.TokenError, IndentationError):
        return 0

    return max_depth


# --------------------------------------------------------------------------
# Halstead metrics
# --------------------------------------------------------------------------

def compute_halstead_metrics(snippet: str, language: str) -> dict[str, float]:
    """Canonical Halstead 1977 formulas; computed on the given snippet."""
    lexer = get_lexer_by_name(LANGUAGE_CONFIG[language]["lexer"])

    distinct_operators: set[str] = set()
    distinct_operands: set[str] = set()
    total_operators = 0
    total_operands = 0

    for token_type, token_value in lexer.get_tokens(snippet):
        cls = classify_token(token_type, token_value)
        if cls == "operator":
            distinct_operators.add(token_value)
            total_operators += 1
        elif cls == "operand":
            distinct_operands.add(token_value)
            total_operands += 1

    eta1 = len(distinct_operators)
    eta2 = len(distinct_operands)
    n1 = total_operators
    n2 = total_operands
    vocabulary = eta1 + eta2
    length = n1 + n2

    if vocabulary > 0 and length > 0:
        volume = length * math.log2(vocabulary)
    else:
        volume = 0.0

    if eta2 > 0:
        difficulty = (eta1 / 2.0) * (n2 / eta2)
    else:
        difficulty = 0.0

    effort = difficulty * volume

    return {
        "distinct_operators": eta1,
        "distinct_operands": eta2,
        "total_operators": n1,
        "total_operands": n2,
        "halstead_volume": volume,
        "halstead_difficulty": difficulty,
        "halstead_effort": effort,
    }


# --------------------------------------------------------------------------
# Maintainability Index (Microsoft-adjusted Oman & Hagemeister 1992)
# --------------------------------------------------------------------------

def compute_maintainability_index(volume: float, ccn: int, nloc: int) -> float:
    """
    Microsoft-adjusted Maintainability Index (Oman & Hagemeister 1992,
    re-parameterized by Microsoft Visual Studio 2007):

        MI = max(0, 100 * (171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC)) / 171)

    The comment-ratio term (+50*sin(sqrt(2.46*perCM))) from the SEI variant
    is intentionally omitted, since docstrings and comments are stripped
    upstream for a consistent cross-author comparison.
    """
    safe_volume = max(volume, 1.0)
    safe_nloc = max(nloc, 1)
    raw = 171.0 - 5.2 * math.log(safe_volume) - 0.23 * ccn - 16.2 * math.log(safe_nloc)
    return max(0.0, 100.0 * raw / 171.0)


# --------------------------------------------------------------------------
# Per-function analysis and degenerate-function filtering
# --------------------------------------------------------------------------

def analyze_code(code: str, language: str) -> list[dict[str, float]]:
    """
    Compute all metrics for every function in `code`. Degenerate functions
    (NLOC <= 0, zero operators, or zero operands) are dropped here so they
    never enter the aggregated statistics.

    Each returned dict additionally carries `function_index_in_row`, the
    0-based position of the function within the analyzed code string,
    counted *before* degenerate-function filtering. This index lets callers
    align cross-author functions originating from the same source row, for
    example to perform paired statistical tests against the function the
    LLM was prompted to produce (typically index 0).
    """
    metrics = []
    for idx, (func, snippet) in enumerate(get_function_snippets(code, language)):
        if func.nloc <= 0:
            continue

        analysis_snippet = snippet
        if language == "python":
            analysis_snippet = strip_python_docstrings(snippet)

        halstead = compute_halstead_metrics(analysis_snippet, language)

        # Degenerate: no operators or no operands -> Halstead is meaningless.
        if halstead["distinct_operators"] == 0 or halstead["distinct_operands"] == 0:
            continue

        max_nesting = compute_max_nesting_depth(analysis_snippet, language)
        mi = compute_maintainability_index(
            halstead["halstead_volume"],
            func.cyclomatic_complexity,
            func.nloc,
        )

        metrics.append(
            {
                "function_index_in_row": idx,
                "nloc": func.nloc,
                "ccn": func.cyclomatic_complexity,
                "parameter_count": func.parameter_count,
                "max_nesting_depth": max_nesting,
                "distinct_operators": halstead["distinct_operators"],
                "distinct_operands": halstead["distinct_operands"],
                "total_operators": halstead["total_operators"],
                "total_operands": halstead["total_operands"],
                "halstead_volume": halstead["halstead_volume"],
                "halstead_difficulty": halstead["halstead_difficulty"],
                "halstead_effort": halstead["halstead_effort"],
                "maintainability_index": mi,
            }
        )
    return metrics


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

METRIC_NAMES = [
    "nloc",
    "ccn",
    "parameter_count",
    "max_nesting_depth",
    "distinct_operators",
    "total_operators",
    "distinct_operands",
    "total_operands",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "maintainability_index",
]


def summarize_metric(values: list[float]) -> str:
    if not values:
        return "No values"
    if len(values) == 1:
        return f"Only one value: {values[0]:.2f}"
    arr = np.asarray(values, dtype=float)
    return (
        f"Avg: {float(arr.mean()):8.2f} | "
        f"Min: {float(arr.min()):8.2f} | "
        f"Median: {float(np.percentile(arr, 50)):8.2f} | "
        f"P90: {float(np.percentile(arr, 90)):8.2f} | "
        f"Max: {float(arr.max()):8.2f} | "
        f"Std: {float(arr.std(ddof=1)):8.2f}"
    )


def print_stats(metrics_by_field: dict[str, list[dict[str, float]]]):
    for field, metrics in metrics_by_field.items():
        print(f"\nStats for {field} ({len(metrics)} functions):")
        for name in METRIC_NAMES:
            values = [m[name] for m in metrics]
            print(f"  {name.upper():22} | {summarize_metric(values)}")

    all_metrics = [m for ms in metrics_by_field.values() for m in ms]
    print(f"\nAggregated Stats across ALL fields ({len(all_metrics)} functions):")
    for name in METRIC_NAMES:
        values = [m[name] for m in all_metrics]
        print(f"  {name.upper():22} | {summarize_metric(values)}")


# --------------------------------------------------------------------------
# Correct Cliff's delta (works regardless of SciPy's U convention)
# --------------------------------------------------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta in [-1, 1]. Positive means values in `a` tend to exceed
    values in `b`. Computed directly from rank sums to avoid the
    U-convention confusion in the previous implementation.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")

    combined = np.concatenate([a, b])
    # Average-rank ties.
    ranks = scipy_stats.rankdata(combined, method="average")
    rank_sum_a = float(ranks[:n1].sum())

    # U statistic for A: U_a = rank_sum_a - n1*(n1+1)/2
    u_a = rank_sum_a - n1 * (n1 + 1) / 2.0
    delta = (2.0 * u_a) / (n1 * n2) - 1.0

    # Clamp tiny floating-point excursions outside [-1, 1].
    return float(max(-1.0, min(1.0, delta)))


def cliffs_delta_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Bootstrap Cliff's delta with a 95% percentile confidence interval.
    Returns (delta_point_estimate, ci_lower, ci_upper).

    Resampling is *with replacement* on each input independently. At large
    n this is the standard non-parametric CI for Cliff's delta and is
    cheap (rank-based, no quadratic comparison).
    """
    if rng is None:
        rng = np.random.default_rng()
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")

    point = cliffs_delta(a, b)

    deltas = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        idx_a = rng.integers(0, n1, size=n1)
        idx_b = rng.integers(0, n2, size=n2)
        deltas[k] = cliffs_delta(a[idx_a], b[idx_b])

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    return point, ci_lo, ci_hi


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Wilcoxon signed-rank test on paired observations.
    Returns (statistic, p_value, dominance_proportion).
    Dominance proportion = fraction of pairs where a > b (ties ignored).
    """
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    nonzero = diff != 0
    if nonzero.sum() == 0:
        return float("nan"), 1.0, 0.5
    res = scipy_stats.wilcoxon(a[nonzero], b[nonzero], alternative="two-sided")
    dom = float((diff > 0).sum()) / float(nonzero.sum())
    return float(res.statistic), float(res.pvalue), dom


def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    if np.array_equal(a, b):
        return len(a) * len(b) / 2.0, 1.0
    res = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    p = float(res.pvalue) if not math.isnan(float(res.pvalue)) else 1.0
    return float(res.statistic), p


# --------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# --------------------------------------------------------------------------

def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """Return BH-adjusted q-values in the original order of `pvalues`."""
    n = len(pvalues)
    if n == 0:
        return []
    arr = np.asarray(pvalues, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    # q_i = min_{k>=i} (n/k * p_k), with ranks 1..n
    q = np.empty(n, dtype=float)
    min_so_far = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = n / rank * ranked[i]
        if val < min_so_far:
            min_so_far = val
        q[i] = min(1.0, min_so_far)
    # Re-order back to original positions.
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


# --------------------------------------------------------------------------
# Statistical tests (with function-level sampling and BH correction)
# --------------------------------------------------------------------------

def run_stats(
    metrics_by_field: dict[str, list[dict[str, float]]],
    sample_size: int | None,
    seed: int,
    bootstrap_resamples: int = 1000,
    paired_metrics: list[str] | None = None,
):
    """
    Pairwise statistical tests across code authors.

    Reports for every (metric, author-pair):
      - Welch's t-test (statistic, p, BH q)
      - Mann-Whitney U (statistic, p, BH q)
      - Cliff's delta (point estimate)
      - Cliff's delta with 95% bootstrap CI [lo, hi]

    If ``paired_metrics`` is non-empty, additionally runs the paired
    Wilcoxon signed-rank test for each (metric, pair) on functions that
    share the same row_id and function_index_in_row == 0. This gives a
    paired-effect-size estimate (dominance proportion = fraction of pairs
    where author A > author B) alongside the unpaired effect (Cliff's delta).

    Bootstrap is applied to the (sub-sampled) data, not to the full
    population, because the pairwise tables are otherwise too costly at
    full corpus size. Sampling is function-level: a single set of indices
    is drawn per field and reused across all metrics for that field, so
    within-function metric coupling is preserved.
    """
    rng = np.random.default_rng(seed)

    sampled_by_field: dict[str, list[dict[str, float]]] = {}
    for field, metrics in metrics_by_field.items():
        if sample_size and len(metrics) > sample_size:
            idx = rng.choice(len(metrics), size=sample_size, replace=False)
            sampled_by_field[field] = [metrics[i] for i in idx]
        else:
            sampled_by_field[field] = metrics

    print("\nStatistical tests (Welch, MWU, Cliff's delta + bootstrap CI)")
    print("p-values use SciPy; q-values use Benjamini-Hochberg FDR.")
    print(f"Bootstrap resamples: {bootstrap_resamples}")
    if sample_size:
        print(f"Sampling: up to {sample_size} functions per field (seed={seed})")

    fields = sorted(sampled_by_field.keys())
    pair_rows: list[dict] = []

    # A separate generator for bootstrap so seed math is independent
    # of test order.
    boot_rng = np.random.default_rng(seed + 1)

    for metric_name in METRIC_NAMES:
        values_by_field = {
            f: np.asarray([m[metric_name] for m in sampled_by_field[f]], dtype=float)
            for f in fields
        }
        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                f1, f2 = fields[i], fields[j]
                a = values_by_field[f1]
                b = values_by_field[f2]
                t_stat, t_p = welch_ttest(a, b)
                u_stat, u_p = mann_whitney_u(a, b)
                delta, ci_lo, ci_hi = cliffs_delta_bootstrap(
                    a, b, n_resamples=bootstrap_resamples, rng=boot_rng
                )
                pair_rows.append(
                    {
                        "metric": metric_name,
                        "f1": f1,
                        "f2": f2,
                        "t": t_stat,
                        "t_p": t_p,
                        "u": u_stat,
                        "u_p": u_p,
                        "delta": delta,
                        "delta_ci_lo": ci_lo,
                        "delta_ci_hi": ci_hi,
                    }
                )

    t_qs = benjamini_hochberg([r["t_p"] for r in pair_rows])
    u_qs = benjamini_hochberg([r["u_p"] for r in pair_rows])
    for row, tq, uq in zip(pair_rows, t_qs, u_qs):
        row["t_q"] = tq
        row["u_q"] = uq

    current_metric = None
    for row in pair_rows:
        if row["metric"] != current_metric:
            current_metric = row["metric"]
            print(f"\nMetric: {current_metric}")
        print(
            f"  {row['f1']} vs {row['f2']} | "
            f"t={row['t']:8.3f}, p={row['t_p']:.3e} (q={row['t_q']:.3e}) | "
            f"U={row['u']:12.1f}, p={row['u_p']:.3e} (q={row['u_q']:.3e}) | "
            f"delta={row['delta']:+.3f} "
            f"[{row['delta_ci_lo']:+.3f}, {row['delta_ci_hi']:+.3f}]"
        )

    # ----------------------------------------------------------------------
    # Paired tests on the unsampled (full) data, joined on row_id with
    # function_index_in_row == 0. This requires row_id and function_index_in_row
    # to be present in the per-function records (added in this version).
    # ----------------------------------------------------------------------
    paired_metrics = paired_metrics or []
    if not paired_metrics:
        return

    print("\nPaired Wilcoxon signed-rank tests")
    print("Pairing: same row_id with function_index_in_row == 0")
    print("Headline metrics:", ", ".join(paired_metrics))

    # Build a (row_id -> per-field metric vector) index for the headline metrics,
    # using the FULL metrics_by_field (not the sampled one). Pairing is done
    # before sampling because sampling would shuffle the row identifiers.
    primary_funcs: dict[str, dict[str, dict[str, float]]] = {f: {} for f in fields}
    for f in fields:
        for m in metrics_by_field[f]:
            if m.get("function_index_in_row") != 0:
                continue
            rid = m.get("row_id")
            if rid is None:
                continue
            # If the same row has multiple index-0 functions (shouldn't, but
            # defensively keep the first), skip duplicates.
            if rid not in primary_funcs[f]:
                primary_funcs[f][rid] = m

    paired_pair_rows: list[dict] = []
    for metric_name in paired_metrics:
        if metric_name not in METRIC_NAMES:
            print(f"[warn] paired metric '{metric_name}' not in METRIC_NAMES; skipping.")
            continue
        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                f1, f2 = fields[i], fields[j]
                shared = set(primary_funcs[f1]) & set(primary_funcs[f2])
                if not shared:
                    continue
                shared_sorted = sorted(shared)
                a = np.asarray(
                    [primary_funcs[f1][r][metric_name] for r in shared_sorted],
                    dtype=float,
                )
                b = np.asarray(
                    [primary_funcs[f2][r][metric_name] for r in shared_sorted],
                    dtype=float,
                )
                w_stat, w_p, dom = paired_wilcoxon(a, b)
                paired_pair_rows.append(
                    {
                        "metric": metric_name,
                        "f1": f1,
                        "f2": f2,
                        "n_pairs": len(shared_sorted),
                        "w": w_stat,
                        "w_p": w_p,
                        "dominance_a_over_b": dom,
                    }
                )

    if paired_pair_rows:
        w_qs = benjamini_hochberg([r["w_p"] for r in paired_pair_rows])
        for row, wq in zip(paired_pair_rows, w_qs):
            row["w_q"] = wq

        current_metric = None
        for row in paired_pair_rows:
            if row["metric"] != current_metric:
                current_metric = row["metric"]
                print(f"\nPaired metric: {current_metric}")
            print(
                f"  {row['f1']} vs {row['f2']} | "
                f"n_pairs={row['n_pairs']:>7d} | "
                f"W={row['w']:14.1f}, p={row['w_p']:.3e} (q={row['w_q']:.3e}) | "
                f"dominance({row['f1']}>{row['f2']})={row['dominance_a_over_b']:.3f}"
            )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute complexity metrics (NLOC, CCN, parameter count, nesting, "
            "Halstead, Maintainability Index) across C / Python / Java code "
            "fields in a JSONL dataset."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to the JSONL dataset.")
    parser.add_argument(
        "--fields",
        nargs="+",
        required=True,
        help="Field specs of the form field_name:language (e.g., human_code:c).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run pairwise statistical tests with BH correction.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Function-level sample size per field for statistical tests.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help=(
            "Number of bootstrap resamples for Cliff's delta confidence "
            "intervals (default 1000). Set to 0 to disable bootstrap, in "
            "which case only the point estimate is reported with NaN CIs."
        ),
    )
    parser.add_argument(
        "--paired-metrics",
        nargs="*",
        default=["nloc", "ccn", "max_nesting_depth", "halstead_volume",
                 "maintainability_index"],
        help=(
            "Metrics on which to additionally run paired Wilcoxon "
            "signed-rank tests, pairing functions by row_id with "
            "function_index_in_row == 0. Pass an empty list to skip "
            "paired analysis."
        ),
    )
    parser.add_argument(
        "--dump-metrics",
        type=str,
        default=None,
        help="Optional path to dump per-function metrics as JSONL (one line per function).",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="hexsha",
        help=(
            "Name of the field in the dataset that identifies each row "
            "(e.g., 'hexsha', 'hm_index'). Used to attach a row identifier "
            "to every per-function dump record so cross-author functions "
            "can be paired by row."
        ),
    )
    args = parser.parse_args()

    field_specs = [parse_field_spec(spec) for spec in args.fields]

    # Per-function metrics, with row id and per-row function index attached.
    metrics_by_field: dict[str, list[dict[str, float]]] = defaultdict(list)

    rows_missing_id = 0
    with open(args.dataset, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(tqdm(fh, desc="Processing"), start=1):
            try:
                item = json.loads(line)
                row_id = item.get(args.id_field)
                if row_id is None:
                    rows_missing_id += 1
                for field_name, language in field_specs:
                    code = item.get(field_name, "")
                    if code and code.strip():
                        per_func = analyze_code(code, language)
                        for row in per_func:
                            row["row_id"] = row_id
                        metrics_by_field[field_name].extend(per_func)
            except Exception as exc:
                print(f"[warn] skipping line {line_number}: {exc}")

    if rows_missing_id:
        print(
            f"[warn] {rows_missing_id} dataset rows had no '{args.id_field}' "
            "field; their functions will have row_id=null in the dump."
        )

    print_stats(metrics_by_field)

    if args.dump_metrics:
        with open(args.dump_metrics, "w", encoding="utf-8") as out:
            for field, metrics in metrics_by_field.items():
                for row in metrics:
                    # Column order: field, row_id, function_index_in_row, then metrics.
                    record = {
                        "field": field,
                        "row_id": row.get("row_id"),
                        "function_index_in_row": row.get("function_index_in_row"),
                    }
                    for k, v in row.items():
                        if k in ("row_id", "function_index_in_row"):
                            continue
                        record[k] = v
                    out.write(json.dumps(record) + "\n")

    if args.stats:
        run_stats(
            metrics_by_field,
            args.sample_size,
            args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            paired_metrics=args.paired_metrics,
        )


if __name__ == "__main__":
    main()
"""
python3 complexity_metrics_extended.py   --dataset ../1_Dataset/java_dataset_dsc_qwen_FINAL.jsonl   --fields human_code:java chatgpt_code:java
 dsc_code:java qwen_code:java   --stats   --dump-metrics java_per_function_metrics.jsonl > java_metrics.txt

python3 complexity_metrics_extended.py   --dataset ../1_Dataset/python_dataset_nodocs_dsc_qwen_FINAL.jsonl   --fields human_code:python chatgpt_code:python
 dsc_code:python qwen_code:python   --stats   --dump-metrics python_per_function_metrics.jsonl > python_metrics.txt
"""
