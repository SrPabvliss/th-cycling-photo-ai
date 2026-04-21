"""Statistical tests for model comparison.

Implements paired bootstrap, McNemar, Wilcoxon as per evaluation_methodology.md.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for comparing two models.

    Returns delta mean, CI95%, and p-value.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    deltas = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = scores_a[idx].mean() - scores_b[idx].mean()

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0)) * 2  # two-tailed

    return {
        "delta_mean": float(deltas.mean()),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": min(p_value, 1.0),
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """McNemar's exact test for paired binary classification.

    Args:
        correct_a: Binary array (1 = model A correct, 0 = incorrect).
        correct_b: Binary array (1 = model B correct, 0 = incorrect).
    """
    # Contingency: b=A_correct_B_wrong, c=A_wrong_B_correct
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    result = stats.binomtest(b, b + c, 0.5)

    return {
        "statistic": float(b),
        "p_value": float(result.pvalue),
        "b": b,
        "c": c,
    }


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired AP values per class."""
    if len(values_a) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "note": "too few classes for Wilcoxon"}

    stat, p_value = stats.wilcoxon(values_a, values_b)
    return {"statistic": float(stat), "p_value": float(p_value)}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values from pairwise comparisons.
        alpha: Family-wise error rate.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [{}] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        results[orig_idx] = {
            "raw_p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
            "rank": rank + 1,
        }

    return results
