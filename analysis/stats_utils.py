"""Statistical helpers used by the revision analysis.

All intervals are two-sided 95% unless stated otherwise.

* ``wilson``      -- Wilson score interval for a binomial proportion.
* ``newcombe``    -- Newcombe (1998, method 10) hybrid-score interval for the
                     difference of two proportions. It assumes independent
                     samples; when both models answer the *same* items and their
                     correctness is positively correlated (the usual case), this
                     interval is conservative relative to a paired analysis.
* ``holm``        -- Holm step-down adjustment of a family of p-values.
* ``two_prop_p``  -- two-sided p-value of the pooled two-proportion z-test,
                     used only to order comparisons for the Holm procedure.
"""
from __future__ import annotations

import math

from scipy.stats import norm

Z95 = norm.ppf(0.975)


def wilson(p: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(p1: float, n1: int, p2: float, n2: int, z: float = Z95) -> tuple[float, float]:
    """Interval for p1 - p2."""
    l1, u1 = wilson(p1, n1, z)
    l2, u2 = wilson(p2, n2, z)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lower, upper


def two_prop_p(p1: float, n1: int, p2: float, n2: int) -> float:
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return 2 * (1 - norm.cdf(abs(p1 - p2) / se))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
