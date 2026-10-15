"""Reusable A/B test statistics: sanity checks, frequentist, bootstrap, Bayesian, power and peeking."""
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------- sanity checks ----------
def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> dict:
    """Sample Ratio Mismatch: did users split between groups as designed? (chi-square goodness of fit)"""
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    chi2, p = stats.chisquare([n_a, n_b], expected)
    return {"n_a": n_a, "n_b": n_b, "share_a": n_a / total, "chi2": float(chi2), "p_value": float(p)}


# ---------- frequentist ----------
@dataclass
class ProportionTest:
    rate_a: float
    rate_b: float
    diff: float              # b - a, in absolute terms
    ci_low: float
    ci_high: float
    relative_lift: float     # (b - a) / a
    z: float
    p_value: float


def two_proportion_ztest(success_a: int, n_a: int, success_b: int, n_b: int, alpha: float = 0.05) -> ProportionTest:
    pa, pb = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pooled
    p = 2 * stats.norm.sf(abs(z))
    # Unpooled standard error for the confidence interval of the difference.
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    zc = stats.norm.ppf(1 - alpha / 2)
    return ProportionTest(pa, pb, pb - pa, pb - pa - zc * se, pb - pa + zc * se, (pb - pa) / pa, float(z), float(p))


# ---------- bootstrap ----------
def bootstrap_diff(a: np.ndarray, b: np.ndarray, stat=np.mean, n_boot: int = 5000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of stat(b) - stat(a)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = stat(rng.choice(b, len(b))) - stat(rng.choice(a, len(a)))
    return out


# ---------- Bayesian ----------
def bayesian_proportions(success_a, n_a, success_b, n_b, draws: int = 200_000, seed: int = 0) -> dict:
    """Beta(1,1) prior -> Beta posterior per group. Reports P(A better) and the expected loss of each choice."""
    rng = np.random.default_rng(seed)
    a = rng.beta(1 + success_a, 1 + n_a - success_a, draws)
    b = rng.beta(1 + success_b, 1 + n_b - success_b, draws)
    return {
        "prob_a_better": float((a > b).mean()),
        # Expected retention we give up (absolute) if we pick this version and it is actually the worse one.
        "expected_loss_choose_a": float(np.maximum(b - a, 0).mean()),
        "expected_loss_choose_b": float(np.maximum(a - b, 0).mean()),
        "diff_ci95": [float(np.quantile(b - a, 0.025)), float(np.quantile(b - a, 0.975))],
    }


# ---------- power ----------
def required_n_per_group(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per group needed to detect an absolute change `mde_abs` from baseline rate `p_base`."""
    p2 = p_base + mde_abs
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p_base + p2) / 2
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2 / mde_abs ** 2
    return int(np.ceil(n))


def minimum_detectable_effect(p_base: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute change detectable with this sample (normal approximation)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za + zb) * np.sqrt(2 * p_base * (1 - p_base) / n_per_group))


# ---------- peeking ----------
def peeking_false_positive_rate(p: float, n_per_group: int, looks: int, sims: int = 2000, alpha: float = 0.05,
                                seed: int = 0) -> dict:
    """Simulate A/A tests (no real difference). Stop at the first look with p < alpha.

    Shows how checking results every day inflates false positives above the nominal 5%.
    """
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_group / looks, n_per_group, looks).astype(int)
    fp_once = fp_peek = 0
    for _ in range(sims):
        a = rng.random(n_per_group) < p
        b = rng.random(n_per_group) < p
        ca, cb = np.cumsum(a), np.cumsum(b)
        significant = [two_proportion_ztest(ca[n - 1], n, cb[n - 1], n, alpha).p_value < alpha for n in checkpoints]
        fp_peek += any(significant)
        fp_once += significant[-1]
    return {"looks": looks, "false_positive_single_look": fp_once / sims, "false_positive_with_peeking": fp_peek / sims}
