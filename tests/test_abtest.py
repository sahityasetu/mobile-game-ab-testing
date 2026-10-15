import numpy as np
from statsmodels.stats.proportion import proportions_ztest

from src.abtest import (bayesian_proportions, minimum_detectable_effect, required_n_per_group, srm_check,
                        two_proportion_ztest)


def test_ztest_matches_statsmodels():
    ours = two_proportion_ztest(450, 1000, 400, 1000)
    z, p = proportions_ztest([400, 450], [1000, 1000])
    assert np.isclose(ours.z, z) and np.isclose(ours.p_value, p)
    assert ours.ci_low < ours.diff < ours.ci_high < 0


def test_srm_flags_unbalanced_split():
    assert srm_check(5000, 5000)["p_value"] > 0.99
    assert srm_check(5300, 4700)["p_value"] < 0.001


def test_power_and_mde_are_consistent():
    n = required_n_per_group(0.2, 0.01)
    assert 20_000 < n < 30_000            # textbook value for 20% -> 21% is ~25.6k per group
    assert abs(minimum_detectable_effect(0.2, n) - 0.01) < 0.001


def test_bayesian_probability_direction():
    r = bayesian_proportions(500, 1000, 450, 1000)
    assert r["prob_a_better"] > 0.95
    assert r["expected_loss_choose_a"] < r["expected_loss_choose_b"]
