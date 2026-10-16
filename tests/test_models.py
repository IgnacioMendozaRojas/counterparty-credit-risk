"""The simulation must be arbitrage-free with respect to the initial curve."""

import numpy as np
import pytest

from ccr import HullWhite, ZeroCurve, simulate


@pytest.mark.parametrize("T", [1.0, 5.0, 10.0])
def test_discount_factor_is_reproduced(sim, curve, T):
    i = sim.index(T)
    assert sim.discount()[:, i].mean() == pytest.approx(float(curve.df(T)), rel=1e-3)


@pytest.mark.parametrize(("t", "T"), [(1.0, 3.0), (3.0, 7.0), (5.0, 15.0)])
def test_discounted_bond_is_a_martingale(sim, curve, t, T):
    i = sim.index(t)
    pv = (sim.discount()[:, i] * sim.bond(i, np.array([T]))[:, 0]).mean()
    assert pv == pytest.approx(float(curve.df(T)), rel=1e-3)


def test_bond_formula_at_time_zero_matches_curve(hw, curve):
    T = np.array([0.5, 2.0, 10.0])
    p = hw.bond(0.0, T, np.zeros(3))
    np.testing.assert_allclose(p[0], curve.df(T))


def test_short_rate_distribution(hw):
    """x(t) is Gaussian with variance sigma^2 (1 - e^{-2at}) / 2a."""
    s = simulate(hw, 5.0, n_paths=20_000, seed=1)
    t = 5.0
    expected = hw.sigma**2 * (1 - np.exp(-2 * hw.a * t)) / (2 * hw.a)
    assert s.x[:, -1].var() == pytest.approx(expected, rel=0.03)
    assert abs(s.x[:, -1].mean()) < 1e-4  # antithetic


def test_discounted_fx_forward_is_a_martingale(sim, curve):
    """E[D(0,t) S(t)] = S0 * exp(-r_f t)."""
    i = sim.index(4.0)
    pv = (sim.discount()[:, i] * sim.fx[:, i]).mean()
    assert pv == pytest.approx(1.10 * np.exp(-0.02 * 4.0), rel=3e-3)


def test_curve_validation():
    with pytest.raises(ValueError):
        ZeroCurve((1.0, 1.0), (0.01, 0.02))
    flat = ZeroCurve.flat(0.05)
    assert float(flat.df(2.0)) == pytest.approx(np.exp(-0.1))
    assert float(flat.shifted(100).zero_rate(3.0)) == pytest.approx(0.06)
    assert HullWhite(flat).B(0.0, np.array([0.0]))[0] == 0.0
