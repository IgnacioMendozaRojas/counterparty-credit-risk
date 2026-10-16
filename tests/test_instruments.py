import numpy as np
import pytest
from scipy.stats import norm

from ccr import FXForward, FXModel, HullWhite, InterestRateSwap, ZeroCurve, simulate


def remaining_cashflows_pv(curve, swap, t):
    """Analytic PV at 0 of the swap cash flows paid after t (single curve)."""
    dates = swap.pay_dates
    resets = np.concatenate([[swap.start], dates[:-1]])
    keep = dates > t + 1e-9
    floating = curve.df(resets[keep]) - curve.df(dates[keep])
    fixed = swap.fixed_rate * (dates[keep] - resets[keep]) * curve.df(dates[keep])
    sign = 1.0 if swap.payer else -1.0
    return sign * swap.notional * float(np.sum(floating - fixed))


def test_par_swap_is_worth_zero(sim, curve):
    k = InterestRateSwap.par_rate(curve, 10.0)
    v = InterestRateSwap(1e8, k, 10.0).mtm(sim)
    assert abs(v[:, 0].mean()) < 1.0  # deterministic at t=0


@pytest.mark.parametrize("t", [0.5, 2.25, 6.0, 9.6])
@pytest.mark.parametrize("payer", [True, False])
def test_expected_discounted_swap_value(sim, curve, t, payer):
    """E[D(0,t) V(t)] equals the PV today of the cash flows still to be paid."""
    swap = InterestRateSwap(1e8, 0.04, 10.0, payer=payer)
    i = sim.index(t)
    mc = (sim.discount()[:, i] * swap.mtm(sim)[:, i]).mean()
    exact = remaining_cashflows_pv(curve, swap, sim.times[i])
    assert mc == pytest.approx(exact, abs=0.002 * swap.notional)


def test_payer_and_receiver_offset(sim):
    payer = InterestRateSwap(5e7, 0.035, 5.0, payer=True).mtm(sim)
    receiver = InterestRateSwap(5e7, 0.035, 5.0, payer=False).mtm(sim)
    np.testing.assert_allclose(payer + receiver, 0.0, atol=1e-6)


def test_swap_is_dead_after_maturity(sim):
    v = InterestRateSwap(1e8, 0.035, 5.0).mtm(sim)
    assert np.all(v[:, sim.index(5.0) :] == 0.0)


def test_forward_starting_swap(sim, curve):
    k = InterestRateSwap.par_rate(curve, 7.0, start=2.0)
    swap = InterestRateSwap(1e8, k, 7.0, start=2.0)
    assert abs(swap.mtm(sim)[:, 0].mean()) < 1.0


def test_fx_forward_value_today(sim, curve):
    fwd = FXForward(1e7, 1.12, 2.0)
    expected = 1e7 * (1.10 * np.exp(-0.02 * 2.0) - 1.12 * float(curve.df(2.0)))
    assert fwd.mtm(sim)[0, 0] == pytest.approx(expected)


def test_fx_forward_expected_exposure_matches_black():
    """With (almost) deterministic rates, EE of an FX forward is a call on the forward."""
    curve = ZeroCurve.flat(0.03)
    hw = HullWhite(curve, a=0.05, sigma=1e-8)
    fx = FXModel(spot=1.0, vol=0.15, foreign_rate=0.01)
    s = simulate(hw, 3.0, n_paths=40_000, fx=fx, seed=11)
    fwd = FXForward(1.0, strike=1.02, maturity=3.0)
    t = 1.5
    i = s.index(t)
    ee_disc = (s.discount()[:, i] * np.maximum(fwd.mtm(s)[:, i], 0)).mean()

    # Discounted to 0: value of a call on S(t) e^{-rf (T-t)} - K e^{-r (T-t)}
    T, r, rf, vol, K = 3.0, 0.03, 0.01, 0.15, 1.02
    k_eff = K * np.exp(-(r - rf) * (T - t))  # strike in units of S(t)
    fwd_s = 1.0 * np.exp((r - rf) * t)
    d1 = (np.log(fwd_s / k_eff) + 0.5 * vol**2 * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    call = np.exp(-r * t) * (fwd_s * norm.cdf(d1) - k_eff * norm.cdf(d2))
    black = np.exp(-rf * (T - t)) * call
    assert ee_disc == pytest.approx(black, rel=0.02)


def test_fx_forward_needs_fx_model(hw):
    s = simulate(hw, 1.0, n_paths=10)
    with pytest.raises(ValueError):
        FXForward(1.0, 1.0, 0.5).mtm(s)
