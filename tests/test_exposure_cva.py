import numpy as np
import pytest

from ccr import (
    CSA,
    CreditCurve,
    FXForward,
    InterestRateSwap,
    NettingSet,
    cva,
    dva,
    exposure_profile,
)
from ccr.cva import cs01


@pytest.fixture(scope="module")
def trades(curve):
    k10 = InterestRateSwap.par_rate(curve, 10.0)
    return [
        InterestRateSwap(1e8, k10, 10.0, payer=True),
        InterestRateSwap(6e7, 0.035, 5.0, payer=False),
        FXForward(2e7, 1.12, 2.0),
    ]


def test_exposure_metrics_are_consistent(sim, trades):
    p = exposure_profile(NettingSet(trades).values(sim), sim)
    assert np.all(p.ee >= 0) and np.all(p.ene <= 0)
    assert p.peak_pfe > p.ee.max()
    assert np.all(np.diff(p.effective_ee) >= 0)
    assert p.eepe >= p.epe
    assert set(p.summary()) == {"EPE (1y)", "EEPE (1y)", "Peak PFE 97.5%", "Peak EE"}


def test_netting_never_increases_exposure(sim, trades):
    netted = exposure_profile(NettingSet(trades).values(sim), sim).ee
    standalone = sum(exposure_profile(t.mtm(sim), sim).ee for t in trades)
    assert np.all(netted <= standalone + 1e-6)
    assert netted.sum() < standalone.sum()


def test_collateral_reduces_exposure(sim, trades):
    plain = NettingSet(trades)
    secured = NettingSet(trades, CSA(mpr_days=10))
    ee_plain = exposure_profile(plain.exposure_values(sim), sim).ee
    ee_secured = exposure_profile(secured.exposure_values(sim), sim).ee
    assert ee_secured.max() < 0.35 * ee_plain.max()


def test_threshold_and_mta_sit_between(sim, trades):
    full = exposure_profile(NettingSet(trades, CSA()).exposure_values(sim), sim).epe
    loose = NettingSet(trades, CSA(threshold_cpty=2e6, mta=5e5))
    partial = exposure_profile(loose.exposure_values(sim), sim).epe
    none = exposure_profile(NettingSet(trades).values(sim), sim).epe
    assert full < partial < none


def test_initial_margin_is_held(sim, trades):
    ns = NettingSet(trades, CSA(initial_margin=1e6))
    c = ns.csa.collateral(ns.values(sim), sim.dt)
    assert np.all(c[:, 0] == 1e6)


def test_cva_closed_form_for_constant_exposure(sim):
    """With constant discounted EE = E, CVA = (1-R) E (1 - S(T))."""
    from ccr.exposure import ExposureProfile

    t = sim.times
    flat = np.full(t.size, 1e6)
    p = ExposureProfile(t, flat, -flat, flat, flat, -flat, 0.975)
    c = CreditCurve(200, recovery=0.4)
    expected = 0.6 * 1e6 * (1 - c.survival(t[-1]))
    assert cva(p, c) == pytest.approx(float(expected))
    assert dva(p, c) == pytest.approx(float(expected))


def test_cva_increases_with_spread(sim, trades):
    p = exposure_profile(NettingSet(trades).values(sim), sim)
    low, high = cva(p, CreditCurve(100)), cva(p, CreditCurve(300))
    assert 0 < low < high
    assert cs01(p, CreditCurve(100)) > 0


def test_credit_curve():
    c = CreditCurve(120, recovery=0.4)
    assert c.hazard == pytest.approx(0.02)
    assert float(c.default_prob(0, 1)) == pytest.approx(1 - np.exp(-0.02))
