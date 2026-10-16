"""Unilateral CVA and DVA, assuming independence between exposure and default
(no wrong-way risk):

    CVA = (1 - R_c) * sum_i EE*(t_i) * PD_c(t_{i-1}, t_i)
    DVA = (1 - R_b) * sum_i |ENE*(t_i)| * PD_b(t_{i-1}, t_i)

EE* and ENE* are discounted expected exposures. Exposure is taken at the end of each
interval, a common and slightly conservative convention.
"""

from __future__ import annotations

import numpy as np

from .credit import CreditCurve
from .exposure import ExposureProfile


def _integrate(times: np.ndarray, exposure: np.ndarray, credit: CreditCurve) -> float:
    pd = credit.default_prob(times[:-1], times[1:])
    return float((1.0 - credit.recovery) * np.sum(exposure[1:] * pd))


def cva(profile: ExposureProfile, counterparty: CreditCurve) -> float:
    return _integrate(profile.times, profile.ee_discounted, counterparty)


def dva(profile: ExposureProfile, own: CreditCurve) -> float:
    return _integrate(profile.times, -profile.ene_discounted, own)


def cs01(profile: ExposureProfile, counterparty: CreditCurve, bump_bp: float = 1.0) -> float:
    """Change in CVA for a +1bp move in the counterparty spread."""
    return (cva(profile, counterparty.bumped(bump_bp)) - cva(profile, counterparty)) / bump_bp
