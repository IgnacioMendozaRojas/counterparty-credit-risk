"""Variation margin under a simplified CSA.

Collateral at t is called on the netted value observed one margin period of risk
(MPR) earlier, so the residual exposure is the move in value over the MPR, which
is the standard way to model the "gap risk" of a collateralised netting set
(Gregory, 2020, ch. 15).

    C(t) = max(V(t - MPR) - TH_cpty, 0) - max(-V(t - MPR) - TH_own, 0)

A minimum transfer amount (MTA) is applied pathwise: collateral only moves when the
required change is at least the MTA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSA:
    threshold_cpty: float = 0.0  # unsecured exposure we accept to the counterparty
    threshold_own: float = 0.0  # unsecured exposure the counterparty accepts to us
    mta: float = 0.0
    mpr_days: int = 10  # margin period of risk, business days
    initial_margin: float = 0.0  # received, fixed amount

    def collateral(self, values: np.ndarray, dt: float) -> np.ndarray:
        """Collateral held by the bank (positive = received), same shape as values."""
        lag = max(int(round(self.mpr_days / 252 / dt)), 0)
        observed = np.zeros_like(values)
        observed[:, lag:] = values[:, : values.shape[1] - lag]
        target = np.maximum(observed - self.threshold_cpty, 0.0) - np.maximum(
            -observed - self.threshold_own, 0.0
        )
        if self.mta <= 0:
            return target + self.initial_margin
        held = np.zeros_like(values)
        for k in range(1, values.shape[1]):
            move = target[:, k] - held[:, k - 1]
            held[:, k] = np.where(np.abs(move) >= self.mta, target[:, k], held[:, k - 1])
        return held + self.initial_margin
