"""Instruments revalued on every simulated path and grid date.

`mtm(sim)` returns the mark-to-market in domestic currency, shape (paths, steps + 1),
from the bank's point of view. A trade is worth zero after its last cash flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import ZeroCurve
from .models import Simulation


@dataclass(frozen=True)
class InterestRateSwap:
    """Vanilla single-curve fixed-for-floating swap.

    payer=True: the bank pays fixed and receives floating.
    The floating rate of each period is fixed at the start of the period from the
    simulated curve, so exposure correctly reflects already-fixed coupons.
    """

    notional: float
    fixed_rate: float
    maturity: float
    frequency: int = 2  # payments per year, both legs
    payer: bool = True
    start: float = 0.0

    @property
    def pay_dates(self) -> np.ndarray:
        n = int(round((self.maturity - self.start) * self.frequency))
        return self.start + np.arange(1, n + 1) / self.frequency

    @staticmethod
    def par_rate(
        curve: ZeroCurve, maturity: float, frequency: int = 2, start: float = 0.0
    ) -> float:
        dates = start + np.arange(1, int(round((maturity - start) * frequency)) + 1) / frequency
        accrual = np.diff(np.concatenate([[start], dates]))
        return float((curve.df(start) - curve.df(dates[-1])) / np.sum(accrual * curve.df(dates)))

    def mtm(self, sim: Simulation) -> np.ndarray:
        dates = self.pay_dates
        resets = np.concatenate([[self.start], dates[:-1]])
        accrual = dates - resets
        reset_idx = [sim.index(t) for t in resets]
        values = np.zeros((sim.n_paths, sim.times.size))
        sign = 1.0 if self.payer else -1.0

        # Floating rate fixed at each reset date, pathwise
        fixings = np.empty((sim.n_paths, dates.size))
        for j, (k, T) in enumerate(zip(reset_idx, dates, strict=True)):
            p = sim.bond(k, np.array([T]))[:, 0]
            fixings[:, j] = (1.0 / p - 1.0) / accrual[j]

        for i, t in enumerate(sim.times):
            if t < self.start - 1e-12 or t >= dates[-1] - 1e-12:
                continue
            j = int(np.searchsorted(dates, t + 1e-12, side="right"))  # current period
            remaining = dates[j:]
            bonds = sim.bond(i, remaining)
            fixed_leg = self.fixed_rate * bonds @ accrual[j:]
            if i >= reset_idx[j]:
                floating = (1.0 + accrual[j] * fixings[:, j]) * bonds[:, 0] - bonds[:, -1]
            else:  # forward-starting, not yet fixed
                floating = sim.bond(i, np.array([resets[j]]))[:, 0] - bonds[:, -1]
            values[:, i] = sign * self.notional * (floating - fixed_leg)
        return values


@dataclass(frozen=True)
class FXForward:
    """The bank buys `notional` units of foreign currency at `strike` on `maturity`."""

    notional: float
    strike: float
    maturity: float
    buy_foreign: bool = True

    def mtm(self, sim: Simulation) -> np.ndarray:
        if sim.fx is None or sim.fx_model is None:
            raise ValueError("an FX forward needs a simulation with an FX model")
        values = np.zeros((sim.n_paths, sim.times.size))
        sign = 1.0 if self.buy_foreign else -1.0
        rf = sim.fx_model.foreign_rate
        T = np.array([self.maturity])
        for i, t in enumerate(sim.times):
            if t >= self.maturity - 1e-12:
                break
            p_dom = sim.bond(i, T)[:, 0]
            p_for = np.exp(-rf * (self.maturity - t))
            values[:, i] = sign * self.notional * (sim.fx[:, i] * p_for - self.strike * p_dom)
        return values
