"""Exposure metrics from simulated netting-set values.

EE(t)     expected exposure, E[max(V, 0)]
ENE(t)    expected negative exposure, E[min(V, 0)] (used for DVA)
PFE_q(t)  potential future exposure, q-quantile of max(V, 0)
EPE       time average of EE over the first year
EEE(t)    effective EE, running maximum of EE (non-decreasing)
EEPE      time average of EEE over the first year: the Basel IMM exposure measure
          (BCBS, "Basel III: CVA risk and counterparty credit risk", CRE53)

Discounted EE*(t) = E[D(0, t) max(V, 0)] uses the pathwise stochastic discount
factor and is the quantity that enters CVA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Simulation


@dataclass
class ExposureProfile:
    times: np.ndarray
    ee: np.ndarray
    ene: np.ndarray
    pfe: np.ndarray
    ee_discounted: np.ndarray
    ene_discounted: np.ndarray
    quantile: float

    @property
    def effective_ee(self) -> np.ndarray:
        return np.maximum.accumulate(self.ee)

    def _one_year_average(self, series: np.ndarray) -> float:
        mask = self.times <= 1.0 + 1e-12
        t, s = self.times[mask], series[mask]
        if t[-1] <= 0:
            return float(s[0])
        return float(np.trapezoid(s, t) / t[-1])

    @property
    def epe(self) -> float:
        return self._one_year_average(self.ee)

    @property
    def eepe(self) -> float:
        return self._one_year_average(self.effective_ee)

    @property
    def peak_pfe(self) -> float:
        return float(self.pfe.max())

    def summary(self) -> dict[str, float]:
        return {
            "EPE (1y)": self.epe,
            "EEPE (1y)": self.eepe,
            f"Peak PFE {self.quantile:.1%}": self.peak_pfe,
            "Peak EE": float(self.ee.max()),
        }


def exposure_profile(
    values: np.ndarray, sim: Simulation, quantile: float = 0.975
) -> ExposureProfile:
    pos = np.maximum(values, 0.0)
    neg = np.minimum(values, 0.0)
    disc = sim.discount()
    return ExposureProfile(
        times=sim.times,
        ee=pos.mean(axis=0),
        ene=neg.mean(axis=0),
        pfe=np.quantile(pos, quantile, axis=0),
        ee_discounted=(disc * pos).mean(axis=0),
        ene_discounted=(disc * neg).mean(axis=0),
        quantile=quantile,
    )
