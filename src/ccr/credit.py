"""Default probabilities implied from a CDS spread.

With a flat hazard rate the "credit triangle" gives lambda = s / (1 - R), and the
survival probability is S(t) = exp(-lambda t). Simple, transparent, and enough to
show the mechanics; a bootstrapped term structure would slot in behind `survival`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CreditCurve:
    spread_bp: float
    recovery: float = 0.40

    @property
    def hazard(self) -> float:
        return self.spread_bp / 1e4 / (1.0 - self.recovery)

    def survival(self, t: ArrayLike) -> np.ndarray:
        return np.exp(-self.hazard * np.asarray(t, dtype=float))

    def default_prob(self, t0: ArrayLike, t1: ArrayLike) -> np.ndarray:
        """Probability of default in (t0, t1]."""
        return self.survival(t0) - self.survival(t1)

    def bumped(self, bp: float) -> CreditCurve:
        return CreditCurve(self.spread_bp + bp, self.recovery)
