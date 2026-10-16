"""Discount curve built from zero-rate pillars (continuous compounding)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ZeroCurve:
    """Zero rates at pillar tenors, linearly interpolated, flat beyond the ends."""

    tenors: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.rates) or not self.tenors:
            raise ValueError("tenors and rates must be non-empty and the same length")
        if any(b <= a for a, b in zip(self.tenors, self.tenors[1:], strict=False)):
            raise ValueError("tenors must be strictly increasing")

    @classmethod
    def flat(cls, rate: float) -> ZeroCurve:
        return cls((1.0,), (rate,))

    def zero_rate(self, t: ArrayLike) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self.tenors, self.rates)

    def df(self, t: ArrayLike) -> np.ndarray:
        """Discount factor P(0, t)."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def inst_forward(self, t: ArrayLike, h: float = 1e-4) -> np.ndarray:
        """Instantaneous forward rate f(0, t) = -d ln P(0, t) / dt (central difference)."""
        t = np.asarray(t, dtype=float)
        lo = np.maximum(t - h, 0.0)
        hi = t + h
        return -(np.log(self.df(hi)) - np.log(self.df(lo))) / (hi - lo)

    def shifted(self, bp: float) -> ZeroCurve:
        """Parallel shift in basis points, used for rate sensitivities."""
        return ZeroCurve(self.tenors, tuple(r + bp / 1e4 for r in self.rates))
