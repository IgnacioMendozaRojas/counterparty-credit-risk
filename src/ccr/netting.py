"""A netting set: trades with one counterparty under one legal agreement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .collateral import CSA
from .models import Simulation


class Trade(Protocol):
    def mtm(self, sim: Simulation) -> np.ndarray: ...


@dataclass
class NettingSet:
    trades: list[Trade] = field(default_factory=list)
    csa: CSA | None = None

    def values(self, sim: Simulation) -> np.ndarray:
        """Netted mark-to-market, before collateral."""
        return np.sum([t.mtm(sim) for t in self.trades], axis=0)

    def exposure_values(self, sim: Simulation) -> np.ndarray:
        """Netted value after collateral: what is lost if the counterparty defaults."""
        v = self.values(sim)
        if self.csa is None:
            return v
        return v - self.csa.collateral(v, sim.dt)
