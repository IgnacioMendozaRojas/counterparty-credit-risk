"""Risk-factor models and joint Monte Carlo simulation.

Interest rates: one-factor Hull-White, fitted exactly to the initial curve.

    dr(t) = [theta(t) - a r(t)] dt + sigma dW1(t)

It is simulated through the Ornstein-Uhlenbeck state x(t) = r(t) - alpha(t), with
exact Gaussian transitions, so there is no discretisation bias in the short rate.
Zero-coupon bonds are available in closed form, which keeps future revaluation of
swaps cheap (Brigo & Mercurio, 2006, ch. 3.3).

FX: geometric Brownian motion under the domestic risk-neutral measure, with the
drift driven by the simulated domestic short rate and a flat foreign rate,
correlated with the rate factor.

    dS/S = [r(t) - r_f] dt + sigma_fx dW2(t),   d<W1, W2> = rho dt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import ZeroCurve


@dataclass(frozen=True)
class HullWhite:
    curve: ZeroCurve
    a: float = 0.03  # mean reversion
    sigma: float = 0.01  # normal (absolute) volatility of the short rate

    def B(self, t: float, T: np.ndarray) -> np.ndarray:
        return (1.0 - np.exp(-self.a * (T - t))) / self.a

    def _V(self, tau: np.ndarray) -> np.ndarray:
        """Variance of the integral of x over a horizon tau."""
        a, s = self.a, self.sigma
        return (s * s / (a * a)) * (
            tau + 2.0 / a * np.exp(-a * tau) - 0.5 / a * np.exp(-2.0 * a * tau) - 1.5 / a
        )

    def alpha(self, t: np.ndarray) -> np.ndarray:
        a, s = self.a, self.sigma
        return self.curve.inst_forward(t) + 0.5 * (s / a) ** 2 * (1.0 - np.exp(-a * t)) ** 2

    def bond(self, t: float, T: np.ndarray, x: np.ndarray) -> np.ndarray:
        """P(t, T_j) for every path. Returns shape (paths, len(T))."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if t <= 0.0:
            return np.broadcast_to(self.curve.df(T), (x.shape[0], T.size)).copy()
        ratio = self.curve.df(T) / self.curve.df(t)
        A = 0.5 * (self._V(T - t) - self._V(T) + self._V(np.array(t)))
        return ratio * np.exp(A - np.outer(x, self.B(t, T)))


@dataclass(frozen=True)
class FXModel:
    spot: float  # domestic units per one unit of foreign currency
    vol: float = 0.10
    foreign_rate: float = 0.02
    corr: float = 0.0  # correlation with the domestic rate factor


@dataclass
class Simulation:
    """Simulated paths on a uniform time grid. Arrays are (paths, steps + 1)."""

    times: np.ndarray
    x: np.ndarray
    short_rate: np.ndarray
    numeraire: np.ndarray  # bank account B(t) = exp(integral of r)
    fx: np.ndarray | None
    hw: HullWhite
    fx_model: FXModel | None

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    @property
    def n_paths(self) -> int:
        return self.x.shape[0]

    def index(self, t: float) -> int:
        """Grid index of a date, snapped to the nearest step."""
        return int(round(t / self.dt))

    def discount(self) -> np.ndarray:
        """Pathwise stochastic discount factor D(0, t) = 1 / B(t)."""
        return 1.0 / self.numeraire

    def bond(self, i: int, maturities: np.ndarray) -> np.ndarray:
        return self.hw.bond(self.times[i], maturities, self.x[:, i])


def simulate(
    hw: HullWhite,
    horizon: float,
    n_paths: int = 5_000,
    steps_per_year: int = 52,
    fx: FXModel | None = None,
    seed: int | None = 42,
    antithetic: bool = True,
) -> Simulation:
    """Simulate the short rate (and optionally FX) jointly on a uniform grid."""
    n_steps = int(round(horizon * steps_per_year))
    dt = 1.0 / steps_per_year
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    rng = np.random.default_rng(seed)

    half = (n_paths + 1) // 2 if antithetic else n_paths
    z1 = rng.standard_normal((half, n_steps))
    z2 = rng.standard_normal((half, n_steps))
    if antithetic:
        z1 = np.vstack([z1, -z1])[:n_paths]
        z2 = np.vstack([z2, -z2])[:n_paths]

    # Exact OU transition for x
    decay = np.exp(-hw.a * dt)
    step_sd = hw.sigma * np.sqrt((1.0 - decay**2) / (2.0 * hw.a))
    x = np.zeros((n_paths, n_steps + 1))
    for k in range(n_steps):
        x[:, k + 1] = x[:, k] * decay + step_sd * z1[:, k]

    r = x + hw.alpha(times)
    integral = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * dt, axis=1)], axis=1
    )
    numeraire = np.exp(integral)

    fx_paths = None
    if fx is not None:
        w2 = fx.corr * z1 + np.sqrt(1.0 - fx.corr**2) * z2
        drift = (r[:, :-1] - fx.foreign_rate - 0.5 * fx.vol**2) * dt
        log_s = np.log(fx.spot) + np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(drift + fx.vol * np.sqrt(dt) * w2, axis=1)], axis=1
        )
        fx_paths = np.exp(log_s)

    return Simulation(times, x, r, numeraire, fx_paths, hw, fx)
