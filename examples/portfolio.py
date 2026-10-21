"""End-to-end example: exposure and CVA of a small netting set, with and without a CSA.

    python examples/portfolio.py            # prints results, writes docs/ figures

All market data and trades are illustrative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ccr import (
    CSA,
    CreditCurve,
    FXForward,
    FXModel,
    HullWhite,
    InterestRateSwap,
    NettingSet,
    ZeroCurve,
    cva,
    dva,
    exposure_profile,
    simulate,
)
from ccr.cva import cs01

DOCS = Path(__file__).resolve().parents[1] / "docs"

# ── Market data (illustrative) ─────────────────────────────────────────
curve = ZeroCurve((0.25, 1, 2, 5, 10, 20), (0.030, 0.032, 0.034, 0.036, 0.038, 0.039))
hw = HullWhite(curve, a=0.03, sigma=0.010)
fx = FXModel(spot=1.10, vol=0.10, foreign_rate=0.020, corr=-0.20)
counterparty = CreditCurve(spread_bp=150, recovery=0.40)
own = CreditCurve(spread_bp=80, recovery=0.40)

# ── Netting set ────────────────────────────────────────────────────────
trades = [
    InterestRateSwap(100e6, InterestRateSwap.par_rate(curve, 10.0), 10.0, payer=True),
    InterestRateSwap(60e6, 0.035, 5.0, payer=False),
    FXForward(20e6, strike=1.12, maturity=2.0),
]
SCENARIOS = {
    "Uncollateralised": NettingSet(trades),
    "CSA, threshold 5M": NettingSet(trades, CSA(threshold_cpty=5e6, threshold_own=5e6, mta=5e5)),
    "CSA, zero threshold": NettingSet(trades, CSA(mta=1e5)),
}


def run(n_paths: int = 10_000, seed: int = 2026):
    sim = simulate(hw, 10.0, n_paths=n_paths, fx=fx, seed=seed)
    profiles = {
        name: exposure_profile(ns.exposure_values(sim), sim) for name, ns in SCENARIOS.items()
    }
    standalone = sum(exposure_profile(t.mtm(sim), sim).ee_discounted for t in trades)
    return sim, profiles, standalone


def results_table(profiles) -> str:
    rows = [
        "| Netting set | EPE (1y) | EEPE (1y) | Peak PFE 97.5% | CVA | DVA | CS01 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, p in profiles.items():
        rows.append(
            f"| {name} | {p.epe / 1e6:,.2f}M | {p.eepe / 1e6:,.2f}M | {p.peak_pfe / 1e6:,.2f}M "
            f"| {cva(p, counterparty) / 1e3:,.0f}k | {dva(p, own) / 1e3:,.0f}k "
            f"| {cs01(p, counterparty) / 1e3:,.1f}k |"
        )
    return "\n".join(rows)


# ── Figures ────────────────────────────────────────────────────────────
SOURCE = (
    "Illustrative netting set: 10y payer IRS 100M, 5y receiver IRS 60M, 2y FX forward 20M. "
    "10,000 Hull-White/GBM paths, weekly grid. "
    "github.com/IgnacioMendozaRojas/counterparty-credit-risk"
)


def _last_nonzero(y):
    idx = np.nonzero(np.abs(y) > 1e-9)[0]
    return idx[-1] if idx.size else len(y) - 1


def figures(profiles, standalone_cva: float) -> None:
    from ccr.figstyle import charts

    unsec = profiles["Uncollateralised"]
    t = unsec.times
    end = _last_nonzero(unsec.ee) + 1
    x = t[:end]
    names = list(profiles)

    # 1. Hero: collateral against the tail of the exposure (PFE)
    peak = {n: p.peak_pfe for n, p in profiles.items()}
    cut = 1 - peak["CSA, zero threshold"] / peak["Uncollateralised"]
    for c in charts(
        f"Daily margining cuts peak exposure by {cut:.0%}",
        "Potential future exposure (97.5%) of one netting set under three collateral "
        "agreements, millions",
        SOURCE,
    ):
        for i, (n, lx, dy) in enumerate(zip(names, (2.6, 3.6, 4.4), (12, 10, 10), strict=True)):
            c.line(x, profiles[n].pfe[:end] / 1e6, i, n, label_x=lx, dy=dy)
        k = int(np.argmax(unsec.pfe[:end]))
        c.note(
            (x[k], unsec.pfe[k] / 1e6),
            f"peak {peak['Uncollateralised'] / 1e6:.1f}M uncollateralised\n"
            f"{peak['CSA, zero threshold'] / 1e6:.1f}M with daily margining",
            (40, -10),
        )
        c.ax.set_ylim(bottom=0)
        c.yaxis("millions")
        c.xlabel("years")
        c.legend("upper left")
        c.save(DOCS, "pfe-by-csa")

    # 2. Anatomy of the uncollateralised profile
    for c in charts(
        "Exposure peaks mid-life, then amortises with every coupon",
        "Uncollateralised netting set: expected exposure (EE), PFE 97.5% and expected "
        "negative exposure, millions",
        SOURCE,
    ):
        for i, (y, label, dy) in enumerate(
            [(unsec.pfe, "PFE 97.5%", 0), (unsec.ee, "EE", 4), (unsec.ene, "ENE", -4)]
        ):
            c.line(x, y[:end] / 1e6, [1, 0, 2][i], label, dy=dy)
        c.ax.axhline(0, color=c.t["muted"], linewidth=0.8)
        j = int(np.argmin(np.abs(x - 2.0)))
        c.note((x[j], unsec.ee[j] / 1e6), "FX forward settles: exposure steps down", (20, 45))
        c.yaxis("millions")
        c.xlabel("years")
        c.legend("upper right")
        c.save(DOCS, "exposure-profile")

    # 3. What each decision is worth in CVA
    bars = {
        "Trades booked standalone": standalone_cva,
        "Netted": cva(profiles["Uncollateralised"], counterparty),
        "Netted + CSA, threshold 5M": cva(profiles["CSA, threshold 5M"], counterparty),
        "Netted + CSA, zero threshold": cva(profiles["CSA, zero threshold"], counterparty),
    }
    for c in charts(
        f"Netting and margining take CVA from {standalone_cva / 1e3:,.0f}k to "
        f"{bars['Netted + CSA, zero threshold'] / 1e3:,.0f}k",
        "Credit valuation adjustment of the same three trades, thousands; "
        "counterparty CDS 150bp, recovery 40%",
        SOURCE,
        size=(9, 3.6),
    ):
        c.hbars(list(bars), [v / 1e3 for v in bars.values()], fmt="{:,.0f}k")
        c.fig.subplots_adjust(left=0.27, right=0.92, top=0.74, bottom=0.24)
        c.save(DOCS, "cva-by-decision")


def main(argv: list[str]) -> int:
    sim, profiles, standalone = run()
    table = results_table(profiles)
    netted = profiles["Uncollateralised"].ee_discounted
    netting_benefit = 1 - cva_from(netted, sim) / cva_from(standalone, sim)
    print(table)
    print(f"\nNetting benefit on CVA vs trades booked standalone: {netting_benefit:.1%}")
    if "--no-plots" not in argv:
        figures(profiles, cva_from(standalone, sim))
        (DOCS / "results.md").write_text(
            table + f"\n\nNetting benefit on CVA vs standalone trades: {netting_benefit:.1%}\n",
            encoding="utf-8",
        )
        print(f"Figures written to {DOCS}")
    return 0


def cva_from(ee_discounted: np.ndarray, sim) -> float:
    from ccr.exposure import ExposureProfile

    z = np.zeros_like(ee_discounted)
    return cva(ExposureProfile(sim.times, z, z, z, ee_discounted, z, 0.975), counterparty)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
