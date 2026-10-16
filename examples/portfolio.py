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
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        text="#0b0b0b",
        muted="#52514e",
        grid="#e4e3df",
        series=["#2a78d6", "#eb6834", "#1baf7a"],
    ),
    "dark": dict(
        surface="#1a1a19",
        text="#f0efec",
        muted="#c3c2b7",
        grid="#34332f",
        series=["#3987e5", "#d95926", "#199e70"],
    ),
}


def _axes(theme, title, subtitle):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Years", color=theme["muted"], fontsize=9)
    ax.set_ylabel("Millions", color=theme["muted"], fontsize=9)
    fig.text(0.07, 0.95, title, color=theme["text"], fontsize=13, weight="bold")
    fig.text(0.07, 0.905, subtitle, color=theme["muted"], fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.12)
    return fig, ax


def _line(ax, theme, x, y, color, label, at: float, dy: int = 8):
    """Plot a series and label it directly at x = `at`, nudged dy points vertically."""
    ax.plot(x, y / 1e6, color=color, linewidth=2, label=label, solid_capstyle="round")
    k = int(np.argmin(np.abs(x - at)))
    ax.annotate(
        label,
        (x[k], y[k] / 1e6),
        xytext=(0, dy),
        textcoords="offset points",
        color=theme["text"],
        fontsize=9,
        ha="center",
        va="bottom" if dy >= 0 else "top",
    )


def _last_nonzero(y):
    idx = np.nonzero(np.abs(y) > 1e-9)[0]
    return idx[-1] if idx.size else len(y) - 1


def figures(profiles) -> None:
    import matplotlib.pyplot as plt

    DOCS.mkdir(exist_ok=True)
    unsec = profiles["Uncollateralised"]
    t = unsec.times
    for mode, theme in THEMES.items():
        # 1. Exposure profile of the uncollateralised netting set
        fig, ax = _axes(
            theme,
            "Exposure profile, uncollateralised netting set",
            "2 interest rate swaps + 1 FX forward · 10,000 Hull-White / GBM paths",
        )
        end = _last_nonzero(unsec.ee) + 1
        for y, color, label, dy in [
            (unsec.pfe, theme["series"][1], "PFE 97.5%", 8),
            (unsec.ee, theme["series"][0], "EE", 8),
            (unsec.ene, theme["series"][2], "ENE", -8),
        ]:
            _line(ax, theme, t[:end], y[:end], color, label, at=3.2, dy=dy)
        ax.axhline(0, color=theme["muted"], linewidth=0.8)
        leg = ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=theme["text"])
        leg.set_bbox_to_anchor((1.0, 1.0))
        fig.savefig(DOCS / f"exposure_profile_{mode}.png", facecolor=theme["surface"])
        plt.close(fig)

        # 2. Expected exposure under different collateral agreements
        fig, ax = _axes(
            theme,
            "Collateral cuts expected exposure",
            "EE by CSA terms · margin period of risk 10 business days",
        )
        label_at = {
            "Uncollateralised": (3.2, 10),
            "CSA, threshold 5M": (3.2, -10),
            "CSA, zero threshold": (4.2, 10),
        }
        for (name, p), color in zip(profiles.items(), theme["series"], strict=True):
            at, dy = label_at[name]
            _line(ax, theme, t[:end], p.ee[:end], color, name, at=at, dy=dy)
        ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=theme["text"])
        fig.savefig(DOCS / f"collateral_effect_{mode}.png", facecolor=theme["surface"])
        plt.close(fig)


def main(argv: list[str]) -> int:
    sim, profiles, standalone = run()
    table = results_table(profiles)
    netted = profiles["Uncollateralised"].ee_discounted
    netting_benefit = 1 - cva_from(netted, sim) / cva_from(standalone, sim)
    print(table)
    print(f"\nNetting benefit on CVA vs trades booked standalone: {netting_benefit:.1%}")
    if "--no-plots" not in argv:
        figures(profiles)
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
