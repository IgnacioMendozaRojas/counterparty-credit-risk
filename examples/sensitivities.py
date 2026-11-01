"""Model choices that move the numbers: the margin period of risk and the rate-FX correlation.

    python examples/sensitivities.py        # writes docs/sensitivities.md, .json and a figure

Same illustrative netting set and market data as examples/portfolio.py, with common random
numbers, so differences come from the parameter, not from simulation noise. The headline
is chosen by a rule fixed before running: the margin period of risk, unless the rate-FX
correlation moves CVA by more than 10% across the range tested.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
from portfolio import DOCS, SOURCE, counterparty, fx, hw, trades

from ccr import CSA, NettingSet, cva, exposure_profile, simulate

MPR_DAYS = (
    5,
    10,
    20,
)  # 10 is the Basel floor for daily-margined OTC netting sets, 20 for large ones
CORRS = (-0.5, -0.2, 0.0, 0.2, 0.5)
N_PATHS = 10_000
SEED = 2026


def margin_period() -> dict:
    sim = simulate(hw, 10.0, n_paths=N_PATHS, fx=fx, seed=SEED)
    out = {}
    for days in MPR_DAYS:
        p = exposure_profile(
            NettingSet(trades, CSA(mta=1e5, mpr_days=days)).exposure_values(sim), sim
        )
        k = int(np.argmax(p.pfe))
        out[days] = {
            "peak_pfe": float(p.peak_pfe),
            "peak_time": float(p.times[k]),
            "median_pfe": float(np.median(p.pfe[p.pfe > 0])),
            "eepe": float(p.eepe),
            "cva": float(cva(p, counterparty)),
            "pfe": p.pfe.tolist(),
        }
    out["times"] = sim.times.tolist()
    return out


def correlation() -> dict:
    out = {}
    for rho in CORRS:
        sim = simulate(hw, 10.0, n_paths=N_PATHS, fx=replace(fx, corr=rho), seed=SEED)
        p = exposure_profile(NettingSet(trades).exposure_values(sim), sim)
        out[rho] = {"cva": float(cva(p, counterparty)), "peak_pfe": float(p.peak_pfe)}
    return out


def headline(mpr: dict, corr: dict) -> dict:
    lo, base, hi = (mpr[d] for d in MPR_DAYS)
    cvas = [v["cva"] for v in corr.values()]
    corr_swing = (max(cvas) - min(cvas)) / corr[0.0]["cva"]
    mpr_rise = hi["peak_pfe"] / base["peak_pfe"] - 1
    spike = base["peak_pfe"] / base["median_pfe"]
    if corr_swing > 0.10:
        text = f"Rate-FX correlation moves CVA by {corr_swing:.0%}"
        chosen = "correlation"
    else:
        cva_rise = hi["cva"] / base["cva"] - 1
        text = (
            f"Doubling the margin period adds {cva_rise:.0%} to CVA but {mpr_rise:.0%} to peak PFE"
        )
        chosen = "margin period"
    return {
        "chosen": chosen,
        "text": text,
        "mpr_rise_10_to_20": mpr_rise,
        "mpr_cva_rise_10_to_20": hi["cva"] / base["cva"] - 1,
        "peak_time_10d": base["peak_time"],
        "peak_over_median_10d": spike,
        "corr_cva_swing": corr_swing,
    }


def figure(mpr: dict, h: dict) -> None:
    from ccr.figstyle import charts

    t = np.asarray(mpr["times"])
    end = int(np.nonzero(np.asarray(mpr[10]["pfe"]) > 1e-9)[0][-1]) + 1
    for c in charts(
        h["text"]
        if h["chosen"] == "margin period"
        else "Margin period of risk and collateralised PFE",
        "PFE 97.5% of the netting set under a zero-threshold CSA, by margin period of risk "
        "(business days), millions",
        SOURCE,
    ):
        for i, days in enumerate(MPR_DAYS):
            c.line(
                t[:end],
                np.asarray(mpr[days]["pfe"][:end]) / 1e6,
                i,
                f"{days} days",
                end_label=False,
            )
        k = int(np.argmax(mpr[10]["pfe"]))
        c.note(
            (t[k], mpr[10]["peak_pfe"] / 1e6),
            f"FX forward settles: peak {mpr[10]['peak_pfe'] / 1e6:.1f}M, "
            f"{h['peak_over_median_10d']:.1f}x the usual level",
            (30, 10),
        )
        c.ax.set_ylim(bottom=0)
        c.yaxis("millions")
        c.xlabel("years")
        c.legend("upper right")
        c.save(DOCS, "margin-period")


def report(mpr: dict, corr: dict, h: dict) -> str:
    rows = [
        f"**{h['text']}.**",
        "",
        "| Margin period of risk | Peak PFE 97.5% | Peak at year | EEPE (1y) | CVA |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in MPR_DAYS:
        m = mpr[d]
        rows.append(
            f"| {d} business days | {m['peak_pfe'] / 1e6:.2f}M | {m['peak_time']:.1f} "
            f"| {m['eepe'] / 1e6:.2f}M | {m['cva'] / 1e3:,.0f}k |"
        )
    rows += [
        "",
        "| Rate-FX correlation | CVA, uncollateralised | Peak PFE 97.5% |",
        "|---:|---:|---:|",
    ]
    for rho, v in corr.items():
        rows.append(f"| {rho:+.1f} | {v['cva'] / 1e3:,.0f}k | {v['peak_pfe'] / 1e6:.2f}M |")
    rows += [
        "",
        f"From 10 to 20 days, peak PFE rises {h['mpr_rise_10_to_20']:.0%} and CVA "
        f"{h['mpr_cva_rise_10_to_20']:.0%} (a diffusion-only model would give about "
        f"{np.sqrt(2) - 1:.0%}). Across correlations of {min(CORRS)} to {max(CORRS)}, CVA moves "
        f"{h['corr_cva_swing']:.0%}.",
    ]
    return "\n".join(rows) + "\n"


def main(argv: list[str]) -> int:
    global N_PATHS
    if "--quick" in argv:  # smoke test only
        N_PATHS = 500
    mpr, corr = margin_period(), correlation()
    h = headline(mpr, corr)
    text = report(mpr, corr, h)
    print(text)
    for level, msg in [
        ("notice", h["text"]),
        ("notice", f"details: {json.dumps(h, default=float)}"),
    ]:
        print(f"::{level} title=Sensitivities::{msg}")
    if "--no-plots" not in argv:
        figure(mpr, h)
        (DOCS / "sensitivities.md").write_text(text, encoding="utf-8")
        slim = {d: {k: v for k, v in mpr[d].items() if k != "pfe"} for d in MPR_DAYS}
        (DOCS / "sensitivities.json").write_text(
            json.dumps({"headline": h, "margin_period": slim, "correlation": corr}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
