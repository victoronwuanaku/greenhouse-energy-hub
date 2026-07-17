"""
Ablation study: isolate where the MPC's economic value comes from.

Runs the winter headline window for the rule-based baseline and four MPC variants:
  * MPC (full)   — all assets, 24 h foresight
  * no H2        — electrolyser + fuel cell locked off
  * no TES       — thermal store locked off
  * myopic (1 h) — 1-step horizon (no multi-step price look-ahead)

Each variant's saving vs the baseline shows the marginal contribution of that
capability. Writes results/scenarios/ablations.csv and results/figures/fig6_ablation.png.

Usage:  python3 experiments/ablations.py [--days 14] [--start-month 1]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.rolling_horizon import ValidRun, load_data, run_simulation
from accounting import saving_pct

VARIANTS = {
    "MPC (full)":   {},
    "no H2":        {"disable_h2": True},
    "no TES":       {"disable_tes": True},
    "myopic (1h)":  {"n_horizon": 1},
}

# A grower cares about total operating cost INCLUDING crop stress. We value comfort-band
# violations at the MPC's internal soft-constraint rate so a controller cannot look cheap
# simply by letting the greenhouse drift out of the band (as the myopic variant does).
COMFORT_PENALTY_EUR_PER_CH = 10.0


def effective_cost(grid_eur: float, viol_ch: float) -> float:
    return grid_eur + COMFORT_PENALTY_EUR_PER_CH * viol_ch


def main():
    ap = argparse.ArgumentParser(description="MPC ablation study.")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start-month", type=int, default=1)
    args = ap.parse_args()

    df = load_data(start_month=args.start_month, n_days=args.days)

    print("\nBaseline...")
    base_outcome = run_simulation(df, mode="baseline")
    if not isinstance(base_outcome, ValidRun):
        print(
            f"INVALID baseline Run at step {base_outcome.failed_step}: "
            f"{base_outcome.failure_code}: {base_outcome.message}"
        )
        return
    base = base_outcome.to_frame()
    base_cost = base["grid_cost_EUR"].sum()
    base_viol = base["T_violation_C"].sum()
    base_eff = effective_cost(base_cost, base_viol)

    rows = []
    for name, kw in VARIANTS.items():
        print(f"\nMPC variant: {name} {kw}")
        outcome = run_simulation(df, mode="mpc", **kw)
        if not isinstance(outcome, ValidRun):
            print(
                f"INVALID {name} Run at step {outcome.failed_step}: "
                f"{outcome.failure_code}: {outcome.message}"
            )
            return
        m = outcome.to_frame()
        grid = m["grid_cost_EUR"].sum()
        viol = m["T_violation_C"].sum()
        rows.append({
            "variant": name,
            "grid_eur": round(grid, 1),
            "Tband_viol_Ch": round(viol, 1),
            "effective_eur": round(effective_cost(grid, viol), 1),
            "grid_saving_pct": round(saving_pct(base_cost, grid), 2),
            "effective_saving_pct": round(saving_pct(base_eff, effective_cost(grid, viol)), 2),
        })

    tab = pd.DataFrame(rows)
    scen = ROOT / "results" / "scenarios"
    scen.mkdir(parents=True, exist_ok=True)
    tab.to_csv(scen / "ablations.csv", index=False)
    print("\n" + "=" * 70)
    print(f"  ABLATION (baseline grid EUR {base_cost:.0f}, comfort viol {base_viol:.0f} degC.h)")
    print("=" * 70)
    print(tab.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#2166ac"] + ["#7fb3d5"] * (len(tab) - 1)
    bars = ax.bar(tab["variant"], tab["effective_saving_pct"], color=colors)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Effective saving vs baseline [%]\n(grid cost + crop-comfort penalty)")
    ax.set_title("Winter ablation — where the MPC's value comes from")
    for b, v in zip(bars, tab["effective_saving_pct"]):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.4 if v >= 0 else -1.2), f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig6_ablation.png", bbox_inches="tight")
    print(f"\nSaved {scen/'ablations.csv'} and results/figures/fig6_ablation.png")


if __name__ == "__main__":
    main()
