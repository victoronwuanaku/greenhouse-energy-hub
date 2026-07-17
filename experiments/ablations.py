"""Evaluate validated MPC variants under one shared Evaluation Policy.

The Baseline is a deliberately limited-capability reference.  This script reports
each named economic quantity and physical comfort violation separately; causal
publication claims remain gated on valid, provenance-backed Run Bundles.

Usage:  python3 experiments/ablations.py [--days 14] [--start-month 1]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accounting import (
    DEFAULT_EVALUATION_POLICY,
    EvaluationReport,
    evaluate_run,
    saving_percent,
)
from control.rolling_horizon import ValidRun, load_data, run_simulation
from models.hub_model import AssetCapabilities, HubConfiguration


EVALUATION_POLICY = DEFAULT_EVALUATION_POLICY

VARIANTS = {
    "MPC (full)": {
        "hub_configuration": HubConfiguration(),
        "mpc_options": {},
    },
    "no H2": {
        "hub_configuration": HubConfiguration(
            capabilities=AssetCapabilities(hydrogen=False)
        ),
        "mpc_options": {},
    },
    "no TES": {
        "hub_configuration": HubConfiguration(
            capabilities=AssetCapabilities(thermal_store=False)
        ),
        "mpc_options": {},
    },
    "myopic (1h)": {
        "hub_configuration": HubConfiguration(),
        "mpc_options": {"n_horizon": 1},
    },
}


def _scorecard_row(
    variant_name: str,
    report: EvaluationReport,
    baseline_report: EvaluationReport,
) -> dict[str, object]:
    nominal = report.nominal
    sensitivities = report.wear_sensitivities
    return {
        "Variant": variant_name,
        "Grid Cost [EUR]": nominal.grid_cost_eur,
        "Operating Cost [EUR]": nominal.operating_cost_eur,
        "Inventory-Adjusted Cost [EUR]": nominal.inventory_adjusted_cost_eur,
        "Comfort Violation [C.h]": nominal.comfort_violation_c_h,
        "Wear 0x Inventory-Adjusted Cost [EUR]": (
            sensitivities["0x"].inventory_adjusted_cost_eur
        ),
        "Wear 1x Inventory-Adjusted Cost [EUR]": (
            sensitivities["1x"].inventory_adjusted_cost_eur
        ),
        "Wear 2x Inventory-Adjusted Cost [EUR]": (
            sensitivities["2x"].inventory_adjusted_cost_eur
        ),
        "Inventory-Adjusted Saving vs Baseline [%]": saving_percent(
            baseline_report.nominal.inventory_adjusted_cost_eur,
            nominal.inventory_adjusted_cost_eur,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MPC ablation study.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-month", type=int, default=1)
    args = parser.parse_args()

    scenario = load_data(start_month=args.start_month, n_days=args.days)

    print("\nBaseline (limited-capability reference)...")
    baseline_outcome = run_simulation(
        scenario,
        mode="baseline",
        hub_config=HubConfiguration(),
        evaluation_policy=EVALUATION_POLICY,
    )
    if not isinstance(baseline_outcome, ValidRun):
        print(
            f"INVALID baseline Run at step {baseline_outcome.failed_step}: "
            f"{baseline_outcome.failure_code}: {baseline_outcome.message}"
        )
        return
    baseline_report = evaluate_run(baseline_outcome, EVALUATION_POLICY)

    rows: list[dict[str, object]] = []
    for name, variant in VARIANTS.items():
        hub_configuration = variant["hub_configuration"]
        mpc_options = variant["mpc_options"]
        print(f"\nMPC variant: {name} {mpc_options}")
        outcome = run_simulation(
            scenario,
            mode="mpc",
            hub_config=hub_configuration,
            evaluation_policy=EVALUATION_POLICY,
            **mpc_options,
        )
        if not isinstance(outcome, ValidRun):
            print(
                f"INVALID {name} Run at step {outcome.failed_step}: "
                f"{outcome.failure_code}: {outcome.message}"
            )
            return
        rows.append(
            _scorecard_row(
                name,
                evaluate_run(outcome, EVALUATION_POLICY),
                baseline_report,
            )
        )

    table = pd.DataFrame(rows)
    scenario_directory = ROOT / "results" / "scenarios"
    figure_directory = ROOT / "results" / "figures"
    scenario_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(scenario_directory / "ablations.csv", index=False)

    baseline = baseline_report.nominal
    print("\n" + "=" * 78)
    print(
        "  ABLATION — baseline Inventory-Adjusted Cost "
        f"EUR {baseline.inventory_adjusted_cost_eur:.2f}; "
        f"Comfort Violation {baseline.comfort_violation_c_h:.2f} C.h"
    )
    print("=" * 78)
    print(table.to_string(index=False))

    saving_column = "Inventory-Adjusted Saving vs Baseline [%]"
    fig, axis = plt.subplots(figsize=(7.5, 4))
    colors = ["#2166ac"] + ["#7fb3d5"] * (len(table) - 1)
    bars = axis.bar(table["Variant"], table[saving_column], color=colors)
    axis.axhline(0, color="k", lw=0.6)
    axis.set_ylabel("Inventory-Adjusted saving vs Baseline [%]")
    axis.set_title("Winter ablation under one Evaluation Policy")
    for bar, value in zip(bars, table[saving_column], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.4 if value >= 0 else -1.2),
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    figure_path = figure_directory / "fig6_ablation.png"
    fig.savefig(figure_path, bbox_inches="tight")
    print(
        f"\nSaved {scenario_directory / 'ablations.csv'} and {figure_path}"
    )


if __name__ == "__main__":
    main()
