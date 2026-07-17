"""Evaluate validated MPC variants under one shared Evaluation Policy.

The Baseline is a deliberately limited-capability reference.  This script reports
each named economic quantity and physical comfort violation separately; causal
publication claims remain gated on valid, provenance-backed Run Bundles.

Usage:  python3 experiments/ablations.py [--days 14] [--start-month 1]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accounting import (
    DEFAULT_EVALUATION_POLICY,
    EvaluationReport,
    RunBundle,
    _capture_publication_context,
    evaluate_run,
    saving_percent,
)
from control.rolling_horizon import (
    ValidRun,
    _persist_outcome,
    load_data,
    run_simulation,
)
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
    baseline_executable_paths = (
        "accounting.py",
        "scenarios.py",
        "models/hub_model.py",
        "control/rolling_horizon.py",
        "experiments/ablations.py",
    )
    baseline_publication_context = _capture_publication_context(
        baseline_executable_paths,
        repository_root=ROOT,
    )
    baseline_outcome = run_simulation(
        scenario,
        mode="baseline",
        hub_config=HubConfiguration(),
        evaluation_policy=EVALUATION_POLICY,
    )
    baseline_artifact = _persist_outcome(
        baseline_outcome,
        EVALUATION_POLICY,
        results_root=ROOT / "results",
        executable_paths=baseline_executable_paths,
        publication_context=baseline_publication_context,
    )
    if not isinstance(baseline_outcome, ValidRun):
        print(
            f"INVALID baseline Run at step {baseline_outcome.failed_step}: "
            f"{baseline_outcome.failure_code}: {baseline_outcome.message}"
        )
        print(f"Diagnostics -> {baseline_artifact}")
        return
    assert isinstance(baseline_artifact, RunBundle)
    print(f"Verified Run Bundle -> {baseline_artifact.path}")
    baseline_report = evaluate_run(baseline_outcome, EVALUATION_POLICY)

    rows: list[dict[str, object]] = []
    for name, variant in VARIANTS.items():
        hub_configuration = variant["hub_configuration"]
        mpc_options = variant["mpc_options"]
        print(f"\nMPC variant: {name} {mpc_options}")
        mpc_executable_paths = (
            "accounting.py",
            "scenarios.py",
            "models/hub_model.py",
            "control/rolling_horizon.py",
            "control/mpc_controller.py",
            "experiments/ablations.py",
        )
        mpc_publication_context = _capture_publication_context(
            mpc_executable_paths,
            repository_root=ROOT,
        )
        outcome = run_simulation(
            scenario,
            mode="mpc",
            hub_config=hub_configuration,
            evaluation_policy=EVALUATION_POLICY,
            **mpc_options,
        )
        artifact = _persist_outcome(
            outcome,
            EVALUATION_POLICY,
            results_root=ROOT / "results",
            executable_paths=mpc_executable_paths,
            publication_context=mpc_publication_context,
        )
        if not isinstance(outcome, ValidRun):
            print(
                f"INVALID {name} Run at step {outcome.failed_step}: "
                f"{outcome.failure_code}: {outcome.message}"
            )
            print(f"Diagnostics -> {artifact}")
            return
        assert isinstance(artifact, RunBundle)
        print(f"Verified Run Bundle -> {artifact.path}")
        rows.append(
            _scorecard_row(
                name,
                evaluate_run(outcome, EVALUATION_POLICY),
                baseline_report,
            )
        )

    table = pd.DataFrame(rows)

    baseline = baseline_report.nominal
    print("\n" + "=" * 78)
    print(
        "  ABLATION — baseline Inventory-Adjusted Cost "
        f"EUR {baseline.inventory_adjusted_cost_eur:.2f}; "
        f"Comfort Violation {baseline.comfort_violation_c_h:.2f} C.h"
    )
    print("=" * 78)
    print(table.to_string(index=False))

    print("\nEvery row above is backed by the verified full-ID Run Bundle printed above.")


if __name__ == "__main__":
    main()
