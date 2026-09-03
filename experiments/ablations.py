"""Run the winter MPC ablations (full, no-H2, no-TES, one-step horizon)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from greenhouse_energy_hub.evaluation import (
    EvaluationPolicy,
    EvaluationReport,
    RunBundle,
    evaluate_run,
    saving_percent,
)
from greenhouse_energy_hub.simulation import InvalidRun, ValidRun

if __package__:
    from .run_scenario import (
        CANDIDATE_INDEX,
        RESULTS_ROOT,
        ROOT,
        _amsterdam_timestamp,
        execute_experiment,
    )
else:
    from run_scenario import (
        CANDIDATE_INDEX,
        RESULTS_ROOT,
        ROOT,
        _amsterdam_timestamp,
        execute_experiment,
    )


VARIANTS = {
    "full": {"horizon_steps": 24},
    "no-h2": {"horizon_steps": 24, "hydrogen": False},
    "no-tes": {"horizon_steps": 24, "thermal_store": False},
    "one-step": {"horizon_steps": 1},
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved greenhouse energy hub MPC ablations."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--start", required=True, type=_amsterdam_timestamp)
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--end", type=_amsterdam_timestamp)
    window.add_argument("--days", type=int)
    parser.add_argument("--scenario-max-horizon", type=int, default=24)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--price-path", type=Path)
    parser.add_argument("--pv-path", type=Path)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--candidate-index", type=Path, default=CANDIDATE_INDEX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    policy = EvaluationPolicy()
    common = {
        "name": args.name,
        "operating_start": args.start,
        "operating_end": args.end,
        "calendar_days": args.days,
        "scenario_max_horizon_steps": args.scenario_max_horizon,
        "terminal_weight": args.terminal_weight,
        "price_path": args.price_path,
        "pv_path": args.pv_path,
        "repository_root": args.repository_root,
        "results_root": args.results_root,
        "extra_executable_paths": ("experiments/ablations.py",),
        "evaluation_policy": policy,
    }

    baseline_result = execute_experiment(
        controller_name="baseline",
        **common,
    )
    if isinstance(baseline_result.outcome, InvalidRun):
        print(
            f"INVALID baseline Run at step {baseline_result.outcome.failed_step}: "
            f"{baseline_result.outcome.failure_code}: "
            f"{baseline_result.outcome.message}"
        )
        print(f"Diagnostics -> {baseline_result.artifact}")
        return 2
    assert isinstance(baseline_result.outcome, ValidRun)
    assert isinstance(baseline_result.artifact, RunBundle)
    print(
        "Verified Baseline Run Bundle "
        f"{baseline_result.artifact.identifier} -> {baseline_result.artifact.path}"
    )
    baseline_report = evaluate_run(baseline_result.outcome, policy)

    rows: list[dict[str, object]] = []
    for variant_name, variant in VARIANTS.items():
        variant_result = execute_experiment(
            controller_name="mpc",
            horizon_steps=variant["horizon_steps"],
            hydrogen=variant.get("hydrogen", True),
            thermal_store=variant.get("thermal_store", True),
            candidate_key=f"ablation-{variant_name}",
            candidate_index=args.candidate_index,
            **common,
        )
        if isinstance(variant_result.outcome, InvalidRun):
            print(
                f"INVALID {variant_name} Run at step "
                f"{variant_result.outcome.failed_step}: "
                f"{variant_result.outcome.failure_code}: "
                f"{variant_result.outcome.message}"
            )
            print(f"Diagnostics -> {variant_result.artifact}")
            return 2
        assert isinstance(variant_result.outcome, ValidRun)
        assert isinstance(variant_result.artifact, RunBundle)
        print(
            f"Verified {variant_name} Run Bundle "
            f"{variant_result.artifact.identifier} -> {variant_result.artifact.path}"
        )
        rows.append(
            _scorecard_row(
                variant_name,
                evaluate_run(variant_result.outcome, policy),
                baseline_report,
            )
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print("Every row is backed by the verified full-ID Run Bundle printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
