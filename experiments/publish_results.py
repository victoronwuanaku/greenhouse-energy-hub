"""Regenerate the publication manifest, figures and README results block.

Every number and figure comes from Run Bundles pinned in the publication
manifest and verified on load. The generated README block replaces only the
text between its two markers; the rest of the README is left byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from greenhouse_energy_hub.evaluation import (
    PublicationEvidence,
    RunBundle,
    build_publication_manifest,
    load_publication_trajectory,
    load_verified_publication_evidence,
    publication_ablation_rows,
    publication_bundle_ids,
    publication_comparison_rows,
)


ROOT = Path(__file__).resolve().parent.parent

README_BEGIN = b"<!-- BEGIN GENERATED RESULTS: DO NOT EDIT -->"
README_END = b"<!-- END GENERATED RESULTS -->"

REPRODUCIBILITY_PROSE = (
    "**Reproducibility.** The tables and figures use [verified Run Bundles]"
    "(results/runs/) committed with the repository. The "
    "[publication manifest](results/publication_manifest.json) maps the published "
    "results and figures to their source runs, and the "
    "[publication tests](tests/test_published_artifacts.py) recompute the reported "
    "values."
)


def render_readme_block(summaries: Mapping[str, Mapping[str, object]]) -> str:
    """Return the Markdown body placed between the README result markers."""
    rows = publication_comparison_rows(summaries)
    ablations = publication_ablation_rows(summaries)
    lines = [
        "| Window | Baseline Inventory-Adjusted Cost | MPC Inventory-Adjusted Cost | Saving (€) | Saving (%) | Comfort Violation (baseline / MPC) |",
        "|--------|----------------------------------:|-----------------------------:|-----------:|:----------:|:-----------------------------------:|",
        *[
            f"| **{row['window']}** | €{row['baseline_inventory_adjusted_cost_eur']:,.0f} | "
            f"€{row['mpc_inventory_adjusted_cost_eur']:,.0f} | "
            f"€{row['saving_eur']:,.0f} | {row['saving_percent']:+.1f} % | "
            f"{row['baseline_comfort_violation_c_h']:.1f} / "
            f"{row['mpc_comfort_violation_c_h']:.1f} °C·h |"
            for row in rows
        ],
        "",
        "| Winter ablation | Inventory-Adjusted Cost | Comfort Violation | Cost difference vs full |",
        "|-----------------|--------------------------:|------------------:|:-----------------------:|",
        *[
            f"| {row['variant']} | €{row['inventory_adjusted_cost_eur']:,.0f} | "
            f"{row['comfort_violation_c_h']:,.1f} °C·h | "
            f"{row['cost_difference_vs_full_percent']:+.1f} % |"
            for row in ablations
        ],
        "",
        REPRODUCIBILITY_PROSE,
    ]
    return "\n" + "\n".join(lines) + "\n"


def rewrite_readme_block(
    readme_path: str | Path,
    summaries: Mapping[str, Mapping[str, object]],
) -> None:
    """Replace the generated block, preserving every byte outside the markers."""
    path = Path(readme_path)
    original = path.read_bytes()
    if original.count(README_BEGIN) != 1 or original.count(README_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    begin = original.index(README_BEGIN) + len(README_BEGIN)
    end = original.index(README_END)
    if end < begin:
        raise ValueError("README generated-results markers are out of order")
    replacement = render_readme_block(summaries).encode("utf-8")
    path.write_bytes(original[:begin] + replacement + original[end:])


def render_figures(evidence: PublicationEvidence, figures_root: str | Path) -> None:
    """Render the six publication figures from verified bundles."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    figures = Path(figures_root)
    figures.mkdir(parents=True, exist_ok=True)
    bundles: Mapping[str, RunBundle] = evidence.bundles
    summaries = evidence.summaries
    trajectories = {key: load_publication_trajectory(bundle) for key, bundle in bundles.items()}
    baseline = trajectories["winter-baseline"]
    mpc = trajectories["winter-mpc"]
    day_format = mdates.DateFormatter("%b %d")
    plt.rcParams.update(
        {"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10}
    )

    # 1. Cumulative grid cost
    baseline_costs = pd.DataFrame(summaries["winter-baseline"]["step_line_items"])
    mpc_costs = pd.DataFrame(summaries["winter-mpc"]["step_line_items"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(baseline.index, baseline_costs["grid_cost_eur"].cumsum(), label="Baseline", lw=2, color="#b2182b")
    ax.plot(mpc.index, mpc_costs["grid_cost_eur"].cumsum(), label="MPC", lw=2, color="#2166ac")
    ax.set(ylabel="Cumulative grid cost [EUR]", title="Winter cumulative grid cost")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "fig1_cumulative_cost.png")
    plt.close(fig)

    # 2. Grid exchange against day-ahead price
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mpc.index, mpc["grid_kw"], color="#2166ac", lw=1.2, label="MPC grid power [kW]")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Grid power [kW]")
    price_axis = ax.twinx()
    price_axis.plot(mpc.index, mpc["price_eur_per_kwh"] * 100, color="#fdae61", alpha=0.75, label="Day-ahead price [ct/kWh]")
    price_axis.set_ylabel("Price [ct/kWh]")
    ax.set_title("MPC grid exchange and day-ahead price")
    ax.xaxis.set_major_formatter(day_format)
    fig.tight_layout()
    fig.savefig(figures / "fig2_grid_vs_price.png")
    plt.close(fig)

    # 3. Storage trajectories
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    storage_columns = (
        ("reached_soc_battery_kwh", "Battery SOC [kWh]"),
        ("reached_soc_hydrogen_kg", "Hydrogen inventory [kg]"),
        ("reached_soc_thermal_kwh", "Thermal-store SOC [kWh]"),
    )
    for axis, (column, label) in zip(axes, storage_columns, strict=True):
        axis.plot(baseline.index, baseline[column], color="#b2182b", alpha=0.7, label="Baseline")
        axis.plot(mpc.index, mpc[column], color="#2166ac", label="MPC")
        axis.set_ylabel(label)
        axis.legend(loc="best")
    axes[-1].xaxis.set_major_formatter(day_format)
    fig.suptitle("Winter storage trajectories")
    fig.tight_layout()
    fig.savefig(figures / "fig3_soc_trajectories.png")
    plt.close(fig)

    # 4. Indoor temperature against the comfort band
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(16, 24, color="#1a9850", alpha=0.08, label="Comfort band")
    ax.plot(mpc.index, mpc["reached_indoor_temperature_c"], color="#2166ac", lw=1.3, label="MPC indoor")
    ax.plot(baseline.index, baseline["reached_indoor_temperature_c"], color="#b2182b", lw=1.0, alpha=0.7, label="Baseline indoor")
    ax.plot(mpc.index, mpc["outdoor_temperature_c"], color="grey", lw=0.9, alpha=0.7, label="Outdoor")
    ax.set(ylabel="Temperature [°C]", title="Winter greenhouse temperature")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(day_format)
    fig.tight_layout()
    fig.savefig(figures / "fig4_temperature.png")
    plt.close(fig)

    # 5. Power-to-heat operation
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mpc.index, mpc["electric_boiler_kw"], color="#d73027", lw=1.2, label="Electric boiler [kW]")
    ax.plot(mpc.index, mpc["thermal_charge_kw"], color="#1a9850", lw=1.1, label="Thermal-store charge [kWth]")
    ax.set(ylabel="Power [kW]", title="MPC power-to-heat operation")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(day_format)
    fig.tight_layout()
    fig.savefig(figures / "fig5_heat_shifting.png")
    plt.close(fig)

    # 6. Ablation cost differences
    ablations = publication_ablation_rows(summaries)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(
        [str(row["variant"]).replace(" horizon", "") for row in ablations],
        [float(row["cost_difference_vs_full_percent"]) for row in ablations],
        color=["#2166ac", "#7fb3d5", "#7fb3d5", "#7fb3d5"],
    )
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Inventory-adjusted cost difference vs full [%]")
    ax.set_title("Winter ablation comparison")
    for bar, row in zip(bars, ablations, strict=True):
        value = float(row["cost_difference_vs_full_percent"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.35 if value >= 0 else -1.0),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(figures / "fig6_ablation.png")
    plt.close(fig)


def regenerate_publication_artifacts(
    *,
    candidate_index: str | Path,
    manifest_path: str | Path,
    repository_root: str | Path,
    runs_root: str | Path,
    figures_root: str | Path,
    readme_path: str | Path,
) -> dict[str, object]:
    """Write the verified manifest, the figures, and the README results block."""
    manifest = build_publication_manifest(
        candidate_index,
        manifest_path=manifest_path,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    evidence = load_verified_publication_evidence(
        manifest, runs_root=runs_root, repository_root=repository_root
    )
    render_figures(evidence, figures_root)
    rewrite_readme_block(readme_path, evidence.summaries)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate publication artifacts from verified pinned Run Bundles."
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "results" / "diagnostics" / "publication-candidates.json",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    manifest = regenerate_publication_artifacts(
        candidate_index=args.candidates,
        manifest_path=args.manifest,
        repository_root=ROOT,
        runs_root=ROOT / "results" / "runs",
        figures_root=ROOT / "results" / "figures",
        readme_path=ROOT / "README.md",
    )
    print(
        "Regenerated publication artifacts from "
        f"{len(publication_bundle_ids(manifest))} verified pinned Run Bundles."
    )


if __name__ == "__main__":
    main()
