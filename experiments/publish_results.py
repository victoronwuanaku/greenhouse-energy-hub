"""Regenerate publication artifacts from verified, full-ID Run Bundles only."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from greenhouse_energy_hub.evaluation import (
    RunBundle,
    load_run_bundle,
    saving_percent,
    verify_run_bundle,
)


ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "results" / "runs"
FIGURES_ROOT = ROOT / "results" / "figures"
README_PATH = ROOT / "README.md"

FULL_IDENTIFIER = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_CANDIDATE_KEYS = frozenset(
    {
        "ablation-full",
        "ablation-no-h2",
        "ablation-no-tes",
        "ablation-one-step",
        "summer-baseline",
        "summer-mpc",
        "winter-baseline",
        "winter-mpc",
    }
)
REQUIRED_SENSITIVITIES = frozenset({"0x", "1x", "2x"})
README_GENERATED_BEGIN = "<!-- BEGIN GENERATED RESULTS: DO NOT EDIT -->"
README_GENERATED_END = "<!-- END GENERATED RESULTS -->"


def _require_full_identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or FULL_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{description} must be a full lowercase SHA-256 identifier")
    return value


def _read_candidate_mapping(candidate_index: str | Path) -> dict[str, str]:
    path = Path(candidate_index)
    if path.is_symlink() or not path.is_file():
        raise ValueError("publication candidates must be a regular JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication candidates must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != EXPECTED_CANDIDATE_KEYS:
        raise ValueError("publication candidates have missing or extra stable keys")

    candidates = {
        key: _require_full_identifier(value, f"publication candidate {key!r}")
        for key, value in payload.items()
    }
    if len(set(candidates.values())) != len(candidates):
        raise ValueError("publication candidates must pin distinct Run Bundle identifiers")
    return candidates


def _summary_for(bundle: RunBundle) -> dict[str, object]:
    try:
        summary = json.loads((bundle.path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Run Bundle {bundle.identifier} has no readable summary") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Run Bundle {bundle.identifier} summary must be an object")
    if not isinstance(summary.get("nominal"), dict):
        raise ValueError(f"Run Bundle {bundle.identifier} summary is missing nominal values")
    sensitivities = summary.get("wear_sensitivities")
    if not isinstance(sensitivities, dict) or set(sensitivities) != REQUIRED_SENSITIVITIES:
        raise ValueError(
            f"Run Bundle {bundle.identifier} is missing required 0x, 1x, and 2x sensitivities"
        )
    return summary


def _policy_for(bundle: RunBundle) -> Mapping[str, object]:
    policy = bundle.manifest.get("evaluation_policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"Run Bundle {bundle.identifier} has no Evaluation Policy")
    return policy


def validate_publication_bundles(bundles: Mapping[str, RunBundle]) -> None:
    """Check publication-specific provenance, sensitivity, and policy gates."""
    if not bundles:
        raise ValueError("publication requires at least one verified Run Bundle")

    policies: list[Mapping[str, object]] = []
    for key, bundle in bundles.items():
        _require_full_identifier(bundle.identifier, f"Run Bundle for {key!r}")
        provenance = bundle.manifest.get("code_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Run Bundle {bundle.identifier} has no provenance")
        if provenance.get("publication_eligible") is not True:
            raise ValueError(f"Run Bundle {bundle.identifier} is not publication-eligible")
        if provenance.get("dirty_executable_paths") or provenance.get(
            "untracked_executable_paths"
        ):
            raise ValueError(f"Run Bundle {bundle.identifier} has dirty provenance")
        if bundle.manifest.get("valid") is not True:
            raise ValueError(f"Run Bundle {bundle.identifier} did not pass validation")
        _summary_for(bundle)
        policies.append(_policy_for(bundle))

    reference_policy = policies[0]
    if any(policy != reference_policy for policy in policies[1:]):
        raise ValueError("publication comparisons use different Evaluation Policies")


def _load_verified_candidates(
    candidates: Mapping[str, str],
    *,
    runs_root: str | Path,
    repository_root: str | Path,
) -> dict[str, RunBundle]:
    bundles: dict[str, RunBundle] = {}
    for key, identifier in candidates.items():
        requested = _require_full_identifier(identifier, f"publication candidate {key!r}")
        loaded = load_run_bundle(
            runs_root,
            requested,
            repository_root=repository_root,
        )
        verified = verify_run_bundle(
            loaded.path,
            expected_identifier=requested,
            repository_root=repository_root,
        )
        if verified.identifier != requested:
            raise ValueError(f"Run Bundle for {key!r} did not retain its requested ID")
        bundles[key] = verified
    validate_publication_bundles(bundles)
    return bundles


def _manifest_from_candidates(candidates: Mapping[str, str]) -> dict[str, object]:
    return {
        "schema_version": "publication-manifest-v1",
        "comparisons": {
            "winter": {
                "baseline_bundle_id": candidates["winter-baseline"],
                "mpc_bundle_id": candidates["winter-mpc"],
            },
            "summer": {
                "baseline_bundle_id": candidates["summer-baseline"],
                "mpc_bundle_id": candidates["summer-mpc"],
            },
        },
        "ablations": {
            "full": candidates["ablation-full"],
            "no-h2": candidates["ablation-no-h2"],
            "no-tes": candidates["ablation-no-tes"],
            "one-step": candidates["ablation-one-step"],
        },
        "figures": {
            "fig1_cumulative_cost.png": [
                candidates["winter-baseline"],
                candidates["winter-mpc"],
            ],
            "fig2_grid_vs_price.png": [candidates["winter-mpc"]],
            "fig3_soc_trajectories.png": [
                candidates["winter-baseline"],
                candidates["winter-mpc"],
            ],
            "fig4_temperature.png": [
                candidates["winter-baseline"],
                candidates["winter-mpc"],
            ],
            "fig5_heat_shifting.png": [candidates["winter-mpc"]],
            "fig6_ablation.png": [
                candidates["ablation-full"],
                candidates["ablation-no-h2"],
                candidates["ablation-no-tes"],
                candidates["ablation-one-step"],
            ],
        },
    }


def build_publication_manifest(
    candidate_index: str | Path,
    *,
    runs_root: str | Path = RUNS_ROOT,
    repository_root: str | Path = ROOT,
) -> dict[str, object]:
    """Load full candidates and reject any bundle unsuitable for publication."""
    candidates = _read_candidate_mapping(candidate_index)
    _load_verified_candidates(
        candidates,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    return _manifest_from_candidates(candidates)


def _manifest_candidates(manifest: Mapping[str, object]) -> dict[str, str]:
    if set(manifest) != {"schema_version", "comparisons", "ablations", "figures"}:
        raise ValueError("publication manifest has missing or extra top-level fields")
    if manifest["schema_version"] != "publication-manifest-v1":
        raise ValueError("publication manifest schema version is unsupported")
    comparisons = manifest["comparisons"]
    ablations = manifest["ablations"]
    figures = manifest["figures"]
    if not isinstance(comparisons, Mapping) or set(comparisons) != {"winter", "summer"}:
        raise ValueError("publication manifest comparisons are incomplete")
    if not isinstance(ablations, Mapping) or set(ablations) != {
        "full",
        "no-h2",
        "no-tes",
        "one-step",
    }:
        raise ValueError("publication manifest ablations are incomplete")
    if not isinstance(figures, Mapping) or set(figures) != {
        "fig1_cumulative_cost.png",
        "fig2_grid_vs_price.png",
        "fig3_soc_trajectories.png",
        "fig4_temperature.png",
        "fig5_heat_shifting.png",
        "fig6_ablation.png",
    }:
        raise ValueError("publication manifest figures are incomplete")

    candidates: dict[str, str] = {}
    for season in ("winter", "summer"):
        comparison = comparisons[season]
        if not isinstance(comparison, Mapping) or set(comparison) != {
            "baseline_bundle_id",
            "mpc_bundle_id",
        }:
            raise ValueError(f"publication manifest {season} comparison is incomplete")
        candidates[f"{season}-baseline"] = _require_full_identifier(
            comparison["baseline_bundle_id"],
            f"publication manifest {season} baseline",
        )
        candidates[f"{season}-mpc"] = _require_full_identifier(
            comparison["mpc_bundle_id"],
            f"publication manifest {season} MPC",
        )
    for key in ("full", "no-h2", "no-tes", "one-step"):
        candidates[f"ablation-{key}"] = _require_full_identifier(
            ablations[key],
            f"publication manifest ablation {key}",
        )

    expected = _manifest_from_candidates(candidates)
    if manifest != expected:
        raise ValueError("publication manifest does not match its pinned recipe")
    return candidates


def publication_bundle_ids(manifest: Mapping[str, object]) -> tuple[str, ...]:
    """Return each full bundle ID once, in recipe order, after schema validation."""
    candidates = _manifest_candidates(manifest)
    return tuple(dict.fromkeys(candidates.values()))


def _read_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("publication manifest must be a regular JSON file")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication manifest must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("publication manifest must be a JSON object")
    return value


def _load_manifest_bundles(
    manifest: Mapping[str, object],
    *,
    runs_root: str | Path,
    repository_root: str | Path,
) -> tuple[dict[str, str], dict[str, RunBundle]]:
    candidates = _manifest_candidates(manifest)
    bundles = _load_verified_candidates(
        candidates,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    return candidates, bundles


def validate_committed_publication_bundles(
    manifest: Mapping[str, object],
    *,
    runs_root: str | Path = RUNS_ROOT,
    repository_root: str | Path = ROOT,
) -> None:
    """Require every pinned bundle member to be retained in the proposed commit."""
    _candidates, bundles = _load_manifest_bundles(
        manifest,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    repository = Path(repository_root)
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    )
    required = {
        member.relative_to(repository).as_posix()
        for bundle in bundles.values()
        for member in bundle.path.iterdir()
    }
    missing = sorted(required - tracked)
    if missing:
        raise ValueError(
            "publication manifest references Run Bundles absent from the proposed commit: "
            + ", ".join(missing)
        )


def _trajectory_for(bundle: RunBundle) -> pd.DataFrame:
    frame = pd.read_csv(bundle.path / "trajectory.csv", parse_dates=["timestamp_utc"])
    return frame.set_index("timestamp_utc")


def _metric(summary: Mapping[str, object], key: str) -> float:
    nominal = summary["nominal"]
    assert isinstance(nominal, Mapping)
    value = nominal[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary metric {key!r} must be numeric")
    return float(value)


def _readme_generated_block(
    candidates: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, object]],
) -> str:
    def comparison_row(label: str, baseline_key: str, mpc_key: str) -> str:
        baseline = summaries[baseline_key]
        mpc = summaries[mpc_key]
        baseline_cost = _metric(baseline, "inventory_adjusted_cost_eur")
        mpc_cost = _metric(mpc, "inventory_adjusted_cost_eur")
        return (
            f"| **{label}** | €{baseline_cost:,.0f} | €{mpc_cost:,.0f} | "
            f"{saving_percent(baseline_cost, mpc_cost):+.1f} % | "
            f"{_metric(baseline, 'comfort_violation_c_h'):.1f} / "
            f"{_metric(mpc, 'comfort_violation_c_h'):.1f} °C·h |"
        )

    full_cost = _metric(summaries["ablation-full"], "inventory_adjusted_cost_eur")
    ablation_rows = []
    for label, key in (
        ("MPC (full)", "ablation-full"),
        ("no H₂", "ablation-no-h2"),
        ("no thermal store", "ablation-no-tes"),
        ("one-step horizon", "ablation-one-step"),
    ):
        summary = summaries[key]
        cost = _metric(summary, "inventory_adjusted_cost_eur")
        ablation_rows.append(
            f"| {label} | €{cost:,.0f} | "
            f"{_metric(summary, 'comfort_violation_c_h'):,.1f} °C·h | "
            f"{saving_percent(full_cost, cost):+.1f} % |"
        )

    full = candidates["ablation-full"]
    no_h2 = candidates["ablation-no-h2"]
    no_tes = candidates["ablation-no-tes"]
    one_step = candidates["ablation-one-step"]
    return "\n".join(
        [
            "| Window | Baseline Inventory-Adjusted Cost | MPC Inventory-Adjusted Cost | Saving | Comfort Violation (baseline / MPC) |",
            "|--------|----------------------------------:|-----------------------------:|:------:|:-----------------------------------:|",
            comparison_row("Winter", "winter-baseline", "winter-mpc"),
            comparison_row("Summer", "summer-baseline", "summer-mpc"),
            "",
            "Pinned Run Bundle IDs:",
            f"- Winter baseline: `{candidates['winter-baseline']}`",
            f"- Winter MPC: `{candidates['winter-mpc']}`",
            f"- Summer baseline: `{candidates['summer-baseline']}`",
            f"- Summer MPC: `{candidates['summer-mpc']}`",
            "",
            "| Winter ablation | Inventory-Adjusted Cost | Comfort Violation | Cost change vs full |",
            "|-----------------|--------------------------:|------------------:|:-------------------:|",
            *ablation_rows,
            "",
            f"Removing hydrogen raises the winter inventory-adjusted cost relative to the full controller (`{no_h2}` versus `{full}`).",
            f"Removing the thermal store raises the winter inventory-adjusted cost relative to the full controller (`{no_tes}` versus `{full}`).",
            f"A one-step horizon has substantial comfort violation in this winter run (`{one_step}` versus `{full}`).",
        ]
    )


def rewrite_readme_generated_block(
    readme_path: str | Path,
    *,
    candidates: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, object]],
) -> None:
    """Replace exactly the delimited generated-results body, leaving all else intact."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if text.count(README_GENERATED_BEGIN) != 1 or text.count(README_GENERATED_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    begin = text.index(README_GENERATED_BEGIN) + len(README_GENERATED_BEGIN)
    end = text.index(README_GENERATED_END)
    if end < begin:
        raise ValueError("README generated-results markers are out of order")
    replacement = "\n" + _readme_generated_block(candidates, summaries) + "\n"
    path.write_text(text[:begin] + replacement + text[end:], encoding="utf-8")


def render_figures(
    bundles: Mapping[str, RunBundle], *, figures_root: str | Path = FIGURES_ROOT
) -> None:
    """Render the six published figures from verified trajectory and evaluation members."""
    figures = Path(figures_root)
    figures.mkdir(parents=True, exist_ok=True)
    trajectories = {key: _trajectory_for(bundle) for key, bundle in bundles.items()}
    summaries = {key: _summary_for(bundle) for key, bundle in bundles.items()}
    baseline = trajectories["winter-baseline"]
    mpc = trajectories["winter-mpc"]

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )

    baseline_costs = pd.DataFrame(summaries["winter-baseline"]["step_line_items"])
    mpc_costs = pd.DataFrame(summaries["winter-mpc"]["step_line_items"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(baseline.index, baseline_costs["grid_cost_eur"].cumsum(), label="Baseline", lw=2, color="#b2182b")
    ax.plot(mpc.index, mpc_costs["grid_cost_eur"].cumsum(), label="MPC", lw=2, color="#2166ac")
    ax.set_ylabel("Cumulative grid cost [EUR]")
    ax.set_title("Winter cumulative grid cost")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "fig1_cumulative_cost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mpc.index, mpc["grid_kw"], color="#2166ac", lw=1.2, label="MPC grid power [kW]")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Grid power [kW]")
    price_axis = ax.twinx()
    price_axis.plot(mpc.index, mpc["price_eur_per_kwh"] * 100, color="#fdae61", alpha=0.75, label="Day-ahead price [ct/kWh]")
    price_axis.set_ylabel("Price [ct/kWh]")
    ax.set_title("MPC grid exchange and day-ahead price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(figures / "fig2_grid_vs_price.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for axis, column, label in zip(
        axes,
        ("reached_soc_battery_kwh", "reached_soc_hydrogen_kg", "reached_soc_thermal_kwh"),
        ("Battery SOC [kWh]", "Hydrogen inventory [kg]", "Thermal-store SOC [kWh]"),
        strict=True,
    ):
        axis.plot(baseline.index, baseline[column], color="#b2182b", alpha=0.7, label="Baseline")
        axis.plot(mpc.index, mpc[column], color="#2166ac", label="MPC")
        axis.set_ylabel(label)
        axis.legend(loc="best")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Winter storage trajectories")
    fig.tight_layout()
    fig.savefig(figures / "fig3_soc_trajectories.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(16, 24, color="#1a9850", alpha=0.08, label="Comfort band")
    ax.plot(mpc.index, mpc["reached_indoor_temperature_c"], color="#2166ac", lw=1.3, label="MPC indoor")
    ax.plot(baseline.index, baseline["reached_indoor_temperature_c"], color="#b2182b", lw=1.0, alpha=0.7, label="Baseline indoor")
    ax.plot(mpc.index, mpc["outdoor_temperature_c"], color="grey", lw=0.9, alpha=0.7, label="Outdoor")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Winter greenhouse temperature")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(figures / "fig4_temperature.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mpc.index, mpc["electric_boiler_kw"], color="#d73027", lw=1.2, label="Electric boiler [kW]")
    ax.plot(mpc.index, mpc["thermal_charge_kw"], color="#1a9850", lw=1.1, label="Thermal-store charge [kWth]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("MPC power-to-heat operation")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(figures / "fig5_heat_shifting.png")
    plt.close(fig)

    ablation_keys = ("ablation-full", "ablation-no-h2", "ablation-no-tes", "ablation-one-step")
    labels = ("MPC (full)", "no H₂", "no thermal store", "one-step")
    full_cost = _metric(summaries["ablation-full"], "inventory_adjusted_cost_eur")
    values = [
        saving_percent(full_cost, _metric(summaries[key], "inventory_adjusted_cost_eur"))
        for key in ablation_keys
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(labels, values, color=["#2166ac", "#7fb3d5", "#7fb3d5", "#7fb3d5"])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Inventory-adjusted cost change vs full [%]")
    ax.set_title("Winter ablation comparison")
    for bar, value in zip(bars, values, strict=True):
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


def regenerate_publication(
    *,
    candidate_index: str | Path,
    manifest_path: str | Path,
    repository_root: str | Path = ROOT,
    runs_root: str | Path = RUNS_ROOT,
    figures_root: str | Path = FIGURES_ROOT,
    readme_path: str | Path = README_PATH,
) -> dict[str, object]:
    """Write the recipe, figures, and generated README block from pinned bundles."""
    manifest = build_publication_manifest(
        candidate_index,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    candidates, bundles = _load_manifest_bundles(
        manifest,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    summaries = {key: _summary_for(bundle) for key, bundle in bundles.items()}
    render_figures(bundles, figures_root=figures_root)
    rewrite_readme_generated_block(
        readme_path,
        candidates=candidates,
        summaries=summaries,
    )
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate publication artifacts from verified pinned Run Bundles."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    manifest = regenerate_publication(
        candidate_index=args.candidates,
        manifest_path=args.manifest,
    )
    print(
        "Regenerated publication artifacts from "
        f"{len(publication_bundle_ids(manifest))} verified pinned Run Bundles."
    )


if __name__ == "__main__":
    main()
