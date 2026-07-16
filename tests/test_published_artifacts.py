from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"

BASELINE_CAPABILITY_POLICY = {
    "hydrogen_dispatch": False,
    "thermal_store_charging": False,
    "grid_battery_charging": False,
    "battery_discharge_price_threshold_eur_per_kwh": 0.12,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_handcrafted_bundle(
    root: Path,
    *,
    validation: dict[str, object] | None = None,
    diagnostics_csv: bytes | None = None,
    publication_eligible: bool = True,
) -> tuple[Path, str]:
    """Write internally hash-consistent bytes so each rejection isolates one gate."""
    validation = validation or {
        "complete": True,
        "valid": True,
        "checked_operating_steps": 1,
        "steps": [
            {
                "operating_step": 0,
                "solver_success": True,
                "solver_return_status": "Solve_Succeeded",
                "physical_invariants_valid": True,
            }
        ],
    }
    diagnostics_csv = diagnostics_csv or (
        b"operating_step,decision_status,solver_success,solver_return_status\n"
        b"0,success,true,Solve_Succeeded\n"
    )
    members = {
        "trajectory.csv": (
            b"operating_step,timestamp_utc,grid_kw\n"
            b"0,2023-01-02T00:00:00Z,100.0\n"
        ),
        "controller_diagnostics.csv": diagnostics_csv,
        "summary.json": _canonical_json_bytes(
            {
                "grid_cost_eur": 12.5,
                "operating_cost_eur": 12.5,
                "inventory_adjusted_cost_eur": 12.5,
                "comfort_violation_c_h": 0.0,
                "wear_sensitivities": {"0x": 12.5, "1x": 12.5, "2x": 12.5},
            }
        ),
        "validation.json": _canonical_json_bytes(validation),
    }
    identity_manifest = {
        "schema_version": "run-bundle-v1",
        "canonicalization_version": "canonical-json-v1",
        "run_specification_identifier": "1" * 64,
        "scenario": {"name": "handcrafted"},
        "controller": {"name": "mpc", "configuration": {}, "capability_policy": {}},
        "asset_capabilities": {"battery": True, "hydrogen": True, "thermal_store": True},
        "evaluation_policy": {"name": "greenhouse-hub-evaluation", "version": "1"},
        "code_provenance": {
            "git_revision": "2" * 40,
            "executable_source_tree_sha256": "3" * 64,
            "executable_path_hashes": {"control/mpc_controller.py": "4" * 64},
            "publication_eligible": publication_eligible,
            "dirty_executable_paths": [] if publication_eligible else ["control/mpc_controller.py"],
        },
        "runtime": {"python": "3.11", "do_mpc": "5.1", "casadi": "3.7"},
        "input_hashes": {"scenario": "5" * 64},
        "member_hashes": {name: _sha256(data) for name, data in members.items()},
        "valid": True,
    }
    bundle_id = _sha256(_canonical_json_bytes(identity_manifest))
    manifest = {**identity_manifest, "run_bundle_identifier": bundle_id}
    bundle_path = root / f"handcrafted--mpc--{bundle_id}"
    bundle_path.mkdir(parents=True)
    for name, data in members.items():
        (bundle_path / name).write_bytes(data)
    (bundle_path / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return bundle_path, bundle_id


@pytest.mark.xfail(strict=True, reason="PF-04: README calls a limited-capability Controller fair")
def test_readme_names_the_baseline_as_limited_capability_not_unqualified_fair():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").casefold()

    assert "fair rule-based baseline" not in readme
    assert "against a fair baseline" not in readme
    assert "limited-capability" in readme


@pytest.mark.xfail(strict=True, reason="PF-04: Baseline Run manifests omit capability policy")
def test_every_baseline_run_manifest_contains_the_exact_capability_policy():
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RUNS.glob("*/manifest.json"))
    ]
    baseline_manifests = [
        manifest
        for manifest in manifests
        if manifest.get("controller", {}).get("name") == "baseline"
    ]

    assert baseline_manifests, "no Baseline Run manifests found"
    for manifest in baseline_manifests:
        assert manifest["controller"]["capability_policy"] == BASELINE_CAPABILITY_POLICY


@pytest.mark.xfail(strict=True, reason="PF-04: causal claims do not cite ablation Run Bundles")
def test_each_causal_claim_cites_its_corresponding_ablation_bundle():
    publication = json.loads(
        (ROOT / "results" / "publication_manifest.json").read_text(encoding="utf-8")
    )
    ablations = publication["ablations"]
    for bundle_id in ablations.values():
        assert re.fullmatch(r"[0-9a-f]{64}", bundle_id)

    claim_requirements = {
        "no-h2": ("h₂ buffer adds", "h2 buffer adds", "hydrogen buffer adds"),
        "no-tes": ("thermal store is the decisive driver", "thermal store, displacing"),
        "one-step": ("price foresight is essential", "myopic controller"),
    }
    readme_lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    for ablation, phrases in claim_requirements.items():
        matching_lines = [
            line
            for line in readme_lines
            if any(phrase in line.casefold() for phrase in phrases)
        ]
        for line in matching_lines:
            assert ablations[ablation] in line


@pytest.mark.xfail(strict=True, reason="PF-06: publication accepts failed validation evidence")
def test_bundle_verification_rejects_a_failed_validation_report(tmp_path):
    from accounting import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(
        tmp_path,
        validation={
            "complete": True,
            "valid": False,
            "checked_operating_steps": 1,
            "steps": [{"operating_step": 0, "solver_success": True}],
        },
    )
    with pytest.raises(ValueError):
        verify_run_bundle(bundle_path, expected_identifier=bundle_id)


@pytest.mark.xfail(strict=True, reason="PF-06: publication accepts missing solver status")
def test_bundle_verification_rejects_missing_solver_status(tmp_path):
    from accounting import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(
        tmp_path,
        diagnostics_csv=(
            b"operating_step,decision_status,solver_return_status\n"
            b"0,success,Solve_Succeeded\n"
        ),
    )
    with pytest.raises(ValueError):
        verify_run_bundle(bundle_path, expected_identifier=bundle_id)


@pytest.mark.xfail(strict=True, reason="PF-06: publication accepts dirty executable source")
def test_bundle_verification_rejects_dirty_executable_source(tmp_path):
    from accounting import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(
        tmp_path,
        publication_eligible=False,
    )
    with pytest.raises(ValueError):
        verify_run_bundle(bundle_path, expected_identifier=bundle_id)


@pytest.mark.xfail(strict=True, reason="PF-06/ADR-0007: changed member bytes are accepted")
def test_bundle_verification_rejects_a_changed_member_byte(tmp_path):
    from accounting import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(tmp_path)
    trajectory = bundle_path / "trajectory.csv"
    changed = bytearray(trajectory.read_bytes())
    changed[-2] = ord("1")
    trajectory.write_bytes(bytes(changed))

    with pytest.raises(ValueError):
        verify_run_bundle(bundle_path, expected_identifier=bundle_id)


@pytest.mark.xfail(strict=True, reason="PF-06/ADR-0007: publication lookup accepts non-authoritative IDs")
@pytest.mark.parametrize("identifier_kind", ["shortened", "unknown"])
def test_publication_lookup_requires_a_known_full_bundle_identifier(tmp_path, identifier_kind):
    from accounting import load_run_bundle

    _bundle_path, bundle_id = _write_handcrafted_bundle(tmp_path)
    requested = bundle_id[:12] if identifier_kind == "shortened" else "f" * 64

    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        load_run_bundle(tmp_path, requested)
