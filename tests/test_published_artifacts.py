from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
DIAGNOSTICS = ROOT / "results" / "diagnostics"
CANDIDATE_INDEX = DIAGNOSTICS / "publication-candidates.json"

EXPECTED_PUBLICATION_CANDIDATE_KEYS = frozenset(
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
EXPECTED_UTC_WINDOWS = {
    "winter-2023-14d": (
        "2023-01-01T23:00:00Z",
        "2023-01-15T23:00:00Z",
        "2023-01-16T23:00:00Z",
    ),
    "summer-2023-14d": (
        "2023-05-31T22:00:00Z",
        "2023-06-14T22:00:00Z",
        "2023-06-15T22:00:00Z",
    ),
}
FULL_ASSET_CAPABILITIES = {
    "battery": True,
    "hydrogen": True,
    "thermal_store": True,
}
EXPECTED_CANDIDATE_SEMANTICS = {
    "ablation-full": ("winter-2023-14d", "mpc", 24, FULL_ASSET_CAPABILITIES),
    "ablation-no-h2": (
        "winter-2023-14d",
        "mpc",
        24,
        {"battery": True, "hydrogen": False, "thermal_store": True},
    ),
    "ablation-no-tes": (
        "winter-2023-14d",
        "mpc",
        24,
        {"battery": True, "hydrogen": True, "thermal_store": False},
    ),
    "ablation-one-step": ("winter-2023-14d", "mpc", 1, FULL_ASSET_CAPABILITIES),
    "summer-baseline": ("summer-2023-14d", "baseline", None, FULL_ASSET_CAPABILITIES),
    "summer-mpc": ("summer-2023-14d", "mpc", 24, FULL_ASSET_CAPABILITIES),
    "winter-baseline": ("winter-2023-14d", "baseline", None, FULL_ASSET_CAPABILITIES),
    "winter-mpc": ("winter-2023-14d", "mpc", 24, FULL_ASSET_CAPABILITIES),
}

BASELINE_CAPABILITY_POLICY = {
    "hydrogen_dispatch": False,
    "thermal_store_charging": False,
    "grid_battery_charging": False,
    "battery_discharge_price_threshold_eur_per_kwh": 0.12,
}

REQUIRED_CAUSAL_ABLATIONS = frozenset({"no-h2", "no-tes", "one-step"})
CAUSAL_TOPIC_PATTERNS = {
    "no-h2": re.compile(r"\b(?:h2|hydrogen|electrolys(?:er|is)|fuel[- ]?cell)\b"),
    "no-tes": re.compile(
        r"\b(?:tes|thermal(?: energy)? (?:store|storage)|heat (?:store|storage))\b"
    ),
    "one-step": re.compile(
        r"\b(?:foresight|look[- ]?ahead|myop(?:ic|ia)|forecast horizon|"
        r"horizon length|one[- ]?step)\b"
    ),
}
CAUSAL_LANGUAGE = re.compile(
    r"\b(?:account(?:s|ed|ing)? for|add(?:s|ed|ing)?|"
    r"attribut(?:e|es|ed|ing|ion)|benefit(?:s|ed|ing)?|buffer(?:s|ed|ing)?|"
    r"caus(?:e|es|ed|ing|al)|contribut(?:e|es|ed|ing|ion)|cut(?:s|ting)?|"
    r"deliver(?:s|ed|ing)?|depend(?:s|ed|ing)? on|displac(?:e|es|ed|ing)|"
    r"driv(?:e|es|en|ing|er)|enabl(?:e|es|ed|ing)|explain(?:s|ed|ing)?|"
    r"improv(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|is essential|is decisive|"
    r"is responsible|is key|is critical|is necessary|lead(?:s|ing)? to|"
    r"lower(?:s|ed|ing)?|produc(?:e|es|ed|ing)|provid(?:e|es|ed|ing)|"
    r"reduc(?:e|es|ed|ing|tion)|sav(?:e|es|ed|ing)|shift(?:s|ed|ing)?|"
    r"win(?:s|ning)?|worsen(?:s|ed|ing)?|contribution|reason|source)\b"
)
OUTCOME_LANGUAGE = re.compile(
    r"\b(?:comfort|costs?|economic|performance|saving|value|violation|"
    r"cheap(?:er)?|expensive|better|worse|points?|percent(?:age)?)\b|[%€]"
)
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


def _readme_result_statements(markdown: str) -> list[str]:
    section = re.search(
        r"(?ms)^## Results\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
        markdown,
    )
    if section is None:
        raise AssertionError("README has no Results section")

    statements: list[str] = []
    for raw_line in section.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "|", "![", "<!--")):
            continue
        statements.extend(
            statement.strip()
            for statement in re.split(r"(?<=[.!?])\s+", line)
            if statement.strip()
        )
    return statements


def _uncited_causal_claims(
    markdown: str,
    ablation_ids: dict[str, str],
) -> list[tuple[str, str]]:
    uncited: list[tuple[str, str]] = []
    for statement in _readme_result_statements(markdown):
        normalized = statement.casefold().replace("h₂", "h2")
        # In a Results section, any capability mention tied to an outcome is an
        # attribution even if its verb is reworded; functional causal verbs are
        # also caught when no numeric/economic outcome term appears.
        is_causal = bool(CAUSAL_LANGUAGE.search(normalized)) or bool(
            OUTCOME_LANGUAGE.search(normalized)
        )
        if not is_causal:
            continue
        for ablation, topic_pattern in CAUSAL_TOPIC_PATTERNS.items():
            if topic_pattern.search(normalized) and ablation_ids[ablation] not in statement:
                uncited.append((ablation, statement))
    return uncited


def _generated_publication_candidates() -> dict[str, str]:
    """Prefer the ignored generator index, else reconstruct the committed recipe."""
    from greenhouse_energy_hub.evaluation import read_publication_candidates

    candidates = read_publication_candidates(
        CANDIDATE_INDEX,
        manifest_path=ROOT / "results" / "publication_manifest.json",
    )
    assert isinstance(candidates, dict)
    assert set(candidates) == EXPECTED_PUBLICATION_CANDIDATE_KEYS
    assert all(
        isinstance(identifier, str)
        and re.fullmatch(r"[0-9a-f]{64}", identifier)
        for identifier in candidates.values()
    )
    assert len(set(candidates.values())) == len(candidates)
    return candidates


def test_candidates_fall_back_to_the_committed_manifest_when_index_is_missing(tmp_path):
    from greenhouse_energy_hub.evaluation import read_publication_candidates

    candidates = read_publication_candidates(
        tmp_path / "publication-candidates.json",
        manifest_path=ROOT / "results" / "publication_manifest.json",
    )

    assert candidates == _generated_publication_candidates()


def _write_handcrafted_bundle(
    root: Path,
    *,
    validation: dict[str, object] | None = None,
    diagnostics_csv: bytes | None = None,
    publication_eligible: bool = True,
    input_hashes: dict[str, object] | None = None,
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
                "control_valid": True,
                "flows_valid": True,
                "successor_valid": True,
                "physical_invariants_valid": True,
            }
        ],
    }
    diagnostics_csv = diagnostics_csv or (
        b"operating_step,adapter,decision_status,solver_success,solver_return_status,"
        b"solver_iterations,solver_wall_seconds,forecast_start_utc,forecast_end_utc,"
        b"terminal_electric_value_eur_per_kwh,terminal_heat_value_eur_per_kwhth\n"
        b"0,mpc,success,true,Solve_Succeeded,3,0.01,2023-01-02T00:00:00Z,"
        b"2023-01-02T00:00:00Z,0.1,0.02\n"
    )
    input_hashes = input_hashes or {
        "scenario": "5" * 64,
        "sources": {"data/source.csv": "6" * 64},
        "sidecars": {"data/source.provenance.json": "7" * 64},
    }
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
            "executable_path_hashes": {"src/greenhouse_energy_hub/controllers/mpc.py": "4" * 64},
            "committed_executable_path_hashes": {
                "src/greenhouse_energy_hub/controllers/mpc.py": "4" * 64
            },
            "publication_eligible": publication_eligible,
            "dirty_executable_paths": [] if publication_eligible else ["src/greenhouse_energy_hub/controllers/mpc.py"],
            "untracked_executable_paths": [],
        },
        "runtime": {
            "python": "3.11",
            "platform": "test-platform",
            "do_mpc": "5.1",
            "casadi": "3.7",
            "numpy": "2.0",
            "pandas": "2.0",
        },
        "input_hashes": input_hashes,
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


def test_readme_names_the_baseline_as_limited_capability_not_unqualified_fair():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").casefold()
    unqualified_fairness = " ".join(("fair", "rule-based", "baseline"))

    assert unqualified_fairness not in readme
    assert "against a fair baseline" not in readme
    assert "limited-capability" in readme


def test_every_baseline_run_manifest_contains_the_exact_capability_policy():
    _generated_publication_candidates()
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


def test_each_causal_claim_cites_its_corresponding_ablation_bundle():
    publication = json.loads(
        (ROOT / "results" / "publication_manifest.json").read_text(encoding="utf-8")
    )
    ablations = publication["ablations"]
    assert REQUIRED_CAUSAL_ABLATIONS <= set(ablations)
    for ablation in sorted(REQUIRED_CAUSAL_ABLATIONS):
        assert re.fullmatch(r"[0-9a-f]{64}", ablations[ablation])

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    unsupported = _uncited_causal_claims(readme, ablations)
    detail = "\n".join(f"{ablation}: {claim}" for ablation, claim in unsupported)
    assert not unsupported, f"causal claims lack exact ablation bundle IDs:\n{detail}"


@pytest.mark.parametrize(
    ("ablation", "reworded_claim"),
    [
        ("no-h2", "Hydrogen dispatch accounts for lower winter operating cost"),
        ("no-tes", "Heat storage enables the controller to avoid expensive hours"),
        ("one-step", "Longer look-ahead reduces the reported comfort violation"),
    ],
)
def test_causal_claim_scan_catches_reworded_unsupported_attribution(
    ablation,
    reworded_claim,
):
    ablations = {
        "no-h2": "a" * 64,
        "no-tes": "b" * 64,
        "one-step": "c" * 64,
    }
    unsupported_readme = f"# Study\n\n## Results\n\n{reworded_claim}.\n\n## Methods\n"
    assert _uncited_causal_claims(unsupported_readme, ablations) == [
        (ablation, f"{reworded_claim}.")
    ]

    cited_readme = (
        f"# Study\n\n## Results\n\n{reworded_claim} "
        f"(`{ablations[ablation]}`).\n\n## Methods\n"
    )
    assert _uncited_causal_claims(cited_readme, ablations) == []


def test_causal_claim_scan_allows_genuinely_noncausal_method_description():
    ablations = {
        "no-h2": "a" * 64,
        "no-tes": "b" * 64,
        "one-step": "c" * 64,
    }
    readme = (
        "# Study\n\n## Results\n\n"
        "The evaluated configuration contains hydrogen, thermal storage, and a 24-hour horizon.\n\n"
        "## Methods\n"
    )
    assert _uncited_causal_claims(readme, ablations) == []


def test_bundle_verification_rejects_a_failed_validation_report(tmp_path):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

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
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


def test_bundle_verification_rejects_missing_solver_status(tmp_path):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(
        tmp_path,
        diagnostics_csv=(
            b"operating_step,decision_status,solver_return_status\n"
            b"0,success,Solve_Succeeded\n"
        ),
    )
    with pytest.raises(ValueError):
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


def test_bundle_verification_rejects_dirty_executable_source(tmp_path):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(
        tmp_path,
        publication_eligible=False,
    )
    with pytest.raises(ValueError):
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


def test_bundle_verification_rejects_a_changed_member_byte(tmp_path):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(tmp_path)
    trajectory = bundle_path / "trajectory.csv"
    changed = bytearray(trajectory.read_bytes())
    changed[-2] = ord("1")
    trajectory.write_bytes(bytes(changed))

    with pytest.raises(ValueError):
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


def test_bundle_verification_rejects_non_full_input_hashes(tmp_path):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    bundle_path, bundle_id = _write_handcrafted_bundle(
        tmp_path,
        input_hashes={
            "scenario": "5" * 12,
            "sources": {"data/source.csv": "6" * 64},
            "sidecars": {"data/source.provenance.json": "7" * 64},
        },
    )
    with pytest.raises(ValueError, match="full lowercase SHA-256"):
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


@pytest.mark.parametrize("identifier_kind", ["shortened", "unknown"])
def test_publication_lookup_requires_a_known_full_bundle_identifier(tmp_path, identifier_kind):
    from greenhouse_energy_hub.evaluation import load_run_bundle

    _bundle_path, bundle_id = _write_handcrafted_bundle(tmp_path)
    requested = bundle_id[:12] if identifier_kind == "shortened" else "f" * 64

    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        load_run_bundle(tmp_path, requested, repository_root=ROOT)


def _csv_member_rows(bundle_path: Path, member: str) -> list[dict[str, str]]:
    with (bundle_path / member).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_generated_bundle_full_scope_evidence():
    """Verify the eight generated Task 13 candidates and all publication evidence."""
    from greenhouse_energy_hub.evaluation import load_run_bundle, verify_run_bundle

    candidates = _generated_publication_candidates()
    if DIAGNOSTICS.exists():
        assert sorted(
            path.relative_to(DIAGNOSTICS).as_posix()
            for path in DIAGNOSTICS.rglob("*")
        ) == ["publication-candidates.json"]
    verified_bundles = {}
    for key, identifier in candidates.items():
        resolved = load_run_bundle(RUNS, identifier, repository_root=ROOT)
        verified = verify_run_bundle(
            resolved.path,
            expected_identifier=identifier,
            repository_root=ROOT,
        )
        assert resolved.identifier == verified.identifier == identifier
        verified_bundles[key] = verified

    for key, bundle in verified_bundles.items():
        manifest_member = json.loads(
            (bundle.path / "manifest.json").read_text(encoding="utf-8")
        )
        scenario = manifest_member["scenario"]
        controller = manifest_member["controller"]
        provenance = manifest_member["code_provenance"]
        (
            expected_scenario_name,
            expected_controller_name,
            expected_horizon_steps,
            expected_asset_capabilities,
        ) = EXPECTED_CANDIDATE_SEMANTICS[key]
        summary = json.loads((bundle.path / "summary.json").read_text(encoding="utf-8"))
        validation = json.loads(
            (bundle.path / "validation.json").read_text(encoding="utf-8")
        )
        trajectory_rows = _csv_member_rows(bundle.path, "trajectory.csv")
        diagnostic_rows = _csv_member_rows(
            bundle.path,
            "controller_diagnostics.csv",
        )

        assert scenario["name"] == expected_scenario_name
        assert controller["name"] == expected_controller_name
        assert manifest_member["asset_capabilities"] == expected_asset_capabilities
        if expected_horizon_steps is None:
            assert "horizon_steps" not in controller["configuration"]
        else:
            assert controller["configuration"]["horizon_steps"] == expected_horizon_steps
        assert scenario["operating_step_count"] == 336
        assert len(validation["steps"]) == 336
        assert len(trajectory_rows) == len(diagnostic_rows) == 336
        assert [row["operating_step"] for row in trajectory_rows] == [
            str(step) for step in range(336)
        ]
        assert [row["operating_step"] for row in diagnostic_rows] == [
            str(step) for step in range(336)
        ]
        assert (
            scenario["operating_start_utc"],
            scenario["operating_end_utc"],
            scenario["forecast_end_utc"],
        ) == EXPECTED_UTC_WINDOWS[scenario["name"]]

        for operating_step, evidence in enumerate(validation["steps"]):
            assert evidence["operating_step"] == operating_step
            assert evidence["decision_status"] == "success"
            assert all(
                evidence[field] is True
                for field in (
                    "control_valid",
                    "flows_valid",
                    "successor_valid",
                    "physical_invariants_valid",
                )
            )

        assert all(row["adapter"] == controller["name"] for row in diagnostic_rows)
        assert all(row["decision_status"] == "success" for row in diagnostic_rows)
        if controller["name"] == "mpc":
            for row in diagnostic_rows:
                assert row["solver_success"] == "true"
                assert row["solver_return_status"] in {
                    "Solve_Succeeded",
                    "Solved_To_Acceptable_Level",
                }
                assert int(row["solver_iterations"]) >= 0
                assert math.isfinite(float(row["solver_wall_seconds"]))
                assert float(row["solver_wall_seconds"]) >= 0.0
        else:
            assert controller["name"] == "baseline"
            assert controller["capability_policy"] == BASELINE_CAPABILITY_POLICY
            assert all(
                not row[field]
                for row in diagnostic_rows
                for field in (
                    "solver_success",
                    "solver_return_status",
                    "solver_iterations",
                    "solver_wall_seconds",
                )
            )

        assert summary["policy"] == manifest_member["evaluation_policy"]
        assert len(summary["step_line_items"]) == 336
        assert set(summary["wear_sensitivities"]) == {"0x", "1x", "2x"}

        assert provenance["publication_eligible"] is True
        assert re.fullmatch(r"[0-9a-f]{40}", provenance["git_revision"])
        assert provenance["dirty_executable_paths"] == []
        assert provenance["untracked_executable_paths"] == []
        assert provenance["executable_path_hashes"]
        assert (
            provenance["executable_path_hashes"]
            == provenance["committed_executable_path_hashes"]
        )
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in provenance["executable_path_hashes"].values()
        ), key


def test_publisher_builds_the_exact_verified_full_id_recipe(tmp_path):
    from greenhouse_energy_hub.evaluation import build_publication_manifest

    candidates = _generated_publication_candidates()
    candidate_index = tmp_path / "publication-candidates.json"
    candidate_index.write_text(json.dumps(candidates), encoding="utf-8")
    publication = build_publication_manifest(
        candidate_index,
        runs_root=RUNS,
        repository_root=ROOT,
    )

    assert publication == {
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


@pytest.mark.parametrize(
    ("candidate_key", "candidate_value"),
    [
        ("unexpected", "0" * 64),
        ("winter-mpc", "a" * 12),
    ],
)
def test_publisher_rejects_extra_candidate_keys_and_identifier_prefixes(
    tmp_path,
    candidate_key,
    candidate_value,
):
    from greenhouse_energy_hub.evaluation import build_publication_manifest

    candidates = _generated_publication_candidates()
    candidates[candidate_key] = candidate_value
    candidate_index = tmp_path / "publication-candidates.json"
    candidate_index.write_text(json.dumps(candidates), encoding="utf-8")

    with pytest.raises(ValueError):
        build_publication_manifest(
            candidate_index,
            runs_root=RUNS,
            repository_root=ROOT,
        )


def test_publisher_rejects_dirty_provenance_and_policy_mismatch():
    from greenhouse_energy_hub.evaluation import (
        load_run_bundle,
        validate_publication_bundles,
    )

    candidates = _generated_publication_candidates()
    baseline = load_run_bundle(
        RUNS,
        candidates["winter-baseline"],
        repository_root=ROOT,
    )
    mpc = load_run_bundle(
        RUNS,
        candidates["winter-mpc"],
        repository_root=ROOT,
    )

    dirty_baseline = type(baseline)(
        identifier=baseline.identifier,
        specification_identifier=baseline.specification_identifier,
        path=baseline.path,
        manifest={
            **baseline.manifest,
            "code_provenance": {
                **baseline.manifest["code_provenance"],
                "publication_eligible": False,
                "dirty_executable_paths": [
                    "src/greenhouse_energy_hub/controllers/baseline.py"
                ],
            },
        },
    )
    with pytest.raises(ValueError, match="dirty|eligible"):
        validate_publication_bundles(
            {"winter-baseline": dirty_baseline, "winter-mpc": mpc}
        )

    mismatched_mpc = type(mpc)(
        identifier=mpc.identifier,
        specification_identifier=mpc.specification_identifier,
        path=mpc.path,
        manifest={
            **mpc.manifest,
            "evaluation_policy": {
                **mpc.manifest["evaluation_policy"],
                "version": "forged-policy-version",
            },
        },
    )
    with pytest.raises(ValueError, match="Evaluation Policies"):
        validate_publication_bundles(
            {"winter-baseline": baseline, "winter-mpc": mismatched_mpc}
        )


def test_publication_manifest_retains_each_resolvable_bundle_in_the_proposed_commit():
    from greenhouse_energy_hub.evaluation import (
        publication_bundle_ids,
        validate_committed_publication_bundles,
    )
    from greenhouse_energy_hub.evaluation import load_run_bundle

    publication = json.loads(
        (ROOT / "results" / "publication_manifest.json").read_text(encoding="utf-8")
    )
    identifiers = publication_bundle_ids(publication)

    assert len(identifiers) == len(EXPECTED_PUBLICATION_CANDIDATE_KEYS)
    validate_committed_publication_bundles(
        publication,
        runs_root=RUNS,
        repository_root=ROOT,
    )
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--cached", "results/runs"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    )
    assert {
        f"results/runs/{bundle.path.name}/manifest.json"
        for bundle in (
            load_run_bundle(RUNS, identifier, repository_root=ROOT)
            for identifier in identifiers
        )
    } <= tracked


def test_publication_core_has_no_matplotlib_backend_import_side_effect():
    rendered_backends = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; before = matplotlib.get_backend(); "
            "import greenhouse_energy_hub.evaluation; "
            "print(before); print(matplotlib.get_backend())",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().casefold().splitlines()

    assert rendered_backends[0] == rendered_backends[1]


def test_publication_rejects_missing_candidate_key_uppercase_and_non_string_identifiers(
    tmp_path,
):
    from greenhouse_energy_hub.evaluation import build_publication_manifest

    candidates = _generated_publication_candidates()
    for name, invalid_candidates in {
        "missing": {key: value for key, value in candidates.items() if key != "winter-mpc"},
        "uppercase": {**candidates, "winter-mpc": candidates["winter-mpc"].upper()},
        "non-string": {**candidates, "winter-mpc": 3},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(invalid_candidates), encoding="utf-8")
        with pytest.raises(ValueError):
            build_publication_manifest(path, runs_root=RUNS, repository_root=ROOT)


def test_publication_rejects_a_summary_without_all_required_sensitivities():
    from greenhouse_energy_hub.evaluation import validate_publication_summary

    candidates = _generated_publication_candidates()
    bundle_path = RUNS / f"winter-2023-14d--mpc--{candidates['winter-mpc']}"
    summary = json.loads((bundle_path / "summary.json").read_text(encoding="utf-8"))
    summary["wear_sensitivities"].pop("2x")

    with pytest.raises(ValueError, match="required 0x, 1x, and 2x sensitivities"):
        validate_publication_summary(summary, identifier=candidates["winter-mpc"])


def test_publication_rejects_malformed_manifest_mapping():
    from greenhouse_energy_hub.evaluation import publication_bundle_ids

    manifest = json.loads(
        (ROOT / "results" / "publication_manifest.json").read_text(encoding="utf-8")
    )
    manifest["figures"] = {"fig1_cumulative_cost.png": []}

    with pytest.raises(ValueError, match="figures are incomplete"):
        publication_bundle_ids(manifest)


def test_readme_rewrite_preserves_bytes_outside_generated_markers(tmp_path):
    from greenhouse_energy_hub.evaluation import (
        read_publication_manifest,
        rewrite_publication_readme_block,
        load_verified_publication_evidence,
    )

    readme = tmp_path / "README.md"
    before = (ROOT / "README.md").read_bytes()
    readme.write_bytes(before)
    evidence = load_verified_publication_evidence(
        read_publication_manifest(ROOT / "results" / "publication_manifest.json"),
        runs_root=RUNS,
        repository_root=ROOT,
    )

    rewrite_publication_readme_block(
        readme,
        candidates=evidence.candidates,
        summaries=evidence.summaries,
    )
    after = readme.read_bytes()
    begin = before.index(b"<!-- BEGIN GENERATED RESULTS: DO NOT EDIT -->")
    before_end = before.index(b"<!-- END GENERATED RESULTS -->")
    after_end = after.index(b"<!-- END GENERATED RESULTS -->")
    assert after[:begin] == before[:begin]
    assert after[after_end:] == before[before_end:]


def test_ablation_cost_difference_is_signed_from_full_to_variant():
    from greenhouse_energy_hub.evaluation import publication_cost_difference_percent

    assert publication_cost_difference_percent(100.0, 102.0) == pytest.approx(2.0)
    assert publication_cost_difference_percent(100.0, 98.0) == pytest.approx(-2.0)
