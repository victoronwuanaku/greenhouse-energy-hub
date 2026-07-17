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
            "executable_path_hashes": {"control/mpc_controller.py": "4" * 64},
            "committed_executable_path_hashes": {
                "control/mpc_controller.py": "4" * 64
            },
            "publication_eligible": publication_eligible,
            "dirty_executable_paths": [] if publication_eligible else ["control/mpc_controller.py"],
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
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


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
        verify_run_bundle(
            bundle_path,
            expected_identifier=bundle_id,
            repository_root=ROOT,
        )


def test_bundle_verification_rejects_dirty_executable_source(tmp_path):
    from accounting import verify_run_bundle

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
    from accounting import verify_run_bundle

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
    from accounting import verify_run_bundle

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
    from accounting import load_run_bundle

    _bundle_path, bundle_id = _write_handcrafted_bundle(tmp_path)
    requested = bundle_id[:12] if identifier_kind == "shortened" else "f" * 64

    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        load_run_bundle(tmp_path, requested, repository_root=ROOT)
