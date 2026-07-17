from __future__ import annotations

import csv
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


BASELINE_CAPABILITY_POLICY = {
    "hydrogen_dispatch": False,
    "thermal_store_charging": False,
    "grid_battery_charging": False,
    "battery_discharge_price_threshold_eur_per_kwh": 0.12,
}

TEST_SOURCE_BYTES = b"timestamp,value\n2023-01-02T00:00:00Z,1.0\n"
TEST_SIDECAR_BYTES = b'{"fixture":"sidecar"}\n'
TEST_SOURCE_SHA256 = hashlib.sha256(TEST_SOURCE_BYTES).hexdigest()
TEST_SIDECAR_SHA256 = hashlib.sha256(TEST_SIDECAR_BYTES).hexdigest()


def _two_step_valid_run(
    *,
    controller_name: str = "baseline",
    controller_configuration: dict[str, object] | None = None,
    capability_policy: dict[str, object] | None = None,
):
    """An exact, hand-built two-step Run with deliberately varied line items."""
    from greenhouse_energy_hub.simulation import (
        DecisionDiagnostics,
        OperatingRecord,
        ValidRun,
        ValidationReport,
    )
    from greenhouse_energy_hub.hub import (
        ExogenousInputs,
        HubConfiguration,
        HubControl,
        HubFlows,
        HubState,
    )
    from greenhouse_energy_hub.scenarios import Scenario, ScenarioPoint

    start = datetime(2023, 1, 2, tzinfo=timezone.utc)
    operating_points = (
        ScenarioPoint(start, 0.10, 0.0, 400.0, 5.0, 0.0),
        ScenarioPoint(start + timedelta(hours=1), 0.20, 0.0, 400.0, 5.0, 0.0),
    )
    points = operating_points + (
        (ScenarioPoint(start + timedelta(hours=2), 0.30, 0.0, 400.0, 5.0, 0.0),)
        if controller_name == "mpc"
        else ()
    )
    scenario = Scenario(
        name="two-step-evaluation",
        operating_start=start,
        operating_end=start + timedelta(hours=2),
        forecast_end=start + timedelta(hours=2 + (controller_name == "mpc")),
        forecast_horizon_capacity_steps=1 if controller_name == "mpc" else 0,
        step_duration=timedelta(hours=1),
        operating_step_count=2,
        points=points,
        provenance=(),
    )

    initial = HubState(100.0, 10.0, 100.0, 19.0)
    reached_first = HubState(110.0, 10.0, 120.0, 25.5)
    terminal = HubState(90.0, 9.0, 100.0, 15.0)
    first_control = HubControl(10.0, 2.0, 30.0, 5.0, 0.0, 0.0, 20.0, 4.0, 0.0)
    second_control = HubControl(
        0.0, 8.0, 0.0, 10.0, 0.0, 0.0, 0.0, 6.0, 0.0
    )
    first_exogenous = ExogenousInputs(0.0, 400.0, 0.10, 5.0, 0.0)
    second_exogenous = ExogenousInputs(0.0, 400.0, 0.20, 5.0, 0.0)

    records = (
        OperatingRecord(
            0,
            points[0].timestamp_utc,
            initial,
            first_control,
            first_exogenous,
            reached_first,
            HubFlows(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        OperatingRecord(
            1,
            points[1].timestamp_utc,
            reached_first,
            second_control,
            second_exogenous,
            terminal,
            HubFlows(-50.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    diagnostics = tuple(
        DecisionDiagnostics(
            adapter=controller_name,
            decision_status="success",
            solver_success=True if controller_name == "mpc" else None,
            solver_return_status="Solve_Succeeded" if controller_name == "mpc" else None,
            solver_iterations=3 if controller_name == "mpc" else None,
            solver_wall_seconds=0.01 if controller_name == "mpc" else None,
            forecast_start_utc=point.timestamp_utc,
            forecast_end_utc=(
                points[index + 1].timestamp_utc
                if controller_name == "mpc"
                else point.timestamp_utc
            ),
            terminal_electric_value_eur_per_kwh=(
                point.price_eur_per_kwh if controller_name == "mpc" else None
            ),
            terminal_heat_value_eur_per_kwhth=(
                point.price_eur_per_kwh / 3.5 if controller_name == "mpc" else None
            ),
        )
        for index, point in enumerate(operating_points)
    )
    config = HubConfiguration()
    return ValidRun(
        scenario=scenario,
        controller_name=controller_name,
        controller_configuration=(
            controller_configuration
            if controller_configuration is not None
            else ({"horizon_steps": 1} if controller_name == "mpc" else {})
        ),
        capability_policy=capability_policy or {},
        hub_configuration=config,
        initial_state=initial,
        records=records,
        controller_diagnostics=diagnostics,
        terminal_state=terminal,
        validation=ValidationReport(True, True, 2, ()),
    )


def _policy():
    from greenhouse_energy_hub.evaluation import EvaluationPolicy, WearCoefficients

    return EvaluationPolicy(
        wear=WearCoefficients(
            battery_eur_per_kwh=0.005,
            thermal_store_eur_per_kwh=0.0005,
            electrolyser_eur_per_kwh=0.002,
            fuel_cell_eur_per_kwh=0.002,
        )
    )


def _publication_ready_run(**run_options):
    """Build a compact, provenance-backed Run whose records obey shared physics."""
    from greenhouse_energy_hub.simulation import OperatingRecord
    from greenhouse_energy_hub.hub import (
        ExogenousInputs,
        HubControl,
        advance_hub,
        initial_state,
    )
    from greenhouse_energy_hub.scenarios import ScenarioPoint, SourceProvenance

    run = _two_step_valid_run(**run_options)
    provenance = SourceProvenance(
        source_name="test-source",
        source_path="data/test-source.csv",
        sha256=TEST_SOURCE_SHA256,
        acquisition_parameters={
            "sidecar_path": "data/test-source.provenance.json",
            "sidecar_sha256": TEST_SIDECAR_SHA256,
            "parameters": {"fixture": "two-step"},
        },
        original_timezone="UTC",
        units={"value": "kW"},
        transformations=("test-fixture",),
    )
    safe_points = tuple(
        ScenarioPoint(
            point.timestamp_utc,
            point.price_eur_per_kwh,
            point.pv_kw,
            point.electric_load_kw,
            19.0,
            point.irradiance_w_per_m2,
        )
        for point in run.scenario.points
    )
    scenario = replace(
        run.scenario,
        points=safe_points,
        provenance=(provenance,),
    )
    state = initial_state(run.hub_configuration)
    records = []
    zero_control = HubControl(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for operating_step, point in enumerate(
        scenario.points[: scenario.operating_step_count]
    ):
        exogenous = ExogenousInputs(
            pv_kw=point.pv_kw,
            electric_load_kw=point.electric_load_kw,
            price_eur_per_kwh=point.price_eur_per_kwh,
            outdoor_temperature_c=point.outdoor_temperature_c,
            irradiance_w_per_m2=point.irradiance_w_per_m2,
        )
        step = advance_hub(state, zero_control, exogenous, run.hub_configuration)
        records.append(
            OperatingRecord(
                operating_step=operating_step,
                timestamp_utc=point.timestamp_utc,
                start_state=state,
                control=zero_control,
                exogenous=exogenous,
                reached_state=step.successor,
                flows=step.flows,
            )
        )
        state = step.successor
    return replace(
        run,
        scenario=scenario,
        initial_state=records[0].start_state,
        records=tuple(records),
        controller_diagnostics=(
            tuple(
                replace(
                    diagnostics,
                    terminal_heat_value_eur_per_kwhth=0.0,
                )
                for diagnostics in run.controller_diagnostics
            )
            if run.controller_name == "mpc"
            else run.controller_diagnostics
        ),
        terminal_state=state,
    )


def _committed_executable_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "executable-repository"
    repository.mkdir(parents=True)
    (repository / "runner.py").write_text(
        "def run():\n    return 'committed'\n",
        encoding="utf-8",
        newline="\n",
    )
    (repository / "data").mkdir()
    (repository / "data" / "test-source.csv").write_bytes(TEST_SOURCE_BYTES)
    (repository / "data" / "test-source.provenance.json").write_bytes(
        TEST_SIDECAR_BYTES
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task 7 Test",
            "-c",
            "user.email=task7@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    return repository


def _rehash_bundle(bundle_path: Path) -> Path:
    """Rehash an adversarially edited bundle without fixing its semantics."""
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, sha256_bytes

    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for member in (
        "trajectory.csv",
        "controller_diagnostics.csv",
        "summary.json",
        "validation.json",
    ):
        manifest["member_hashes"][member] = sha256_bytes(
            (bundle_path / member).read_bytes()
        )
    identity_manifest = dict(manifest)
    identity_manifest.pop("run_bundle_identifier", None)
    identifier = sha256_bytes(canonical_json_bytes(identity_manifest))
    manifest["run_bundle_identifier"] = identifier
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    renamed = bundle_path.with_name(
        bundle_path.name.rsplit("--", 1)[0] + f"--{identifier}"
    )
    bundle_path.rename(renamed)
    return renamed


def _rehash_specification_and_bundle(bundle_path: Path) -> Path:
    """Recompute both public identities after an internally consistent forgery."""
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, run_specification_identifier

    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specification_content = {
        "schema_version": "run-specification-v1",
        "canonicalization_version": "canonical-json-v1",
        "scenario": manifest["scenario"],
        "controller": manifest["controller"],
        "asset_capabilities": manifest["asset_capabilities"],
        "hub_configuration": {
            "capabilities": manifest["asset_capabilities"],
        },
        "evaluation_policy": manifest["evaluation_policy"],
        "code_provenance": manifest["code_provenance"],
        "runtime": manifest["runtime"],
        "input_hashes": manifest["input_hashes"],
    }
    manifest["run_specification_identifier"] = run_specification_identifier(
        specification_content
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return _rehash_bundle(bundle_path)


def _rewrite_csv(path: Path, mutate) -> None:
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    fieldnames = list(rows[0])
    mutate(fieldnames, rows)
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(target.getvalue(), encoding="utf-8", newline="")


def _created_test_bundle(tmp_path: Path, *, controller_name: str = "baseline"):
    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run(
        controller_name=controller_name,
        capability_policy=(
            BASELINE_CAPABILITY_POLICY
            if controller_name == "baseline"
            else {"battery": True, "hydrogen": True, "thermal_store": True}
        ),
    )
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    bundle = create_run_bundle(
        run,
        report,
        specification,
        tmp_path / "results" / "runs",
        repository_root=repository,
    )
    return repository, run, report, specification, bundle


def test_evaluation_stable_interfaces_have_exact_fields():
    from greenhouse_energy_hub.evaluation import (
        CostSummary,
        EvaluationPolicy,
        EvaluationReport,
        StepLineItems,
        WearCoefficients,
    )

    assert [field.name for field in fields(WearCoefficients)] == [
        "battery_eur_per_kwh",
        "thermal_store_eur_per_kwh",
        "electrolyser_eur_per_kwh",
        "fuel_cell_eur_per_kwh",
    ]
    assert [field.name for field in fields(EvaluationPolicy)] == [
        "name",
        "version",
        "grid_import_fee_eur_per_kwh",
        "wear",
        "sensitivity_multipliers",
    ]
    assert [field.name for field in fields(StepLineItems)] == [
        "operating_step",
        "grid_cost_eur",
        "battery_wear_eur",
        "thermal_store_wear_eur",
        "electrolyser_wear_eur",
        "fuel_cell_wear_eur",
        "operating_cost_eur",
        "comfort_violation_c_h",
    ]
    assert [field.name for field in fields(CostSummary)] == [
        "grid_cost_eur",
        "battery_wear_eur",
        "thermal_store_wear_eur",
        "electrolyser_wear_eur",
        "fuel_cell_wear_eur",
        "operating_cost_eur",
        "inventory_settlement_eur",
        "inventory_adjusted_cost_eur",
        "comfort_violation_c_h",
    ]
    assert [field.name for field in fields(EvaluationReport)] == [
        "policy",
        "nominal",
        "wear_sensitivities",
        "step_line_items",
    ]


def test_two_step_grid_and_each_wear_term_use_exact_policy_formulas():
    from greenhouse_energy_hub.evaluation import evaluate_run

    first, second = evaluate_run(
        _two_step_valid_run(), _policy()
    ).step_line_items

    assert first.grid_cost_eur == pytest.approx(12.5)
    assert first.battery_wear_eur == pytest.approx(0.06)
    assert first.thermal_store_wear_eur == pytest.approx(0.012)
    assert first.electrolyser_wear_eur == pytest.approx(0.06)
    assert first.fuel_cell_wear_eur == pytest.approx(0.01)
    assert second.grid_cost_eur == pytest.approx(-10.0)
    assert second.battery_wear_eur == pytest.approx(0.04)
    assert second.thermal_store_wear_eur == pytest.approx(0.003)
    assert second.electrolyser_wear_eur == pytest.approx(0.0)
    assert second.fuel_cell_wear_eur == pytest.approx(0.02)


def test_two_step_operating_cost_is_grid_plus_four_wear_terms():
    from greenhouse_energy_hub.evaluation import evaluate_run

    report = evaluate_run(_two_step_valid_run(), _policy())
    first, second = report.step_line_items

    assert first.operating_cost_eur == pytest.approx(12.642)
    assert second.operating_cost_eur == pytest.approx(-9.937)
    assert report.nominal.grid_cost_eur == pytest.approx(2.5)
    assert report.nominal.battery_wear_eur == pytest.approx(0.10)
    assert report.nominal.thermal_store_wear_eur == pytest.approx(0.015)
    assert report.nominal.electrolyser_wear_eur == pytest.approx(0.06)
    assert report.nominal.fuel_cell_wear_eur == pytest.approx(0.03)
    assert report.nominal.operating_cost_eur == pytest.approx(2.705)


def test_two_step_inventory_uses_mean_operating_price_and_terminal_state():
    from greenhouse_energy_hub.evaluation import evaluate_run, recoverable_inventory_kwh

    run = _two_step_valid_run()
    report = evaluate_run(run, _policy())
    initial_inventory = 100.0 * 0.96 + 10.0 * 0.50 * 33.33 + 100.0 / 3.5
    terminal_inventory = 90.0 * 0.96 + 9.0 * 0.50 * 33.33 + 100.0 / 3.5
    inventory_settlement = 0.15 * (initial_inventory - terminal_inventory)

    assert recoverable_inventory_kwh(
        run.initial_state, run.hub_configuration
    ) == pytest.approx(initial_inventory)
    assert recoverable_inventory_kwh(
        run.terminal_state, run.hub_configuration
    ) == pytest.approx(terminal_inventory)
    assert report.nominal.inventory_settlement_eur == pytest.approx(
        inventory_settlement
    )
    assert report.nominal.inventory_adjusted_cost_eur == pytest.approx(
        2.705 + inventory_settlement
    )


def test_two_step_comfort_uses_reached_state_and_is_never_monetized():
    from greenhouse_energy_hub.evaluation import evaluate_run

    report = evaluate_run(_two_step_valid_run(), _policy())
    first, second = report.step_line_items

    assert first.comfort_violation_c_h == pytest.approx(1.5)
    assert second.comfort_violation_c_h == pytest.approx(1.0)
    assert report.nominal.comfort_violation_c_h == pytest.approx(2.5)
    serialized_summary = asdict(report.nominal)
    assert "comfort_cost_eur" not in serialized_summary
    assert "effective_cost_eur" not in serialized_summary


def test_wear_sensitivities_reuse_records_at_zero_one_and_two_times():
    from greenhouse_energy_hub.evaluation import evaluate_run

    run = _two_step_valid_run()
    records_before = run.records
    report = evaluate_run(run, _policy())
    zero = report.wear_sensitivities["0x"]
    one = report.wear_sensitivities["1x"]
    two = report.wear_sensitivities["2x"]
    inventory_settlement = report.nominal.inventory_settlement_eur

    assert run.records is records_before
    assert zero.operating_cost_eur == pytest.approx(2.5)
    assert zero.inventory_adjusted_cost_eur == pytest.approx(
        2.5 + inventory_settlement
    )
    assert one == report.nominal
    assert two.operating_cost_eur == pytest.approx(2.91)
    assert two.inventory_adjusted_cost_eur == pytest.approx(
        2.91 + inventory_settlement
    )
    with pytest.raises(TypeError):
        report.wear_sensitivities["3x"] = two


@pytest.mark.parametrize(
    "multipliers",
    [
        (),
        (1.0,),
        (0.0, 1.0),
        (1.0, 2.0),
        (0.0, 2.0),
    ],
)
def test_evaluation_policy_requires_zero_nominal_and_double_wear_evidence(
    multipliers,
):
    from greenhouse_energy_hub.evaluation import EvaluationPolicy

    with pytest.raises(ValueError, match="0x, 1x, and 2x"):
        EvaluationPolicy(sensitivity_multipliers=multipliers)


def test_evaluation_policy_allows_additional_unique_nonnegative_sensitivities():
    from greenhouse_energy_hub.evaluation import EvaluationPolicy, evaluate_run

    policy = EvaluationPolicy(
        sensitivity_multipliers=(0.0, 0.5, 1.0, 2.0, 3.0)
    )

    report = evaluate_run(_two_step_valid_run(), policy)

    assert tuple(report.wear_sensitivities) == (
        "0x",
        "0.5x",
        "1x",
        "2x",
        "3x",
    )


def test_serialized_policy_metadata_labels_provisional_coefficients_and_is_immutable():
    from types import MappingProxyType

    policy = _policy()
    metadata = policy.to_metadata()
    assert isinstance(metadata, MappingProxyType)
    assert isinstance(metadata["wear"], MappingProxyType)
    assert isinstance(metadata["sensitivity_multipliers"], tuple)
    with pytest.raises(TypeError):
        dict.__setitem__(metadata, "name", "bypassed-freeze")
    with pytest.raises(TypeError):
        dict.__setitem__(
            metadata["wear"],
            "battery_eur_per_kwh",
            999.0,
        )
    with pytest.raises(TypeError):
        json.dumps(metadata, sort_keys=True, allow_nan=False)

    normalized = policy.to_serializable_metadata()

    canonical_json = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert '"coefficient_status":"provisional"' in canonical_json
    assert json.loads(canonical_json) == {
        "comfort_valuation": None,
        "grid_import_fee_eur_per_kwh": 0.025,
        "name": "greenhouse-hub-evaluation",
        "sensitivity_multipliers": [0.0, 1.0, 2.0],
        "settlement_rule": "arithmetic-mean-operating-wholesale-price",
        "version": "1",
        "wear": {
            "battery_eur_per_kwh": 0.005,
            "coefficient_status": "provisional",
            "electrolyser_eur_per_kwh": 0.002,
            "fuel_cell_eur_per_kwh": 0.002,
            "thermal_store_eur_per_kwh": 0.0005,
        },
    }
    assert isinstance(normalized, dict)
    assert isinstance(normalized["wear"], dict)
    assert isinstance(normalized["sensitivity_multipliers"], list)
    with pytest.raises(TypeError):
        metadata["name"] = "changed"
    with pytest.raises(TypeError):
        metadata["wear"]["battery_eur_per_kwh"] = 999.0

    normalized["wear"]["battery_eur_per_kwh"] = 999.0
    assert (
        policy.to_serializable_metadata()["wear"]["battery_eur_per_kwh"]
        == 0.005
    )


@pytest.mark.parametrize("capability", ["battery", "hydrogen", "thermal_store"])
def test_disabled_asset_recoverable_inventory_is_exactly_zero(capability):
    from greenhouse_energy_hub.evaluation import recoverable_inventory_kwh
    from greenhouse_energy_hub.hub import AssetCapabilities, HubConfiguration, HubState

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )
    inventory = {
        "battery": HubState(123.0, 0.0, 0.0, 19.0),
        "hydrogen": HubState(0.0, 123.0, 0.0, 19.0),
        "thermal_store": HubState(0.0, 0.0, 123.0, 19.0),
    }[capability]

    assert recoverable_inventory_kwh(inventory, config) == 0.0


def test_evaluate_run_rejects_invalid_incomplete_and_wrong_record_count():
    from greenhouse_energy_hub.evaluation import evaluate_run
    from greenhouse_energy_hub.simulation import InvalidRun, ValidationReport

    run = _two_step_valid_run()
    invalid = InvalidRun(
        scenario=run.scenario,
        controller_name=run.controller_name,
        controller_configuration=run.controller_configuration,
        capability_policy=run.capability_policy,
        hub_configuration=run.hub_configuration,
        failed_step=1,
        failure_code="solver_failure",
        message="forced",
        partial_records=run.records[:1],
        controller_diagnostics=run.controller_diagnostics,
    )
    with pytest.raises(TypeError, match="ValidRun"):
        evaluate_run(invalid, _policy())

    incomplete = replace(
        run,
        validation=ValidationReport(
            complete=False,
            valid=False,
            checked_operating_steps=1,
            issues=(),
        ),
    )
    with pytest.raises(ValueError, match="complete and valid"):
        evaluate_run(incomplete, _policy())

    wrong_count = replace(run, records=run.records[:1])
    with pytest.raises(AssertionError, match="Operating Step count"):
        evaluate_run(wrong_count, _policy())


def test_evaluate_run_recomputes_from_records_without_serialized_cost_columns(
    monkeypatch,
):
    from greenhouse_energy_hub.evaluation import evaluate_run
    from greenhouse_energy_hub.simulation import ValidRun

    def serialized_columns_must_not_be_read(_run):
        raise AssertionError("evaluate_run trusted serialized/precomputed columns")

    monkeypatch.setattr(ValidRun, "to_frame", serialized_columns_must_not_be_read)

    report = evaluate_run(_two_step_valid_run(), _policy())

    assert report.nominal.grid_cost_eur == pytest.approx(2.5)
    assert report.nominal.operating_cost_eur == pytest.approx(2.705)


def test_baseline_and_mpc_use_same_policy_without_changing_baseline_capabilities():
    from greenhouse_energy_hub.evaluation import evaluate_run

    policy = _policy()
    baseline = _two_step_valid_run(
        controller_name="baseline",
        capability_policy=BASELINE_CAPABILITY_POLICY,
    )
    mpc = _two_step_valid_run(
        controller_name="mpc",
        capability_policy={
            "battery": True,
            "hydrogen": True,
            "thermal_store": True,
        },
    )

    baseline_report = evaluate_run(baseline, policy)
    mpc_report = evaluate_run(mpc, policy)

    assert baseline_report.policy is policy
    assert mpc_report.policy is policy
    assert baseline.capability_policy == BASELINE_CAPABILITY_POLICY
    assert baseline_report.nominal == mpc_report.nominal


def test_solver_only_configuration_changes_do_not_change_evaluation():
    from greenhouse_energy_hub.evaluation import evaluate_run
    from greenhouse_energy_hub.controllers.mpc import MpcConfiguration

    policy = _policy()
    first_config = MpcConfiguration.from_evaluation_policy(policy)
    second_config = MpcConfiguration.from_evaluation_policy(
        policy,
        terminal_weight=9.0,
        complementarity_weight=0.75,
        comfort_slack_weight=123.0,
        input_move_weight=0.2,
    )
    first_run = _two_step_valid_run(
        controller_name="mpc",
        controller_configuration=first_config.to_controller_metadata(),
    )
    second_run = _two_step_valid_run(
        controller_name="mpc",
        controller_configuration=second_config.to_controller_metadata(),
    )

    assert (
        first_run.controller_configuration["solver_diagnostics"]
        != second_run.controller_configuration["solver_diagnostics"]
    )
    assert evaluate_run(first_run, policy) == evaluate_run(second_run, policy)


def test_ablation_variants_use_hub_configuration_and_publish_separate_scorecard():
    from greenhouse_energy_hub.evaluation import evaluate_run
    from experiments import ablations

    assert not hasattr(ablations, "effective_cost")
    assert not hasattr(ablations, "COMFORT_PENALTY_EUR_PER_CH")
    assert ablations.VARIANTS == {
        "full": {"horizon_steps": 24},
        "no-h2": {"horizon_steps": 24, "hydrogen": False},
        "no-tes": {"horizon_steps": 24, "thermal_store": False},
        "one-step": {"horizon_steps": 1},
    }

    policy = _policy()
    report = evaluate_run(_two_step_valid_run(), policy)
    row = ablations._scorecard_row("test", report, report)

    assert row == {
        "Variant": "test",
        "Grid Cost [EUR]": pytest.approx(report.nominal.grid_cost_eur),
        "Operating Cost [EUR]": pytest.approx(
            report.nominal.operating_cost_eur
        ),
        "Inventory-Adjusted Cost [EUR]": pytest.approx(
            report.nominal.inventory_adjusted_cost_eur
        ),
        "Comfort Violation [C.h]": pytest.approx(
            report.nominal.comfort_violation_c_h
        ),
        "Wear 0x Inventory-Adjusted Cost [EUR]": pytest.approx(
            report.wear_sensitivities["0x"].inventory_adjusted_cost_eur
        ),
        "Wear 1x Inventory-Adjusted Cost [EUR]": pytest.approx(
            report.wear_sensitivities["1x"].inventory_adjusted_cost_eur
        ),
        "Wear 2x Inventory-Adjusted Cost [EUR]": pytest.approx(
            report.wear_sensitivities["2x"].inventory_adjusted_cost_eur
        ),
        "Inventory-Adjusted Saving vs Baseline [%]": pytest.approx(0.0),
    }
    assert not any(
        "effective" in column.casefold() or "comfort cost" in column.casefold()
        for column in row
    )


def test_run_identity_interfaces_and_canonical_json_are_exact_and_finite():
    from greenhouse_energy_hub.evaluation import (
        RunBundle,
        RunSpecification,
        canonical_json_bytes,
        sha256_bytes,
    )

    assert [field.name for field in fields(RunSpecification)] == [
        "identifier",
        "canonical_content",
        "publication_eligible",
    ]
    assert [field.name for field in fields(RunBundle)] == [
        "identifier",
        "specification_identifier",
        "path",
        "manifest",
    ]
    left = {"z": ["é", 1.0], "a": {"b": True}}
    right = {"a": {"b": True}, "z": ["é", 1.0]}
    expected = b'{"a":{"b":true},"z":["\xc3\xa9",1.0]}'

    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected
    assert re.fullmatch(r"[0-9a-f]{64}", sha256_bytes(expected))
    with pytest.raises(ValueError):
        canonical_json_bytes({"not_finite": float("nan")})


def test_run_specification_hashes_all_inputs_and_changes_with_every_identity_axis(
    tmp_path,
    monkeypatch,
):
    import greenhouse_energy_hub.evaluation as accounting
    from greenhouse_energy_hub.evaluation import build_run_specification
    from greenhouse_energy_hub.scenarios import ScenarioPoint

    repository = _committed_executable_repository(tmp_path)
    run = _publication_ready_run(
        controller_configuration={"horizon_steps": 0},
        capability_policy=BASELINE_CAPABILITY_POLICY,
    )
    base = build_run_specification(
        run,
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    source = run.scenario.provenance[0]
    changed_source = replace(source, sha256="3" * 64)
    source_specification = build_run_specification(
        replace(run, scenario=replace(run.scenario, provenance=(changed_source,))),
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    first_point = run.scenario.points[0]
    changed_point = ScenarioPoint(
        first_point.timestamp_utc,
        first_point.price_eur_per_kwh + 0.001,
        first_point.pv_kw,
        first_point.electric_load_kw,
        first_point.outdoor_temperature_c,
        first_point.irradiance_w_per_m2,
    )
    input_specification = build_run_specification(
        replace(
            run,
            scenario=replace(
                run.scenario,
                points=(changed_point, *run.scenario.points[1:]),
            ),
        ),
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    configuration_specification = build_run_specification(
        replace(run, controller_configuration={"horizon_steps": 1}),
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    changed_runtime = dict(base.runtime)
    changed_runtime["platform"] = f"{changed_runtime['platform']}-changed"
    monkeypatch.setattr(
        accounting,
        "_actual_runtime_manifest",
        lambda: changed_runtime,
    )
    runtime_specification = build_run_specification(
        run,
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    monkeypatch.undo()
    (repository / "runner.py").write_text(
        "def run():\n    return 'dirty-working-bytes'\n",
        encoding="utf-8",
        newline="\n",
    )
    code_specification = build_run_specification(
        run,
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )

    specifications = (
        base,
        source_specification,
        input_specification,
        configuration_specification,
        runtime_specification,
        code_specification,
    )
    assert len({item.identifier for item in specifications}) == len(specifications)
    assert all(re.fullmatch(r"[0-9a-f]{64}", item.identifier) for item in specifications)
    assert base.publication_eligible is True
    assert code_specification.publication_eligible is False
    assert base.input_hashes == {
        "scenario": base.input_hashes["scenario"],
        "sources": {"data/test-source.csv": TEST_SOURCE_SHA256},
        "sidecars": {
            "data/test-source.provenance.json": TEST_SIDECAR_SHA256
        },
    }
    assert re.fullmatch(r"[0-9a-f]{64}", base.input_hashes["scenario"])
    assert set(base.runtime) == {
        "python",
        "platform",
        "do_mpc",
        "casadi",
        "numpy",
        "pandas",
    }
    assert base.code_provenance["executable_path_hashes"] == {
        "runner.py": base.code_provenance["executable_path_hashes"]["runner.py"]
    }
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        base.code_provenance["executable_source_tree_sha256"],
    )


def test_run_specification_runtime_is_actual_and_not_publicly_overrideable(tmp_path):
    import inspect

    from greenhouse_energy_hub.evaluation import build_run_specification

    repository = _committed_executable_repository(tmp_path)
    assert "runtime" not in inspect.signature(build_run_specification).parameters
    with pytest.raises(TypeError, match="runtime"):
        build_run_specification(
            _publication_ready_run(),
            _policy(),
            executable_paths=("runner.py",),
            repository_root=repository,
            runtime={"python": "forged"},
        )


def test_code_provenance_scopes_dirty_checks_to_explicit_executables(tmp_path):
    from greenhouse_energy_hub.evaluation import collect_code_provenance

    repository = _committed_executable_repository(tmp_path)
    (repository / "REVIEW_RESPONSE.md").write_text("unrelated\n", encoding="utf-8")
    clean = collect_code_provenance(
        ("runner.py",), repository_root=repository
    )
    (repository / "generated.py").write_text("untracked executable\n", encoding="utf-8")
    untracked = collect_code_provenance(
        ("runner.py", "generated.py"), repository_root=repository
    )

    assert clean["publication_eligible"] is True
    assert clean["dirty_executable_paths"] == []
    assert clean["untracked_executable_paths"] == []
    assert "REVIEW_RESPONSE.md" not in clean["executable_path_hashes"]
    assert untracked["publication_eligible"] is False
    assert untracked["untracked_executable_paths"] == ["generated.py"]


@pytest.mark.parametrize("symlink_component", [False, True])
def test_code_provenance_rejects_symlinked_executable_selectors(
    tmp_path,
    symlink_component,
):
    from greenhouse_energy_hub.evaluation import collect_code_provenance

    repository = _committed_executable_repository(tmp_path)
    if symlink_component:
        (repository / "real").mkdir()
        shutil.copy2(repository / "runner.py", repository / "real" / "runner.py")
        (repository / "linked").symlink_to("real", target_is_directory=True)
        selector = "linked/runner.py"
    else:
        (repository / "runner-link.py").symlink_to("runner.py")
        selector = "runner-link.py"

    with pytest.raises(ValueError, match="symlink"):
        collect_code_provenance((selector,), repository_root=repository)


def test_run_specification_rejects_nonfinite_missing_provenance_and_bad_mpc_evidence(
    tmp_path,
):
    from greenhouse_energy_hub.evaluation import build_run_specification

    repository = _committed_executable_repository(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        build_run_specification(
            _two_step_valid_run(),
            _policy(),
            executable_paths=("runner.py",),
            repository_root=repository,
        )

    finite_run = _publication_ready_run()
    with pytest.raises((TypeError, ValueError), match="finite|JSON"):
        build_run_specification(
            replace(finite_run, controller_configuration={"weight": float("inf")}),
            _policy(),
            executable_paths=("runner.py",),
            repository_root=repository,
        )

    mpc_run = _publication_ready_run(controller_name="mpc")
    bad_diagnostics = replace(
        mpc_run.controller_diagnostics[0], solver_return_status=None
    )
    with pytest.raises(ValueError, match="solver.*evidence|return status"):
        build_run_specification(
            replace(
                mpc_run,
                controller_diagnostics=(
                    bad_diagnostics,
                    *mpc_run.controller_diagnostics[1:],
                ),
            ),
            _policy(),
            executable_paths=("runner.py",),
            repository_root=repository,
        )


def test_valid_run_serialization_is_fixed_finite_utc_and_one_row_per_step():
    from greenhouse_energy_hub.evaluation import evaluate_run, serialize_valid_run

    run = _publication_ready_run(capability_policy=BASELINE_CAPABILITY_POLICY)
    report = evaluate_run(run, _policy())
    members = serialize_valid_run(run, report)

    assert tuple(members) == (
        "trajectory.csv",
        "controller_diagnostics.csv",
        "summary.json",
        "validation.json",
    )
    assert all(data.endswith(b"\n") for name, data in members.items() if name.endswith(".csv"))
    assert b"\r\n" not in members["trajectory.csv"]
    trajectory_rows = list(
        csv.DictReader(io.StringIO(members["trajectory.csv"].decode("utf-8")))
    )
    diagnostics_rows = list(
        csv.DictReader(
            io.StringIO(members["controller_diagnostics.csv"].decode("utf-8"))
        )
    )
    assert list(trajectory_rows[0]) == [
        "operating_step",
        "timestamp_utc",
        "start_soc_battery_kwh",
        "start_soc_hydrogen_kg",
        "start_soc_thermal_kwh",
        "start_indoor_temperature_c",
        "battery_charge_kw",
        "battery_discharge_kw",
        "electrolyser_kw",
        "fuel_cell_kw",
        "heat_pump_kw",
        "electric_boiler_kw",
        "thermal_charge_kw",
        "thermal_discharge_kw",
        "ventilation_fraction",
        "pv_kw",
        "electric_load_kw",
        "price_eur_per_kwh",
        "outdoor_temperature_c",
        "irradiance_w_per_m2",
        "reached_soc_battery_kwh",
        "reached_soc_hydrogen_kg",
        "reached_soc_thermal_kwh",
        "reached_indoor_temperature_c",
        "grid_kw",
        "generated_heat_kw",
        "heat_to_air_kw",
        "thermal_charge_margin_kw",
        "hydrogen_production_kg_per_h",
        "hydrogen_consumption_kg_per_h",
    ]
    assert list(diagnostics_rows[0]) == [
        "operating_step",
        "adapter",
        "decision_status",
        "solver_success",
        "solver_return_status",
        "solver_iterations",
        "solver_wall_seconds",
        "forecast_start_utc",
        "forecast_end_utc",
        "terminal_electric_value_eur_per_kwh",
        "terminal_heat_value_eur_per_kwhth",
    ]
    assert len(trajectory_rows) == len(diagnostics_rows) == run.scenario.operating_step_count
    assert all(row["timestamp_utc"].endswith("Z") for row in trajectory_rows)
    summary = json.loads(members["summary.json"])
    validation = json.loads(members["validation.json"])
    assert set(summary) == {
        "schema_version",
        "policy",
        "nominal",
        "wear_sensitivities",
        "step_line_items",
    }
    assert summary["schema_version"] == "evaluation-report-v1"
    assert summary["policy"] == report.policy.to_serializable_metadata()
    assert summary["nominal"] == asdict(report.nominal)
    assert len(summary["step_line_items"]) == run.scenario.operating_step_count
    assert set(validation) == {
        "schema_version",
        "complete",
        "valid",
        "checked_operating_steps",
        "issues",
        "steps",
    }
    assert validation["schema_version"] == "run-validation-v1"
    assert validation["complete"] is validation["valid"] is True
    assert validation["checked_operating_steps"] == run.scenario.operating_step_count
    assert len(validation["steps"]) == run.scenario.operating_step_count
    assert all(step["physical_invariants_valid"] for step in validation["steps"])


@pytest.mark.parametrize("member", ["trajectory.csv", "controller_diagnostics.csv"])
def test_bundle_verification_requires_exact_ordered_csv_headers(tmp_path, member):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)

    def truncate(fieldnames, rows):
        removed = fieldnames.pop()
        for row in rows:
            row.pop(removed)

    _rewrite_csv(bundle.path / member, truncate)
    mutated = _rehash_bundle(bundle.path)
    with pytest.raises(ValueError, match="schema|header|columns"):
        verify_run_bundle(mutated, repository_root=repository)


@pytest.mark.parametrize(
    ("member", "mutation"),
    [
        (
            "summary.json",
            lambda value: value.update({"nominal": {"operating_cost_eur": 0.0}}),
        ),
        (
            "validation.json",
            lambda value: value["steps"][0].update({"control_valid": False}),
        ),
        (
            "validation.json",
            lambda value: value["steps"][0].update({"flows_valid": "true"}),
        ),
    ],
)
def test_bundle_verification_rejects_internally_rehashed_semantic_json_mutations(
    tmp_path,
    member,
    mutation,
):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    path = bundle.path / member
    content = json.loads(path.read_text(encoding="utf-8"))
    mutation(content)
    path.write_bytes(canonical_json_bytes(content))
    mutated = _rehash_bundle(bundle.path)

    with pytest.raises(ValueError, match="summary|validation|evidence|schema"):
        verify_run_bundle(mutated, repository_root=repository)


@pytest.mark.parametrize(
    ("controller_name", "column", "value"),
    [
        ("baseline", "adapter", "mpc"),
        ("baseline", "solver_success", "true"),
        ("baseline", "solver_return_status", "Solve_Succeeded"),
        ("mpc", "solver_success", "false"),
        ("mpc", "decision_status", "failure"),
    ],
)
def test_bundle_verification_binds_controller_and_solver_semantics(
    tmp_path,
    controller_name,
    column,
    value,
):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(
        tmp_path,
        controller_name=controller_name,
    )

    def contradict(_fieldnames, rows):
        rows[0][column] = value

    _rewrite_csv(bundle.path / "controller_diagnostics.csv", contradict)
    mutated = _rehash_bundle(bundle.path)
    with pytest.raises(ValueError, match="controller|diagnostic|solver|status|adapter"):
        verify_run_bundle(mutated, repository_root=repository)


@pytest.mark.parametrize(
    "forgery",
    [
        "recorded-horizon",
        "forecast-end",
        "terminal-electric-value",
        "terminal-heat-value",
    ],
)
def test_bundle_verification_rejects_rehashed_forecast_semantic_forgery(
    tmp_path,
    forgery,
):
    """Hashes and identities cannot make causal diagnostic evidence truthful."""
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    root = Path(__file__).resolve().parent.parent
    source = next(root.glob("results/runs/winter-2023-14d--mpc--1e9ee84c*"))
    copied = tmp_path / source.name
    shutil.copytree(source, copied)

    if forgery == "recorded-horizon":
        manifest_path = copied / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["controller"]["configuration"]["horizon_steps"] = 23
        from greenhouse_energy_hub.evaluation import canonical_json_bytes

        manifest_path.write_bytes(canonical_json_bytes(manifest))
        forged = _rehash_specification_and_bundle(copied)
    else:
        def forge_diagnostics(_fieldnames, rows):
            if forgery == "forecast-end":
                rows[0]["forecast_end_utc"] = "2023-01-02T22:00:00Z"
            elif forgery == "terminal-electric-value":
                rows[0]["terminal_electric_value_eur_per_kwh"] = str(
                    float(rows[0]["terminal_electric_value_eur_per_kwh"]) + 0.01
                )
            else:
                rows[0]["terminal_heat_value_eur_per_kwhth"] = str(
                    float(rows[0]["terminal_heat_value_eur_per_kwhth"]) + 0.01
                )

        _rewrite_csv(copied / "controller_diagnostics.csv", forge_diagnostics)
        forged = _rehash_bundle(copied)

    with pytest.raises(ValueError, match="horizon|forecast|terminal"):
        verify_run_bundle(forged, repository_root=root)


def test_bundle_verification_rejects_rehashed_baseline_terminal_coefficients(tmp_path):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    root = Path(__file__).resolve().parent.parent
    source = next(root.glob("results/runs/winter-2023-14d--baseline--509c81dd*"))
    copied = tmp_path / source.name
    shutil.copytree(source, copied)

    def forge_terminal(_fieldnames, rows):
        rows[0]["terminal_electric_value_eur_per_kwh"] = "0.1"

    _rewrite_csv(copied / "controller_diagnostics.csv", forge_terminal)
    forged = _rehash_bundle(copied)

    with pytest.raises(ValueError, match="Baseline.*terminal|terminal.*Baseline"):
        verify_run_bundle(forged, repository_root=root)


@pytest.mark.parametrize(
    "column",
    [
        "battery_charge_kw",
        "grid_kw",
        "reached_soc_battery_kwh",
        "start_indoor_temperature_c",
        "pv_kw",
    ],
)
def test_bundle_verification_recomputes_physics_and_record_continuity(
    tmp_path,
    column,
):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)

    def change_physics(_fieldnames, rows):
        rows[0][column] = str(float(rows[0][column]) + 1.0)

    _rewrite_csv(bundle.path / "trajectory.csv", change_physics)
    mutated = _rehash_bundle(bundle.path)
    with pytest.raises(ValueError, match="physics|flow|state|control|Scenario|continuity"):
        verify_run_bundle(mutated, repository_root=repository)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(
            {"run_specification_identifier": "f" * 64}
        ),
        lambda manifest: manifest["input_hashes"].update(
            {"scenario": "f" * 64}
        ),
        lambda manifest: manifest["scenario"]["provenance"][0].update(
            {"sha256": "f" * 64}
        ),
        lambda manifest: manifest["input_hashes"]["sources"].update(
            {"data/test-source.csv": "f" * 64}
        ),
        lambda manifest: manifest["code_provenance"].update(
            {"executable_source_tree_sha256": "f" * 64}
        ),
    ],
)
def test_bundle_verification_reconstructs_the_complete_specification_identity_graph(
    tmp_path,
    mutation,
):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    mutated = _rehash_bundle(bundle.path)

    with pytest.raises(ValueError, match="Specification|scenario|provenance|source|digest|input"):
        verify_run_bundle(mutated, repository_root=repository)


@pytest.mark.parametrize(
    "section",
    [
        "scenario",
        "controller",
        "asset_capabilities",
        "evaluation_policy",
        "code_provenance",
        "runtime",
        "input_hashes",
    ],
)
def test_bundle_verification_rejects_unknown_nested_identity_fields(
    tmp_path,
    section,
):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[section]["unknown_field"] = "forged"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    mutated = _rehash_bundle(bundle.path)

    with pytest.raises(ValueError, match="schema|unknown|incomplete|Specification"):
        verify_run_bundle(mutated, repository_root=repository)


def test_bundle_verification_rejects_an_internally_rehashed_forged_runtime_identity(
    tmp_path,
):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["python"] = "forged-runtime"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    mutated = _rehash_bundle(bundle.path)

    with pytest.raises(ValueError, match="runtime|Specification"):
        verify_run_bundle(mutated, repository_root=repository)


def test_bundle_verification_is_runtime_portable_unless_strict_match_is_requested(
    tmp_path,
    monkeypatch,
):
    import greenhouse_energy_hub.evaluation as evaluation
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    different_runtime = dict(evaluation._actual_runtime_manifest())
    different_runtime["python"] = "99.99.99"
    different_runtime["platform"] = "different-verifier-platform"
    monkeypatch.setattr(
        evaluation,
        "_actual_runtime_manifest",
        lambda: different_runtime,
    )

    portable = verify_run_bundle(bundle.path, repository_root=repository)
    assert portable.identifier == bundle.identifier
    with pytest.raises(ValueError, match="strict runtime compatibility"):
        verify_run_bundle(
            bundle.path,
            repository_root=repository,
            require_runtime_match=True,
        )


def test_create_run_bundle_requires_explicit_repository_authority(tmp_path):
    import inspect

    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run()
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    parameter = inspect.signature(create_run_bundle).parameters["repository_root"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="repository_root"):
        create_run_bundle(
            run,
            report,
            specification,
            tmp_path / "results" / "runs",
        )


def test_verify_and_load_require_explicit_repository_authority(tmp_path):
    import inspect

    from greenhouse_energy_hub.evaluation import load_run_bundle, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    for function in (verify_run_bundle, load_run_bundle):
        parameter = inspect.signature(function).parameters["repository_root"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="repository_root"):
        verify_run_bundle(bundle.path)
    with pytest.raises(TypeError, match="repository_root"):
        load_run_bundle(bundle.path.parent, bundle.identifier)


@pytest.mark.parametrize(
    ("member", "mutate"),
    [
        (
            "summary.json",
            lambda value: value["nominal"].update(
                {"battery_wear_eur": False}
            ),
        ),
        (
            "validation.json",
            lambda value: value["steps"][0].update({"control_valid": 1}),
        ),
    ],
)
def test_bundle_verification_is_type_exact_for_recomputed_members(
    tmp_path,
    member,
    mutate,
):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    path = bundle.path / member
    content = json.loads(path.read_text(encoding="utf-8"))
    mutate(content)
    path.write_bytes(canonical_json_bytes(content))
    mutated = _rehash_bundle(bundle.path)

    with pytest.raises(ValueError, match="summary|validation|type|canonical"):
        verify_run_bundle(mutated, repository_root=repository)


def test_authoritative_git_rejects_fully_rehashed_executable_map_forgery(tmp_path):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, sha256_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_map = {"missing/runner.py": "f" * 64}
    manifest["code_provenance"].update(
        {
            "executable_path_hashes": forged_map,
            "committed_executable_path_hashes": forged_map,
            "executable_source_tree_sha256": sha256_bytes(
                canonical_json_bytes(forged_map)
            ),
        }
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    mutated = _rehash_specification_and_bundle(bundle.path)

    with pytest.raises(ValueError, match="Git|revision|executable|missing"):
        verify_run_bundle(mutated, repository_root=repository)


def test_authoritative_git_rejects_fully_rehashed_scenario_input_forgery(tmp_path):
    from greenhouse_energy_hub.evaluation import canonical_json_bytes, sha256_bytes, verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    manifest_path = bundle.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["scenario"]["provenance"][0]
    provenance["sha256"] = "e" * 64
    provenance["acquisition_parameters"]["sidecar_sha256"] = "f" * 64
    manifest["input_hashes"] = {
        "scenario": sha256_bytes(canonical_json_bytes(manifest["scenario"])),
        "sources": {provenance["source_path"]: "e" * 64},
        "sidecars": {
            provenance["acquisition_parameters"]["sidecar_path"]: "f" * 64
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    mutated = _rehash_specification_and_bundle(bundle.path)

    with pytest.raises(ValueError, match="Git|source|sidecar|provenance|input"):
        verify_run_bundle(mutated, repository_root=repository)


def test_historical_bundle_verification_uses_recorded_revision_not_current_head(
    tmp_path,
):
    from greenhouse_energy_hub.evaluation import verify_run_bundle

    repository, _, _, _, bundle = _created_test_bundle(tmp_path)
    (repository / "runner.py").write_text(
        "def run():\n    return 'new-head'\n",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(["git", "add", "runner.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task 7 Test",
            "-c",
            "user.email=task7@example.invalid",
            "commit",
            "-q",
            "-m",
            "new head",
        ],
        cwd=repository,
        check=True,
    )

    verified = verify_run_bundle(bundle.path, repository_root=repository)
    assert verified.identifier == bundle.identifier


@pytest.mark.parametrize(
    "relative_path",
    ["data/test-source.csv", "data/test-source.provenance.json"],
)
def test_creation_revalidates_scenario_source_bytes_after_capture(
    tmp_path,
    relative_path,
):
    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run()
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    (repository / relative_path).write_bytes(b"dirtied after capture\n")

    with pytest.raises(ValueError, match="source|sidecar|input|changed|dirty"):
        create_run_bundle(
            run,
            report,
            specification,
            tmp_path / "results" / "runs",
            repository_root=repository,
        )


def test_bundle_creation_is_atomic_deduplicated_and_collision_safe(tmp_path, monkeypatch):
    import greenhouse_energy_hub.evaluation as accounting
    from greenhouse_energy_hub.evaluation import (
        BundleCollisionError,
        build_run_specification,
        create_run_bundle,
        evaluate_run,
    )

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run(capability_policy=BASELINE_CAPABILITY_POLICY)
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    runs_root = tmp_path / "results" / "runs"
    fsync_calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(accounting.os, "fsync", recording_fsync)
    first = create_run_bundle(
        run, report, specification, runs_root, repository_root=repository
    )
    inode = first.path.stat().st_ino
    second = create_run_bundle(
        run, report, specification, runs_root, repository_root=repository
    )

    assert first == second
    assert second.path.stat().st_ino == inode
    assert first.path.name.endswith(first.identifier)
    assert re.fullmatch(r"[0-9a-f]{64}", first.identifier)
    assert len(fsync_calls) >= 7
    assert not list(runs_root.glob(".tmp-*"))

    (first.path / "summary.json").write_bytes(b"{}")
    with pytest.raises(BundleCollisionError):
        create_run_bundle(
            run, report, specification, runs_root, repository_root=repository
        )


@pytest.mark.parametrize("claimant_kind", ["empty-directory", "file", "broken-symlink"])
def test_atomic_publication_never_clobbers_a_racing_claimant(
    tmp_path,
    monkeypatch,
    claimant_kind,
):
    import greenhouse_energy_hub.evaluation as accounting
    from greenhouse_energy_hub.evaluation import (
        BundleCollisionError,
        build_run_specification,
        create_run_bundle,
        evaluate_run,
    )

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run(capability_policy=BASELINE_CAPABILITY_POLICY)
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    runs_root = tmp_path / "results" / "runs"
    real_publish = getattr(accounting, "_atomic_noreplace_directory", os.replace)
    claimant: dict[str, object] = {}

    def racing_publish(source, destination):
        destination = Path(destination)
        if claimant_kind == "empty-directory":
            destination.mkdir()
        elif claimant_kind == "file":
            destination.write_bytes(b"other owner's exact bytes")
        else:
            destination.symlink_to("missing-other-owner")
        claimant["path"] = destination
        claimant["inode"] = destination.lstat().st_ino
        claimant["bytes"] = (
            destination.read_bytes() if claimant_kind == "file" else None
        )
        claimant["target"] = (
            os.readlink(destination) if claimant_kind == "broken-symlink" else None
        )
        return real_publish(source, destination)

    monkeypatch.setattr(
        accounting,
        "_atomic_noreplace_directory",
        racing_publish,
        raising=False,
    )
    monkeypatch.setattr(accounting.os, "replace", racing_publish)
    with pytest.raises(BundleCollisionError):
        create_run_bundle(
            run,
            report,
            specification,
            runs_root,
            repository_root=repository,
        )

    destination = claimant["path"]
    assert isinstance(destination, Path)
    assert destination.lstat().st_ino == claimant["inode"]
    if claimant_kind == "empty-directory":
        assert destination.is_dir() and not any(destination.iterdir())
    elif claimant_kind == "file":
        assert destination.read_bytes() == claimant["bytes"]
    else:
        assert destination.is_symlink()
        assert os.readlink(destination) == claimant["target"]


@pytest.mark.parametrize("winner_matches", [True, False])
def test_atomic_publication_verifies_a_cooperative_racing_winner(
    tmp_path,
    monkeypatch,
    winner_matches,
):
    import greenhouse_energy_hub.evaluation as accounting
    from greenhouse_energy_hub.evaluation import (
        BundleCollisionError,
        build_run_specification,
        create_run_bundle,
        evaluate_run,
    )

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run(capability_policy=BASELINE_CAPABILITY_POLICY)
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    runs_root = tmp_path / "results" / "runs"
    real_publish = getattr(accounting, "_atomic_noreplace_directory", os.replace)
    winner_inode = []

    def racing_publish(source, destination):
        source = Path(source)
        destination = Path(destination)
        shutil.copytree(source, destination)
        if not winner_matches:
            (destination / "summary.json").write_bytes(b"{}")
        winner_inode.append(destination.stat().st_ino)
        return real_publish(source, destination)

    monkeypatch.setattr(
        accounting,
        "_atomic_noreplace_directory",
        racing_publish,
        raising=False,
    )
    monkeypatch.setattr(accounting.os, "replace", racing_publish)
    if winner_matches:
        bundle = create_run_bundle(
            run,
            report,
            specification,
            runs_root,
            repository_root=repository,
        )
        assert bundle.path.stat().st_ino == winner_inode[0]
    else:
        with pytest.raises(BundleCollisionError):
            create_run_bundle(
                run,
                report,
                specification,
                runs_root,
                repository_root=repository,
            )


def test_publication_revalidates_code_bytes_after_specification_capture(tmp_path):
    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run()
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    (repository / "runner.py").write_text(
        "def run():\n    return 'dirtied-after-capture'\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="dirty|changed|executable"):
        create_run_bundle(
            run,
            report,
            specification,
            tmp_path / "results" / "runs",
            repository_root=repository,
        )


def test_publication_revalidates_actual_runtime_after_specification_capture(
    tmp_path,
    monkeypatch,
):
    import greenhouse_energy_hub.evaluation as accounting
    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run()
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    forged = dict(specification.runtime)
    forged["python"] = "runtime-changed-after-capture"
    monkeypatch.setattr(accounting, "_actual_runtime_manifest", lambda: forged)

    with pytest.raises(ValueError, match="runtime"):
        create_run_bundle(
            run,
            report,
            specification,
            tmp_path / "results" / "runs",
            repository_root=repository,
        )


def test_entrypoints_capture_complete_publication_context_before_execution():
    import inspect

    import greenhouse_energy_hub.simulation as rolling_horizon
    from experiments import ablations, run_scenario

    cli_source = inspect.getsource(run_scenario.main)
    execution_source = inspect.getsource(run_scenario.execute_experiment)
    ablation_source = inspect.getsource(ablations.main)
    marker = "_capture_publication_context"
    assert not hasattr(rolling_horizon, "main")
    assert "execute_experiment(" in cli_source
    assert "execute_experiment(" in ablation_source
    assert marker in execution_source
    assert execution_source.index(marker) < execution_source.index("simulate_run(")
    executable_paths = "\n".join(run_scenario.CORE_EXECUTABLE_PATHS)
    for dependency in (
        "src/greenhouse_energy_hub/evaluation.py",
        "src/greenhouse_energy_hub/scenarios.py",
        "src/greenhouse_energy_hub/hub.py",
        "src/greenhouse_energy_hub/simulation.py",
    ):
        assert dependency in executable_paths
    assert (
        "src/greenhouse_energy_hub/controllers/mpc.py"
        in inspect.getsource(run_scenario.executable_paths_for_controller)
    )
    assert '"experiments/ablations.py"' in ablation_source


def test_one_specification_retains_divergent_valid_outputs(tmp_path):
    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run(controller_name="mpc")
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    changed_diagnostics = (
        replace(run.controller_diagnostics[0], solver_wall_seconds=0.02),
        *run.controller_diagnostics[1:],
    )
    divergent_run = replace(run, controller_diagnostics=changed_diagnostics)
    runs_root = tmp_path / "results" / "runs"

    first = create_run_bundle(
        run, report, specification, runs_root, repository_root=repository
    )
    second = create_run_bundle(
        divergent_run,
        report,
        specification,
        runs_root,
        repository_root=repository,
    )

    assert first.specification_identifier == second.specification_identifier
    assert first.identifier != second.identifier
    assert first.path.exists() and second.path.exists()


def test_interrupted_bundle_write_removes_only_its_owned_temporary_directory(
    tmp_path,
    monkeypatch,
):
    import greenhouse_energy_hub.evaluation as accounting
    from greenhouse_energy_hub.evaluation import build_run_specification, create_run_bundle, evaluate_run

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    run = _publication_ready_run()
    report = evaluate_run(run, _policy())
    specification = build_run_specification(
        run,
        report.policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    runs_root = tmp_path / "results" / "runs"
    runs_root.mkdir(parents=True)
    unrelated = runs_root / ".tmp-unrelated-owner"
    unrelated.mkdir()

    def interrupt_replace(_source, _destination):
        raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(
        accounting,
        "_atomic_noreplace_directory",
        interrupt_replace,
        raising=False,
    )
    monkeypatch.setattr(accounting.os, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt):
        create_run_bundle(
            run,
            report,
            specification,
            runs_root,
            repository_root=repository,
        )

    assert unrelated.is_dir()
    assert sorted(path.name for path in runs_root.glob(".tmp-*")) == [
        ".tmp-unrelated-owner"
    ]
    assert not list(runs_root.glob(f"*--{specification.identifier}"))


def test_invalid_run_writes_separate_non_bundle_diagnostics(tmp_path):
    from greenhouse_energy_hub.evaluation import (
        build_run_specification,
        verify_run_bundle,
        write_failure_diagnostics,
    )
    from greenhouse_energy_hub.simulation import InvalidRun

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    valid = _publication_ready_run()
    invalid = InvalidRun(
        scenario=valid.scenario,
        controller_name=valid.controller_name,
        controller_configuration=valid.controller_configuration,
        capability_policy=valid.capability_policy,
        hub_configuration=valid.hub_configuration,
        failed_step=1,
        failure_code="forced_failure",
        message="diagnostic evidence only",
        partial_records=valid.records[:1],
        controller_diagnostics=valid.controller_diagnostics[:1],
    )
    specification = build_run_specification(
        invalid,
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    diagnostics_root = tmp_path / "results" / "diagnostics"

    path = write_failure_diagnostics(
        invalid,
        specification,
        diagnostics_root,
        policy=_policy(),
    )
    second_path = write_failure_diagnostics(
        invalid,
        specification,
        diagnostics_root,
        policy=_policy(),
    )

    assert path.parent.name == specification.identifier
    assert path.is_relative_to(diagnostics_root)
    assert second_path.parent == path.parent
    assert second_path != path
    assert (path / "failure.json").is_file()
    assert (path / "controller_diagnostics.csv").is_file()
    assert not (path / "manifest.json").exists()
    with pytest.raises(ValueError, match="Run Bundle|manifest"):
        verify_run_bundle(path, repository_root=repository)


@pytest.mark.parametrize(
    "identity_axis",
    [
        "controller_name",
        "controller_configuration",
        "capability_policy",
        "hub_configuration",
        "evaluation_policy",
        "input_hashes",
    ],
)
def test_failure_diagnostics_binds_every_specification_identity_axis(
    tmp_path,
    identity_axis,
):
    from greenhouse_energy_hub.evaluation import (
        RunSpecification,
        build_run_specification,
        run_specification_identifier,
        write_failure_diagnostics,
    )
    from greenhouse_energy_hub.simulation import InvalidRun
    from greenhouse_energy_hub.hub import AssetCapabilities, HubConfiguration

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    valid = _publication_ready_run()
    invalid = InvalidRun(
        scenario=valid.scenario,
        controller_name=valid.controller_name,
        controller_configuration=valid.controller_configuration,
        capability_policy=valid.capability_policy,
        hub_configuration=valid.hub_configuration,
        failed_step=0,
        failure_code="forced_failure",
        message="diagnostic evidence only",
        partial_records=(),
        controller_diagnostics=(),
    )
    policy = _policy()
    mutated = invalid
    specification_policy = policy
    if identity_axis == "controller_name":
        mutated = replace(invalid, controller_name="mpc")
    elif identity_axis == "controller_configuration":
        mutated = replace(invalid, controller_configuration={"horizon_steps": 9})
    elif identity_axis == "capability_policy":
        mutated = replace(invalid, capability_policy={"hydrogen_dispatch": False})
    elif identity_axis == "hub_configuration":
        mutated = replace(
            invalid,
            hub_configuration=HubConfiguration(
                capabilities=AssetCapabilities(hydrogen=False)
            ),
        )
    elif identity_axis == "evaluation_policy":
        specification_policy = replace(
            policy,
            grid_import_fee_eur_per_kwh=0.123,
        )

    specification = build_run_specification(
        mutated,
        specification_policy,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    if identity_axis == "input_hashes":
        content = json.loads(
            json.dumps(
                specification.canonical_content,
                default=lambda value: dict(value),
            )
        )
        content["input_hashes"]["scenario"] = "f" * 64
        specification = RunSpecification(
            identifier=run_specification_identifier(content),
            canonical_content=content,
            publication_eligible=specification.publication_eligible,
        )

    with pytest.raises(ValueError, match="Specification|controller|capabil|Hub|policy|input"):
        write_failure_diagnostics(
            invalid,
            specification,
            tmp_path / "results" / "diagnostics",
            policy=policy,
        )


def test_legacy_entry_point_persists_valid_bundles_and_invalid_diagnostics(tmp_path):
    from greenhouse_energy_hub.evaluation import RunBundle
    from greenhouse_energy_hub.simulation import InvalidRun, _persist_outcome

    repository = _committed_executable_repository(tmp_path / "repo-fixture")
    valid = _publication_ready_run()
    policy = _policy()
    results_root = tmp_path / "results"

    valid_artifact = _persist_outcome(
        valid,
        policy,
        results_root=results_root,
        executable_paths=("runner.py",),
        repository_root=repository,
    )
    invalid = InvalidRun(
        scenario=valid.scenario,
        controller_name=valid.controller_name,
        controller_configuration=valid.controller_configuration,
        capability_policy=valid.capability_policy,
        hub_configuration=valid.hub_configuration,
        failed_step=0,
        failure_code="forced_failure",
        message="failure path",
        partial_records=(),
        controller_diagnostics=(),
    )
    invalid_artifact = _persist_outcome(
        invalid,
        policy,
        results_root=results_root,
        executable_paths=("runner.py",),
        repository_root=repository,
    )

    assert isinstance(valid_artifact, RunBundle)
    assert valid_artifact.path.is_relative_to(results_root / "runs")
    assert invalid_artifact.is_relative_to(results_root / "diagnostics")
    assert not (invalid_artifact / "manifest.json").exists()


def test_legacy_entry_points_no_longer_write_mutable_csv_or_figure_artifacts():
    import inspect

    import greenhouse_energy_hub.simulation as rolling_horizon
    from experiments import ablations, run_scenario

    assert not hasattr(rolling_horizon, "main")
    rolling_source = inspect.getsource(run_scenario.main)
    ablation_source = inspect.getsource(ablations.main)
    assert ".to_csv(" not in rolling_source
    for controller in ("baseline", "mpc"):
        assert f"{controller}_results.csv" not in rolling_source
    assert "summary.csv" not in rolling_source
    assert ".to_csv(" not in ablation_source
    assert "ablations.csv" not in ablation_source
    assert ".savefig(" not in ablation_source
