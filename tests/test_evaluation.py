from __future__ import annotations

import csv
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess

import pytest


BASELINE_CAPABILITY_POLICY = {
    "hydrogen_dispatch": False,
    "thermal_store_charging": False,
    "grid_battery_charging": False,
    "battery_discharge_price_threshold_eur_per_kwh": 0.12,
}


def _two_step_valid_run(
    *,
    controller_name: str = "baseline",
    controller_configuration: dict[str, object] | None = None,
    capability_policy: dict[str, object] | None = None,
):
    """An exact, hand-built two-step Run with deliberately varied line items."""
    from control.rolling_horizon import (
        DecisionDiagnostics,
        OperatingRecord,
        ValidRun,
        ValidationReport,
    )
    from models.hub_model import (
        ExogenousInputs,
        HubConfiguration,
        HubControl,
        HubFlows,
        HubState,
    )
    from scenarios import Scenario, ScenarioPoint

    start = datetime(2023, 1, 2, tzinfo=timezone.utc)
    points = (
        ScenarioPoint(start, 0.10, 0.0, 400.0, 5.0, 0.0),
        ScenarioPoint(start + timedelta(hours=1), 0.20, 0.0, 400.0, 5.0, 0.0),
    )
    scenario = Scenario(
        name="two-step-evaluation",
        operating_start=start,
        operating_end=start + timedelta(hours=2),
        forecast_end=start + timedelta(hours=2),
        forecast_horizon_capacity_steps=0,
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
            forecast_end_utc=point.timestamp_utc,
            terminal_electric_value_eur_per_kwh=(
                point.price_eur_per_kwh if controller_name == "mpc" else None
            ),
            terminal_heat_value_eur_per_kwhth=(
                point.price_eur_per_kwh / 3.5 if controller_name == "mpc" else None
            ),
        )
        for point in points
    )
    config = HubConfiguration()
    return ValidRun(
        scenario=scenario,
        controller_name=controller_name,
        controller_configuration=controller_configuration or {},
        capability_policy=capability_policy or {},
        hub_configuration=config,
        initial_state=initial,
        records=records,
        controller_diagnostics=diagnostics,
        terminal_state=terminal,
        validation=ValidationReport(True, True, 2, ()),
    )


def _policy():
    from accounting import EvaluationPolicy, WearCoefficients

    return EvaluationPolicy(
        wear=WearCoefficients(
            battery_eur_per_kwh=0.005,
            thermal_store_eur_per_kwh=0.0005,
            electrolyser_eur_per_kwh=0.002,
            fuel_cell_eur_per_kwh=0.002,
        )
    )


def _publication_ready_run(**run_options):
    """Attach source and sidecar provenance to the compact evaluation fixture."""
    from scenarios import SourceProvenance

    run = _two_step_valid_run(**run_options)
    provenance = SourceProvenance(
        source_name="test-source",
        source_path="data/test-source.csv",
        sha256="1" * 64,
        acquisition_parameters={
            "sidecar_path": "data/test-source.provenance.json",
            "sidecar_sha256": "2" * 64,
            "parameters": {"fixture": "two-step"},
        },
        original_timezone="UTC",
        units={"value": "kW"},
        transformations=("test-fixture",),
    )
    return replace(
        run,
        scenario=replace(run.scenario, provenance=(provenance,)),
    )


def _committed_executable_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "executable-repository"
    repository.mkdir(parents=True)
    (repository / "runner.py").write_text(
        "def run():\n    return 'committed'\n",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
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
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    return repository


def test_evaluation_stable_interfaces_have_exact_fields():
    from accounting import (
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
    from accounting import evaluate_run

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
    from accounting import evaluate_run

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
    from accounting import evaluate_run, recoverable_inventory_kwh

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
    from accounting import evaluate_run

    report = evaluate_run(_two_step_valid_run(), _policy())
    first, second = report.step_line_items

    assert first.comfort_violation_c_h == pytest.approx(1.5)
    assert second.comfort_violation_c_h == pytest.approx(1.0)
    assert report.nominal.comfort_violation_c_h == pytest.approx(2.5)
    serialized_summary = asdict(report.nominal)
    assert "comfort_cost_eur" not in serialized_summary
    assert "effective_cost_eur" not in serialized_summary


def test_wear_sensitivities_reuse_records_at_zero_one_and_two_times():
    from accounting import evaluate_run

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
    from accounting import EvaluationPolicy

    with pytest.raises(ValueError, match="0x, 1x, and 2x"):
        EvaluationPolicy(sensitivity_multipliers=multipliers)


def test_evaluation_policy_allows_additional_unique_nonnegative_sensitivities():
    from accounting import EvaluationPolicy, evaluate_run

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
    from accounting import recoverable_inventory_kwh
    from models.hub_model import AssetCapabilities, HubConfiguration, HubState

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
    from accounting import evaluate_run
    from control.rolling_horizon import InvalidRun, ValidationReport

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
    from accounting import evaluate_run
    from control.rolling_horizon import ValidRun

    def serialized_columns_must_not_be_read(_run):
        raise AssertionError("evaluate_run trusted serialized/precomputed columns")

    monkeypatch.setattr(ValidRun, "to_frame", serialized_columns_must_not_be_read)

    report = evaluate_run(_two_step_valid_run(), _policy())

    assert report.nominal.grid_cost_eur == pytest.approx(2.5)
    assert report.nominal.operating_cost_eur == pytest.approx(2.705)


def test_baseline_and_mpc_use_same_policy_without_changing_baseline_capabilities():
    from accounting import evaluate_run

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
    from accounting import evaluate_run
    from control.mpc_controller import MpcConfiguration

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
    from accounting import evaluate_run
    from experiments import ablations
    from models.hub_model import HubConfiguration

    assert not hasattr(ablations, "effective_cost")
    assert not hasattr(ablations, "COMFORT_PENALTY_EUR_PER_CH")
    for variant in ablations.VARIANTS.values():
        assert isinstance(variant["hub_configuration"], HubConfiguration)
        assert not {
            "disable_h2",
            "disable_tes",
        }.intersection(variant["mpc_options"])

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
    from accounting import (
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
):
    from accounting import build_run_specification
    from scenarios import ScenarioPoint

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
    runtime_specification = build_run_specification(
        run,
        _policy(),
        executable_paths=("runner.py",),
        repository_root=repository,
        runtime=changed_runtime,
    )
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
        "sources": {"data/test-source.csv": "1" * 64},
        "sidecars": {"data/test-source.provenance.json": "2" * 64},
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


def test_code_provenance_scopes_dirty_checks_to_explicit_executables(tmp_path):
    from accounting import collect_code_provenance

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


def test_run_specification_rejects_nonfinite_missing_provenance_and_bad_mpc_evidence(
    tmp_path,
):
    from accounting import build_run_specification

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
    from accounting import evaluate_run, serialize_valid_run

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
    assert summary["policy"] == report.policy.to_serializable_metadata()
    assert summary["nominal"] == asdict(report.nominal)
    assert len(summary["step_line_items"]) == run.scenario.operating_step_count
    assert validation["complete"] is validation["valid"] is True
    assert validation["checked_operating_steps"] == run.scenario.operating_step_count
    assert len(validation["steps"]) == run.scenario.operating_step_count
    assert all(step["physical_invariants_valid"] for step in validation["steps"])


def test_bundle_creation_is_atomic_deduplicated_and_collision_safe(tmp_path, monkeypatch):
    import accounting
    from accounting import (
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
    first = create_run_bundle(run, report, specification, runs_root)
    inode = first.path.stat().st_ino
    second = create_run_bundle(run, report, specification, runs_root)

    assert first == second
    assert second.path.stat().st_ino == inode
    assert first.path.name.endswith(first.identifier)
    assert re.fullmatch(r"[0-9a-f]{64}", first.identifier)
    assert len(fsync_calls) >= 7
    assert not list(runs_root.glob(".tmp-*"))

    (first.path / "summary.json").write_bytes(b"{}")
    with pytest.raises(BundleCollisionError):
        create_run_bundle(run, report, specification, runs_root)


def test_one_specification_retains_divergent_valid_outputs(tmp_path):
    from accounting import build_run_specification, create_run_bundle, evaluate_run

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

    first = create_run_bundle(run, report, specification, runs_root)
    second = create_run_bundle(divergent_run, report, specification, runs_root)

    assert first.specification_identifier == second.specification_identifier
    assert first.identifier != second.identifier
    assert first.path.exists() and second.path.exists()


def test_interrupted_bundle_write_removes_only_its_owned_temporary_directory(
    tmp_path,
    monkeypatch,
):
    import accounting
    from accounting import build_run_specification, create_run_bundle, evaluate_run

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

    monkeypatch.setattr(accounting.os, "replace", interrupt_replace)
    with pytest.raises(KeyboardInterrupt):
        create_run_bundle(run, report, specification, runs_root)

    assert unrelated.is_dir()
    assert sorted(path.name for path in runs_root.glob(".tmp-*")) == [
        ".tmp-unrelated-owner"
    ]
    assert not list(runs_root.glob(f"*--{specification.identifier}"))


def test_invalid_run_writes_separate_non_bundle_diagnostics(tmp_path):
    from accounting import (
        build_run_specification,
        verify_run_bundle,
        write_failure_diagnostics,
    )
    from control.rolling_horizon import InvalidRun

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

    path = write_failure_diagnostics(invalid, specification, diagnostics_root)

    assert path.parent.name == specification.identifier
    assert path.is_relative_to(diagnostics_root)
    assert (path / "failure.json").is_file()
    assert (path / "controller_diagnostics.csv").is_file()
    assert not (path / "manifest.json").exists()
    with pytest.raises(ValueError, match="Run Bundle|manifest"):
        verify_run_bundle(path)


def test_legacy_entry_point_persists_valid_bundles_and_invalid_diagnostics(tmp_path):
    from accounting import RunBundle
    from control.rolling_horizon import InvalidRun, _persist_outcome

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

    from control import rolling_horizon
    from experiments import ablations

    rolling_source = inspect.getsource(rolling_horizon.main)
    ablation_source = inspect.getsource(ablations.main)
    assert ".to_csv(" not in rolling_source
    assert "baseline_results.csv" not in rolling_source
    assert "mpc_results.csv" not in rolling_source
    assert "summary.csv" not in rolling_source
    assert ".to_csv(" not in ablation_source
    assert "ablations.csv" not in ablation_source
    assert ".savefig(" not in ablation_source
