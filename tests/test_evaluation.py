from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta, timezone
import json

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


def test_serialized_policy_metadata_labels_provisional_coefficients_and_is_immutable():
    policy = _policy()
    metadata = policy.to_metadata()

    assert json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False)) == {
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
    with pytest.raises(TypeError):
        metadata["name"] = "changed"
    with pytest.raises(TypeError):
        metadata["wear"]["battery_eur_per_kwh"] = 999.0


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
