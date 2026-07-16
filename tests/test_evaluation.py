from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.xfail(strict=True, reason="PF-05: reported economics omit policy wear and monetize comfort")
def test_two_step_trajectory_has_one_exact_economic_scorecard():
    from accounting import (
        EvaluationPolicy,
        WearCoefficients,
        evaluate_run,
        recoverable_inventory_kwh,
    )
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
    second_control = HubControl(0.0, 8.0, 0.0, 10.0, 0.0, 0.0, 0.0, 6.0, 0.0)
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
            adapter="baseline",
            decision_status="success",
            solver_success=None,
            solver_return_status=None,
            solver_iterations=None,
            solver_wall_seconds=None,
            forecast_start_utc=point.timestamp_utc,
            forecast_end_utc=point.timestamp_utc,
            terminal_electric_value_eur_per_kwh=None,
            terminal_heat_value_eur_per_kwhth=None,
        )
        for point in points
    )
    config = HubConfiguration()
    run = ValidRun(
        scenario=scenario,
        controller_name="baseline",
        controller_configuration={},
        capability_policy={},
        hub_configuration=config,
        initial_state=initial,
        records=records,
        controller_diagnostics=diagnostics,
        terminal_state=terminal,
        validation=ValidationReport(True, True, 2, ()),
    )
    policy = EvaluationPolicy(
        wear=WearCoefficients(
            battery_eur_per_kwh=0.005,
            thermal_store_eur_per_kwh=0.0005,
            electrolyser_eur_per_kwh=0.002,
            fuel_cell_eur_per_kwh=0.002,
        )
    )

    report = evaluate_run(run, policy)

    first, second = report.step_line_items
    assert first.grid_cost_eur == pytest.approx(12.5)
    assert first.battery_wear_eur == pytest.approx(0.06)
    assert first.thermal_store_wear_eur == pytest.approx(0.012)
    assert first.electrolyser_wear_eur == pytest.approx(0.06)
    assert first.fuel_cell_wear_eur == pytest.approx(0.01)
    assert first.operating_cost_eur == pytest.approx(12.642)
    assert first.comfort_violation_c_h == pytest.approx(1.5)

    assert second.grid_cost_eur == pytest.approx(-10.0)
    assert second.battery_wear_eur == pytest.approx(0.04)
    assert second.thermal_store_wear_eur == pytest.approx(0.003)
    assert second.electrolyser_wear_eur == pytest.approx(0.0)
    assert second.fuel_cell_wear_eur == pytest.approx(0.02)
    assert second.operating_cost_eur == pytest.approx(-9.937)
    assert second.comfort_violation_c_h == pytest.approx(1.0)

    initial_inventory = 100.0 * 0.96 + 10.0 * 0.50 * 33.33 + 100.0 / 3.5
    terminal_inventory = 90.0 * 0.96 + 9.0 * 0.50 * 33.33 + 100.0 / 3.5
    inventory_settlement = 0.15 * (initial_inventory - terminal_inventory)
    assert recoverable_inventory_kwh(initial, config) == pytest.approx(initial_inventory)
    assert recoverable_inventory_kwh(terminal, config) == pytest.approx(terminal_inventory)

    nominal = report.nominal
    assert nominal.grid_cost_eur == pytest.approx(2.5)
    assert nominal.battery_wear_eur == pytest.approx(0.10)
    assert nominal.thermal_store_wear_eur == pytest.approx(0.015)
    assert nominal.electrolyser_wear_eur == pytest.approx(0.06)
    assert nominal.fuel_cell_wear_eur == pytest.approx(0.03)
    assert nominal.operating_cost_eur == pytest.approx(2.705)
    assert nominal.inventory_settlement_eur == pytest.approx(inventory_settlement)
    assert nominal.inventory_adjusted_cost_eur == pytest.approx(2.705 + inventory_settlement)
    assert nominal.comfort_violation_c_h == pytest.approx(2.5)

    zero = report.wear_sensitivities["0x"]
    one = report.wear_sensitivities["1x"]
    two = report.wear_sensitivities["2x"]
    assert zero.operating_cost_eur == pytest.approx(2.5)
    assert zero.inventory_adjusted_cost_eur == pytest.approx(2.5 + inventory_settlement)
    assert one == nominal
    assert two.operating_cost_eur == pytest.approx(2.91)
    assert two.inventory_adjusted_cost_eur == pytest.approx(2.91 + inventory_settlement)

    serialized_summary = asdict(nominal)
    assert "comfort_cost_eur" not in serialized_summary
    assert "effective_cost_eur" not in serialized_summary
