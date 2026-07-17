from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


INPUT_NAMES = (
    "P_bat_ch",
    "P_bat_dis",
    "P_elz",
    "P_fc",
    "P_hp",
    "P_eboiler",
    "Q_tes_ch",
    "Q_tes_dis",
    "vent",
)


def _test_diagnostics(forecast):
    from control.rolling_horizon import DecisionDiagnostics

    return DecisionDiagnostics(
        adapter="test",
        decision_status="success",
        solver_success=None,
        solver_return_status=None,
        solver_iterations=None,
        solver_wall_seconds=None,
        forecast_start_utc=forecast[0].timestamp_utc,
        forecast_end_utc=forecast[-1].timestamp_utc,
        terminal_electric_value_eur_per_kwh=None,
        terminal_heat_value_eur_per_kwhth=None,
    )


def _zero_control():
    from models.hub_model import HubControl

    return HubControl(
        battery_charge_kw=0.0,
        battery_discharge_kw=0.0,
        electrolyser_kw=0.0,
        fuel_cell_kw=0.0,
        heat_pump_kw=0.0,
        electric_boiler_kw=0.0,
        thermal_charge_kw=0.0,
        thermal_discharge_kw=0.0,
        ventilation_fraction=0.0,
    )


@pytest.mark.xfail(strict=True, reason="PF-01: one-step terminal state is unconstrained")
def test_one_step_mpc_keeps_every_reached_state_valid():
    from control.rolling_horizon import load_data, run_simulation
    from models.hub_model import state_bounds

    outcome = run_simulation(
        load_data(start_month=1, n_days=2),
        mode="mpc",
        n_horizon=1,
    )
    bounds = state_bounds()
    frame = outcome if isinstance(outcome, pd.DataFrame) else outcome.to_frame()
    for state, column in {
        "SOC_bat": "SOC_bat_kWh",
        "SOC_h2": "SOC_h2_kg",
        "SOC_tes": "SOC_tes_kWh",
        "T_in": "T_in_C",
    }.items():
        lower, upper = bounds[state]
        assert frame[column].between(lower - 1e-6, upper + 1e-6).all()


class _FailedMpc:
    """Complete double for the do-mpc surface consumed by the legacy loop."""

    def __init__(self) -> None:
        self.x0 = None
        self.solver_stats: dict[str, object] = {}
        self.u0 = {name: np.array([[1.0]]) for name in INPUT_NAMES}
        self.make_step_calls: list[np.ndarray] = []

    def set_initial_guess(self) -> None:
        return None

    def make_step(self, x0: np.ndarray) -> np.ndarray:
        self.make_step_calls.append(np.asarray(x0).copy())
        self.solver_stats = {
            "success": False,
            "return_status": "Infeasible_Problem_Detected",
            "iter_count": 17,
            "t_wall_total": 0.01,
        }
        return np.ones((len(INPUT_NAMES), 1))


def test_solver_failure_returns_invalid_run_without_advancing_plant(monkeypatch, hourly_frame):
    import control.mpc_controller as mpc_controller
    import control.rolling_horizon as rolling_horizon
    import models.hub_model as hub_model

    failed_mpc = _FailedMpc()
    monkeypatch.setattr(mpc_controller, "build_mpc", lambda **_kwargs: (failed_mpc, object()))

    real_hub_dynamics = hub_model.hub_dynamics
    plant_calls = 0

    def counted_hub_dynamics(*args, **kwargs):
        nonlocal plant_calls
        plant_calls += 1
        return real_hub_dynamics(*args, **kwargs)

    # The legacy loop holds a direct import, so instrument both definition and consumer.
    monkeypatch.setattr(hub_model, "hub_dynamics", counted_hub_dynamics)
    monkeypatch.setattr(rolling_horizon, "hub_dynamics", counted_hub_dynamics)

    # Forty-nine points give the default 24-step controller ample initial coverage,
    # so only the injected solver failure may classify this Run as invalid.
    outcome = rolling_horizon.run_simulation(hourly_frame, mode="mpc")

    observed = (
        len(failed_mpc.make_step_calls),
        plant_calls,
        getattr(outcome, "failure_code", None),
    )
    assert observed == (1, 0, "solver_failure")

    # Keep the not-yet-existing outcome type inside the regression body so legacy
    # collection remains possible until the fail-closed simulation interface lands.
    from control.rolling_horizon import InvalidRun

    assert isinstance(outcome, InvalidRun)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("battery_charge_kw", np.nan),
        ("battery_charge_kw", 500.01),
    ],
)
def test_invalid_control_returns_invalid_run_without_applying_control(
    monkeypatch, hourly_frame, field, value
):
    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, InvalidRun

    control = _zero_control()
    control = type(control)(**{**control.__dict__, field: value})

    def decide(_self, _state, forecast):
        return ControlDecision(control=control, diagnostics=_test_diagnostics(forecast))

    monkeypatch.setattr(rolling_horizon.BaselineControllerAdapter, "decide", decide)

    def must_not_advance(*_args, **_kwargs):
        raise AssertionError("invalid controls must not reach the hub")

    monkeypatch.setattr(rolling_horizon, "advance_hub", must_not_advance)

    outcome = rolling_horizon.run_simulation(hourly_frame.iloc[:1], mode="baseline")

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == "invalid_control"
    assert outcome.failed_step == 0
    assert outcome.partial_records == ()


@pytest.mark.parametrize(
    ("invalid_step", "failure_code"),
    [
        (
            "successor",
            "invalid_successor",
        ),
        (
            "flows",
            "invalid_flows",
        ),
    ],
)
def test_invalid_successor_or_flows_never_append_an_operating_record(
    monkeypatch, hourly_frame, invalid_step, failure_code
):
    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, InvalidRun
    from models.hub_model import HubFlows, HubState, HubStep

    def decide(_self, _state, forecast):
        return ControlDecision(
            control=_zero_control(), diagnostics=_test_diagnostics(forecast)
        )

    monkeypatch.setattr(rolling_horizon.BaselineControllerAdapter, "decide", decide)

    valid_state = HubState(
        soc_battery_kwh=500.0,
        soc_hydrogen_kg=60.0,
        soc_thermal_kwh=1600.0,
        indoor_temperature_c=19.0,
    )
    valid_flows = HubFlows(
        grid_kw=0.0,
        generated_heat_kw=0.0,
        heat_to_air_kw=0.0,
        thermal_charge_margin_kw=0.0,
        hydrogen_production_kg_per_h=0.0,
        hydrogen_consumption_kg_per_h=0.0,
    )
    step = HubStep(
        successor=(
            HubState(
                soc_battery_kwh=1000.01,
                soc_hydrogen_kg=60.0,
                soc_thermal_kwh=1600.0,
                indoor_temperature_c=19.0,
            )
            if invalid_step == "successor"
            else valid_state
        ),
        flows=(
            HubFlows(**{**valid_flows.__dict__, "grid_kw": 2000.01})
            if invalid_step == "flows"
            else valid_flows
        ),
    )
    monkeypatch.setattr(rolling_horizon, "advance_hub", lambda *_args, **_kwargs: step)

    outcome = rolling_horizon.run_simulation(hourly_frame.iloc[:1], mode="baseline")

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == failure_code
    assert outcome.failed_step == 0
    assert outcome.partial_records == ()


def test_sub_tolerance_opposing_flow_is_zeroed_only_when_serialized(
    monkeypatch, hourly_frame
):
    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, ValidRun
    from models.hub_model import ETA_BAT_CH, ETA_BAT_DIS

    tiny_discharge_kw = 1e-5
    charge_to_operational_cap_kw = (
        400.0 + tiny_discharge_kw / ETA_BAT_DIS
    ) / ETA_BAT_CH
    control = type(_zero_control())(
        **{
            **_zero_control().__dict__,
            "battery_charge_kw": charge_to_operational_cap_kw,
            "battery_discharge_kw": tiny_discharge_kw,
        }
    )

    def decide(_self, _state, forecast):
        return ControlDecision(control=control, diagnostics=_test_diagnostics(forecast))

    monkeypatch.setattr(rolling_horizon.BaselineControllerAdapter, "decide", decide)
    monkeypatch.setattr(
        rolling_horizon.BaselineControllerAdapter,
        "requires_operational_storage_bounds",
        True,
    )

    outcome = rolling_horizon.run_simulation(
        hourly_frame.iloc[:1], mode="baseline"
    )

    assert isinstance(outcome, ValidRun)
    assert outcome.terminal_state.soc_battery_kwh == pytest.approx(900.0)
    assert outcome.records[0].control.battery_discharge_kw == tiny_discharge_kw
    assert outcome.to_frame().iloc[0].u_P_bat_dis == 0.0


def test_run_configuration_mappings_are_read_only(hourly_frame):
    from control.rolling_horizon import ValidRun, run_simulation

    outcome = run_simulation(hourly_frame.iloc[:1], mode="baseline")

    assert isinstance(outcome, ValidRun)
    with pytest.raises(TypeError):
        outcome.controller_configuration["mutated"] = True
    with pytest.raises(TypeError):
        outcome.capability_policy["hydrogen"] = True


def test_mpc_adapter_configuration_mappings_are_read_only_copies():
    from control.mpc_controller import MpcControllerAdapter

    configuration = {"horizon_steps": 24}
    capability_policy = {"battery": True}
    adapter = MpcControllerAdapter(
        mpc=object(),
        forecast_horizon_steps=24,
        configuration=configuration,
        capability_policy=capability_policy,
    )
    configuration["horizon_steps"] = 1
    capability_policy["battery"] = False

    assert adapter.configuration == {"horizon_steps": 24}
    assert adapter.capability_policy == {"battery": True}
    with pytest.raises(TypeError):
        adapter.configuration["horizon_steps"] = 1
    with pytest.raises(TypeError):
        adapter.capability_policy["battery"] = False


def test_cli_does_not_serialize_invalid_run(monkeypatch, tmp_path, hourly_frame):
    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControllerFailure, DecisionDiagnostics

    def decide(_self, _state, forecast):
        diagnostics = DecisionDiagnostics(
            **{
                **_test_diagnostics(forecast).__dict__,
                "decision_status": "failure",
            }
        )
        return ControllerFailure(
            code="test_failure", message="forced failure", diagnostics=diagnostics
        )

    monkeypatch.setattr(rolling_horizon.BaselineControllerAdapter, "decide", decide)
    monkeypatch.setattr(
        rolling_horizon, "load_data", lambda **_kwargs: hourly_frame.iloc[:1]
    )
    monkeypatch.setattr(rolling_horizon, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        rolling_horizon.sys,
        "argv",
        ["rolling_horizon.py", "--days", "1", "--mode", "baseline"],
    )

    csv_writes = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda _self, path, **_kwargs: csv_writes.append(path),
    )

    rolling_horizon.main()

    assert csv_writes == []


@pytest.mark.xfail(strict=True, reason="PF-02: disabled assets retain non-zero state and capacity")
@pytest.mark.parametrize(
    ("capability", "state_field", "control_fields"),
    [
        ("hydrogen", "soc_hydrogen_kg", ("electrolyser_kw", "fuel_cell_kw")),
        ("thermal_store", "soc_thermal_kwh", ("thermal_charge_kw", "thermal_discharge_kw")),
    ],
)
def test_disabled_asset_is_inert_zero_capacity(capability, state_field, control_fields):
    from models.hub_model import (
        AssetCapabilities,
        HubConfiguration,
        control_bounds,
        initial_state,
        physical_state_bounds,
    )

    capabilities = AssetCapabilities(**{capability: False})
    config = HubConfiguration(capabilities=capabilities)
    state = initial_state(config)

    assert getattr(state, state_field) == 0.0
    assert physical_state_bounds(config)[state_field] == (0.0, 0.0)
    for field in control_fields:
        assert control_bounds(config)[field] == (0.0, 0.0)


def _first_mpc_control(prices: np.ndarray) -> np.ndarray:
    from control.mpc_controller import build_mpc
    from models.hub_model import initial_state

    n = len(prices)
    controller, _ = build_mpc(
        price_forecast=prices,
        pv_forecast=np.zeros(n),
        load_elec_forecast=np.full(n, 400.0),
        temp_out_forecast=np.full(n, 5.0),
        irr_forecast=np.zeros(n),
        n_horizon=24,
    )
    initial = initial_state()
    x0 = np.array(
        [[initial["SOC_bat"]], [initial["SOC_h2"]], [initial["SOC_tes"]], [initial["T_in"]]]
    )
    controller.x0 = x0
    controller.set_initial_guess()
    controller.make_step(x0)
    assert controller.solver_stats["success"] is True
    return np.array([float(np.squeeze(controller.u0[name])) for name in INPUT_NAMES])


@pytest.mark.xfail(strict=True, reason="PF-03: terminal value reads beyond the forecast horizon")
def test_first_control_is_independent_of_out_of_horizon_prices():
    shared_horizon = np.full(25, 0.10)
    low_tail = np.concatenate([shared_horizon, np.full(24, -1.0)])
    high_tail = np.concatenate([shared_horizon, np.full(24, 1.0)])

    np.testing.assert_allclose(
        _first_mpc_control(low_tail),
        _first_mpc_control(high_tail),
        atol=1e-4,
        rtol=0.0,
    )
