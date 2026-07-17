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


def _test_diagnostics(forecast, **overrides):
    from control.rolling_horizon import DecisionDiagnostics

    values = {
        "adapter": "baseline",
        "decision_status": "success",
        "solver_success": None,
        "solver_return_status": None,
        "solver_iterations": None,
        "solver_wall_seconds": None,
        "forecast_start_utc": forecast[0].timestamp_utc,
        "forecast_end_utc": forecast[-1].timestamp_utc,
        "terminal_electric_value_eur_per_kwh": None,
        "terminal_heat_value_eur_per_kwhth": None,
    }
    values.update(overrides)
    return DecisionDiagnostics(
        **values,
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


def _forecast_points(
    prices,
    *,
    temperatures=None,
    start="2023-01-02 00:00:00+00:00",
):
    from control.rolling_horizon import _LegacyScenarioPoint

    prices = tuple(float(price) for price in prices)
    if temperatures is None:
        temperatures = (5.0,) * len(prices)
    timestamps = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    return tuple(
        _LegacyScenarioPoint(
            timestamp_utc=timestamp.to_pydatetime(),
            price_eur_per_kwh=price,
            pv_kw=0.0,
            electric_load_kw=400.0,
            outdoor_temperature_c=float(temperature),
            irradiance_w_per_m2=0.0,
        )
        for timestamp, price, temperature in zip(
            timestamps, prices, temperatures, strict=True
        )
    )


def test_one_step_mpc_keeps_every_reached_state_valid():
    from control.rolling_horizon import ValidRun, load_data, run_simulation
    from models.hub_model import state_bounds

    outcome = run_simulation(
        load_data(start_month=1, n_days=2, forecast_hours=1),
        mode="mpc",
        n_horizon=1,
    )
    assert isinstance(outcome, ValidRun)
    bounds = state_bounds()
    frame = outcome.to_frame()
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
    monkeypatch.setattr(
        mpc_controller,
        "build_mpc",
        lambda *_args, **_kwargs: (failed_mpc, object()),
    )

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
    "case",
    [
        "failure_status",
        "solver_failure",
        "wrong_adapter",
        "missing_mpc_status",
        "invalid_iteration_type",
        "non_finite_wall_time",
        "reversed_forecast",
        "non_finite_terminal_value",
        "wrong_schema",
    ],
)
def test_malformed_success_diagnostics_fail_before_physics(
    monkeypatch, hourly_frame, case
):
    from datetime import timedelta
    from types import SimpleNamespace

    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, InvalidRun
    from models.hub_model import HubConfiguration

    def decide(_state, forecast):
        if case == "wrong_schema":
            diagnostics = object()
        else:
            overrides = {
                "adapter": "mpc",
                "solver_success": True,
                "solver_return_status": "Solve_Succeeded",
                "solver_iterations": 3,
                "solver_wall_seconds": 0.01,
            }
            if case == "failure_status":
                overrides["decision_status"] = "failure"
            elif case == "solver_failure":
                overrides["solver_success"] = False
            elif case == "wrong_adapter":
                overrides["adapter"] = "baseline"
            elif case == "missing_mpc_status":
                overrides["solver_return_status"] = None
            elif case == "invalid_iteration_type":
                overrides["solver_iterations"] = 3.5
            elif case == "non_finite_wall_time":
                overrides["solver_wall_seconds"] = np.nan
            elif case == "reversed_forecast":
                overrides["forecast_end_utc"] = (
                    forecast[0].timestamp_utc - timedelta(hours=1)
                )
            elif case == "non_finite_terminal_value":
                overrides["terminal_electric_value_eur_per_kwh"] = np.inf
            diagnostics = _test_diagnostics(forecast, **overrides)
        return ControlDecision(control=_zero_control(), diagnostics=diagnostics)

    controller = SimpleNamespace(
        name="mpc",
        configuration={},
        capability_policy={},
        forecast_horizon_steps=0,
        requires_operational_storage_bounds=True,
        decide=decide,
    )
    scenario = rolling_horizon._legacy_scenario_from_frame(
        hourly_frame.iloc[:1], forecast_horizon_steps=0
    )
    plant_calls = 0

    def must_not_advance(*_args, **_kwargs):
        nonlocal plant_calls
        plant_calls += 1
        raise AssertionError("invalid diagnostics must not reach the hub")

    monkeypatch.setattr(rolling_horizon, "advance_hub", must_not_advance)

    outcome = rolling_horizon.simulate_run(
        scenario, controller, HubConfiguration()
    )

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == "invalid_diagnostics"
    assert outcome.failed_step == 0
    assert outcome.partial_records == ()
    assert plant_calls == 0


def test_controller_failure_requires_failure_diagnostics(monkeypatch, hourly_frame):
    from types import SimpleNamespace

    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControllerFailure, InvalidRun
    from models.hub_model import HubConfiguration

    def decide(_state, forecast):
        return ControllerFailure(
            code="forced_failure",
            message="forced failure",
            diagnostics=_test_diagnostics(forecast, decision_status="success"),
        )

    controller = SimpleNamespace(
        name="baseline",
        configuration={},
        capability_policy={},
        forecast_horizon_steps=0,
        requires_operational_storage_bounds=False,
        decide=decide,
    )
    scenario = rolling_horizon._legacy_scenario_from_frame(
        hourly_frame.iloc[:1], forecast_horizon_steps=0
    )
    monkeypatch.setattr(
        rolling_horizon,
        "advance_hub",
        lambda *_args, **_kwargs: pytest.fail("failure diagnostics reached the hub"),
    )

    outcome = rolling_horizon.simulate_run(
        scenario, controller, HubConfiguration()
    )

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == "invalid_diagnostics"
    assert outcome.partial_records == ()


def test_baseline_success_requires_empty_solver_diagnostics(monkeypatch, hourly_frame):
    from types import SimpleNamespace

    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, InvalidRun
    from models.hub_model import HubConfiguration

    def decide(_state, forecast):
        return ControlDecision(
            control=_zero_control(),
            diagnostics=_test_diagnostics(
                forecast,
                solver_success=True,
                solver_return_status="not-applicable",
            ),
        )

    controller = SimpleNamespace(
        name="baseline",
        configuration={},
        capability_policy={},
        forecast_horizon_steps=0,
        requires_operational_storage_bounds=False,
        decide=decide,
    )
    scenario = rolling_horizon._legacy_scenario_from_frame(
        hourly_frame.iloc[:1], forecast_horizon_steps=0
    )
    monkeypatch.setattr(
        rolling_horizon,
        "advance_hub",
        lambda *_args, **_kwargs: pytest.fail("solver diagnostics reached the hub"),
    )

    outcome = rolling_horizon.simulate_run(
        scenario, controller, HubConfiguration()
    )

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == "invalid_diagnostics"
    assert outcome.partial_records == ()


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


@pytest.mark.parametrize("malformed_shape", ["object", "missing_flows"])
def test_malformed_physics_schema_returns_invalid_run_without_record(
    monkeypatch, hourly_frame, malformed_shape
):
    from types import SimpleNamespace

    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, InvalidRun
    from models.hub_model import HubState

    def decide(_self, _state, forecast):
        return ControlDecision(
            control=_zero_control(), diagnostics=_test_diagnostics(forecast)
        )

    monkeypatch.setattr(rolling_horizon.BaselineControllerAdapter, "decide", decide)
    malformed_step = (
        object()
        if malformed_shape == "object"
        else SimpleNamespace(
            successor=HubState(
                soc_battery_kwh=500.0,
                soc_hydrogen_kg=60.0,
                soc_thermal_kwh=1600.0,
                indoor_temperature_c=19.0,
            )
        )
    )
    monkeypatch.setattr(
        rolling_horizon,
        "advance_hub",
        lambda *_args, **_kwargs: malformed_step,
    )

    outcome = rolling_horizon.run_simulation(
        hourly_frame.iloc[:1], mode="baseline"
    )

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == "physics_schema_error"
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
        outcome.capability_policy["hydrogen_dispatch"] = True


def test_baseline_run_records_exact_capability_policy(hourly_frame):
    from control.rolling_horizon import ValidRun, run_simulation

    outcome = run_simulation(hourly_frame.iloc[:1], mode="baseline")

    assert isinstance(outcome, ValidRun)
    assert outcome.capability_policy == {
        "hydrogen_dispatch": False,
        "thermal_store_charging": False,
        "grid_battery_charging": False,
        "battery_discharge_price_threshold_eur_per_kwh": 0.12,
    }


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


class _ForecastAwareMpc:
    def __init__(self):
        self.x0 = None
        self.solver_stats = {
            "success": True,
            "return_status": "Solve_Succeeded",
            "iter_count": 2,
            "t_wall_total": 0.01,
        }
        self.make_step_calls = []
        self.forecast_activations = []
        self.forecast_clears = 0

        owner = self

        class _ForecastSource:
            def activate(
                self,
                forecast,
                terminal_electric_value,
                terminal_heat_value,
            ):
                owner.forecast_activations.append(
                    (
                        tuple(forecast),
                        terminal_electric_value,
                        terminal_heat_value,
                    )
                )

            def clear(self):
                owner.forecast_clears += 1

        self._forecast_source = _ForecastSource()

    def make_step(self, x0):
        self.make_step_calls.append(np.asarray(x0).copy())
        return np.zeros((len(INPUT_NAMES), 1))


def _mpc_adapter(mpc, horizon_steps):
    from control.mpc_controller import MpcControllerAdapter

    return MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=horizon_steps,
        configuration={"horizon_steps": horizon_steps},
        capability_policy={"battery": True},
    )


def test_mpc_rejects_wrong_length_forecast_before_solver():
    from control.rolling_horizon import ControllerFailure
    from models.hub_model import initial_state

    mpc = _ForecastAwareMpc()
    adapter = _mpc_adapter(mpc, horizon_steps=2)

    decision = adapter.decide(initial_state(), _forecast_points([0.1, 0.2]))

    assert isinstance(decision, ControllerFailure)
    assert decision.code == "forecast_coverage"
    assert decision.diagnostics.decision_status == "failure"
    assert decision.diagnostics.solver_success is None
    assert decision.diagnostics.solver_return_status is None
    assert mpc.make_step_calls == []
    assert mpc.forecast_activations == []


def test_terminal_coefficients_use_only_controlled_stage_points():
    from control.rolling_horizon import ControlDecision
    from models.hub_model import HP_COP, initial_state

    mpc = _ForecastAwareMpc()
    adapter = _mpc_adapter(mpc, horizon_steps=2)
    forecast = _forecast_points(
        [0.10, 0.30, 9.90],
        temperatures=[5.0, 25.0, -20.0],
    )

    decision = adapter.decide(initial_state(), forecast)

    assert isinstance(decision, ControlDecision)
    assert decision.diagnostics.terminal_electric_value_eur_per_kwh == pytest.approx(
        0.20
    )
    assert decision.diagnostics.terminal_heat_value_eur_per_kwhth == pytest.approx(
        0.10 / HP_COP
    )
    assert len(mpc.forecast_activations) == 1
    active_forecast, electric_value, heat_value = mpc.forecast_activations[0]
    assert active_forecast == forecast
    assert electric_value == pytest.approx(0.20)
    assert heat_value == pytest.approx(0.10 / HP_COP)
    assert mpc.forecast_clears == 1


def test_mpc_enables_operational_terminal_bounds():
    from control.mpc_controller import MpcConfiguration, build_mpc
    from models.hub_model import (
        HubConfiguration,
        STATE_MODEL_NAMES,
        operational_state_bounds,
    )

    hub_config = HubConfiguration()
    mpc, _ = build_mpc(hub_config, MpcConfiguration(horizon_steps=2))

    assert mpc.settings.use_terminal_bounds is True
    for field_name, (lower, upper) in operational_state_bounds(hub_config).items():
        model_name = STATE_MODEL_NAMES[field_name]
        assert float(mpc.terminal_bounds["lower", model_name]) == pytest.approx(lower)
        assert float(mpc.terminal_bounds["upper", model_name]) == pytest.approx(upper)


@pytest.mark.parametrize("capability", ["battery", "hydrogen", "thermal_store"])
def test_disabled_asset_keeps_positive_nominal_mpc_scaling(capability):
    from control.mpc_controller import MpcConfiguration, build_mpc
    from models.hub_model import (
        INPUT_SCALE,
        STATE_SCALE,
        AssetCapabilities,
        HubConfiguration,
    )

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )
    mpc, _ = build_mpc(config, MpcConfiguration(horizon_steps=1))

    assert all(value > 0.0 for value in STATE_SCALE.values())
    assert all(value > 0.0 for value in INPUT_SCALE.values())
    for model_name, nominal_scale in STATE_SCALE.items():
        assert float(mpc.scaling["_x", model_name]) == nominal_scale
    for model_name, nominal_scale in INPUT_SCALE.items():
        assert float(mpc.scaling["_u", model_name]) == nominal_scale


def test_load_data_separates_operating_window_and_rejects_missing_coverage():
    from control.rolling_horizon import (
        CoveredFrame,
        ForecastCoverageError,
        load_data,
    )

    covered = load_data(start_month=1, n_days=1, forecast_hours=3)

    assert isinstance(covered, CoveredFrame)
    assert covered.operating_step_count == 24
    assert len(covered.frame) == 27
    assert covered.frame.index[-1] == covered.frame.index[23] + pd.Timedelta(hours=3)

    with pytest.raises(ForecastCoverageError):
        load_data(start_month=12, n_days=30, forecast_hours=49)


def test_covered_frame_iterates_only_operating_window(monkeypatch, hourly_frame):
    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import ControlDecision, CoveredFrame, ValidRun
    from models.hub_model import HubFlows, HubStep

    observed_forecasts = []

    def decide(_self, _state, forecast):
        observed_forecasts.append(tuple(point.timestamp_utc for point in forecast))
        return ControlDecision(
            control=_zero_control(), diagnostics=_test_diagnostics(forecast)
        )

    monkeypatch.setattr(rolling_horizon.BaselineControllerAdapter, "decide", decide)
    monkeypatch.setattr(
        rolling_horizon,
        "advance_hub",
        lambda state, *_args: HubStep(
            successor=state,
            flows=HubFlows(
                grid_kw=0.0,
                generated_heat_kw=0.0,
                heat_to_air_kw=0.0,
                thermal_charge_margin_kw=0.0,
                hydrogen_production_kg_per_h=0.0,
                hydrogen_consumption_kg_per_h=0.0,
            ),
        ),
    )
    covered = CoveredFrame(frame=hourly_frame.iloc[:4], operating_step_count=2)

    outcome = rolling_horizon.run_simulation(covered, mode="baseline")

    assert isinstance(outcome, ValidRun)
    assert len(outcome.records) == 2
    assert [window[0] for window in observed_forecasts] == [
        timestamp.to_pydatetime() for timestamp in hourly_frame.index[:2]
    ]


def test_missing_final_forecast_coverage_fails_without_clamping(hourly_frame):
    import control.rolling_horizon as rolling_horizon
    from control.rolling_horizon import InvalidRun
    from models.hub_model import HubConfiguration

    # Two operating steps with a two-stage controller require four points in total;
    # three points deliberately leave the final N+1 view one point short.
    scenario = rolling_horizon._legacy_scenario_from_frame(
        hourly_frame.iloc[:3],
        forecast_horizon_steps=2,
        operating_step_count=2,
    )
    mpc = _ForecastAwareMpc()
    adapter = _mpc_adapter(mpc, horizon_steps=2)

    outcome = rolling_horizon.simulate_run(
        scenario, adapter, HubConfiguration()
    )

    assert isinstance(outcome, InvalidRun)
    assert outcome.failure_code == "forecast_coverage"
    assert outcome.failed_step == 1
    assert len(outcome.partial_records) == 1
    assert len(mpc.make_step_calls) == 1
    assert [len(activation[0]) for activation in mpc.forecast_activations] == [3]


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


def test_cli_constructs_one_asset_capability_configuration(
    monkeypatch, tmp_path, hourly_frame
):
    from types import SimpleNamespace

    import control.rolling_horizon as rolling_horizon

    observed_configurations = []

    def capture_run(_data, *, mode, hub_config, **_kwargs):
        observed_configurations.append((mode, hub_config))
        return SimpleNamespace(
            failed_step=0,
            failure_code="test_stop",
            message="configuration captured",
        )

    monkeypatch.setattr(
        rolling_horizon, "load_data", lambda **_kwargs: hourly_frame.iloc[:1]
    )
    monkeypatch.setattr(rolling_horizon, "run_simulation", capture_run)
    monkeypatch.setattr(rolling_horizon, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        rolling_horizon.sys,
        "argv",
        [
            "rolling_horizon.py",
            "--days",
            "1",
            "--mode",
            "baseline",
            "--disable-h2",
            "--disable-tes",
        ],
    )

    rolling_horizon.main()

    assert len(observed_configurations) == 1
    mode, config = observed_configurations[0]
    assert mode == "baseline"
    assert config.capabilities.battery is True
    assert config.capabilities.hydrogen is False
    assert config.capabilities.thermal_store is False


@pytest.mark.parametrize(
    ("capability", "state_field", "control_fields"),
    [
        ("battery", "soc_battery_kwh", ("battery_charge_kw", "battery_discharge_kw")),
        ("hydrogen", "soc_hydrogen_kg", ("electrolyser_kw", "fuel_cell_kw")),
        ("thermal_store", "soc_thermal_kwh", ("thermal_charge_kw", "thermal_discharge_kw")),
    ],
)
def test_disabled_asset_configuration_is_exact_zero_capacity(
    capability, state_field, control_fields
):
    from models.hub_model import (
        AssetCapabilities,
        HubConfiguration,
        control_bounds,
        initial_state,
        operational_state_bounds,
        physical_state_bounds,
    )

    capabilities = AssetCapabilities(**{capability: False})
    config = HubConfiguration(capabilities=capabilities)
    state = initial_state(config)

    assert getattr(state, state_field) == 0.0
    assert physical_state_bounds(config)[state_field] == (0.0, 0.0)
    assert operational_state_bounds(config)[state_field] == (0.0, 0.0)
    for field in control_fields:
        assert control_bounds(config)[field] == (0.0, 0.0)


@pytest.mark.parametrize(
    ("capability", "state_field", "control_fields", "disabled_commands"),
    [
        (
            "battery",
            "soc_battery_kwh",
            ("battery_charge_kw", "battery_discharge_kw"),
            {"battery_charge_kw": 100.0, "battery_discharge_kw": 50.0},
        ),
        (
            "hydrogen",
            "soc_hydrogen_kg",
            ("electrolyser_kw", "fuel_cell_kw"),
            {"electrolyser_kw": 100.0, "fuel_cell_kw": 40.0},
        ),
        (
            "thermal_store",
            "soc_thermal_kwh",
            ("thermal_charge_kw", "thermal_discharge_kw"),
            {"thermal_charge_kw": 100.0, "thermal_discharge_kw": 50.0},
        ),
    ],
)
def test_disabled_asset_dynamics_flows_and_validation_are_inert(
    capability, state_field, control_fields, disabled_commands
):
    from models.hub_model import (
        AssetCapabilities,
        ExogenousInputs,
        HubConfiguration,
        HubControl,
        HubState,
        advance_hub,
        initial_state,
        validate_control,
        validate_successor,
    )

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )
    configured_start = initial_state(config)
    invalid_state = HubState(
        **{**configured_start.__dict__, state_field: 123.0}
    )
    control = HubControl(
        **{**_zero_control().__dict__, **disabled_commands, "heat_pump_kw": 10.0}
    )
    exogenous = ExogenousInputs(
        pv_kw=0.0,
        electric_load_kw=100.0,
        price_eur_per_kwh=0.10,
        outdoor_temperature_c=5.0,
        irradiance_w_per_m2=0.0,
    )
    step = advance_hub(invalid_state, control, exogenous, config)
    zeroed_step = advance_hub(
        configured_start,
        HubControl(
            **{
                **control.__dict__,
                **{field_name: 0.0 for field_name in control_fields},
            }
        ),
        exogenous,
        config,
    )

    assert getattr(step.successor, state_field) == 0.0
    assert step.flows == zeroed_step.flows
    if capability == "hydrogen":
        assert step.flows.hydrogen_production_kg_per_h == 0.0
        assert step.flows.hydrogen_consumption_kg_per_h == 0.0
    assert {issue.field for issue in validate_control(control, config)} >= set(
        control_fields
    )
    assert [issue.field for issue in validate_successor(
        invalid_state, config, require_operational_storage=False
    )] == [state_field]


@pytest.mark.parametrize(
    ("capability", "inventories"),
    [
        ("battery", (123.0, 0.0, 0.0)),
        ("hydrogen", (0.0, 123.0, 0.0)),
        ("thermal_store", (0.0, 0.0, 123.0)),
    ],
)
def test_disabled_asset_recoverable_inventory_contribution_is_zero(
    capability, inventories
):
    from accounting import stored_equiv_kwh
    from models.hub_model import AssetCapabilities, HubConfiguration

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )

    assert stored_equiv_kwh(*inventories, config=config) == 0.0


@pytest.mark.parametrize(
    ("capability", "disabled_state", "disabled_controls", "disabled_flows"),
    [
        (
            "hydrogen",
            "soc_hydrogen_kg",
            ("electrolyser_kw", "fuel_cell_kw"),
            ("hydrogen_production_kg_per_h", "hydrogen_consumption_kg_per_h"),
        ),
        (
            "thermal_store",
            "soc_thermal_kwh",
            ("thermal_charge_kw", "thermal_discharge_kw"),
            (),
        ),
    ],
)
def test_two_day_winter_mpc_keeps_disabled_assets_exactly_zero(
    capability, disabled_state, disabled_controls, disabled_flows
):
    from control.rolling_horizon import ValidRun, load_data, run_simulation
    from models.hub_model import AssetCapabilities, HubConfiguration

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )
    outcome = run_simulation(
        load_data(start_month=1, n_days=2),
        mode="mpc",
        hub_config=config,
    )

    assert isinstance(outcome, ValidRun)
    assert outcome.hub_configuration is config
    assert getattr(outcome.initial_state, disabled_state) == 0.0
    assert getattr(outcome.terminal_state, disabled_state) == 0.0
    for record in outcome.records:
        assert getattr(record.start_state, disabled_state) == 0.0
        assert getattr(record.reached_state, disabled_state) == 0.0
        for field_name in disabled_controls:
            assert getattr(record.control, field_name) == 0.0
        for field_name in disabled_flows:
            assert getattr(record.flows, field_name) == 0.0


def _first_mpc_control(prices: np.ndarray) -> np.ndarray:
    from datetime import timedelta

    from control.mpc_controller import (
        MpcConfiguration,
        MpcControllerAdapter,
        build_mpc,
    )
    from control.rolling_horizon import ControlDecision, _LegacyScenario
    from models.hub_model import HubConfiguration, initial_state

    points = _forecast_points(prices)
    step_duration = timedelta(hours=1)
    scenario = _LegacyScenario(
        name="causal_test",
        operating_start=points[0].timestamp_utc,
        operating_end=points[0].timestamp_utc + step_duration,
        forecast_end=points[-1].timestamp_utc,
        forecast_horizon_capacity_steps=24,
        step_duration=step_duration,
        operating_step_count=1,
        points=points,
    )
    hub_config = HubConfiguration()
    mpc, _ = build_mpc(hub_config, MpcConfiguration(horizon_steps=24))
    initial = initial_state()
    mpc.x0 = np.array(
        [
            [initial["SOC_bat"]],
            [initial["SOC_h2"]],
            [initial["SOC_tes"]],
            [initial["T_in"]],
        ]
    )
    mpc.set_initial_guess()
    adapter = MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=24,
        configuration={"horizon_steps": 24},
        capability_policy={"battery": True},
    )

    decision = adapter.decide(initial, scenario.forecast_view(0, 24))

    assert isinstance(decision, ControlDecision)
    return np.array([float(decision.control[name]) for name in INPUT_NAMES])


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
