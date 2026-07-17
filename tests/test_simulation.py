from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import venv

import numpy as np
import pandas as pd
import pytest

import greenhouse_energy_hub.controllers.baseline as baseline_controller


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


def _boundary_violations(source: str, origin: str) -> list[str]:
    """Return forbidden repository-root imports and interpreter path mutations."""
    tree = ast.parse(source, filename=origin)
    forbidden_roots = {"models", "control", "accounting"}
    system_modules: set[str] = set()
    system_paths: set[str] = set()
    importlib_modules: set[str] = set()
    import_module_functions: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    system_modules.add(alias.asname or alias.name)
                elif alias.name == "importlib":
                    importlib_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "sys":
            for alias in node.names:
                if alias.name == "path":
                    system_paths.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_functions.add(alias.asname or alias.name)

    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id in system_modules
        ):
            violations.append(f"{origin}:{node.lineno}: interpreter path access")
        elif isinstance(node, ast.Name) and node.id in system_paths:
            violations.append(f"{origin}:{node.lineno}: interpreter path access")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in forbidden_roots:
                    violations.append(f"{origin}:{node.lineno}: root import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in forbidden_roots:
                violations.append(f"{origin}:{node.lineno}: root import {node.module}")
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            is_dynamic_import = (
                isinstance(function, ast.Name)
                and function.id in import_module_functions | {"__import__"}
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
                and isinstance(function.value, ast.Name)
                and function.value.id in importlib_modules
            )
            module_name = node.args[0]
            if (
                is_dynamic_import
                and isinstance(module_name, ast.Constant)
                and isinstance(module_name.value, str)
                and module_name.value.split(".", 1)[0] in forbidden_roots
            ):
                violations.append(
                    f"{origin}:{node.lineno}: dynamic root import {module_name.value}"
                )
    return violations


@pytest.mark.parametrize(
    "source",
    (
        "import sys as runtime\nruntime.path.insert(0, '.')\n",
        "from sys import path as runtime_path\nruntime_path.insert(0, '.')\n",
        (
            "import importlib as loader\n"
            f"loader.import_module({('mod' + 'els.hub_model')!r})\n"
        ),
        (
            "from importlib import import_module as load_module\n"
            f"load_module({('con' + 'trol.mpc')!r})\n"
        ),
        f"__import__({('account' + 'ing')!r})\n",
    ),
)
def test_boundary_guard_rejects_aliased_paths_and_constant_dynamic_imports(source):
    assert _boundary_violations(source, "characterization.py")


def test_project_sources_use_only_installed_package_imports():
    repository_root = Path(__file__).resolve().parent.parent
    violations: list[str] = []
    for directory in ("src", "experiments", "scripts", "tests"):
        for path in sorted((repository_root / directory).rglob("*.py")):
            violations.extend(
                _boundary_violations(
                    path.read_text(encoding="utf-8"),
                    str(path.relative_to(repository_root)),
                )
            )

    notebook_path = repository_root / "notebooks" / "results_analysis.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            violations.extend(
                _boundary_violations(
                    "".join(cell.get("source", [])),
                    f"{notebook_path.relative_to(repository_root)}:cell-{index}",
                )
            )

    assert violations == []


def test_editable_install_imports_all_owning_modules_outside_repository(tmp_path):
    repository_root = Path(__file__).resolve().parent.parent
    environment = tmp_path / "installed-package"
    outside_repository = tmp_path / "outside-repository"
    outside_repository.mkdir()
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(repository_root),
        ],
        cwd=outside_repository,
        check=True,
        capture_output=True,
        text=True,
    )
    module_names = (
        "greenhouse_energy_hub.hub",
        "greenhouse_energy_hub.simulation",
        "greenhouse_energy_hub.scenarios",
        "greenhouse_energy_hub.evaluation",
    )
    probe = (
        "import importlib, pathlib\n"
        f"expected = pathlib.Path({str(repository_root / 'src')!r}).resolve()\n"
        f"names = {module_names!r}\n"
        "for name in names:\n"
        "    module_path = pathlib.Path(importlib.import_module(name).__file__).resolve()\n"
        "    assert module_path.is_relative_to(expected), (name, module_path, expected)\n"
    )
    subprocess.run(
        [str(python), "-c", probe],
        cwd=outside_repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _test_diagnostics(forecast, **overrides):
    from greenhouse_energy_hub.simulation import DecisionDiagnostics

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
    from greenhouse_energy_hub.hub import HubControl

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
    from greenhouse_energy_hub.scenarios import ScenarioPoint

    prices = tuple(float(price) for price in prices)
    if temperatures is None:
        temperatures = (5.0,) * len(prices)
    timestamps = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    return tuple(
        ScenarioPoint(
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
    from greenhouse_energy_hub.simulation import ValidRun, load_data, run_simulation
    from greenhouse_energy_hub.hub import state_bounds

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
    import greenhouse_energy_hub.controllers.mpc as mpc_controller
    import greenhouse_energy_hub.simulation as rolling_horizon
    import greenhouse_energy_hub.hub as hub_model

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
    from greenhouse_energy_hub.simulation import InvalidRun

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

    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, InvalidRun
    from greenhouse_energy_hub.hub import HubConfiguration

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
    scenario = rolling_horizon._scenario_from_frame(
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

    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControllerFailure, InvalidRun
    from greenhouse_energy_hub.hub import HubConfiguration

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
    scenario = rolling_horizon._scenario_from_frame(
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

    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, InvalidRun
    from greenhouse_energy_hub.hub import HubConfiguration

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
    scenario = rolling_horizon._scenario_from_frame(
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
    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, InvalidRun

    control = _zero_control()
    control = type(control)(**{**control.__dict__, field: value})

    def decide(_self, _state, forecast):
        return ControlDecision(control=control, diagnostics=_test_diagnostics(forecast))

    monkeypatch.setattr(baseline_controller.BaselineControllerAdapter, "decide", decide)

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
    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, InvalidRun
    from greenhouse_energy_hub.hub import HubFlows, HubState, HubStep

    def decide(_self, _state, forecast):
        return ControlDecision(
            control=_zero_control(), diagnostics=_test_diagnostics(forecast)
        )

    monkeypatch.setattr(baseline_controller.BaselineControllerAdapter, "decide", decide)

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

    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, InvalidRun
    from greenhouse_energy_hub.hub import HubState

    def decide(_self, _state, forecast):
        return ControlDecision(
            control=_zero_control(), diagnostics=_test_diagnostics(forecast)
        )

    monkeypatch.setattr(baseline_controller.BaselineControllerAdapter, "decide", decide)
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
    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, ValidRun
    from greenhouse_energy_hub.hub import ETA_BAT_CH, ETA_BAT_DIS

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

    monkeypatch.setattr(baseline_controller.BaselineControllerAdapter, "decide", decide)
    monkeypatch.setattr(
        baseline_controller.BaselineControllerAdapter,
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
    from greenhouse_energy_hub.simulation import ValidRun, run_simulation

    outcome = run_simulation(hourly_frame.iloc[:1], mode="baseline")

    assert isinstance(outcome, ValidRun)
    with pytest.raises(TypeError):
        outcome.controller_configuration["mutated"] = True
    with pytest.raises(TypeError):
        outcome.capability_policy["hydrogen_dispatch"] = True


def test_valid_run_frame_uses_shared_evaluation_step_line_items(
    monkeypatch,
    hourly_frame,
):
    from greenhouse_energy_hub.evaluation import DEFAULT_EVALUATION_POLICY, StepLineItems
    import greenhouse_energy_hub.simulation as rolling_horizon

    outcome = rolling_horizon.run_simulation(
        hourly_frame.iloc[:1], mode="baseline"
    )
    calls = []

    def shared_line_items(record, policy, step_hours):
        calls.append((record, policy, step_hours))
        return StepLineItems(
            operating_step=record.operating_step,
            grid_cost_eur=123.456,
            battery_wear_eur=0.0,
            thermal_store_wear_eur=0.0,
            electrolyser_wear_eur=0.0,
            fuel_cell_wear_eur=0.0,
            operating_cost_eur=123.456,
            comfort_violation_c_h=7.89,
        )

    monkeypatch.setattr(rolling_horizon, "evaluate_step", shared_line_items)

    frame = outcome.to_frame()

    assert frame.iloc[0]["grid_cost_EUR"] == pytest.approx(123.456)
    assert frame.iloc[0]["T_violation_C"] == pytest.approx(7.89)
    assert calls == [
        (outcome.records[0], DEFAULT_EVALUATION_POLICY, 1.0)
    ]


def test_baseline_run_records_exact_capability_policy(hourly_frame):
    from greenhouse_energy_hub.simulation import ValidRun, run_simulation

    outcome = run_simulation(hourly_frame.iloc[:1], mode="baseline")

    assert isinstance(outcome, ValidRun)
    assert outcome.capability_policy == {
        "hydrogen_dispatch": False,
        "thermal_store_charging": False,
        "grid_battery_charging": False,
        "battery_discharge_price_threshold_eur_per_kwh": 0.12,
    }


def test_mpc_adapter_configuration_mappings_are_read_only_copies():
    from greenhouse_energy_hub.controllers.mpc import MpcControllerAdapter

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
    from greenhouse_energy_hub.controllers.mpc import MpcControllerAdapter

    return MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=horizon_steps,
        configuration={"horizon_steps": horizon_steps},
        capability_policy={"battery": True},
    )


def test_both_owned_controller_adapters_cross_simulate_run(
    monkeypatch,
    hourly_frame,
):
    import greenhouse_energy_hub.simulation as simulation
    from greenhouse_energy_hub.controllers.baseline import BaselineControllerAdapter
    from greenhouse_energy_hub.hub import (
        HubConfiguration,
        HubFlows,
        HubStep,
    )
    from greenhouse_energy_hub.simulation import ValidRun

    scenario = simulation._scenario_from_frame(
        hourly_frame.iloc[:2],
        forecast_horizon_steps=1,
        operating_step_count=1,
    )
    hub_configuration = HubConfiguration()
    monkeypatch.setattr(
        simulation,
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

    baseline = simulation.simulate_run(
        scenario,
        BaselineControllerAdapter(hub_configuration),
        hub_configuration,
    )
    mpc = simulation.simulate_run(
        scenario,
        _mpc_adapter(_ForecastAwareMpc(), horizon_steps=1),
        hub_configuration,
    )

    assert isinstance(baseline, ValidRun)
    assert isinstance(mpc, ValidRun)
    assert (baseline.controller_name, mpc.controller_name) == ("baseline", "mpc")
    assert len(baseline.records) == len(mpc.records) == 1


def test_mpc_rejects_wrong_length_forecast_before_solver():
    from greenhouse_energy_hub.simulation import ControllerFailure
    from greenhouse_energy_hub.hub import initial_state

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
    from greenhouse_energy_hub.simulation import ControlDecision
    from greenhouse_energy_hub.hub import HP_COP, initial_state

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
    from greenhouse_energy_hub.controllers.mpc import MpcConfiguration, build_mpc
    from greenhouse_energy_hub.hub import (
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


def test_horizon_length_does_not_change_successor_state_bounds():
    from greenhouse_energy_hub.controllers.mpc import MpcConfiguration, build_mpc
    from greenhouse_energy_hub.hub import (
        HubConfiguration,
        STATE_MODEL_NAMES,
        operational_state_bounds,
    )

    hub_config = HubConfiguration()
    one_step, _ = build_mpc(hub_config, MpcConfiguration(horizon_steps=1))
    publication_horizon, _ = build_mpc(
        hub_config,
        MpcConfiguration(horizon_steps=24),
    )

    assert one_step.settings.use_terminal_bounds is True
    assert publication_horizon.settings.use_terminal_bounds is True
    for field_name, expected_bounds in operational_state_bounds(hub_config).items():
        model_name = STATE_MODEL_NAMES[field_name]
        one_step_bounds = (
            float(one_step.terminal_bounds["lower", model_name]),
            float(one_step.terminal_bounds["upper", model_name]),
        )
        publication_bounds = (
            float(publication_horizon.terminal_bounds["lower", model_name]),
            float(publication_horizon.terminal_bounds["upper", model_name]),
        )
        assert one_step_bounds == pytest.approx(expected_bounds)
        assert publication_bounds == pytest.approx(expected_bounds)
        assert one_step_bounds == pytest.approx(publication_bounds)


@pytest.mark.parametrize("capability", ["battery", "hydrogen", "thermal_store"])
def test_disabled_asset_keeps_positive_nominal_mpc_scaling(capability):
    from greenhouse_energy_hub.controllers.mpc import MpcConfiguration, build_mpc
    from greenhouse_energy_hub.hub import (
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


def test_load_data_returns_scenario_and_rejects_missing_coverage():
    from greenhouse_energy_hub.simulation import load_data
    from greenhouse_energy_hub.scenarios import Scenario, ScenarioCoverageError

    scenario = load_data(start_month=1, n_days=1, forecast_hours=3)

    assert isinstance(scenario, Scenario)
    assert scenario.operating_step_count == 24
    assert len(scenario.points) == 27
    assert scenario.points[-1].timestamp_utc == (
        scenario.points[23].timestamp_utc + pd.Timedelta(hours=3)
    )

    with pytest.raises(ScenarioCoverageError):
        load_data(start_month=12, n_days=30, forecast_hours=49)


def test_scenario_iterates_only_operating_window(monkeypatch, hourly_frame):
    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, ValidRun
    from greenhouse_energy_hub.hub import HubFlows, HubStep

    observed_forecasts = []

    def decide(_self, _state, forecast):
        observed_forecasts.append(tuple(point.timestamp_utc for point in forecast))
        return ControlDecision(
            control=_zero_control(), diagnostics=_test_diagnostics(forecast)
        )

    monkeypatch.setattr(baseline_controller.BaselineControllerAdapter, "decide", decide)
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
    scenario = rolling_horizon._scenario_from_frame(
        hourly_frame.iloc[:4], forecast_horizon_steps=2, operating_step_count=2
    )

    outcome = rolling_horizon.run_simulation(scenario, mode="baseline")

    assert isinstance(outcome, ValidRun)
    assert len(outcome.records) == 2
    assert [window[0] for window in observed_forecasts] == [
        timestamp.to_pydatetime() for timestamp in hourly_frame.index[:2]
    ]


def test_baseline_and_mpc_runs_share_max_horizon_scenario_canonical_content(
    monkeypatch,
):
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from datetime import datetime, timedelta
    import json
    from types import SimpleNamespace

    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import (
        ControlDecision,
        DecisionDiagnostics,
        ValidRun,
    )
    from greenhouse_energy_hub.controllers.baseline import BaselineControllerAdapter
    from greenhouse_energy_hub.hub import HubConfiguration, HubFlows, HubStep

    scenario = rolling_horizon.load_data(
        start_month=1, n_days=1, forecast_hours=3
    )
    config = HubConfiguration()

    def identity_step(state, *_args):
        return HubStep(
            successor=state,
            flows=HubFlows(
                grid_kw=0.0,
                generated_heat_kw=0.0,
                heat_to_air_kw=0.0,
                thermal_charge_margin_kw=0.0,
                hydrogen_production_kg_per_h=0.0,
                hydrogen_consumption_kg_per_h=0.0,
            ),
        )

    def mpc_decide(_state, forecast):
        return ControlDecision(
            control=_zero_control(),
            diagnostics=DecisionDiagnostics(
                adapter="mpc",
                decision_status="success",
                solver_success=True,
                solver_return_status="Solve_Succeeded",
                solver_iterations=1,
                solver_wall_seconds=0.001,
                forecast_start_utc=forecast[0].timestamp_utc,
                forecast_end_utc=forecast[-1].timestamp_utc,
                terminal_electric_value_eur_per_kwh=0.0,
                terminal_heat_value_eur_per_kwhth=0.0,
            ),
        )

    monkeypatch.setattr(rolling_horizon, "advance_hub", identity_step)
    baseline = rolling_horizon.simulate_run(
        scenario, BaselineControllerAdapter(config), config
    )
    mpc = rolling_horizon.simulate_run(
        scenario,
        SimpleNamespace(
            name="mpc",
            configuration={"horizon_steps": 3},
            capability_policy={},
            forecast_horizon_steps=3,
            requires_operational_storage_bounds=True,
            decide=mpc_decide,
        ),
        config,
    )

    assert isinstance(baseline, ValidRun)
    assert isinstance(mpc, ValidRun)
    assert baseline.scenario is scenario
    assert mpc.scenario is scenario
    assert baseline.scenario == mpc.scenario
    assert scenario.forecast_horizon_capacity_steps == 3
    assert len(baseline.records) == len(mpc.records) == scenario.operating_step_count
    assert len(scenario.points) == scenario.operating_step_count + 3

    def canonical_value(value):
        if is_dataclass(value):
            return {
                field.name: canonical_value(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {key: canonical_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical_value(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, timedelta):
            return value.total_seconds()
        return value

    def canonical_bytes(run):
        return json.dumps(
            canonical_value(run.scenario),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    assert canonical_bytes(baseline) == canonical_bytes(mpc)


def test_missing_final_forecast_coverage_fails_before_run(hourly_frame):
    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.scenarios import ScenarioCoverageError

    # Two operating steps with a two-stage controller require four points in total;
    # three points deliberately leave the final N+1 view one point short. Scenario
    # construction must reject this before any Controller can run.
    with pytest.raises(ScenarioCoverageError):
        rolling_horizon._scenario_from_frame(
            hourly_frame.iloc[:3],
            forecast_horizon_steps=2,
            operating_step_count=2,
        )


def test_experiment_configuration_constructs_one_asset_capability_owner():
    from experiments.run_scenario import build_hub_configuration

    config = build_hub_configuration(
        battery=True,
        hydrogen=False,
        thermal_store=False,
    )

    assert config.capabilities.battery is True
    assert config.capabilities.hydrogen is False
    assert config.capabilities.thermal_store is False


def test_execute_experiment_captures_package_initializers_and_their_dirty_state(
    monkeypatch,
    tmp_path,
):
    from datetime import datetime
    import subprocess

    from experiments import run_scenario

    repository = tmp_path / "repository"
    required_initializers = {
        "src/greenhouse_energy_hub/__init__.py",
        "src/greenhouse_energy_hub/controllers/__init__.py",
    }
    executable_paths = set(
        run_scenario.executable_paths_for_controller("baseline")
    ) | required_initializers
    for relative_path in executable_paths:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# committed executable input\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Task 10 Test"),
        ("git", "config", "user.email", "task10@example.invalid"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(
            command,
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    dirty_initializer = "src/greenhouse_energy_hub/__init__.py"
    (repository / dirty_initializer).write_text(
        "# dirty executable input\n",
        encoding="utf-8",
    )
    observed = {}
    real_capture = run_scenario._capture_publication_context

    class CaptureComplete(Exception):
        pass

    def capture_then_stop(paths, *, repository_root):
        context = real_capture(paths, repository_root=repository_root)
        observed["paths"] = tuple(paths)
        observed["code_provenance"] = context.code_provenance
        raise CaptureComplete

    monkeypatch.setattr(
        run_scenario,
        "_capture_publication_context",
        capture_then_stop,
    )

    with pytest.raises(CaptureComplete):
        run_scenario.execute_experiment(
            name="capture-only",
            operating_start=datetime.fromisoformat("2023-01-02T00:00:00+01:00"),
            calendar_days=1,
            controller_name="baseline",
            scenario_max_horizon_steps=24,
            repository_root=repository,
            results_root=repository / "results",
        )

    assert required_initializers.issubset(observed["paths"])
    provenance = observed["code_provenance"]
    assert provenance["publication_eligible"] is False
    assert provenance["dirty_executable_paths"] == (dirty_initializer,)
    assert set(provenance["executable_path_hashes"]) >= required_initializers


def test_publication_candidate_map_records_only_a_verified_full_identifier(
    monkeypatch,
    tmp_path,
):
    import json

    from experiments import run_scenario
    from greenhouse_energy_hub.evaluation import RunBundle

    identifier = "a" * 64
    bundle = RunBundle(
        identifier=identifier,
        specification_identifier="b" * 64,
        path=tmp_path / "runs" / f"scenario--baseline--{identifier}",
        manifest={},
    )
    verified = []

    def verify(path, requested_identifier, *, repository_root):
        verified.append((path, requested_identifier, repository_root))
        return bundle

    monkeypatch.setattr(run_scenario, "verify_run_bundle", verify)
    candidate_index = tmp_path / "diagnostics" / "publication-candidates.json"

    run_scenario.record_publication_candidate(
        "winter-baseline",
        bundle,
        candidate_index=candidate_index,
        repository_root=tmp_path,
    )

    assert json.loads(candidate_index.read_text(encoding="utf-8")) == {
        "winter-baseline": identifier
    }
    assert verified == [(bundle.path, identifier, tmp_path)]
    assert list(candidate_index.parent.glob(".publication-candidates.json.*")) == []

    candidate_index.write_text('{"legacy-short-id":"abc123"}', encoding="utf-8")
    with pytest.raises(ValueError, match="full lowercase SHA-256"):
        run_scenario.record_publication_candidate(
            "winter-baseline",
            bundle,
            candidate_index=candidate_index,
            repository_root=tmp_path,
        )


def test_baseline_adapter_policy_metadata_has_one_stable_owner():
    from greenhouse_energy_hub.controllers.baseline import (
        BASELINE_TARGET_C,
        BaselineControllerAdapter,
    )
    from greenhouse_energy_hub.hub import T_MIN_C

    assert BaselineControllerAdapter.__module__ == (
        "greenhouse_energy_hub.controllers.baseline"
    )
    assert BASELINE_TARGET_C == T_MIN_C + 0.5
    assert BaselineControllerAdapter.configuration == {
        "target_indoor_temperature_c": BASELINE_TARGET_C
    }
    assert BaselineControllerAdapter.capability_policy == {
        "hydrogen_dispatch": False,
        "thermal_store_charging": False,
        "grid_battery_charging": False,
        "battery_discharge_price_threshold_eur_per_kwh": 0.12,
    }
    with pytest.raises(TypeError):
        BaselineControllerAdapter.capability_policy["hydrogen_dispatch"] = True


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
    from greenhouse_energy_hub.hub import (
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


@pytest.mark.parametrize("signed_value", [5e-7, -5e-7])
@pytest.mark.parametrize(
    ("capability", "state_field"),
    [
        ("battery", "soc_battery_kwh"),
        ("hydrogen", "soc_hydrogen_kg"),
        ("thermal_store", "soc_thermal_kwh"),
    ],
)
def test_public_successor_validator_rejects_every_nonzero_disabled_state(
    capability, state_field, signed_value
):
    from greenhouse_energy_hub.hub import (
        AssetCapabilities,
        HubConfiguration,
        HubState,
        initial_state,
        validate_successor,
    )

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )
    state = initial_state(config)
    nonzero_state = HubState(
        **{**state.__dict__, state_field: signed_value}
    )

    issues = validate_successor(
        nonzero_state,
        config,
        require_operational_storage=False,
        tolerance=1e-4,
    )

    assert [(issue.code, issue.field) for issue in issues] == [
        ("out_of_bounds", state_field)
    ]


@pytest.mark.parametrize("signed_value", [5e-7, -5e-7])
@pytest.mark.parametrize(
    ("capability", "control_field"),
    [
        ("battery", "battery_charge_kw"),
        ("battery", "battery_discharge_kw"),
        ("hydrogen", "electrolyser_kw"),
        ("hydrogen", "fuel_cell_kw"),
        ("thermal_store", "thermal_charge_kw"),
        ("thermal_store", "thermal_discharge_kw"),
    ],
)
def test_public_control_validator_rejects_every_nonzero_disabled_control(
    capability, control_field, signed_value
):
    from greenhouse_energy_hub.hub import (
        AssetCapabilities,
        HubConfiguration,
        HubControl,
        validate_control,
    )

    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )
    control = HubControl(
        **{**_zero_control().__dict__, control_field: signed_value}
    )

    issues = validate_control(control, config, tolerance=1e-4)

    assert [(issue.code, issue.field) for issue in issues] == [
        ("out_of_bounds", control_field)
    ]


@pytest.mark.parametrize("signed_value", [5e-7, -5e-7])
@pytest.mark.parametrize(
    ("capability", "control_field"),
    [
        ("battery", "battery_charge_kw"),
        ("battery", "battery_discharge_kw"),
        ("hydrogen", "electrolyser_kw"),
        ("hydrogen", "fuel_cell_kw"),
        ("thermal_store", "thermal_charge_kw"),
        ("thermal_store", "thermal_discharge_kw"),
    ],
)
def test_simulation_zeroes_disabled_solver_noise_before_validation_and_recording(
    monkeypatch, hourly_frame, capability, control_field, signed_value
):
    import greenhouse_energy_hub.simulation as rolling_horizon
    from greenhouse_energy_hub.simulation import ControlDecision, ValidRun
    from greenhouse_energy_hub.hub import AssetCapabilities, HubConfiguration, HubControl

    raw_control = HubControl(
        **{**_zero_control().__dict__, control_field: signed_value}
    )

    def decide(_self, _state, forecast):
        return ControlDecision(
            control=raw_control,
            diagnostics=_test_diagnostics(forecast),
        )

    monkeypatch.setattr(baseline_controller.BaselineControllerAdapter, "decide", decide)
    config = HubConfiguration(
        capabilities=AssetCapabilities(**{capability: False})
    )

    outcome = rolling_horizon.run_simulation(
        hourly_frame.iloc[:1],
        mode="baseline",
        hub_config=config,
    )

    assert isinstance(outcome, ValidRun)
    assert getattr(outcome.records[0].control, control_field) == 0.0


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
    from greenhouse_energy_hub.hub import (
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
    from greenhouse_energy_hub.evaluation import stored_equiv_kwh
    from greenhouse_energy_hub.hub import AssetCapabilities, HubConfiguration

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
    from greenhouse_energy_hub.simulation import ValidRun, load_data, run_simulation
    from greenhouse_energy_hub.hub import AssetCapabilities, HubConfiguration

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

    from greenhouse_energy_hub.controllers.mpc import (
        MpcConfiguration,
        MpcControllerAdapter,
        build_mpc,
    )
    from greenhouse_energy_hub.simulation import ControlDecision
    from greenhouse_energy_hub.hub import HubConfiguration, initial_state
    from greenhouse_energy_hub.scenarios import Scenario

    points = _forecast_points(prices)
    step_duration = timedelta(hours=1)
    scenario = Scenario(
        name="causal_test",
        operating_start=points[0].timestamp_utc,
        operating_end=points[0].timestamp_utc + step_duration,
        forecast_end=points[-1].timestamp_utc + step_duration,
        forecast_horizon_capacity_steps=len(points) - 1,
        step_duration=step_duration,
        operating_step_count=1,
        points=points,
        provenance=(),
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


def test_mpc_configuration_has_exact_stable_fields_and_policy_separation():
    from dataclasses import fields

    from greenhouse_energy_hub.evaluation import EvaluationPolicy, WearCoefficients
    from greenhouse_energy_hub.controllers.mpc import MpcConfiguration

    assert [field.name for field in fields(MpcConfiguration)] == [
        "horizon_steps",
        "terminal_weight",
        "battery_wear_eur_per_kwh",
        "thermal_store_wear_eur_per_kwh",
        "electrolyser_wear_eur_per_kwh",
        "fuel_cell_wear_eur_per_kwh",
        "complementarity_weight",
        "comfort_slack_weight",
        "input_move_weight",
        "solver_max_iterations",
        "solver_tolerance",
    ]
    policy = EvaluationPolicy(
        wear=WearCoefficients(0.011, 0.022, 0.033, 0.044)
    )
    config = MpcConfiguration.from_evaluation_policy(
        policy,
        horizon_steps=6,
        terminal_weight=2.0,
        complementarity_weight=0.3,
        comfort_slack_weight=40.0,
        input_move_weight=0.05,
    )
    metadata = config.to_controller_metadata()

    assert metadata["solver_objective_economic_terms"] == {
        "battery_wear_eur_per_kwh": 0.011,
        "thermal_store_wear_eur_per_kwh": 0.022,
        "electrolyser_wear_eur_per_kwh": 0.033,
        "fuel_cell_wear_eur_per_kwh": 0.044,
    }
    assert metadata["solver_diagnostics"] == {
        "terminal_weight": 2.0,
        "complementarity_weight": 0.3,
        "comfort_slack_weight": 40.0,
        "input_move_weight": 0.05,
        "solver_max_iterations": 800,
        "solver_tolerance": 1e-6,
    }
    assert metadata["horizon_steps"] == 6
    assert set(metadata) == {
        "horizon_steps",
        "solver_objective_economic_terms",
        "solver_diagnostics",
    }


def test_run_simulation_threads_named_evaluation_policy_into_mpc_configuration(
    monkeypatch,
    hourly_frame,
):
    from greenhouse_energy_hub.evaluation import EvaluationPolicy, WearCoefficients
    import greenhouse_energy_hub.controllers.mpc as mpc_controller
    import greenhouse_energy_hub.simulation as rolling_horizon

    captured = {}

    class InertMpc:
        x0 = None

        def set_initial_guess(self):
            return None

    def capture_build_mpc(hub_config, config):
        captured["hub_config"] = hub_config
        captured["config"] = config
        return InertMpc(), object()

    def capture_simulate_run(scenario, controller, hub_config):
        captured["controller_configuration"] = controller.configuration
        return "simulation-not-needed"

    monkeypatch.setattr(mpc_controller, "build_mpc", capture_build_mpc)
    monkeypatch.setattr(rolling_horizon, "simulate_run", capture_simulate_run)
    policy = EvaluationPolicy(
        name="shared-test-policy",
        wear=WearCoefficients(0.101, 0.202, 0.303, 0.404),
    )

    outcome = rolling_horizon.run_simulation(
        hourly_frame.iloc[:2],
        mode="mpc",
        n_horizon=1,
        evaluation_policy=policy,
    )

    assert outcome == "simulation-not-needed"
    assert captured["config"].battery_wear_eur_per_kwh == 0.101
    assert captured["config"].thermal_store_wear_eur_per_kwh == 0.202
    assert captured["config"].electrolyser_wear_eur_per_kwh == 0.303
    assert captured["config"].fuel_cell_wear_eur_per_kwh == 0.404
    assert captured["controller_configuration"][
        "solver_objective_economic_terms"
    ] == {
        "battery_wear_eur_per_kwh": 0.101,
        "thermal_store_wear_eur_per_kwh": 0.202,
        "electrolyser_wear_eur_per_kwh": 0.303,
        "fuel_cell_wear_eur_per_kwh": 0.404,
    }
    with pytest.raises(TypeError):
        captured["controller_configuration"][
            "solver_objective_economic_terms"
        ]["battery_wear_eur_per_kwh"] = 999.0
    with pytest.raises(TypeError):
        captured["controller_configuration"]["solver_diagnostics"][
            "terminal_weight"
        ] = 999.0
