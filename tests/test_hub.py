"""
Physical and control invariants for the greenhouse energy hub.

Two groups of tests:
  1. Unit tests on the plant model (`hub_dynamics`) — fast, no solver.
  2. A short live MPC roll-out — builds the do-mpc controller and checks that the
     applied control respects the electricity balance and bounds.
  3. Invariants on the saved simulation results (run rolling_horizon.py first):
     balance ~0, no simultaneous charge/discharge, SOC within bounds, comfort band,
     and MPC total cost <= baseline.

Run:  pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hub_model import (  # noqa: E402
    hub_dynamics, initial_state, state_bounds, input_bounds,
    ETA_ELZ, ETA_FC_E, E_H2_LHV_KWH_KG,
    BAT_CAPACITY_KWH, H2_CAPACITY_KG, TES_CAPACITY_KWH,
    T_HARD_MIN_C, T_HARD_MAX_C,
)

RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# 1. Plant-model unit tests
# ---------------------------------------------------------------------------
def _sample_controls():
    return {
        "P_bat_ch": 100.0, "P_bat_dis": 0.0, "P_elz": 50.0, "P_fc": 0.0,
        "P_hp": 100.0, "P_eboiler": 200.0, "Q_tes_ch": 150.0, "Q_tes_dis": 0.0,
        "vent": 0.0,
    }


def _sample_params():
    return {"P_pv": 80.0, "P_load": 600.0, "price": 0.10, "T_out": 5.0, "G_Wm2": 0.0}


def test_characterizes_nominal_multicarrier_step_and_exact_import_fee():
    """Freeze verified legacy plant outputs before the expression layer moves."""
    x_next, metrics = hub_dynamics(initial_state(), _sample_controls(), _sample_params())

    assert x_next == pytest.approx(
        {
            "SOC_bat": 596.0,
            "SOC_h2": 60.975097509750974,
            "SOC_tes": 1742.0,
            "T_in": 15.919402985074626,
        }
    )
    assert metrics["P_grid_kW"] == pytest.approx(970.0)
    assert metrics["grid_cost_EUR"] == pytest.approx(121.25)
    assert metrics["Q_hp_kW"] == pytest.approx(350.0)
    assert metrics["Q_eboiler_kW"] == pytest.approx(198.0)
    assert metrics["m_h2_prod_kg_h"] == pytest.approx(0.9750975097509752)
    assert metrics["elec_residual_kW"] == pytest.approx(0.0, abs=1e-12)


def test_characterizes_limited_baseline_policy_at_fixed_fixtures():
    """Freeze policy decisions without endorsing legacy fairness claims."""
    from control.rolling_horizon import baseline_control

    cold_expensive = baseline_control(
        initial_state(),
        {"P_pv": 0.0, "P_load": 400.0, "price": 0.20, "T_out": 5.0, "G_Wm2": 0.0},
    )
    assert cold_expensive == pytest.approx(
        {
            "P_bat_ch": 0.0,
            "P_bat_dis": 384.0,
            "P_elz": 0.0,
            "P_fc": 0.0,
            "P_hp": 122.97619047619044,
            "P_eboiler": 0.0,
            "Q_tes_ch": 0.0,
            "Q_tes_dis": 0.0,
            "vent": 0.0,
        }
    )

    sunny_surplus = baseline_control(
        initial_state(),
        {"P_pv": 1000.0, "P_load": 100.0, "price": 0.05, "T_out": 20.0, "G_Wm2": 700.0},
    )
    assert sunny_surplus == pytest.approx(
        {
            "P_bat_ch": 416.6666666666667,
            "P_bat_dis": 0.0,
            "P_elz": 0.0,
            "P_fc": 0.0,
            "P_hp": 0.0,
            "P_eboiler": 0.0,
            "Q_tes_ch": 0.0,
            "Q_tes_dis": 0.0,
            "vent": 1.0,
        }
    )


def test_electricity_balance_is_exact():
    """Derived P_grid must close the electricity balance exactly (slack bus)."""
    x, u, p = initial_state(), _sample_controls(), _sample_params()
    _, m = hub_dynamics(x, u, p)
    expected = (p["P_load"] + u["P_bat_ch"] + u["P_elz"] + u["P_hp"] + u["P_eboiler"]
                - p["P_pv"] - u["P_bat_dis"] - u["P_fc"])
    assert m["elec_residual_kW"] == pytest.approx(0.0, abs=1e-9)
    assert m["P_grid_kW"] == pytest.approx(expected, abs=1e-9)


def test_h2_mass_conservation():
    """SOC_h2 change equals electrolyser production minus fuel-cell consumption."""
    x = initial_state()
    u = {**_sample_controls(), "P_elz": 100.0, "P_fc": 40.0}
    p = _sample_params()
    x_next, _ = hub_dynamics(x, u, p)
    prod = ETA_ELZ * u["P_elz"] / E_H2_LHV_KWH_KG
    cons = (u["P_fc"] / ETA_FC_E) / E_H2_LHV_KWH_KG
    assert x_next["SOC_h2"] - x["SOC_h2"] == pytest.approx(prod - cons, rel=1e-9)


def test_tes_cannot_charge_from_nothing_flag():
    """tes_charge_excess flags charging beyond generated heat (model invariant)."""
    x = initial_state()
    u = {**_sample_controls(), "P_hp": 0.0, "P_eboiler": 0.0, "P_fc": 0.0,
         "Q_tes_ch": 100.0}                       # charging with zero generated heat
    _, m = hub_dynamics(x, u, _sample_params())
    assert m["tes_charge_excess_kW"] > 0.0
    # With enough generation the flag clears
    u2 = {**u, "P_eboiler": 200.0}                # 200 * 0.99 ~ 198 kW heat > 100
    _, m2 = hub_dynamics(x, u2, _sample_params())
    assert m2["tes_charge_excess_kW"] == pytest.approx(0.0, abs=1e-9)


def test_temperature_update_is_bounded_and_warms_with_heat():
    """Implicit-Euler thermal node is stable and responds correctly to heating."""
    x = {**initial_state(), "T_in": 18.0}
    p = {"P_pv": 0.0, "P_load": 0.0, "price": 0.0, "T_out": 0.0, "G_Wm2": 0.0}
    cold = {**_sample_controls(), "P_hp": 0.0, "P_eboiler": 0.0,
            "Q_tes_ch": 0.0, "Q_tes_dis": 0.0}
    warm = {**cold, "P_hp": 175.0}                # full heat pump
    t_cold = hub_dynamics(x, cold, p)[0]["T_in"]
    t_warm = hub_dynamics(x, warm, p)[0]["T_in"]
    assert t_warm > t_cold                        # heating raises temperature
    assert T_HARD_MIN_C < t_cold < T_HARD_MAX_C   # remains finite/bounded


@pytest.mark.parametrize(
    ("field", "value", "issue_code"),
    [
        ("battery_charge_kw", np.inf, "non_finite"),
        ("battery_discharge_kw", 500.01, "out_of_bounds"),
    ],
)
def test_invalid_control_reports_non_finite_and_out_of_bounds_fields(
    field, value, issue_code
):
    from models.hub_model import HubConfiguration, HubControl, validate_control

    values = {
        "battery_charge_kw": 0.0,
        "battery_discharge_kw": 0.0,
        "electrolyser_kw": 0.0,
        "fuel_cell_kw": 0.0,
        "heat_pump_kw": 0.0,
        "electric_boiler_kw": 0.0,
        "thermal_charge_kw": 0.0,
        "thermal_discharge_kw": 0.0,
        "ventilation_fraction": 0.0,
    }
    values[field] = value

    issues = validate_control(HubControl(**values), HubConfiguration())

    assert [(issue.code, issue.field) for issue in issues] == [(issue_code, field)]


def test_invalid_control_reports_simultaneous_flows_above_exact_tolerance():
    from models.hub_model import HubConfiguration, HubControl, validate_control

    control = HubControl(
        battery_charge_kw=0.0011,
        battery_discharge_kw=0.0011,
        electrolyser_kw=0.0,
        fuel_cell_kw=0.0,
        heat_pump_kw=0.0,
        electric_boiler_kw=0.0,
        thermal_charge_kw=0.0,
        thermal_discharge_kw=0.0,
        ventilation_fraction=0.0,
    )

    issues = validate_control(control, HubConfiguration())

    assert [(issue.code, issue.field) for issue in issues] == [
        ("simultaneous_charge_discharge", "battery")
    ]


def _parity_case_values():
    """Cover nominal, every state/control edge, and deterministic interiors."""
    from models.hub_model import (
        CONTROL_MODEL_NAMES,
        STATE_MODEL_NAMES,
        AssetCapabilities,
        HubConfiguration,
        control_bounds,
        operational_state_bounds,
        physical_state_bounds,
    )

    config = HubConfiguration()
    nominal_state = {
        "soc_battery_kwh": 500.0,
        "soc_hydrogen_kg": 60.0,
        "soc_thermal_kwh": 1600.0,
        "indoor_temperature_c": 19.0,
    }
    nominal_control = {
        "battery_charge_kw": 100.0,
        "battery_discharge_kw": 0.0,
        "electrolyser_kw": 50.0,
        "fuel_cell_kw": 0.0,
        "heat_pump_kw": 100.0,
        "electric_boiler_kw": 200.0,
        "thermal_charge_kw": 150.0,
        "thermal_discharge_kw": 0.0,
        "ventilation_fraction": 0.0,
    }
    nominal_exogenous = {
        "pv_kw": 80.0,
        "electric_load_kw": 600.0,
        "price_eur_per_kwh": 0.10,
        "outdoor_temperature_c": 5.0,
        "irradiance_w_per_m2": 0.0,
    }

    cases = [("nominal", config, nominal_state, nominal_control, nominal_exogenous)]
    for bound_policy, bounds in (
        ("physical", physical_state_bounds(config)),
        ("operational", operational_state_bounds(config)),
    ):
        for field_name in STATE_MODEL_NAMES:
            for edge_name, value in zip(
                ("lower", "upper"),
                bounds[field_name],
                strict=True,
            ):
                cases.append(
                    (
                        f"state-{bound_policy}-{field_name}-{edge_name}",
                        config,
                        {**nominal_state, field_name: value},
                        nominal_control,
                        nominal_exogenous,
                    )
                )
    for field_name in CONTROL_MODEL_NAMES:
        for edge_name, value in zip(
            ("lower", "upper"),
            control_bounds(config)[field_name],
            strict=True,
        ):
            cases.append(
                (
                    f"control-{field_name}-{edge_name}",
                    config,
                    nominal_state,
                    {**nominal_control, field_name: value},
                    nominal_exogenous,
                )
            )

    rng = np.random.default_rng(20260717)
    state_limits = physical_state_bounds(config)
    control_limits = control_bounds(config)
    for index in range(8):
        random_state = {
            field_name: rng.uniform(lower, upper)
            for field_name, (lower, upper) in state_limits.items()
        }
        random_control = {
            field_name: rng.uniform(lower, upper)
            for field_name, (lower, upper) in control_limits.items()
        }
        random_exogenous = {
            "pv_kw": rng.uniform(0.0, 500.0),
            "electric_load_kw": rng.uniform(0.0, 1000.0),
            "price_eur_per_kwh": rng.uniform(-0.10, 0.40),
            "outdoor_temperature_c": rng.uniform(-10.0, 35.0),
            "irradiance_w_per_m2": rng.uniform(0.0, 1000.0),
        }
        cases.append(
            (
                f"random-interior-{index}",
                config,
                random_state,
                random_control,
                random_exogenous,
            )
        )

    for name, capabilities in (
        ("no-hydrogen", AssetCapabilities(hydrogen=False)),
        ("no-thermal-store", AssetCapabilities(thermal_store=False)),
    ):
        cases.append(
            (
                name,
                HubConfiguration(capabilities=capabilities),
                nominal_state,
                nominal_control,
                nominal_exogenous,
            )
        )
    return cases


@pytest.mark.parametrize(
    ("_case_name", "config", "state_values", "control_values", "exogenous_values"),
    _parity_case_values(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_shared_hub_expressions_match_numerical_adapter(
    _case_name, config, state_values, control_values, exogenous_values
):
    """The CasADi and numerical adapters must evaluate one physical owner."""
    from casadi import Function, SX, vertcat

    from models.hub_model import (
        CONTROL_MODEL_NAMES,
        EXOGENOUS_MODEL_NAMES,
        STATE_MODEL_NAMES,
        ExogenousInputs,
        HubControl,
        HubState,
        advance_hub,
        hub_step_expressions,
    )

    state_symbol = SX.sym("state", len(STATE_MODEL_NAMES))
    control_symbol = SX.sym("control", len(CONTROL_MODEL_NAMES))
    exogenous_symbol = SX.sym("exogenous", len(EXOGENOUS_MODEL_NAMES))
    symbolic_step = hub_step_expressions(
        HubState(
            **{
                field_name: state_symbol[index]
                for index, field_name in enumerate(STATE_MODEL_NAMES)
            }
        ),
        HubControl(
            **{
                field_name: control_symbol[index]
                for index, field_name in enumerate(CONTROL_MODEL_NAMES)
            }
        ),
        ExogenousInputs(
            **{
                field_name: exogenous_symbol[index]
                for index, field_name in enumerate(EXOGENOUS_MODEL_NAMES)
            }
        ),
        config,
    )
    symbolic_function = Function(
        "shared_hub_step",
        [state_symbol, control_symbol, exogenous_symbol],
        [
            vertcat(
                *(
                    getattr(symbolic_step.successor, field_name)
                    for field_name in STATE_MODEL_NAMES
                ),
                symbolic_step.flows.grid_kw,
                symbolic_step.flows.generated_heat_kw,
                symbolic_step.flows.heat_to_air_kw,
                symbolic_step.flows.thermal_charge_margin_kw,
                symbolic_step.flows.hydrogen_production_kg_per_h,
                symbolic_step.flows.hydrogen_consumption_kg_per_h,
            )
        ],
    )

    numerical_step = advance_hub(
        HubState(**state_values),
        HubControl(**control_values),
        ExogenousInputs(**exogenous_values),
        config,
    )
    numerical_values = np.asarray(
        [
            *(getattr(numerical_step.successor, name) for name in STATE_MODEL_NAMES),
            numerical_step.flows.grid_kw,
            numerical_step.flows.generated_heat_kw,
            numerical_step.flows.heat_to_air_kw,
            numerical_step.flows.thermal_charge_margin_kw,
            numerical_step.flows.hydrogen_production_kg_per_h,
            numerical_step.flows.hydrogen_consumption_kg_per_h,
        ],
        dtype=float,
    )
    symbolic_values = np.asarray(
        symbolic_function(
            [state_values[name] for name in STATE_MODEL_NAMES],
            [control_values[name] for name in CONTROL_MODEL_NAMES],
            [exogenous_values[name] for name in EXOGENOUS_MODEL_NAMES],
        )
    ).reshape(-1)

    np.testing.assert_allclose(
        symbolic_values,
        numerical_values,
        rtol=0.0,
        atol=1e-8,
    )


def test_shared_thermal_charge_margin_is_signed_and_validated_numerically():
    from models.hub_model import (
        ExogenousInputs,
        HubConfiguration,
        HubControl,
        HubState,
        advance_hub,
        validate_flows,
    )

    state = HubState(500.0, 60.0, 1600.0, 19.0)
    exogenous = ExogenousInputs(0.0, 0.0, 0.0, 5.0, 0.0)
    feasible = advance_hub(
        state,
        HubControl(0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 0.0),
        exogenous,
    )
    infeasible = advance_hub(
        state,
        HubControl(0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 36.0, 0.0, 0.0),
        exogenous,
        HubConfiguration(),
    )

    assert feasible.flows.generated_heat_kw == pytest.approx(35.0)
    assert feasible.flows.thermal_charge_margin_kw == pytest.approx(-30.0)
    assert validate_flows(feasible.flows) == ()
    assert infeasible.flows.thermal_charge_margin_kw == pytest.approx(1.0)
    assert [issue.code for issue in validate_flows(infeasible.flows)] == [
        "thermal_charge_infeasible"
    ]


def test_legacy_wrapper_delegates_physics_and_conversions_to_shared_owners():
    """Prevent compatibility metrics from becoming a second equation owner."""
    import ast
    import inspect
    import textwrap

    import models.hub_model as hub_model

    wrapper_tree = ast.parse(
        textwrap.dedent(inspect.getsource(hub_model.hub_dynamics))
    )
    expression_tree = ast.parse(
        textwrap.dedent(inspect.getsource(hub_model.hub_step_expressions))
    )

    def called_functions(tree):
        return {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    wrapper_calls = called_functions(wrapper_tree)
    expression_calls = called_functions(expression_tree)
    wrapper_names = {
        node.id for node in ast.walk(wrapper_tree) if isinstance(node, ast.Name)
    }

    assert {"advance_hub", "_hub_conversions"} <= wrapper_calls
    assert "_hub_conversions" in expression_calls
    assert wrapper_names.isdisjoint(
        {
            "HP_COP",
            "ETA_EBOILER",
            "ETA_ELZ",
            "ETA_FC_E",
            "ETA_FC_H",
            "E_H2_LHV_KWH_KG",
        }
    )
    assert not hasattr(hub_model, "fuel_cell_outputs")


# ---------------------------------------------------------------------------
# 2. Short live MPC roll-out
# ---------------------------------------------------------------------------
def _mpc_forecast_points(price, pv, load, tout, irr):
    from control.rolling_horizon import _LegacyScenarioPoint

    timestamps = pd.date_range("2023-01-01", periods=len(price), freq="h", tz="UTC")
    return tuple(
        _LegacyScenarioPoint(
            timestamp_utc=timestamp.to_pydatetime(),
            price_eur_per_kwh=float(price_value),
            pv_kw=float(pv_value),
            electric_load_kw=float(load_value),
            outdoor_temperature_c=float(tout_value),
            irradiance_w_per_m2=float(irr_value),
        )
        for timestamp, price_value, pv_value, load_value, tout_value, irr_value in zip(
            timestamps, price, pv, load, tout, irr, strict=True
        )
    )


def _configured_mpc(horizon_steps=24):
    from control.mpc_controller import (
        MpcConfiguration,
        MpcControllerAdapter,
        build_mpc,
    )
    from models.hub_model import HubConfiguration

    config = MpcConfiguration(horizon_steps=horizon_steps)
    mpc, model = build_mpc(HubConfiguration(), config)
    adapter = MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=horizon_steps,
        configuration={"horizon_steps": horizon_steps},
        capability_policy={"battery": True},
    )
    mpc.x0 = np.asarray(list(initial_state().values())).reshape(-1, 1)
    mpc.set_initial_guess()
    return mpc, model, adapter


def test_solver_stat_storage_preserves_required_stats_and_numeric_data():
    from control.rolling_horizon import ControlDecision
    from models.hub_model import BALANCE_STATE_TOLERANCE, STATE_SCALE

    n = 25
    forecast = _mpc_forecast_points(
        np.full(n, 0.10),
        np.zeros(n),
        np.full(n, 400.0),
        np.full(n, 5.0),
        np.zeros(n),
    )
    mpc, _, adapter = _configured_mpc()
    state = initial_state()
    x0 = np.array(
        [state["SOC_bat"], state["SOC_h2"], state["SOC_tes"], state["T_in"]]
    ).reshape(-1, 1)

    decision = adapter.decide(state, forecast)

    assert isinstance(decision, ControlDecision)
    required_statistics = (
        "success",
        "return_status",
        "iter_count",
        "t_wall_total",
    )
    assert tuple(mpc.settings.store_solver_stats) == required_statistics
    assert mpc.settings.nlpsol_opts["ipopt.constr_viol_tol"] == (
        BALANCE_STATE_TOLERANCE / max(STATE_SCALE.values())
    )
    for statistic in required_statistics:
        stored = getattr(mpc.data, statistic)
        if statistic in mpc.solver_stats:
            assert stored.shape[0] == 1
            assert stored[-1, 0] == mpc.solver_stats[statistic]
        else:
            assert stored.shape[0] == 0
    np.testing.assert_allclose(mpc.data._x[-1], x0.reshape(-1))
    np.testing.assert_allclose(
        mpc.data._u[-1],
        np.array([decision.control[name] for name in input_bounds()]),
    )


def test_mpc_builds_steps_and_respects_balance():
    """A few closed-loop MPC steps solve and yield balance-feasible controls."""
    from control.rolling_horizon import ControlDecision
    from models.hub_model import HubState
    n = 30
    rng = np.random.default_rng(0)
    price = 0.05 + 0.05 * np.sin(np.linspace(0, 6, n)) + 0.01 * rng.standard_normal(n)
    pv = np.zeros(n)
    load = np.full(n, 400.0)
    tout = np.full(n, 5.0)
    irr = np.zeros(n)

    forecast = _mpc_forecast_points(price, pv, load, tout, irr)
    mpc, _, adapter = _configured_mpc()
    x = initial_state()

    ib = input_bounds()
    for k in range(5):
        stable_state = (
            x
            if isinstance(x, HubState)
            else HubState(
                soc_battery_kwh=x["SOC_bat"],
                soc_hydrogen_kg=x["SOC_h2"],
                soc_thermal_kwh=x["SOC_tes"],
                indoor_temperature_c=x["T_in"],
            )
        )
        decision = adapter.decide(stable_state, forecast[k : k + 25])
        assert isinstance(decision, ControlDecision)
        if k == 0:
            configured_statistics = (
                "success",
                "return_status",
                "iter_count",
                "t_wall_total",
            )
            assert tuple(mpc.settings.store_solver_stats) == configured_statistics
            for statistic in configured_statistics:
                expected_rows = 1 if statistic in mpc.solver_stats else 0
                assert getattr(mpc.data, statistic).shape[0] == expected_rows
            assert mpc.data.return_status[-1, 0] == "Solve_Succeeded"
        u = {name: float(decision.control[name]) for name in ib}
        for name, (lo, hi) in ib.items():
            # 1e-4 tolerance absorbs IPOPT's constraint-satisfaction noise
            assert lo - 1e-4 <= u[name] <= hi + 1e-4, f"{name} out of bounds: {u[name]}"
        p = {"P_pv": pv[k], "P_load": load[k], "price": price[k],
             "T_out": tout[k], "G_Wm2": irr[k]}
        x, m = hub_dynamics(x, u, p)
        assert m["elec_residual_kW"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Shared physical-invariant suite — used by BOTH the regenerated integration run and
# the committed-CSV checks, so they assert exactly the same things.
# ---------------------------------------------------------------------------
def operating_rows(df):
    """Drop the terminal-state row (NaN exogenous inputs); keep true operating hours."""
    return df[df["price_EUR_kWh"].notna()]


def assert_physical_invariants(df):
    op = operating_rows(df)
    # electricity balance closes exactly (slack-bus construction)
    assert df["elec_residual_kW"].abs().max() < 1e-6
    # no store charges and discharges in the same operating hour (battery, TES, H2)
    assert ((op.u_P_bat_ch > 1.0) & (op.u_P_bat_dis > 1.0)).sum() == 0
    assert ((op.u_Q_tes_ch > 1.0) & (op.u_Q_tes_dis > 1.0)).sum() == 0
    assert ((op.u_P_elz > 1.0) & (op.u_P_fc > 1.0)).sum() == 0
    # storage within physical limits [0, capacity]
    for col, cap in [("SOC_bat_kWh", BAT_CAPACITY_KWH), ("SOC_h2_kg", H2_CAPACITY_KG),
                     ("SOC_tes_kWh", TES_CAPACITY_KWH)]:
        assert df[col].min() >= -1.0, f"{col} below 0"
        assert df[col].max() <= cap + 1.0, f"{col} above capacity"
    # temperature within hard safety bounds
    assert df.T_in_C.min() >= T_HARD_MIN_C - 1e-6
    assert df.T_in_C.max() <= T_HARD_MAX_C + 1e-6


# ---------------------------------------------------------------------------
# 2b. Fast deterministic end-to-end integration (does NOT rely on committed CSVs)
# ---------------------------------------------------------------------------
def test_integration_short_run_full_invariants_and_mpc_not_worse():
    """Regenerate a short real-data window for both controllers and assert the SAME
    physical-invariant suite used on committed results, plus MPC <= baseline cost."""
    from control.rolling_horizon import ValidRun, load_data, run_simulation
    df = load_data(start_month=1, n_days=2)          # deterministic 2-day winter window
    base_run = run_simulation(df, mode="baseline")
    mpc_run = run_simulation(df, mode="mpc")
    assert isinstance(base_run, ValidRun)
    assert isinstance(mpc_run, ValidRun)
    base = base_run.to_frame()
    mpc = mpc_run.to_frame()
    assert_physical_invariants(base)
    assert_physical_invariants(mpc)
    assert mpc["grid_cost_EUR"].sum() <= base["grid_cost_EUR"].sum() + 1e-6


# ---------------------------------------------------------------------------
# 3. Invariants on saved simulation results
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results():
    b, m = RESULTS / "baseline_results.csv", RESULTS / "mpc_results.csv"
    if not (b.exists() and m.exists()):
        pytest.skip("Run `python3 control/rolling_horizon.py` first to generate results.")
    return pd.read_csv(b), pd.read_csv(m)


def test_results_physical_invariants(results):
    for d in results:
        assert_physical_invariants(d)


def test_results_terminal_row_schema(results):
    """Each results frame has exactly one terminal row, flagged both by is_terminal
    and by NaN exogenous price (so downstream code can filter it reliably)."""
    for d in results:
        assert "is_terminal" in d.columns
        is_term = d["is_terminal"].astype(str).str.strip().str.lower().isin(["true", "1"])
        assert is_term.sum() == 1
        assert (is_term.values == d["price_EUR_kWh"].isna().values).all()


def test_mpc_respects_operational_soc_bounds(results):
    """The MPC enforces operational SOC reserves (10-90%) as hard constraints.

    (The naive baseline does not — e.g. its TES slowly drains via standing
    losses below the operational floor, which is physically valid.)
    """
    _, m = results
    sb = state_bounds()
    tol = 1.0
    assert m.SOC_bat_kWh.min() >= sb["SOC_bat"][0] - tol
    assert m.SOC_bat_kWh.max() <= sb["SOC_bat"][1] + tol
    assert m.SOC_h2_kg.min() >= sb["SOC_h2"][0] - tol
    assert m.SOC_h2_kg.max() <= sb["SOC_h2"][1] + tol
    assert m.SOC_tes_kWh.min() >= sb["SOC_tes"][0] - tol
    assert m.SOC_tes_kWh.max() <= sb["SOC_tes"][1] + tol


def test_mpc_beats_or_matches_baseline(results):
    b, m = results
    assert m["grid_cost_EUR"].sum() <= b["grid_cost_EUR"].sum()


# ---------------------------------------------------------------------------
# 4. Published-artifact consistency and accounting bounds
# ---------------------------------------------------------------------------
def test_summary_consistent_with_scenario_csvs():
    """Characterize only the committed pre-migration grid-only summary.

    Task 14 deletes this test and its private accounting adapter when regenerated
    artifacts use valid Run Bundles and the current Evaluation Policy.
    """
    scen = RESULTS / "scenarios"
    if not (scen / "summary.csv").exists():
        pytest.skip("No scenario summary committed yet.")
    import accounting
    from accounting import (
        _legacy_grid_inventory_adjusted_cost,
        saving_pct,
        stored_equiv_kwh,
    )
    assert not hasattr(accounting, "inventory_adjusted_cost")
    x0 = initial_state()
    init_eq = stored_equiv_kwh(x0["SOC_bat"], x0["SOC_h2"], x0["SOC_tes"])
    summ = pd.read_csv(scen / "summary.csv")
    for _, row in summ.iterrows():
        b = pd.read_csv(scen / f"{row.scenario}_baseline.csv")
        m = pd.read_csv(scen / f"{row.scenario}_mpc.csv")
        settle = m["price_EUR_kWh"].mean()
        assert b["grid_cost_EUR"].sum() == pytest.approx(row.baseline_eur, abs=1.0)
        assert m["grid_cost_EUR"].sum() == pytest.approx(row.mpc_eur, abs=1.0)
        assert _legacy_grid_inventory_adjusted_cost(
            b, init_eq, settle
        ) == pytest.approx(row.baseline_adj_eur, abs=1.0)
        assert _legacy_grid_inventory_adjusted_cost(
            m, init_eq, settle
        ) == pytest.approx(row.mpc_adj_eur, abs=1.0)
        assert saving_pct(b["grid_cost_EUR"].sum(), m["grid_cost_EUR"].sum()) == pytest.approx(row.saving_pct, abs=0.2)


def test_import_fee_smoothing_error_is_bounded():
    """The MPC's smooth import volume tracks the exact max(0, P_grid) used by the plant
    to within the documented epsilon (worst case 0.5 kW at P_grid=0), so the solver
    objective stays close to the realised/reported cost."""
    P = np.linspace(-2000.0, 2000.0, 4001)
    smooth = 0.5 * (P + np.sqrt(P**2 + 1.0))   # must match mpc_controller IMPORT_SMOOTH_EPS2
    exact = np.maximum(0.0, P)
    assert np.max(np.abs(smooth - exact)) <= 0.5 + 1e-9
