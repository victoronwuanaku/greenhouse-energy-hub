"""Test-only adapters around the package interfaces.

These helpers let tests build Scenarios from small DataFrames, run a named
controller with keyword overrides, and view a ValidRun as a DataFrame. None of
this is needed by the package itself, so it lives with the tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from greenhouse_energy_hub import simulation
from greenhouse_energy_hub.evaluation import (
    DEFAULT_EVALUATION_POLICY,
    EvaluationPolicy,
    evaluate_step,
)
from greenhouse_energy_hub.hub import (
    CONTROL_MODEL_NAMES,
    DT_H,
    GRID_IMPORT_FEE_EUR_KWH,
    STATE_MODEL_NAMES,
    T_MAX_C,
    T_MIN_C,
    ExogenousInputs,
    HubConfiguration,
    HubControl,
    HubState,
    advance_hub,
    control_bounds,
    hub_state_array,
    initial_state,
    operational_state_bounds,
)
from greenhouse_energy_hub.scenarios import (
    Scenario,
    ScenarioCoverageError,
    ScenarioPoint,
    ScenarioValidationError,
    build_scenario,
)
from greenhouse_energy_hub.simulation import (
    ControllerAdapter,
    InvalidRun,
    ValidRun,
)

INPUT_NAMES = tuple(CONTROL_MODEL_NAMES.values())


def state_bounds(config: HubConfiguration = HubConfiguration()) -> dict[str, tuple[float, float]]:
    """Operational state bounds keyed by model name (SOC_bat, SOC_h2, ...)."""
    return {
        STATE_MODEL_NAMES[field]: bounds
        for field, bounds in operational_state_bounds(config).items()
    }


def input_bounds(config: HubConfiguration = HubConfiguration()) -> dict[str, tuple[float, float]]:
    """Control bounds keyed by model name (P_bat_ch, P_bat_dis, ...)."""
    return {
        CONTROL_MODEL_NAMES[field]: bounds
        for field, bounds in control_bounds(config).items()
    }


def hub_step(
    x: Mapping[str, object],
    u: Mapping[str, object],
    p: Mapping[str, object],
    config: HubConfiguration = HubConfiguration(),
) -> tuple[dict[str, float], dict[str, float]]:
    """Advance the hub one step from model-name dictionaries.

    Returns the successor state and a few derived flows/metrics, all keyed by
    model name, so physics tests can be written against plain numbers.
    """
    state = HubState(
        soc_battery_kwh=x["SOC_bat"],
        soc_hydrogen_kg=x["SOC_h2"],
        soc_thermal_kwh=x["SOC_tes"],
        indoor_temperature_c=x["T_in"],
    )
    control = HubControl(
        **{field: u[name] for field, name in CONTROL_MODEL_NAMES.items()}
    )
    exogenous = ExogenousInputs(
        pv_kw=p["P_pv"],
        electric_load_kw=p["P_load"],
        price_eur_per_kwh=p.get("price", 0.0),
        outdoor_temperature_c=p["T_out"],
        irradiance_w_per_m2=p.get("G_Wm2", 0.0),
    )
    step = advance_hub(state, control, exogenous, config)
    grid_kw = float(step.flows.grid_kw)
    reached_temperature = float(step.successor.indoor_temperature_c)
    successor = {
        name: float(getattr(step.successor, field))
        for field, name in STATE_MODEL_NAMES.items()
    }
    metrics = {
        "P_grid_kW": grid_kw,
        "tes_charge_excess_kW": max(0.0, float(step.flows.thermal_charge_margin_kw)),
        "Q_gen_kW": float(step.flows.generated_heat_kw),
        "Q_air_kW": float(step.flows.heat_to_air_kw),
        "m_h2_prod_kg_h": float(step.flows.hydrogen_production_kg_per_h),
        "m_h2_fc_kg_h": float(step.flows.hydrogen_consumption_kg_per_h),
        "grid_cost_EUR": (
            grid_kw * float(exogenous.price_eur_per_kwh)
            + GRID_IMPORT_FEE_EUR_KWH * max(0.0, grid_kw)
        )
        * DT_H,
        "T_violation_C": max(0.0, reached_temperature - T_MAX_C)
        + max(0.0, T_MIN_C - reached_temperature),
    }
    return successor, metrics


def _timestamp_utc(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ScenarioValidationError("frame timestamps must be timezone-aware")
    return timestamp.tz_convert(timezone.utc).to_pydatetime()


def scenario_from_frame(
    frame: pd.DataFrame,
    forecast_horizon_steps: int,
    operating_step_count: int | None = None,
) -> Scenario:
    """Build a Scenario from an hourly frame with the raw source column names."""
    points = tuple(
        ScenarioPoint(
            timestamp_utc=_timestamp_utc(timestamp),
            price_eur_per_kwh=float(row.price_EUR_kWh),
            pv_kw=float(row.P_pv_kW),
            electric_load_kw=float(row.P_elec_kW),
            outdoor_temperature_c=float(row.T_out_C),
            irradiance_w_per_m2=float(row.G_Wm2),
        )
        for timestamp, row in frame.iterrows()
    )
    if not points:
        raise ValueError("frame must contain at least one operating step")
    if operating_step_count is None:
        operating_step_count = len(points)
    if not 0 < operating_step_count <= len(points):
        raise ValueError(
            "operating_step_count must be positive and no greater than point count"
        )
    duration = timedelta(hours=DT_H)
    return Scenario(
        name="frame",
        operating_start=points[0].timestamp_utc,
        operating_end=points[operating_step_count - 1].timestamp_utc + duration,
        forecast_end=points[operating_step_count - 1].timestamp_utc
        + duration
        + forecast_horizon_steps * duration,
        forecast_horizon_capacity_steps=forecast_horizon_steps,
        step_duration=duration,
        operating_step_count=operating_step_count,
        points=points,
        provenance=(),
    )


def scenario_tag(start_month: int, n_days: int) -> str:
    season = {12: "winter", 1: "winter", 2: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}[start_month]
    return f"{season}_m{start_month:02d}_{n_days}d"


def load_window(
    start_month: int = 6,
    n_days: int = 14,
    forecast_hours: int = 24,
) -> Scenario:
    """Build a 2023 Scenario from the committed data by month and day count.

    The price bytes begin at 2023-01-01T00:00Z, one hour after January 1 local
    midnight, so January windows start on January 2 (the first complete
    local-midnight day).
    """
    start_day = 2 if start_month == 1 else 1
    start = pd.Timestamp(
        f"2023-{start_month:02d}-{start_day:02d} 00:00",
        tz="Europe/Amsterdam",
    )
    return build_scenario(
        name=scenario_tag(start_month, n_days),
        operating_start=start,
        calendar_days=n_days,
        max_horizon_steps=forecast_hours,
    )


def run_controller(
    data: Scenario | pd.DataFrame,
    mode: str = "mpc",
    hub_config: HubConfiguration = HubConfiguration(),
    evaluation_policy: EvaluationPolicy = DEFAULT_EVALUATION_POLICY,
    **mpc_kwargs: object,
) -> ValidRun | InvalidRun:
    """Run the named controller over a Scenario or an hourly frame."""
    if mode not in ("mpc", "baseline"):
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "baseline":
        from greenhouse_energy_hub.controllers.baseline import (
            BaselineControllerAdapter,
        )

        if mpc_kwargs:
            raise TypeError(f"unexpected Baseline options: {', '.join(sorted(mpc_kwargs))}")
        controller: ControllerAdapter = BaselineControllerAdapter(hub_config)
        scenario = (
            data
            if isinstance(data, Scenario)
            else scenario_from_frame(
                data, controller.forecast_horizon_steps, operating_step_count=len(data)
            )
        )
        return simulation.simulate_run(scenario, controller, hub_config)

    from greenhouse_energy_hub.controllers import mpc as mpc_controller

    horizon_steps = int(mpc_kwargs.pop("n_horizon", mpc_controller.N_HORIZON))
    config_field_names = {field.name for field in fields(mpc_controller.MpcConfiguration)}
    unexpected = set(mpc_kwargs) - config_field_names
    if unexpected:
        raise TypeError(f"unexpected MPC options: {', '.join(sorted(unexpected))}")
    mpc_config = mpc_controller.MpcConfiguration.from_evaluation_policy(
        evaluation_policy,
        horizon_steps=horizon_steps,
        **mpc_kwargs,
    )
    mpc, forecast_source = mpc_controller.build_mpc(hub_config, mpc_config)
    mpc.x0 = hub_state_array(initial_state(hub_config))
    mpc.set_initial_guess()
    controller = mpc_controller.MpcControllerAdapter(
        mpc=mpc,
        forecast_source=forecast_source,
        forecast_horizon_steps=horizon_steps,
        configuration=mpc_config.to_controller_metadata(),
        capability_policy=asdict(hub_config.capabilities),
    )
    if isinstance(data, Scenario):
        scenario = data
    else:
        operating_step_count = len(data) - horizon_steps
        if operating_step_count <= 0:
            raise ScenarioCoverageError(
                "frame must contain Operating Steps plus Forecast Coverage"
            )
        scenario = scenario_from_frame(data, horizon_steps, operating_step_count)
    return simulation.simulate_run(scenario, controller, hub_config)


def run_to_frame(run: ValidRun) -> pd.DataFrame:
    """One row per operating hour plus a final ``is_terminal`` row."""
    rows: list[dict[str, object]] = []
    step_hours = run.scenario.step_duration.total_seconds() / 3600.0
    for record in run.records:
        line_items = evaluate_step(record, DEFAULT_EVALUATION_POLICY, step_hours)
        rows.append(
            {
                "timestamp": pd.Timestamp(record.timestamp_utc),
                "SOC_bat_kWh": float(record.start_state.soc_battery_kwh),
                "SOC_h2_kg": float(record.start_state.soc_hydrogen_kg),
                "SOC_tes_kWh": float(record.start_state.soc_thermal_kwh),
                "T_in_C": float(record.start_state.indoor_temperature_c),
                **{f"u_{name}": float(value) for name, value in record.control.items()},
                "u_P_grid": float(record.flows.grid_kw),
                "P_pv_kW": float(record.exogenous.pv_kw),
                "P_load_kW": float(record.exogenous.electric_load_kw),
                "price_EUR_kWh": float(record.exogenous.price_eur_per_kwh),
                "T_out_C": float(record.exogenous.outdoor_temperature_c),
                "G_Wm2": float(record.exogenous.irradiance_w_per_m2),
                "grid_cost_EUR": line_items.grid_cost_eur,
                "Q_air_kW": float(record.flows.heat_to_air_kw),
                "T_violation_C": line_items.comfort_violation_c_h,
                "is_terminal": False,
            }
        )
    terminal_timestamp = (
        pd.Timestamp(rows[-1]["timestamp"]) + run.scenario.step_duration
        if rows
        else pd.Timestamp(run.scenario.operating_start)
    )
    terminal = run.terminal_state
    rows.append(
        {
            "timestamp": terminal_timestamp,
            "SOC_bat_kWh": float(terminal.soc_battery_kwh),
            "SOC_h2_kg": float(terminal.soc_hydrogen_kg),
            "SOC_tes_kWh": float(terminal.soc_thermal_kwh),
            "T_in_C": float(terminal.indoor_temperature_c),
            **{f"u_{name}": 0.0 for name in INPUT_NAMES},
            "u_P_grid": 0.0,
            "P_pv_kW": np.nan,
            "P_load_kW": np.nan,
            "price_EUR_kWh": np.nan,
            "T_out_C": np.nan,
            "G_Wm2": np.nan,
            "grid_cost_EUR": 0.0,
            "Q_air_kW": np.nan,
            "T_violation_C": 0.0,
            "is_terminal": True,
        }
    )
    return pd.DataFrame(rows).set_index("timestamp")
