"""
Rolling-horizon simulation loop for the greenhouse energy hub MPC.

Workflow
--------
1. Load aligned PV, weather, price and electrical-demand data for a window.
2. Initialise hub state (battery, H2, TES, indoor temperature).
3. At each timestep k:
   a. (MPC) solve the N-step optimisation with perfect-foresight forecasts,
      apply the first control action; (baseline) apply the rule-based action.
   b. Advance the plant via hub_dynamics().
   c. Record states, controls, costs and diagnostics.
4. Run the identical loop for both controllers and compare total grid cost.

Heat is implicit: there is no prescribed heat-demand series. Both controllers
must keep the greenhouse temperature inside the comfort band by supplying heat
(heat pump, e-boiler, fuel-cell heat, TES) and opening ventilation.

Baseline controller (fair, naive, no look-ahead)
------------------------------------------------
  - A frugal thermostat: reactively heats to hold the LOWER comfort bound
    (BASELINE_TARGET_C = T_MIN + 0.5 = 16.5 degC), the cheapest in-band temperature,
    via heat pump first, then e-boiler, then TES discharge.
  - Holding the lower bound (not a 19 degC setpoint) makes the comparison fair: any
    MPC saving reflects dispatch timing/arbitrage, not simply running colder.
  - Vent fully when solar gain would push the air above the comfort band.
  - Battery charges from PV surplus, discharges when price > 0.12 EUR/kWh.
  - No hydrogen use, no thermal pre-storage, no price look-ahead.

Result schema note: each results frame has one row per simulated hour plus a final
`is_terminal=True` row carrying the true terminal state (zero controls/cost, NaN
exogenous inputs) so inventory settlement uses the real end state. Operating-step
aggregations should filter `is_terminal == False`.

Usage
-----
    python3 control/rolling_horizon.py [--days 14] [--start-month 6] [--mode both]
    # winter fortnight: --start-month 1 ;  summer fortnight: --start-month 6
"""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hub_model import (
    AssetCapabilities, ExogenousInputs, HubConfiguration, HubControl, HubFlows,
    HubState, ValidationIssue, advance_hub, hub_dynamics, hub_state_array,
    initial_state, normalize_control, state_bounds, validate_control, validate_flows,
    validate_successor, SOLVER_BOUND_TOLERANCE_KW,
    BAT_P_MAX_KW, ETA_BAT_CH, ETA_BAT_DIS,
    HP_P_MAX_KW, HP_COP, EBOILER_P_MAX_KW, ETA_EBOILER,
    TES_P_MAX_KW,
    C_AIR_KWH_K, U_EFF_KW_K, SOLAR_GAIN_FRAC, FLOOR_AREA_M2,
    GRID_IMPORT_FEE_EUR_KWH, Q_CROP_LATENT_KW, T_MIN_C, T_MAX_C, DT_H,
)
from accounting import stored_equiv_kwh, inventory_adjusted_cost, saving_pct
# NOTE: build_mpc is imported lazily inside run_simulation() so that importing this
# module (e.g. for load_data or the baseline) does not pull in the do-mpc/IPOPT stack.

# Fair baseline: a frugal thermostat that holds the LOWER comfort bound (cheapest
# myopic temperature), so any MPC saving reflects dispatch timing/arbitrage rather
# than simply running the greenhouse colder than a 19 degC setpoint.
BASELINE_TARGET_C = T_MIN_C + 0.5

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"   # created in main(), not at import time

# Decision variables solved by the MPC (P_grid is the derived slack bus, not a control)
INPUT_NAMES = ["P_bat_ch", "P_bat_dis", "P_elz", "P_fc", "P_hp",
               "P_eboiler", "Q_tes_ch", "Q_tes_dis", "vent"]

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _read_only_mapping(
    values: Mapping[str, JSONValue],
) -> Mapping[str, JSONValue]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class _LegacyScenarioPoint:
    timestamp_utc: datetime
    price_eur_per_kwh: float
    pv_kw: float
    electric_load_kw: float
    outdoor_temperature_c: float
    irradiance_w_per_m2: float


@dataclass(frozen=True)
class _LegacyScenario:
    """Small DataFrame bridge retained only until the Scenario migration task."""

    name: str
    operating_start: datetime
    operating_end: datetime
    forecast_end: datetime
    forecast_horizon_capacity_steps: int
    step_duration: timedelta
    operating_step_count: int
    points: tuple[_LegacyScenarioPoint, ...]
    provenance: tuple[object, ...] = ()

    def forecast_view(
        self, operating_step: int, horizon_steps: int
    ) -> tuple[_LegacyScenarioPoint, ...]:
        required = horizon_steps + 1
        view = self.points[operating_step : operating_step + required]
        if view and len(view) < required:
            # Preserve the legacy end-of-frame TVP clamp. Task 5 replaces this
            # temporary bridge with strict Scenario coverage semantics.
            view = (*view, *((view[-1],) * (required - len(view))))
        return tuple(view)


@dataclass(frozen=True)
class DecisionDiagnostics:
    adapter: str
    decision_status: str
    solver_success: bool | None
    solver_return_status: str | None
    solver_iterations: int | None
    solver_wall_seconds: float | None
    forecast_start_utc: datetime | None
    forecast_end_utc: datetime | None
    terminal_electric_value_eur_per_kwh: float | None
    terminal_heat_value_eur_per_kwhth: float | None


@dataclass(frozen=True)
class ControlDecision:
    control: HubControl
    diagnostics: DecisionDiagnostics


@dataclass(frozen=True)
class ControllerFailure:
    code: str
    message: str
    diagnostics: DecisionDiagnostics


class ControllerAdapter(Protocol):
    name: str
    configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    forecast_horizon_steps: int
    requires_operational_storage_bounds: bool

    def decide(
        self,
        state: HubState,
        forecast: tuple[_LegacyScenarioPoint, ...],
    ) -> ControlDecision | ControllerFailure:
        raise NotImplementedError


@dataclass(frozen=True)
class ValidationReport:
    complete: bool
    valid: bool
    checked_operating_steps: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class OperatingRecord:
    operating_step: int
    timestamp_utc: datetime
    start_state: HubState
    control: HubControl
    exogenous: ExogenousInputs
    reached_state: HubState
    flows: HubFlows


@dataclass(frozen=True)
class ValidRun:
    scenario: _LegacyScenario
    controller_name: str
    controller_configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    hub_configuration: HubConfiguration
    initial_state: HubState
    records: tuple[OperatingRecord, ...]
    controller_diagnostics: tuple[DecisionDiagnostics, ...]
    terminal_state: HubState
    validation: ValidationReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_configuration",
            _read_only_mapping(self.controller_configuration),
        )
        object.__setattr__(
            self,
            "capability_policy",
            _read_only_mapping(self.capability_policy),
        )

    def to_frame(self) -> pd.DataFrame:
        """Explicit legacy serialization, available only for a Valid Run."""
        return _valid_run_to_frame(self)


@dataclass(frozen=True)
class InvalidRun:
    scenario: _LegacyScenario
    controller_name: str
    controller_configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    hub_configuration: HubConfiguration
    failed_step: int
    failure_code: str
    message: str
    partial_records: tuple[OperatingRecord, ...]
    controller_diagnostics: tuple[DecisionDiagnostics, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_configuration",
            _read_only_mapping(self.controller_configuration),
        )
        object.__setattr__(
            self,
            "capability_policy",
            _read_only_mapping(self.capability_policy),
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _key_by_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a frame by an explicit (month, day, hour) key, flooring sub-hour stamps.

    PVGIS timestamps fall on HH:11 (solar-time offset); flooring to the hour and
    keying by (month, day, hour) makes the price/PV/demand alignment explicit and
    robust to the cross-year (and leap-day) mismatch, instead of relying on row order.
    """
    f = frame.copy()
    f.index = f.index.floor("h")
    f["_key"] = list(zip(f.index.month, f.index.day, f.index.hour))
    return f.drop_duplicates("_key").set_index("_key")


def load_data(start_month: int = 6, n_days: int = 14) -> pd.DataFrame:
    """
    Load PV/weather (PVGIS) and price (energy-charts) data and align them on an
    explicit (month, day, hour) key for the requested window. PV/weather and prices
    may come from different years (and PVGIS 2020 is a leap year); keying by calendar
    hour — not array position — makes that alignment explicit. Feb 29 has no price
    counterpart in a non-leap price year and is simply dropped by the key join.
    """
    pv = pd.read_csv(DATA_DIR / "pv_profile.csv",
                     index_col="timestamp", parse_dates=True)
    prices = pd.read_csv(DATA_DIR / "grid_price_signal.csv",
                         index_col="timestamp", parse_dates=True)
    demand = pd.read_csv(DATA_DIR / "demand_profile.csv",
                         index_col="timestamp", parse_dates=True)

    price_year = prices.index[0].year
    start = pd.Timestamp(f"{price_year}-{start_month:02d}-01", tz="UTC")
    end = start + pd.Timedelta(days=n_days)
    prices_window = prices.loc[start:end].iloc[:-1]

    pv_k, dem_k = _key_by_hour(pv), _key_by_hour(demand)
    keys = list(zip(prices_window.index.month, prices_window.index.day,
                    prices_window.index.hour))

    df = pd.DataFrame(index=prices_window.index)
    df["price_EUR_kWh"] = prices_window["price_EUR_kWh"].values
    df["price_EUR_MWh"] = prices_window["price_EUR_MWh"].values
    df["P_pv_kW"] = pv_k["P_kW"].reindex(keys).values * 500.0   # scale 1 kWp -> 500 kWp
    df["G_Wm2"] = pv_k["G_Wm2"].reindex(keys).values
    df["T_out_C"] = pv_k["T2m_C"].reindex(keys).values
    df["P_elec_kW"] = dem_k["P_elec_kW"].reindex(keys).values

    df = df.dropna()
    print(f"Simulation: {df.index[0]} -> {df.index[-1]}  ({len(df)} steps)")
    print(f"  Price: {df.price_EUR_MWh.min():.1f} - {df.price_EUR_MWh.max():.1f} EUR/MWh "
          f"(negative: {(df.price_EUR_MWh < 0).sum()} h)")
    print(f"  PV peak: {df.P_pv_kW.max():.0f} kW   Elec load: "
          f"{df.P_elec_kW.min():.0f}-{df.P_elec_kW.max():.0f} kW   "
          f"T_out: {df.T_out_C.min():.1f}-{df.T_out_C.max():.1f} C")
    return df


# ---------------------------------------------------------------------------
# Baseline controller (rule-based, no look-ahead)
# ---------------------------------------------------------------------------
def baseline_control(x: dict, p: dict) -> dict:
    """Naive reactive dispatch: hold the lower comfort bound (BASELINE_TARGET_C)
    with HP/e-boiler/TES, plus a simple PV-charge / high-price-discharge battery rule."""
    sb = state_bounds()
    T_in, T_out, G = x["T_in"], p["T_out"], p["G_Wm2"]
    P_pv, P_load, price = p["P_pv"], p["P_load"], p["price"]

    Q_solar = SOLAR_GAIN_FRAC * G * FLOOR_AREA_M2 / 1000.0
    C = C_AIR_KWH_K / DT_H
    U = U_EFF_KW_K

    # Generated heat needed to reach the target this step (vents closed, no TES charge)
    Q_air_req = BASELINE_TARGET_C * (C + U) - C * T_in - U * T_out + Q_CROP_LATENT_KW
    Q_heat_req = max(0.0, Q_air_req - Q_solar)

    # Heat dispatch: heat pump (cheapest) -> e-boiler -> TES discharge
    Q_hp = min(Q_heat_req, HP_COP * HP_P_MAX_KW)
    P_hp = Q_hp / HP_COP
    rem = Q_heat_req - Q_hp
    Q_eb = min(rem, ETA_EBOILER * EBOILER_P_MAX_KW)
    P_eboiler = Q_eb / ETA_EBOILER
    rem -= Q_eb
    tes_avail = max(0.0, x["SOC_tes"] - sb["SOC_tes"][0])
    Q_tes_dis = min(rem, TES_P_MAX_KW, tes_avail)
    Q_tes_ch = 0.0

    # Ventilation: vent fully if solar gain would overheat the (unheated) greenhouse
    Q_air_novent = Q_hp + Q_eb + Q_tes_dis + Q_solar
    T_next_novent = (C * T_in + Q_air_novent + U * T_out - Q_CROP_LATENT_KW) / (C + U)
    vent = 1.0 if T_next_novent > T_MAX_C else 0.0

    P_elz = 0.0
    P_fc = 0.0

    # Battery: charge PV surplus, discharge when expensive (clamped to SOC bounds)
    P_bat_ch = 0.0
    P_bat_dis = 0.0
    headroom = max(0.0, sb["SOC_bat"][1] - x["SOC_bat"]) / (ETA_BAT_CH * DT_H)
    available = max(0.0, x["SOC_bat"] - sb["SOC_bat"][0]) * ETA_BAT_DIS / DT_H
    pv_surplus = P_pv - P_load - P_hp - P_eboiler
    if pv_surplus > 0:
        P_bat_ch = min(BAT_P_MAX_KW, pv_surplus, headroom)
    elif price > 0.12:
        P_bat_dis = min(BAT_P_MAX_KW, available)

    return {
        "P_bat_ch": P_bat_ch, "P_bat_dis": P_bat_dis,
        "P_elz": P_elz, "P_fc": P_fc,
        "P_hp": P_hp, "P_eboiler": P_eboiler,
        "Q_tes_ch": Q_tes_ch, "Q_tes_dis": Q_tes_dis,
        "vent": vent,
    }


class BaselineControllerAdapter:
    name = "baseline"
    configuration: Mapping[str, JSONValue] = _read_only_mapping(
        {"target_indoor_temperature_c": BASELINE_TARGET_C}
    )
    capability_policy: Mapping[str, JSONValue] = _read_only_mapping(
        {
            "hydrogen": False,
            "thermal_store_charging": False,
            "grid_battery_charging": False,
            "battery_discharge_price_threshold_eur_per_kwh": 0.12,
        }
    )
    forecast_horizon_steps = 0
    requires_operational_storage_bounds = False

    def decide(
        self,
        state: HubState,
        forecast: tuple[_LegacyScenarioPoint, ...],
    ) -> ControlDecision | ControllerFailure:
        point = forecast[0]
        legacy_control = baseline_control(
            state,
            {
                "P_pv": point.pv_kw,
                "P_load": point.electric_load_kw,
                "price": point.price_eur_per_kwh,
                "T_out": point.outdoor_temperature_c,
                "G_Wm2": point.irradiance_w_per_m2,
            },
        )
        stable_values = {
            field_name: legacy_control[model_name]
            for field_name, model_name in {
                "battery_charge_kw": "P_bat_ch",
                "battery_discharge_kw": "P_bat_dis",
                "electrolyser_kw": "P_elz",
                "fuel_cell_kw": "P_fc",
                "heat_pump_kw": "P_hp",
                "electric_boiler_kw": "P_eboiler",
                "thermal_charge_kw": "Q_tes_ch",
                "thermal_discharge_kw": "Q_tes_dis",
                "ventilation_fraction": "vent",
            }.items()
        }
        return ControlDecision(
            control=HubControl(**stable_values),
            diagnostics=DecisionDiagnostics(
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
            ),
        )


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
def _timestamp_utc(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.to_pydatetime()


def _legacy_scenario_from_frame(
    frame: pd.DataFrame, forecast_horizon_steps: int
) -> _LegacyScenario:
    points = tuple(
        _LegacyScenarioPoint(
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
        raise ValueError("legacy simulation frame must contain at least one operating step")
    duration = timedelta(hours=DT_H)
    return _LegacyScenario(
        name="legacy_frame",
        operating_start=points[0].timestamp_utc,
        operating_end=points[-1].timestamp_utc + duration,
        forecast_end=points[-1].timestamp_utc,
        forecast_horizon_capacity_steps=forecast_horizon_steps,
        step_duration=duration,
        operating_step_count=len(points),
        points=points,
    )


def _invalid_run(
    scenario: _LegacyScenario,
    controller: ControllerAdapter,
    hub_config: HubConfiguration,
    failed_step: int,
    failure_code: str,
    message: str,
    records: list[OperatingRecord],
    diagnostics: list[DecisionDiagnostics],
) -> InvalidRun:
    return InvalidRun(
        scenario=scenario,
        controller_name=controller.name,
        controller_configuration=dict(controller.configuration),
        capability_policy=dict(controller.capability_policy),
        hub_configuration=hub_config,
        failed_step=failed_step,
        failure_code=failure_code,
        message=message,
        partial_records=tuple(records),
        controller_diagnostics=tuple(diagnostics),
    )


def _issue_message(issues: tuple[ValidationIssue, ...]) -> str:
    return "; ".join(
        f"{issue.code} [{issue.field}]: {issue.message}" for issue in issues
    )


def _exogenous_from_point(point: _LegacyScenarioPoint) -> ExogenousInputs:
    exogenous = ExogenousInputs(
        pv_kw=float(point.pv_kw),
        electric_load_kw=float(point.electric_load_kw),
        price_eur_per_kwh=float(point.price_eur_per_kwh),
        outdoor_temperature_c=float(point.outdoor_temperature_c),
        irradiance_w_per_m2=float(point.irradiance_w_per_m2),
    )
    if not all(np.isfinite(float(value)) for value in exogenous.values()):
        raise ValueError("scenario point contains a non-finite exogenous value")
    return exogenous


def simulate_run(
    scenario: _LegacyScenario,
    controller: ControllerAdapter,
    hub_config: HubConfiguration,
) -> ValidRun | InvalidRun:
    start = initial_state(hub_config)
    state = start
    records: list[OperatingRecord] = []
    diagnostics: list[DecisionDiagnostics] = []

    for operating_step in range(scenario.operating_step_count):
        try:
            forecast = scenario.forecast_view(
                operating_step, controller.forecast_horizon_steps
            )
            if not forecast:
                raise ValueError("controller forecast is empty")
        except Exception as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "scenario_error",
                str(exc),
                records,
                diagnostics,
            )

        # Gate 1: request a decision and reject controller failure immediately.
        try:
            decision = controller.decide(state, forecast)
        except Exception as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "controller_error",
                str(exc),
                records,
                diagnostics,
            )
        if isinstance(decision, ControllerFailure):
            diagnostics.append(decision.diagnostics)
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                decision.code,
                decision.message,
                records,
                diagnostics,
            )
        if not isinstance(decision, ControlDecision):
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "controller_schema_error",
                "controller returned neither ControlDecision nor ControllerFailure",
                records,
                diagnostics,
            )
        diagnostics.append(decision.diagnostics)

        # Gate 2: validate and normalize the returned control before physics.
        control_issues = validate_control(
            decision.control,
            hub_config,
            tolerance=SOLVER_BOUND_TOLERANCE_KW,
        )
        if control_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_control",
                _issue_message(control_issues),
                records,
                diagnostics,
            )
        control = normalize_control(
            decision.control,
            hub_config,
            zero_small_flows=False,
        )

        # Gate 3: evaluate existing hub physics only after the control is accepted.
        try:
            exogenous = _exogenous_from_point(forecast[0])
            step = advance_hub(state, control, exogenous, hub_config)
        except Exception as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "physics_error",
                str(exc),
                records,
                diagnostics,
            )

        # Gate 4: validate reached flows and state before recording or advancing.
        flow_issues = validate_flows(step.flows)
        if flow_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_flows",
                _issue_message(flow_issues),
                records,
                diagnostics,
            )
        successor_issues = validate_successor(
            step.successor,
            hub_config,
            controller.requires_operational_storage_bounds,
        )
        if successor_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_successor",
                _issue_message(successor_issues),
                records,
                diagnostics,
            )

        records.append(
            OperatingRecord(
                operating_step=operating_step,
                timestamp_utc=forecast[0].timestamp_utc,
                start_state=state,
                control=control,
                exogenous=exogenous,
                reached_state=step.successor,
                flows=step.flows,
            )
        )
        state = step.successor

    return ValidRun(
        scenario=scenario,
        controller_name=controller.name,
        controller_configuration=dict(controller.configuration),
        capability_policy=dict(controller.capability_policy),
        hub_configuration=hub_config,
        initial_state=start,
        records=tuple(records),
        controller_diagnostics=tuple(diagnostics),
        terminal_state=state,
        validation=ValidationReport(
            complete=True,
            valid=True,
            checked_operating_steps=len(records),
            issues=(),
        ),
    )


def _valid_run_to_frame(run: ValidRun) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in run.records:
        serialized_control = normalize_control(
            record.control,
            run.hub_configuration,
        )
        grid_kw = float(record.flows.grid_kw)
        price = float(record.exogenous.price_eur_per_kwh)
        reached_temperature = float(record.reached_state.indoor_temperature_c)
        row = {
            "timestamp": pd.Timestamp(record.timestamp_utc),
            "SOC_bat_kWh": float(record.start_state.soc_battery_kwh),
            "SOC_h2_kg": float(record.start_state.soc_hydrogen_kg),
            "SOC_tes_kWh": float(record.start_state.soc_thermal_kwh),
            "T_in_C": float(record.start_state.indoor_temperature_c),
            **{
                f"u_{name}": float(value)
                for name, value in serialized_control.items()
            },
            "u_P_grid": grid_kw,
            "P_pv_kW": float(record.exogenous.pv_kw),
            "P_load_kW": float(record.exogenous.electric_load_kw),
            "price_EUR_kWh": price,
            "T_out_C": float(record.exogenous.outdoor_temperature_c),
            "G_Wm2": float(record.exogenous.irradiance_w_per_m2),
            "grid_cost_EUR": (
                grid_kw * price
                + GRID_IMPORT_FEE_EUR_KWH * max(0.0, grid_kw)
            )
            * DT_H,
            "elec_residual_kW": 0.0,
            "Q_air_kW": float(record.flows.heat_to_air_kw),
            "T_violation_C": max(0.0, reached_temperature - T_MAX_C)
            + max(0.0, T_MIN_C - reached_temperature),
            "is_terminal": False,
        }
        rows.append(row)

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
            "elec_residual_kW": 0.0,
            "Q_air_kW": np.nan,
            "T_violation_C": 0.0,
            "is_terminal": True,
        }
    )
    return pd.DataFrame(rows).set_index("timestamp")


def run_simulation(
    df: pd.DataFrame, mode: str = "mpc", **mpc_kwargs
) -> ValidRun | InvalidRun:
    """Compatibility wrapper from a legacy frame to the stable Run outcome union."""
    if mode not in ("mpc", "baseline"):
        raise ValueError(f"Unknown mode: {mode}")

    hub_config = HubConfiguration()
    if mode == "baseline":
        controller: ControllerAdapter = BaselineControllerAdapter()
        scenario = _legacy_scenario_from_frame(df, controller.forecast_horizon_steps)
        return simulate_run(scenario, controller, hub_config)

    from control.mpc_controller import MpcControllerAdapter, N_HORIZON, build_mpc

    horizon_steps = int(mpc_kwargs.get("n_horizon", N_HORIZON))
    print(
        f"\nBuilding MPC controller (horizon={horizon_steps}h, "
        f"{', '.join(key for key, value in mpc_kwargs.items() if value) or 'full'})..."
    )
    mpc, _ = build_mpc(
        price_forecast=df["price_EUR_kWh"].to_numpy(),
        pv_forecast=df["P_pv_kW"].to_numpy(),
        load_elec_forecast=df["P_elec_kW"].to_numpy(),
        temp_out_forecast=df["T_out_C"].to_numpy(),
        irr_forecast=df["G_Wm2"].to_numpy(),
        **mpc_kwargs,
    )
    mpc.x0 = hub_state_array(initial_state(hub_config))
    mpc.set_initial_guess()
    controller = MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=horizon_steps,
        configuration={"horizon_steps": horizon_steps, **mpc_kwargs},
        capability_policy={
            "battery": True,
            "hydrogen": not bool(mpc_kwargs.get("disable_h2", False)),
            "thermal_store": not bool(mpc_kwargs.get("disable_tes", False)),
        },
    )
    scenario = _legacy_scenario_from_frame(df, horizon_steps)
    print("MPC controller ready.\n")
    return simulate_run(scenario, controller, hub_config)


def scenario_tag(start_month: int, n_days: int) -> str:
    """Human-readable scenario label, e.g. 'winter_m01_14d'."""
    season = {12: "winter", 1: "winter", 2: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}[start_month]
    return f"{season}_m{start_month:02d}_{n_days}d"


def update_summary(summary_path: Path, row: dict):
    """Append/replace this scenario's row in an auditable summary table."""
    cols = list(row.keys())
    if summary_path.exists():
        table = pd.read_csv(summary_path)
        table = table[table["scenario"] != row["scenario"]]   # replace if rerun
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row], columns=cols)
    table.sort_values("scenario").to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Greenhouse energy hub MPC rolling-horizon simulation.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-month", type=int, default=6)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["mpc", "baseline", "both"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    scen_dir = RESULTS_DIR / "scenarios"
    scen_dir.mkdir(exist_ok=True)
    tag = scenario_tag(args.start_month, args.days)

    print("=" * 65)
    print(f"  Greenhouse Energy Hub MPC - {tag}")
    print("  Location: Westland, Netherlands")
    print("=" * 65)

    df = load_data(start_month=args.start_month, n_days=args.days)
    results = {}

    for i, mode in enumerate(["baseline", "mpc"]):
        if args.mode not in (mode, "both"):
            continue
        print(f"\n[{i+1}/2] {mode.upper()}...")
        outcome = run_simulation(df, mode=mode)
        if not isinstance(outcome, ValidRun):
            print(
                f"  INVALID RUN at step {outcome.failed_step}: "
                f"{outcome.failure_code}: {outcome.message}"
            )
            continue
        res = outcome.to_frame()
        results[mode] = res
        res.to_csv(RESULTS_DIR / f"{mode}_results.csv")          # canonical (latest run)
        res.to_csv(scen_dir / f"{tag}_{mode}.csv")               # scenario archive

    if "baseline" in results and "mpc" in results:
        x0 = initial_state()
        init_eq = stored_equiv_kwh(x0["SOC_bat"], x0["SOC_h2"], x0["SOC_tes"])
        settle = results["mpc"]["price_EUR_kWh"].mean()
        base, mpc_c = (results["baseline"]["grid_cost_EUR"].sum(),
                       results["mpc"]["grid_cost_EUR"].sum())
        base_adj = inventory_adjusted_cost(results["baseline"], init_eq, settle)
        mpc_adj = inventory_adjusted_cost(results["mpc"], init_eq, settle)
        bv, mv = (results["baseline"]["T_violation_C"].sum(),
                  results["mpc"]["T_violation_C"].sum())

        print("\n" + "=" * 65)
        print(f"  RESULTS SUMMARY - {tag}")
        print("=" * 65)
        print(f"  Baseline grid cost      : EUR {base:>9.2f}   T-band viol {bv:>6.1f} degC.h")
        print(f"  MPC grid cost           : EUR {mpc_c:>9.2f}   T-band viol {mv:>6.1f} degC.h")
        print(f"  Saving (raw grid cost)  : EUR {base - mpc_c:>9.2f}   ({saving_pct(base, mpc_c):+.1f}%)")
        print(f"  Saving (inventory-adj.) : EUR {base_adj - mpc_adj:>9.2f}   ({saving_pct(base_adj, mpc_adj):+.1f}%)")
        print("=" * 65)

        update_summary(scen_dir / "summary.csv", {
            "scenario": tag, "window_start": str(df.index[0].date()), "days": args.days,
            "baseline_eur": round(base, 1), "mpc_eur": round(mpc_c, 1),
            "saving_pct": round(saving_pct(base, mpc_c), 2),
            "baseline_adj_eur": round(base_adj, 1), "mpc_adj_eur": round(mpc_adj, 1),
            "adj_saving_pct": round(saving_pct(base_adj, mpc_adj), 2),
            "base_viol_Ch": round(bv, 1), "mpc_viol_Ch": round(mv, 1),
        })
        print(f"  Scenario archive -> {scen_dir}/{tag}_*.csv  | summary -> {scen_dir}/summary.csv")


if __name__ == "__main__":
    main()
