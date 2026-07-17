"""
Rolling-horizon simulation loop for the greenhouse energy hub MPC.

Workflow
--------
1. Load aligned PV, weather, price and electrical-demand data for a window.
2. Initialise hub state (battery, H2, TES, indoor temperature).
3. At each timestep k:
   a. (MPC) solve the N-step optimisation with perfect-foresight forecasts,
      apply the first control action; (baseline) apply the rule-based action.
   b. Advance the plant through the shared numerical hub adapter.
   c. Record states, controls, costs and diagnostics.
4. Run the identical loop for both controllers and apply one named scorecard.

Heat is implicit: there is no prescribed heat-demand series. Both controllers
must keep the greenhouse temperature inside the comfort band by supplying heat
(heat pump, e-boiler, fuel-cell heat, TES) and opening ventilation.

Baseline controller (limited capability, naive, no look-ahead)
---------------------------------------------------------------
  - A frugal thermostat: reactively heats to hold the LOWER comfort bound
    (BASELINE_TARGET_C = T_MIN + 0.5 = 16.5 degC), the cheapest in-band temperature,
    via heat pump first, then e-boiler, then TES discharge.
  - Holding the lower bound (not a 19 degC setpoint) aligns the comfort target, while
    its exact capability limits remain explicit in every comparison.
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
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from numbers import Integral, Real
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
    HubState, HubStep, ValidationIssue, advance_hub, hub_dynamics, hub_state_array,
    initial_state, normalize_control, state_bounds, validate_control, validate_flows,
    validate_successor, SOLVER_BOUND_TOLERANCE_KW,
    BAT_P_MAX_KW, ETA_BAT_CH, ETA_BAT_DIS,
    HP_P_MAX_KW, HP_COP, EBOILER_P_MAX_KW, ETA_EBOILER,
    TES_P_MAX_KW,
    C_AIR_KWH_K, U_EFF_KW_K, SOLAR_GAIN_FRAC, FLOOR_AREA_M2,
    GRID_IMPORT_FEE_EUR_KWH, Q_CROP_LATENT_KW, T_MIN_C, T_MAX_C, DT_H,
)
from accounting import (
    DEFAULT_EVALUATION_POLICY,
    EvaluationPolicy,
    evaluate_run,
    saving_percent,
)
from scenarios import (
    Scenario,
    ScenarioCoverageError,
    ScenarioPoint,
    ScenarioValidationError,
    build_scenario,
)
# NOTE: build_mpc is imported lazily inside run_simulation() so that importing this
# module (e.g. for load_data or the baseline) does not pull in the do-mpc/IPOPT stack.

# Limited-capability Baseline: a frugal thermostat that holds the lower comfort
# bound. Its exact missing capabilities are declared by BaselineControllerAdapter.
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
    def freeze(value: object) -> object:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {str(key): freeze(item) for key, item in value.items()}
            )
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return value

    return freeze(values)


# Temporary import aliases keep pre-migration characterization helpers importable;
# both names resolve to the Scenario Module's real immutable types.
_LegacyScenarioPoint = ScenarioPoint


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
        forecast: tuple[ScenarioPoint, ...],
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
    scenario: Scenario
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
    scenario: Scenario
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
def load_data(
    start_month: int = 6,
    n_days: int = 14,
    forecast_hours: int = 24,
) -> Scenario:
    """Build the legacy month/day request through the validated Scenario path.

    The acquired 2023 price bytes begin at 2023-01-01T00:00Z, one hour after
    January 1 local midnight.  January requests therefore start January 2 local,
    the first complete local-midnight window; direct ``build_scenario`` calls for
    January 1 fail closed and never clamp or synthesize the missing instant.
    """
    if isinstance(start_month, bool) or not isinstance(start_month, Integral):
        raise ValueError("start_month must be an integer from 1 through 12")
    if not 1 <= int(start_month) <= 12:
        raise ValueError("start_month must be an integer from 1 through 12")
    if isinstance(n_days, bool) or not isinstance(n_days, Integral) or n_days <= 0:
        raise ValueError("n_days must be a positive integer")
    if (
        isinstance(forecast_hours, bool)
        or not isinstance(forecast_hours, Integral)
        or forecast_hours < 0
    ):
        raise ValueError("forecast_hours must be a nonnegative integer")

    start_day = 2 if int(start_month) == 1 else 1
    start = pd.Timestamp(
        f"2023-{int(start_month):02d}-{start_day:02d} 00:00",
        tz="Europe/Amsterdam",
    )
    scenario = build_scenario(
        name=scenario_tag(int(start_month), int(n_days)),
        operating_start=start,
        calendar_days=int(n_days),
        max_horizon_steps=int(forecast_hours),
    )
    operating_points = scenario.points[: scenario.operating_step_count]
    prices_mwh = np.array(
        [point.price_eur_per_kwh * 1000.0 for point in scenario.points]
    )
    print(
        f"Simulation: {scenario.operating_start} -> "
        f"{operating_points[-1].timestamp_utc}  "
        f"({scenario.operating_step_count} operating steps + "
        f"{scenario.forecast_horizon_capacity_steps} forecast hours)"
    )
    print(
        f"  Price: {prices_mwh.min():.1f} - {prices_mwh.max():.1f} EUR/MWh "
        f"(negative: {(prices_mwh < 0).sum()} h)"
    )
    print(
        f"  PV peak: {max(point.pv_kw for point in scenario.points):.0f} kW   "
        f"Elec load: {min(point.electric_load_kw for point in scenario.points):.0f}-"
        f"{max(point.electric_load_kw for point in scenario.points):.0f} kW   "
        f"T_out: {min(point.outdoor_temperature_c for point in scenario.points):.1f}-"
        f"{max(point.outdoor_temperature_c for point in scenario.points):.1f} C"
    )
    return scenario


# ---------------------------------------------------------------------------
# Baseline controller (rule-based, no look-ahead)
# ---------------------------------------------------------------------------
def baseline_control(
    x: Mapping[str, object],
    p: Mapping[str, object],
    hub_config: HubConfiguration = HubConfiguration(),
) -> dict[str, object]:
    """Naive reactive dispatch: hold the lower comfort bound (BASELINE_TARGET_C)
    with HP/e-boiler/TES, plus a simple PV-charge / high-price-discharge battery rule."""
    sb = state_bounds(hub_config)
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
    if hub_config.capabilities.battery and pv_surplus > 0:
        P_bat_ch = min(BAT_P_MAX_KW, pv_surplus, headroom)
    elif hub_config.capabilities.battery and price > 0.12:
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
            "hydrogen_dispatch": False,
            "thermal_store_charging": False,
            "grid_battery_charging": False,
            "battery_discharge_price_threshold_eur_per_kwh": 0.12,
        }
    )
    forecast_horizon_steps = 0
    requires_operational_storage_bounds = False

    def __init__(
        self, hub_config: HubConfiguration = HubConfiguration()
    ) -> None:
        self._hub_config = hub_config

    def decide(
        self,
        state: HubState,
        forecast: tuple[ScenarioPoint, ...],
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
            self._hub_config,
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
        raise ScenarioValidationError(
            "frame compatibility timestamps must be timezone-aware"
        )
    timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.to_pydatetime()


def _scenario_from_frame(
    frame: pd.DataFrame,
    forecast_horizon_steps: int,
    operating_step_count: int | None = None,
) -> Scenario:
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
        raise ValueError("simulation frame must contain at least one operating step")
    if operating_step_count is None:
        operating_step_count = len(points)
    if not 0 < operating_step_count <= len(points):
        raise ValueError(
            "operating_step_count must be positive and no greater than point count"
        )
    duration = timedelta(hours=DT_H)
    return Scenario(
        name="frame_compatibility",
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


def _invalid_run(
    scenario: Scenario,
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


def _validate_decision_diagnostics(
    diagnostics: object,
    controller: ControllerAdapter,
    expected_status: str,
    failure_code: str | None = None,
) -> tuple[ValidationIssue, ...]:
    """Validate controller evidence without reaching back into controller internals."""
    if not isinstance(diagnostics, DecisionDiagnostics):
        return (
            ValidationIssue(
                code="schema_error",
                field="diagnostics",
                message="controller diagnostics must be DecisionDiagnostics",
            ),
        )

    issues: list[ValidationIssue] = []

    if not isinstance(diagnostics.adapter, str) or not diagnostics.adapter:
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="adapter",
                message="adapter must be a nonempty string",
            )
        )
    elif diagnostics.adapter != controller.name:
        issues.append(
            ValidationIssue(
                code="adapter_mismatch",
                field="adapter",
                message=(
                    f"diagnostics adapter {diagnostics.adapter!r} does not match "
                    f"controller {controller.name!r}"
                ),
            )
        )

    if diagnostics.decision_status != expected_status:
        issues.append(
            ValidationIssue(
                code="branch_mismatch",
                field="decision_status",
                message=(
                    f"{type(diagnostics).__name__} status must be "
                    f"{expected_status!r}"
                ),
            )
        )

    if diagnostics.solver_success is not None and type(
        diagnostics.solver_success
    ) is not bool:
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="solver_success",
                message="solver_success must be bool or None",
            )
        )

    if diagnostics.solver_return_status is not None and not isinstance(
        diagnostics.solver_return_status, str
    ):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="solver_return_status",
                message="solver_return_status must be str or None",
            )
        )

    iterations = diagnostics.solver_iterations
    if iterations is not None and (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations < 0
    ):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="solver_iterations",
                message="solver_iterations must be a nonnegative int or None",
            )
        )

    wall_seconds = diagnostics.solver_wall_seconds
    if wall_seconds is not None and (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, Real)
        or not np.isfinite(float(wall_seconds))
        or wall_seconds < 0
    ):
        issues.append(
            ValidationIssue(
                code="non_finite",
                field="solver_wall_seconds",
                message="solver_wall_seconds must be finite, nonnegative, or None",
                actual=(
                    float(wall_seconds)
                    if isinstance(wall_seconds, Real)
                    and not isinstance(wall_seconds, bool)
                    else None
                ),
            )
        )

    forecast_start = diagnostics.forecast_start_utc
    forecast_end = diagnostics.forecast_end_utc
    if forecast_start is not None and not isinstance(forecast_start, datetime):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="forecast_start_utc",
                message="forecast_start_utc must be datetime or None",
            )
        )
    if forecast_end is not None and not isinstance(forecast_end, datetime):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="forecast_end_utc",
                message="forecast_end_utc must be datetime or None",
            )
        )
    if (forecast_start is None) != (forecast_end is None):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="forecast_window_utc",
                message="forecast timestamps must both be present or both be None",
            )
        )
    elif isinstance(forecast_start, datetime) and isinstance(forecast_end, datetime):
        try:
            reversed_window = forecast_start > forecast_end
        except TypeError:
            reversed_window = True
        if reversed_window:
            issues.append(
                ValidationIssue(
                    code="chronology_error",
                    field="forecast_window_utc",
                    message="forecast_start_utc must not follow forecast_end_utc",
                )
            )

    for field_name in (
        "terminal_electric_value_eur_per_kwh",
        "terminal_heat_value_eur_per_kwhth",
    ):
        value = getattr(diagnostics, field_name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
        ):
            issues.append(
                ValidationIssue(
                    code="non_finite",
                    field=field_name,
                    message=f"{field_name} must be finite numeric or None",
                    actual=(
                        float(value)
                        if isinstance(value, Real) and not isinstance(value, bool)
                        else None
                    ),
                )
            )

    if expected_status == "success":
        if controller.name == "mpc":
            if diagnostics.solver_success is not True:
                issues.append(
                    ValidationIssue(
                        code="branch_mismatch",
                        field="solver_success",
                        message="successful MPC decision requires solver_success=True",
                    )
                )
            if not (
                isinstance(diagnostics.solver_return_status, str)
                and diagnostics.solver_return_status.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="schema_error",
                        field="solver_return_status",
                        message=(
                            "successful MPC decision requires a nonempty "
                            "solver_return_status"
                        ),
                    )
                )
        elif controller.name == "baseline":
            for field_name in (
                "solver_success",
                "solver_return_status",
                "solver_iterations",
                "solver_wall_seconds",
            ):
                if getattr(diagnostics, field_name) is not None:
                    issues.append(
                        ValidationIssue(
                            code="branch_mismatch",
                            field=field_name,
                            message=(
                                "successful Baseline decision requires empty "
                                "solver diagnostics"
                            ),
                        )
                    )
        elif diagnostics.solver_success is False:
            issues.append(
                ValidationIssue(
                    code="branch_mismatch",
                    field="solver_success",
                    message="successful decision cannot report solver_success=False",
                )
            )
    else:
        if diagnostics.solver_success is True:
            issues.append(
                ValidationIssue(
                    code="branch_mismatch",
                    field="solver_success",
                    message="ControllerFailure cannot report solver_success=True",
                )
            )
        if controller.name == "mpc" and failure_code == "forecast_coverage":
            for field_name in (
                "solver_success",
                "solver_return_status",
                "solver_iterations",
                "solver_wall_seconds",
            ):
                if getattr(diagnostics, field_name) is not None:
                    issues.append(
                        ValidationIssue(
                            code="branch_mismatch",
                            field=field_name,
                            message=(
                                "pre-solver forecast coverage failure requires empty "
                                "solver diagnostics"
                            ),
                        )
                    )
        elif controller.name == "mpc":
            if diagnostics.solver_success is not False:
                issues.append(
                    ValidationIssue(
                        code="branch_mismatch",
                        field="solver_success",
                        message="MPC ControllerFailure requires solver_success=False",
                    )
                )
            if not (
                isinstance(diagnostics.solver_return_status, str)
                and diagnostics.solver_return_status.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="schema_error",
                        field="solver_return_status",
                        message="MPC ControllerFailure requires a nonempty return status",
                    )
                )

    return tuple(issues)


def _exogenous_from_point(point: ScenarioPoint) -> ExogenousInputs:
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
    scenario: Scenario,
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
                return _invalid_run(
                    scenario,
                    controller,
                    hub_config,
                    operating_step,
                    "forecast_coverage",
                    "controller forecast is empty",
                    records,
                    diagnostics,
                )
        except ScenarioCoverageError as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "forecast_coverage",
                str(exc),
                records,
                diagnostics,
            )
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
        if not isinstance(decision, (ControlDecision, ControllerFailure)):
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
        expected_status = (
            "success" if isinstance(decision, ControlDecision) else "failure"
        )
        diagnostics_issues = _validate_decision_diagnostics(
            decision.diagnostics,
            controller,
            expected_status,
            decision.code if isinstance(decision, ControllerFailure) else None,
        )
        if isinstance(decision.diagnostics, DecisionDiagnostics):
            diagnostics.append(decision.diagnostics)
        if diagnostics_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_diagnostics",
                _issue_message(diagnostics_issues),
                records,
                diagnostics,
            )
        if isinstance(decision, ControllerFailure):
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

        # Gate 2: absorb only accepted solver-bound noise, including residuals on
        # Disabled Asset fields, then apply the exact configured public validator.
        try:
            control = normalize_control(
                decision.control,
                hub_config,
                zero_small_flows=False,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_control",
                f"control normalization failed: {exc}",
                records,
                diagnostics,
            )
        control_issues = validate_control(
            control,
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
        if not isinstance(step, HubStep):
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "physics_schema_error",
                (
                    "advance_hub returned "
                    f"{type(step).__name__}; expected HubStep"
                ),
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
    data: Scenario | pd.DataFrame,
    mode: str = "mpc",
    hub_config: HubConfiguration = HubConfiguration(),
    evaluation_policy: EvaluationPolicy = DEFAULT_EVALUATION_POLICY,
    **mpc_kwargs: object,
) -> ValidRun | InvalidRun:
    """Run one Controller against a Scenario (or a test-only frame adapter)."""
    if mode not in ("mpc", "baseline"):
        raise ValueError(f"Unknown mode: {mode}")
    if not isinstance(evaluation_policy, EvaluationPolicy):
        raise TypeError("evaluation_policy must be an EvaluationPolicy")

    if mode == "baseline":
        if mpc_kwargs:
            unexpected = ", ".join(sorted(mpc_kwargs))
            raise TypeError(f"unexpected Baseline options: {unexpected}")
        controller: ControllerAdapter = BaselineControllerAdapter(hub_config)
        if isinstance(data, Scenario):
            scenario = data
        elif isinstance(data, pd.DataFrame):
            scenario = _scenario_from_frame(
                data,
                controller.forecast_horizon_steps,
                operating_step_count=len(data),
            )
        else:
            raise TypeError("data must be a Scenario or DataFrame")
        return simulate_run(scenario, controller, hub_config)

    from control.mpc_controller import (
        MpcConfiguration,
        MpcControllerAdapter,
        N_HORIZON,
        build_mpc,
    )

    horizon_steps = int(mpc_kwargs.get("n_horizon", N_HORIZON))
    print(
        f"\nBuilding MPC controller (horizon={horizon_steps}h, "
        f"{', '.join(key for key, value in mpc_kwargs.items() if value) or 'full'})..."
    )
    config_field_names = {field.name for field in fields(MpcConfiguration)}
    unexpected = set(mpc_kwargs) - config_field_names - {"n_horizon"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(
            f"unexpected MPC options: {names}; pass asset capabilities via "
            "hub_config"
        )
    economic_field_names = {
        "battery_wear_eur_per_kwh",
        "thermal_store_wear_eur_per_kwh",
        "electrolyser_wear_eur_per_kwh",
        "fuel_cell_wear_eur_per_kwh",
    }
    economic_overrides = economic_field_names.intersection(mpc_kwargs)
    if economic_overrides:
        names = ", ".join(sorted(economic_overrides))
        raise TypeError(
            f"MPC wear terms come from evaluation_policy, not overrides: {names}"
        )
    config_values = {
        key: value
        for key, value in mpc_kwargs.items()
        if key in config_field_names and key not in economic_field_names
    }
    config_values["horizon_steps"] = horizon_steps
    mpc_config = MpcConfiguration.from_evaluation_policy(
        evaluation_policy,
        **config_values,
    )
    mpc, _ = build_mpc(
        hub_config,
        mpc_config,
    )
    mpc.x0 = hub_state_array(initial_state(hub_config))
    mpc.set_initial_guess()
    controller = MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=horizon_steps,
        configuration=mpc_config.to_controller_metadata(),
        capability_policy=asdict(hub_config.capabilities),
    )
    if isinstance(data, Scenario):
        scenario = data
    elif isinstance(data, pd.DataFrame):
        operating_step_count = len(data) - horizon_steps
        if operating_step_count <= 0:
            raise ScenarioCoverageError(
                "frame compatibility requires Operating Steps plus explicit "
                "Forecast Coverage"
            )
        scenario = _scenario_from_frame(
            data,
            horizon_steps,
            operating_step_count=operating_step_count,
        )
    else:
        raise TypeError("data must be a Scenario or DataFrame")
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
    parser.add_argument("--disable-h2", action="store_true")
    parser.add_argument("--disable-tes", action="store_true")
    args = parser.parse_args()

    hub_config = HubConfiguration(
        capabilities=AssetCapabilities(
            hydrogen=not args.disable_h2,
            thermal_store=not args.disable_tes,
        )
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    scen_dir = RESULTS_DIR / "scenarios"
    scen_dir.mkdir(exist_ok=True)
    tag = scenario_tag(args.start_month, args.days)

    print("=" * 65)
    print(f"  Greenhouse Energy Hub MPC - {tag}")
    print("  Location: Westland, Netherlands")
    print("=" * 65)

    scenario = load_data(start_month=args.start_month, n_days=args.days)
    runs: dict[str, ValidRun] = {}

    for i, mode in enumerate(["baseline", "mpc"]):
        if args.mode not in (mode, "both"):
            continue
        print(f"\n[{i+1}/2] {mode.upper()}...")
        outcome = run_simulation(
            scenario,
            mode=mode,
            hub_config=hub_config,
            evaluation_policy=DEFAULT_EVALUATION_POLICY,
        )
        if not isinstance(outcome, ValidRun):
            print(
                f"  INVALID RUN at step {outcome.failed_step}: "
                f"{outcome.failure_code}: {outcome.message}"
            )
            continue
        res = outcome.to_frame()
        runs[mode] = outcome
        res.to_csv(RESULTS_DIR / f"{mode}_results.csv")          # canonical (latest run)
        res.to_csv(scen_dir / f"{tag}_{mode}.csv")               # scenario archive

    if "baseline" in runs and "mpc" in runs:
        baseline_report = evaluate_run(
            runs["baseline"], DEFAULT_EVALUATION_POLICY
        )
        mpc_report = evaluate_run(runs["mpc"], DEFAULT_EVALUATION_POLICY)
        baseline = baseline_report.nominal
        mpc = mpc_report.nominal

        print("\n" + "=" * 65)
        print(f"  RESULTS SUMMARY - {tag}")
        print("=" * 65)
        print(
            "  Baseline (limited capability): "
            f"Grid EUR {baseline.grid_cost_eur:>9.2f}; "
            f"Operating EUR {baseline.operating_cost_eur:>9.2f}; "
            f"Inventory-Adjusted EUR "
            f"{baseline.inventory_adjusted_cost_eur:>9.2f}; "
            f"Comfort {baseline.comfort_violation_c_h:>6.1f} C.h"
        )
        print(
            "  MPC                          : "
            f"Grid EUR {mpc.grid_cost_eur:>9.2f}; "
            f"Operating EUR {mpc.operating_cost_eur:>9.2f}; "
            f"Inventory-Adjusted EUR "
            f"{mpc.inventory_adjusted_cost_eur:>9.2f}; "
            f"Comfort {mpc.comfort_violation_c_h:>6.1f} C.h"
        )
        print(
            "  Inventory-Adjusted saving   : "
            f"EUR {baseline.inventory_adjusted_cost_eur - mpc.inventory_adjusted_cost_eur:>9.2f} "
            f"({saving_percent(baseline.inventory_adjusted_cost_eur, mpc.inventory_adjusted_cost_eur):+.1f}%)"
        )
        print("=" * 65)

        update_summary(scen_dir / "summary.csv", {
            "scenario": tag,
            "window_start": str(pd.Timestamp(scenario.operating_start).date()),
            "days": args.days,
            "baseline_grid_eur": round(baseline.grid_cost_eur, 1),
            "mpc_grid_eur": round(mpc.grid_cost_eur, 1),
            "baseline_operating_eur": round(baseline.operating_cost_eur, 1),
            "mpc_operating_eur": round(mpc.operating_cost_eur, 1),
            "baseline_inventory_adjusted_eur": round(
                baseline.inventory_adjusted_cost_eur, 1
            ),
            "mpc_inventory_adjusted_eur": round(
                mpc.inventory_adjusted_cost_eur, 1
            ),
            "inventory_adjusted_saving_pct": round(
                saving_percent(
                    baseline.inventory_adjusted_cost_eur,
                    mpc.inventory_adjusted_cost_eur,
                ),
                2,
            ),
            "baseline_comfort_violation_Ch": round(
                baseline.comfort_violation_c_h, 1
            ),
            "mpc_comfort_violation_Ch": round(
                mpc.comfort_violation_c_h, 1
            ),
            "baseline_wear_0x_inventory_adjusted_eur": round(
                baseline_report.wear_sensitivities[
                    "0x"
                ].inventory_adjusted_cost_eur,
                1,
            ),
            "baseline_wear_2x_inventory_adjusted_eur": round(
                baseline_report.wear_sensitivities[
                    "2x"
                ].inventory_adjusted_cost_eur,
                1,
            ),
            "mpc_wear_0x_inventory_adjusted_eur": round(
                mpc_report.wear_sensitivities[
                    "0x"
                ].inventory_adjusted_cost_eur,
                1,
            ),
            "mpc_wear_2x_inventory_adjusted_eur": round(
                mpc_report.wear_sensitivities[
                    "2x"
                ].inventory_adjusted_cost_eur,
                1,
            ),
        })
        print(f"  Scenario archive -> {scen_dir}/{tag}_*.csv  | summary -> {scen_dir}/summary.csv")


if __name__ == "__main__":
    main()
