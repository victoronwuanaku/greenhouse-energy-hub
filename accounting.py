"""Named evaluation policy and reproducible Run-level economic scorecards.

Realized evaluation is intentionally independent of the MPC's solver-only
regularization.  Every Controller is scored from immutable Operating Records by
the same named policy; serialized trajectory columns are compatibility output,
not an accounting input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

from models.hub_model import (
    E_H2_LHV_KWH_KG,
    ETA_BAT_DIS,
    ETA_FC_E,
    HP_COP,
    T_MAX_C,
    T_MIN_C,
    HubConfiguration,
    HubState,
)

if TYPE_CHECKING:
    from control.rolling_horizon import (
        OperatingRecord,
        ValidRun,
    )


PROVISIONAL_COEFFICIENT_STATUS = "provisional"
SETTLEMENT_RULE = "arithmetic-mean-operating-wholesale-price"


def _immutable_metadata(value: object) -> object:
    """Recursively freeze policy metadata without inheriting mutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _immutable_metadata(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_metadata(item) for item in value)
    return value


def _normal_json_primitives(value: object) -> object:
    """Copy immutable metadata into ordinary JSON dict/list/scalar primitives."""
    if isinstance(value, Mapping):
        return {
            str(key): _normal_json_primitives(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normal_json_primitives(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"metadata value {value!r} is not JSON-compatible")


def _finite_nonnegative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite and nonnegative")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return converted


@dataclass(frozen=True)
class WearCoefficients:
    battery_eur_per_kwh: float = 0.005
    thermal_store_eur_per_kwh: float = 0.0005
    electrolyser_eur_per_kwh: float = 0.002
    fuel_cell_eur_per_kwh: float = 0.002

    def __post_init__(self) -> None:
        for field_name in (
            "battery_eur_per_kwh",
            "thermal_store_eur_per_kwh",
            "electrolyser_eur_per_kwh",
            "fuel_cell_eur_per_kwh",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class EvaluationPolicy:
    name: str = "greenhouse-hub-evaluation"
    version: str = "1"
    grid_import_fee_eur_per_kwh: float = 0.025
    wear: WearCoefficients = WearCoefficients()
    sensitivity_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("EvaluationPolicy name must be nonempty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("EvaluationPolicy version must be nonempty")
        object.__setattr__(
            self,
            "grid_import_fee_eur_per_kwh",
            _finite_nonnegative(
                self.grid_import_fee_eur_per_kwh,
                "grid_import_fee_eur_per_kwh",
            ),
        )
        if not isinstance(self.wear, WearCoefficients):
            raise TypeError("wear must be WearCoefficients")
        multipliers = tuple(
            _finite_nonnegative(value, "sensitivity_multipliers")
            for value in self.sensitivity_multipliers
        )
        if len(set(multipliers)) != len(multipliers):
            raise ValueError("sensitivity_multipliers must be unique")
        if not {0.0, 1.0, 2.0}.issubset(multipliers):
            raise ValueError(
                "sensitivity_multipliers must contain mandatory 0x, 1x, and 2x"
            )
        object.__setattr__(self, "sensitivity_multipliers", multipliers)

    def to_metadata(self) -> Mapping[str, object]:
        """Return deterministic metadata backed by genuine immutable containers."""
        metadata = _immutable_metadata(
            {
                "name": self.name,
                "version": self.version,
                "grid_import_fee_eur_per_kwh": self.grid_import_fee_eur_per_kwh,
                "wear": {
                    **asdict(self.wear),
                    "coefficient_status": PROVISIONAL_COEFFICIENT_STATUS,
                },
                "sensitivity_multipliers": self.sensitivity_multipliers,
                "settlement_rule": SETTLEMENT_RULE,
                "comfort_valuation": None,
            }
        )
        assert isinstance(metadata, Mapping)
        return metadata

    def to_serializable_metadata(self) -> dict[str, object]:
        """Return a fresh ordinary JSON-primitive copy of policy metadata."""
        normalized = _normal_json_primitives(self.to_metadata())
        assert isinstance(normalized, dict)
        return normalized


DEFAULT_EVALUATION_POLICY = EvaluationPolicy()


@dataclass(frozen=True)
class StepLineItems:
    operating_step: int
    grid_cost_eur: float
    battery_wear_eur: float
    thermal_store_wear_eur: float
    electrolyser_wear_eur: float
    fuel_cell_wear_eur: float
    operating_cost_eur: float
    comfort_violation_c_h: float


@dataclass(frozen=True)
class CostSummary:
    grid_cost_eur: float
    battery_wear_eur: float
    thermal_store_wear_eur: float
    electrolyser_wear_eur: float
    fuel_cell_wear_eur: float
    operating_cost_eur: float
    inventory_settlement_eur: float
    inventory_adjusted_cost_eur: float
    comfort_violation_c_h: float


@dataclass(frozen=True)
class EvaluationReport:
    policy: EvaluationPolicy
    nominal: CostSummary
    wear_sensitivities: Mapping[str, CostSummary]
    step_line_items: tuple[StepLineItems, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wear_sensitivities",
            MappingProxyType(dict(self.wear_sensitivities)),
        )
        object.__setattr__(self, "step_line_items", tuple(self.step_line_items))


def evaluate_step(
    record: OperatingRecord,
    policy: EvaluationPolicy,
    step_hours: float,
) -> StepLineItems:
    """Recompute one Operating Record's exact realized evaluation line items."""
    step_hours = _finite_nonnegative(step_hours, "step_hours")
    control = record.control
    flows = record.flows
    price = float(record.exogenous.price_eur_per_kwh)
    grid_kw = float(flows.grid_kw)

    grid_cost = (
        price * grid_kw
        + policy.grid_import_fee_eur_per_kwh * max(grid_kw, 0.0)
    ) * step_hours
    battery_wear = policy.wear.battery_eur_per_kwh * (
        float(control.battery_charge_kw) + float(control.battery_discharge_kw)
    ) * step_hours
    thermal_wear = policy.wear.thermal_store_eur_per_kwh * (
        float(control.thermal_charge_kw) + float(control.thermal_discharge_kw)
    ) * step_hours
    electrolyser_wear = (
        policy.wear.electrolyser_eur_per_kwh
        * float(control.electrolyser_kw)
        * step_hours
    )
    fuel_cell_wear = (
        policy.wear.fuel_cell_eur_per_kwh
        * float(control.fuel_cell_kw)
        * step_hours
    )
    reached_temperature = float(record.reached_state.indoor_temperature_c)
    comfort_violation = (
        max(reached_temperature - T_MAX_C, 0.0)
        + max(T_MIN_C - reached_temperature, 0.0)
    ) * step_hours
    operating_cost = (
        grid_cost
        + battery_wear
        + thermal_wear
        + electrolyser_wear
        + fuel_cell_wear
    )
    return StepLineItems(
        operating_step=record.operating_step,
        grid_cost_eur=grid_cost,
        battery_wear_eur=battery_wear,
        thermal_store_wear_eur=thermal_wear,
        electrolyser_wear_eur=electrolyser_wear,
        fuel_cell_wear_eur=fuel_cell_wear,
        operating_cost_eur=operating_cost,
        comfort_violation_c_h=comfort_violation,
    )


def recoverable_inventory_kwh(
    state: HubState,
    config: HubConfiguration = HubConfiguration(),
) -> float:
    """Recoverable electricity-equivalent inventory under Asset capabilities."""
    capabilities = config.capabilities
    battery = (
        ETA_BAT_DIS * float(state.soc_battery_kwh)
        if capabilities.battery
        else 0.0
    )
    hydrogen = (
        ETA_FC_E * E_H2_LHV_KWH_KG * float(state.soc_hydrogen_kg)
        if capabilities.hydrogen
        else 0.0
    )
    thermal = (
        float(state.soc_thermal_kwh) / HP_COP
        if capabilities.thermal_store
        else 0.0
    )
    return battery + hydrogen + thermal


def _summary(
    line_items: tuple[StepLineItems, ...],
    inventory_settlement_eur: float,
) -> CostSummary:
    grid_cost = sum(item.grid_cost_eur for item in line_items)
    battery_wear = sum(item.battery_wear_eur for item in line_items)
    thermal_wear = sum(item.thermal_store_wear_eur for item in line_items)
    electrolyser_wear = sum(item.electrolyser_wear_eur for item in line_items)
    fuel_cell_wear = sum(item.fuel_cell_wear_eur for item in line_items)
    operating_cost = (
        grid_cost
        + battery_wear
        + thermal_wear
        + electrolyser_wear
        + fuel_cell_wear
    )
    return CostSummary(
        grid_cost_eur=grid_cost,
        battery_wear_eur=battery_wear,
        thermal_store_wear_eur=thermal_wear,
        electrolyser_wear_eur=electrolyser_wear,
        fuel_cell_wear_eur=fuel_cell_wear,
        operating_cost_eur=operating_cost,
        inventory_settlement_eur=inventory_settlement_eur,
        inventory_adjusted_cost_eur=operating_cost + inventory_settlement_eur,
        comfort_violation_c_h=sum(
            item.comfort_violation_c_h for item in line_items
        ),
    )


def _scaled_policy(
    policy: EvaluationPolicy,
    multiplier: float,
) -> EvaluationPolicy:
    return replace(
        policy,
        wear=WearCoefficients(
            battery_eur_per_kwh=(
                policy.wear.battery_eur_per_kwh * multiplier
            ),
            thermal_store_eur_per_kwh=(
                policy.wear.thermal_store_eur_per_kwh * multiplier
            ),
            electrolyser_eur_per_kwh=(
                policy.wear.electrolyser_eur_per_kwh * multiplier
            ),
            fuel_cell_eur_per_kwh=(
                policy.wear.fuel_cell_eur_per_kwh * multiplier
            ),
        ),
    )


def _multiplier_label(multiplier: float) -> str:
    return f"{int(multiplier) if multiplier.is_integer() else multiplier:g}x"


def evaluate_run(
    run: ValidRun,
    policy: EvaluationPolicy = DEFAULT_EVALUATION_POLICY,
) -> EvaluationReport:
    """Evaluate one complete ValidRun exclusively from immutable records."""
    from control.rolling_horizon import ValidRun

    if not isinstance(run, ValidRun):
        raise TypeError("evaluate_run accepts only a complete ValidRun")
    if not run.validation.complete or not run.validation.valid:
        raise ValueError("evaluate_run requires a complete and valid Run")
    expected_count = run.scenario.operating_step_count
    assert len(run.records) == expected_count, (
        "ValidRun record count must equal Scenario Operating Step count"
    )
    assert run.validation.checked_operating_steps == expected_count, (
        "ValidationReport count must equal Scenario Operating Step count"
    )

    step_hours = run.scenario.step_duration.total_seconds() / 3600.0
    nominal_line_items = tuple(
        evaluate_step(record, policy, step_hours) for record in run.records
    )
    settlement_price = sum(
        float(record.exogenous.price_eur_per_kwh) for record in run.records
    ) / expected_count
    inventory_settlement = settlement_price * (
        recoverable_inventory_kwh(run.initial_state, run.hub_configuration)
        - recoverable_inventory_kwh(run.terminal_state, run.hub_configuration)
    )
    nominal = _summary(nominal_line_items, inventory_settlement)

    sensitivities: dict[str, CostSummary] = {}
    for multiplier in policy.sensitivity_multipliers:
        if multiplier == 1.0:
            sensitivity_summary = nominal
        else:
            sensitivity_policy = _scaled_policy(policy, multiplier)
            sensitivity_line_items = tuple(
                evaluate_step(record, sensitivity_policy, step_hours)
                for record in run.records
            )
            sensitivity_summary = _summary(
                sensitivity_line_items,
                inventory_settlement,
            )
        sensitivities[_multiplier_label(multiplier)] = sensitivity_summary

    return EvaluationReport(
        policy=policy,
        nominal=nominal,
        wear_sensitivities=sensitivities,
        step_line_items=nominal_line_items,
    )


def saving_percent(baseline_cost: float, comparison_cost: float) -> float:
    """Percentage reduction of comparison cost relative to Baseline cost."""
    return (
        100.0 * (baseline_cost - comparison_cost) / abs(baseline_cost)
        if baseline_cost
        else 0.0
    )


# Compatibility wrappers retained until the structural migration removes the
# DataFrame-facing entry points.  New evaluation code uses the stable interfaces.
def stored_equiv_kwh(
    soc_bat: float,
    soc_h2: float,
    soc_tes: float,
    config: HubConfiguration = HubConfiguration(),
) -> float:
    return recoverable_inventory_kwh(
        HubState(soc_bat, soc_h2, soc_tes, 0.0),
        config,
    )


def stored_equiv_from_row(
    row: Mapping[str, object],
    config: HubConfiguration = HubConfiguration(),
) -> float:
    return stored_equiv_kwh(
        float(row["SOC_bat_kWh"]),
        float(row["SOC_h2_kg"]),
        float(row["SOC_tes_kWh"]),
        config,
    )


def _legacy_grid_inventory_adjusted_cost(
    results_df: object,
    init_equiv: float,
    settle_price: float,
    config: HubConfiguration = HubConfiguration(),
) -> float:
    """Characterize committed pre-migration grid-only artifact summaries.

    This private adapter is not a current evaluation API: it intentionally omits
    policy wear so the old committed CSV/summary characterization remains readable.
    Task 14 must delete it with that characterization when artifacts regenerate from
    valid Run Bundles under ``evaluate_run``.
    """
    final_equiv = stored_equiv_from_row(results_df.iloc[-1], config)
    return float(results_df["grid_cost_EUR"].sum()) + settle_price * (
        init_equiv - final_equiv
    )


def saving_pct(baseline_cost: float, comparison_cost: float) -> float:
    return saving_percent(baseline_cost, comparison_cost)
