"""
Multi-carrier greenhouse energy hub — component models.

Framework: Geidl & Andersson (2007) energy hub formulation.
Location:  Representative Dutch (Westland) greenhouse, Netherlands (1 ha high-tech).

This module defines the physical plant: asset sizing, control/state bounds, and the
discrete-time state-transition function used by both the MPC internal model
(`control/mpc_controller.py`, CasADi adapter) and the rolling-horizon numerical
simulation (`control/rolling_horizon.py`).

Energy carriers
---------------
  Electricity : grid, PV, battery, electrolyser, fuel cell, heat pump, e-boiler, load
  Heat        : heat pump, e-boiler, fuel-cell waste heat, thermal store, greenhouse
  Hydrogen    : electrolyser (production) + tank + fuel cell (reconversion)

Hub assets (sized for a high-tech, fully-lit 1 ha greenhouse)
-------------------------------------------------------------
  1. Solar PV array       — 500 kWp co-located (adjacent field / north roof)
  2. Li-ion battery       — 1000 kWh / 500 kW (2 h duration)
  3. PEM electrolyser     — 250 kW electrical input
  4. PEM fuel cell        — 200 kW electrical output + recovered heat (CHP analogue)
  5. Compressed H2 tank   — 200 kg storage (~6670 kWh LHV)
  6. Air-source heat pump — 175 kW elec → 612 kW thermal (COP 3.5)
  7. Electric boiler      — 600 kW elec → 594 kW heat (power-to-heat flexibility)
  8. Hot-water TES        — 4000 kWh buffer (steel tank, common in NL greenhouses)
  9. Grid connection      — bidirectional, +-2000 kW (realistic for a lit 1 ha site)

The electrolyser + tank + fuel cell together are the green analogue of the gas CHP
being phased out: cheap/surplus electricity -> H2 -> electricity + heat on demand.

State variables (x)
-------------------
  SOC_bat   : battery state-of-charge        [kWh]  in [100, 900]
  SOC_h2    : hydrogen tank inventory         [kg]   in [8,   192]
  SOC_tes   : thermal energy store            [kWh]  in [400, 3600]
  T_in      : greenhouse indoor temperature   [degC] soft band [16, 24]

Control inputs (u)
------------------
  P_bat_ch  : battery charge power            [kW]   in [0, 500]
  P_bat_dis : battery discharge power         [kW]   in [0, 500]
  P_elz     : electrolyser power              [kW]   in [0, 250]
  P_fc      : fuel-cell electrical output     [kW]   in [0, 200]
  P_hp      : heat pump electrical input      [kW]   in [0, 175]  -> COP x 175 = 612 kW heat
  P_eboiler : electric boiler electrical input[kW]   in [0, 600]  -> eta x 600 = 594 kW heat
  Q_tes_ch  : thermal store charge rate       [kW]   in [0, 800]
  Q_tes_dis : thermal store discharge rate    [kW]   in [0, 800]
  vent      : ventilation fraction (cooling)  [-]    in [0, 1]
  P_grid    : net grid import (+) / export (-)[kW]   in [-2000, 2000]

Parameters (p) — time-varying exogenous inputs
-----------------------------------------------
  P_pv      : PV generation [kW]
  P_load    : electrical demand (lighting + base) [kW]
  price     : grid electricity price [EUR/kWh]
  T_out     : outdoor temperature [degC]
  G_Wm2     : global tilted-plane irradiance [W/m2]

Note: heat "demand" is NOT prescribed. It is implicit in the greenhouse temperature
ODE: the controller supplies heat (and opens vents) to keep T_in within the comfort
band. This is a single, internally consistent thermal model.

Efficiency parameters (literature values)
-----------------------------------------
  eta_bat_ch / eta_bat_dis : 0.96 / 0.96  -> round-trip 92.2% (Li-ion, Hesse et al. 2017)
  eta_elz                  : 0.65         -> PEM electrolyser, LHV basis (IEA 2023)
  eta_fc_e / eta_fc_h      : 0.50 / 0.35  -> PEM fuel cell elec / recovered heat (LHV)
  COP_hp                   : 3.5          -> air-source HP at 7 degC outdoor (EN 14511)
  eta_eboiler              : 0.99         -> resistive / electrode boiler
  eta_tes                  : 0.995/h      -> 0.5%/h standing loss (insulated tank)
  E_H2_LHV                 : 33.33 kWh/kg -> hydrogen lower heating value (LHV)

References
----------
  Geidl, M. & Andersson, G. (2007). Optimal power flow of multiple energy carriers.
    IEEE Trans. Power Syst. 22(1), 145-155.
  Hesse, H. et al. (2017). Lithium-ion battery storage for the grid. Energies 10(12).
  IEA (2023). Hydrogen. IEA, Paris. https://www.iea.org/reports/hydrogen
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str
    actual: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class AssetCapabilities:
    battery: bool = True
    hydrogen: bool = True
    thermal_store: bool = True


@dataclass(frozen=True)
class HubConfiguration:
    capabilities: AssetCapabilities = AssetCapabilities()


@dataclass(frozen=True)
class HubState(Mapping[str, object]):
    soc_battery_kwh: object
    soc_hydrogen_kg: object
    soc_thermal_kwh: object
    indoor_temperature_c: object

    def __getitem__(self, model_name: str) -> object:
        for field_name, mapped_name in STATE_MODEL_NAMES.items():
            if model_name == mapped_name:
                return getattr(self, field_name)
        raise KeyError(model_name)

    def __iter__(self) -> Iterator[str]:
        return iter(STATE_MODEL_NAMES.values())

    def __len__(self) -> int:
        return len(STATE_MODEL_NAMES)


@dataclass(frozen=True)
class HubControl(Mapping[str, object]):
    battery_charge_kw: object
    battery_discharge_kw: object
    electrolyser_kw: object
    fuel_cell_kw: object
    heat_pump_kw: object
    electric_boiler_kw: object
    thermal_charge_kw: object
    thermal_discharge_kw: object
    ventilation_fraction: object

    def __getitem__(self, model_name: str) -> object:
        for field_name, mapped_name in CONTROL_MODEL_NAMES.items():
            if model_name == mapped_name:
                return getattr(self, field_name)
        raise KeyError(model_name)

    def __iter__(self) -> Iterator[str]:
        return iter(CONTROL_MODEL_NAMES.values())

    def __len__(self) -> int:
        return len(CONTROL_MODEL_NAMES)


@dataclass(frozen=True)
class ExogenousInputs(Mapping[str, object]):
    pv_kw: object
    electric_load_kw: object
    price_eur_per_kwh: object
    outdoor_temperature_c: object
    irradiance_w_per_m2: object

    def __getitem__(self, model_name: str) -> object:
        for field_name, mapped_name in EXOGENOUS_MODEL_NAMES.items():
            if model_name == mapped_name:
                return getattr(self, field_name)
        raise KeyError(model_name)

    def __iter__(self) -> Iterator[str]:
        return iter(EXOGENOUS_MODEL_NAMES.values())

    def __len__(self) -> int:
        return len(EXOGENOUS_MODEL_NAMES)


@dataclass(frozen=True)
class HubFlows:
    grid_kw: object
    generated_heat_kw: object
    heat_to_air_kw: object
    thermal_charge_margin_kw: object
    hydrogen_production_kg_per_h: object
    hydrogen_consumption_kg_per_h: object


@dataclass(frozen=True)
class HubStep:
    successor: HubState
    flows: HubFlows


@dataclass(frozen=True)
class _HubConversions:
    battery_charge_kw: object
    battery_discharge_kw: object
    electrolyser_kw: object
    fuel_cell_kw: object
    thermal_charge_kw: object
    thermal_discharge_kw: object
    heat_pump_heat_kw: object
    electric_boiler_heat_kw: object
    fuel_cell_heat_kw: object
    generated_heat_kw: object
    hydrogen_production_kg_per_h: object
    hydrogen_consumption_kg_per_h: object


STATE_MODEL_NAMES = {
    "soc_battery_kwh": "SOC_bat",
    "soc_hydrogen_kg": "SOC_h2",
    "soc_thermal_kwh": "SOC_tes",
    "indoor_temperature_c": "T_in",
}

CONTROL_MODEL_NAMES = {
    "battery_charge_kw": "P_bat_ch",
    "battery_discharge_kw": "P_bat_dis",
    "electrolyser_kw": "P_elz",
    "fuel_cell_kw": "P_fc",
    "heat_pump_kw": "P_hp",
    "electric_boiler_kw": "P_eboiler",
    "thermal_charge_kw": "Q_tes_ch",
    "thermal_discharge_kw": "Q_tes_dis",
    "ventilation_fraction": "vent",
}

EXOGENOUS_MODEL_NAMES = {
    "pv_kw": "P_pv",
    "electric_load_kw": "P_load",
    "price_eur_per_kwh": "price",
    "outdoor_temperature_c": "T_out",
    "irradiance_w_per_m2": "G_Wm2",
}

# Validation tolerances are declared here so every execution and serialization
# path applies the same numerical policy.
BALANCE_STATE_TOLERANCE = 1e-6
SOLVER_BOUND_TOLERANCE_KW = 1e-4
SIMULTANEOUS_FLOW_TOLERANCE_KW = 1e-3

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
E_H2_LHV_KWH_KG = 33.33      # hydrogen LOWER heating value [kWh/kg]
DT_H = 1.0                   # timestep [hours]

# ---------------------------------------------------------------------------
# Asset capacity parameters  (sized for a fully-lit 1 ha greenhouse;
# provisional values, calibrated so peak heat/elec demand is coverable)
# ---------------------------------------------------------------------------
PV_PEAK_KWP = 500.0          # co-located PV array

BAT_CAPACITY_KWH = 1000.0    # usable battery capacity
BAT_SOC_MIN = 0.10           # minimum SOC (10%)
BAT_SOC_MAX = 0.90           # maximum SOC (90%)
BAT_P_MAX_KW = 500.0         # max charge / discharge power
ETA_BAT_CH = 0.96            # charge efficiency
ETA_BAT_DIS = 0.96           # discharge efficiency

ELZ_P_MAX_KW = 250.0         # electrolyser rated electrical power
ETA_ELZ = 0.65               # electrical -> hydrogen efficiency (LHV basis)

FC_P_MAX_KW = 200.0          # fuel-cell rated electrical output
ETA_FC_E = 0.50              # H2 chemical -> electricity efficiency (LHV)
ETA_FC_H = 0.35              # H2 chemical -> recovered heat efficiency (LHV)

H2_CAPACITY_KG = 200.0       # hydrogen tank capacity
H2_SOC_MIN = 0.04            # min fill (8 kg — pressure floor)
H2_SOC_MAX = 0.96            # max fill (192 kg)

HP_P_MAX_KW = 175.0          # heat pump max electrical input -> 612 kW thermal
HP_COP = 3.5                 # coefficient of performance (air-source, 7 degC outdoor)

EBOILER_P_MAX_KW = 600.0     # electric boiler max electrical input
ETA_EBOILER = 0.99           # electrical -> heat efficiency

TES_CAPACITY_KWH = 4000.0    # thermal energy store capacity
TES_SOC_MIN = 0.10           # min fill
TES_SOC_MAX = 0.90           # max fill
TES_P_MAX_KW = 800.0         # max charge/discharge rate
ETA_TES_STANDING = 0.995     # standing efficiency per hour (0.5%/h loss)

GRID_P_MAX_KW = 2000.0       # grid connection capacity (+- )

# Grid tariff asymmetry: imported energy carries grid-transport + levy charges on
# top of the wholesale day-ahead price; exported energy is settled at wholesale.
# This breaks the symmetric buy=sell arbitrage and tempers negative-price gaming.
# Indicative NL large-consumer transport/levy component (provisional).
GRID_IMPORT_FEE_EUR_KWH = 0.025

# ---------------------------------------------------------------------------
# Greenhouse thermal model (single setpoint ODE; WUR-informed parameterisation)
# ---------------------------------------------------------------------------
FLOOR_AREA_M2 = 10_000.0
C_AIR_KJ_K = 7.5 * FLOOR_AREA_M2     # air+mass heat capacity: 7.5 kJ/(m2.K) total
C_AIR_KWH_K = C_AIR_KJ_K / 3600.0    # -> ~20.8 kWh/K

# Envelope conductance (conduction/radiation), modern Dutch double-glass:
#   3.5 W/(m2.K) x 10,000 m2 / 1000 = 35 kW/K
U_EFF_KW_K = 35.0

# Ventilation: roof vents are a CONTROL. At full opening (vent=1) the effective
# ventilation conductance is K_VENT_KW_K (~8 air changes/hour). The controller
# modulates vent in [0,1] for cooling authority.
K_VENT_KW_K = 200.0

# Sensible solar gain fraction of tilted-plane irradiance entering the air node
# (glass transmission x shade screens x transpiration latent split). See README.
SOLAR_GAIN_FRAC = 0.08

# Constant latent heat withdrawal from crop transpiration [kW]
Q_CROP_LATENT_KW = 80.0

# Comfort band and setpoint
T_MIN_C = 16.0
T_MAX_C = 24.0
T_SETPOINT_C = 19.0

# Hard safety bounds on the temperature state (band is enforced softly by the MPC)
T_HARD_MIN_C = 5.0
T_HARD_MAX_C = 40.0


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
def physical_state_bounds(
    config: HubConfiguration,
) -> dict[str, tuple[float, float]]:
    """Physical Run bounds, with Disabled Assets pinned exactly to zero."""
    capabilities = config.capabilities
    return {
        "soc_battery_kwh": (
            (0.0, BAT_CAPACITY_KWH) if capabilities.battery else (0.0, 0.0)
        ),
        "soc_hydrogen_kg": (
            (0.0, H2_CAPACITY_KG) if capabilities.hydrogen else (0.0, 0.0)
        ),
        "soc_thermal_kwh": (
            (0.0, TES_CAPACITY_KWH)
            if capabilities.thermal_store
            else (0.0, 0.0)
        ),
        "indoor_temperature_c": (T_HARD_MIN_C, T_HARD_MAX_C),
    }


def operational_state_bounds(
    config: HubConfiguration,
) -> dict[str, tuple[float, float]]:
    """Existing MPC reserve policy plus the common hard temperature limits."""
    capabilities = config.capabilities
    return {
        "soc_battery_kwh": (
            (
                BAT_SOC_MIN * BAT_CAPACITY_KWH,
                BAT_SOC_MAX * BAT_CAPACITY_KWH,
            )
            if capabilities.battery
            else (0.0, 0.0)
        ),
        "soc_hydrogen_kg": (
            (H2_SOC_MIN * H2_CAPACITY_KG, H2_SOC_MAX * H2_CAPACITY_KG)
            if capabilities.hydrogen
            else (0.0, 0.0)
        ),
        "soc_thermal_kwh": (
            (TES_SOC_MIN * TES_CAPACITY_KWH, TES_SOC_MAX * TES_CAPACITY_KWH)
            if capabilities.thermal_store
            else (0.0, 0.0)
        ),
        "indoor_temperature_c": (T_HARD_MIN_C, T_HARD_MAX_C),
    }


def control_bounds(config: HubConfiguration) -> dict[str, tuple[float, float]]:
    """Physical bounds for each stable HubControl field."""
    capabilities = config.capabilities
    return {
        "battery_charge_kw": (
            (0.0, BAT_P_MAX_KW) if capabilities.battery else (0.0, 0.0)
        ),
        "battery_discharge_kw": (
            (0.0, BAT_P_MAX_KW) if capabilities.battery else (0.0, 0.0)
        ),
        "electrolyser_kw": (
            (0.0, ELZ_P_MAX_KW) if capabilities.hydrogen else (0.0, 0.0)
        ),
        "fuel_cell_kw": (
            (0.0, FC_P_MAX_KW) if capabilities.hydrogen else (0.0, 0.0)
        ),
        "heat_pump_kw": (0.0, HP_P_MAX_KW),
        "electric_boiler_kw": (0.0, EBOILER_P_MAX_KW),
        "thermal_charge_kw": (
            (0.0, TES_P_MAX_KW)
            if capabilities.thermal_store
            else (0.0, 0.0)
        ),
        "thermal_discharge_kw": (
            (0.0, TES_P_MAX_KW)
            if capabilities.thermal_store
            else (0.0, 0.0)
        ),
        "ventilation_fraction": (0.0, 1.0),
        # P_grid is NOT a control: it is the derived slack bus (see hub_dynamics).
    }


def state_bounds(
    config: HubConfiguration = HubConfiguration(),
) -> dict[str, tuple[float, float]]:
    """Legacy model-name view of MPC operational state bounds."""
    stable = operational_state_bounds(config)
    return {STATE_MODEL_NAMES[field]: bounds for field, bounds in stable.items()}


def input_bounds(
    config: HubConfiguration = HubConfiguration(),
) -> dict[str, tuple[float, float]]:
    """Legacy model-name view of physical control bounds."""
    stable = control_bounds(config)
    return {CONTROL_MODEL_NAMES[field]: bounds for field, bounds in stable.items()}


# ---------------------------------------------------------------------------
# Derived hub quantities (shared by plant and symbolic model)
# ---------------------------------------------------------------------------
def _hub_conversions(
    control: HubControl,
    config: HubConfiguration,
) -> _HubConversions:
    """Apply capabilities and own every asset conversion expression once."""
    capabilities = config.capabilities
    battery_charge = (
        control.battery_charge_kw if capabilities.battery else 0.0
    )
    battery_discharge = (
        control.battery_discharge_kw if capabilities.battery else 0.0
    )
    electrolyser = control.electrolyser_kw if capabilities.hydrogen else 0.0
    fuel_cell = control.fuel_cell_kw if capabilities.hydrogen else 0.0
    thermal_charge = (
        control.thermal_charge_kw if capabilities.thermal_store else 0.0
    )
    thermal_discharge = (
        control.thermal_discharge_kw if capabilities.thermal_store else 0.0
    )

    heat_pump_heat = HP_COP * control.heat_pump_kw
    electric_boiler_heat = ETA_EBOILER * control.electric_boiler_kw
    hydrogen_consumption = fuel_cell / ETA_FC_E / E_H2_LHV_KWH_KG
    fuel_cell_heat = ETA_FC_H * fuel_cell / ETA_FC_E
    hydrogen_production = ETA_ELZ * electrolyser / E_H2_LHV_KWH_KG

    return _HubConversions(
        battery_charge_kw=battery_charge,
        battery_discharge_kw=battery_discharge,
        electrolyser_kw=electrolyser,
        fuel_cell_kw=fuel_cell,
        thermal_charge_kw=thermal_charge,
        thermal_discharge_kw=thermal_discharge,
        heat_pump_heat_kw=heat_pump_heat,
        electric_boiler_heat_kw=electric_boiler_heat,
        fuel_cell_heat_kw=fuel_cell_heat,
        generated_heat_kw=(
            heat_pump_heat + electric_boiler_heat + fuel_cell_heat
        ),
        hydrogen_production_kg_per_h=hydrogen_production,
        hydrogen_consumption_kg_per_h=hydrogen_consumption,
    )


def greenhouse_temperature_next(T_in, Q_air, vent, T_out):
    """
    Implicit-Euler update of greenhouse air temperature (Δt = 1 h).

        C dT/dt = Q_air + (U + K_vent*vent)*(T_out - T_next) - Q_latent

    Solved for T_next (smooth in `vent`, so differentiable for IPOPT):

        T_next = (C*T_in + Q_air + Geff*T_out - Q_latent) / (C + Geff)
        Geff   = U_EFF + K_VENT*vent
    """
    C = C_AIR_KWH_K / DT_H
    Geff = U_EFF_KW_K + K_VENT_KW_K * vent
    return (C * T_in + Q_air + Geff * T_out - Q_CROP_LATENT_KW) / (C + Geff)


# ---------------------------------------------------------------------------
# Discrete-time state update (Euler, Δt = 1 h)  — the "plant"
# ---------------------------------------------------------------------------
def hub_dynamics(
    x: Mapping[str, object],
    u: Mapping[str, object],
    p: Mapping[str, object],
    config: HubConfiguration = HubConfiguration(),
) -> tuple[dict[str, object], dict[str, object]]:
    """Temporary dictionary compatibility wrapper over :func:`advance_hub`.

    Exact realized maxima and legacy presentation metrics intentionally live here,
    outside the shared expression layer consumed by CasADi.
    """
    state = HubState(
        soc_battery_kwh=x["SOC_bat"],
        soc_hydrogen_kg=x["SOC_h2"],
        soc_thermal_kwh=x["SOC_tes"],
        indoor_temperature_c=x["T_in"],
    )
    control = HubControl(
        battery_charge_kw=u["P_bat_ch"],
        battery_discharge_kw=u["P_bat_dis"],
        electrolyser_kw=u["P_elz"],
        fuel_cell_kw=u["P_fc"],
        heat_pump_kw=u["P_hp"],
        electric_boiler_kw=u["P_eboiler"],
        thermal_charge_kw=u["Q_tes_ch"],
        thermal_discharge_kw=u["Q_tes_dis"],
        ventilation_fraction=u["vent"],
    )
    exogenous = ExogenousInputs(
        pv_kw=p["P_pv"],
        electric_load_kw=p["P_load"],
        price_eur_per_kwh=p.get("price", 0.0),
        outdoor_temperature_c=p["T_out"],
        irradiance_w_per_m2=p.get("G_Wm2", 0.0),
    )
    step = advance_hub(state, control, exogenous, config)
    conversions = _hub_conversions(control, config)
    grid_kw = step.flows.grid_kw
    reached_temperature = step.successor.indoor_temperature_c
    thermal_charge_excess = max(0.0, step.flows.thermal_charge_margin_kw)

    return (
        {
            "SOC_bat": step.successor.soc_battery_kwh,
            "SOC_h2": step.successor.soc_hydrogen_kg,
            "SOC_tes": step.successor.soc_thermal_kwh,
            "T_in": step.successor.indoor_temperature_c,
        },
        {
            "P_grid_kW": grid_kw,
            "elec_residual_kW": 0.0,
            "tes_charge_excess_kW": thermal_charge_excess,
            "Q_hp_kW": conversions.heat_pump_heat_kw,
            "Q_eboiler_kW": conversions.electric_boiler_heat_kw,
            "Q_fc_heat_kW": conversions.fuel_cell_heat_kw,
            "Q_air_kW": step.flows.heat_to_air_kw,
            "m_h2_prod_kg_h": step.flows.hydrogen_production_kg_per_h,
            "m_h2_fc_kg_h": step.flows.hydrogen_consumption_kg_per_h,
            "grid_cost_EUR": (
                grid_kw * exogenous.price_eur_per_kwh
                + GRID_IMPORT_FEE_EUR_KWH * max(0.0, grid_kw)
            )
            * DT_H,
            "T_violation_C": max(0.0, reached_temperature - T_MAX_C)
            + max(0.0, T_MIN_C - reached_temperature),
        },
    )


# ---------------------------------------------------------------------------
# Stable hub interface and fail-closed validation
# ---------------------------------------------------------------------------
def hub_state_array(state: HubState) -> np.ndarray:
    return np.asarray(
        [getattr(state, field_name) for field_name in STATE_MODEL_NAMES], dtype=float
    ).reshape(-1, 1)


def hub_control_from_array(values: Sequence[float]) -> HubControl:
    raw = np.asarray(values).reshape(-1)
    if raw.size != len(CONTROL_MODEL_NAMES):
        raise ValueError(
            f"expected {len(CONTROL_MODEL_NAMES)} control values, received {raw.size}"
        )
    return HubControl(
        **{
            field_name: raw[index]
            for index, field_name in enumerate(CONTROL_MODEL_NAMES)
        }
    )


def hub_step_expressions(
    state: HubState,
    control: HubControl,
    exogenous: ExogenousInputs,
    config: HubConfiguration = HubConfiguration(),
) -> HubStep:
    """Return one physical step using float- and CasADi-compatible arithmetic."""
    capabilities = config.capabilities
    conversions = _hub_conversions(control, config)
    solar_heat = (
        SOLAR_GAIN_FRAC
        * exogenous.irradiance_w_per_m2
        * FLOOR_AREA_M2
        / 1000.0
    )
    heat_to_air = (
        conversions.generated_heat_kw
        - conversions.thermal_charge_kw
        + conversions.thermal_discharge_kw
        + solar_heat
    )
    grid_kw = (
        exogenous.electric_load_kw
        + conversions.battery_charge_kw
        + conversions.electrolyser_kw
        + control.heat_pump_kw
        + control.electric_boiler_kw
        - exogenous.pv_kw
        - conversions.battery_discharge_kw
        - conversions.fuel_cell_kw
    )

    soc_battery_next = (
        state.soc_battery_kwh
        + ETA_BAT_CH * conversions.battery_charge_kw * DT_H
        - conversions.battery_discharge_kw / ETA_BAT_DIS * DT_H
        if capabilities.battery
        else 0.0
    )
    soc_hydrogen_next = (
        state.soc_hydrogen_kg
        + (
            conversions.hydrogen_production_kg_per_h
            - conversions.hydrogen_consumption_kg_per_h
        )
        * DT_H
        if capabilities.hydrogen
        else 0.0
    )
    soc_thermal_next = (
        ETA_TES_STANDING * state.soc_thermal_kwh
        + conversions.thermal_charge_kw * DT_H
        - conversions.thermal_discharge_kw * DT_H
        if capabilities.thermal_store
        else 0.0
    )
    indoor_temperature_next = greenhouse_temperature_next(
        state.indoor_temperature_c,
        heat_to_air,
        control.ventilation_fraction,
        exogenous.outdoor_temperature_c,
    )

    return HubStep(
        successor=HubState(
            soc_battery_kwh=soc_battery_next,
            soc_hydrogen_kg=soc_hydrogen_next,
            soc_thermal_kwh=soc_thermal_next,
            indoor_temperature_c=indoor_temperature_next,
        ),
        flows=HubFlows(
            grid_kw=grid_kw,
            generated_heat_kw=conversions.generated_heat_kw,
            heat_to_air_kw=heat_to_air,
            thermal_charge_margin_kw=(
                conversions.thermal_charge_kw - conversions.generated_heat_kw
            ),
            hydrogen_production_kg_per_h=(
                conversions.hydrogen_production_kg_per_h
            ),
            hydrogen_consumption_kg_per_h=(
                conversions.hydrogen_consumption_kg_per_h
            ),
        ),
    )


def advance_hub(
    state: HubState,
    control: HubControl,
    exogenous: ExogenousInputs,
    config: HubConfiguration = HubConfiguration(),
) -> HubStep:
    return hub_step_expressions(state, control, exogenous, config)


def _finite_field_values(instance: object, expected_type: type) -> tuple[
    dict[str, float], tuple[ValidationIssue, ...]
]:
    if not isinstance(instance, expected_type):
        return {}, (
            ValidationIssue(
                code="schema_error",
                field=expected_type.__name__,
                message=f"expected {expected_type.__name__}",
            ),
        )

    values: dict[str, float] = {}
    issues: list[ValidationIssue] = []
    for item in fields(expected_type):
        raw = getattr(instance, item.name)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            issues.append(
                ValidationIssue(
                    code="non_finite",
                    field=item.name,
                    message=f"{item.name} must be a finite scalar",
                )
            )
            continue
        if not math.isfinite(value):
            issues.append(
                ValidationIssue(
                    code="non_finite",
                    field=item.name,
                    message=f"{item.name} must be finite",
                    actual=value,
                )
            )
            continue
        values[item.name] = value
    return values, tuple(issues)


def validate_control(
    control: HubControl,
    config: HubConfiguration,
    tolerance: float = BALANCE_STATE_TOLERANCE,
) -> tuple[ValidationIssue, ...]:
    values, finite_issues = _finite_field_values(control, HubControl)
    issues = list(finite_issues)
    bounds = control_bounds(config)
    for field_name, value in values.items():
        lower, upper = bounds[field_name]
        exact_zero_bound = lower == 0.0 and upper == 0.0
        field_tolerance = 0.0 if exact_zero_bound else (
            tolerance
            if field_name != "ventilation_fraction"
            else BALANCE_STATE_TOLERANCE
        )
        if value < lower - field_tolerance or value > upper + field_tolerance:
            issues.append(
                ValidationIssue(
                    code="out_of_bounds",
                    field=field_name,
                    message=f"{field_name} is outside [{lower}, {upper}]",
                    actual=value,
                    lower=lower,
                    upper=upper,
                )
            )

    for label, charge_field, discharge_field in (
        ("battery", "battery_charge_kw", "battery_discharge_kw"),
        ("hydrogen", "electrolyser_kw", "fuel_cell_kw"),
        ("thermal_store", "thermal_charge_kw", "thermal_discharge_kw"),
    ):
        charge = values.get(charge_field)
        discharge = values.get(discharge_field)
        if (
            charge is not None
            and discharge is not None
            and charge > SIMULTANEOUS_FLOW_TOLERANCE_KW
            and discharge > SIMULTANEOUS_FLOW_TOLERANCE_KW
        ):
            issues.append(
                ValidationIssue(
                    code="simultaneous_charge_discharge",
                    field=label,
                    message=(
                        f"{label} charge and discharge both exceed "
                        f"{SIMULTANEOUS_FLOW_TOLERANCE_KW} kW"
                    ),
                    actual=min(charge, discharge),
                    upper=SIMULTANEOUS_FLOW_TOLERANCE_KW,
                )
            )
    return tuple(issues)


def normalize_control(
    control: HubControl,
    config: HubConfiguration,
    bound_tolerance: float = SOLVER_BOUND_TOLERANCE_KW,
    *,
    zero_small_flows: bool = True,
) -> HubControl:
    """Clip accepted bound noise and optionally zero sub-threshold flow noise."""
    bounds = control_bounds(config)
    normalized: dict[str, float] = {}
    for field_name in CONTROL_MODEL_NAMES:
        value = float(getattr(control, field_name))
        lower, upper = bounds[field_name]
        tolerance = (
            bound_tolerance
            if field_name != "ventilation_fraction"
            else BALANCE_STATE_TOLERANCE
        )
        if (
            zero_small_flows
            and field_name != "ventilation_fraction"
            and abs(value) <= SIMULTANEOUS_FLOW_TOLERANCE_KW
        ):
            value = 0.0
        if lower - tolerance <= value < lower:
            value = lower
        elif upper < value <= upper + tolerance:
            value = upper
        normalized[field_name] = value
    return HubControl(**normalized)


def validate_successor(
    state: HubState,
    config: HubConfiguration,
    require_operational_storage: bool,
    tolerance: float = BALANCE_STATE_TOLERANCE,
) -> tuple[ValidationIssue, ...]:
    values, finite_issues = _finite_field_values(state, HubState)
    issues = list(finite_issues)
    bounds = (
        operational_state_bounds(config)
        if require_operational_storage
        else physical_state_bounds(config)
    )
    for field_name, value in values.items():
        lower, upper = bounds[field_name]
        field_tolerance = (
            0.0 if lower == 0.0 and upper == 0.0 else tolerance
        )
        if (
            value < lower - field_tolerance
            or value > upper + field_tolerance
        ):
            issues.append(
                ValidationIssue(
                    code="out_of_bounds",
                    field=field_name,
                    message=f"{field_name} is outside [{lower}, {upper}]",
                    actual=value,
                    lower=lower,
                    upper=upper,
                )
            )
    return tuple(issues)


def validate_grid_flow(
    flows: HubFlows,
    tolerance: float = BALANCE_STATE_TOLERANCE,
) -> tuple[ValidationIssue, ...]:
    try:
        grid_kw = float(flows.grid_kw)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ()
    if math.isfinite(grid_kw) and abs(grid_kw) > GRID_P_MAX_KW + tolerance:
        return (
            ValidationIssue(
                code="grid_limit",
                field="grid_kw",
                message=f"grid flow exceeds +/-{GRID_P_MAX_KW} kW",
                actual=grid_kw,
                lower=-GRID_P_MAX_KW,
                upper=GRID_P_MAX_KW,
            ),
        )
    return ()


def validate_thermal_charge(
    flows: HubFlows,
    tolerance: float = BALANCE_STATE_TOLERANCE,
) -> tuple[ValidationIssue, ...]:
    try:
        margin_kw = float(flows.thermal_charge_margin_kw)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ()
    if math.isfinite(margin_kw) and margin_kw > tolerance:
        return (
            ValidationIssue(
                code="thermal_charge_infeasible",
                field="thermal_charge_margin_kw",
                message="thermal-store charging exceeds generated heat",
                actual=margin_kw,
                upper=0.0,
            ),
        )
    return ()


def validate_flows(
    flows: HubFlows,
    tolerance: float = BALANCE_STATE_TOLERANCE,
) -> tuple[ValidationIssue, ...]:
    _, finite_issues = _finite_field_values(flows, HubFlows)
    return (
        *finite_issues,
        *validate_grid_flow(flows, tolerance),
        *validate_thermal_charge(flows, tolerance),
    )


# ---------------------------------------------------------------------------
# Initial state (typical operating point, mid-inventory)
# ---------------------------------------------------------------------------
def initial_state(config: HubConfiguration = HubConfiguration()) -> HubState:
    """Physically reasonable initial hub state."""
    capabilities = config.capabilities
    return HubState(
        soc_battery_kwh=(
            0.50 * BAT_CAPACITY_KWH if capabilities.battery else 0.0
        ),
        soc_hydrogen_kg=(
            0.30 * H2_CAPACITY_KG if capabilities.hydrogen else 0.0
        ),
        soc_thermal_kwh=(
            0.40 * TES_CAPACITY_KWH if capabilities.thermal_store else 0.0
        ),
        indoor_temperature_c=T_SETPOINT_C,  # at setpoint
    )


# ---------------------------------------------------------------------------
# Scaling factors for do-mpc (improves numerical conditioning)
# ---------------------------------------------------------------------------
STATE_SCALE = {
    "SOC_bat": BAT_CAPACITY_KWH,
    "SOC_h2":  H2_CAPACITY_KG,
    "SOC_tes": TES_CAPACITY_KWH,
    "T_in":    10.0,
}

INPUT_SCALE = {
    "P_bat_ch":  BAT_P_MAX_KW,
    "P_bat_dis": BAT_P_MAX_KW,
    "P_elz":     ELZ_P_MAX_KW,
    "P_fc":      FC_P_MAX_KW,
    "P_hp":      HP_P_MAX_KW,
    "P_eboiler": EBOILER_P_MAX_KW,
    "Q_tes_ch":  TES_P_MAX_KW,
    "Q_tes_dis": TES_P_MAX_KW,
    "vent":      1.0,
}
