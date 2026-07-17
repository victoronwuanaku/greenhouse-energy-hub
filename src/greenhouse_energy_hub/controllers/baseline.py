"""Limited-capability Baseline Controller Adapter.

The characterized frugal thermostat and battery rule remain intentionally
unchanged.  Its exact missing capabilities are explicit policy metadata rather
than an implicit claim of capability-equivalent Controller performance.
"""

from collections.abc import Mapping

from greenhouse_energy_hub.hub import (
    BAT_P_MAX_KW,
    C_AIR_KWH_K,
    DT_H,
    EBOILER_P_MAX_KW,
    ETA_BAT_CH,
    ETA_BAT_DIS,
    ETA_EBOILER,
    FLOOR_AREA_M2,
    HP_COP,
    HP_P_MAX_KW,
    Q_CROP_LATENT_KW,
    SOLAR_GAIN_FRAC,
    TES_P_MAX_KW,
    T_MAX_C,
    T_MIN_C,
    U_EFF_KW_K,
    HubConfiguration,
    HubControl,
    HubState,
    state_bounds,
)
from greenhouse_energy_hub.scenarios import ScenarioPoint
from greenhouse_energy_hub.simulation import (
    ControlDecision,
    ControllerFailure,
    DecisionDiagnostics,
    JSONValue,
    _read_only_mapping,
)


# Limited-capability Baseline: a frugal thermostat that holds the lower comfort
# bound. Its exact missing capabilities are declared by BaselineControllerAdapter.
BASELINE_TARGET_C = T_MIN_C + 0.5


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
