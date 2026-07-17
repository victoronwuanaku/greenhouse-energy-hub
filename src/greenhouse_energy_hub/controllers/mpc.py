"""
Rolling-horizon economic MPC for the greenhouse energy hub.

Formulation
-----------
Solver:    do-mpc v5 + CasADi v3.7 (IPOPT backend)
Model:     Nonlinear discrete-time model (implicit-Euler thermal node, Δt = 1 h)
Horizon:   N = 24 steps (full day-night price cycle look-ahead)
Objective: Minimise total grid electricity cost + degradation/cycling costs,
           subject to a hard electricity balance and a SOFT indoor-temperature band.

Finite-horizon optimal control problem (solved at each timestep k):

    min   Σ_{i=0}^{N-1} [ price(k+i)·P_grid(k+i)·Δt              (grid cost)
      u                  + c_bat·(P_bat_ch+P_bat_dis)·Δt          (battery wear)
                         + c_tes·(Q_tes_ch+Q_tes_dis)·Δt          (TES throughput)
                         + c_elz·P_elz·Δt + c_fc·P_fc·Δt          (H2 stack wear)
                         + c_cmpl·(complementarity)               (anti-cycling)
                         + ρ·(T_band slack)                       (soft comfort band) ]
          - β·λ̄·V_stored(x_{k+N})                                 (shadow-price terminal value)

    s.t.  x_{k+i+1} = f(x_{k+i}, u_{k+i})                          dynamics
          x_min ≤ x ≤ x_max,   u_min ≤ u ≤ u_max                  bounds
          P_grid + P_pv + P_bat_dis + P_fc                         electricity balance
            = P_load + P_bat_ch + P_elz + P_hp + P_eboiler         (hard equality)
          Q_tes_ch ≤ Q_hp + Q_eboiler + Q_fc_heat                  TES charges from generated heat
          16 ≤ T_in ≤ 24 °C                                        soft (slack-penalised)
          x(k) = x_measured

Design notes
------------
* Heat is NOT a prescribed demand: the controller supplies heat and opens vents to
  hold T_in in band. This removes the old conflicting two-thermal-model formulation.
* Hydrogen has a discharge path (fuel cell -> electricity + heat), so the H2 buffer
  carries economic value instead of being a pure cost sink.
* Battery/TES throughput costs + a complementarity penalty eliminate the physically
  meaningless simultaneous charge+discharge that the old formulation exhibited.
* The terminal cost values stored energy at the horizon-average price (a simple
  cost-to-go proxy), avoiding both end-of-horizon dumping and free hoarding.

References
----------
  Fiedler, F. et al. (2023). do-mpc: Towards FAIR nonlinear and robust MPC.
    Control Engineering Practice, 140, 105676.
  Andersson, J.A.E. et al. (2019). CasADi. Math. Prog. Comp. 11(1).
  McAllister, R.D. et al. (2025). RL-Guided MPC for Autonomous Greenhouse
    Control. arXiv:2506.13278.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import do_mpc
from casadi import DM, sqrt

from greenhouse_energy_hub.evaluation import EvaluationPolicy
from greenhouse_energy_hub.simulation import (
    ControlDecision,
    ControllerFailure,
    DecisionDiagnostics,
    JSONValue,
)
from greenhouse_energy_hub.hub import (
    ExogenousInputs, HubConfiguration, HubControl, HubState,
    hub_control_from_array, hub_state_array, hub_step_expressions,
    BAT_P_MAX_KW, ETA_BAT_DIS,
    E_H2_LHV_KWH_KG,
    FC_P_MAX_KW, ETA_FC_E,
    HP_COP,
    TES_P_MAX_KW,
    GRID_P_MAX_KW, GRID_IMPORT_FEE_EUR_KWH,
    T_MIN_C, T_MAX_C, T_SETPOINT_C,
    STATE_MODEL_NAMES, CONTROL_MODEL_NAMES,
    operational_state_bounds, control_bounds,
    BALANCE_STATE_TOLERANCE, STATE_SCALE, INPUT_SCALE,
)

# ---------------------------------------------------------------------------
# MPC hyper-parameters
# ---------------------------------------------------------------------------
N_HORIZON = 24          # prediction horizon [hours]
DT_H = 1.0              # timestep [hours]

# Cost weights
W_BAT_THRU  = 5e-3      # battery throughput / degradation [EUR/kWh]
W_TES_THRU  = 5e-4      # thermal store throughput [EUR/kWh]
W_ELZ_WEAR  = 2e-3      # electrolyser stack wear [EUR/kWh]
W_FC_WEAR   = 2e-3      # fuel-cell stack wear [EUR/kWh]
W_COMPL     = 1e-3      # complementarity penalty (anti simultaneous ch/dis)
W_TBAND     = 10.0      # soft temperature-band violation penalty [EUR per slack]
W_TERMINAL  = 1.0       # terminal stored-energy value weight
W_RTERM     = 1e-4      # input-move smoothing (rterm)


def _immutable_metadata(values: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(value: object) -> object:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {str(key): freeze(item) for key, item in value.items()}
            )
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return value

    return freeze(values)


@dataclass(frozen=True)
class MpcConfiguration:
    horizon_steps: int = N_HORIZON
    terminal_weight: float = W_TERMINAL
    battery_wear_eur_per_kwh: float = W_BAT_THRU
    thermal_store_wear_eur_per_kwh: float = W_TES_THRU
    electrolyser_wear_eur_per_kwh: float = W_ELZ_WEAR
    fuel_cell_wear_eur_per_kwh: float = W_FC_WEAR
    complementarity_weight: float = W_COMPL
    comfort_slack_weight: float = W_TBAND
    input_move_weight: float = W_RTERM
    solver_max_iterations: int = 800
    solver_tolerance: float = 1e-6

    @classmethod
    def from_evaluation_policy(
        cls,
        policy: EvaluationPolicy,
        **solver_configuration: object,
    ) -> "MpcConfiguration":
        """Bind declared evaluation wear terms into the Solver Objective."""
        if not isinstance(policy, EvaluationPolicy):
            raise TypeError("policy must be an EvaluationPolicy")
        if policy.grid_import_fee_eur_per_kwh != GRID_IMPORT_FEE_EUR_KWH:
            raise ValueError(
                "MPC currently supports only the declared grid import fee "
                f"{GRID_IMPORT_FEE_EUR_KWH} EUR/kWh"
            )
        economic_names = {
            "battery_wear_eur_per_kwh",
            "thermal_store_wear_eur_per_kwh",
            "electrolyser_wear_eur_per_kwh",
            "fuel_cell_wear_eur_per_kwh",
        }
        conflicting = economic_names.intersection(solver_configuration)
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise TypeError(
                f"Solver economic terms come from EvaluationPolicy, not overrides: {names}"
            )
        return cls(
            battery_wear_eur_per_kwh=policy.wear.battery_eur_per_kwh,
            thermal_store_wear_eur_per_kwh=(
                policy.wear.thermal_store_eur_per_kwh
            ),
            electrolyser_wear_eur_per_kwh=(
                policy.wear.electrolyser_eur_per_kwh
            ),
            fuel_cell_wear_eur_per_kwh=policy.wear.fuel_cell_eur_per_kwh,
            **solver_configuration,
        )

    def to_controller_metadata(self) -> dict[str, object]:
        """Separate realized economic terms from solver-only diagnostics."""
        return {
            "horizon_steps": self.horizon_steps,
            "solver_objective_economic_terms": {
                "battery_wear_eur_per_kwh": self.battery_wear_eur_per_kwh,
                "thermal_store_wear_eur_per_kwh": (
                    self.thermal_store_wear_eur_per_kwh
                ),
                "electrolyser_wear_eur_per_kwh": (
                    self.electrolyser_wear_eur_per_kwh
                ),
                "fuel_cell_wear_eur_per_kwh": self.fuel_cell_wear_eur_per_kwh,
            },
            "solver_diagnostics": {
                "terminal_weight": self.terminal_weight,
                "complementarity_weight": self.complementarity_weight,
                "comfort_slack_weight": self.comfort_slack_weight,
                "input_move_weight": self.input_move_weight,
                "solver_max_iterations": self.solver_max_iterations,
                "solver_tolerance": self.solver_tolerance,
            },
        }


@dataclass(frozen=True)
class _NeutralForecastPoint:
    price_eur_per_kwh: float = 0.0
    pv_kw: float = 0.0
    electric_load_kw: float = 0.0
    outdoor_temperature_c: float = T_SETPOINT_C
    irradiance_w_per_m2: float = 0.0


class _ForecastTVPSource:
    """Expose only the active N+1 immutable forecast view to do-mpc."""

    def __init__(self, tvp_template: object, horizon_steps: int) -> None:
        self._tvp_template = tvp_template
        self._horizon_steps = horizon_steps
        self._neutral_forecast = tuple(
            _NeutralForecastPoint() for _ in range(horizon_steps + 1)
        )
        self.clear()

    def activate(
        self,
        forecast: tuple[object, ...],
        terminal_electric_value: float,
        terminal_heat_value: float,
    ) -> None:
        self._active_forecast = forecast
        self._terminal_electric_value = terminal_electric_value
        self._terminal_heat_value = terminal_heat_value

    def clear(self) -> None:
        self._active_forecast = self._neutral_forecast
        self._terminal_electric_value = 0.0
        self._terminal_heat_value = 0.0

    def __call__(self, _t_now: object) -> object:
        for step in range(self._horizon_steps + 1):
            point = self._active_forecast[step]
            self._tvp_template["_tvp", step, "price"] = float(
                point.price_eur_per_kwh
            )
            self._tvp_template["_tvp", step, "P_pv"] = float(point.pv_kw)
            self._tvp_template["_tvp", step, "P_load"] = float(
                point.electric_load_kw
            )
            self._tvp_template["_tvp", step, "T_out"] = float(
                point.outdoor_temperature_c
            )
            self._tvp_template["_tvp", step, "G_Wm2"] = float(
                point.irradiance_w_per_m2
            )
            self._tvp_template[
                "_tvp", step, "terminal_electric_value"
            ] = self._terminal_electric_value
            self._tvp_template[
                "_tvp", step, "terminal_heat_value"
            ] = self._terminal_heat_value
        return self._tvp_template


class MpcControllerAdapter:
    name = "mpc"
    requires_operational_storage_bounds = True

    def __init__(
        self,
        mpc: object,
        forecast_horizon_steps: int,
        configuration: Mapping[str, JSONValue],
        capability_policy: Mapping[str, JSONValue],
    ) -> None:
        self._mpc = mpc
        self.forecast_horizon_steps = forecast_horizon_steps
        self.configuration = _immutable_metadata(configuration)
        self.capability_policy = _immutable_metadata(capability_policy)

    def decide(
        self,
        state: HubState,
        forecast: tuple[object, ...],
    ) -> ControlDecision | ControllerFailure:
        required_points = self.forecast_horizon_steps + 1
        if not isinstance(forecast, tuple) or len(forecast) != required_points:
            received_points = len(forecast) if hasattr(forecast, "__len__") else 0
            return ControllerFailure(
                code="forecast_coverage",
                message=(
                    f"MPC horizon {self.forecast_horizon_steps} requires "
                    f"{required_points} immutable forecast points; received "
                    f"{received_points}"
                ),
                diagnostics=DecisionDiagnostics(
                    adapter="mpc",
                    decision_status="failure",
                    solver_success=None,
                    solver_return_status=None,
                    solver_iterations=None,
                    solver_wall_seconds=None,
                    forecast_start_utc=(
                        forecast[0].timestamp_utc if received_points else None
                    ),
                    forecast_end_utc=(
                        forecast[-1].timestamp_utc if received_points else None
                    ),
                    terminal_electric_value_eur_per_kwh=None,
                    terminal_heat_value_eur_per_kwhth=None,
                ),
            )

        stage_points = forecast[:-1]
        terminal_electric_value = float(
            np.mean([point.price_eur_per_kwh for point in stage_points])
        )
        heating_prices = [
            point.price_eur_per_kwh
            for point in stage_points
            if point.outdoor_temperature_c < T_SETPOINT_C
        ]
        terminal_heat_value = (
            float(np.mean(heating_prices)) / HP_COP if heating_prices else 0.0
        )

        x0 = hub_state_array(state)
        self._mpc.x0 = x0
        forecast_source = getattr(self._mpc, "_forecast_source", None)
        if forecast_source is not None:
            forecast_source.activate(
                forecast,
                terminal_electric_value,
                terminal_heat_value,
            )
        try:
            raw_control = self._mpc.make_step(x0)
        finally:
            if forecast_source is not None:
                forecast_source.clear()
        stats = dict(self._mpc.solver_stats)
        diagnostics = DecisionDiagnostics(
            adapter="mpc",
            decision_status="success" if stats.get("success") is True else "failure",
            solver_success=bool(stats.get("success", False)),
            solver_return_status=str(stats.get("return_status", "")),
            solver_iterations=int(stats["iter_count"]) if "iter_count" in stats else None,
            solver_wall_seconds=(
                float(stats["t_wall_total"]) if "t_wall_total" in stats else None
            ),
            forecast_start_utc=forecast[0].timestamp_utc,
            forecast_end_utc=forecast[-1].timestamp_utc,
            terminal_electric_value_eur_per_kwh=terminal_electric_value,
            terminal_heat_value_eur_per_kwhth=terminal_heat_value,
        )
        if stats.get("success") is not True:
            return ControllerFailure(
                code="solver_failure",
                message=f"MPC solve failed: {stats.get('return_status', 'unknown')}",
                diagnostics=diagnostics,
            )
        control = hub_control_from_array(np.asarray(raw_control).reshape(-1))
        return ControlDecision(control=control, diagnostics=diagnostics)


def _enable_text_solver_stat_storage(mpc: object) -> None:
    """Let do-mpc 5.1 store its declared string-valued return_status field.

    do-mpc accepts ``return_status`` in ``store_solver_stats`` but its numeric data
    writer calls ``reshape`` on the raw string. Converting just text statistics to
    arrays preserves the configured field and the original data update behavior.
    """
    original_update = mpc.data.update

    def update_with_text_arrays(**values: object) -> None:
        original_update(
            **{
                key: np.asarray(value) if isinstance(value, str) else value
                for key, value in values.items()
            }
        )

    mpc.data.update = update_with_text_arrays


def build_mpc(
    hub_config: HubConfiguration,
    config: MpcConfiguration,
) -> tuple[object, object]:
    """
    Construct and return a configured do-mpc MPC controller for the hub.

    The controller is configured from physical and solver policy only. Forecast data
    is supplied later as an immutable N+1 view to ``MpcControllerAdapter.decide``.

    Returns
    -------
    (mpc, model) : configured do_mpc controller + symbolic model
    """
    if config.horizon_steps < 1:
        raise ValueError("MPC horizon_steps must be at least one")
    # ------------------------------------------------------------------
    # 1. Symbolic model
    # ------------------------------------------------------------------
    model = do_mpc.model.Model(model_type="discrete", symvar_type="SX")

    SOC_bat = model.set_variable("_x", "SOC_bat")   # [kWh]
    SOC_h2  = model.set_variable("_x", "SOC_h2")    # [kg]
    SOC_tes = model.set_variable("_x", "SOC_tes")   # [kWh]
    T_in    = model.set_variable("_x", "T_in")      # [degC]

    P_bat_ch  = model.set_variable("_u", "P_bat_ch")
    P_bat_dis = model.set_variable("_u", "P_bat_dis")
    P_elz     = model.set_variable("_u", "P_elz")
    P_fc      = model.set_variable("_u", "P_fc")
    P_hp      = model.set_variable("_u", "P_hp")
    P_eboiler = model.set_variable("_u", "P_eboiler")
    Q_tes_ch  = model.set_variable("_u", "Q_tes_ch")
    Q_tes_dis = model.set_variable("_u", "Q_tes_dis")
    vent      = model.set_variable("_u", "vent")

    price  = model.set_variable("_tvp", "price")
    P_pv   = model.set_variable("_tvp", "P_pv")
    P_load = model.set_variable("_tvp", "P_load")
    T_out  = model.set_variable("_tvp", "T_out")
    G_Wm2  = model.set_variable("_tvp", "G_Wm2")
    terminal_electric_value = model.set_variable(
        "_tvp", "terminal_electric_value"
    )
    terminal_heat_value = model.set_variable("_tvp", "terminal_heat_value")

    # Numerical simulation and this CasADi model bind the same physical owner.
    shared_step = hub_step_expressions(
        HubState(
            soc_battery_kwh=SOC_bat,
            soc_hydrogen_kg=SOC_h2,
            soc_thermal_kwh=SOC_tes,
            indoor_temperature_c=T_in,
        ),
        HubControl(
            battery_charge_kw=P_bat_ch,
            battery_discharge_kw=P_bat_dis,
            electrolyser_kw=P_elz,
            fuel_cell_kw=P_fc,
            heat_pump_kw=P_hp,
            electric_boiler_kw=P_eboiler,
            thermal_charge_kw=Q_tes_ch,
            thermal_discharge_kw=Q_tes_dis,
            ventilation_fraction=vent,
        ),
        ExogenousInputs(
            pv_kw=P_pv,
            electric_load_kw=P_load,
            price_eur_per_kwh=price,
            outdoor_temperature_c=T_out,
            irradiance_w_per_m2=G_Wm2,
        ),
        hub_config,
    )
    for field_name, model_name in STATE_MODEL_NAMES.items():
        expression = getattr(shared_step.successor, field_name)
        model.set_rhs(
            model_name,
            DM(expression) if isinstance(expression, (int, float)) else expression,
        )
    model.set_expression("P_grid", shared_step.flows.grid_kw)
    model.set_expression(
        "tes_charge_feas", shared_step.flows.thermal_charge_margin_kw
    )
    model.set_expression("Q_gen", shared_step.flows.generated_heat_kw)

    model.setup()

    # ------------------------------------------------------------------
    # 2. Controller
    # ------------------------------------------------------------------
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=config.horizon_steps,
        t_step=DT_H * 3600,      # do-mpc expects seconds
        n_robust=0,
        use_terminal_bounds=True,
        store_full_solution=True,
        store_solver_stats=[
            "success",
            "return_status",
            "iter_count",
            "t_wall_total",
        ],
        nlpsol_opts={
            "record_time": True,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": config.solver_max_iterations,
            "ipopt.tol": config.solver_tolerance,
            # do-mpc enforces continuity in state-scaled NLP coordinates.
            # Convert the physical state tolerance to the tightest scale.
            "ipopt.constr_viol_tol": (
                BALANCE_STATE_TOLERANCE / max(STATE_SCALE.values())
            ),
        },
    )

    # Variable scaling (wide magnitude range: T_in ~20 vs SOC_tes ~4000)
    for name, val in STATE_SCALE.items():
        mpc.scaling["_x", name] = val
    for name, val in INPUT_SCALE.items():
        mpc.scaling["_u", name] = val

    # ------------------------------------------------------------------
    # 3. Objective
    # ------------------------------------------------------------------
    # Imported energy pays wholesale + a transport/levy surcharge; exports earn wholesale.
    # import_kw is a smooth max(0, P_grid) = 0.5*(P + sqrt(P^2 + eps^2)) so the objective
    # stays C-infinity for IPOPT. eps = 1 kW; this charges a tiny phantom import (~0.5 kW
    # at P_grid=0, i.e. ~0.0125 EUR/h of surcharge) that biases the SOLVER objective only.
    # The realised/reported grid cost uses the EXACT max(0, P_grid), so published
    # savings are unaffected; tests/test_hub.py bounds this approximation error.
    IMPORT_SMOOTH_EPS2 = 1.0
    P_grid_expr = model.aux["P_grid"]
    import_kw = 0.5 * (P_grid_expr + sqrt(P_grid_expr ** 2 + IMPORT_SMOOTH_EPS2))
    lterm = (
        (price * P_grid_expr + GRID_IMPORT_FEE_EUR_KWH * import_kw) * DT_H
        + config.battery_wear_eur_per_kwh * (P_bat_ch + P_bat_dis) * DT_H
        + config.thermal_store_wear_eur_per_kwh
        * (Q_tes_ch + Q_tes_dis)
        * DT_H
        + config.electrolyser_wear_eur_per_kwh * P_elz * DT_H
        + config.fuel_cell_wear_eur_per_kwh * P_fc * DT_H
        + config.complementarity_weight
        * (
            P_bat_ch * P_bat_dis / BAT_P_MAX_KW
            + Q_tes_ch * Q_tes_dis / TES_P_MAX_KW
            + P_elz * P_fc / FC_P_MAX_KW
        )
    )

    # Terminal cost: reward stored energy valued at the price it would be used at
    # (a cost-to-go proxy). Battery and H2 convert back to electricity, so they are
    # valued at the horizon-average price. TES heat only displaces FUTURE HEATING
    # electricity, so it is valued at the average price during heating hours / COP —
    # this is ~0 in summer (no heating need), which prevents pointless heat hoarding.
    stored_value = DM(0.0)
    if hub_config.capabilities.battery:
        stored_value += terminal_electric_value * ETA_BAT_DIS * SOC_bat
    if hub_config.capabilities.hydrogen:
        stored_value += (
            terminal_electric_value * ETA_FC_E * E_H2_LHV_KWH_KG * SOC_h2
        )
    if hub_config.capabilities.thermal_store:
        stored_value += terminal_heat_value * SOC_tes
    mterm = -config.terminal_weight * stored_value

    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(
        P_bat_ch=config.input_move_weight,
        P_bat_dis=config.input_move_weight,
        P_hp=config.input_move_weight,
        P_eboiler=config.input_move_weight,
    )

    # ------------------------------------------------------------------
    # 4. Constraints
    # ------------------------------------------------------------------
    # Grid connection limits on the derived net import (proper one-sided inequalities)
    mpc.set_nl_cons("grid_import_max", model.aux["P_grid"], ub=GRID_P_MAX_KW)
    mpc.set_nl_cons("grid_export_max", -model.aux["P_grid"], ub=GRID_P_MAX_KW)
    # TES can only charge from generated heat
    mpc.set_nl_cons("tes_charge_feas", model.aux["tes_charge_feas"], ub=0.0)
    # Soft indoor-temperature comfort band [16, 24] degC
    hard_temperature_lower, hard_temperature_upper = operational_state_bounds(
        hub_config
    )["indoor_temperature_c"]
    mpc.set_nl_cons(
        "T_upper",
        T_in,
        ub=T_MAX_C,
        soft_constraint=True,
        penalty_term_cons=config.comfort_slack_weight,
        maximum_violation=hard_temperature_upper - T_MAX_C,
    )
    mpc.set_nl_cons(
        "T_lower",
        -T_in,
        ub=-T_MIN_C,
        soft_constraint=True,
        penalty_term_cons=config.comfort_slack_weight,
        maximum_violation=T_MIN_C - hard_temperature_lower,
    )

    # ------------------------------------------------------------------
    # 5. Bounds
    # ------------------------------------------------------------------
    for field_name, (lower, upper) in operational_state_bounds(hub_config).items():
        model_name = STATE_MODEL_NAMES[field_name]
        mpc.bounds["lower", "_x", model_name] = lower
        mpc.bounds["upper", "_x", model_name] = upper
        mpc.terminal_bounds["lower", model_name] = lower
        mpc.terminal_bounds["upper", model_name] = upper
    for field_name, (lower, upper) in control_bounds(hub_config).items():
        model_name = CONTROL_MODEL_NAMES[field_name]
        mpc.bounds["lower", "_u", model_name] = lower
        mpc.bounds["upper", "_u", model_name] = upper

    # ------------------------------------------------------------------
    # 6. Time-varying parameters (causal N+1 forecast view)
    # ------------------------------------------------------------------
    tvp_template = mpc.get_tvp_template()
    forecast_source = _ForecastTVPSource(tvp_template, config.horizon_steps)
    mpc.set_tvp_fun(forecast_source)
    mpc.setup()
    _enable_text_solver_stat_storage(mpc)
    mpc._forecast_source = forecast_source
    return mpc, model
