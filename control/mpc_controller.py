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

import numpy as np
import do_mpc
from casadi import sqrt

from models.hub_model import (
    BAT_P_MAX_KW, ETA_BAT_CH, ETA_BAT_DIS,
    ETA_ELZ, E_H2_LHV_KWH_KG,
    FC_P_MAX_KW, ETA_FC_E, ETA_FC_H,
    HP_COP, ETA_EBOILER,
    TES_P_MAX_KW, ETA_TES_STANDING,
    GRID_P_MAX_KW, GRID_IMPORT_FEE_EUR_KWH,
    C_AIR_KWH_K, U_EFF_KW_K, K_VENT_KW_K, SOLAR_GAIN_FRAC, FLOOR_AREA_M2,
    Q_CROP_LATENT_KW, T_MIN_C, T_MAX_C, T_SETPOINT_C,
    state_bounds, input_bounds, STATE_SCALE, INPUT_SCALE,
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


def build_mpc(price_forecast: np.ndarray,
              pv_forecast: np.ndarray,
              load_elec_forecast: np.ndarray,
              temp_out_forecast: np.ndarray,
              irr_forecast: np.ndarray,
              n_horizon: int = N_HORIZON,
              disable_h2: bool = False,
              disable_tes: bool = False,
              terminal_weight: float = W_TERMINAL) -> tuple:
    """
    Construct and return a configured do-mpc MPC controller for the hub.

    Parameters
    ----------
    price_forecast     : [n] array, EUR/kWh
    pv_forecast        : [n] array, kW
    load_elec_forecast : [n] array, kW (lighting + base electrical load)
    temp_out_forecast  : [n] array, degC
    irr_forecast       : [n] array, W/m2
    n_horizon          : prediction horizon [h]; set 1 for a myopic (no-foresight) ablation
    disable_h2         : if True, lock electrolyser + fuel cell off (no-H2 ablation)
    disable_tes        : if True, lock the thermal store off (no-TES ablation)
    terminal_weight    : weight on the terminal stored-energy value (0 disables it)

    Returns
    -------
    (mpc, model) : configured do_mpc controller + symbolic model
    """
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

    # --- Conversions ---
    Q_hp      = HP_COP * P_hp
    Q_eboiler = ETA_EBOILER * P_eboiler
    h2_chem   = P_fc / ETA_FC_E
    m_h2_fc   = h2_chem / E_H2_LHV_KWH_KG
    Q_fc_heat = ETA_FC_H * h2_chem
    m_h2_prod = ETA_ELZ * P_elz / E_H2_LHV_KWH_KG

    Q_gen   = Q_hp + Q_eboiler + Q_fc_heat
    Q_solar = SOLAR_GAIN_FRAC * G_Wm2 * FLOOR_AREA_M2 / 1000.0
    Q_air   = Q_gen - Q_tes_ch + Q_tes_dis + Q_solar

    # --- Dynamics ---
    SOC_bat_next = (SOC_bat
                    + ETA_BAT_CH * P_bat_ch * DT_H
                    - (P_bat_dis / ETA_BAT_DIS) * DT_H)
    SOC_h2_next = SOC_h2 + (m_h2_prod - m_h2_fc) * DT_H
    SOC_tes_next = (ETA_TES_STANDING * SOC_tes
                    + Q_tes_ch * DT_H - Q_tes_dis * DT_H)

    # Implicit-Euler greenhouse temperature (smooth in vent for IPOPT)
    C_over_dt = C_AIR_KWH_K / DT_H
    Geff = U_EFF_KW_K + K_VENT_KW_K * vent
    T_in_next = (C_over_dt * T_in + Q_air + Geff * T_out - Q_CROP_LATENT_KW) / (C_over_dt + Geff)

    model.set_rhs("SOC_bat", SOC_bat_next)
    model.set_rhs("SOC_h2",  SOC_h2_next)
    model.set_rhs("SOC_tes", SOC_tes_next)
    model.set_rhs("T_in",    T_in_next)

    # Grid is the slack bus: net import is fully determined by the electricity
    # balance, so it is an EXPRESSION (not a free input). This makes the balance
    # hold exactly by construction and avoids the degenerate squared-equality
    # constraint (zero gradient at feasibility) used in the original formulation.
    P_grid = (P_load + P_bat_ch + P_elz + P_hp + P_eboiler
              - P_pv - P_bat_dis - P_fc)
    tes_charge_feas = Q_tes_ch - Q_gen   # <= 0 : cannot charge TES from nothing
    model.set_expression("P_grid", P_grid)
    model.set_expression("tes_charge_feas", tes_charge_feas)
    model.set_expression("Q_gen", Q_gen)

    model.setup()

    # ------------------------------------------------------------------
    # 2. Controller
    # ------------------------------------------------------------------
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=n_horizon,
        t_step=DT_H * 3600,      # do-mpc expects seconds
        n_robust=0,
        store_full_solution=True,
        nlpsol_opts={
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 800,
            "ipopt.tol": 1e-6,
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
    # Imported energy pays wholesale + a transport/levy surcharge; exports earn
    # wholesale only. import_kw is a smooth max(0, P_grid) so IPOPT stays differentiable.
    P_grid_expr = model.aux["P_grid"]
    import_kw = 0.5 * (P_grid_expr + sqrt(P_grid_expr ** 2 + 1.0))
    lterm = (
        (price * P_grid_expr + GRID_IMPORT_FEE_EUR_KWH * import_kw) * DT_H
        + W_BAT_THRU * (P_bat_ch + P_bat_dis) * DT_H
        + W_TES_THRU * (Q_tes_ch + Q_tes_dis) * DT_H
        + W_ELZ_WEAR * P_elz * DT_H
        + W_FC_WEAR * P_fc * DT_H
        + W_COMPL * (P_bat_ch * P_bat_dis / BAT_P_MAX_KW
                     + Q_tes_ch * Q_tes_dis / TES_P_MAX_KW
                     + P_elz * P_fc / FC_P_MAX_KW)
    )

    # Terminal cost: reward stored energy valued at the price it would be used at
    # (a cost-to-go proxy). Battery and H2 convert back to electricity, so they are
    # valued at the horizon-average price. TES heat only displaces FUTURE HEATING
    # electricity, so it is valued at the average price during heating hours / COP —
    # this is ~0 in summer (no heating need), which prevents pointless heat hoarding.
    lam_avg = float(np.mean(price_forecast))
    heating_mask = temp_out_forecast < T_SETPOINT_C
    if heating_mask.any():
        lam_heat = float(np.mean(price_forecast[heating_mask])) / HP_COP
    else:
        lam_heat = 0.0
    stored_value = (lam_avg * ETA_BAT_DIS * SOC_bat                       # battery -> elec
                    + lam_avg * ETA_FC_E * E_H2_LHV_KWH_KG * SOC_h2       # H2 -> elec (lossy)
                    + lam_heat * SOC_tes)                                 # TES -> heating elec
    mterm = -terminal_weight * stored_value

    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(P_bat_ch=W_RTERM, P_bat_dis=W_RTERM,
                  P_hp=W_RTERM, P_eboiler=W_RTERM)

    # ------------------------------------------------------------------
    # 4. Constraints
    # ------------------------------------------------------------------
    # Grid connection limits on the derived net import (proper one-sided inequalities)
    mpc.set_nl_cons("grid_import_max", model.aux["P_grid"], ub=GRID_P_MAX_KW)
    mpc.set_nl_cons("grid_export_max", -model.aux["P_grid"], ub=GRID_P_MAX_KW)
    # TES can only charge from generated heat
    mpc.set_nl_cons("tes_charge_feas", model.aux["tes_charge_feas"], ub=0.0)
    # Soft indoor-temperature comfort band [16, 24] degC
    mpc.set_nl_cons("T_upper", T_in, ub=T_MAX_C,
                    soft_constraint=True, penalty_term_cons=W_TBAND, maximum_violation=10.0)
    mpc.set_nl_cons("T_lower", -T_in, ub=-T_MIN_C,
                    soft_constraint=True, penalty_term_cons=W_TBAND, maximum_violation=10.0)

    # ------------------------------------------------------------------
    # 5. Bounds
    # ------------------------------------------------------------------
    sb, ib = state_bounds(), input_bounds()
    for s in ("SOC_bat", "SOC_h2", "SOC_tes", "T_in"):
        mpc.bounds["lower", "_x", s] = sb[s][0]
        mpc.bounds["upper", "_x", s] = sb[s][1]
    for uname, (lo, hi) in ib.items():
        mpc.bounds["lower", "_u", uname] = lo
        mpc.bounds["upper", "_u", uname] = hi

    # Ablations: lock selected assets off by pinning their upper bound to zero.
    if disable_h2:
        mpc.bounds["upper", "_u", "P_elz"] = 0.0
        mpc.bounds["upper", "_u", "P_fc"] = 0.0
    if disable_tes:
        mpc.bounds["upper", "_u", "Q_tes_ch"] = 0.0
        mpc.bounds["upper", "_u", "Q_tes_dis"] = 0.0

    # ------------------------------------------------------------------
    # 6. Time-varying parameters (perfect-foresight forecasts)
    # ------------------------------------------------------------------
    forecasts = {
        "price":  price_forecast,
        "P_pv":   pv_forecast,
        "P_load": load_elec_forecast,
        "T_out":  temp_out_forecast,
        "G_Wm2":  irr_forecast,
    }
    n = len(price_forecast)
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        k = int(round(float(np.squeeze(t_now)) / (DT_H * 3600)))
        for i in range(n_horizon + 1):
            idx = min(k + i, n - 1)
            for key, arr in forecasts.items():
                tvp_template["_tvp", i, key] = float(arr[idx])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    return mpc, model
