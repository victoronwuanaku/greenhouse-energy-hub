"""
Multi-carrier greenhouse energy hub — component models.

Framework: Geidl & Andersson (2007) energy hub formulation.
Location:  Division Q, Monster / Westland, Netherlands (1 ha high-tech greenhouse).

This module defines the physical plant: asset sizing, control/state bounds, and the
discrete-time state-transition function used by both the MPC internal model
(`control/mpc_controller.py`, symbolic CasADi mirror) and the rolling-horizon plant
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
that SPROUT replaces: cheap/surplus electricity -> H2 -> electricity + heat on demand.

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
def state_bounds() -> dict:
    """(lower, upper) bounds for each state [kWh / kg / kWh / degC]."""
    return {
        "SOC_bat": (BAT_SOC_MIN * BAT_CAPACITY_KWH, BAT_SOC_MAX * BAT_CAPACITY_KWH),
        "SOC_h2":  (H2_SOC_MIN * H2_CAPACITY_KG,    H2_SOC_MAX * H2_CAPACITY_KG),
        "SOC_tes": (TES_SOC_MIN * TES_CAPACITY_KWH, TES_SOC_MAX * TES_CAPACITY_KWH),
        "T_in":    (T_HARD_MIN_C, T_HARD_MAX_C),
    }


def input_bounds() -> dict:
    """(lower, upper) bounds for each control input."""
    return {
        "P_bat_ch":  (0.0, BAT_P_MAX_KW),
        "P_bat_dis": (0.0, BAT_P_MAX_KW),
        "P_elz":     (0.0, ELZ_P_MAX_KW),
        "P_fc":      (0.0, FC_P_MAX_KW),
        "P_hp":      (0.0, HP_P_MAX_KW),
        "P_eboiler": (0.0, EBOILER_P_MAX_KW),
        "Q_tes_ch":  (0.0, TES_P_MAX_KW),
        "Q_tes_dis": (0.0, TES_P_MAX_KW),
        "vent":      (0.0, 1.0),
        # P_grid is NOT a control: it is the derived slack bus (see hub_dynamics).
    }


# ---------------------------------------------------------------------------
# Derived hub quantities (shared by plant and symbolic model)
# ---------------------------------------------------------------------------
def fuel_cell_outputs(P_fc):
    """Fuel-cell H2 chemical draw [kW], H2 mass rate [kg/h], recovered heat [kW]."""
    h2_chem_kW = P_fc / ETA_FC_E
    m_h2_kg_h = h2_chem_kW / E_H2_LHV_KWH_KG
    Q_fc_heat = ETA_FC_H * h2_chem_kW
    return h2_chem_kW, m_h2_kg_h, Q_fc_heat


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
def hub_dynamics(x: dict, u: dict, p: dict) -> dict:
    """
    One-step discrete-time state transition for the greenhouse energy hub.

    Parameters
    ----------
    x : dict — {SOC_bat [kWh], SOC_h2 [kg], SOC_tes [kWh], T_in [degC]}
    u : dict — control inputs (see module docstring)
    p : dict — {P_pv [kW], P_load [kW], price [EUR/kWh], T_out [degC], G_Wm2 [W/m2]}

    Returns
    -------
    x_next : dict — states at next timestep
    metrics : dict — derived diagnostics (residuals, costs, heat/H2 flows)
    """
    SOC_bat, SOC_h2, SOC_tes, T_in = x["SOC_bat"], x["SOC_h2"], x["SOC_tes"], x["T_in"]

    P_bat_ch, P_bat_dis = u["P_bat_ch"], u["P_bat_dis"]
    P_elz, P_fc = u["P_elz"], u["P_fc"]
    P_hp, P_eboiler = u["P_hp"], u["P_eboiler"]
    Q_tes_ch, Q_tes_dis = u["Q_tes_ch"], u["Q_tes_dis"]
    vent = u["vent"]

    P_pv, P_load = p["P_pv"], p["P_load"]
    T_out = p["T_out"]
    G_Wm2 = p.get("G_Wm2", 0.0)

    # --- Conversions ---
    Q_hp = HP_COP * P_hp                          # heat pump thermal output [kW]
    Q_eboiler = ETA_EBOILER * P_eboiler           # e-boiler thermal output [kW]
    _, m_h2_fc, Q_fc_heat = fuel_cell_outputs(P_fc)
    m_h2_prod = ETA_ELZ * P_elz / E_H2_LHV_KWH_KG  # H2 produced [kg/h]

    Q_gen = Q_hp + Q_eboiler + Q_fc_heat          # total generated heat [kW]
    Q_solar = SOLAR_GAIN_FRAC * G_Wm2 * FLOOR_AREA_M2 / 1000.0  # solar gain [kW]
    Q_air = Q_gen - Q_tes_ch + Q_tes_dis + Q_solar             # net heat to air [kW]

    # --- Grid is the slack bus: net import determined by the electricity balance ---
    #   P_grid + P_pv + P_bat_dis + P_fc = P_load + P_bat_ch + P_elz + P_hp + P_eboiler
    # so the balance holds exactly by construction (no equality constraint needed).
    P_grid = (P_load + P_bat_ch + P_elz + P_hp + P_eboiler
              - P_pv - P_bat_dis - P_fc)

    # --- State updates ---
    SOC_bat_next = (SOC_bat
                    + ETA_BAT_CH * P_bat_ch * DT_H
                    - (P_bat_dis / ETA_BAT_DIS) * DT_H)
    SOC_h2_next = SOC_h2 + (m_h2_prod - m_h2_fc) * DT_H
    SOC_tes_next = (ETA_TES_STANDING * SOC_tes
                    + Q_tes_ch * DT_H
                    - Q_tes_dis * DT_H)
    T_in_next = greenhouse_temperature_next(T_in, Q_air, vent, T_out)

    x_next = {
        "SOC_bat": SOC_bat_next,
        "SOC_h2":  SOC_h2_next,
        "SOC_tes": SOC_tes_next,
        "T_in":    T_in_next,
    }

    metrics = {
        "P_grid_kW": P_grid,
        "elec_residual_kW": 0.0,   # balance is exact by construction (slack bus)
        "tes_charge_excess_kW": max(0.0, Q_tes_ch - Q_gen),  # >0 => infeasible TES charge
        "Q_hp_kW": Q_hp,
        "Q_eboiler_kW": Q_eboiler,
        "Q_fc_heat_kW": Q_fc_heat,
        "Q_air_kW": Q_air,
        "m_h2_prod_kg_h": m_h2_prod,
        "m_h2_fc_kg_h": m_h2_fc,
        # import pays wholesale + transport/levy; export earns wholesale only
        "grid_cost_EUR": (P_grid * p.get("price", 0.0)
                          + GRID_IMPORT_FEE_EUR_KWH * max(0.0, P_grid)) * DT_H,
        # violation of the temperature REACHED during the hour (the controlled result)
        "T_violation_C": max(0.0, T_in_next - T_MAX_C) + max(0.0, T_MIN_C - T_in_next),
    }
    return x_next, metrics


# ---------------------------------------------------------------------------
# Initial state (typical operating point, mid-inventory)
# ---------------------------------------------------------------------------
def initial_state() -> dict:
    """Physically reasonable initial hub state."""
    return {
        "SOC_bat": 0.50 * BAT_CAPACITY_KWH,   # 500 kWh (50%)
        "SOC_h2":  0.30 * H2_CAPACITY_KG,     # 60 kg (30%)
        "SOC_tes": 0.40 * TES_CAPACITY_KWH,   # 1600 kWh (40%)
        "T_in":    T_SETPOINT_C,              # at setpoint
    }


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
