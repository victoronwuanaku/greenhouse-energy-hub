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

Baseline controller (naive, no look-ahead)
------------------------------------------
  - Heat to hold the setpoint reactively: heat pump first, then e-boiler, then TES.
  - Vent fully when solar gain would push the air above the comfort band.
  - Battery charges from PV surplus, discharges when price > 0.12 EUR/kWh.
  - No hydrogen use, no thermal pre-storage, no price look-ahead.

Usage
-----
    python control/rolling_horizon.py [--days 14] [--start-month 6] [--mode both]
    # winter fortnight: --start-month 1 ;  summer fortnight: --start-month 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hub_model import (
    hub_dynamics, initial_state, state_bounds,
    BAT_P_MAX_KW, ETA_BAT_CH, ETA_BAT_DIS,
    HP_P_MAX_KW, HP_COP, EBOILER_P_MAX_KW, ETA_EBOILER,
    TES_P_MAX_KW, ETA_FC_E, E_H2_LHV_KWH_KG,
    C_AIR_KWH_K, U_EFF_KW_K, SOLAR_GAIN_FRAC, FLOOR_AREA_M2,
    Q_CROP_LATENT_KW, T_SETPOINT_C, T_MAX_C, DT_H,
)
from control.mpc_controller import build_mpc, N_HORIZON

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Decision variables solved by the MPC (P_grid is the derived slack bus, not a control)
INPUT_NAMES = ["P_bat_ch", "P_bat_dis", "P_elz", "P_fc", "P_hp",
               "P_eboiler", "Q_tes_ch", "Q_tes_dis", "vent"]


def stored_equiv_kwh(soc_bat: float, soc_h2: float, soc_tes: float) -> float:
    """
    Electricity-equivalent value of stored energy [kWh]: the recoverable
    electricity (battery, H2 via fuel cell) plus heat valued at the HP COP.
    Used to mark-to-market terminal inventory so the controller comparison is
    not skewed by one controller ending with more/less stored energy.
    """
    return (ETA_BAT_DIS * soc_bat
            + ETA_FC_E * E_H2_LHV_KWH_KG * soc_h2
            + (1.0 / HP_COP) * soc_tes)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(start_month: int = 6, n_days: int = 14) -> pd.DataFrame:
    """
    Load and align PV/weather (PVGIS) and price (energy-charts) data to a common
    hourly index for the requested window. PV/weather and prices may come from
    different years; they are aligned by day-of-year and hour.
    """
    pv = pd.read_csv(DATA_DIR / "pv_profile.csv",
                     index_col="timestamp", parse_dates=True)
    prices = pd.read_csv(DATA_DIR / "grid_price_signal.csv",
                         index_col="timestamp", parse_dates=True)
    demand = pd.read_csv(DATA_DIR / "demand_profile.csv",
                         index_col="timestamp", parse_dates=True)

    price_year = prices.index[0].year
    pv_year = pv.index[0].year

    start = pd.Timestamp(f"{price_year}-{start_month:02d}-01", tz="UTC")
    end = start + pd.Timedelta(days=n_days)
    prices_window = prices.loc[start:end].iloc[:-1]
    n_steps = len(prices_window)

    pv_start = pd.Timestamp(f"{pv_year}-{start_month:02d}-01", tz="UTC")
    pv_end = pv_start + pd.Timedelta(hours=n_steps)
    pv_window = pv.loc[pv_start:pv_end].iloc[:n_steps]
    dem_window = demand.loc[pv_start:pv_end].iloc[:n_steps]

    df = pd.DataFrame(index=prices_window.index)
    df["price_EUR_kWh"] = prices_window["price_EUR_kWh"].values
    df["price_EUR_MWh"] = prices_window["price_EUR_MWh"].values
    df["P_pv_kW"] = pv_window["P_kW"].values * 500.0   # scale 1 kWp -> 500 kWp
    df["G_Wm2"] = pv_window["G_Wm2"].values
    df["T_out_C"] = pv_window["T2m_C"].values
    df["P_elec_kW"] = dem_window["P_elec_kW"].values

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
    """Naive reactive dispatch: hold setpoint with HP/e-boiler/TES, simple battery."""
    sb = state_bounds()
    T_in, T_out, G = x["T_in"], p["T_out"], p["G_Wm2"]
    P_pv, P_load, price = p["P_pv"], p["P_load"], p["price"]

    Q_solar = SOLAR_GAIN_FRAC * G * FLOOR_AREA_M2 / 1000.0
    C = C_AIR_KWH_K / DT_H
    U = U_EFF_KW_K

    # Generated heat needed to reach the setpoint this step (vents closed, no TES charge)
    Q_air_req = T_SETPOINT_C * (C + U) - C * T_in - U * T_out + Q_CROP_LATENT_KW
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


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
def run_simulation(df: pd.DataFrame, mode: str = "mpc") -> pd.DataFrame:
    """Run one full simulation (mode = 'mpc' or 'baseline') over the window."""
    assert mode in ("mpc", "baseline"), f"Unknown mode: {mode}"
    n = len(df)
    prices = df["price_EUR_kWh"].values
    pv = df["P_pv_kW"].values
    p_load = df["P_elec_kW"].values
    t_out = df["T_out_C"].values
    irr = df["G_Wm2"].values

    x = initial_state()

    if mode == "mpc":
        print(f"\nBuilding MPC controller (horizon={N_HORIZON}h)...")
        mpc, _ = build_mpc(
            price_forecast=prices, pv_forecast=pv,
            load_elec_forecast=p_load, temp_out_forecast=t_out,
            irr_forecast=irr,
        )
        x0 = np.array([x["SOC_bat"], x["SOC_h2"], x["SOC_tes"], x["T_in"]])
        mpc.x0 = x0
        mpc.set_initial_guess()
        print("MPC controller ready.\n")

    records = []
    for k in range(n):
        p = {"P_pv": pv[k], "P_load": p_load[k], "price": prices[k],
             "T_out": t_out[k], "G_Wm2": irr[k]}

        if mode == "mpc":
            x0 = np.array([[x["SOC_bat"]], [x["SOC_h2"]], [x["SOC_tes"]], [x["T_in"]]])
            mpc.x0 = x0
            mpc.make_step(x0)
            u = {name: float(np.squeeze(mpc.u0[name])) for name in INPUT_NAMES}
        else:
            u = baseline_control(x, p)

        x_next, metrics = hub_dynamics(x, u, p)

        rec = {
            "timestamp": df.index[k],
            "SOC_bat_kWh": x["SOC_bat"], "SOC_h2_kg": x["SOC_h2"],
            "SOC_tes_kWh": x["SOC_tes"], "T_in_C": x["T_in"],
            **{f"u_{key}": val for key, val in u.items()},
            "u_P_grid": metrics["P_grid_kW"],   # derived slack bus
            "P_pv_kW": pv[k], "P_load_kW": p_load[k],
            "price_EUR_kWh": prices[k], "T_out_C": t_out[k], "G_Wm2": irr[k],
            "grid_cost_EUR": metrics["grid_cost_EUR"],
            "elec_residual_kW": metrics["elec_residual_kW"],
            "Q_air_kW": metrics["Q_air_kW"],
            "T_violation_C": metrics["T_violation_C"],
        }
        records.append(rec)
        x = x_next

        if (k + 1) % 24 == 0:
            cum = sum(r["grid_cost_EUR"] for r in records)
            print(f"  Step {k+1:4d}/{n}  SOC_bat={x['SOC_bat']:.0f} "
                  f"SOC_h2={x['SOC_h2']:.1f} SOC_tes={x['SOC_tes']:.0f} "
                  f"T_in={x['T_in']:.1f}C  cum cost EUR{cum:.0f}")

    return pd.DataFrame(records).set_index("timestamp")


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

    print("=" * 65)
    print("  Greenhouse Energy Hub MPC - Rolling Horizon Simulation")
    print("  Location: Westland/Monster, Netherlands")
    print("=" * 65)

    df = load_data(start_month=args.start_month, n_days=args.days)
    results = {}

    if args.mode in ("baseline", "both"):
        print("\n[1/2] BASELINE (rule-based)...")
        results["baseline"] = run_simulation(df, mode="baseline")
        results["baseline"].to_csv(RESULTS_DIR / "baseline_results.csv")

    if args.mode in ("mpc", "both"):
        print("\n[2/2] MPC...")
        results["mpc"] = run_simulation(df, mode="mpc")
        results["mpc"].to_csv(RESULTS_DIR / "mpc_results.csv")

    if "baseline" in results and "mpc" in results:
        base = results["baseline"]["grid_cost_EUR"].sum()
        mpc_c = results["mpc"]["grid_cost_EUR"].sum()
        saving = base - mpc_c
        pct = 100 * saving / abs(base) if base != 0 else 0.0
        bv = results["baseline"]["T_violation_C"].sum()
        mv = results["mpc"]["T_violation_C"].sum()

        # Inventory-adjusted (mark-to-market) cost: charge each controller for the
        # net stored energy it consumed over the window, valued at the mean price.
        x0 = initial_state()
        init_eq = stored_equiv_kwh(x0["SOC_bat"], x0["SOC_h2"], x0["SOC_tes"])
        settle = results["mpc"]["price_EUR_kWh"].mean()

        def adjusted(r):
            fin_eq = stored_equiv_kwh(r["SOC_bat_kWh"].iloc[-1],
                                      r["SOC_h2_kg"].iloc[-1],
                                      r["SOC_tes_kWh"].iloc[-1])
            return r["grid_cost_EUR"].sum() + settle * (init_eq - fin_eq)

        base_adj, mpc_adj = adjusted(results["baseline"]), adjusted(results["mpc"])
        adj_saving = base_adj - mpc_adj
        adj_pct = 100 * adj_saving / abs(base_adj) if base_adj != 0 else 0.0

        print("\n" + "=" * 65)
        print("  RESULTS SUMMARY")
        print("=" * 65)
        print(f"  Baseline grid cost      : EUR {base:>9.2f}   T-band viol {bv:>6.1f} degC.h")
        print(f"  MPC grid cost           : EUR {mpc_c:>9.2f}   T-band viol {mv:>6.1f} degC.h")
        print(f"  Saving (raw grid cost)  : EUR {saving:>9.2f}   ({pct:+.1f}%)")
        print(f"  Saving (inventory-adj.) : EUR {adj_saving:>9.2f}   ({adj_pct:+.1f}%)")
        print("=" * 65)


if __name__ == "__main__":
    main()
