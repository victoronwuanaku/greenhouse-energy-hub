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
    TES_P_MAX_KW,
    C_AIR_KWH_K, U_EFF_KW_K, SOLAR_GAIN_FRAC, FLOOR_AREA_M2,
    Q_CROP_LATENT_KW, T_MIN_C, T_MAX_C, DT_H,
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


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
def run_simulation(df: pd.DataFrame, mode: str = "mpc", **mpc_kwargs) -> pd.DataFrame:
    """
    Run one full simulation (mode = 'mpc' or 'baseline') over the window.

    Extra keyword arguments (n_horizon, disable_h2, disable_tes, terminal_weight)
    are forwarded to build_mpc to support ablation studies.
    """
    assert mode in ("mpc", "baseline"), f"Unknown mode: {mode}"
    n = len(df)
    prices = df["price_EUR_kWh"].values
    pv = df["P_pv_kW"].values
    p_load = df["P_elec_kW"].values
    t_out = df["T_out_C"].values
    irr = df["G_Wm2"].values

    x = initial_state()

    if mode == "mpc":
        from control.mpc_controller import build_mpc, N_HORIZON   # lazy (heavy do-mpc import)
        print(f"\nBuilding MPC controller (horizon={mpc_kwargs.get('n_horizon', N_HORIZON)}h, "
              f"{', '.join(k for k, v in mpc_kwargs.items() if v) or 'full'})...")
        mpc, _ = build_mpc(
            price_forecast=prices, pv_forecast=pv,
            load_elec_forecast=p_load, temp_out_forecast=t_out,
            irr_forecast=irr, **mpc_kwargs,
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
            "is_terminal": False,
        }
        records.append(rec)
        x = x_next

        if (k + 1) % 24 == 0:
            cum = sum(r["grid_cost_EUR"] for r in records)
            print(f"  Step {k+1:4d}/{n}  SOC_bat={x['SOC_bat']:.0f} "
                  f"SOC_h2={x['SOC_h2']:.1f} SOC_tes={x['SOC_tes']:.0f} "
                  f"T_in={x['T_in']:.1f}C  cum cost EUR{cum:.0f}")

    # Append the TRUE terminal state (state after the final control) as the last row,
    # so inventory settlement and SOC plots use the real end state, not the pre-step
    # state of the last step. Controls/cost are zero here (no step is taken).
    term = {"timestamp": df.index[-1] + pd.Timedelta(hours=DT_H),
            "SOC_bat_kWh": x["SOC_bat"], "SOC_h2_kg": x["SOC_h2"],
            "SOC_tes_kWh": x["SOC_tes"], "T_in_C": x["T_in"],
            **{f"u_{name}": 0.0 for name in INPUT_NAMES}, "u_P_grid": 0.0,
            "P_pv_kW": np.nan, "P_load_kW": np.nan, "price_EUR_kWh": np.nan,
            "T_out_C": np.nan, "G_Wm2": np.nan, "grid_cost_EUR": 0.0,
            "elec_residual_kW": 0.0, "Q_air_kW": np.nan, "T_violation_C": 0.0,
            "is_terminal": True}
    records.append(term)

    return pd.DataFrame(records).set_index("timestamp")


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
        res = run_simulation(df, mode=mode)
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
