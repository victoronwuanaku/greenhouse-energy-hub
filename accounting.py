"""
Centralised economic & physical accounting for the greenhouse energy hub.

Single source of truth so the CLI summary, the analysis notebook, and the tests
all compute savings the same way (avoids drift between reported numbers).

Two cost notions:
  * raw grid cost            — sum of per-step grid cost (import pays wholesale +
                               transport/levy surcharge; export earns wholesale).
  * inventory-adjusted cost  — raw cost plus the value of net stored energy consumed
                               over the window, marked to market at the mean price,
                               so a controller is not rewarded for merely ending with
                               more/less storage.
"""

from models.hub_model import ETA_BAT_DIS, ETA_FC_E, E_H2_LHV_KWH_KG, HP_COP


def stored_equiv_kwh(soc_bat: float, soc_h2: float, soc_tes: float) -> float:
    """Electricity-equivalent value of stored energy [kWh]: recoverable electricity
    (battery, H2 via fuel cell) plus heat valued at the heat-pump COP."""
    return (ETA_BAT_DIS * soc_bat
            + ETA_FC_E * E_H2_LHV_KWH_KG * soc_h2
            + (1.0 / HP_COP) * soc_tes)


def stored_equiv_from_row(row) -> float:
    """Electricity-equivalent of a results row (uses the SOC_* columns)."""
    return stored_equiv_kwh(row["SOC_bat_kWh"], row["SOC_h2_kg"], row["SOC_tes_kWh"])


def inventory_adjusted_cost(results_df, init_equiv: float, settle_price: float) -> float:
    """Raw grid cost + mark-to-market of net stored energy consumed over the window.

    The final row of a results frame is the true terminal state (see
    rolling_horizon.run_simulation, which appends it), so iloc[-1] is exact.
    """
    final_equiv = stored_equiv_from_row(results_df.iloc[-1])
    return results_df["grid_cost_EUR"].sum() + settle_price * (init_equiv - final_equiv)


def saving_pct(baseline_cost: float, mpc_cost: float) -> float:
    """Percentage cost reduction of MPC vs baseline."""
    return 100.0 * (baseline_cost - mpc_cost) / abs(baseline_cost) if baseline_cost else 0.0
