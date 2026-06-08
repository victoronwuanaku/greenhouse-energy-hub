"""
Physical and control invariants for the greenhouse energy hub.

Two groups of tests:
  1. Unit tests on the plant model (`hub_dynamics`) — fast, no solver.
  2. A short live MPC roll-out — builds the do-mpc controller and checks that the
     applied control respects the electricity balance and bounds.
  3. Invariants on the saved simulation results (run rolling_horizon.py first):
     balance ~0, no simultaneous charge/discharge, SOC within bounds, comfort band,
     and MPC total cost <= baseline.

Run:  pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.hub_model import (  # noqa: E402
    hub_dynamics, initial_state, state_bounds, input_bounds,
    ETA_ELZ, ETA_FC_E, E_H2_LHV_KWH_KG,
    BAT_CAPACITY_KWH, H2_CAPACITY_KG, TES_CAPACITY_KWH,
    T_HARD_MIN_C, T_HARD_MAX_C,
)

RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# 1. Plant-model unit tests
# ---------------------------------------------------------------------------
def _sample_controls():
    return {
        "P_bat_ch": 100.0, "P_bat_dis": 0.0, "P_elz": 50.0, "P_fc": 0.0,
        "P_hp": 100.0, "P_eboiler": 200.0, "Q_tes_ch": 150.0, "Q_tes_dis": 0.0,
        "vent": 0.0,
    }


def _sample_params():
    return {"P_pv": 80.0, "P_load": 600.0, "price": 0.10, "T_out": 5.0, "G_Wm2": 0.0}


def test_electricity_balance_is_exact():
    """Derived P_grid must close the electricity balance exactly (slack bus)."""
    x, u, p = initial_state(), _sample_controls(), _sample_params()
    _, m = hub_dynamics(x, u, p)
    expected = (p["P_load"] + u["P_bat_ch"] + u["P_elz"] + u["P_hp"] + u["P_eboiler"]
                - p["P_pv"] - u["P_bat_dis"] - u["P_fc"])
    assert m["elec_residual_kW"] == pytest.approx(0.0, abs=1e-9)
    assert m["P_grid_kW"] == pytest.approx(expected, abs=1e-9)


def test_h2_mass_conservation():
    """SOC_h2 change equals electrolyser production minus fuel-cell consumption."""
    x = initial_state()
    u = {**_sample_controls(), "P_elz": 100.0, "P_fc": 40.0}
    p = _sample_params()
    x_next, _ = hub_dynamics(x, u, p)
    prod = ETA_ELZ * u["P_elz"] / E_H2_LHV_KWH_KG
    cons = (u["P_fc"] / ETA_FC_E) / E_H2_LHV_KWH_KG
    assert x_next["SOC_h2"] - x["SOC_h2"] == pytest.approx(prod - cons, rel=1e-9)


def test_tes_cannot_charge_from_nothing_flag():
    """tes_charge_excess flags charging beyond generated heat (model invariant)."""
    x = initial_state()
    u = {**_sample_controls(), "P_hp": 0.0, "P_eboiler": 0.0, "P_fc": 0.0,
         "Q_tes_ch": 100.0}                       # charging with zero generated heat
    _, m = hub_dynamics(x, u, _sample_params())
    assert m["tes_charge_excess_kW"] > 0.0
    # With enough generation the flag clears
    u2 = {**u, "P_eboiler": 200.0}                # 200 * 0.99 ~ 198 kW heat > 100
    _, m2 = hub_dynamics(x, u2, _sample_params())
    assert m2["tes_charge_excess_kW"] == pytest.approx(0.0, abs=1e-9)


def test_temperature_update_is_bounded_and_warms_with_heat():
    """Implicit-Euler thermal node is stable and responds correctly to heating."""
    x = {**initial_state(), "T_in": 18.0}
    p = {"P_pv": 0.0, "P_load": 0.0, "price": 0.0, "T_out": 0.0, "G_Wm2": 0.0}
    cold = {**_sample_controls(), "P_hp": 0.0, "P_eboiler": 0.0,
            "Q_tes_ch": 0.0, "Q_tes_dis": 0.0}
    warm = {**cold, "P_hp": 175.0}                # full heat pump
    t_cold = hub_dynamics(x, cold, p)[0]["T_in"]
    t_warm = hub_dynamics(x, warm, p)[0]["T_in"]
    assert t_warm > t_cold                        # heating raises temperature
    assert T_HARD_MIN_C < t_cold < T_HARD_MAX_C   # remains finite/bounded


# ---------------------------------------------------------------------------
# 2. Short live MPC roll-out
# ---------------------------------------------------------------------------
def test_mpc_builds_steps_and_respects_balance():
    """A few closed-loop MPC steps solve and yield balance-feasible controls."""
    from control.mpc_controller import build_mpc
    n = 30
    rng = np.random.default_rng(0)
    price = 0.05 + 0.05 * np.sin(np.linspace(0, 6, n)) + 0.01 * rng.standard_normal(n)
    pv = np.zeros(n)
    load = np.full(n, 400.0)
    tout = np.full(n, 5.0)
    irr = np.zeros(n)

    mpc, _ = build_mpc(price, pv, load, tout, irr)
    x = initial_state()
    x0 = np.array([x["SOC_bat"], x["SOC_h2"], x["SOC_tes"], x["T_in"]])
    mpc.x0 = x0
    mpc.set_initial_guess()

    sb, ib = state_bounds(), input_bounds()
    for k in range(5):
        x0 = np.array([[x["SOC_bat"]], [x["SOC_h2"]], [x["SOC_tes"]], [x["T_in"]]])
        mpc.x0 = x0
        mpc.make_step(x0)
        u = {name: float(np.squeeze(mpc.u0[name])) for name in ib}
        for name, (lo, hi) in ib.items():
            # 1e-4 tolerance absorbs IPOPT's constraint-satisfaction noise
            assert lo - 1e-4 <= u[name] <= hi + 1e-4, f"{name} out of bounds: {u[name]}"
        p = {"P_pv": pv[k], "P_load": load[k], "price": price[k],
             "T_out": tout[k], "G_Wm2": irr[k]}
        x, m = hub_dynamics(x, u, p)
        assert m["elec_residual_kW"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2b. Fast deterministic end-to-end integration (does NOT rely on committed CSVs)
# ---------------------------------------------------------------------------
def test_integration_short_run_mpc_not_worse_than_baseline():
    """Run a short real-data window for both controllers and check the headline
    claim (MPC <= baseline cost) plus physical invariants, regenerated from code."""
    from control.rolling_horizon import load_data, run_simulation
    df = load_data(start_month=1, n_days=2)          # deterministic 2-day winter window
    base = run_simulation(df, mode="baseline")
    mpc = run_simulation(df, mode="mpc")
    for d in (base, mpc):
        assert d["elec_residual_kW"].abs().max() < 1e-6
        assert ((d.u_P_bat_ch > 1) & (d.u_P_bat_dis > 1)).sum() == 0
        assert ((d.u_Q_tes_ch > 1) & (d.u_Q_tes_dis > 1)).sum() == 0
    assert mpc["grid_cost_EUR"].sum() <= base["grid_cost_EUR"].sum() + 1e-6


# ---------------------------------------------------------------------------
# 3. Invariants on saved simulation results
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results():
    b, m = RESULTS / "baseline_results.csv", RESULTS / "mpc_results.csv"
    if not (b.exists() and m.exists()):
        pytest.skip("Run `python control/rolling_horizon.py` first to generate results.")
    return pd.read_csv(b), pd.read_csv(m)


def test_results_balance_residual_zero(results):
    for d in results:
        assert d["elec_residual_kW"].abs().max() < 1e-6


def test_results_no_simultaneous_charge_discharge(results):
    for d in results:
        assert ((d.u_P_bat_ch > 1.0) & (d.u_P_bat_dis > 1.0)).sum() == 0
        assert ((d.u_Q_tes_ch > 1.0) & (d.u_Q_tes_dis > 1.0)).sum() == 0
        assert ((d.u_P_elz > 1.0) & (d.u_P_fc > 1.0)).sum() == 0


def test_results_soc_within_physical_bounds(results):
    """Both controllers must keep storage within physical limits [0, capacity]."""
    caps = {"SOC_bat_kWh": BAT_CAPACITY_KWH, "SOC_h2_kg": H2_CAPACITY_KG,
            "SOC_tes_kWh": TES_CAPACITY_KWH}
    for d in results:
        for col, cap in caps.items():
            assert d[col].min() >= -1.0
            assert d[col].max() <= cap + 1.0


def test_mpc_respects_operational_soc_bounds(results):
    """The MPC enforces operational SOC reserves (10-90%) as hard constraints.

    (The naive baseline does not — e.g. its TES slowly drains via standing
    losses below the operational floor, which is physically valid.)
    """
    _, m = results
    sb = state_bounds()
    tol = 1.0
    assert m.SOC_bat_kWh.min() >= sb["SOC_bat"][0] - tol
    assert m.SOC_bat_kWh.max() <= sb["SOC_bat"][1] + tol
    assert m.SOC_h2_kg.min() >= sb["SOC_h2"][0] - tol
    assert m.SOC_h2_kg.max() <= sb["SOC_h2"][1] + tol
    assert m.SOC_tes_kWh.min() >= sb["SOC_tes"][0] - tol
    assert m.SOC_tes_kWh.max() <= sb["SOC_tes"][1] + tol


def test_results_temperature_within_hard_bounds(results):
    for d in results:
        assert d.T_in_C.min() >= T_HARD_MIN_C - 1e-6
        assert d.T_in_C.max() <= T_HARD_MAX_C + 1e-6


def test_mpc_beats_or_matches_baseline(results):
    b, m = results
    assert m["grid_cost_EUR"].sum() <= b["grid_cost_EUR"].sum()
