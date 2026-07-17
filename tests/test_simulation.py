from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


INPUT_NAMES = (
    "P_bat_ch",
    "P_bat_dis",
    "P_elz",
    "P_fc",
    "P_hp",
    "P_eboiler",
    "Q_tes_ch",
    "Q_tes_dis",
    "vent",
)


@pytest.mark.xfail(strict=True, reason="PF-01: one-step terminal state is unconstrained")
def test_one_step_mpc_keeps_every_reached_state_valid():
    from control.rolling_horizon import load_data, run_simulation
    from models.hub_model import state_bounds

    outcome = run_simulation(
        load_data(start_month=1, n_days=2),
        mode="mpc",
        n_horizon=1,
    )
    bounds = state_bounds()
    frame = outcome if isinstance(outcome, pd.DataFrame) else outcome.to_frame()
    for state, column in {
        "SOC_bat": "SOC_bat_kWh",
        "SOC_h2": "SOC_h2_kg",
        "SOC_tes": "SOC_tes_kWh",
        "T_in": "T_in_C",
    }.items():
        lower, upper = bounds[state]
        assert frame[column].between(lower - 1e-6, upper + 1e-6).all()


class _FailedMpc:
    """Complete double for the do-mpc surface consumed by the legacy loop."""

    def __init__(self) -> None:
        self.x0 = None
        self.solver_stats: dict[str, object] = {}
        self.u0 = {name: np.array([[1.0]]) for name in INPUT_NAMES}
        self.make_step_calls: list[np.ndarray] = []

    def set_initial_guess(self) -> None:
        return None

    def make_step(self, x0: np.ndarray) -> np.ndarray:
        self.make_step_calls.append(np.asarray(x0).copy())
        self.solver_stats = {
            "success": False,
            "return_status": "Infeasible_Problem_Detected",
            "iter_count": 17,
            "t_wall_total": 0.01,
        }
        return np.ones((len(INPUT_NAMES), 1))


@pytest.mark.xfail(strict=True, reason="PF-02: failed solver output is applied to the plant")
def test_solver_failure_returns_invalid_run_without_advancing_plant(monkeypatch, hourly_frame):
    import control.mpc_controller as mpc_controller
    import control.rolling_horizon as rolling_horizon
    import models.hub_model as hub_model

    failed_mpc = _FailedMpc()
    monkeypatch.setattr(mpc_controller, "build_mpc", lambda **_kwargs: (failed_mpc, object()))

    real_hub_dynamics = hub_model.hub_dynamics
    plant_calls = 0

    def counted_hub_dynamics(*args, **kwargs):
        nonlocal plant_calls
        plant_calls += 1
        return real_hub_dynamics(*args, **kwargs)

    # The legacy loop holds a direct import, so instrument both definition and consumer.
    monkeypatch.setattr(hub_model, "hub_dynamics", counted_hub_dynamics)
    monkeypatch.setattr(rolling_horizon, "hub_dynamics", counted_hub_dynamics)

    # Forty-nine points give the default 24-step controller ample initial coverage,
    # so only the injected solver failure may classify this Run as invalid.
    outcome = rolling_horizon.run_simulation(hourly_frame, mode="mpc")

    observed = (
        len(failed_mpc.make_step_calls),
        plant_calls,
        getattr(outcome, "failure_code", None),
    )
    assert observed == (1, 0, "solver_failure")

    # Keep the not-yet-existing outcome type inside the regression body so legacy
    # collection remains possible until the fail-closed simulation interface lands.
    from control.rolling_horizon import InvalidRun

    assert isinstance(outcome, InvalidRun)


@pytest.mark.xfail(strict=True, reason="PF-02: disabled assets retain non-zero state and capacity")
@pytest.mark.parametrize(
    ("capability", "state_field", "control_fields"),
    [
        ("hydrogen", "soc_hydrogen_kg", ("electrolyser_kw", "fuel_cell_kw")),
        ("thermal_store", "soc_thermal_kwh", ("thermal_charge_kw", "thermal_discharge_kw")),
    ],
)
def test_disabled_asset_is_inert_zero_capacity(capability, state_field, control_fields):
    from models.hub_model import (
        AssetCapabilities,
        HubConfiguration,
        control_bounds,
        initial_state,
        physical_state_bounds,
    )

    capabilities = AssetCapabilities(**{capability: False})
    config = HubConfiguration(capabilities=capabilities)
    state = initial_state(config)

    assert getattr(state, state_field) == 0.0
    assert physical_state_bounds(config)[state_field] == (0.0, 0.0)
    for field in control_fields:
        assert control_bounds(config)[field] == (0.0, 0.0)


def _first_mpc_control(prices: np.ndarray) -> np.ndarray:
    from control.mpc_controller import build_mpc
    from models.hub_model import initial_state

    n = len(prices)
    controller, _ = build_mpc(
        price_forecast=prices,
        pv_forecast=np.zeros(n),
        load_elec_forecast=np.full(n, 400.0),
        temp_out_forecast=np.full(n, 5.0),
        irr_forecast=np.zeros(n),
        n_horizon=24,
    )
    initial = initial_state()
    x0 = np.array(
        [[initial["SOC_bat"]], [initial["SOC_h2"]], [initial["SOC_tes"]], [initial["T_in"]]]
    )
    controller.x0 = x0
    controller.set_initial_guess()
    controller.make_step(x0)
    assert controller.solver_stats["success"] is True
    return np.array([float(np.squeeze(controller.u0[name])) for name in INPUT_NAMES])


@pytest.mark.xfail(strict=True, reason="PF-03: terminal value reads beyond the forecast horizon")
def test_first_control_is_independent_of_out_of_horizon_prices():
    shared_horizon = np.full(25, 0.10)
    low_tail = np.concatenate([shared_horizon, np.full(24, -1.0)])
    high_tail = np.concatenate([shared_horizon, np.full(24, 1.0)])

    np.testing.assert_allclose(
        _first_mpc_control(low_tail),
        _first_mpc_control(high_tail),
        atol=1e-4,
        rtol=0.0,
    )
