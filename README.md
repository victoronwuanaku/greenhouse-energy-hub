# Greenhouse Energy Hub MPC

Rolling-horizon economic **Model Predictive Control (MPC)** for a multi-carrier greenhouse energy hub (electricity · heat · hydrogen). The controller coordinates solar PV, a battery, a hydrogen electrolyser/fuel-cell buffer, a heat pump, an electric boiler and thermal storage against real Dutch day-ahead electricity prices, minimising operating cost while keeping the crop inside its temperature comfort band.

**System modelled:** a representative 1 ha (10,000 m²) high-tech, lit Dutch (Westland) tomato greenhouse.

---

## Motivation

Dutch greenhouses have historically provided grid flexibility through **Combined Heat and Power (CHP)** units — gas engines producing electricity (sold to the grid) and heat (for the crop). As the Netherlands phases out fossil CHP, that dispatchable flexibility disappears. Replacing it requires an MPC-orchestrated multi-carrier hub combining:

- **Battery** — short-term price arbitrage and peak shaving
- **Electrolyser + H₂ tank + fuel cell** — a multi-day buffer; the *green analogue of the CHP* (surplus electricity → H₂ → electricity **and** heat on demand)
- **Heat pump + electric boiler + thermal store** — power-to-heat that decouples crop heating from the real-time electricity price
- **Predictive control** — exploiting day-ahead price forecasts and the crop's thermal comfort band as flexibility

This repo implements the **economic-dispatch layer**: a rolling-horizon MPC that coordinates conversion and storage against real NL day-ahead prices to minimise operating cost while keeping the greenhouse inside its comfort band.

---

## Results

14-day rolling-horizon simulations use do-mpc / IPOPT with a 24-hour receding horizon, hourly steps, and perfect-foresight forecasts. The MPC is compared with a **limited-capability rule-based baseline**: it cannot dispatch hydrogen (`9b36719b929fffcef26084f5efa37bcb2385b9b9199c7e429dec37a86993ad97`), charge the thermal store (`787abda250b5b2f47005452322aa8d65ee5c7bcb8848246797e4e98ea8e1b8de`), or charge the battery from the grid; it may discharge the battery only when the price is at least €0.12/kWh. It reactively holds the 16.5 °C lower comfort bound and has no look-ahead (`df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac`). Imports pay wholesale plus a transport/levy surcharge; exports earn wholesale.

<!-- BEGIN GENERATED RESULTS: DO NOT EDIT -->
| Window | Baseline Inventory-Adjusted Cost | MPC Inventory-Adjusted Cost | Saving | Comfort Violation (baseline / MPC) |
|--------|----------------------------------:|-----------------------------:|:------:|:-----------------------------------:|
| **Winter** | €45,389 | €42,560 | +6.2 % | 0.0 / 0.0 °C·h |
| **Summer** | €3,571 | €1,872 | +47.6 % | 29.6 / 28.6 °C·h |

Pinned Run Bundle IDs:
- Winter baseline: `509c81dd1bea108aac1be2073b2b16646771bf65d5cc7f257081bcf81784713a`
- Winter MPC: `1e9ee84ca78896c082b6d3216bab0d841f027b65f901aeec76bc78f253a90253`
- Summer baseline: `8d9f6657c04b552f8d4f83f484ba004bcc46a2e7fa9c95cdc67936da9d8998e2`
- Summer MPC: `5975dafc40c7bb296fa8861eb1e78b2afc21ff4eeff14ed3e2575b77768cecb1`

| Winter ablation | Inventory-Adjusted Cost | Comfort Violation | Cost change vs full |
|-----------------|--------------------------:|------------------:|:-------------------:|
| MPC (full) | €42,560 | 0.0 °C·h | +0.0 % |
| no H₂ | €43,057 | 0.0 °C·h | -1.2 % |
| no thermal store | €43,303 | 0.0 °C·h | -1.7 % |
| one-step horizon | €41,082 | 2,826.4 °C·h | +3.5 % |

Removing hydrogen raises the winter inventory-adjusted cost relative to the full controller (`9b36719b929fffcef26084f5efa37bcb2385b9b9199c7e429dec37a86993ad97` versus `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`).
Removing the thermal store raises the winter inventory-adjusted cost relative to the full controller (`787abda250b5b2f47005452322aa8d65ee5c7bcb8848246797e4e98ea8e1b8de` versus `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`).
A one-step horizon has substantial comfort violation in this winter run (`df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac` versus `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`).
<!-- END GENERATED RESULTS -->

Published tables and figures are regenerated only from the pinned, verified Run Bundles under `results/runs/`. The table reports **Inventory-Adjusted Cost** and **Comfort Violation** separately; it does not use a comfort-priced composite metric. These are simulation-prototype results, not decision-grade evidence for a specific site.

### Verified figures

![Cumulative cost](results/figures/fig1_cumulative_cost.png)

![Grid exchange vs price](results/figures/fig2_grid_vs_price.png)
![Temperature vs comfort band](results/figures/fig4_temperature.png)
![Storage trajectories](results/figures/fig3_soc_trajectories.png)
![Power-to-heat load-shifting](results/figures/fig5_heat_shifting.png)

### Verified ablation evidence

![Ablation](results/figures/fig6_ablation.png)

---

## System architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     GREENHOUSE ENERGY HUB (1 ha)                 │
│                                                                  │
│  ☀ PV (500 kWp) ─────────────┐                                  │
│  🔋 Battery (1000 kWh/500 kW) ┤                                  │
│  ⚡ Electrolyser (250 kW) ────┤── Electricity bus ⇄ Grid (±2 MW) │
│  🔌 Fuel cell (200 kWe) ──────┤                                  │
│  🌡 Heat pump (175 kWe→612 kWth, COP 3.5)                        │
│  🔥 Electric boiler (600 kWe→594 kWth)                           │
│       │  🫙 H₂ tank (200 kg)                                     │
│       └──┴──▶ Heat bus ──▶ 🪣 Thermal store (4000 kWh) ──▶ Crop │
│  ☀ Solar gain (passive)   🍃 Ventilation (control)              │
│                                                                  │
│  🌡 Indoor temperature  T_in ∈ [16, 24] °C  (soft band)          │
└────────────────────────────────────────────────────────────────┘
            ▲ MPC — 24-hour receding horizon, hourly steps
            │ minimise: Σ price·P_grid·Δt + wear/cycling + band slack
            │ subject to: storage dynamics, electricity balance,
            │             power-to-heat balance, comfort band
```

The electrolyser + tank + fuel cell together replace the dispatchable CHP: cheap/surplus electricity is stored as hydrogen and later reconverted to **both** electricity and (recovered) heat.

---

## MPC formulation

At each timestep *k* the controller solves a finite-horizon optimal control problem:

$$\min_{u_k,\ldots,u_{k+N-1}} \;\; \sum_{i=0}^{N-1}\Big[\; \underbrace{\lambda(k{+}i)\,P_{\text{grid}}\,\Delta t}_{\text{grid cost}} \;+\; \underbrace{c_{\text{wear}}(u)}_{\text{degradation}} \;+\; \underbrace{c_{\text{cmpl}}(u)}_{\text{anti-cycling}} \;+\; \underbrace{\rho\,s_T}_{\text{comfort slack}}\;\Big] \;-\; \underbrace{\beta\,V_{\text{stored}}(x_{k+N})}_{\text{terminal value}}$$

subject to

- **Storage dynamics** — battery SOC, H₂ inventory, thermal store, greenhouse temperature (implicit-Euler thermal node, unconditionally stable).
- **Electricity balance** — $P_{\text{grid}} + P_{\text{pv}} + P_{\text{bat,dis}} + P_{\text{fc}} = P_{\text{load}} + P_{\text{bat,ch}} + P_{\text{elz}} + P_{\text{hp}} + P_{\text{eb}}$. The grid is the **slack bus** (a derived expression), so the balance holds *exactly by construction* — avoiding the degenerate squared-equality constraint of the original prototype.
- **Asymmetric grid tariff** — imported energy pays wholesale price + a transport/levy surcharge; exports earn wholesale only. This breaks symmetric buy=sell arbitrage and tempers negative-price gaming. The import volume uses a smooth $\max(0, P_{\text{grid}})$ so the objective stays differentiable for IPOPT.
- **Power-to-heat feasibility** — the thermal store can only charge from generated heat.
- **Comfort band** — $T_{\text{in}}\in[16,24]\,°\mathrm{C}$ as a *soft* constraint (slack-penalised), so the problem stays feasible when summer solar gain physically exceeds ventilation capacity.
- **Anti-cycling** — per-kWh throughput costs plus a complementarity penalty eliminate simultaneous charge/discharge.
- **Terminal value** — end-of-horizon storage is valued at a price-based cost-to-go proxy (battery & H₂ at the horizon-average price; heat at the average *heating-hour* price ÷ COP, which is ≈ 0 in summer and so prevents pointless heat hoarding).

**Horizon** N = 24 h · **Step** Δt = 1 h · **Solver** IPOPT (via CasADi / do-mpc).

---

## Repository layout

```
greenhouse-energy-hub-mpc/
├── data/
│   ├── fetch_pvgis.py        # PVGIS API — Westland NL solar + weather
│   ├── fetch_prices.py       # energy-charts.info — NL day-ahead prices
│   ├── generate_demand.py    # synthetic greenhouse ELECTRICITY load (WUR params)
│   └── *.csv                 # generated inputs (hourly UTC)
├── models/
│   └── hub_model.py          # assets, bounds, plant dynamics (Geidl–Andersson hub)
├── control/
│   ├── mpc_controller.py     # do-mpc symbolic MPC (CasADi / IPOPT)
│   └── rolling_horizon.py    # data alignment, simulation loop, limited-capability baseline, CLI
├── accounting.py             # shared cost/saving calc (CLI, notebook, tests agree)
├── experiments/
│   └── ablations.py          # winter ablation study (no-H2 / no-TES / myopic)
├── tests/
│   └── test_hub.py           # physical + control invariants & integration (pytest)
├── notebooks/
│   └── results_analysis.ipynb
├── pyproject.toml            # metadata, Python ≥3.11, pytest config
└── results/
    ├── runs/                 # content-addressed verified Run Bundles
    ├── publication_manifest.json  # pinned publication recipe
    └── figures/              # regenerated fig1–fig6 used by the README
```

---

## Installation and usage

```bash
git clone https://github.com/victoronwuanaku/greenhouse-energy-hub-mpc
cd greenhouse-energy-hub-mpc
pip install -r requirements.txt      # or: pip install -e ".[dev]"

# Generate inputs (PV/weather + NL prices, then the electrical load)
python3 data/fetch_pvgis.py
python3 data/fetch_prices.py
python3 data/generate_demand.py

# Generate verified candidate Run Bundles, then publish the pinned recipe
python3 experiments/run_scenario.py --name winter-2023-14d --start 2023-01-01T00:00:00+01:00 --days 14 --controller baseline --candidate-key winter-baseline
python3 experiments/publish_results.py --candidates results/diagnostics/publication-candidates.json --manifest results/publication_manifest.json

# Tests and analysis
python3 -m pytest tests/ -q
jupyter nbconvert --to notebook --execute --inplace notebooks/results_analysis.ipynb
```

---

## Data sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| Solar PV + weather | PVGIS-SARAH2 (EU JRC), Westland 52.0 °N 4.25 °E | 2020, hourly |
| NL day-ahead prices | energy-charts.info (Fraunhofer ISE / ENTSO-E) | 2023, hourly |
| Greenhouse electrical load | Synthetic, WUR-parameterised (Warmenhoven et al. 2023) | hourly |

PV/weather (2020) and prices (2023) come from different years; `load_data` aligns them **explicitly by (month, day, hour)** after flooring the PVGIS `:11` solar-time stamps to the hour (Feb 29 has no non-leap-year price counterpart and is dropped by the key join). This is a deliberate simplification for a synthetic study — switch the PVGIS database to ERA5/2023 for a fully single-year dataset. Heat demand is **not** prescribed: it is implicit in the greenhouse temperature ODE.

---

## Tools and methods

| Tool | Role | Basis |
|------|------|-------|
| **do-mpc v5** | MPC formulation, rolling-horizon solver | Fiedler et al. (2023), *Control Eng. Practice* |
| **CasADi v3.7** | Symbolic differentiation, NLP generation | Andersson et al. (2019), *Math. Prog. Comp.* |
| **IPOPT** | Interior-point NLP solver | Wächter & Biegler (2006) |
| **PVGIS** | Solar irradiance + temperature data | EC Joint Research Centre |
| **Energy-hub framework** | Multi-carrier coupling | Geidl & Andersson (2007) |

---

## Honest limitations

- **Perfect foresight.** Forecasts are the realised data — an upper bound on achievable savings. A natural next step is forecast error / robust or stochastic MPC.
- **Naive limited-capability baseline.** The baseline is a frugal reactive thermostat, not a tuned commercial greenhouse EMS; the comparison is not against the state of the art.
- **Simplified market.** A flat import surcharge over the wholesale price — no capacity charges, time-of-use network tariffs, imbalance settlement, or explicit export limits beyond the grid power cap. Arbitrage value is therefore still somewhat optimistic. The MPC objective uses a smooth `max(0, P_grid)` (ε = 1 kW) for the import fee; the reported cost uses the exact `max`, so published savings are unaffected (bounded in `tests/`).
- **Toy thermal model.** A single-zone lumped-capacitance ODE; humidity, CO₂ and crop growth are out of scope.
- **Summer overheating.** On hot, high-irradiance hours the solar gain physically exceeds ventilation capacity, so the comfort band is violated by both controllers (a real greenhouse would add active cooling). Reported honestly rather than hidden.
- **Indicative asset sizing.** Capacities are reasonable for a high-tech 1 ha greenhouse but are provisional, not calibrated to a specific site.

---

## Key references

1. **McAllister, R.D. et al. (2025).** RL-Guided MPC for Autonomous Greenhouse Control. *arXiv:2506.13278* — recent work combining reinforcement learning with predictive control for greenhouse climate management.
2. **Fiedler, F. et al. (2023).** do-mpc: Towards FAIR nonlinear and robust MPC. *Control Engineering Practice, 140*, 105676.
3. **Geidl, M. & Andersson, G. (2007).** Optimal power flow of multiple energy carriers. *IEEE Trans. Power Syst. 22*(1), 145–155 — the energy-hub framework used in `hub_model.py`.
4. **Coordinated distributed MPC for multi-energy carrier systems** (2024). *Scientific Reports.*
5. **Andersson, J.A.E. et al. (2019).** CasADi: a software framework for nonlinear optimization and optimal control. *Math. Prog. Computation, 11*(1), 1–36.

---

## License

MIT — see `pyproject.toml`.
