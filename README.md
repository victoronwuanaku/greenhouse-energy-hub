# Greenhouse Energy Hub MPC

Rolling-horizon economic **Model Predictive Control (MPC)** for a multi-carrier greenhouse energy hub (electricity · heat · hydrogen). A portfolio project demonstrating engagement with predictive control and energy-systems optimisation, motivated by the [SPROUT project](https://www.tudelft.nl/) — a €5.6M RVO-funded consortium replacing the grid flexibility of Dutch greenhouse CHP units with MPC-controlled hybrid storage hubs.

**Site modelled:** Division Q / Westland greenhouse cluster, Monster, Netherlands · **Scale:** 1 ha (10,000 m²) high-tech lit tomato greenhouse.

---

## Motivation

Dutch greenhouses have historically provided grid flexibility through **Combined Heat and Power (CHP)** units — gas engines producing electricity (sold to the grid) and heat (for the crop). As the Netherlands phases out fossil CHP, that dispatchable flexibility disappears. SPROUT replaces it with an MPC-orchestrated multi-carrier hub combining:

- **Battery** — short-term price arbitrage and peak shaving
- **Electrolyser + H₂ tank + fuel cell** — a multi-day buffer; the *green analogue of the CHP* (surplus electricity → H₂ → electricity **and** heat on demand)
- **Heat pump + electric boiler + thermal store** — power-to-heat that decouples crop heating from the real-time electricity price
- **Predictive control** — exploiting day-ahead price forecasts and the crop's thermal comfort band as flexibility

This repo implements the **economic-dispatch layer**: a rolling-horizon MPC that coordinates conversion and storage against real NL day-ahead prices to minimise operating cost while keeping the greenhouse inside its comfort band.

---

## Results

14-day rolling-horizon simulation, do-mpc / IPOPT, 24-hour receding horizon, hourly steps, perfect-foresight forecasts. The MPC is compared against a **fair rule-based baseline** — a frugal thermostat that holds the *lower* comfort bound (16.5 °C) reactively, with a simple battery rule and no look-ahead. Holding the lower bound (rather than a 19 °C setpoint) means the MPC's saving reflects dispatch skill, not merely running the greenhouse colder. Imports pay wholesale + a transport/levy surcharge; exports earn wholesale. Cost is reported as raw grid cost and **inventory-adjusted** (net stored-energy change marked to market at the mean price, so neither controller is rewarded for merely ending with more/less storage).

| Window | Baseline cost | MPC cost | Saving (raw) | Saving (inv.-adjusted) | Comfort-band violation |
|--------|--------------:|---------:|:------------:|:----------------------:|:----------------------:|
| **Winter** (Jan 1–14) | €44,308 | €41,154 | **+7.1 %** | **+6.9 %** (€3,073) | 0 °C·h (both) |
| **Summer** (Jun 1–14) | €3,461 | €1,574 | +54.5 % | +53.1 % (€1,869) | ~29 °C·h (both) |

**Winter is the headline result** and the one that matters for SPROUT: a credible **~7 % operating-cost reduction** on large absolute winter costs, achieved purely by smarter dispatch against a fair baseline. The summer percentage is large because absolute costs are small (high PV makes the hub near net-zero) and partly reflects paid consumption during negative-price hours — a real but flattering demand-response effect; the absolute summer saving is modest.

Both scenarios are reproducible from committed scenario files (`results/scenarios/`) and summarised in `results/scenarios/summary.csv`. Every reported run satisfies the invariants checked in `tests/`: the electricity balance closes exactly (residual < 10⁻⁶ kW), no store ever charges and discharges in the same hour, and all states stay within bounds.

### How the MPC wins

![Cumulative cost](results/figures/fig1_cumulative_cost.png)

- **Storage arbitrage** — the battery cycles daily (charge cheap, discharge expensive); the H₂ tank buffers across multiple days via electrolyser/fuel-cell round-trips.
- **Power-to-heat load-shifting** — the MPC runs the electric boiler in cheap hours and banks heat in the thermal store, displacing expensive-hour heating.
- **Thermal-mass pre-heating** — both controllers sit near the 16 °C floor, but the MPC additionally pre-heats the greenhouse's thermal mass when power is cheap and coasts when it is expensive — demand-side flexibility within the crop comfort band.

![Grid exchange vs price](results/figures/fig2_grid_vs_price.png)
![Temperature vs comfort band](results/figures/fig4_temperature.png)
![Storage trajectories](results/figures/fig3_soc_trajectories.png)
![Power-to-heat load-shifting](results/figures/fig5_heat_shifting.png)

### Ablation — where the value comes from

Re-running the winter window with capabilities disabled isolates each contribution. *Effective* cost charges comfort-band violations at the MPC's internal rate (10 €/°C·h), so a controller cannot look cheap by letting the greenhouse drift out of band.

| Variant | Grid cost | Comfort violation | Effective saving vs baseline |
|---------|----------:|------------------:|:----------------------------:|
| **MPC (full)** | €41,154 | 0 °C·h | **+7.1 %** |
| no H₂ | €41,908 | 0 °C·h | +5.4 % |
| no thermal store | €46,037 | 0 °C·h | **−3.9 %** |
| myopic (1 h horizon) | €23,094 | 2,992 °C·h | **−19.7 %** |

![Ablation](results/figures/fig6_ablation.png)

The **thermal store is the decisive driver** — remove it and the MPC *loses* to the baseline (−3.9 %), underscoring power-to-heat as the CHP-flexibility replacement. The **H₂ buffer adds ~1.7 points**. The **myopic controller** looks 48 % cheaper on grid cost alone, but only by under-heating and racking up 2,992 °C·h of crop-comfort violation; once that is valued, it is far *worse* (−19.7 %) — i.e. **price foresight is essential to hold the comfort band**, which is the core justification for MPC here.

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
│   └── rolling_horizon.py    # data alignment, simulation loop, fair baseline, CLI
├── accounting.py             # shared cost/saving calc (CLI, notebook, tests agree)
├── experiments/
│   └── ablations.py          # winter ablation study (no-H2 / no-TES / myopic)
├── tests/
│   └── test_hub.py           # physical + control invariants & integration (pytest)
├── notebooks/
│   └── results_analysis.ipynb
├── pyproject.toml            # metadata, Python ≥3.11, pytest config
└── results/
    ├── *.csv                 # canonical (latest) run
    ├── scenarios/            # per-scenario archives + summary.csv + ablations.csv
    └── figures/              # fig1–fig6 (committed, used by the README)
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

# Run both controllers (scenario-tagged outputs land in results/scenarios/)
python3 control/rolling_horizon.py --days 14 --start-month 1   # winter (headline)
python3 control/rolling_horizon.py --days 14 --start-month 6   # summer

# Ablation study, tests, and analysis
python3 experiments/ablations.py --days 14 --start-month 1
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
- **Naive (if fair) baseline.** The baseline is a frugal reactive thermostat, not a tuned commercial greenhouse EMS; the saving is "MPC vs. a sensible naive rule," not vs. the state of the art.
- **Simplified market.** A flat import surcharge over the wholesale price — no capacity charges, time-of-use network tariffs, imbalance settlement, or explicit export limits beyond the grid power cap. Arbitrage value is therefore still somewhat optimistic.
- **Toy thermal model.** A single-zone lumped-capacitance ODE; humidity, CO₂ and crop growth are out of scope.
- **Summer overheating.** On hot, high-irradiance hours the solar gain physically exceeds ventilation capacity, so the comfort band is violated by both controllers (a real greenhouse would add active cooling). Reported honestly rather than hidden.
- **Indicative asset sizing.** Capacities are reasonable for a high-tech 1 ha greenhouse but are provisional, not calibrated to a specific site.

---

## Key references

1. **McAllister, R.D. et al. (2025).** RL-Guided MPC for Autonomous Greenhouse Control. *arXiv:2506.13278* — foundational paper from the SPROUT supervisor; this project implements the economic-MPC layer that RL guidance targets.
2. **Fiedler, F. et al. (2023).** do-mpc: Towards FAIR nonlinear and robust MPC. *Control Engineering Practice, 140*, 105676.
3. **Geidl, M. & Andersson, G. (2007).** Optimal power flow of multiple energy carriers. *IEEE Trans. Power Syst. 22*(1), 145–155 — the energy-hub framework used in `hub_model.py`.
4. **Coordinated distributed MPC for multi-energy carrier systems** (2024). *Scientific Reports.*
5. **Andersson, J.A.E. et al. (2019).** CasADi: a software framework for nonlinear optimization and optimal control. *Math. Prog. Computation, 11*(1), 1–36.

---

## SPROUT context

This project addresses SPROUT's core control challenge: replacing the dispatchable flexibility of phased-out CHP units with an MPC-orchestrated multi-carrier storage hub. The economic-dispatch layer demonstrated here — coordinating battery, hydrogen and thermal storage and power-to-heat against real NL price signals — is the foundation on which the hierarchical RL-guided MPC of McAllister et al. (2025) operates.

Viktor Onwuanaku holds an MSc from Wageningen University (2025), a SPROUT consortium partner, and brings energy-systems modelling experience (PyPSA, HOMER Pro, DIgSILENT PowerFactory) from a techno-economic HRES optimisation thesis — a direct bridge to the SPROUT team's modelling and experimental infrastructure.
