# Greenhouse Energy Hub Architecture Migration Implementation Plan

> This implementation checklist uses checkbox (`- [ ]`) syntax for task tracking.

**Goal:** Refactor the repository into four deep, installable Modules while making every controller Run fail closed, causal, physically valid, evaluated under one named policy, and publishable only as a verified provenance-backed Run Bundle.

**Architecture:** Stabilize behavior behind the existing files first, then move those stabilized responsibilities into greenhouse-hub physics, validated simulation, Scenario data, and evaluation/artifact Modules under src/greenhouse_energy_hub. Numerical simulation and CasADi optimization share one expression layer; Controller Adapters receive immutable horizon-limited Scenario views; only ValidRun values can cross into evaluation and atomic Run Bundle persistence.

**Tech Stack:** Python 3.11+, dataclasses and typing.Protocol, NumPy, pandas, CasADi 3.7, do-mpc 5.1, IPOPT, pytest, hashlib/json/csv/pathlib/subprocess from the standard library, Matplotlib/Jupyter for publication consumers.

## Global Constraints

- Work from the repository root.
- Preserve unrelated user-owned local files. Never edit, stage, delete, or include them in a commit.
- Keep each behavioral correction, structural move, and artifact regeneration in a separate commit. Do not combine phases to reduce commit count.
- Keep the existing Baseline Controller policy unchanged during this migration: lower-band thermostat, PV-surplus battery charging, fixed high-price battery discharge, no hydrogen, no TES charging, and no grid battery charging.
- Do not add a new optimizer, new greenhouse physics, a capability-matched baseline, a generic backend framework, a database, or a remote artifact registry.
- Do not overwrite or treat results/baseline_results.csv, results/mpc_results.csv, or results/scenarios/*.csv as authoritative. They remain legacy evidence until the final artifact task removes them.
- Do not fetch network data during implementation. Existing input bytes are the migration inputs; acquisition scripts gain deterministic provenance output without silently replacing those bytes.
- Every known regression is first committed as a strict xfail linked to its finding identifier. Prove the defect with --runxfail, implement the fix, prove the test passes with --runxfail, then remove only that test's xfail marker.
- Every commit ends with the smallest relevant test command and a clean check of the files intended for that commit.
- Use these cache settings for commands that import do-mpc or Matplotlib:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest -q
~~~

- A new ADR-level trade-off stops execution for user review. An implementation detail already determined by ADR-0001 through ADR-0007 does not reopen design.
- No published claim is regenerated before the behavioral and structural gates pass.
- The goal is not complete until every PF-01 through PF-07 and AR-01 through AR-02 entry has full-scope evidence in docs/reviews/2026-07-16-finding-verification-matrix.md.

---

## Responsibility and File Map

| Current file | Responsibility stabilized before moving | Final file |
|---|---|---|
| models/hub_model.py | Asset capabilities, Hub State/control types, shared expressions, bounds, numerical Adapter, invariants | src/greenhouse_energy_hub/hub.py |
| control/rolling_horizon.py | Run outcome types, Controller seam, operating-step loop, fail-closed validation | src/greenhouse_energy_hub/simulation.py |
| control/mpc_controller.py | do-mpc/CasADi Adapter and solver diagnostics | src/greenhouse_energy_hub/controllers/mpc.py |
| Baseline functions inside control/rolling_horizon.py | Characterized limited-capability Controller Adapter | src/greenhouse_energy_hub/controllers/baseline.py |
| New root scenarios.py | Scenario construction, immutable horizon views, timezone rules, exact coverage, provenance | src/greenhouse_energy_hub/scenarios.py |
| accounting.py | Evaluation Policy, scorecards, identities, Run Bundle persistence and verification | src/greenhouse_energy_hub/evaluation.py |
| CLI code inside control/rolling_horizon.py | Select Scenario/Controller and call Modules only | experiments/run_scenario.py |
| experiments/ablations.py | Select validated variant specifications and compare scorecards | experiments/ablations.py |
| data/fetch_prices.py | Price acquisition Adapter | scripts/fetch_prices.py |
| data/fetch_pvgis.py | PV/weather acquisition Adapter | scripts/fetch_pvgis.py |
| data/generate_demand.py | Demand derivation Adapter calling Scenario-owned temporal rules | scripts/generate_demand.py |
| data/grid_price_signal.csv, data/pv_profile.csv | Immutable acquired input bytes | data/source/ |
| data/demand_profile.csv | Reproducible derived input | data/derived/ |
| tests/test_hub.py | Physics and numerical/CasADi parity | tests/test_hub.py |
| New regression files | Simulation, Scenario, evaluation, identity and publication contracts | tests/test_simulation.py, tests/test_scenarios.py, tests/test_evaluation.py, tests/test_published_artifacts.py |
| notebooks/results_analysis.ipynb | Explicit verified Run Bundle consumer | notebooks/results_analysis.ipynb |
| Mutable legacy result files | Removed after replacement evidence exists | results/runs/, results/diagnostics/, results/figures/, results/publication_manifest.json |

## Stable Interface Contract

The legacy files expose these names before the structural phase. The structural phase changes import paths only.

~~~python
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
~~~

### Hub Module

~~~python
@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str
    actual: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class AssetCapabilities:
    battery: bool = True
    hydrogen: bool = True
    thermal_store: bool = True


@dataclass(frozen=True)
class HubConfiguration:
    capabilities: AssetCapabilities = AssetCapabilities()


@dataclass(frozen=True)
class HubState:
    soc_battery_kwh: object
    soc_hydrogen_kg: object
    soc_thermal_kwh: object
    indoor_temperature_c: object


@dataclass(frozen=True)
class HubControl:
    battery_charge_kw: object
    battery_discharge_kw: object
    electrolyser_kw: object
    fuel_cell_kw: object
    heat_pump_kw: object
    electric_boiler_kw: object
    thermal_charge_kw: object
    thermal_discharge_kw: object
    ventilation_fraction: object


@dataclass(frozen=True)
class ExogenousInputs:
    pv_kw: object
    electric_load_kw: object
    price_eur_per_kwh: object
    outdoor_temperature_c: object
    irradiance_w_per_m2: object


@dataclass(frozen=True)
class HubFlows:
    grid_kw: object
    generated_heat_kw: object
    heat_to_air_kw: object
    thermal_charge_margin_kw: object
    hydrogen_production_kg_per_h: object
    hydrogen_consumption_kg_per_h: object


@dataclass(frozen=True)
class HubStep:
    successor: HubState
    flows: HubFlows


STATE_MODEL_NAMES = {
    "soc_battery_kwh": "SOC_bat",
    "soc_hydrogen_kg": "SOC_h2",
    "soc_thermal_kwh": "SOC_tes",
    "indoor_temperature_c": "T_in",
}

CONTROL_MODEL_NAMES = {
    "battery_charge_kw": "P_bat_ch",
    "battery_discharge_kw": "P_bat_dis",
    "electrolyser_kw": "P_elz",
    "fuel_cell_kw": "P_fc",
    "heat_pump_kw": "P_hp",
    "electric_boiler_kw": "P_eboiler",
    "thermal_charge_kw": "Q_tes_ch",
    "thermal_discharge_kw": "Q_tes_dis",
    "ventilation_fraction": "vent",
}
~~~

Public functions:

~~~python
physical_state_bounds(config: HubConfiguration) -> dict[str, tuple[float, float]]
operational_state_bounds(config: HubConfiguration) -> dict[str, tuple[float, float]]
control_bounds(config: HubConfiguration) -> dict[str, tuple[float, float]]
initial_state(config: HubConfiguration = HubConfiguration()) -> HubState
hub_state_array(state: HubState) -> np.ndarray
hub_control_from_array(values: Sequence[float]) -> HubControl
hub_step_expressions(
    state: HubState,
    control: HubControl,
    exogenous: ExogenousInputs,
    config: HubConfiguration = HubConfiguration(),
) -> HubStep
advance_hub(
    state: HubState,
    control: HubControl,
    exogenous: ExogenousInputs,
    config: HubConfiguration = HubConfiguration(),
) -> HubStep
validate_control(
    control: HubControl,
    config: HubConfiguration,
    tolerance: float = 1e-6,
) -> tuple[ValidationIssue, ...]
validate_successor(
    state: HubState,
    config: HubConfiguration,
    require_operational_storage: bool,
    tolerance: float = 1e-6,
) -> tuple[ValidationIssue, ...]
~~~

Disabled capabilities keep every field but force associated capacities, initial values, bounds, effective flows, dynamics, and inventory contributions to exactly zero. Physical storage bounds are zero-to-capacity; MPC operational reserve bounds remain the existing 10-90%, 4-96%, and 10-90% policies. All Controllers share physical validation; the MPC additionally declares operational storage bounds as required.

### Scenario Module

~~~python
@dataclass(frozen=True)
class ScenarioPoint:
    timestamp_utc: datetime
    price_eur_per_kwh: float
    pv_kw: float
    electric_load_kw: float
    outdoor_temperature_c: float
    irradiance_w_per_m2: float


@dataclass(frozen=True)
class SourceProvenance:
    source_name: str
    source_path: str
    sha256: str
    acquisition_parameters: Mapping[str, JSONValue]
    original_timezone: str
    units: Mapping[str, str]
    transformations: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    name: str
    operating_start: datetime
    operating_end: datetime
    forecast_end: datetime
    forecast_horizon_capacity_steps: int
    step_duration: timedelta
    operating_step_count: int
    points: tuple[ScenarioPoint, ...]
    provenance: tuple[SourceProvenance, ...]

    def forecast_view(
        self, operating_step: int, horizon_steps: int
    ) -> tuple[ScenarioPoint, ...]:
        view = self.points[operating_step:operating_step + horizon_steps + 1]
        if len(view) != horizon_steps + 1:
            raise ScenarioCoverageError(
                f"{self.name}: operating step {operating_step} requires "
                f"{horizon_steps + 1} points; received {len(view)}"
            )
        return view
~~~

The N+1 points are the N controlled stages plus the terminal TVP point required by do-mpc. Terminal inventory prices use only view[:N]; the terminal point never enters realized evaluation. Scenario.points includes Operating Window points followed by Forecast Coverage points. Only points[:operating_step_count] are applied or evaluated. A comparison constructs one Scenario at the maximum configured horizon (24 for the publication set), so Baseline and MPC Run Specifications share identical Scenario content even though the Baseline requests only its current point.

### Simulation Module

~~~python
@dataclass(frozen=True)
class DecisionDiagnostics:
    adapter: str
    decision_status: str
    solver_success: bool | None
    solver_return_status: str | None
    solver_iterations: int | None
    solver_wall_seconds: float | None
    forecast_start_utc: datetime | None
    forecast_end_utc: datetime | None
    terminal_electric_value_eur_per_kwh: float | None
    terminal_heat_value_eur_per_kwhth: float | None


@dataclass(frozen=True)
class ControlDecision:
    control: HubControl
    diagnostics: DecisionDiagnostics


@dataclass(frozen=True)
class ControllerFailure:
    code: str
    message: str
    diagnostics: DecisionDiagnostics


class ControllerAdapter(Protocol):
    name: str
    configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    forecast_horizon_steps: int
    requires_operational_storage_bounds: bool

    def decide(
        self,
        state: HubState,
        forecast: tuple[ScenarioPoint, ...],
    ) -> ControlDecision | ControllerFailure:
        raise NotImplementedError


@dataclass(frozen=True)
class ValidationReport:
    complete: bool
    valid: bool
    checked_operating_steps: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class OperatingRecord:
    operating_step: int
    timestamp_utc: datetime
    start_state: HubState
    control: HubControl
    exogenous: ExogenousInputs
    reached_state: HubState
    flows: HubFlows


@dataclass(frozen=True)
class ValidRun:
    scenario: Scenario
    controller_name: str
    controller_configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    hub_configuration: HubConfiguration
    initial_state: HubState
    records: tuple[OperatingRecord, ...]
    controller_diagnostics: tuple[DecisionDiagnostics, ...]
    terminal_state: HubState
    validation: ValidationReport


@dataclass(frozen=True)
class InvalidRun:
    scenario: Scenario
    controller_name: str
    controller_configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    hub_configuration: HubConfiguration
    failed_step: int
    failure_code: str
    message: str
    partial_records: tuple[OperatingRecord, ...]
    controller_diagnostics: tuple[DecisionDiagnostics, ...]


@dataclass(frozen=True)
class MpcConfiguration:
    horizon_steps: int = 24
    terminal_weight: float = 1.0
    battery_wear_eur_per_kwh: float = 0.005
    thermal_store_wear_eur_per_kwh: float = 0.0005
    electrolyser_wear_eur_per_kwh: float = 0.002
    fuel_cell_wear_eur_per_kwh: float = 0.002
    complementarity_weight: float = 0.001
    comfort_slack_weight: float = 10.0
    input_move_weight: float = 0.0001
    solver_max_iterations: int = 800
    solver_tolerance: float = 1e-6
~~~

simulate_run(scenario: Scenario, controller: ControllerAdapter, hub_config: HubConfiguration) returns ValidRun | InvalidRun and never raises away the diagnostic evidence. It returns immediately on ControllerFailure, non-finite/missing/out-of-bound control, invalid successor, grid limit, thermal-charge feasibility, or schema error. Evaluation and bundle creation perform an isinstance(run, ValidRun) guard.

### Evaluation and Run Bundle Module

~~~python
@dataclass(frozen=True)
class WearCoefficients:
    battery_eur_per_kwh: float = 0.005
    thermal_store_eur_per_kwh: float = 0.0005
    electrolyser_eur_per_kwh: float = 0.002
    fuel_cell_eur_per_kwh: float = 0.002


@dataclass(frozen=True)
class EvaluationPolicy:
    name: str = "greenhouse-hub-evaluation"
    version: str = "1"
    grid_import_fee_eur_per_kwh: float = 0.025
    wear: WearCoefficients = WearCoefficients()
    sensitivity_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0)


@dataclass(frozen=True)
class StepLineItems:
    operating_step: int
    grid_cost_eur: float
    battery_wear_eur: float
    thermal_store_wear_eur: float
    electrolyser_wear_eur: float
    fuel_cell_wear_eur: float
    operating_cost_eur: float
    comfort_violation_c_h: float


@dataclass(frozen=True)
class CostSummary:
    grid_cost_eur: float
    battery_wear_eur: float
    thermal_store_wear_eur: float
    electrolyser_wear_eur: float
    fuel_cell_wear_eur: float
    operating_cost_eur: float
    inventory_settlement_eur: float
    inventory_adjusted_cost_eur: float
    comfort_violation_c_h: float


@dataclass(frozen=True)
class EvaluationReport:
    policy: EvaluationPolicy
    nominal: CostSummary
    wear_sensitivities: Mapping[str, CostSummary]
    step_line_items: tuple[StepLineItems, ...]


@dataclass(frozen=True)
class RunSpecification:
    identifier: str
    canonical_content: Mapping[str, JSONValue]
    publication_eligible: bool


@dataclass(frozen=True)
class RunBundle:
    identifier: str
    specification_identifier: str
    path: Path
    manifest: Mapping[str, JSONValue]
~~~

Canonical JSON is UTF-8 with sorted keys, separators comma/colon, ensure_ascii=False, and allow_nan=False. Timestamps serialize as UTC ISO-8601 ending in Z. SHA-256 hex digests are always full 64-character identifiers.

Each Run Bundle contains:

~~~text
manifest.json
trajectory.csv
controller_diagnostics.csv
summary.json
validation.json
~~~

The bundle identifier is SHA-256 over canonical manifest content with only run_bundle_identifier omitted. That content already includes the SHA-256 of every non-manifest member. The bundle directory is results/runs/<scenario>--<controller>--<full-bundle-id>. A same-ID directory deduplicates only after byte-for-byte member verification; a different bundle ID for the same Run Specification is retained.

All Mapping inputs crossing these contracts are normalized immediately to canonical dictionaries for identity/serialization or read-only MappingProxyType values for runtime metadata. A frozen dataclass may not retain a caller-owned mutable dictionary or DataFrame.

## Finding-to-Task Traceability

| Ledger item | Reproduction | Behavioral/structural resolution | Full-scope evidence |
|---|---|---|---|
| PF-01 one-step invalid reached states | Task 1 | Task 3 | Tasks 13 and 15 |
| PF-02 Disabled Assets and failed solves | Task 1 | Tasks 2, 4, and 7 | Tasks 13-15 |
| PF-03 out-of-horizon leakage | Task 1 | Tasks 3 and 5 | Tasks 13 and 15 |
| PF-04 overstated baseline fairness/causality | Task 1 | Tasks 6, 7, and 14 | Tasks 14 and 15 |
| PF-05 objective/report mismatch | Task 1 | Task 6 | Tasks 13-15 |
| PF-06 incomplete publication validation | Task 1 | Task 7 | Tasks 13-15 |
| PF-07 Scenario/reproduction inconsistency | Task 1 | Tasks 5 and 11 | Tasks 13-15 |
| AR-01 duplicated numerical/symbolic physics | Task 1 characterization | Task 4 | Task 15 parity evidence |
| AR-02 unstable packaging/path mutation | Task 1 baseline | Tasks 9-12 | Task 15 install/import evidence |

### ADR Traceability

| ADR | Implemented by |
|---|---|
| ADR-0001 four deep Modules | Tasks 4-12 |
| ADR-0002 fail-closed Valid Runs | Tasks 2 and 7 |
| ADR-0003 causal horizons and valid Terminal States | Tasks 3 and 5 |
| ADR-0004 provenance-backed publication | Tasks 7 and 13-14 |
| ADR-0005 inert zero-capacity Disabled Assets | Task 4 |
| ADR-0006 Inventory-Adjusted Cost scorecard | Task 6 |
| ADR-0007 two-level content identity | Task 7 |

---

## Phase 1 — Evidence Before Behavioral Change

### Task 1: Commit characterization coverage and finding-linked regressions

**Files:**

- Create: tests/conftest.py
- Modify: tests/test_hub.py
- Create: tests/test_simulation.py
- Create: tests/test_scenarios.py
- Create: tests/test_evaluation.py
- Create: tests/test_published_artifacts.py

**Interfaces:** Tests use current legacy imports. New-interface imports stay inside strict-xfail test bodies so missing names are test failures rather than collection errors.

- [ ] Add deterministic helpers in tests/conftest.py:

~~~python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=49, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "price_EUR_kWh": np.full(49, 0.10),
            "P_pv_kW": np.zeros(49),
            "P_elec_kW": np.full(49, 400.0),
            "T_out_C": np.full(49, 5.0),
            "G_Wm2": np.zeros(49),
        },
        index=index,
    )
~~~

- [ ] Extend tests/test_hub.py with passing characterization cases for the current battery, H2, TES, temperature, grid slack, exact realized import fee, and Baseline Controller decisions at fixed fixtures. Assert numeric values with pytest.approx; do not characterize the known horizon, time, failure, publication, or economic defects.

- [ ] Add the PF-01 reproduction to tests/test_simulation.py:

~~~python
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
~~~

- [ ] Add a fake unsuccessful do-mpc object and the PF-02 fail-closed reproduction. The fake's make_step sets solver_stats to success=False and exposes deliberately stale non-zero u0 values. Patch control.mpc_controller.build_mpc and models.hub_model.hub_dynamics, then assert no plant call occurs and the returned outcome is InvalidRun. This must fail because the current loop applies u0.

- [ ] Add the PF-03 causal reproduction to tests/test_simulation.py. Build two current controllers with 49 samples, identical indices 0 through 24 at 0.10 EUR/kWh, and tails of -1.0 and +1.0 EUR/kWh. Solve from the same initial state and assert all first controls agree within absolute tolerance 1e-4. The current maximum difference is approximately 250 kW.

- [ ] Add PF-07 regressions to tests/test_scenarios.py:

  - load_data(start_month=12, n_days=31) must return 744 Operating Steps or explicitly reject insufficient Forecast Coverage; it may not silently return 743.
  - lighting at 06:00 Europe/Amsterdam is on even though the corresponding UTC hour differs in winter and summer.
  - a spring-forward local calendar day has 23 Operating Steps.
  - a fall-back local calendar day has 25 Operating Steps.
  - missing, duplicate, naive, and off-hour source timestamps raise ScenarioValidationError.

- [ ] Add PF-05 regressions to tests/test_evaluation.py using a two-step hand-built trajectory. Assert exact grid, each wear term, Operating Cost, settlement, Inventory-Adjusted Cost, reached-state Comfort Violation, 0x/1x/2x sensitivities, and that no comfort-cost field exists.

- [ ] Add PF-04 regressions to tests/test_published_artifacts.py: the README may not call the current limited-capability Controller an unqualified fair baseline, every Baseline Run manifest must contain the exact capability policy, and every causal statement about H2, TES, or foresight must cite the corresponding ablation bundle.

- [ ] Add PF-06/ADR-0007 regressions to tests/test_published_artifacts.py. Assert that publication rejects a failed validation report, a missing solver status, a dirty executable source, a changed member byte, a shortened digest, and an unknown bundle ID.

- [ ] Prove the existing suite remains green with explicit expected regressions:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest -q
~~~

Expected: all characterization tests pass and each known finding is reported xfailed, not skipped.

- [ ] Prove the tests really reproduce defects:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest tests/test_simulation.py tests/test_scenarios.py \
        tests/test_evaluation.py tests/test_published_artifacts.py \
        --runxfail -q
~~~

Expected: failures corresponding to PF-01 through PF-07.

- [ ] Commit only tests:

~~~bash
git add tests
git commit -m "test: capture architecture regressions"
~~~

### Task 2: Make Run execution fail closed before any control is applied

**Files:**

- Modify: control/rolling_horizon.py
- Modify: control/mpc_controller.py
- Modify: models/hub_model.py
- Modify: experiments/ablations.py
- Modify: tests/test_simulation.py
- Modify: tests/test_hub.py

**Interfaces:** Add ControlDecision, ControllerFailure, DecisionDiagnostics, OperatingRecord, ValidRun, InvalidRun, ControllerAdapter, and simulate_run using the Stable Interface Contract. Keep run_simulation as a temporary compatibility wrapper returning the same Run outcome union.

- [ ] Run the PF-02 test with --runxfail and confirm it fails because hub_dynamics is called after solver failure.

- [ ] In control/mpc_controller.py, add MpcControllerAdapter.decide. It must use the array returned from make_step only after this check:

~~~python
raw_control = self._mpc.make_step(x0)
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
    terminal_electric_value_eur_per_kwh=None,
    terminal_heat_value_eur_per_kwhth=None,
)
if stats.get("success") is not True:
    return ControllerFailure(
        code="solver_failure",
        message=f"MPC solve failed: {stats.get('return_status', 'unknown')}",
        diagnostics=diagnostics,
    )
control = hub_control_from_array(np.asarray(raw_control).reshape(-1))
return ControlDecision(control=control, diagnostics=diagnostics)
~~~

Do not read mpc.u0 in this path.

- [ ] Configure store_solver_stats to include success, return_status, iter_count, and t_wall_total.

- [ ] In models/hub_model.py, add ValidationIssue, AssetCapabilities, HubConfiguration, HubState, HubControl, ExogenousInputs, HubFlows, and HubStep plus explicit field/model-name maps. Add physical_state_bounds, operational_state_bounds, control_bounds, and finite/control/grid/thermal-charge and successor validation helpers. Do not unify symbolic equations yet. Use physical storage bounds zero-to-capacity and hard temperature bounds for common Run validity.

- [ ] Declare numeric validation tolerances in one place: 1e-6 for balance/state comparisons, 1e-4 kW for solver-bound noise, and 1e-3 kW for simultaneous battery, TES, or H2 charge/discharge. Values below tolerance are normalized to zero in serialized controls; values above tolerance invalidate the Run.

- [ ] In control/rolling_horizon.py, implement simulate_run so the sequence is exactly: request decision, reject ControllerFailure, validate controls, evaluate shared physics, validate flows and reached state, append record, then advance state. Return InvalidRun immediately at the first failed gate.

- [ ] Wrap baseline_control in BaselineControllerAdapter. Its diagnostics use solver_success=None, solver_return_status=None, and decision_status="success"; its capability_policy records no hydrogen, no TES charging, no grid battery charging, and fixed-threshold battery discharge.

- [ ] Change the CLI and experiments/ablations.py to require isinstance(outcome, ValidRun) before serializing or comparing a trajectory. InvalidRun may write diagnostics to stdout in this task but may not write any result CSV.

- [ ] Run the PF-02 test with --runxfail. Expected: pass and the fake plant call count remains zero.

- [ ] Remove only the PF-02 solver-failure xfail marker and run:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest tests/test_simulation.py -k "solver_failure or invalid_control or invalid_successor" -q
~~~

Expected: pass.

- [ ] Run the full suite. Remaining finding tests stay xfailed.

- [ ] Commit:

~~~bash
git add control/rolling_horizon.py control/mpc_controller.py models/hub_model.py \
        experiments/ablations.py tests/test_simulation.py tests/test_hub.py
git commit -m "fix: fail closed on invalid controller steps"
~~~

### Task 3: Enforce causal Forecast Horizons and valid Terminal States

**Files:**

- Modify: control/mpc_controller.py
- Modify: control/rolling_horizon.py
- Modify: tests/test_simulation.py

**Interfaces:** build_mpc accepts HubConfiguration and MpcConfiguration, not complete forecast arrays. MpcControllerAdapter.decide accepts exactly horizon_steps + 1 immutable points.

- [ ] Prove PF-01 and PF-03 fail under --runxfail before editing.

- [ ] Replace complete-array terminal coefficients with model TVPs terminal_electric_value and terminal_heat_value. In decide, calculate:

~~~python
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
~~~

Store both in DecisionDiagnostics. No function in controllers/mpc.py may retain a Scenario, DataFrame, or array longer than the supplied view.

- [ ] Replace the clamp in tvp_fun. The closure reads only self._active_forecast and performs direct indexing 0 through horizon_steps. A wrong-length view returns ControllerFailure(code="forecast_coverage") before make_step.

- [ ] Enable terminal bounds using the installed do-mpc 5.1 API:

~~~python
mpc.set_param(
    n_horizon=config.horizon_steps,
    t_step=DT_H * 3600,
    n_robust=0,
    use_terminal_bounds=True,
    store_full_solution=True,
    store_solver_stats=["success", "return_status", "iter_count", "t_wall_total"],
    nlpsol_opts={
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 800,
        "ipopt.tol": 1e-6,
    },
)
for field_name, (lower, upper) in operational_state_bounds(hub_config).items():
    model_name = STATE_MODEL_NAMES[field_name]
    mpc.terminal_bounds["lower", model_name] = lower
    mpc.terminal_bounds["upper", model_name] = upper
~~~

For indoor temperature, terminal_bounds uses the existing hard 5-40 C bounds; comfort remains a separately reported soft constraint.

- [ ] Introduce a temporary CoveredFrame in control/rolling_horizon.py with frame and operating_step_count. load_data accepts forecast_hours, loads Operating Window plus that explicit coverage, and rejects missing coverage. simulate_run iterates only operating_step_count and slices N+1 points per decision. This is a bridge; Task 5 replaces it with Scenario.

- [ ] Add a diagnostic test proving the terminal coefficients equal only the stage points in the supplied view.

- [ ] Add a final-step test proving missing coverage returns InvalidRun(code="forecast_coverage") and does not clamp, repeat, synthesize, or shorten.

- [ ] Run PF-01 and PF-03 with --runxfail. Expected: both pass. The 48-step one-hour winter Run remains within operational storage bounds and hard temperature bounds.

- [ ] Remove only the PF-01 and PF-03 xfail markers, then run:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest tests/test_simulation.py -k "one_step or horizon or forecast or terminal" -q
~~~

Expected: pass.

- [ ] Commit:

~~~bash
git add control/mpc_controller.py control/rolling_horizon.py tests/test_simulation.py
git commit -m "fix: constrain causal horizon terminal states"
~~~

### Task 4: Give numerical and CasADi physics one owner and make Disabled Assets inert (AR-01)

**Files:**

- Modify: models/hub_model.py
- Modify: control/mpc_controller.py
- Modify: control/rolling_horizon.py
- Modify: accounting.py
- Modify: tests/test_hub.py
- Modify: tests/test_simulation.py

**Interfaces:** Implement every Hub type and function in the Stable Interface Contract. hub_step_expressions must contain only ordinary arithmetic and Python configuration branches so it accepts both floats and CasADi symbols.

- [ ] Add parameterized tests over nominal values, every edge bound, deterministic random interior points, no-H2, and no-TES. Construct a CasADi Function from hub_step_expressions and compare successor states and flows to advance_hub with absolute tolerance 1e-8.

- [ ] Add PF-02 Disabled Asset tests asserting exact zero initial state, state bounds, control bounds, successor state, dynamic flows, and recoverable inventory contribution.

- [ ] Run the new tests with --runxfail where marked. Expected: fail because physics is duplicated and disabled storage retains non-zero state semantics.

- [ ] Move conversions, balance, state updates, and heat equations into hub_step_expressions. Keep exact realized max operations out of shared physics. thermal_charge_margin_kw is thermal_charge_kw - generated_heat_kw and is validated numerically as <= tolerance.

- [ ] Make hub_dynamics a temporary dictionary compatibility wrapper over advance_hub. Delete the duplicate conversion/dynamics block from control/mpc_controller.py and bind the returned shared expressions into model.set_rhs and model.set_expression.

- [ ] Apply capabilities before expressions:

~~~python
battery_charge = control.battery_charge_kw if config.capabilities.battery else 0.0
battery_discharge = control.battery_discharge_kw if config.capabilities.battery else 0.0
electrolyser = control.electrolyser_kw if config.capabilities.hydrogen else 0.0
fuel_cell = control.fuel_cell_kw if config.capabilities.hydrogen else 0.0
thermal_charge = control.thermal_charge_kw if config.capabilities.thermal_store else 0.0
thermal_discharge = control.thermal_discharge_kw if config.capabilities.thermal_store else 0.0

soc_battery_next = (
    state.soc_battery_kwh + ETA_BAT_CH * battery_charge * DT_H
    - battery_discharge / ETA_BAT_DIS * DT_H
    if config.capabilities.battery else 0.0
)
soc_hydrogen_next = (
    state.soc_hydrogen_kg + (hydrogen_production - hydrogen_consumption) * DT_H
    if config.capabilities.hydrogen else 0.0
)
soc_thermal_next = (
    ETA_TES_STANDING * state.soc_thermal_kwh
    + thermal_charge * DT_H - thermal_discharge * DT_H
    if config.capabilities.thermal_store else 0.0
)
~~~

- [ ] Keep numerical scaling factors at their positive nominal capacities even for a Disabled Asset. Zero capacity is represented by state/control bounds and expressions; it must not create a zero scaling divisor.

- [ ] Pass one HubConfiguration instance through initial state, numerical advance, Controller construction, validation, and inventory helpers. Remove disable_h2 and disable_tes booleans from lower-level functions; CLI options construct AssetCapabilities instead.

- [ ] Run a two-day winter full, no-H2, and no-TES MPC validation test. All must return ValidRun with every disabled field exactly zero.

- [ ] Remove the Disabled Asset xfail markers and run:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest tests/test_hub.py tests/test_simulation.py -q
~~~

Expected: pass with only later-phase regressions xfailed.

- [ ] Commit:

~~~bash
git add models/hub_model.py control/mpc_controller.py control/rolling_horizon.py \
        accounting.py tests/test_hub.py tests/test_simulation.py
git commit -m "refactor: unify hub physics and asset capabilities"
~~~

## Phase 2 — Deepen Scenario and Evaluation Modules

### Task 5: Implement exact Scenario time, coverage, and provenance semantics

**Files:**

- Create: scenarios.py
- Modify: control/rolling_horizon.py
- Modify: data/fetch_prices.py
- Modify: data/fetch_pvgis.py
- Modify: data/generate_demand.py
- Create: data/grid_price_signal.provenance.json
- Create: data/pv_profile.provenance.json
- Create: data/demand_profile.provenance.json
- Regenerate: data/demand_profile.csv
- Modify: tests/test_scenarios.py
- Modify: tests/test_simulation.py

**Interfaces:** Implement ScenarioPoint, SourceProvenance, Scenario, ScenarioValidationError, ScenarioCoverageError, build_scenario, lighting_schedule, and derive_electrical_demand.

- [ ] Prove every PF-07 regression fails under --runxfail.

- [ ] In build_scenario, require timezone-aware Europe/Amsterdam start/end values, max_horizon_steps, and a one-hour step. Calendar-day requests additionally require a local-midnight start and compute end by advancing local dates, not by adding 24 times days.

- [ ] Build expected indices exactly:

~~~python
start_date = operating_start.date()
end_date = start_date + datetime.timedelta(days=calendar_days)
operating_end = pd.Timestamp(
    datetime.datetime.combine(end_date, datetime.time.min),
    tz="Europe/Amsterdam",
)
operating_index = pd.date_range(
    start=operating_start.tz_convert("UTC"),
    end=operating_end.tz_convert("UTC"),
    freq="h",
    inclusive="left",
)
forecast_end = operating_end + max_horizon_steps * pd.Timedelta(hours=1)
coverage_index = pd.date_range(
    start=operating_index[0],
    end=forecast_end.tz_convert("UTC"),
    freq="h",
    inclusive="left",
)
~~~

- [ ] Validate each source before alignment: DatetimeIndex, timezone-aware, UTC-convertible, unique, exactly hourly, finite required columns, and no missing expected instants. Never call dropna, drop_duplicates, forward-fill, backward-fill, repeat, clamp, or implicit resample.

- [ ] Reindex 2023 price bytes directly by UTC instant. Map the 2020 PV/weather profile to target UTC instants through the explicit transformation source_utc_calendar_transplant using (month, day, hour), rejecting any missing or duplicate key. Record that transformation and both source/target coverage in provenance.

- [ ] Make derive_electrical_demand consume the target UTC index and aligned irradiance, convert the target index to Europe/Amsterdam for lighting_schedule, and then apply the existing seasonal envelope, 06:00-22:00 local photoperiod, daylight dimming, and base load. Scenario construction no longer trusts the legacy UTC-scheduled demand CSV.

- [ ] Change data/generate_demand.py into a thin caller of derive_electrical_demand. The generated demand is a non-authoritative materialized view; Scenario construction derives the same values from target timestamps and PV irradiance so temporal semantics have one owner. During the legacy layout invoke it as:

~~~bash
python3 -m data.generate_demand --target-year 2023
~~~

It writes the derived bytes and a JSON provenance sidecar.

- [ ] Add --provenance-only to data/fetch_prices.py and data/fetch_pvgis.py. It hashes and describes existing bytes without a network call. The sidecars include source URL/name, parameters, original timezone, units, transformations, row count, UTC coverage, and output digest. Historic acquisition time is null and provenance_status is legacy-import; never invent missing provenance.

- [ ] Hash every input file with streaming SHA-256. SourceProvenance records source identity, path relative to repository root, full digest, acquisition parameters, original timezone, units, transformations, requested Operating Window, Forecast Coverage, and exact UTC coverage.

- [ ] Replace CoveredFrame in control/rolling_horizon.py with Scenario. The simulation loops range(scenario.operating_step_count) and calls scenario.forecast_view.

- [ ] Add exact tests for:

  - winter and summer 14-day windows;
  - a December window that has sufficient source coverage;
  - a cross-month window;
  - 2023-03-26 spring-forward with 23 steps;
  - 2023-10-29 fall-back with 25 steps;
  - winter/summer 06:00 local lighting;
  - missing, duplicate, naive, and off-grid timestamps;
  - insufficient final Forecast Coverage;
  - forecast-only points excluded from Operating Steps.
  - Baseline and MPC comparison Runs reference byte-identical Scenario canonical content built with the comparison's maximum horizon.

- [ ] Run all Scenario tests with --runxfail. Expected: pass.

- [ ] Remove PF-07 temporal xfail markers and run:

~~~bash
python3 -m pytest tests/test_scenarios.py tests/test_simulation.py \
    -k "scenario or coverage or operating or lighting or dst" -q
~~~

Expected: pass.

- [ ] Materialize and validate provenance without network access:

~~~bash
python3 -m data.fetch_prices --provenance-only
python3 -m data.fetch_pvgis --provenance-only
python3 -m data.generate_demand --target-year 2023
python3 -m pytest tests/test_scenarios.py -q
~~~

Expected: acquired price/PV CSV bytes are unchanged; demand is a 2023 UTC hourly materialization whose local lighting agrees exactly with derive_electrical_demand; all three sidecars validate.

- [ ] Commit Scenario behavior and its input-provenance materialization; do not move files yet:

~~~bash
git add scenarios.py control/rolling_horizon.py data/fetch_prices.py \
        data/fetch_pvgis.py data/generate_demand.py data/*.provenance.json \
        data/demand_profile.csv tests/test_scenarios.py tests/test_simulation.py
git commit -m "feat: validate scenario time and provenance"
~~~

### Task 6: Apply one named Evaluation Policy to every Controller

**Files:**

- Modify: accounting.py
- Modify: control/rolling_horizon.py
- Modify: control/mpc_controller.py
- Modify: experiments/ablations.py
- Modify: tests/test_evaluation.py
- Modify: tests/test_simulation.py

**Interfaces:** Implement WearCoefficients, EvaluationPolicy, StepLineItems, CostSummary, EvaluationReport, evaluate_step, evaluate_run, recoverable_inventory_kwh, and saving_percent.

- [ ] Prove PF-05 fails with --runxfail.

- [ ] evaluate_step must calculate exactly:

~~~python
grid_cost = (
    point.price_eur_per_kwh * flows.grid_kw
    + policy.grid_import_fee_eur_per_kwh * max(flows.grid_kw, 0.0)
) * step_hours
battery_wear = policy.wear.battery_eur_per_kwh * (
    control.battery_charge_kw + control.battery_discharge_kw
) * step_hours
thermal_wear = policy.wear.thermal_store_eur_per_kwh * (
    control.thermal_charge_kw + control.thermal_discharge_kw
) * step_hours
electrolyser_wear = (
    policy.wear.electrolyser_eur_per_kwh
    * control.electrolyser_kw * step_hours
)
fuel_cell_wear = (
    policy.wear.fuel_cell_eur_per_kwh
    * control.fuel_cell_kw * step_hours
)
comfort_violation = (
    max(record.reached_state.indoor_temperature_c - T_MAX_C, 0.0)
    + max(T_MIN_C - record.reached_state.indoor_temperature_c, 0.0)
) * step_hours
~~~

- [ ] CostSummary.operating_cost_eur is grid plus all four wear terms. Settlement price is the arithmetic mean wholesale price over Operating Records. Inventory settlement is settlement_price times initial recoverable inventory minus Terminal State recoverable inventory. Disabled Assets contribute exactly zero.

- [ ] evaluate_run rejects InvalidRun and asserts that the count of records equals the Scenario Operating Step count. It recomputes all values from records; it never trusts a precomputed cost column.

- [ ] Produce 0x, 1x, and 2x wear sensitivity summaries from the same immutable records. The nominal summary is 1x and the provisional coefficient label appears in serialized policy metadata.

- [ ] Pass the Evaluation Policy wear coefficients into MpcConfiguration as explicitly named Solver Objective economic terms. Keep complementarity, smoothing, terminal value, and soft comfort weights in separate solver_diagnostics metadata. Add a reconciliation test proving solver-only weight changes do not alter evaluate_run output.

- [ ] Delete effective_cost and COMFORT_PENALTY_EUR_PER_CH from experiments/ablations.py. Its table publishes Grid Cost, Operating Cost, Inventory-Adjusted Cost, Comfort Violation, and wear sensitivities separately.

- [ ] Add a Baseline/MPC same-policy comparison test and assert that the Baseline capability policy remains unchanged.

- [ ] Run the PF-04 capability-policy test with --runxfail and make its manifest assertions pass here. Keep the README-language and causal-citation portions strict xfail until Task 14 regenerates publication text from valid ablations.

- [ ] Run with --runxfail, remove PF-05 xfail markers, and run:

~~~bash
python3 -m pytest tests/test_evaluation.py tests/test_simulation.py \
    -k "evaluation or cost or wear or inventory or comfort or baseline" -q
~~~

Expected: pass.

- [ ] Commit:

~~~bash
git add accounting.py control/rolling_horizon.py control/mpc_controller.py \
        experiments/ablations.py tests/test_evaluation.py tests/test_simulation.py
git commit -m "feat: define shared experiment evaluation policy"
~~~

### Task 7: Add two-level identities and atomic Run Bundles

**Files:**

- Modify: accounting.py
- Modify: control/rolling_horizon.py
- Modify: experiments/ablations.py
- Modify: tests/test_evaluation.py
- Modify: tests/test_published_artifacts.py
- Modify: .gitignore

**Interfaces:** Add canonical_json_bytes, sha256_bytes, collect_code_provenance, build_run_specification, run_specification_identifier, serialize_valid_run, create_run_bundle, verify_run_bundle, load_run_bundle, and write_failure_diagnostics.

- [ ] Prove PF-06 and identity tests fail with --runxfail.

- [ ] Implement canonical JSON exactly:

~~~python
def canonical_json_bytes(value: JSONValue) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
~~~

- [ ] collect_code_provenance receives an explicit tuple of executable paths. For each path, compare working bytes with git show HEAD:<path>; record the full SHA-256 and fail publication eligibility if any executable/configuration input differs or is untracked. Ignore unrelated untracked files, generated outputs, caches, and other non-executable files.

- [ ] Run Specification canonical content includes canonicalization/schema versions, Scenario content/provenance, Controller configuration and capability policy, HubConfiguration, EvaluationPolicy, git revision, executable path/hash map and aggregate source digest, Python/platform versions, actual do-mpc/CasADi/NumPy/pandas versions, and input hashes.

- [ ] Serialize trajectory.csv and controller_diagnostics.csv with fixed column order, UTF-8, newline="\n", finite numeric values, UTC Z timestamps, and one row per Operating Step. summary.json is EvaluationReport. validation.json contains every step's solver and invariant evidence plus complete=true and valid=true.

- [ ] Compute member hashes, then the bundle identifier:

~~~python
identity_manifest = {
    "schema_version": "run-bundle-v1",
    "canonicalization_version": "canonical-json-v1",
    "run_specification_identifier": specification.identifier,
    "scenario": specification.scenario,
    "controller": specification.controller,
    "asset_capabilities": specification.asset_capabilities,
    "evaluation_policy": specification.evaluation_policy,
    "code_provenance": specification.code_provenance,
    "runtime": specification.runtime,
    "input_hashes": specification.input_hashes,
    "member_hashes": member_hashes,
    "valid": True,
}
bundle_id = sha256_bytes(canonical_json_bytes(identity_manifest))
manifest = {**identity_manifest, "run_bundle_identifier": bundle_id}
~~~

- [ ] create_run_bundle accepts only ValidRun, a valid EvaluationReport, a publication-eligible RunSpecification, and a target root. Write all members into a sibling temporary directory, fsync files, verify the temporary bundle, and atomically os.replace it into the full-ID path. On interruption, remove only that owned temporary directory.

- [ ] If the destination exists, verify its full identifier and every member byte. Return it only when identical. Raise BundleCollisionError on any mismatch. Never overwrite.

- [ ] write_failure_diagnostics accepts InvalidRun and writes under results/diagnostics/<run-specification-id>/<UTC-timestamp>/; it must not call create_run_bundle and its directory cannot be parsed as a Run Bundle.

- [ ] Add tests for input/source/config/runtime identity changes, canonical key order, tampering, full-ID enforcement, prefix collision rejection, exact deduplication, divergent bundles for one specification, and interrupted writes.

- [ ] Update legacy experiment entry points so valid executions create Run Bundles and failures create diagnostics. Stop writing mutable canonical/scenario CSVs.

- [ ] Add exactly results/runs/.tmp-*/ and results/diagnostics/ to .gitignore. Final Run Bundle directories under results/runs remain trackable.

- [ ] Run with --runxfail; remove identity/publication-gating xfail markers only when:

~~~bash
python3 -m pytest tests/test_evaluation.py tests/test_published_artifacts.py -q
~~~

Expected: pass.

- [ ] Commit:

~~~bash
git add accounting.py control/rolling_horizon.py experiments/ablations.py \
        tests/test_evaluation.py tests/test_published_artifacts.py .gitignore
git commit -m "feat: persist verified content-addressed run bundles"
~~~

### Task 8: Behavioral review gate before any source move

**Files:** No production edits. Test edits are allowed only to correct an evidenced test defect.

- [ ] Run:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    python3 -m pytest -q
~~~

Expected: all implemented behavioral tests pass; only final-publication tests that require regenerated bundles may remain strict xfail.

- [ ] Execute two-day winter Baseline, full MPC, one-step MPC, no-H2, and no-TES specifications into a temporary directory. Verify every outcome is ValidRun, every bundle verifies, and all variants use the same Evaluation Policy.

- [ ] Inspect:

~~~bash
git status --short
git diff --check
rg -n "min\(k \+ i|dropna\(|drop_duplicates\(|effective_cost|COMFORT_PENALTY" \
    control models accounting.py scenarios.py experiments tests
~~~

Expected: no forecast clamp, silent temporal discard, or comfort monetization remains. git status may show only intentionally preserved unrelated files and ignored temporary outputs.

- [ ] Do not commit generated temporary bundles. Record the gate command and result in the implementation handoff.

## Phase 3 — Structural Migration Only

### Task 9: Move stabilized Modules into the installable src package

**Files:**

- Create: src/greenhouse_energy_hub/__init__.py
- Create: src/greenhouse_energy_hub/controllers/__init__.py
- Move: models/hub_model.py -> src/greenhouse_energy_hub/hub.py
- Move: control/rolling_horizon.py -> src/greenhouse_energy_hub/simulation.py
- Move: control/mpc_controller.py -> src/greenhouse_energy_hub/controllers/mpc.py
- Move: scenarios.py -> src/greenhouse_energy_hub/scenarios.py
- Move: accounting.py -> src/greenhouse_energy_hub/evaluation.py
- Modify: pyproject.toml
- Modify: imports in tests and experiments only

**Interfaces:** No public name, signature, formula, default, tolerance, or serialized field changes are allowed in this task.

- [ ] Use git mv for every whole-file move.

- [ ] Configure setuptools src discovery:

~~~toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
~~~

- [ ] Keep __init__.py minimal. Export package __version__ only; callers import from owning Modules.

- [ ] Update imports to greenhouse_energy_hub.hub, greenhouse_energy_hub.simulation, greenhouse_energy_hub.scenarios, greenhouse_energy_hub.evaluation, and greenhouse_energy_hub.controllers.mpc. Do not add compatibility shims at legacy paths.

- [ ] Run the full suite against an editable installation:

~~~bash
python3 -m venv --system-site-packages /tmp/geh-architecture-env
/tmp/geh-architecture-env/bin/python -m pip install \
    --no-deps --no-build-isolation -e .
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    /tmp/geh-architecture-env/bin/python -m pytest -q
~~~

Expected: same result as Task 8.

- [ ] Commit only the package moves/config/import rewrites:

~~~bash
git add -A src models control scenarios.py accounting.py pyproject.toml tests experiments
git status --short
git commit -m "refactor: move deep modules into src package"
~~~

### Task 10: Extract the Baseline Adapter and thin scenario CLI

**Files:**

- Create: src/greenhouse_energy_hub/controllers/baseline.py
- Modify: src/greenhouse_energy_hub/simulation.py
- Create: experiments/run_scenario.py
- Modify: experiments/ablations.py
- Delete empty legacy models/ and control/ package files if still present
- Modify: tests/test_simulation.py

**Interfaces:** Move BaselineControllerAdapter and its characterized policy unchanged. Move CLI parsing/printing out of simulation; do not change Run behavior.

- [ ] Copy the existing Baseline Controller block verbatim into controllers/baseline.py, switch only imports, then delete the old block.

- [ ] experiments/run_scenario.py may parse paths, start/end/days, controller name, controller horizon, Scenario maximum horizon, terminal weight, capability flags, and an optional publication candidate key. It constructs Scenario, HubConfiguration, Controller, EvaluationPolicy, and RunSpecification; calls simulate_run; then calls create_run_bundle or write_failure_diagnostics. It owns no physics, validation, evaluation formula, or serialization. When a candidate key is supplied for a verified bundle, atomically record the key-to-full-ID mapping in the ignored, non-authoritative results/diagnostics/publication-candidates.json file.

- [ ] experiments/ablations.py imports the same construction functions and defines only these variants:

~~~python
VARIANTS = {
    "full": {"horizon_steps": 24},
    "no-h2": {"horizon_steps": 24, "hydrogen": False},
    "no-tes": {"horizon_steps": 24, "thermal_store": False},
    "one-step": {"horizon_steps": 1},
}
~~~

- [ ] Add tests proving both Controller Adapters cross simulate_run and that the Baseline capability metadata is stable.

- [ ] Run:

~~~bash
/tmp/geh-architecture-env/bin/python -m pytest tests/test_simulation.py -q
/tmp/geh-architecture-env/bin/python experiments/run_scenario.py --help
/tmp/geh-architecture-env/bin/python experiments/ablations.py --help
~~~

Expected: pass/help exit zero.

- [ ] Commit:

~~~bash
git add src/greenhouse_energy_hub experiments tests/test_simulation.py models control
git commit -m "refactor: separate controller adapters and experiment cli"
~~~

### Task 11: Separate acquisition scripts from source and derived data

**Files:**

- Move: data/fetch_prices.py -> scripts/fetch_prices.py
- Move: data/fetch_pvgis.py -> scripts/fetch_pvgis.py
- Move: data/generate_demand.py -> scripts/generate_demand.py
- Move: data/grid_price_signal.csv -> data/source/grid_price_signal.csv
- Move: data/pv_profile.csv -> data/source/pv_profile.csv
- Move: data/demand_profile.csv -> data/derived/demand_profile.csv
- Move: data/grid_price_signal.provenance.json -> data/source/grid_price_signal.provenance.json
- Move: data/pv_profile.provenance.json -> data/source/pv_profile.provenance.json
- Move: data/demand_profile.provenance.json -> data/derived/demand_profile.provenance.json
- Modify: src/greenhouse_energy_hub/scenarios.py
- Modify: scripts/*.py
- Modify: tests/test_scenarios.py

**Interfaces:** This task contains file moves and path rewrites only. Acquisition, derivation, temporal semantics, hashes, and provenance content were stabilized in Task 5.

- [ ] Use git mv for scripts, CSVs, and sidecars; update path constants/imports only.

~~~bash
/tmp/geh-architecture-env/bin/python scripts/fetch_prices.py --provenance-only
/tmp/geh-architecture-env/bin/python scripts/fetch_pvgis.py --provenance-only
/tmp/geh-architecture-env/bin/python scripts/generate_demand.py --target-year 2023
/tmp/geh-architecture-env/bin/python -m pytest tests/test_scenarios.py -q
~~~

Expected: source and derived CSV hashes are unchanged by the move/check commands; all sidecars validate at their new paths.

- [ ] Commit structural changes only:

~~~bash
git add -A scripts data src/greenhouse_energy_hub/scenarios.py tests/test_scenarios.py
git commit -m "refactor: separate scripts from scenario data"
~~~

### Task 12: Prove installed-package ownership and remove path mutation (AR-02)

**Files:**

- Modify: tests/test_hub.py
- Modify: tests/test_simulation.py
- Modify: tests/test_scenarios.py
- Modify: tests/test_evaluation.py
- Modify: tests/test_published_artifacts.py
- Modify: experiments/*.py
- Modify: notebooks/results_analysis.ipynb code cells only
- Modify: pyproject.toml optional dev dependencies if required

**Interfaces:** No legacy import or sys.path mutation remains.

- [ ] Remove every sys.path insertion and root-module import.

- [ ] Add an install smoke test that creates a temporary venv with system site packages, installs this repository with --no-deps, changes cwd outside the repository, and imports all four Modules.

- [ ] Add a static test that fails if project Python or notebook code contains sys.path.insert or imports models., control., or bare accounting.

- [ ] Update notebook imports to the installed package, but do not change its legacy input/result cells yet; publication conversion is Task 14.

- [ ] Run:

~~~bash
python3 -m venv --system-site-packages /tmp/geh-install-test
/tmp/geh-install-test/bin/python -m pip install \
    --no-deps --no-build-isolation -e .
cd /tmp
/tmp/geh-install-test/bin/python -c \
  "import greenhouse_energy_hub.hub, greenhouse_energy_hub.simulation, greenhouse_energy_hub.scenarios, greenhouse_energy_hub.evaluation"
cd - >/dev/null
rg -n "sys\.path|from models|from control|import accounting|from accounting" \
    src experiments scripts tests notebooks
/tmp/geh-architecture-env/bin/python -m pytest -q
~~~

Expected: import succeeds outside the repository; rg has no matches; tests pass except final-publication xfails.

- [ ] Commit:

~~~bash
git add pyproject.toml src experiments scripts tests notebooks/results_analysis.ipynb
git commit -m "test: enforce installed package boundaries"
~~~

## Phase 4 — Validated Regeneration and Completion Audit

### Task 13: Execute all required full-scope Valid Runs

**Files generated:** results/runs/* or results/diagnostics/* only.

**Review gate:** All executable source/configuration changes must already be committed. Unrelated local files may remain untracked because they are excluded from executable provenance.

- [ ] Verify:

~~~bash
git diff --check
git status --short
/tmp/geh-architecture-env/bin/python -m pytest -q
~~~

Expected: no tracked executable changes; only intentionally preserved unrelated files and final-publication xfails may remain.

- [ ] Use these publication windows so source coverage is exact under Europe/Amsterdam semantics:

  - Winter: 2023-01-02T00:00:00+01:00 for 14 local calendar days.
  - Summer: 2023-06-01T00:00:00+02:00 for 14 local calendar days.

The winter date deliberately starts on January 2 because the retained 2023 price input has no 2022-12-31 23:00 UTC sample required for January 1 local midnight. Do not synthesize that missing hour.

- [ ] Execute Baseline and full MPC for winter and summer:

~~~bash
/tmp/geh-architecture-env/bin/python experiments/run_scenario.py \
    --name winter-2023-14d --start 2023-01-02T00:00:00+01:00 \
    --days 14 --controller baseline --scenario-max-horizon 24 \
    --candidate-key winter-baseline
/tmp/geh-architecture-env/bin/python experiments/run_scenario.py \
    --name winter-2023-14d --start 2023-01-02T00:00:00+01:00 \
    --days 14 --controller mpc --horizon 24 --scenario-max-horizon 24 \
    --candidate-key winter-mpc
/tmp/geh-architecture-env/bin/python experiments/run_scenario.py \
    --name summer-2023-14d --start 2023-06-01T00:00:00+02:00 \
    --days 14 --controller baseline --scenario-max-horizon 24 \
    --candidate-key summer-baseline
/tmp/geh-architecture-env/bin/python experiments/run_scenario.py \
    --name summer-2023-14d --start 2023-06-01T00:00:00+02:00 \
    --days 14 --controller mpc --horizon 24 --scenario-max-horizon 24 \
    --candidate-key summer-mpc
~~~

- [ ] Execute the complete winter ablation set:

~~~bash
/tmp/geh-architecture-env/bin/python experiments/ablations.py \
    --name winter-2023-14d --start 2023-01-02T00:00:00+01:00 --days 14 \
    --scenario-max-horizon 24 \
    --candidate-index results/diagnostics/publication-candidates.json
~~~

Required MPC bundles: full horizon, no-H2, no-TES, and one-step. Reuse an already verified full bundle only when the full content identity matches.

- [ ] If any execution returns InvalidRun, stop publication. Retain its diagnostics, diagnose within existing ADRs, add a failing regression, fix in a separate behavioral commit, rerun the full suite, and restart Task 13 from clean committed source.

- [ ] Verify every generated bundle:

~~~bash
/tmp/geh-architecture-env/bin/python -m pytest tests/test_published_artifacts.py \
    -k "generated_bundle or full_scope" -q
~~~

Expected: all bundles have complete successful per-step solver evidence where applicable, physical/temporal/evaluation validation, full provenance, and 0x/1x/2x wear sensitivity.

- [ ] Verify results/diagnostics/publication-candidates.json contains exactly these keys with full 64-character identifiers: winter-baseline, winter-mpc, summer-baseline, summer-mpc, ablation-full, ablation-no-h2, ablation-no-tes, and ablation-one-step. Do not commit this non-authoritative candidate file.

### Task 14: Regenerate publication artifacts only from pinned Run Bundles

**Files:**

- Create: experiments/publish_results.py
- Create: results/publication_manifest.json
- Regenerate: results/figures/*.png
- Modify: notebooks/results_analysis.ipynb
- Modify: README.md
- Modify: tests/test_published_artifacts.py
- Delete: results/baseline_results.csv
- Delete: results/mpc_results.csv
- Delete: results/scenarios/

**Interfaces:** publication_manifest.json is a publication recipe, not a mutable alias. It pins full bundle IDs for winter/summer comparisons, every ablation, each figure, and every README metric.

- [ ] Add experiments/publish_results.py as a thin caller of evaluation Module verification/loading/rendering functions. It refuses a prefix, invalid bundle, failed validation, dirty provenance, missing sensitivity, or comparison with different Evaluation Policies.

- [ ] Create the publication manifest by loading the exact full IDs from the Task 13 candidate file, rejecting missing/extra keys and any non-64-character value, verifying every referenced bundle, and then serializing this mapping:

~~~python
manifest = {
    "schema_version": "publication-manifest-v1",
    "comparisons": {
        "winter": {
            "baseline_bundle_id": candidates["winter-baseline"],
            "mpc_bundle_id": candidates["winter-mpc"],
        },
        "summer": {
            "baseline_bundle_id": candidates["summer-baseline"],
            "mpc_bundle_id": candidates["summer-mpc"],
        },
    },
    "ablations": {
        "full": candidates["ablation-full"],
        "no-h2": candidates["ablation-no-h2"],
        "no-tes": candidates["ablation-no-tes"],
        "one-step": candidates["ablation-one-step"],
    },
    "figures": {
        "fig1_cumulative_cost.png": [
            candidates["winter-baseline"], candidates["winter-mpc"]
        ],
        "fig2_grid_vs_price.png": [candidates["winter-mpc"]],
        "fig3_soc_trajectories.png": [
            candidates["winter-baseline"], candidates["winter-mpc"]
        ],
        "fig4_temperature.png": [
            candidates["winter-baseline"], candidates["winter-mpc"]
        ],
        "fig5_heat_shifting.png": [candidates["winter-mpc"]],
        "fig6_ablation.png": [
            candidates["ablation-full"],
            candidates["ablation-no-h2"],
            candidates["ablation-no-tes"],
            candidates["ablation-one-step"],
        ],
    },
}
~~~

- [ ] Replace notebook loading of mutable CSVs with load_run_bundle using IDs from publication_manifest.json. Recompute every table and figure through evaluate_run/load verification, not stored display values.

- [ ] Replace README's unqualified fair-baseline language with its exact capability policy. Replace effective comfort cost with separate Inventory-Adjusted Cost and Comfort Violation. Retain causal statements only when the valid ablation bundle comparison supports them.

- [ ] Put generated README numbers inside explicit generated markers and include full bundle IDs immediately below the table. experiments/publish_results.py rewrites only that marked block.

- [ ] Remove all legacy mutable CSV/scenario artifacts after equivalent verified bundles and the publication manifest exist.

- [ ] Regenerate:

~~~bash
/tmp/geh-architecture-env/bin/python experiments/publish_results.py \
    --candidates results/diagnostics/publication-candidates.json \
    --manifest results/publication_manifest.json
/tmp/geh-architecture-env/bin/python -m jupyter nbconvert \
    --to notebook --execute --inplace notebooks/results_analysis.ipynb
/tmp/geh-architecture-env/bin/python -m pytest tests/test_published_artifacts.py -q
~~~

Expected: every figure/table/README metric recomputes from pinned valid bundles; no publication xfail remains.

- [ ] Run the full suite:

~~~bash
env MPLCONFIGDIR=/tmp/greenhouse-mpl-cache \
    XDG_CACHE_HOME=/tmp/greenhouse-cache \
    /tmp/geh-architecture-env/bin/python -m pytest -q
~~~

Expected: all tests pass, no xfail or skip for required evidence.

- [ ] Add a retention test proving every full ID in publication_manifest.json resolves to a committed Run Bundle; no cleanup path may remove a referenced bundle.

- [ ] Commit regenerated artifacts separately. Keep the ignored diagnostics candidate file out of the commit:

~~~bash
git add experiments/publish_results.py results/runs results/figures \
        results/publication_manifest.json notebooks/results_analysis.ipynb \
        README.md tests/test_published_artifacts.py
git add -u results
git commit -m "results: publish validated provenance-backed runs"
~~~

### Task 15: Close every finding with authoritative evidence

**Files:**

- Modify: docs/reviews/2026-07-16-finding-verification-matrix.md
- Modify: README.md if only evidence links are needed

- [ ] For PF-01 through PF-07 and AR-01 through AR-02, change Status to Resolved only when its full required evidence exists. Replace Completion evidence: Missing with:

  - exact test file and test name;
  - exact verification command;
  - relevant full Run Bundle Identifier(s);
  - relevant source/ADR path;
  - result of the full-scope command.

- [ ] Run the final audit:

~~~bash
/tmp/geh-architecture-env/bin/python -m pytest -q
/tmp/geh-architecture-env/bin/python -m pytest \
    tests/test_published_artifacts.py -q
git diff --check
rg -n "Completion evidence: Missing|Status: Confirmed|Status: Unproven" \
    docs/reviews/2026-07-16-finding-verification-matrix.md
rg -n "sys\.path|fair rule-based|effective cost|baseline_results\.csv|mpc_results\.csv|results/scenarios" \
    README.md src experiments scripts tests notebooks
git status --short
~~~

Expected: tests pass; no unresolved ledger status; no path mutation, unqualified fairness claim, comfort monetization, or mutable-result reference; git status shows at most intentionally preserved unrelated files.

- [ ] Independently verify every publication bundle one final time from its full ID and recompute every manifest summary.

- [ ] Commit the audit:

~~~bash
git add docs/reviews/2026-07-16-finding-verification-matrix.md README.md
git commit -m "docs: close architecture finding verification ledger"
~~~

- [ ] Only after the audit commit and final green suite, record the final verification evidence.

---

## Review Boundaries

1. **Evidence gate:** Task 1 contains tests only.
2. **Behavior gate:** Tasks 2-7 change behavior and Module ownership without source-tree moves or publication regeneration.
3. **Structural gate:** Tasks 9-12 move stabilized code and data without changing scientific formulas or claims.
4. **Regeneration gate:** Tasks 13-14 run only committed executable source and publish only fully verified bundles.
5. **Completion gate:** Task 15 maps every original finding to authoritative, reproducible evidence.

At each gate, present the relevant diff, test output, and remaining finding statuses before continuing if the user requests review.
