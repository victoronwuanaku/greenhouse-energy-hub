# Finding Verification Matrix

This matrix is the completion ledger for the architecture migration. A finding is not resolved by code movement or by a narrow passing test; its completion evidence must prove the required behavior across the scope of the original claim.

## Status language

- **Confirmed** — current source or an authoritative reproduction demonstrates the defect.
- **Unproven** — the current evidence does not establish the required outcome.
- **Resolved** — the required behavior and full-scope verification evidence both exist.

No finding in this document is **Resolved** at the start of the migration.

## Priority findings

### PF-01 — One-step MPC produces invalid reached states

**Status:** Confirmed

**Current evidence:** `experiments/ablations.py` selects `n_horizon=1`, while `control/mpc_controller.py` applies state and comfort constraints to stage states without an explicit constraint on the reached horizon Terminal State. The review reproduction observed storage and temperature outside physical bounds and subsequent infeasible solves.

**Required outcome:** A valid one-operating-step Forecast Horizon changes foresight only; every reached state remains subject to the same physical and operational policy as longer horizons.

**Owning Modules:** Greenhouse-hub physics; validated simulation

**Decision:** ADR-0003

**Required regression evidence:**

- A one-step controller test that inspects its reached battery, hydrogen, thermal-store, and temperature state.
- A multi-step one-hour-horizon Run that remains physically valid at every operating step.
- A parity test showing that horizon length does not change successor-state bounds.

**Completion evidence:** Missing

### PF-02 — Disabled TES and failed solves are published as valid results

**Status:** Confirmed

**Current evidence:** `control/mpc_controller.py` pins TES flows to zero while retaining a positive lower state bound and hourly standing loss. `control/rolling_horizon.py` applies `mpc.u0` without checking `mpc.solver_stats`, and `experiments/ablations.py` writes aggregate artifacts without validating the Run.

**Required outcome:** A Disabled Asset remains in the shared schema with zero capacity, zero initial and admissible state, zero associated controls, no nonzero dynamic contribution, and no inventory value. Every solver failure stops before control application, and no failed Run enters publication.

**Owning Modules:** Validated simulation; greenhouse-hub physics; evaluation/artifact

**Decision:** ADR-0002 and ADR-0005

**Required regression evidence:**

- A forced solver-failure test proving that no control is applied and no Run Bundle is created.
- Configuration and Adapter-parity tests proving that a Disabled Asset remains identically zero and contributes no inventory value.
- Valid no-TES and no-H2 Runs covering the complete published scenario window.
- Manifest and artifact tests proving that every published ablation has successful per-step solver evidence and physical invariant evidence.

**Completion evidence:** Missing

### PF-03 — The controller leaks information beyond its declared horizon

**Status:** Confirmed

**Current evidence:** `control/rolling_horizon.py` passes the complete simulation arrays to `build_mpc`; `control/mpc_controller.py` calculates terminal prices once from those complete arrays. The review reproduction showed different first actions for forecasts that were identical inside the horizon and differed only beyond it.

**Required outcome:** Current control and terminal valuation depend only on the current Hub State, declared Controller configuration, and Scenario information inside the Forecast Horizon.

**Owning Modules:** MPC Controller Adapter; Scenario data; greenhouse-hub physics

**Decision:** ADR-0003

**Required regression evidence:**

- Two scenarios identical through the complete Forecast Horizon but different afterward produce equivalent current controls within a declared numerical tolerance.
- Terminal valuation inputs are observable in diagnostics and contain only horizon-local data.
- Final Operating Steps require explicit Forecast Coverage and never clamp, repeat, or synthesize missing values.

**Completion evidence:** Missing

### PF-04 — Baseline fairness and causal attribution are overstated

**Status:** Confirmed

**Current evidence:** The Baseline Controller never uses hydrogen, never charges TES, cannot charge the battery from cheap grid energy, and uses a fixed discharge threshold. Existing prose describes the comparison as fair and attributes savings to smarter dispatch, while the invalid ablations cannot currently isolate those causes.

**Required outcome:** The Baseline Controller is described by an explicit capability policy; comparisons distinguish temperature fairness from capability parity; causal claims are made only when supported by valid ablations.

**Owning Modules:** Controller Adapters; evaluation/artifact

**Decision:** ADR-0004 and ADR-0006

**Required verification evidence:**

- The Run Bundle records the Baseline Controller capability policy.
- Published tables and README language identify the exact comparison rather than using an unqualified fairness label.
- Every causal attribution links to a valid ablation Run Bundle.

**Completion evidence:** Missing

### PF-05 — Optimized and reported economic quantities do not match

**Status:** Confirmed

**Current evidence:** The MPC objective includes battery, TES, electrolyser, and fuel-cell wear terms, while `accounting.py` reports Grid Cost and inventory settlement only. The ablation script monetizes Comfort Violation with the solver slack weight without an external valuation policy.

**Required outcome:** Every economic quantity has one explicit definition; objective regularization is reconciled with reported Operating Cost; Comfort Violation remains physical unless an approved valuation policy applies.

**Owning Module:** Evaluation/artifact

**Decision:** ADR-0006

**Required regression evidence:**

- Step-level terms sum exactly to each named Run-level economic quantity.
- Changing a solver-only regularization term cannot silently change the definition of a Published Result.
- Comfort valuation is absent unless a named evaluation policy supplies it.
- Baseline and MPC comparisons use the same evaluation policy.
- Nominal, zero-times, and two-times provisional wear sensitivities are reproducible from the same Run trajectories.

**Completion evidence:** Missing

### PF-06 — Passing tests do not validate all published experiments

**Status:** Confirmed

**Current evidence:** `tests/test_hub.py` validates canonical CSVs and selected summary columns but does not require per-step solver success, ablation trajectories, complete summary fields, code/input provenance, or a valid manifest for every Published Result.

**Required outcome:** Every Published Result is reconstructable from Valid Run Bundles whose complete solver, physical, temporal, evaluation, and provenance evidence is checked automatically.

**Owning Modules:** Validated simulation; evaluation/artifact

**Decision:** ADR-0002, ADR-0004, ADR-0006, and ADR-0007

**Required regression evidence:**

- Publication rejects a missing, invalid, or failed Run manifest.
- Publication verifies and pins the full Run Bundle Identifier rather than trusting a filename, alias, or digest prefix.
- Every summary field is recomputed from trajectories and evaluation policy.
- Every figure and README metric identifies its source Run Bundles.
- Tests enumerate every published bundle rather than a mutable canonical subset.

**Completion evidence:** Missing

### PF-07 — Scenario time semantics and documented reproduction are inconsistent

**Status:** Confirmed

**Current evidence:** `load_data` unconditionally drops the last selected price row, silently drops missing aligned rows, and does not enforce requested length. The lighting schedule interprets UTC timestamp hours as operating-clock hours. Canonical output files are overwritten, while the notebook reads whichever canonical files were written most recently.

**Required outcome:** A Scenario has exact validated coverage, explicit source and operating timezones, deterministic alignment, and provenance; publication selects explicit Run Bundles rather than mutable latest-run files.

**Owning Modules:** Scenario data; evaluation/artifact

**Decision:** ADR-0004, ADR-0007, and the architecture design's Scenario temporal policy

**Required regression evidence:**

- Exact-length tests for winter, summer, December year-end, and cross-month windows.
- Tests for UTC source timestamps, Europe/Amsterdam operating-clock schedules, and both daylight-saving transitions.
- Missing and duplicate input hours fail with diagnostic evidence instead of being dropped.
- Insufficient Forecast Coverage fails before a Run starts and forecast-only rows never enter evaluation totals.
- Notebook and figure generation consumes explicit Run Bundle identifiers.

**Completion evidence:** Missing

## Architecture requirements

### AR-01 — Numerical and symbolic physics have one owner

**Status:** Unproven

**Current evidence:** `models/hub_model.py` and `control/mpc_controller.py` contain parallel Implementations of battery, hydrogen, TES, temperature, heat-flow, and grid-balance equations.

**Required outcome:** Physical knowledge has Locality in the greenhouse-hub physics Module, with numerical and CasADi Adapters crossing one physical test surface.

**Decision:** ADR-0001

**Required verification evidence:** Property or parameterized parity tests over representative states, controls, exogenous inputs, enabled/disabled Asset configurations, and edge bounds.

**Completion evidence:** Missing

### AR-02 — Module ownership is reflected by stable packaging

**Status:** Unproven

**Current evidence:** Source code is divided between root modules and technical `models/` and `control/` packages; executable scripts mutate `sys.path`; tests run primarily from the repository root.

**Required outcome:** The four approved Modules live in an installable `src/greenhouse_energy_hub/` package, scripts are thin callers, and imports work without repository-root path mutation.

**Decision:** ADR-0001

**Required verification evidence:** Build/install tests, import tests from outside the repository root, absence of runtime path mutation, and the full test suite against the installed package.

**Completion evidence:** Missing

## Final audit rule

At goal completion, every **Completion evidence** entry above must name an authoritative file, test, command result, and—where applicable—Run Bundle. Evidence that covers only a short run, only canonical CSVs, or only one Controller cannot prove a repository-wide or publication-wide requirement.
