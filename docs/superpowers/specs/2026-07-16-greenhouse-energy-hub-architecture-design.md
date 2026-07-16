# Greenhouse Energy Hub Architecture Design

**Review status:** Complete written design awaiting user approval

## Purpose

Refactor the repository into four deep, testable Modules while correcting the architectural causes of invalid controller runs, duplicated physics, ambiguous scenario construction, and drifting evaluation/publication logic. The migration must preserve verified behavior, separate behavioral corrections from structural moves, and regenerate scientific artifacts only from Valid Runs.

## Approved architecture decisions

The active goal and accepted ADRs establish these decisions:

1. The repository is organized around four deep Modules: greenhouse-hub physics, validated simulation, scenario data, and experiment evaluation/artifacts.
2. Run execution fails closed on solver failure or invariant violation.
3. Controller decisions are causal with respect to the declared Forecast Horizon, and reached Terminal States remain valid.
4. Published Results come only from immutable, provenance-backed Run Bundles.
5. A Disabled Asset remains in the shared state and control schemas as an inert zero-capacity state with exactly zero admissible state and control values.
6. Every comparison publishes a named economic scorecard and uses Inventory-Adjusted Cost as its primary economic quantity under one shared Evaluation Policy.
7. Run Specifications and completed Run Bundles use separate deterministic content identities; full Run Bundle Identifiers pin exact publication evidence.
8. The final source layout is a shallow `src/greenhouse_energy_hub/` package, without enterprise-style layering or speculative extension points.

## Target repository shape

```text
CONTEXT.md
docs/
├── adr/
└── superpowers/
    ├── specs/
    └── plans/

src/greenhouse_energy_hub/
├── hub.py
├── simulation.py
├── scenarios.py
├── evaluation.py
└── controllers/
    ├── baseline.py
    └── mpc.py

experiments/
├── run_scenario.py
└── ablations.py

scripts/
├── fetch_prices.py
├── fetch_pvgis.py
└── generate_demand.py

data/
├── source/
└── derived/

results/
├── runs/
├── diagnostics/
└── figures/

tests/
├── test_hub.py
├── test_simulation.py
├── test_scenarios.py
├── test_evaluation.py
└── test_published_artifacts.py
```

The target tree is an outcome of Module ownership, not the first migration step. Existing files remain in place until characterization and regression tests protect the behavior being moved.

## Module design

### Validated simulation Module

This Module owns a complete Run: controller lifecycle, operating-step sequencing, solver-status handling, state advancement, invariant validation, terminal-state capture, and the decision whether the Run is valid. The Baseline Controller and MPC Controller are two Adapters at one real Seam; both must produce controls that cross the same validation path.

Experiment scripts may choose a Scenario and Controller configuration, but they may not reproduce rollout logic or decide whether output is publishable. The Module returns either a Valid Run with complete diagnostics or an explicit failure that cannot be mistaken for a result.

At each Operating Step, a Controller Adapter receives the current Hub State plus a read-only Scenario view restricted to its Forecast Horizon and returns either a control decision with Adapter-specific diagnostics or an explicit failure. The validated simulation Module checks status, finiteness, control bounds, and required diagnostic fields before it advances the Hub State. A failed MPC solve therefore cannot expose stale `u0` values as an applicable decision; the Baseline Controller crosses the same Seam without pretending to have solver diagnostics.

The Baseline Controller retains its characterized frugal thermostat and battery rule rather than gaining new scientific behavior during this architecture migration. Its capability policy explicitly records that it does not use hydrogen, charge the thermal store, or grid-charge the battery. Publications describe it as a limited-capability reference—not a capability-equivalent or unqualified "fair" Controller—and make causal claims only from validated ablations. Any future capability-matched baseline is a separate scientific decision.

### Greenhouse-hub physics Module

This Module owns asset parameters, Hub State semantics, control bounds, coupled electricity/heat/hydrogen equations, physical state transitions, and physical invariants. Numerical simulation and CasADi optimization are two Adapters over the same physical knowledge rather than independent equation copies.

State transitions, conversions, balances, and derived physical flows are written once as pure expression functions that accept either numerical values or CasADi symbols. The numerical Adapter evaluates those functions directly; the CasADi Adapter binds the same expressions into the do-mpc model. Solver-only smooth approximations belong to the Solver Objective, not to shared physics or realized evaluation. No generic backend framework is introduced beyond the two real Adapters.

The Interface must make reached-state semantics explicit. Every successor Hub State, including the horizon Terminal State, crosses the same invariant checks. A one-step Forecast Horizon and a longer Forecast Horizon differ only in foresight, not in physical feasibility or which successor state is constrained. The MPC Controller calculates terminal value from only its supplied Forecast Horizon view; it cannot receive or aggregate the complete Scenario arrays.

For a Disabled Asset, the Module retains the normal schema position while setting capacity, initial state, state bounds, and associated control bounds to exactly zero. Its dynamics cannot create a nonzero successor through standing loss or flow, and evaluation excludes it from inventory valuation. Both physics Adapters consume this same configured shape. This is the accepted decision in ADR-0005.

### Scenario data Module

This Module owns the meaning and construction of a Scenario: input sources, timezone conversion, calendar alignment, exact requested window, scaling, missing/duplicate-hour rejection, and provenance. Acquisition and demand-generation scripts are source Adapters; they do not define scenario semantics independently.

The Scenario supplied to a Run is immutable and validated. It distinguishes the evaluated Operating Window from the extra Forecast Coverage needed by the configured Controller. The same Scenario construction path is used for baseline comparisons, full MPC, ablations, tests, and publication.

The Operating Window is the half-open interval `[start, end)` between timezone-aware Europe/Amsterdam wall-clock instants. A request expressed in calendar days ends at local midnight after that many dates; it therefore contains 23- or 25-hour days across daylight-saving transitions rather than silently forcing `24 × days` samples. Forecast Coverage extends far enough beyond the Operating Window to supply every stage required by the maximum configured Forecast Horizon at the last Operating Step. Insufficient coverage invalidates the Scenario; values are never clamped, repeated, synthesized, or silently shortened.

All source timestamps are normalized to unique UTC instants for alignment, while schedules such as lighting are evaluated in Europe/Amsterdam local time. Every expected hourly UTC instant across the Operating Window and required Forecast Coverage must have exactly one sample from every required series; missing, duplicate, naive, off-grid, or silently discarded samples invalidate the Scenario. Only Operating Window steps contribute to Run evaluation. Provenance records source identity and hash, acquisition parameters, original timezone, units, transformation steps, Operating Window, Forecast Coverage, and exact UTC coverage.

### Experiment evaluation and artifact Module

This Module owns named economic quantities, comfort measurement, inventory settlement, controller comparison, Run Bundle manifests, artifact persistence, and publication eligibility. Solver Objective regularization remains distinct from reported Operating Cost; only asset throughput and wear terms declared by the Evaluation Policy enter Operating Cost.

One named and versioned Evaluation Policy applies to every compared Controller Adapter. It publishes Grid Cost, Operating Cost, Inventory-Adjusted Cost, and Comfort Violation separately; Inventory-Adjusted Cost is the primary economic comparison. Operating Cost includes only declared throughput and wear terms, while terminal value, anti-cycling, input smoothing, and comfort slack remain Solver Objective diagnostics. Current throughput and wear coefficients are provisional policy inputs and require explicit labeling and sensitivity evidence. Comfort Violation is not monetized without an independently justified policy. This is the accepted decision in ADR-0006.

The initial Evaluation Policy preserves the repository's declared tariff and wear assumptions while making them explicit:

```text
Grid Cost = sum((price × grid power + import fee × max(grid power, 0)) × step duration)
Wear      = sum((0.005 × battery throughput
               + 0.0005 × thermal-store throughput
               + 0.002 × electrolyser power
               + 0.002 × fuel-cell power) × step duration)
Operating Cost = Grid Cost + Wear
Inventory-Adjusted Cost = Operating Cost
                        + settlement price × (initial recoverable inventory
                                            - terminal recoverable inventory)
```

The settlement price is the arithmetic mean wholesale price over the Operating Window, shared by every compared Controller. Recoverable inventory uses the existing discharge efficiencies for battery and hydrogen and the declared heat-pump COP for thermal storage; a Disabled Asset contributes zero. Comfort Violation integrates reached-state degrees outside the band over Operating Window steps. The nominal wear coefficients are the primary policy, with at least zero-times and two-times coefficient sensitivity reported until independently justified.

The Module assigns each requested experiment a Run Specification Identifier over its canonical inputs and provenance, then assigns each completed bundle a separate Run Bundle Identifier over its exact manifest and members. Both use versioned canonical UTF-8 JSON and SHA-256. Bundle creation is atomic; identical content deduplicates only after complete verification, divergent outputs for one specification remain distinct, and Published Results pin full Run Bundle Identifiers. Failed diagnostics are stored under `results/diagnostics/`, never `results/runs/`. This is the accepted decision in ADR-0007.

Each Run Bundle contains `manifest.json`, `trajectory.csv`, `controller_diagnostics.csv`, `summary.json`, and `validation.json`. The manifest records schema and canonicalization versions, both identifiers, Scenario and Controller configuration, Asset capabilities, Evaluation Policy, code revision and executable source-tree digest, actual runtime/dependency versions, input and member hashes, and validity status. The Run Bundle Identifier excludes only its own manifest field to avoid circular hashing. Run Bundle creation requires every executable source and configuration file used by the Run to match its recorded committed content; dirty executable inputs or incomplete executions remain diagnostics, while unrelated working files and generated output paths do not affect eligibility.

The notebook, figures, summary tables, and README consume explicit valid Run Bundles rather than whichever mutable CSV was written most recently.

## Control and data flow

```text
source data → Scenario data Module → validated Scenario
                                         ↓
                                  Run Specification
                                         ↓
Controller Adapter → validated simulation Module ← greenhouse-hub physics Module
                                         ↓
                                     Valid Run
                                         ↓
                              evaluation/artifact Module
                                         ↓
immutable Run Bundle → notebook / figures / summary / README
```

Failure moves in the opposite direction only as diagnostics. It never enters the publication flow.

## Error handling and validity

- A failed or non-success solver status ends the Run before its controls are applied.
- Non-finite controls, out-of-range controls, invalid reached states, grid-limit violations, or scenario-schema violations invalidate the Run.
- Missing Forecast Coverage ends construction before a Run starts; implicit padding is forbidden.
- Validation errors identify the operating step, Controller, Scenario, violated invariant, and relevant values.
- Partial diagnostic traces may be retained under `results/diagnostics/`, but only complete Valid Runs receive Run Bundles.
- Artifact generation verifies the full Run Bundle Identifier, every member hash, validity evidence, and clean source provenance before accepting input.

## Testing strategy

### Characterization tests

Protect only verified current behavior before moves: physical equations, electricity balance, realized Grid Cost, known-good fixture alignment, and valid full-horizon controller behavior. Documented temporal, evaluation, horizon, and publication defects belong in failing regression tests rather than characterization expectations.

### Regression tests

Reproduce each priority finding before its correction:

- one-step terminal state escaping storage and temperature bounds;
- failed solves being accepted and applied;
- disabled TES becoming infeasible through retained standing losses;
- out-of-horizon data changing the current action;
- December scenario truncation and UTC/local lighting mismatch;
- optimized wear terms being absent from the named reported metric;
- published artifact checks omitting solver status, provenance, and ablation trajectories.

### Interface tests

- Numerical and CasADi Adapters produce equivalent successor states and derived flows within declared tolerances.
- Baseline and MPC Adapters cross the same simulation test surface.
- Every Scenario reports exact temporal coverage and provenance, including spring-forward and fall-back calendar windows.
- Step-level Evaluation Policy terms sum exactly to every reported Run-level quantity for both Controller Adapters.
- Run Specification and Run Bundle identity tests cover canonicalization, source/input changes, member tampering, digest-prefix collisions, deduplication, divergent executions, and interrupted atomic writes.
- Every Published Result is reconstructable from its Run Bundles.

### Completion tests

The complete winter and summer Baseline and MPC Runs, plus full-horizon, no-H2, no-TES, and valid one-step MPC Runs, must complete without failed solves or invariant violations before replacement claims are published. README values, figures, and summaries must be regenerated and checked from their pinned authoritative Run Bundle Identifiers.

## Migration sequence

1. Add `CONTEXT.md`, accepted ADRs, and this design specification.
2. Obtain user approval of the complete written design.
3. Write a task-level implementation plan with exact files, tests, commands, and review gates.
4. Add characterization and failing regression tests without moving source files.
5. Correct fail-closed run behavior and deepen the validated simulation Module.
6. Deepen the greenhouse-hub physics Module and prove numerical/CasADi parity.
7. Deepen the Scenario data Module and separate scripts from stored inputs.
8. Deepen the evaluation/artifact Module and introduce immutable Run Bundles.
9. Complete the mechanical `src/` and folder migration after responsibilities stabilize.
10. Regenerate, validate, and audit all published artifacts and claims.

Each behavioral correction, structural move, and artifact regeneration remains a separate reviewable change.

## Non-goals

- Introduce a new optimizer, stochastic or robust MPC, or richer greenhouse physics.
- Add a commercial greenhouse controller or retune scientific parameters beyond what is required to make existing experiments valid.
- Create a generic solver plugin system while do-mpc is the only solver Adapter.
- Split each physical Asset into a shallow Module.
- Add dependency-injection machinery, factories, event buses, or generic helper folders.
- Add a database, remote artifact store, signing infrastructure, or general artifact registry for Run Bundles.
- Preserve an invalid historical claim merely to minimize changes to the README.

## Completion criteria

The migration is complete only when every original finding maps to a behavioral correction and authoritative verification evidence; all Module Interface, regression, scenario, and publication tests pass; no legacy path or mutable canonical artifact remains authoritative; and every published claim is reproducible from provenance-backed Valid Run Bundles.

The authoritative completion ledger is [`docs/reviews/2026-07-16-finding-verification-matrix.md`](../../reviews/2026-07-16-finding-verification-matrix.md). Every entry in that matrix must name full-scope completion evidence before the goal can be marked complete.
