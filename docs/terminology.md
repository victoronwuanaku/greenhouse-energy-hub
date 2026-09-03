# Terminology

Capitalised terms used in the README, docstrings and architecture decision records.

## Physical system

- **Greenhouse Energy Hub** — the coupled electricity, heat, hydrogen, storage and greenhouse-climate system studied here.
- **Asset** — one conversion or storage capability in the hub: battery, thermal store, electrolyser, fuel cell, heat pump, electric boiler, PV.
- **Hub State** — the battery, hydrogen, thermal-store and indoor-temperature quantities needed to continue a simulation.
- **Terminal State** — the Hub State reached after the last operating step of a horizon or run.
- **Disabled Asset** — an Asset kept in the state and control schema for an ablation with zero capacity, zero admissible state and controls, no standing loss and no inventory value.

## Control and simulation

- **Controller** — a policy that selects hub controls from the current Hub State and the scenario information it is allowed to see.
- **MPC Controller** — the rolling-horizon Controller that optimises controls over a declared Forecast Horizon.
- **Baseline Controller** — the rule-based comparison Controller; its missing capabilities are declared in `BASELINE_CAPABILITY_POLICY`.
- **Forecast Horizon** — the future interval whose scenario information the MPC Controller may use for its current decision.
- **Operating Window** — the half-open calendar interval whose Operating Steps a Run applies and evaluates.
- **Forecast Coverage** — the scenario data beyond the Operating Window needed to fill every Forecast Horizon.
- **Operating Step** — one interval in which a Controller selects controls and the hub advances to the next Hub State.
- **Run** — one Controller rollout over one Scenario, including diagnostics and the reached Terminal State.
- **Valid Run** — a Run in which every solve succeeded, every step satisfied the physical invariants, and all provenance is present.
- **Solver Objective** — the quantity the MPC Controller minimises over its horizon (economic estimate plus terminal value, penalties and regularisation). It is not a realised cost.

## Scenario and publication

- **Scenario** — the validated Operating Window plus Forecast Coverage of prices, PV, electrical demand, weather and time-zone semantics supplied to a Run.
- **Run Specification** — the declared Scenario, Controller configuration, Asset capabilities, Evaluation Policy, code/runtime provenance and input hashes requested for a Run.
- **Run Specification Identifier** — SHA-256 of the canonical Run Specification.
- **Run Bundle** — the immutable trajectory, summary, diagnostics, manifest and validation evidence for one Valid Run, stored under `results/runs/`.
- **Run Bundle Identifier** — SHA-256 of the canonical manifest, which includes the hash of every member. One Run Specification may map to several Run Bundles if executions differ.
- **Published Result** — a number or figure derived only from Valid Runs pinned in `results/publication_manifest.json`.

## Evaluation

- **Evaluation Policy** — a named, versioned set of rules that turns a Run's records into economic and physical measures. All compared Runs use the same policy.
- **Grid Cost** — realised import and export settlement, excluding wear and inventory settlement.
- **Operating Cost** — Grid Cost plus the asset throughput and wear terms declared by the Evaluation Policy.
- **Inventory-Adjusted Cost** — Operating Cost adjusted for the change in recoverable stored energy between the initial and Terminal State, priced by the policy's settlement rule. This is the primary economic comparison.
- **Comfort Violation** — accumulated °C·h by which the reached indoor temperature lies outside the comfort band.
