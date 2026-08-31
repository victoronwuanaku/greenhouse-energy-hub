# Greenhouse Energy Hub Experiments

This context defines the language used to model, control, simulate, evaluate, and publish experiments for the representative greenhouse energy hub.

## Language

### Physical system

**Greenhouse Energy Hub**:
The coupled electricity, heat, hydrogen, storage, and greenhouse-climate system studied by this repository.
_Avoid_: Plant, site, system when the coupled hub is intended

**Asset**:
A physical conversion or storage capability within the **Greenhouse Energy Hub**, such as the battery, thermal store, electrolyser, or fuel cell.
_Avoid_: Component, device

**Hub State**:
The battery, hydrogen, thermal-store, and indoor-temperature quantities required to continue a simulation.
_Avoid_: State vector when discussing the domain concept

**Terminal State**:
The **Hub State** reached after the final operating step in a prediction horizon or simulation window.
_Avoid_: Final row, end value

**Disabled Asset**:
A zero-capacity **Asset** that remains in the configured **Hub State** schema for an ablation but has exactly zero admissible state and control values, contributes no standing-loss dynamics or inventory value, and leaves the remaining **Greenhouse Energy Hub** physically valid.
_Avoid_: Zeroed control, removed feature

### Control and simulation

**Controller**:
A policy that selects hub controls from the current **Hub State** and available scenario information.
_Avoid_: Solver, dispatcher

**MPC Controller**:
The rolling-horizon **Controller** that optimizes controls over a declared **Forecast Horizon**.
_Avoid_: Optimizer when the complete controller is intended

**Baseline Controller**:
The explicitly defined comparison **Controller** used to interpret the value of the **MPC Controller**.
_Avoid_: Fair controller, naive controller without stating its capabilities

**Forecast Horizon**:
The future interval whose scenario information the **MPC Controller** is permitted to use for its current decision.
_Avoid_: Full forecast, simulation window

**Operating Window**:
The half-open calendar interval whose **Operating Steps** are applied and evaluated by a **Run**.
_Avoid_: Forecast buffer, complete input range

**Forecast Coverage**:
The validated scenario information required to fill every declared **Forecast Horizon** across an **Operating Window**.
_Avoid_: Operating Window, padded forecast

**Operating Step**:
One interval in which a **Controller** selects controls and the **Greenhouse Energy Hub** advances to the next **Hub State**.
_Avoid_: Row, sample

**Run**:
One controller rollout over one **Scenario**, including diagnostics and its reached **Terminal State**.
_Avoid_: Result, simulation when the complete evaluated execution is intended

**Valid Run**:
A **Run** in which every solve succeeds, every operating step satisfies required invariants, and all required provenance is present.
_Avoid_: Completed run, successful script

**Solver Objective**:
The mathematical quantity minimized by an **MPC Controller** over its **Forecast Horizon**, which may combine economic estimates, terminal-value proxies, constraint penalties, and numerical regularization. It is not a realized economic quantity for a **Run**.
_Avoid_: Operating Cost, total cost

### Scenario and publication

**Scenario**:
A validated **Operating Window** and **Forecast Coverage** of price, PV, electrical demand, weather, timezone semantics, and configuration supplied to a **Run**.
_Avoid_: Dataset, forecast when the complete experiment input is intended

**Run Specification**:
The immutable declared **Scenario**, **Controller** configuration, **Asset** capabilities, **Evaluation Policy**, code/runtime provenance, and input provenance requested for a **Run**.
_Avoid_: Run Bundle, configuration file

**Run Specification Identifier**:
The full content identity of one canonical **Run Specification**.
_Avoid_: Run ID, input filename

**Run Bundle**:
The immutable trajectories, summary, diagnostics, configuration, provenance, and validation evidence for one **Run**.
_Avoid_: Results CSV, output folder

**Run Bundle Identifier**:
The full content identity of one exact **Run Bundle** and all of its members.
_Avoid_: Run ID, directory name, latest run

**Published Result**:
A numerical claim or figure derived exclusively from one or more **Valid Runs**.
_Avoid_: Saved output

### Evaluation

**Evaluation Policy**:
A named and versioned set of rules applied consistently to controller trajectories to calculate economic quantities, inventory settlement, and physical performance measures for a **Run**.
_Avoid_: Solver weights, report logic

**Grid Cost**:
The realized import and export settlement cost for a **Run**, excluding wear, comfort valuation, and inventory settlement.
_Avoid_: Operating cost

**Operating Cost**:
The explicitly configured economic evaluation of a **Run**, including every cost term declared by its **Evaluation Policy**.
_Avoid_: Cost without a qualifier

**Inventory-Adjusted Cost**:
An **Operating Cost** adjusted for the difference between initial and terminal recoverable stored energy under a declared settlement policy.
_Avoid_: Adjusted cost without naming the adjustment

**Comfort Violation**:
The accumulated magnitude and duration by which reached indoor temperature lies outside the declared comfort band.
_Avoid_: Crop damage, comfort cost

## Relationships

- A **Scenario** is consumed by one or more **Runs**.
- Every **Scenario** contains one **Operating Window** plus enough **Forecast Coverage** for its declared controller configurations.
- Every **Run** is governed by exactly one **Run Specification**, which has exactly one **Run Specification Identifier**.
- A **Run** uses exactly one **Controller**.
- A **Controller** advances the **Greenhouse Energy Hub** through one or more **Operating Steps**.
- Every **Operating Step** starts from one **Hub State** and reaches one successor **Hub State**.
- Every **Run** has exactly one reached **Terminal State**.
- An **MPC Controller** chooses controls using a **Solver Objective**; an **Evaluation Policy** evaluates the resulting **Run** independently.
- Runs compared in one experiment use the same **Evaluation Policy**.
- Every **Valid Run** resolves to exactly one **Run Bundle Identifier**; byte-identical Valid Runs may resolve to the same immutable **Run Bundle**.
- Every **Run Bundle** has exactly one **Run Bundle Identifier** and references exactly one **Run Specification Identifier**.
- One **Run Specification Identifier** may correspond to multiple **Run Bundle Identifiers** when executions produce different evidence.
- A **Run Bundle** identifies its **Evaluation Policy**.
- A **Published Result** pins full **Run Bundle Identifiers** belonging to **Valid Runs**.
- **Grid Cost** is one input to **Operating Cost**; **Inventory-Adjusted Cost** applies an additional declared settlement policy.

## Worked example

A **Run** is not a **Valid Run** if its reached **Terminal State** violates a storage bound, even if the solver returned controls for that operating step.

## Flagged ambiguities

- "cost" previously meant **Solver Objective**, **Grid Cost**, and **Inventory-Adjusted Cost**; use the explicit term for the intended quantity.
- "fair baseline" previously meant temperature-fair while omitting several asset capabilities; describe the **Baseline Controller** by its declared capabilities instead.
- "terminal row" previously mixed a storage-accounting record with the domain **Terminal State**; the state is domain data, while its tabular representation is an implementation detail.
- "myopic" previously meant a one-step numerical configuration that did not preserve normal state constraints; reserve the term for a valid **MPC Controller** with a one-operating-step **Forecast Horizon**.
- "run ID" previously conflated the requested inputs with the produced evidence; use **Run Specification Identifier** or **Run Bundle Identifier** explicitly.
- "scenario window" previously mixed evaluated **Operating Window** steps with extra **Forecast Coverage**; name the intended interval explicitly.
