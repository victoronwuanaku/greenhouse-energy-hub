# Economic Comparison Decision Brief

**Review status:** Approved — Approach A, recorded by ADR-0006

## Decision

Choose the primary economic comparison and define how the evaluation/artifact Module separates realized economics from the MPC Controller's Solver Objective. The choice must apply one Evaluation Policy to every Controller, preserve Comfort Violation as a physical measure unless an external valuation is supplied, and prevent terminal inventory from distorting comparisons.

## Current evidence

| Quantity | Current Implementation | Architectural problem |
|---|---|---|
| Grid Cost | `models/hub_model.py` records exact wholesale import/export settlement plus the import surcharge. | It excludes all asset-throughput terms and terminal inventory. |
| Solver Objective | `control/mpc_controller.py` combines a smooth Grid Cost proxy, four throughput/wear terms, anti-cycling, input-move smoothing, comfort slack, and terminal value. | Several terms are Controller tuning devices rather than realized Run economics. |
| Inventory-Adjusted Cost | `accounting.py` adjusts Grid Cost for the change in recoverable stored energy at the scenario mean wholesale price. | It currently bypasses the four throughput/wear terms optimized by the MPC Controller. |
| "Effective cost" | `experiments/ablations.py` adds Comfort Violation at the Solver Objective slack weight of EUR 10/degree-C-hour. | A feasibility/tuning weight is presented as crop economics without an external valuation policy. |

The stored trajectories contain the controls needed to apply the four throughput/wear terms to both the Baseline Controller and MPC Controller. However, their coefficients are declared only in `control/mpc_controller.py`; the repository provides no independent economic source or sensitivity analysis for them.

## Materiality check

Applying the existing four throughput/wear coefficients to the committed trajectories gives the following diagnostic values:

| Scenario | Controller | Grid Cost | Declared throughput/wear | Share of Grid Cost |
|---|---:|---:|---:|---:|
| Winter | Baseline | EUR 44,308.13 | EUR 1.92 | 0.004% |
| Winter | MPC | EUR 41,153.83 | EUR 250.00 | 0.607% |
| Summer | Baseline | EUR 3,461.02 | EUR 33.82 | 0.977% |
| Summer | MPC | EUR 1,573.91 | EUR 180.35 | 11.459% |

Including those terms before inventory settlement would change the saved-trajectory Inventory-Adjusted Cost reduction from 6.92% to approximately 6.37% in winter and from 53.14% to approximately 48.50% in summer. These calculations demonstrate that the policy choice is material; they are not replacement Published Results because the existing Runs have not passed the new validity and provenance requirements.

## Approaches

### A. Named scorecard with Inventory-Adjusted Cost primary — recommended

Publish all of the following from one Evaluation Policy:

1. Grid Cost: exact realized grid settlement.
2. Operating Cost: Grid Cost plus policy-declared asset throughput/wear terms, evaluated identically for every Controller.
3. Inventory-Adjusted Cost: Operating Cost plus the value of initial recoverable inventory minus Terminal State recoverable inventory, using one Scenario-derived settlement price for every compared Run.
4. Comfort Violation: a separate physical measure, never converted to money by a Solver Objective weight.
5. Solver Objective diagnostics: reported only as Controller diagnostics, with terminal value, anti-cycling, smoothing, and slack terms kept outside Operating Cost.

The current throughput/wear coefficients may be retained initially only as explicit provisional Evaluation Policy inputs, not as silently authoritative equipment economics. Their values and provenance must be stored in every Run Bundle. This approach reconciles what the Controller optimizes with what the repository reports, prevents inventory depletion from looking like a saving, and still exposes the simpler Grid Cost.

Trade-off: it gives provisional coefficients visible influence over the primary metric, so publication must label them and include sensitivity evidence until independently justified.

### B. Evidence-conservative inventory-adjusted grid settlement

Make inventory-adjusted Grid Cost the primary comparison, classify all existing throughput/wear weights as Solver Objective regularization, and publish asset throughput separately rather than monetizing it.

This avoids implying unsupported degradation economics and stays closest to the current reports. The trade-off is that a Controller can optimize materially different asset use without that use appearing in the headline economic metric; Operating Cost would remain incomplete until a justified wear policy is supplied.

### C. No primary metric

Publish Grid Cost, provisional Operating Cost, Inventory-Adjusted Cost, asset throughput, and Comfort Violation as a scorecard without one headline saving.

This is the most cautious presentation, but it weakens comparability and leaves every downstream publication to decide which quantity leads. That reduces Locality and reintroduces interpretation drift outside the evaluation/artifact Module.

## Recommendation

Adopt Approach A. It gives the evaluation/artifact Module one deep Interface, applies identical economics to both Controller Adapters, and keeps Solver Objective tuning separate from realized evaluation. Treat the current throughput/wear coefficients as provisional policy inputs rather than validated equipment economics, and never monetize Comfort Violation without an independently approved valuation.

Approach A was adopted. ADR-0006 records the decision; the verification matrix now treats it as authoritative.
