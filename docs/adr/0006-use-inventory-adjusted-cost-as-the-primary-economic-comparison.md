---
status: accepted
---

# Use inventory-adjusted cost as the primary economic comparison

Every controller comparison uses one named and versioned Evaluation Policy that publishes Grid Cost, Operating Cost, Inventory-Adjusted Cost, and Comfort Violation separately. Operating Cost is Grid Cost plus policy-declared asset throughput and wear terms evaluated identically for every Controller. Inventory-Adjusted Cost additionally settles the difference between initial and Terminal State recoverable inventory using one Scenario-derived settlement rule, and it is the primary economic comparison.

Terminal value, anti-cycling, input smoothing, and comfort slack remain Solver Objective diagnostics rather than realized Run economics. Comfort Violation remains a physical measure unless an independently justified Evaluation Policy explicitly values it; a Solver Objective weight is not such a valuation.

## Consequences

Every Run Bundle records the Evaluation Policy version, settlement rule, coefficients, line items, and sensitivity evidence required to reproduce its totals. The existing throughput and wear coefficients are provisional policy inputs, not validated equipment economics, and publication must label them accordingly. Step-level values must sum exactly to each Run-level quantity, all compared Controller Adapters use the same policy, and the previous internally weighted "effective cost" is not publishable.
