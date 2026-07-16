---
status: accepted
---

# Represent disabled assets as inert zero-capacity states

A Disabled Asset retains its position in the configured Hub State and control schemas, but its capacity, initial state, admissible state, and associated controls are all exactly zero. Its dynamics may not introduce a nonzero successor through standing loss, inflow, or outflow. It is excluded from initial and terminal inventory valuation, and Run provenance records that its capability is disabled.

## Consequences

Numerical and CasADi Adapters use the same state and control shapes for full and ablation Runs. Configuration and validation must reject any nonzero state, control, dynamic contribution, or inventory value for a Disabled Asset. This stable shape reduces variant complexity while preserving physical validity.
