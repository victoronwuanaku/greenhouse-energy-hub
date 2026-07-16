---
status: accepted
---

# Treat solver or invariant failure as an invalid run

A Run is valid only when every controller solve succeeds and every required physical, temporal, and schema invariant is satisfied. Simulation execution must fail closed: unsuccessful controls may not be applied, and invalid or incomplete runs may not produce publishable Run Bundles.

## Consequences

Run validity belongs to the validated simulation Module rather than individual scripts, notebooks, or artifact tests. Failure diagnostics may be retained for debugging, but they must be unmistakably separate from published results.
