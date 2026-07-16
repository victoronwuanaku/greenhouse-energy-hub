---
status: accepted
---

# Publish only provenance-backed run bundles

Published Results will be generated only from immutable Run Bundles that identify their Scenario, controller configuration, solver outcome, code and input provenance, trajectories, evaluation policy, and validation evidence. Mutable canonical CSVs may remain as convenience views, but they are not authoritative scientific artifacts.

## Consequences

The notebook, figures, summaries, and README claims must select explicit valid Run Bundles and be reproducible from them. Regeneration happens only after behavioral and structural migration is complete and verified.
