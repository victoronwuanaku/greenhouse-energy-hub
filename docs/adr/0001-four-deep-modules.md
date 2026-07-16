---
status: accepted
---

# Organize the repository around four deep modules

The repository will concentrate its behavior in four deep Modules: greenhouse-hub physics, validated simulation, scenario data, and experiment evaluation/artifacts. This shape is preferred over adding technical layers because it gives callers Leverage while keeping physical knowledge, run validity, temporal semantics, and publication rules local to their owning Module.

## Consequences

The eventual `src/greenhouse_energy_hub/` package will reflect these Modules, while experiment and acquisition scripts remain thin callers. File moves alone do not satisfy this decision; each Module must own its declared invariants behind a small Interface.
