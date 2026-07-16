---
status: accepted
---

# Keep control decisions causal and terminal states valid

An MPC Controller may use scenario information only from its declared Forecast Horizon, and every Hub State reached by a planned operating step—including the horizon Terminal State—must satisfy the applicable invariants. Horizon length may change foresight, but it may not silently change which physical constraints apply.

## Consequences

Terminal valuation must be derived from horizon-local information, and tests must prove that data beyond the Forecast Horizon cannot change the current action. One-step and longer-horizon controllers must cross the same physical test surface.
