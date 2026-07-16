---
status: accepted
---

# Use two-level content identity for run bundles

Each requested experiment receives a Run Specification Identifier: a SHA-256 digest over a versioned canonical representation of its Scenario, Controller configuration, Asset capabilities, Evaluation Policy, code revision, exact executable source-tree content, runtime/dependency manifest, and input hashes. Each completed Run Bundle receives a separate Run Bundle Identifier derived from its canonical manifest and the SHA-256 hashes of every trajectory, summary, diagnostic, and validation member. This distinction identifies both what was requested and the exact evidence produced, including divergent executions of one specification.

## Consequences

Run Bundles use human-readable Scenario and Controller prefixes for navigation, but only full identifiers are authoritative. Valid bundles are written atomically, are never overwritten, deduplicate only after full-content verification, and retain distinct outputs for the same Run Specification. Bundles referenced by Published Results are permanent; unreferenced bundles may be removed only after proving that no publication references them. Failed or incomplete diagnostics remain outside `results/runs/`, and aliases such as `latest` are non-authoritative. The Implementation remains filesystem-based at the repository's current scale.
