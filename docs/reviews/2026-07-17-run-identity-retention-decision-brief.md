# Run Identity and Retention Decision Brief

**Review status:** Approved — Approach A, recorded by ADR-0007

## Decision

Choose how immutable Run Bundles are identified, deduplicated, retained, and selected for publication. Identity must distinguish the requested experiment inputs from the actual output produced, detect tampering, prevent accidental overwrite, and remain usable by people inspecting `results/runs/`.

## Current evidence

- `control/rolling_horizon.py` writes mutable `results/{baseline,mpc}_results.csv` files and replaces scenario files under names such as `winter_m01_14d_mpc.csv`.
- `summary.csv` replaces rows by a human-readable scenario tag, so configuration, code, input, solver, or Evaluation Policy changes can silently reuse the same identity.
- The notebook and README select filenames rather than verified Run Bundle manifests.
- The repository has no manifest, canonical serialization rule, content digest, collision policy, atomic-write rule, or separation between failed diagnostics and authoritative Runs.
- The complete current `results/` tree is approximately 1.2 MiB, and its four scenario trajectories total approximately 420 KiB. Append-only filesystem bundles are therefore proportionate; an external registry or artifact store is not currently justified.

A deterministic identifier derived only from requested inputs would improve the current state but would still conflate the request with its output. The same code, inputs, configuration, and environment can produce a different trajectory because of solver, platform, dependency, or nondeterministic behavior; those outputs must not overwrite one another under one identifier.

## Approaches

### A. Two-level content identity — recommended

Use two deterministic identifiers:

1. A Run Specification Identifier hashes a canonical description of the manifest schema version, Scenario manifest, Controller Adapter and configuration, Asset capabilities, Evaluation Policy, code revision, exact executable source-tree digest, dependency/runtime manifest, and all input hashes. A commit hash alone is insufficient because a dirty worktree can execute different source under the same revision.
2. A Run Bundle Identifier hashes the canonical completed manifest payload plus the hashes of trajectories, summary, per-step solver diagnostics, and validation evidence. The Run Bundle Identifier field itself is excluded from the hashed payload to avoid circularity.

Both identifiers use SHA-256 over a versioned canonical UTF-8 JSON encoding; output-member hashes use SHA-256 over their exact bytes. The manifest records the algorithm and canonicalization version so identity rules can evolve without reinterpretation.

Store a Valid Run Bundle under a human-readable path such as:

```text
results/runs/<scenario>--<controller>--<bundle-digest-prefix>/
```

The full Run Specification Identifier and Run Bundle Identifier remain in the manifest; the readable prefix is only navigation. A digest-prefix collision extends the directory prefix rather than overwriting either bundle. Valid bundles are append-only and never overwritten. Byte-identical bundles deduplicate after full-digest verification. Distinct bundles with the same Run Specification Identifier are both retained and explicitly reveal execution variance. Published Results pin full Run Bundle Identifiers.

Failed or incomplete execution diagnostics live outside `results/runs/` under timestamped attempt paths and can never satisfy publication lookup. Mutable aliases such as `latest` may exist only as non-authoritative convenience views that resolve to a full bundle identifier.

This approach gives the evaluation/artifact Module Locality for identity, integrity, retention, and publication selection while keeping a readable filesystem.

Trade-off: it requires canonical serialization and two related identifiers rather than one filename.

### B. Single input-derived identifier

Hash the Scenario, Controller configuration, Evaluation Policy, code revision, dependencies, and inputs, then store one bundle at that path.

This is simpler and makes repeated specifications easy to find. The trade-off is serious: a second execution with different output must either overwrite the first, fail, or introduce an ad hoc suffix. The identifier proves which experiment was requested but not which bytes were actually published.

### C. Timestamp or UUID identity with stored hashes

Give every execution a timestamp or random UUID and record input and output hashes in its manifest.

This preserves every attempt and is operationally simple. However, equivalent bundles do not share identity, deduplication requires a separate index, reproducibility checks cannot compare identifiers directly, and chronological names encourage consumers to select the newest output rather than an explicitly verified one.

## Retention policy under Approach A

- Every Run Bundle referenced by a Published Result is retained permanently and immutably.
- Every distinct Valid Run Bundle remains append-only unless an explicit garbage-collection operation proves that no Published Result, manifest, summary, figure, or README claim references it.
- Identical bundles may be deduplicated only after full digest and member-hash verification.
- Failed and incomplete diagnostics are never stored as Run Bundles; their local retention is operational policy and cannot affect publication.
- Convenience views and indexes are regenerable and non-authoritative.

## Required verification

- Reordering equivalent manifest keys produces the same Run Specification Identifier and Run Bundle Identifier.
- Changing any Scenario input, Controller configuration, Asset capability, Evaluation Policy, code revision, or dependency manifest changes the Run Specification Identifier.
- Changing executable source content without changing the Git revision changes the executable source-tree digest and Run Specification Identifier.
- Changing any trajectory, summary, diagnostic, or validation member changes the Run Bundle Identifier and fails integrity verification.
- Re-running identical content deduplicates without overwrite; different output for one specification remains separately addressable.
- Publication rejects aliases, failed attempts, truncated digests, missing members, and unverified manifests as authoritative identities.
- Bundle creation is atomic: interruption cannot leave a directory that is discoverable as a Valid Run Bundle.

## Recommendation

Adopt Approach A. It separates reproducibility of the requested experiment from integrity of the produced evidence, preserves divergent executions instead of hiding them, and keeps publication pinned to exact immutable bytes.

Keep the Implementation filesystem-based for this repository's present scale. No database, remote object store, signing infrastructure, or general artifact registry is required.

The user approved Approach A. ADR-0007 records the decision, `CONTEXT.md` defines its domain terms, and the architecture design now treats it as authoritative.
