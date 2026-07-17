# Finding Verification Matrix

This matrix is the completion ledger for the architecture migration. A finding is not resolved by code movement or by a narrow passing test; its completion evidence must prove the required behavior across the scope of the original claim.

## Status language

- **Confirmed** — current source or an authoritative reproduction demonstrates the defect.
- **Unproven** — the current evidence does not establish the required outcome.
- **Resolved** — the required behavior and full-scope verification evidence both exist.

No finding in this document is **Resolved** at the start of the migration.

The **Current evidence** paragraphs below are retained as historical, pre-migration
descriptions. Paths in those paragraphs may therefore name the former repository
layout; all completion evidence names the current committed `src/` package and
publication artifacts.

## Priority findings

### PF-01 — One-step MPC produces invalid reached states

**Status:** Resolved

**Current evidence:** `experiments/ablations.py` selects `n_horizon=1`, while `control/mpc_controller.py` applies state and comfort constraints to stage states without an explicit constraint on the reached horizon Terminal State. The review reproduction observed storage and temperature outside physical bounds and subsequent infeasible solves.

**Required outcome:** A valid one-operating-step Forecast Horizon changes foresight only; every reached state remains subject to the same physical and operational policy as longer horizons.

**Owning Modules:** Greenhouse-hub physics; validated simulation

**Decision:** ADR-0003

**Required regression evidence:**

- A one-step controller test that inspects its reached battery, hydrogen, thermal-store, and temperature state.
- A multi-step one-hour-horizon Run that remains physically valid at every operating step.
- A parity test showing that horizon length does not change successor-state bounds.

**Completion evidence:**

- **Implementation evidence:** Reached-state bounds and shared successor physics are owned by `src/greenhouse_energy_hub/hub.py`; `src/greenhouse_energy_hub/controllers/mpc.py` applies operational bounds to every planned successor, including the Terminal State; `src/greenhouse_energy_hub/simulation.py` validates every applied control, flow set, and reached state before appending an Operating Record.
- **Regression evidence:** `tests/test_simulation.py::test_one_step_mpc_keeps_every_reached_state_valid` and `tests/test_simulation.py::test_mpc_enables_operational_terminal_bounds`, together with `tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`; their combined command `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_simulation.py::test_one_step_mpc_keeps_every_reached_state_valid tests/test_simulation.py::test_mpc_enables_operational_terminal_bounds tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence` observed **3 passed**. The explicit policy-parity test is `tests/test_simulation.py::test_horizon_length_does_not_change_successor_state_bounds`; command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_simulation.py::test_horizon_length_does_not_change_successor_state_bounds` observed **1 passed in 15.17s** after comparing every one-step and 24-step Terminal State lower/upper bound against the same `operational_state_bounds` policy.
- **Publication/full-scope evidence:** The verified 336-step one-step Run Bundle is `df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac`; the verified 336-step, 24-step-foresight ablation control is `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`. `tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence` reconstructs all 336 records in each bundle and checks 336/336 successful solver decisions plus 336/336 valid control/flow/successor/physical-invariant evidence rows. Final repository command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`; observed: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0003-causal-horizon-and-terminal-validity.md`.

### PF-02 — Disabled TES and failed solves are published as valid results

**Status:** Resolved

**Current evidence:** `control/mpc_controller.py` pins TES flows to zero while retaining a positive lower state bound and hourly standing loss. `control/rolling_horizon.py` applies `mpc.u0` without checking `mpc.solver_stats`, and `experiments/ablations.py` writes aggregate artifacts without validating the Run.

**Required outcome:** A Disabled Asset remains in the shared schema with zero capacity, zero initial and admissible state, zero associated controls, no nonzero dynamic contribution, and no inventory value. Every solver failure stops before control application, and no failed Run enters publication.

**Owning Modules:** Validated simulation; greenhouse-hub physics; evaluation/artifact

**Decision:** ADR-0002 and ADR-0005

**Required regression evidence:**

- A forced solver-failure test proving that no control is applied and no Run Bundle is created.
- Configuration and Adapter-parity tests proving that a Disabled Asset remains identically zero and contributes no inventory value.
- Valid no-TES and no-H2 Runs covering the complete published scenario window.
- Manifest and artifact tests proving that every published ablation has successful per-step solver evidence and physical invariant evidence.

**Completion evidence:**

- **Implementation evidence:** Disabled-Asset zero-capacity configuration, admissible states, controls, dynamics, and validation are centralized in `src/greenhouse_energy_hub/hub.py`; fail-closed control application and invalid-run handling are in `src/greenhouse_energy_hub/simulation.py`; bundle eligibility, validation, and zero inventory valuation are enforced by `src/greenhouse_energy_hub/evaluation.py`.
- **Regression evidence:** `tests/test_simulation.py::test_solver_failure_returns_invalid_run_without_advancing_plant`, `tests/test_evaluation.py::test_invalid_run_writes_separate_non_bundle_diagnostics`, `tests/test_hub.py::test_shared_hub_expressions_match_numerical_adapter`, `tests/test_simulation.py::test_disabled_asset_configuration_is_exact_zero_capacity`, `tests/test_simulation.py::test_disabled_asset_dynamics_flows_and_validation_are_inert`, `tests/test_simulation.py::test_disabled_asset_recoverable_inventory_contribution_is_zero`, `tests/test_simulation.py::test_two_day_winter_mpc_keeps_disabled_assets_exactly_zero`, and `tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_simulation.py::test_solver_failure_returns_invalid_run_without_advancing_plant tests/test_evaluation.py::test_invalid_run_writes_separate_non_bundle_diagnostics tests/test_hub.py::test_shared_hub_expressions_match_numerical_adapter tests/test_simulation.py::test_disabled_asset_configuration_is_exact_zero_capacity tests/test_simulation.py::test_disabled_asset_dynamics_flows_and_validation_are_inert tests/test_simulation.py::test_disabled_asset_recoverable_inventory_contribution_is_zero tests/test_simulation.py::test_two_day_winter_mpc_keeps_disabled_assets_exactly_zero tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`; observed: **59 passed** across all parameterizations, including disabled-H2 and disabled-TES numerical/symbolic parity.
- **Publication/full-scope evidence:** The valid no-H2 bundle is `9b36719b929fffcef26084f5efa37bcb2385b9b9199c7e429dec37a86993ad97`; the valid no-TES bundle is `787abda250b5b2f47005452322aa8d65ee5c7bcb8848246797e4e98ea8e1b8de`. Public verification reconstructed 336/336 records, found 336/336 successful MPC solver decisions, and recomputed 336/336 valid physical-evidence rows and each summary for both bundles. `tests/test_published_artifacts.py::test_bundle_verification_rejects_a_failed_validation_report`, `tests/test_published_artifacts.py::test_bundle_verification_rejects_missing_solver_status`, and `tests/test_published_artifacts.py::test_publication_manifest_retains_each_resolvable_bundle_in_the_proposed_commit` are included in `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py`; observed: **28 passed, zero failures/skips/xfails/XPASS**. Final full-scope result: **307 passed, zero failures/skips/xfails/XPASS** from `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`.
- **Decision evidence:** `docs/adr/0002-valid-runs-fail-closed.md` and `docs/adr/0005-represent-disabled-assets-as-inert-zero-capacity-states.md`.

### PF-03 — The controller leaks information beyond its declared horizon

**Status:** Resolved

**Current evidence:** `control/rolling_horizon.py` passes the complete simulation arrays to `build_mpc`; `control/mpc_controller.py` calculates terminal prices once from those complete arrays. The review reproduction showed different first actions for forecasts that were identical inside the horizon and differed only beyond it.

**Required outcome:** Current control and terminal valuation depend only on the current Hub State, declared Controller configuration, and Scenario information inside the Forecast Horizon.

**Owning Modules:** MPC Controller Adapter; Scenario data; greenhouse-hub physics

**Decision:** ADR-0003

**Required regression evidence:**

- Two scenarios identical through the complete Forecast Horizon but different afterward produce equivalent current controls within a declared numerical tolerance.
- Terminal valuation inputs are observable in diagnostics and contain only horizon-local data.
- Final Operating Steps require explicit Forecast Coverage and never clamp, repeat, or synthesize missing values.

**Completion evidence:**

- **Implementation evidence:** `src/greenhouse_energy_hub/controllers/mpc.py` receives only the declared Forecast Horizon and derives terminal coefficients from controlled stage points; `src/greenhouse_energy_hub/scenarios.py` requires explicit Forecast Coverage; `src/greenhouse_energy_hub/simulation.py` rejects a wrong-length or incomplete forecast before solver use.
- **Regression evidence:** `tests/test_simulation.py::test_first_control_is_independent_of_out_of_horizon_prices`, `tests/test_simulation.py::test_terminal_coefficients_use_only_controlled_stage_points`, `tests/test_simulation.py::test_mpc_rejects_wrong_length_forecast_before_solver`, and `tests/test_simulation.py::test_missing_final_forecast_coverage_fails_before_run`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_simulation.py::test_first_control_is_independent_of_out_of_horizon_prices tests/test_simulation.py::test_terminal_coefficients_use_only_controlled_stage_points tests/test_simulation.py::test_mpc_rejects_wrong_length_forecast_before_solver tests/test_simulation.py::test_missing_final_forecast_coverage_fails_before_run`; observed: **4 passed**.
- **Publication/full-scope evidence:** `df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac` records horizon-local diagnostics for all 336 one-step decisions, while `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a` does so for all 336 24-step-foresight decisions; both are exercised by `tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`. Final full-scope command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0003-causal-horizon-and-terminal-validity.md`.

### PF-04 — Baseline fairness and causal attribution are overstated

**Status:** Resolved

**Current evidence:** The Baseline Controller never uses hydrogen, never charges TES, cannot charge the battery from cheap grid energy, and uses a fixed discharge threshold. Existing prose describes the comparison as fair and attributes savings to smarter dispatch, while the invalid ablations cannot currently isolate those causes.

**Required outcome:** The Baseline Controller is described by an explicit capability policy; comparisons distinguish temperature fairness from capability parity; causal claims are made only when supported by valid ablations.

**Owning Modules:** Controller Adapters; evaluation/artifact

**Decision:** ADR-0004 and ADR-0006

**Required verification evidence:**

- The Run Bundle records the Baseline Controller capability policy.
- Published tables and README language identify the exact comparison rather than using an unqualified fairness label.
- Every causal attribution links to a valid ablation Run Bundle.

**Completion evidence:**

- **Implementation evidence:** The exact limited Baseline capability policy is owned by `src/greenhouse_energy_hub/controllers/baseline.py`, serialized by `src/greenhouse_energy_hub/evaluation.py`, and stated without an unqualified fairness label in `README.md`; `experiments/publish_results.py` publishes only the pinned recipe and its valid ablation evidence.
- **Regression evidence:** `tests/test_simulation.py::test_baseline_run_records_exact_capability_policy`, `tests/test_published_artifacts.py::test_readme_names_the_baseline_as_limited_capability_not_unqualified_fair`, `tests/test_published_artifacts.py::test_every_baseline_run_manifest_contains_the_exact_capability_policy`, and `tests/test_published_artifacts.py::test_each_causal_claim_cites_its_corresponding_ablation_bundle`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_simulation.py::test_baseline_run_records_exact_capability_policy tests/test_published_artifacts.py::test_readme_names_the_baseline_as_limited_capability_not_unqualified_fair tests/test_published_artifacts.py::test_every_baseline_run_manifest_contains_the_exact_capability_policy tests/test_published_artifacts.py::test_each_causal_claim_cites_its_corresponding_ablation_bundle`; observed: **4 passed**.
- **Publication/full-scope evidence:** Baseline policy evidence is pinned by winter `509c81dd1bea108aac1be2073b2b16646771bf65d5cc7f257081bcf81784713a` and summer `8d9f6657c04b552f8d4f83f484ba004bcc46a2e7fa9c95cdc67936da9d8998e2`. Each causal statement names the valid full/no-H2/no-TES/one-step evidence: `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`, `9b36719b929fffcef26084f5efa37bcb2385b9b9199c7e429dec37a86993ad97`, `787abda250b5b2f47005452322aa8d65ee5c7bcb8848246797e4e98ea8e1b8de`, and `df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac`. All six are covered by the public verifier and summary recomputation. Publication command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py`; observed: **28 passed, zero failures/skips/xfails/XPASS**. Full command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`; observed: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0004-publish-provenance-backed-run-bundles.md` and `docs/adr/0006-use-inventory-adjusted-cost-as-the-primary-economic-comparison.md`.

### PF-05 — Optimized and reported economic quantities do not match

**Status:** Resolved

**Current evidence:** The MPC objective includes battery, TES, electrolyser, and fuel-cell wear terms, while `accounting.py` reports Grid Cost and inventory settlement only. The ablation script monetizes Comfort Violation with the solver slack weight without an external valuation policy.

**Required outcome:** Every economic quantity has one explicit definition; objective regularization is reconciled with reported Operating Cost; Comfort Violation remains physical unless an approved valuation policy applies.

**Owning Module:** Evaluation/artifact

**Decision:** ADR-0006

**Required regression evidence:**

- Step-level terms sum exactly to each named Run-level economic quantity.
- Changing a solver-only regularization term cannot silently change the definition of a Published Result.
- Comfort valuation is absent unless a named evaluation policy supplies it.
- Baseline and MPC comparisons use the same evaluation policy.
- Nominal, zero-times, and two-times provisional wear sensitivities are reproducible from the same Run trajectories.

**Completion evidence:**

- **Implementation evidence:** `src/greenhouse_energy_hub/evaluation.py` owns the named Evaluation Policy, exact step line items, Grid/Operating/Inventory-Adjusted Cost definitions, recoverable-inventory settlement, physical Comfort Violation, and 0x/1x/2x provisional-wear sensitivities; solver-only objective terms remain in `src/greenhouse_energy_hub/controllers/mpc.py` and do not define Published Results.
- **Regression evidence:** `tests/test_evaluation.py::test_two_step_grid_and_each_wear_term_use_exact_policy_formulas`, `tests/test_evaluation.py::test_two_step_operating_cost_is_grid_plus_four_wear_terms`, `tests/test_evaluation.py::test_two_step_comfort_uses_reached_state_and_is_never_monetized`, `tests/test_evaluation.py::test_wear_sensitivities_reuse_records_at_zero_one_and_two_times`, `tests/test_evaluation.py::test_baseline_and_mpc_use_same_policy_without_changing_baseline_capabilities`, and `tests/test_evaluation.py::test_solver_only_configuration_changes_do_not_change_evaluation`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_evaluation.py::test_two_step_grid_and_each_wear_term_use_exact_policy_formulas tests/test_evaluation.py::test_two_step_operating_cost_is_grid_plus_four_wear_terms tests/test_evaluation.py::test_two_step_comfort_uses_reached_state_and_is_never_monetized tests/test_evaluation.py::test_wear_sensitivities_reuse_records_at_zero_one_and_two_times tests/test_evaluation.py::test_baseline_and_mpc_use_same_policy_without_changing_baseline_capabilities tests/test_evaluation.py::test_solver_only_configuration_changes_do_not_change_evaluation`; observed: **6 passed**.
- **Publication/full-scope evidence:** `tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary` is the committed audit of all eight unique manifest bundles, 2,688 step line items, and 24 sensitivity summaries. It verifies every full ID and committed member; re-sums every nominal economic field; proves every step/Run Operating Cost and Inventory-Adjusted Cost identity; reproduces 0x/1x/2x policy sensitivities, two comparison rows, and four ablation rows. Exact command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary`; observed: **1 passed in 5.82s**. Verified IDs: `509c81dd1bea108aac1be2073b2b16646771bf65d5cc7f257081bcf81784713a`, `1e9ee84ca78896c082b6d3216bab0d841f027b65f901aeec76bc78f253a90253`, `8d9f6657c04b552f8d4f83f484ba004bcc46a2e7fa9c95cdc67936da9d8998e2`, `5975dafc40c7bb296fa8861eb1e78b2afc21ff4eeff14ed3e2575b77768cecb1`, `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`, `9b36719b929fffcef26084f5efa37bcb2385b9b9199c7e429dec37a86993ad97`, `787abda250b5b2f47005452322aa8d65ee5c7bcb8848246797e4e98ea8e1b8de`, and `df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac`. Publication command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py`; observed: **28 passed, zero failures/skips/xfails/XPASS**. Full command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`; observed: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0006-use-inventory-adjusted-cost-as-the-primary-economic-comparison.md`.

### PF-06 — Passing tests do not validate all published experiments

**Status:** Resolved

**Current evidence:** `tests/test_hub.py` validates canonical CSVs and selected summary columns but does not require per-step solver success, ablation trajectories, complete summary fields, code/input provenance, or a valid manifest for every Published Result.

**Required outcome:** Every Published Result is reconstructable from Valid Run Bundles whose complete solver, physical, temporal, evaluation, and provenance evidence is checked automatically.

**Owning Modules:** Validated simulation; evaluation/artifact

**Decision:** ADR-0002, ADR-0004, ADR-0006, and ADR-0007

**Required regression evidence:**

- Publication rejects a missing, invalid, or failed Run manifest.
- Publication verifies and pins the full Run Bundle Identifier rather than trusting a filename, alias, or digest prefix.
- Every summary field is recomputed from trajectories and evaluation policy.
- Every figure and README metric identifies its source Run Bundles.
- Tests enumerate every published bundle rather than a mutable canonical subset.

**Completion evidence:**

- **Implementation evidence:** `src/greenhouse_energy_hub/evaluation.py` verifies full identifiers, provenance, every bundle member, solver/physical/temporal evidence, exact recomputed summary bytes, manifest schema, and committed retention; `experiments/publish_results.py`, `notebooks/results_analysis.ipynb`, and `README.md` consume the pinned `results/publication_manifest.json` rather than mutable latest-run files.
- **Regression evidence:** `tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`, `tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary`, `tests/test_published_artifacts.py::test_publisher_builds_the_exact_verified_full_id_recipe`, `tests/test_published_artifacts.py::test_publication_manifest_retains_each_resolvable_bundle_in_the_proposed_commit`, `tests/test_published_artifacts.py::test_bundle_verification_rejects_a_failed_validation_report`, `tests/test_published_artifacts.py::test_bundle_verification_rejects_missing_solver_status`, and `tests/test_published_artifacts.py::test_readme_rewrite_preserves_bytes_outside_generated_markers`. Command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py`; observed: **28 passed, zero failures/skips/xfails/XPASS**.
- **Publication/full-scope evidence:** The manifest has eight unique full identifiers: `509c81dd1bea108aac1be2073b2b16646771bf65d5cc7f257081bcf81784713a`, `1e9ee84ca78896c082b6d3216bab0d841f027b65f901aeec76bc78f253a90253`, `8d9f6657c04b552f8d4f83f484ba004bcc46a2e7fa9c95cdc67936da9d8998e2`, `5975dafc40c7bb296fa8861eb1e78b2afc21ff4eeff14ed3e2575b77768cecb1`, `9c0b37e82655b6ccef9e1b1d831ee729d37eb4afc289a78045bab85fbe287e4a`, `9b36719b929fffcef26084f5efa37bcb2385b9b9199c7e429dec37a86993ad97`, `787abda250b5b2f47005452322aa8d65ee5c7bcb8848246797e4e98ea8e1b8de`, and `df487e8da6980b2d87c7629ee276d92952630f0feebfb6962d2b291838edc5ac`. `tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary` resolves and publicly verifies every full ID, verifies all five members per selected bundle as committed, reconstructs all 2,688 trajectory/diagnostic rows, recomputes all eight summaries, and reproduces two comparison rows and four ablation rows. Exact node command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary`; observed: **1 passed in 5.82s**. All nine retained bundle directories also passed final public verification with all 45 members committed. Final command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`; observed: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0002-valid-runs-fail-closed.md`, `docs/adr/0004-publish-provenance-backed-run-bundles.md`, `docs/adr/0006-use-inventory-adjusted-cost-as-the-primary-economic-comparison.md`, and `docs/adr/0007-use-two-level-content-identity-for-run-bundles.md`.

### PF-07 — Scenario time semantics and documented reproduction are inconsistent

**Status:** Resolved

**Current evidence:** `load_data` unconditionally drops the last selected price row, silently drops missing aligned rows, and does not enforce requested length. The lighting schedule interprets UTC timestamp hours as operating-clock hours. Canonical output files are overwritten, while the notebook reads whichever canonical files were written most recently.

**Required outcome:** A Scenario has exact validated coverage, explicit source and operating timezones, deterministic alignment, and provenance; publication selects explicit Run Bundles rather than mutable latest-run files.

**Owning Modules:** Scenario data; evaluation/artifact

**Decision:** ADR-0004, ADR-0007, and the architecture design's Scenario temporal policy

**Required regression evidence:**

- Exact-length tests for winter, summer, December year-end, and cross-month windows.
- Tests for UTC source timestamps, Europe/Amsterdam operating-clock schedules, and both daylight-saving transitions.
- Missing and duplicate input hours fail with diagnostic evidence instead of being dropped.
- Insufficient Forecast Coverage fails before a Run starts and forecast-only rows never enter evaluation totals.
- Notebook and figure generation consumes explicit Run Bundle identifiers.

**Completion evidence:**

- **Implementation evidence:** Exact UTC source coverage, Europe/Amsterdam operating-clock semantics, DST-aware windows, strict alignment, and provenance are owned by `src/greenhouse_energy_hub/scenarios.py`; `src/greenhouse_energy_hub/simulation.py` separates Forecast Coverage from the Operating Window; `src/greenhouse_energy_hub/evaluation.py`, `experiments/publish_results.py`, and `notebooks/results_analysis.ipynb` load explicit manifest-selected bundles.
- **Regression evidence:** `tests/test_scenarios.py::test_winter_and_summer_fourteen_day_windows_are_exact`, `tests/test_scenarios.py::test_december_window_with_available_forecast_coverage_is_complete`, `tests/test_scenarios.py::test_cross_month_calendar_window_is_complete`, `tests/test_scenarios.py::test_dst_local_calendar_day_has_exact_operating_steps`, `tests/test_scenarios.py::test_lighting_turns_on_at_six_local_in_winter_and_summer`, `tests/test_scenarios.py::test_invalid_source_timestamps_raise_scenario_validation_error`, `tests/test_simulation.py::test_missing_final_forecast_coverage_fails_before_run`, and `tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_scenarios.py tests/test_simulation.py::test_missing_final_forecast_coverage_fails_before_run tests/test_published_artifacts.py::test_generated_bundle_full_scope_evidence`; observed: **44 passed** across all scenario parameterizations.
- **Publication/full-scope evidence:** Winter bundles `509c81dd1bea108aac1be2073b2b16646771bf65d5cc7f257081bcf81784713a` and `1e9ee84ca78896c082b6d3216bab0d841f027b65f901aeec76bc78f253a90253` each verify an exact 336-step `2023-01-01T23:00:00Z`–`2023-01-15T23:00:00Z` Operating Window with Forecast Coverage through `2023-01-16T23:00:00Z`. Summer bundles `8d9f6657c04b552f8d4f83f484ba004bcc46a2e7fa9c95cdc67936da9d8998e2` and `5975dafc40c7bb296fa8861eb1e78b2afc21ff4eeff14ed3e2575b77768cecb1` verify `2023-05-31T22:00:00Z`–`2023-06-14T22:00:00Z` with Forecast Coverage through `2023-06-15T22:00:00Z`. The four ablation IDs in `results/publication_manifest.json` are also resolved and verified by `tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary` before notebook/figure publication. Publication command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra tests/test_published_artifacts.py`; observed: **28 passed, zero failures/skips/xfails/XPASS**. Full command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`; observed: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0004-publish-provenance-backed-run-bundles.md`, `docs/adr/0007-use-two-level-content-identity-for-run-bundles.md`, and the temporal policy implemented by `src/greenhouse_energy_hub/scenarios.py`.

## Architecture requirements

### AR-01 — Numerical and symbolic physics have one owner

**Status:** Resolved

**Current evidence:** `models/hub_model.py` and `control/mpc_controller.py` contain parallel Implementations of battery, hydrogen, TES, temperature, heat-flow, and grid-balance equations.

**Required outcome:** Physical knowledge has Locality in the greenhouse-hub physics Module, with numerical and CasADi Adapters crossing one physical test surface.

**Decision:** ADR-0001

**Required verification evidence:** Property or parameterized parity tests over representative states, controls, exogenous inputs, enabled/disabled Asset configurations, and edge bounds.

**Completion evidence:**

- **Implementation evidence:** `src/greenhouse_energy_hub/hub.py` is the single owner of numerical and symbolic state transitions, flows, bounds, and validation; `src/greenhouse_energy_hub/controllers/mpc.py` consumes that symbolic surface, and `src/greenhouse_energy_hub/simulation.py` consumes the numerical surface.
- **Regression evidence:** `tests/test_hub.py::test_shared_hub_expressions_match_numerical_adapter`, `tests/test_hub.py::test_shared_thermal_charge_margin_is_signed_and_validated_numerically`, and `tests/test_hub.py::test_legacy_wrapper_delegates_physics_and_conversions_to_shared_owners`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_hub.py::test_shared_hub_expressions_match_numerical_adapter tests/test_hub.py::test_shared_thermal_charge_margin_is_signed_and_validated_numerically tests/test_hub.py::test_legacy_wrapper_delegates_physics_and_conversions_to_shared_owners`; observed: **47 passed** across nominal, every physical/operational state edge, every control edge, eight random interior cases, and disabled-H2/disabled-TES configurations.
- **Publication/full-scope evidence:** No Run Bundle ID is required for this structural ownership finding. `tests/test_published_artifacts.py::test_publication_manifest_recomputes_every_pinned_summary` publicly verifies all eight pinned bundles, whose verifier replays the shared numerical physics over all 2,688 records and matches stored validation bytes. Final command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra`: **307 passed, zero failures/skips/xfails/XPASS**.
- **Decision evidence:** `docs/adr/0001-four-deep-modules.md`.

### AR-02 — Module ownership is reflected by stable packaging

**Status:** Resolved

**Current evidence:** Source code is divided between root modules and technical `models/` and `control/` packages; executable scripts mutate `sys.path`; tests run primarily from the repository root.

**Required outcome:** The four approved Modules live in an installable `src/greenhouse_energy_hub/` package, scripts are thin callers, and imports work without repository-root path mutation.

**Decision:** ADR-0001

**Required verification evidence:** Build/install tests, import tests from outside the repository root, absence of runtime path mutation, and the full test suite against the installed package.

**Completion evidence:**

- **Implementation evidence:** The installable package is rooted at `src/greenhouse_energy_hub/` with the four owning Modules in `hub.py`, `simulation.py`, `scenarios.py`, and `evaluation.py`; Controller Adapters live in `src/greenhouse_energy_hub/controllers/`; `experiments/` and `scripts/` are thin callers; packaging metadata is in `pyproject.toml`.
- **Regression evidence:** `tests/test_simulation.py::test_boundary_guard_rejects_aliased_paths_and_constant_dynamic_imports`, `tests/test_simulation.py::test_project_sources_use_only_installed_package_imports`, and `tests/test_simulation.py::test_editable_install_imports_all_owning_modules_outside_repository`. Command: `/tmp/geh-architecture-env/bin/python -m pytest -q tests/test_simulation.py::test_boundary_guard_rejects_aliased_paths_and_constant_dynamic_imports tests/test_simulation.py::test_project_sources_use_only_installed_package_imports tests/test_simulation.py::test_editable_install_imports_all_owning_modules_outside_repository`; observed: **7 passed** across the five guarded mutation/import forms plus the project-source and outside-root editable-install tests.
- **Publication/full-scope evidence:** No Run Bundle ID is required for this packaging finding. Every retained bundle's executable map names installed `src/greenhouse_energy_hub/` modules plus only the applicable thin `experiments/` caller, and the final full-suite command `/tmp/geh-architecture-env/bin/python -m pytest -o addopts='' -q -ra` ran against `/tmp/geh-architecture-env`: **307 passed, zero failures/skips/xfails/XPASS**. The required source audit `rg -n "sys\\.path|fair rule-based|effective cost|baseline_results\\.csv|mpc_results\\.csv|results/scenarios" README.md src experiments scripts tests notebooks` produced no matches after the runtime-equivalent negative assertions were made audit-compatible in commit `a89f29e`.
- **Decision evidence:** `docs/adr/0001-four-deep-modules.md`.

## Final audit rule

At goal completion, every **Completion evidence** entry above must name an authoritative file, test, command result, and—where applicable—Run Bundle. Evidence that covers only a short run, only canonical CSVs, or only one Controller cannot prove a repository-wide or publication-wide requirement.
