"""
Rolling-horizon simulation loop for the greenhouse energy hub.

At each operating step the controller is asked for a decision from the current
Hub State and its forecast view; the decision is validated, the plant advances
through the shared hub physics, and the reached flows and state are validated
before being recorded. Any failure ends the Run as an explicit ``InvalidRun``;
only a fully validated rollout becomes a ``ValidRun``.

Heat is implicit: there is no prescribed heat-demand series. Both controllers
keep the greenhouse temperature inside the comfort band by supplying heat
(heat pump, e-boiler, fuel-cell heat, TES) and opening ventilation.

Controller implementations live in ``controllers/``; the command-line entry
point is ``experiments/run_scenario.py``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from typing import Protocol

import numpy as np

from greenhouse_energy_hub.hub import (
    ExogenousInputs, HubConfiguration, HubControl, HubFlows,
    HubState, HubStep, ValidationIssue, advance_hub,
    initial_state, normalize_control, validate_control, validate_flows,
    validate_successor, SOLVER_BOUND_TOLERANCE_KW,
)
from greenhouse_energy_hub.scenarios import (
    JSONValue,
    Scenario,
    ScenarioCoverageError,
    ScenarioPoint,
    freeze_json,
)


@dataclass(frozen=True)
class DecisionDiagnostics:
    adapter: str
    decision_status: str
    solver_success: bool | None
    solver_return_status: str | None
    solver_iterations: int | None
    solver_wall_seconds: float | None
    forecast_start_utc: datetime | None
    forecast_end_utc: datetime | None
    terminal_electric_value_eur_per_kwh: float | None
    terminal_heat_value_eur_per_kwhth: float | None


@dataclass(frozen=True)
class ControlDecision:
    control: HubControl
    diagnostics: DecisionDiagnostics


@dataclass(frozen=True)
class ControllerFailure:
    code: str
    message: str
    diagnostics: DecisionDiagnostics


class ControllerAdapter(Protocol):
    name: str
    configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    forecast_horizon_steps: int
    requires_operational_storage_bounds: bool

    def decide(
        self,
        state: HubState,
        forecast: tuple[ScenarioPoint, ...],
    ) -> ControlDecision | ControllerFailure:
        raise NotImplementedError


@dataclass(frozen=True)
class ValidationReport:
    complete: bool
    valid: bool
    checked_operating_steps: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class OperatingRecord:
    operating_step: int
    timestamp_utc: datetime
    start_state: HubState
    control: HubControl
    exogenous: ExogenousInputs
    reached_state: HubState
    flows: HubFlows


@dataclass(frozen=True)
class ValidRun:
    scenario: Scenario
    controller_name: str
    controller_configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    hub_configuration: HubConfiguration
    initial_state: HubState
    records: tuple[OperatingRecord, ...]
    controller_diagnostics: tuple[DecisionDiagnostics, ...]
    terminal_state: HubState
    validation: ValidationReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_configuration",
            freeze_json(self.controller_configuration),
        )
        object.__setattr__(
            self,
            "capability_policy",
            freeze_json(self.capability_policy),
        )


@dataclass(frozen=True)
class InvalidRun:
    scenario: Scenario
    controller_name: str
    controller_configuration: Mapping[str, JSONValue]
    capability_policy: Mapping[str, JSONValue]
    hub_configuration: HubConfiguration
    failed_step: int
    failure_code: str
    message: str
    partial_records: tuple[OperatingRecord, ...]
    controller_diagnostics: tuple[DecisionDiagnostics, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_configuration",
            freeze_json(self.controller_configuration),
        )
        object.__setattr__(
            self,
            "capability_policy",
            freeze_json(self.capability_policy),
        )


def _invalid_run(
    scenario: Scenario,
    controller: ControllerAdapter,
    hub_config: HubConfiguration,
    failed_step: int,
    failure_code: str,
    message: str,
    records: list[OperatingRecord],
    diagnostics: list[DecisionDiagnostics],
) -> InvalidRun:
    return InvalidRun(
        scenario=scenario,
        controller_name=controller.name,
        controller_configuration=dict(controller.configuration),
        capability_policy=dict(controller.capability_policy),
        hub_configuration=hub_config,
        failed_step=failed_step,
        failure_code=failure_code,
        message=message,
        partial_records=tuple(records),
        controller_diagnostics=tuple(diagnostics),
    )


def _issue_message(issues: tuple[ValidationIssue, ...]) -> str:
    return "; ".join(
        f"{issue.code} [{issue.field}]: {issue.message}" for issue in issues
    )


def _validate_decision_diagnostics(
    diagnostics: object,
    controller: ControllerAdapter,
    expected_status: str,
    failure_code: str | None = None,
) -> tuple[ValidationIssue, ...]:
    """Validate controller evidence without reaching back into controller internals."""
    if not isinstance(diagnostics, DecisionDiagnostics):
        return (
            ValidationIssue(
                code="schema_error",
                field="diagnostics",
                message="controller diagnostics must be DecisionDiagnostics",
            ),
        )

    issues: list[ValidationIssue] = []

    if not isinstance(diagnostics.adapter, str) or not diagnostics.adapter:
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="adapter",
                message="adapter must be a nonempty string",
            )
        )
    elif diagnostics.adapter != controller.name:
        issues.append(
            ValidationIssue(
                code="adapter_mismatch",
                field="adapter",
                message=(
                    f"diagnostics adapter {diagnostics.adapter!r} does not match "
                    f"controller {controller.name!r}"
                ),
            )
        )

    if diagnostics.decision_status != expected_status:
        issues.append(
            ValidationIssue(
                code="branch_mismatch",
                field="decision_status",
                message=(
                    f"{type(diagnostics).__name__} status must be "
                    f"{expected_status!r}"
                ),
            )
        )

    if diagnostics.solver_success is not None and type(
        diagnostics.solver_success
    ) is not bool:
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="solver_success",
                message="solver_success must be bool or None",
            )
        )

    if diagnostics.solver_return_status is not None and not isinstance(
        diagnostics.solver_return_status, str
    ):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="solver_return_status",
                message="solver_return_status must be str or None",
            )
        )

    iterations = diagnostics.solver_iterations
    if iterations is not None and (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations < 0
    ):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="solver_iterations",
                message="solver_iterations must be a nonnegative int or None",
            )
        )

    wall_seconds = diagnostics.solver_wall_seconds
    if wall_seconds is not None and (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, Real)
        or not np.isfinite(float(wall_seconds))
        or wall_seconds < 0
    ):
        issues.append(
            ValidationIssue(
                code="non_finite",
                field="solver_wall_seconds",
                message="solver_wall_seconds must be finite, nonnegative, or None",
                actual=(
                    float(wall_seconds)
                    if isinstance(wall_seconds, Real)
                    and not isinstance(wall_seconds, bool)
                    else None
                ),
            )
        )

    forecast_start = diagnostics.forecast_start_utc
    forecast_end = diagnostics.forecast_end_utc
    if forecast_start is not None and not isinstance(forecast_start, datetime):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="forecast_start_utc",
                message="forecast_start_utc must be datetime or None",
            )
        )
    if forecast_end is not None and not isinstance(forecast_end, datetime):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="forecast_end_utc",
                message="forecast_end_utc must be datetime or None",
            )
        )
    if (forecast_start is None) != (forecast_end is None):
        issues.append(
            ValidationIssue(
                code="schema_error",
                field="forecast_window_utc",
                message="forecast timestamps must both be present or both be None",
            )
        )
    elif isinstance(forecast_start, datetime) and isinstance(forecast_end, datetime):
        try:
            reversed_window = forecast_start > forecast_end
        except TypeError:
            reversed_window = True
        if reversed_window:
            issues.append(
                ValidationIssue(
                    code="chronology_error",
                    field="forecast_window_utc",
                    message="forecast_start_utc must not follow forecast_end_utc",
                )
            )

    for field_name in (
        "terminal_electric_value_eur_per_kwh",
        "terminal_heat_value_eur_per_kwhth",
    ):
        value = getattr(diagnostics, field_name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
        ):
            issues.append(
                ValidationIssue(
                    code="non_finite",
                    field=field_name,
                    message=f"{field_name} must be finite numeric or None",
                    actual=(
                        float(value)
                        if isinstance(value, Real) and not isinstance(value, bool)
                        else None
                    ),
                )
            )

    if expected_status == "success":
        if controller.name == "mpc":
            if diagnostics.solver_success is not True:
                issues.append(
                    ValidationIssue(
                        code="branch_mismatch",
                        field="solver_success",
                        message="successful MPC decision requires solver_success=True",
                    )
                )
            if not (
                isinstance(diagnostics.solver_return_status, str)
                and diagnostics.solver_return_status.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="schema_error",
                        field="solver_return_status",
                        message=(
                            "successful MPC decision requires a nonempty "
                            "solver_return_status"
                        ),
                    )
                )
        elif controller.name == "baseline":
            for field_name in (
                "solver_success",
                "solver_return_status",
                "solver_iterations",
                "solver_wall_seconds",
            ):
                if getattr(diagnostics, field_name) is not None:
                    issues.append(
                        ValidationIssue(
                            code="branch_mismatch",
                            field=field_name,
                            message=(
                                "successful Baseline decision requires empty "
                                "solver diagnostics"
                            ),
                        )
                    )
        elif diagnostics.solver_success is False:
            issues.append(
                ValidationIssue(
                    code="branch_mismatch",
                    field="solver_success",
                    message="successful decision cannot report solver_success=False",
                )
            )
    else:
        if diagnostics.solver_success is True:
            issues.append(
                ValidationIssue(
                    code="branch_mismatch",
                    field="solver_success",
                    message="ControllerFailure cannot report solver_success=True",
                )
            )
        if controller.name == "mpc" and failure_code == "forecast_coverage":
            for field_name in (
                "solver_success",
                "solver_return_status",
                "solver_iterations",
                "solver_wall_seconds",
            ):
                if getattr(diagnostics, field_name) is not None:
                    issues.append(
                        ValidationIssue(
                            code="branch_mismatch",
                            field=field_name,
                            message=(
                                "pre-solver forecast coverage failure requires empty "
                                "solver diagnostics"
                            ),
                        )
                    )
        elif controller.name == "mpc":
            if diagnostics.solver_success is not False:
                issues.append(
                    ValidationIssue(
                        code="branch_mismatch",
                        field="solver_success",
                        message="MPC ControllerFailure requires solver_success=False",
                    )
                )
            if not (
                isinstance(diagnostics.solver_return_status, str)
                and diagnostics.solver_return_status.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="schema_error",
                        field="solver_return_status",
                        message="MPC ControllerFailure requires a nonempty return status",
                    )
                )

    return tuple(issues)


def _exogenous_from_point(point: ScenarioPoint) -> ExogenousInputs:
    exogenous = ExogenousInputs(
        pv_kw=float(point.pv_kw),
        electric_load_kw=float(point.electric_load_kw),
        price_eur_per_kwh=float(point.price_eur_per_kwh),
        outdoor_temperature_c=float(point.outdoor_temperature_c),
        irradiance_w_per_m2=float(point.irradiance_w_per_m2),
    )
    if not all(np.isfinite(float(value)) for value in exogenous.values()):
        raise ValueError("scenario point contains a non-finite exogenous value")
    return exogenous


def simulate_run(
    scenario: Scenario,
    controller: ControllerAdapter,
    hub_config: HubConfiguration,
) -> ValidRun | InvalidRun:
    start = initial_state(hub_config)
    state = start
    records: list[OperatingRecord] = []
    diagnostics: list[DecisionDiagnostics] = []

    for operating_step in range(scenario.operating_step_count):
        try:
            forecast = scenario.forecast_view(
                operating_step, controller.forecast_horizon_steps
            )
            if not forecast:
                return _invalid_run(
                    scenario,
                    controller,
                    hub_config,
                    operating_step,
                    "forecast_coverage",
                    "controller forecast is empty",
                    records,
                    diagnostics,
                )
        except ScenarioCoverageError as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "forecast_coverage",
                str(exc),
                records,
                diagnostics,
            )
        except Exception as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "scenario_error",
                str(exc),
                records,
                diagnostics,
            )

        # Gate 1: request a decision and reject controller failure immediately.
        try:
            decision = controller.decide(state, forecast)
        except Exception as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "controller_error",
                str(exc),
                records,
                diagnostics,
            )
        if not isinstance(decision, (ControlDecision, ControllerFailure)):
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "controller_schema_error",
                "controller returned neither ControlDecision nor ControllerFailure",
                records,
                diagnostics,
            )
        expected_status = (
            "success" if isinstance(decision, ControlDecision) else "failure"
        )
        diagnostics_issues = _validate_decision_diagnostics(
            decision.diagnostics,
            controller,
            expected_status,
            decision.code if isinstance(decision, ControllerFailure) else None,
        )
        if isinstance(decision.diagnostics, DecisionDiagnostics):
            diagnostics.append(decision.diagnostics)
        if diagnostics_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_diagnostics",
                _issue_message(diagnostics_issues),
                records,
                diagnostics,
            )
        if isinstance(decision, ControllerFailure):
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                decision.code,
                decision.message,
                records,
                diagnostics,
            )

        # Gate 2: clip solver noise within tolerance (including on Disabled Asset
        # fields), then validate the control against the configured bounds.
        try:
            control = normalize_control(
                decision.control,
                hub_config,
                zero_small_flows=False,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_control",
                f"control normalization failed: {exc}",
                records,
                diagnostics,
            )
        control_issues = validate_control(
            control,
            hub_config,
            tolerance=SOLVER_BOUND_TOLERANCE_KW,
        )
        if control_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_control",
                _issue_message(control_issues),
                records,
                diagnostics,
            )

        # Gate 3: advance the plant only with an accepted control.
        try:
            exogenous = _exogenous_from_point(forecast[0])
            step = advance_hub(state, control, exogenous, hub_config)
        except Exception as exc:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "physics_error",
                str(exc),
                records,
                diagnostics,
            )
        if not isinstance(step, HubStep):
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "physics_schema_error",
                (
                    "advance_hub returned "
                    f"{type(step).__name__}; expected HubStep"
                ),
                records,
                diagnostics,
            )

        # Gate 4: validate reached flows and state before recording or advancing.
        flow_issues = validate_flows(step.flows)
        if flow_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_flows",
                _issue_message(flow_issues),
                records,
                diagnostics,
            )
        successor_issues = validate_successor(
            step.successor,
            hub_config,
            controller.requires_operational_storage_bounds,
        )
        if successor_issues:
            return _invalid_run(
                scenario,
                controller,
                hub_config,
                operating_step,
                "invalid_successor",
                _issue_message(successor_issues),
                records,
                diagnostics,
            )

        records.append(
            OperatingRecord(
                operating_step=operating_step,
                timestamp_utc=forecast[0].timestamp_utc,
                start_state=state,
                control=control,
                exogenous=exogenous,
                reached_state=step.successor,
                flows=step.flows,
            )
        )
        state = step.successor

    return ValidRun(
        scenario=scenario,
        controller_name=controller.name,
        controller_configuration=dict(controller.configuration),
        capability_policy=dict(controller.capability_policy),
        hub_configuration=hub_config,
        initial_state=start,
        records=tuple(records),
        controller_diagnostics=tuple(diagnostics),
        terminal_state=state,
        validation=ValidationReport(
            complete=True,
            valid=True,
            checked_operating_steps=len(records),
            issues=(),
        ),
    )
