"""Run one validated greenhouse-hub experiment and persist its evidence.

This entry point selects configurations only. Physics, Run validation,
evaluation, and serialization remain owned by their package Modules.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Sequence
import uuid
from zoneinfo import ZoneInfo

from greenhouse_energy_hub.evaluation import (
    EvaluationPolicy,
    RunBundle,
    RunSpecification,
    _capture_publication_context,
    build_run_specification,
    canonical_json_bytes,
    create_run_bundle,
    evaluate_run,
    verify_run_bundle,
    write_failure_diagnostics,
)
from greenhouse_energy_hub.hub import (
    AssetCapabilities,
    HubConfiguration,
    hub_state_array,
    initial_state,
)
from greenhouse_energy_hub.scenarios import Scenario, build_scenario
from greenhouse_energy_hub.simulation import (
    ControllerAdapter,
    InvalidRun,
    ValidRun,
    simulate_run,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
CANDIDATE_INDEX = RESULTS_ROOT / "diagnostics" / "publication-candidates.json"
AMSTERDAM = ZoneInfo("Europe/Amsterdam")
FULL_IDENTIFIER = re.compile(r"[0-9a-f]{64}")
CANDIDATE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DEFAULT_HORIZON_STEPS = 24

CORE_EXECUTABLE_PATHS = (
    "src/greenhouse_energy_hub/__init__.py",
    "src/greenhouse_energy_hub/controllers/__init__.py",
    "src/greenhouse_energy_hub/evaluation.py",
    "src/greenhouse_energy_hub/scenarios.py",
    "src/greenhouse_energy_hub/hub.py",
    "src/greenhouse_energy_hub/simulation.py",
    "experiments/run_scenario.py",
)


@dataclass(frozen=True)
class ExperimentResult:
    outcome: ValidRun | InvalidRun
    specification: RunSpecification
    artifact: RunBundle | Path


def build_hub_configuration(
    *,
    battery: bool = True,
    hydrogen: bool = True,
    thermal_store: bool = True,
) -> HubConfiguration:
    """Construct the one Hub Configuration shared by every Run component."""
    return HubConfiguration(
        capabilities=AssetCapabilities(
            battery=battery,
            hydrogen=hydrogen,
            thermal_store=thermal_store,
        )
    )


def build_controller(
    controller_name: str,
    hub_configuration: HubConfiguration,
    evaluation_policy: EvaluationPolicy,
    *,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    terminal_weight: float = 1.0,
) -> ControllerAdapter:
    """Construct one Controller Adapter without reproducing rollout logic."""
    if controller_name == "baseline":
        from greenhouse_energy_hub.controllers.baseline import (
            BaselineControllerAdapter,
        )

        return BaselineControllerAdapter(hub_configuration)
    if controller_name != "mpc":
        raise ValueError(f"unknown Controller: {controller_name}")

    from greenhouse_energy_hub.controllers.mpc import (
        MpcConfiguration,
        MpcControllerAdapter,
        build_mpc,
    )

    configuration = MpcConfiguration.from_evaluation_policy(
        evaluation_policy,
        horizon_steps=horizon_steps,
        terminal_weight=terminal_weight,
    )
    mpc, _ = build_mpc(hub_configuration, configuration)
    mpc.x0 = hub_state_array(initial_state(hub_configuration))
    mpc.set_initial_guess()
    return MpcControllerAdapter(
        mpc=mpc,
        forecast_horizon_steps=horizon_steps,
        configuration=configuration.to_controller_metadata(),
        capability_policy=asdict(hub_configuration.capabilities),
    )


def build_experiment_scenario(
    *,
    name: str,
    operating_start: datetime,
    scenario_max_horizon_steps: int,
    calendar_days: int | None = None,
    operating_end: datetime | None = None,
    price_path: str | Path | None = None,
    pv_path: str | Path | None = None,
) -> Scenario:
    """Construct a Scenario through the single validated Scenario interface."""
    source_paths: dict[str, str | Path] = {}
    if price_path is not None:
        source_paths["price_path"] = price_path
    if pv_path is not None:
        source_paths["pv_path"] = pv_path
    return build_scenario(
        name=name,
        operating_start=operating_start,
        operating_end=operating_end,
        calendar_days=calendar_days,
        max_horizon_steps=scenario_max_horizon_steps,
        **source_paths,
    )


def executable_paths_for_controller(
    controller_name: str,
    *,
    extra_paths: tuple[str | Path, ...] = (),
) -> tuple[str | Path, ...]:
    if controller_name == "baseline":
        controller_path = "src/greenhouse_energy_hub/controllers/baseline.py"
    elif controller_name == "mpc":
        controller_path = "src/greenhouse_energy_hub/controllers/mpc.py"
    else:
        raise ValueError(f"unknown Controller: {controller_name}")
    return tuple(dict.fromkeys((*CORE_EXECUTABLE_PATHS, controller_path, *extra_paths)))


def _validated_candidate_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("publication candidate index must be a JSON object")
    normalized: dict[str, str] = {}
    for key, identifier in value.items():
        if not isinstance(key, str) or CANDIDATE_KEY.fullmatch(key) is None:
            raise ValueError("publication candidate keys must be stable names")
        if not isinstance(identifier, str) or FULL_IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError(
                "publication candidate values must be full lowercase SHA-256 identifiers"
            )
        normalized[key] = identifier
    return normalized


def _read_candidate_mapping(candidate_index: Path) -> dict[str, str]:
    if not candidate_index.exists():
        return {}
    if candidate_index.is_symlink() or not candidate_index.is_file():
        raise ValueError("publication candidate index must be a regular file")
    try:
        payload = json.loads(candidate_index.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication candidate index must be valid UTF-8 JSON") from exc
    return _validated_candidate_mapping(payload)


def _atomic_write_candidate_mapping(
    candidate_index: Path,
    mapping: dict[str, str],
) -> None:
    candidate_index.parent.mkdir(parents=True, exist_ok=True)
    if candidate_index.parent.is_symlink() or not candidate_index.parent.is_dir():
        raise ValueError("publication candidate directory must be a real directory")
    temporary = candidate_index.parent / (
        f".{candidate_index.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(mapping))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, candidate_index)
        directory_descriptor = os.open(candidate_index.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def record_publication_candidate(
    candidate_key: str,
    bundle: RunBundle,
    *,
    candidate_index: str | Path = CANDIDATE_INDEX,
    repository_root: str | Path = ROOT,
) -> None:
    """Atomically update a non-authoritative key-to-verified-full-ID map."""
    if not isinstance(candidate_key, str) or CANDIDATE_KEY.fullmatch(candidate_key) is None:
        raise ValueError("publication candidate key must be a stable nonempty name")
    if not isinstance(bundle, RunBundle):
        raise TypeError("publication candidates require a verified RunBundle")
    if FULL_IDENTIFIER.fullmatch(bundle.identifier) is None:
        raise ValueError("publication candidates require a full Run Bundle identifier")
    root = Path(repository_root)
    verified = verify_run_bundle(
        bundle.path,
        bundle.identifier,
        repository_root=root,
    )
    if verified.identifier != bundle.identifier:
        raise ValueError("verified Run Bundle identifier changed during candidate update")

    destination = Path(candidate_index)
    if not destination.is_absolute():
        destination = root / destination
    mapping = _read_candidate_mapping(destination)
    mapping[candidate_key] = verified.identifier
    _atomic_write_candidate_mapping(destination, mapping)


def execute_experiment(
    *,
    name: str,
    operating_start: datetime,
    controller_name: str,
    scenario_max_horizon_steps: int,
    calendar_days: int | None = None,
    operating_end: datetime | None = None,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    terminal_weight: float = 1.0,
    battery: bool = True,
    hydrogen: bool = True,
    thermal_store: bool = True,
    price_path: str | Path | None = None,
    pv_path: str | Path | None = None,
    repository_root: str | Path = ROOT,
    results_root: str | Path = RESULTS_ROOT,
    candidate_key: str | None = None,
    candidate_index: str | Path = CANDIDATE_INDEX,
    extra_executable_paths: tuple[str | Path, ...] = (),
    evaluation_policy: EvaluationPolicy | None = None,
) -> ExperimentResult:
    """Construct, validate, evaluate, and route one experiment outcome."""
    root = Path(repository_root).resolve()
    paths = executable_paths_for_controller(
        controller_name,
        extra_paths=extra_executable_paths,
    )
    publication_context = _capture_publication_context(
        paths,
        repository_root=root,
    )
    scenario = build_experiment_scenario(
        name=name,
        operating_start=operating_start,
        operating_end=operating_end,
        calendar_days=calendar_days,
        scenario_max_horizon_steps=scenario_max_horizon_steps,
        price_path=price_path,
        pv_path=pv_path,
    )
    hub_configuration = build_hub_configuration(
        battery=battery,
        hydrogen=hydrogen,
        thermal_store=thermal_store,
    )
    policy = evaluation_policy or EvaluationPolicy()
    controller = build_controller(
        controller_name,
        hub_configuration,
        policy,
        horizon_steps=horizon_steps,
        terminal_weight=terminal_weight,
    )
    outcome = simulate_run(scenario, controller, hub_configuration)
    specification = build_run_specification(
        outcome,
        policy,
        executable_paths=paths,
        repository_root=root,
        _publication_context=publication_context,
    )
    output_root = Path(results_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    if isinstance(outcome, ValidRun):
        report = evaluate_run(outcome, policy)
        artifact: RunBundle | Path = create_run_bundle(
            outcome,
            report,
            specification,
            output_root / "runs",
            repository_root=root,
        )
        if candidate_key is not None:
            assert isinstance(artifact, RunBundle)
            record_publication_candidate(
                candidate_key,
                artifact,
                candidate_index=candidate_index,
                repository_root=root,
            )
    else:
        artifact = write_failure_diagnostics(
            outcome,
            specification,
            output_root / "diagnostics",
            policy=policy,
        )
    return ExperimentResult(outcome, specification, artifact)


def _amsterdam_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be valid ISO 8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed.astimezone(AMSTERDAM)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one validated greenhouse energy hub Scenario."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--start", required=True, type=_amsterdam_timestamp)
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--end", type=_amsterdam_timestamp)
    window.add_argument("--days", type=int)
    parser.add_argument(
        "--controller",
        choices=("baseline", "mpc"),
        required=True,
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument(
        "--scenario-max-horizon",
        type=int,
        default=DEFAULT_HORIZON_STEPS,
    )
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--price-path", type=Path)
    parser.add_argument("--pv-path", type=Path)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--disable-battery", action="store_true")
    parser.add_argument("--disable-h2", action="store_true")
    parser.add_argument("--disable-tes", action="store_true")
    parser.add_argument("--candidate-key")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = execute_experiment(
        name=args.name,
        operating_start=args.start,
        operating_end=args.end,
        calendar_days=args.days,
        controller_name=args.controller,
        horizon_steps=args.horizon,
        scenario_max_horizon_steps=args.scenario_max_horizon,
        terminal_weight=args.terminal_weight,
        battery=not args.disable_battery,
        hydrogen=not args.disable_h2,
        thermal_store=not args.disable_tes,
        price_path=args.price_path,
        pv_path=args.pv_path,
        repository_root=args.repository_root,
        results_root=args.results_root,
        candidate_key=args.candidate_key,
        candidate_index=Path(args.results_root)
        / "diagnostics"
        / "publication-candidates.json",
    )
    if isinstance(result.outcome, InvalidRun):
        print(
            f"INVALID RUN at step {result.outcome.failed_step}: "
            f"{result.outcome.failure_code}: {result.outcome.message}"
        )
        print(f"Diagnostics -> {result.artifact}")
        return 2
    assert isinstance(result.artifact, RunBundle)
    print(f"Verified Run Bundle {result.artifact.identifier} -> {result.artifact.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
