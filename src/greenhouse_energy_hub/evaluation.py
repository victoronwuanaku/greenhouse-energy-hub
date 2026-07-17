"""Named evaluation policy and reproducible Run-level economic scorecards.

Realized evaluation is intentionally independent of the MPC's solver-only
regularization.  Every Controller is scored from immutable Operating Records by
the same named policy; serialized trajectory columns are compatibility output,
not an accounting input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import ctypes.util
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta, timezone
import errno
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import math
import os
from pathlib import Path
import platform as platform_module
import re
import shutil
import subprocess
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias
import uuid

from greenhouse_energy_hub.hub import (
    BALANCE_STATE_TOLERANCE,
    E_H2_LHV_KWH_KG,
    ETA_BAT_DIS,
    ETA_FC_E,
    HP_COP,
    T_SETPOINT_C,
    T_MAX_C,
    T_MIN_C,
    AssetCapabilities,
    ExogenousInputs,
    HubConfiguration,
    HubControl,
    HubFlows,
    HubState,
    advance_hub,
    initial_state,
    validate_control,
    validate_flows,
    validate_successor,
)
from greenhouse_energy_hub.scenarios import Scenario, ScenarioPoint, SourceProvenance

if TYPE_CHECKING:
    from greenhouse_energy_hub.simulation import (
        InvalidRun,
        OperatingRecord,
        ValidRun,
    )


PROVISIONAL_COEFFICIENT_STATUS = "provisional"
SETTLEMENT_RULE = "arithmetic-mean-operating-wholesale-price"
RUN_SPECIFICATION_SCHEMA_VERSION = "run-specification-v1"
RUN_BUNDLE_SCHEMA_VERSION = "run-bundle-v1"
EVALUATION_REPORT_SCHEMA_VERSION = "evaluation-report-v1"
RUN_VALIDATION_SCHEMA_VERSION = "run-validation-v1"
CANONICALIZATION_VERSION = "canonical-json-v1"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_BUNDLE_MEMBERS = (
    "trajectory.csv",
    "controller_diagnostics.csv",
    "summary.json",
    "validation.json",
)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _immutable_metadata(value: object) -> object:
    """Recursively freeze policy metadata without inheriting mutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _immutable_metadata(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_metadata(item) for item in value)
    return value


def _normal_json_primitives(value: object) -> object:
    """Copy immutable metadata into ordinary JSON dict/list/scalar primitives."""
    if isinstance(value, Mapping):
        return {
            str(key): _normal_json_primitives(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normal_json_primitives(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"metadata value {value!r} is not JSON-compatible")


def _finite_nonnegative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite and nonnegative")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return converted


@dataclass(frozen=True)
class WearCoefficients:
    battery_eur_per_kwh: float = 0.005
    thermal_store_eur_per_kwh: float = 0.0005
    electrolyser_eur_per_kwh: float = 0.002
    fuel_cell_eur_per_kwh: float = 0.002

    def __post_init__(self) -> None:
        for field_name in (
            "battery_eur_per_kwh",
            "thermal_store_eur_per_kwh",
            "electrolyser_eur_per_kwh",
            "fuel_cell_eur_per_kwh",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class EvaluationPolicy:
    name: str = "greenhouse-hub-evaluation"
    version: str = "1"
    grid_import_fee_eur_per_kwh: float = 0.025
    wear: WearCoefficients = WearCoefficients()
    sensitivity_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("EvaluationPolicy name must be nonempty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("EvaluationPolicy version must be nonempty")
        object.__setattr__(
            self,
            "grid_import_fee_eur_per_kwh",
            _finite_nonnegative(
                self.grid_import_fee_eur_per_kwh,
                "grid_import_fee_eur_per_kwh",
            ),
        )
        if not isinstance(self.wear, WearCoefficients):
            raise TypeError("wear must be WearCoefficients")
        multipliers = tuple(
            _finite_nonnegative(value, "sensitivity_multipliers")
            for value in self.sensitivity_multipliers
        )
        if len(set(multipliers)) != len(multipliers):
            raise ValueError("sensitivity_multipliers must be unique")
        if not {0.0, 1.0, 2.0}.issubset(multipliers):
            raise ValueError(
                "sensitivity_multipliers must contain mandatory 0x, 1x, and 2x"
            )
        object.__setattr__(self, "sensitivity_multipliers", multipliers)

    def to_metadata(self) -> Mapping[str, object]:
        """Return deterministic metadata backed by genuine immutable containers."""
        metadata = _immutable_metadata(
            {
                "name": self.name,
                "version": self.version,
                "grid_import_fee_eur_per_kwh": self.grid_import_fee_eur_per_kwh,
                "wear": {
                    **asdict(self.wear),
                    "coefficient_status": PROVISIONAL_COEFFICIENT_STATUS,
                },
                "sensitivity_multipliers": self.sensitivity_multipliers,
                "settlement_rule": SETTLEMENT_RULE,
                "comfort_valuation": None,
            }
        )
        assert isinstance(metadata, Mapping)
        return metadata

    def to_serializable_metadata(self) -> dict[str, object]:
        """Return a fresh ordinary JSON-primitive copy of policy metadata."""
        normalized = _normal_json_primitives(self.to_metadata())
        assert isinstance(normalized, dict)
        return normalized


DEFAULT_EVALUATION_POLICY = EvaluationPolicy()


@dataclass(frozen=True)
class StepLineItems:
    operating_step: int
    grid_cost_eur: float
    battery_wear_eur: float
    thermal_store_wear_eur: float
    electrolyser_wear_eur: float
    fuel_cell_wear_eur: float
    operating_cost_eur: float
    comfort_violation_c_h: float


@dataclass(frozen=True)
class CostSummary:
    grid_cost_eur: float
    battery_wear_eur: float
    thermal_store_wear_eur: float
    electrolyser_wear_eur: float
    fuel_cell_wear_eur: float
    operating_cost_eur: float
    inventory_settlement_eur: float
    inventory_adjusted_cost_eur: float
    comfort_violation_c_h: float


@dataclass(frozen=True)
class EvaluationReport:
    policy: EvaluationPolicy
    nominal: CostSummary
    wear_sensitivities: Mapping[str, CostSummary]
    step_line_items: tuple[StepLineItems, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wear_sensitivities",
            MappingProxyType(dict(self.wear_sensitivities)),
        )
        object.__setattr__(self, "step_line_items", tuple(self.step_line_items))


@dataclass(frozen=True)
class RunSpecification:
    identifier: str
    canonical_content: Mapping[str, JSONValue]
    publication_eligible: bool

    def __post_init__(self) -> None:
        _require_full_sha256(self.identifier, "Run Specification identifier")
        normalized = _to_json_primitives(self.canonical_content)
        if not isinstance(normalized, dict):
            raise TypeError("Run Specification content must be a JSON object")
        object.__setattr__(
            self,
            "canonical_content",
            _freeze_json_mapping(normalized),
        )
        if not isinstance(self.publication_eligible, bool):
            raise TypeError("publication_eligible must be bool")

    def _mapping(self, key: str) -> Mapping[str, JSONValue]:
        value = self.canonical_content[key]
        if not isinstance(value, Mapping):
            raise TypeError(f"Run Specification {key} must be a mapping")
        return value

    @property
    def scenario(self) -> Mapping[str, JSONValue]:
        return self._mapping("scenario")

    @property
    def controller(self) -> Mapping[str, JSONValue]:
        return self._mapping("controller")

    @property
    def asset_capabilities(self) -> Mapping[str, JSONValue]:
        return self._mapping("asset_capabilities")

    @property
    def hub_configuration(self) -> Mapping[str, JSONValue]:
        return self._mapping("hub_configuration")

    @property
    def evaluation_policy(self) -> Mapping[str, JSONValue]:
        return self._mapping("evaluation_policy")

    @property
    def code_provenance(self) -> Mapping[str, JSONValue]:
        return self._mapping("code_provenance")

    @property
    def runtime(self) -> Mapping[str, JSONValue]:
        return self._mapping("runtime")

    @property
    def input_hashes(self) -> Mapping[str, JSONValue]:
        return self._mapping("input_hashes")


@dataclass(frozen=True)
class RunBundle:
    identifier: str
    specification_identifier: str
    path: Path
    manifest: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        _require_full_sha256(self.identifier, "Run Bundle identifier")
        _require_full_sha256(
            self.specification_identifier,
            "Run Specification identifier",
        )
        normalized = _to_json_primitives(self.manifest)
        if not isinstance(normalized, dict):
            raise TypeError("Run Bundle manifest must be a JSON object")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "manifest", _freeze_json_mapping(normalized))


class BundleCollisionError(RuntimeError):
    """An authoritative Run Bundle identifier resolves to different bytes."""


def evaluate_step(
    record: OperatingRecord,
    policy: EvaluationPolicy,
    step_hours: float,
) -> StepLineItems:
    """Recompute one Operating Record's exact realized evaluation line items."""
    step_hours = _finite_nonnegative(step_hours, "step_hours")
    control = record.control
    flows = record.flows
    price = float(record.exogenous.price_eur_per_kwh)
    grid_kw = float(flows.grid_kw)

    grid_cost = (
        price * grid_kw
        + policy.grid_import_fee_eur_per_kwh * max(grid_kw, 0.0)
    ) * step_hours
    battery_wear = policy.wear.battery_eur_per_kwh * (
        float(control.battery_charge_kw) + float(control.battery_discharge_kw)
    ) * step_hours
    thermal_wear = policy.wear.thermal_store_eur_per_kwh * (
        float(control.thermal_charge_kw) + float(control.thermal_discharge_kw)
    ) * step_hours
    electrolyser_wear = (
        policy.wear.electrolyser_eur_per_kwh
        * float(control.electrolyser_kw)
        * step_hours
    )
    fuel_cell_wear = (
        policy.wear.fuel_cell_eur_per_kwh
        * float(control.fuel_cell_kw)
        * step_hours
    )
    reached_temperature = float(record.reached_state.indoor_temperature_c)
    comfort_violation = (
        max(reached_temperature - T_MAX_C, 0.0)
        + max(T_MIN_C - reached_temperature, 0.0)
    ) * step_hours
    operating_cost = (
        grid_cost
        + battery_wear
        + thermal_wear
        + electrolyser_wear
        + fuel_cell_wear
    )
    return StepLineItems(
        operating_step=record.operating_step,
        grid_cost_eur=grid_cost,
        battery_wear_eur=battery_wear,
        thermal_store_wear_eur=thermal_wear,
        electrolyser_wear_eur=electrolyser_wear,
        fuel_cell_wear_eur=fuel_cell_wear,
        operating_cost_eur=operating_cost,
        comfort_violation_c_h=comfort_violation,
    )


def recoverable_inventory_kwh(
    state: HubState,
    config: HubConfiguration = HubConfiguration(),
) -> float:
    """Recoverable electricity-equivalent inventory under Asset capabilities."""
    capabilities = config.capabilities
    battery = (
        ETA_BAT_DIS * float(state.soc_battery_kwh)
        if capabilities.battery
        else 0.0
    )
    hydrogen = (
        ETA_FC_E * E_H2_LHV_KWH_KG * float(state.soc_hydrogen_kg)
        if capabilities.hydrogen
        else 0.0
    )
    thermal = (
        float(state.soc_thermal_kwh) / HP_COP
        if capabilities.thermal_store
        else 0.0
    )
    return battery + hydrogen + thermal


def _summary(
    line_items: tuple[StepLineItems, ...],
    inventory_settlement_eur: float,
) -> CostSummary:
    grid_cost = sum(item.grid_cost_eur for item in line_items)
    battery_wear = sum(item.battery_wear_eur for item in line_items)
    thermal_wear = sum(item.thermal_store_wear_eur for item in line_items)
    electrolyser_wear = sum(item.electrolyser_wear_eur for item in line_items)
    fuel_cell_wear = sum(item.fuel_cell_wear_eur for item in line_items)
    operating_cost = (
        grid_cost
        + battery_wear
        + thermal_wear
        + electrolyser_wear
        + fuel_cell_wear
    )
    return CostSummary(
        grid_cost_eur=grid_cost,
        battery_wear_eur=battery_wear,
        thermal_store_wear_eur=thermal_wear,
        electrolyser_wear_eur=electrolyser_wear,
        fuel_cell_wear_eur=fuel_cell_wear,
        operating_cost_eur=operating_cost,
        inventory_settlement_eur=inventory_settlement_eur,
        inventory_adjusted_cost_eur=operating_cost + inventory_settlement_eur,
        comfort_violation_c_h=sum(
            item.comfort_violation_c_h for item in line_items
        ),
    )


def _scaled_policy(
    policy: EvaluationPolicy,
    multiplier: float,
) -> EvaluationPolicy:
    return replace(
        policy,
        wear=WearCoefficients(
            battery_eur_per_kwh=(
                policy.wear.battery_eur_per_kwh * multiplier
            ),
            thermal_store_eur_per_kwh=(
                policy.wear.thermal_store_eur_per_kwh * multiplier
            ),
            electrolyser_eur_per_kwh=(
                policy.wear.electrolyser_eur_per_kwh * multiplier
            ),
            fuel_cell_eur_per_kwh=(
                policy.wear.fuel_cell_eur_per_kwh * multiplier
            ),
        ),
    )


def _multiplier_label(multiplier: float) -> str:
    return f"{int(multiplier) if multiplier.is_integer() else multiplier:g}x"


def evaluate_run(
    run: ValidRun,
    policy: EvaluationPolicy = DEFAULT_EVALUATION_POLICY,
) -> EvaluationReport:
    """Evaluate one complete ValidRun exclusively from immutable records."""
    from greenhouse_energy_hub.simulation import ValidRun

    if not isinstance(run, ValidRun):
        raise TypeError("evaluate_run accepts only a complete ValidRun")
    if not run.validation.complete or not run.validation.valid:
        raise ValueError("evaluate_run requires a complete and valid Run")
    expected_count = run.scenario.operating_step_count
    assert len(run.records) == expected_count, (
        "ValidRun record count must equal Scenario Operating Step count"
    )
    assert run.validation.checked_operating_steps == expected_count, (
        "ValidationReport count must equal Scenario Operating Step count"
    )

    step_hours = run.scenario.step_duration.total_seconds() / 3600.0
    nominal_line_items = tuple(
        evaluate_step(record, policy, step_hours) for record in run.records
    )
    settlement_price = sum(
        float(record.exogenous.price_eur_per_kwh) for record in run.records
    ) / expected_count
    inventory_settlement = settlement_price * (
        recoverable_inventory_kwh(run.initial_state, run.hub_configuration)
        - recoverable_inventory_kwh(run.terminal_state, run.hub_configuration)
    )
    nominal = _summary(nominal_line_items, inventory_settlement)

    sensitivities: dict[str, CostSummary] = {}
    for multiplier in policy.sensitivity_multipliers:
        if multiplier == 1.0:
            sensitivity_summary = nominal
        else:
            sensitivity_policy = _scaled_policy(policy, multiplier)
            sensitivity_line_items = tuple(
                evaluate_step(record, sensitivity_policy, step_hours)
                for record in run.records
            )
            sensitivity_summary = _summary(
                sensitivity_line_items,
                inventory_settlement,
            )
        sensitivities[_multiplier_label(multiplier)] = sensitivity_summary

    return EvaluationReport(
        policy=policy,
        nominal=nominal,
        wear_sensitivities=sensitivities,
        step_line_items=nominal_line_items,
    )


def saving_percent(baseline_cost: float, comparison_cost: float) -> float:
    """Percentage reduction of comparison cost relative to Baseline cost."""
    return (
        100.0 * (baseline_cost - comparison_cost) / abs(baseline_cost)
        if baseline_cost
        else 0.0
    )


# ---------------------------------------------------------------------------
# Deterministic Run Specifications and immutable Run Bundles
# ---------------------------------------------------------------------------
def canonical_json_bytes(value: JSONValue) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("sha256_bytes accepts bytes")
    return hashlib.sha256(data).hexdigest()


def _require_full_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a full lowercase SHA-256 digest")
    return value


def _to_json_primitives(value: object, field_name: str = "value") -> JSONValue:
    """Copy identity inputs into ordinary, finite JSON primitives."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} JSON object keys must be strings")
        return {
            key: _to_json_primitives(item, f"{field_name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _to_json_primitives(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, datetime):
        return _utc_z(value, field_name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite JSON numeric data")
        return value
    raise TypeError(f"{field_name} value {value!r} is not JSON-compatible")


def _freeze_json(value: JSONValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_json_mapping(
    value: dict[str, JSONValue],
) -> Mapping[str, JSONValue]:
    frozen = _freeze_json(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _utc_z(value: object, field_name: str = "timestamp") -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    converted = value.astimezone(timezone.utc)
    return converted.isoformat().replace("+00:00", "Z")


def _package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"required runtime distribution {distribution!r} is not installed"
        ) from exc


def _actual_runtime_manifest() -> dict[str, JSONValue]:
    return {
        "python": platform_module.python_version(),
        "platform": platform_module.platform(),
        "do_mpc": _package_version("do-mpc"),
        "casadi": _package_version("casadi"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
    }


def _repository_relative_path(path: str | Path, repository_root: Path) -> str:
    """Return the lexical Git selector and reject every symlink in its path."""
    resolved_root = repository_root.resolve(strict=True)
    candidate = Path(path)
    lexical = Path(
        os.path.abspath(
            os.fspath(candidate if candidate.is_absolute() else resolved_root / candidate)
        )
    )
    try:
        relative = lexical.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("executable paths must remain inside repository_root") from exc
    if not relative.parts:
        raise ValueError("executable path must identify a file")
    current = resolved_root
    for part in relative.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise ValueError(
                    f"executable path contains a symlink component: {relative.as_posix()}"
                )
            current.lstat()
        except FileNotFoundError:
            break
    return relative.as_posix()


def collect_code_provenance(
    executable_paths: tuple[str | Path, ...],
    *,
    repository_root: str | Path = Path(__file__).resolve().parents[2],
) -> dict[str, JSONValue]:
    """Hash only the explicit executable/configuration inputs used by a Run."""
    if not isinstance(executable_paths, tuple) or not executable_paths:
        raise ValueError("executable_paths must be a nonempty explicit tuple")
    root = Path(repository_root).resolve()
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if revision_result.returncode != 0:
        raise ValueError("repository_root must be a Git worktree")
    revision = revision_result.stdout.decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Git revision must be a full 40-character object ID")

    normalized_paths = tuple(
        _repository_relative_path(path, root) for path in executable_paths
    )
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("executable_paths must not contain duplicates")

    working_hashes: dict[str, JSONValue] = {}
    dirty: list[JSONValue] = []
    untracked: list[JSONValue] = []
    committed_hashes: dict[str, JSONValue] = {}
    for relative in sorted(normalized_paths):
        working_path = root / relative
        if not working_path.is_file():
            raise FileNotFoundError(f"executable input does not exist: {relative}")
        working_bytes = working_path.read_bytes()
        working_hashes[relative] = sha256_bytes(working_bytes)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if committed.returncode != 0:
            untracked.append(relative)
            dirty.append(relative)
            continue
        committed_hashes[relative] = sha256_bytes(committed.stdout)
        if committed.stdout != working_bytes:
            dirty.append(relative)

    source_digest = sha256_bytes(canonical_json_bytes(working_hashes))
    return {
        "git_revision": revision,
        "executable_source_tree_sha256": source_digest,
        "executable_path_hashes": working_hashes,
        "committed_executable_path_hashes": committed_hashes,
        "publication_eligible": not dirty,
        "dirty_executable_paths": dirty,
        "untracked_executable_paths": untracked,
    }


def _repository_relative_selector(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a nonempty repository-relative path")
    selector = Path(value)
    if (
        selector.is_absolute()
        or ".." in selector.parts
        or selector.as_posix() != value
        or not selector.parts
    ):
        raise ValueError(f"{description} must be an exact repository-relative path")
    return value


def _git_object_bytes(
    repository_root: str | Path,
    revision: str,
    relative_path: str,
    description: str,
) -> bytes:
    _require_full_git_revision(revision, "recorded Git revision")
    selector = _repository_relative_selector(relative_path, description)
    root = Path(repository_root).resolve(strict=True)
    result = subprocess.run(
        ["git", "show", f"{revision}:{selector}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(
            f"recorded Git revision lacks authoritative {description}: {selector}"
        )
    return result.stdout


def _require_full_git_revision(value: object, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field_name} must be a full 40-character object ID")
    return value


def _verify_git_hash_map(
    repository_root: str | Path,
    revision: str,
    hashes: Mapping[str, str],
    description: str,
) -> None:
    for relative_path, expected_hash in hashes.items():
        committed_bytes = _git_object_bytes(
            repository_root,
            revision,
            relative_path,
            description,
        )
        if sha256_bytes(committed_bytes) != expected_hash:
            raise ValueError(
                f"authoritative Git {description} bytes do not match recorded hash: "
                f"{relative_path}"
            )


@dataclass(frozen=True)
class _PublicationContext:
    executable_paths: tuple[str, ...]
    repository_root: Path
    code_provenance: Mapping[str, JSONValue]
    runtime: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "executable_paths", tuple(self.executable_paths))
        object.__setattr__(self, "repository_root", Path(self.repository_root).resolve())
        object.__setattr__(
            self,
            "code_provenance",
            _freeze_json_mapping(
                _to_json_primitives(self.code_provenance, "code_provenance")
            ),
        )
        object.__setattr__(
            self,
            "runtime",
            _freeze_json_mapping(_to_json_primitives(self.runtime, "runtime")),
        )


def _capture_publication_context(
    executable_paths: tuple[str | Path, ...],
    *,
    repository_root: str | Path = Path(__file__).resolve().parents[2],
) -> _PublicationContext:
    """Capture immutable identity inputs before a potentially long execution."""
    root = Path(repository_root).resolve()
    code_provenance = collect_code_provenance(
        executable_paths,
        repository_root=root,
    )
    normalized_paths = tuple(
        sorted(code_provenance["executable_path_hashes"])
    )
    return _PublicationContext(
        executable_paths=normalized_paths,
        repository_root=root,
        code_provenance=code_provenance,
        runtime=_actual_runtime_manifest(),
    )


def _scenario_content(scenario: object) -> dict[str, JSONValue]:
    points = []
    for point in scenario.points:
        points.append(
            {
                "timestamp_utc": _utc_z(point.timestamp_utc),
                "price_eur_per_kwh": float(point.price_eur_per_kwh),
                "pv_kw": float(point.pv_kw),
                "electric_load_kw": float(point.electric_load_kw),
                "outdoor_temperature_c": float(point.outdoor_temperature_c),
                "irradiance_w_per_m2": float(point.irradiance_w_per_m2),
            }
        )
    provenance = []
    for item in scenario.provenance:
        provenance.append(
            {
                "source_name": item.source_name,
                "source_path": item.source_path,
                "sha256": item.sha256,
                "acquisition_parameters": _to_json_primitives(
                    item.acquisition_parameters,
                    f"scenario.provenance.{item.source_name}.acquisition_parameters",
                ),
                "original_timezone": item.original_timezone,
                "units": _to_json_primitives(item.units),
                "transformations": list(item.transformations),
            }
        )
    content: dict[str, JSONValue] = {
        "name": scenario.name,
        "operating_start_utc": _utc_z(scenario.operating_start),
        "operating_end_utc": _utc_z(scenario.operating_end),
        "forecast_end_utc": _utc_z(scenario.forecast_end),
        "forecast_horizon_capacity_steps": scenario.forecast_horizon_capacity_steps,
        "step_duration_seconds": scenario.step_duration.total_seconds(),
        "operating_step_count": scenario.operating_step_count,
        "points": points,
        "provenance": provenance,
    }
    normalized = _to_json_primitives(content, "scenario")
    assert isinstance(normalized, dict)
    return normalized


def _scenario_input_hashes(
    scenario_content: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    provenance = scenario_content["provenance"]
    assert isinstance(provenance, list)
    source_hashes: dict[str, JSONValue] = {}
    sidecar_hashes: dict[str, JSONValue] = {}
    for item in provenance:
        assert isinstance(item, dict)
        source_path = item["source_path"]
        source_hash = item["sha256"]
        assert isinstance(source_path, str) and isinstance(source_hash, str)
        _require_full_sha256(source_hash, f"source hash for {source_path}")
        source_hashes[source_path] = source_hash
        acquisition = item["acquisition_parameters"]
        assert isinstance(acquisition, dict)
        sidecar_path = acquisition.get("sidecar_path")
        sidecar_hash = acquisition.get("sidecar_sha256")
        if not isinstance(sidecar_path, str) or not sidecar_path:
            raise ValueError(f"source {source_path} is missing sidecar provenance path")
        _require_full_sha256(sidecar_hash, f"sidecar hash for {source_path}")
        sidecar_hashes[sidecar_path] = sidecar_hash
    return {
        "scenario": sha256_bytes(canonical_json_bytes(dict(scenario_content))),
        "sources": source_hashes,
        "sidecars": sidecar_hashes,
    }


def _validate_publication_run_evidence(run: object) -> None:
    from greenhouse_energy_hub.simulation import InvalidRun, ValidRun

    if not isinstance(run, (ValidRun, InvalidRun)):
        raise TypeError("Run Specification requires a ValidRun or InvalidRun")
    if not run.scenario.provenance:
        raise ValueError(
            "Run Specification rejects provenance-empty compatibility Scenarios"
        )
    if isinstance(run, InvalidRun):
        return
    expected = run.scenario.operating_step_count
    if not run.validation.complete or not run.validation.valid:
        raise ValueError("Run Specification requires complete valid Run evidence")
    if (
        len(run.records) != expected
        or len(run.controller_diagnostics) != expected
        or run.validation.checked_operating_steps != expected
    ):
        raise ValueError("Run evidence must contain one record and diagnostic per step")
    for index, diagnostics in enumerate(run.controller_diagnostics):
        if diagnostics.adapter != run.controller_name or diagnostics.decision_status != "success":
            raise ValueError(f"step {index} has invalid controller diagnostic evidence")
        if run.controller_name == "mpc":
            if diagnostics.solver_success is not True:
                raise ValueError(f"step {index} is missing successful MPC solver evidence")
            if (
                not isinstance(diagnostics.solver_return_status, str)
                or not diagnostics.solver_return_status
            ):
                raise ValueError(f"step {index} is missing MPC solver return status evidence")
            if (
                isinstance(diagnostics.solver_iterations, bool)
                or not isinstance(diagnostics.solver_iterations, int)
                or diagnostics.solver_iterations < 0
            ):
                raise ValueError(f"step {index} has invalid MPC solver iteration evidence")
            if (
                isinstance(diagnostics.solver_wall_seconds, bool)
                or not isinstance(diagnostics.solver_wall_seconds, (int, float))
                or not math.isfinite(float(diagnostics.solver_wall_seconds))
                or diagnostics.solver_wall_seconds < 0
            ):
                raise ValueError(f"step {index} has invalid MPC solver timing evidence")
        elif any(
            value is not None
            for value in (
                diagnostics.solver_success,
                diagnostics.solver_return_status,
                diagnostics.solver_iterations,
                diagnostics.solver_wall_seconds,
            )
        ):
            raise ValueError("non-MPC diagnostics must not claim solver evidence")


def run_specification_identifier(
    canonical_content: Mapping[str, object],
) -> str:
    normalized = _to_json_primitives(canonical_content, "Run Specification")
    if not isinstance(normalized, dict):
        raise TypeError("Run Specification content must be a JSON object")
    return sha256_bytes(canonical_json_bytes(normalized))


def build_run_specification(
    run: ValidRun | InvalidRun,
    policy: EvaluationPolicy,
    *,
    executable_paths: tuple[str | Path, ...],
    repository_root: str | Path = Path(__file__).resolve().parents[2],
    _publication_context: _PublicationContext | None = None,
) -> RunSpecification:
    """Build the exact requested-input identity, independent of output bytes."""
    if not isinstance(policy, EvaluationPolicy):
        raise TypeError("policy must be an EvaluationPolicy")
    _validate_publication_run_evidence(run)
    scenario = _scenario_content(run.scenario)
    input_hashes = _scenario_input_hashes(scenario)
    root = Path(repository_root).resolve()
    context = _publication_context or _capture_publication_context(
        executable_paths,
        repository_root=root,
    )
    if context.repository_root != root:
        raise ValueError("publication context repository_root does not match")
    requested_paths = tuple(
        sorted(_repository_relative_path(path, root) for path in executable_paths)
    )
    if requested_paths != context.executable_paths:
        raise ValueError("publication context executable paths do not match")
    code_provenance = _to_json_primitives(context.code_provenance)
    runtime = _to_json_primitives(context.runtime)
    assert isinstance(code_provenance, dict) and isinstance(runtime, dict)
    controller = _to_json_primitives(
        {
            "name": run.controller_name,
            "configuration": run.controller_configuration,
            "capability_policy": run.capability_policy,
        },
        "controller",
    )
    capabilities = _to_json_primitives(asdict(run.hub_configuration.capabilities))
    hub_configuration = _to_json_primitives(asdict(run.hub_configuration))
    evaluation_policy = _to_json_primitives(policy.to_serializable_metadata())
    assert isinstance(controller, dict)
    assert isinstance(capabilities, dict)
    assert isinstance(hub_configuration, dict)
    assert isinstance(evaluation_policy, dict)
    content: dict[str, JSONValue] = {
        "schema_version": RUN_SPECIFICATION_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "scenario": scenario,
        "controller": controller,
        "asset_capabilities": capabilities,
        "hub_configuration": hub_configuration,
        "evaluation_policy": evaluation_policy,
        "code_provenance": code_provenance,
        "runtime": runtime,
        "input_hashes": input_hashes,
    }
    identifier = run_specification_identifier(content)
    return RunSpecification(
        identifier=identifier,
        canonical_content=content,
        publication_eligible=bool(code_provenance["publication_eligible"]),
    )


TRAJECTORY_COLUMNS = (
    "operating_step",
    "timestamp_utc",
    "start_soc_battery_kwh",
    "start_soc_hydrogen_kg",
    "start_soc_thermal_kwh",
    "start_indoor_temperature_c",
    "battery_charge_kw",
    "battery_discharge_kw",
    "electrolyser_kw",
    "fuel_cell_kw",
    "heat_pump_kw",
    "electric_boiler_kw",
    "thermal_charge_kw",
    "thermal_discharge_kw",
    "ventilation_fraction",
    "pv_kw",
    "electric_load_kw",
    "price_eur_per_kwh",
    "outdoor_temperature_c",
    "irradiance_w_per_m2",
    "reached_soc_battery_kwh",
    "reached_soc_hydrogen_kg",
    "reached_soc_thermal_kwh",
    "reached_indoor_temperature_c",
    "grid_kw",
    "generated_heat_kw",
    "heat_to_air_kw",
    "thermal_charge_margin_kw",
    "hydrogen_production_kg_per_h",
    "hydrogen_consumption_kg_per_h",
)

DIAGNOSTIC_COLUMNS = (
    "operating_step",
    "adapter",
    "decision_status",
    "solver_success",
    "solver_return_status",
    "solver_iterations",
    "solver_wall_seconds",
    "forecast_start_utc",
    "forecast_end_utc",
    "terminal_electric_value_eur_per_kwh",
    "terminal_heat_value_eur_per_kwhth",
)


def _finite_csv_number(value: object, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be finite numeric data")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite numeric data")
    return value


def _optional_csv_number(value: object, field_name: str) -> str | int | float:
    if value is None:
        return ""
    return _finite_csv_number(value, field_name)


def _optional_timestamp(value: object, field_name: str) -> str:
    return "" if value is None else _utc_z(value, field_name)


def _csv_bytes(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return target.getvalue().encode("utf-8")


def _trajectory_row(record: object) -> dict[str, object]:
    state = record.start_state
    control = record.control
    exogenous = record.exogenous
    reached = record.reached_state
    flows = record.flows
    row = {
        "operating_step": record.operating_step,
        "timestamp_utc": _utc_z(record.timestamp_utc),
        "start_soc_battery_kwh": state.soc_battery_kwh,
        "start_soc_hydrogen_kg": state.soc_hydrogen_kg,
        "start_soc_thermal_kwh": state.soc_thermal_kwh,
        "start_indoor_temperature_c": state.indoor_temperature_c,
        "battery_charge_kw": control.battery_charge_kw,
        "battery_discharge_kw": control.battery_discharge_kw,
        "electrolyser_kw": control.electrolyser_kw,
        "fuel_cell_kw": control.fuel_cell_kw,
        "heat_pump_kw": control.heat_pump_kw,
        "electric_boiler_kw": control.electric_boiler_kw,
        "thermal_charge_kw": control.thermal_charge_kw,
        "thermal_discharge_kw": control.thermal_discharge_kw,
        "ventilation_fraction": control.ventilation_fraction,
        "pv_kw": exogenous.pv_kw,
        "electric_load_kw": exogenous.electric_load_kw,
        "price_eur_per_kwh": exogenous.price_eur_per_kwh,
        "outdoor_temperature_c": exogenous.outdoor_temperature_c,
        "irradiance_w_per_m2": exogenous.irradiance_w_per_m2,
        "reached_soc_battery_kwh": reached.soc_battery_kwh,
        "reached_soc_hydrogen_kg": reached.soc_hydrogen_kg,
        "reached_soc_thermal_kwh": reached.soc_thermal_kwh,
        "reached_indoor_temperature_c": reached.indoor_temperature_c,
        "grid_kw": flows.grid_kw,
        "generated_heat_kw": flows.generated_heat_kw,
        "heat_to_air_kw": flows.heat_to_air_kw,
        "thermal_charge_margin_kw": flows.thermal_charge_margin_kw,
        "hydrogen_production_kg_per_h": flows.hydrogen_production_kg_per_h,
        "hydrogen_consumption_kg_per_h": flows.hydrogen_consumption_kg_per_h,
    }
    for column in TRAJECTORY_COLUMNS:
        if column not in {"operating_step", "timestamp_utc"}:
            row[column] = _finite_csv_number(row[column], column)
    return row


def _diagnostic_row(operating_step: int, diagnostics: object) -> dict[str, object]:
    return {
        "operating_step": operating_step,
        "adapter": diagnostics.adapter,
        "decision_status": diagnostics.decision_status,
        "solver_success": (
            "" if diagnostics.solver_success is None else str(diagnostics.solver_success).lower()
        ),
        "solver_return_status": diagnostics.solver_return_status or "",
        "solver_iterations": _optional_csv_number(
            diagnostics.solver_iterations,
            "solver_iterations",
        ),
        "solver_wall_seconds": _optional_csv_number(
            diagnostics.solver_wall_seconds,
            "solver_wall_seconds",
        ),
        "forecast_start_utc": _optional_timestamp(
            diagnostics.forecast_start_utc,
            "forecast_start_utc",
        ),
        "forecast_end_utc": _optional_timestamp(
            diagnostics.forecast_end_utc,
            "forecast_end_utc",
        ),
        "terminal_electric_value_eur_per_kwh": _optional_csv_number(
            diagnostics.terminal_electric_value_eur_per_kwh,
            "terminal_electric_value_eur_per_kwh",
        ),
        "terminal_heat_value_eur_per_kwhth": _optional_csv_number(
            diagnostics.terminal_heat_value_eur_per_kwhth,
            "terminal_heat_value_eur_per_kwhth",
        ),
    }


def _summary_content(report: EvaluationReport) -> dict[str, JSONValue]:
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "policy": report.policy.to_serializable_metadata(),
        "nominal": asdict(report.nominal),
        "wear_sensitivities": {
            label: asdict(summary)
            for label, summary in report.wear_sensitivities.items()
        },
        "step_line_items": [asdict(item) for item in report.step_line_items],
    }


def _numeric_dataclass_matches(
    observed: object,
    expected: object,
    expected_type: type,
    *,
    tolerance: float = BALANCE_STATE_TOLERANCE,
) -> bool:
    if not isinstance(observed, expected_type) or not isinstance(expected, expected_type):
        return False
    for item in fields(expected_type):
        try:
            left = float(getattr(observed, item.name))
            right = float(getattr(expected, item.name))
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        if abs(left - right) > tolerance:
            return False
    return True


def _validation_content(run: object) -> dict[str, JSONValue]:
    steps = []
    for record, diagnostics in zip(
        run.records,
        run.controller_diagnostics,
        strict=True,
    ):
        control_valid = not validate_control(record.control, run.hub_configuration)
        flows_valid = not validate_flows(record.flows)
        successor_valid = not validate_successor(
            record.reached_state,
            run.hub_configuration,
            run.controller_name == "mpc",
        )
        recomputed = advance_hub(
            record.start_state,
            record.control,
            record.exogenous,
            run.hub_configuration,
        )
        physics_valid = (
            _numeric_dataclass_matches(
                record.reached_state,
                recomputed.successor,
                HubState,
            )
            and _numeric_dataclass_matches(
                record.flows,
                recomputed.flows,
                HubFlows,
            )
        )
        steps.append(
            {
                "operating_step": record.operating_step,
                "decision_status": diagnostics.decision_status,
                "solver_success": diagnostics.solver_success,
                "solver_return_status": diagnostics.solver_return_status,
                "control_valid": control_valid,
                "flows_valid": flows_valid,
                "successor_valid": successor_valid,
                "physical_invariants_valid": (
                    control_valid and flows_valid and successor_valid and physics_valid
                ),
            }
        )
    return {
        "schema_version": RUN_VALIDATION_SCHEMA_VERSION,
        "complete": True,
        "valid": True,
        "checked_operating_steps": run.validation.checked_operating_steps,
        "issues": [],
        "steps": steps,
    }


def _validate_record_sequence(run: object) -> None:
    expected_initial = initial_state(run.hub_configuration)
    if not _numeric_dataclass_matches(run.initial_state, expected_initial, HubState):
        raise ValueError("ValidRun initial state does not match HubConfiguration")
    expected_start = run.initial_state
    for operating_step, record in enumerate(run.records):
        point = run.scenario.points[operating_step]
        if record.operating_step != operating_step:
            raise ValueError("Operating Records must be in exact step order")
        if record.timestamp_utc != point.timestamp_utc:
            raise ValueError("Operating Record timestamp does not match Scenario")
        if not _numeric_dataclass_matches(record.start_state, expected_start, HubState):
            raise ValueError("Operating Record state continuity is invalid")
        expected_exogenous = ExogenousInputs(
            pv_kw=point.pv_kw,
            electric_load_kw=point.electric_load_kw,
            price_eur_per_kwh=point.price_eur_per_kwh,
            outdoor_temperature_c=point.outdoor_temperature_c,
            irradiance_w_per_m2=point.irradiance_w_per_m2,
        )
        if not _numeric_dataclass_matches(
            record.exogenous,
            expected_exogenous,
            ExogenousInputs,
            tolerance=0.0,
        ):
            raise ValueError("Operating Record exogenous inputs do not match Scenario")
        expected_start = record.reached_state
    if not _numeric_dataclass_matches(run.terminal_state, expected_start, HubState):
        raise ValueError("ValidRun terminal state breaks record continuity")


def serialize_valid_run(
    run: ValidRun,
    report: EvaluationReport,
) -> dict[str, bytes]:
    """Serialize authoritative output members without a terminal pseudo-row."""
    from greenhouse_energy_hub.simulation import ValidRun

    if not isinstance(run, ValidRun):
        raise TypeError("serialize_valid_run accepts only ValidRun")
    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    _validate_publication_run_evidence(run)
    expected_report = evaluate_run(run, report.policy)
    if report != expected_report:
        raise ValueError("EvaluationReport does not match the ValidRun records")
    _validate_record_sequence(run)
    trajectory_rows = [_trajectory_row(record) for record in run.records]
    diagnostic_rows = [
        _diagnostic_row(index, diagnostics)
        for index, diagnostics in enumerate(run.controller_diagnostics)
    ]
    summary = _to_json_primitives(_summary_content(report), "summary")
    validation = _to_json_primitives(_validation_content(run), "validation")
    assert isinstance(summary, dict) and isinstance(validation, dict)
    if any(
        step["physical_invariants_valid"] is not True
        for step in validation["steps"]
    ):
        raise ValueError("ValidRun records fail derived physical validation")
    return {
        "trajectory.csv": _csv_bytes(TRAJECTORY_COLUMNS, trajectory_rows),
        "controller_diagnostics.csv": _csv_bytes(
            DIAGNOSTIC_COLUMNS,
            diagnostic_rows,
        ),
        "summary.json": canonical_json_bytes(summary),
        "validation.json": canonical_json_bytes(validation),
    }


def _read_canonical_json(path: Path, description: str) -> dict[str, JSONValue]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Run Bundle is missing {description}: {path.name}") from exc
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Run Bundle {description} is not finite UTF-8 JSON") from exc
    normalized = _to_json_primitives(parsed, description)
    if not isinstance(normalized, dict):
        raise ValueError(f"Run Bundle {description} must be a JSON object")
    if raw != canonical_json_bytes(normalized):
        raise ValueError(f"Run Bundle {description} is not canonical JSON")
    return normalized


def _require_hash_mapping(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    if not isinstance(value, dict) or (not value and not allow_empty):
        raise ValueError(f"{field_name} must be a nonempty hash mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be nonempty strings")
        result[key] = _require_full_sha256(digest, f"{field_name}.{key}")
    return result


def _csv_rows(path: Path, description: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Run Bundle is missing {description}") from exc
    if not raw.endswith(b"\n") or b"\r\n" in raw:
        raise ValueError(f"Run Bundle {description} must use LF and end with LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Run Bundle {description} must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"Run Bundle {description} has an invalid header")
    return list(reader.fieldnames), list(reader)


def _verify_operating_steps(
    rows: Sequence[Mapping[str, str]],
    expected_count: int,
    description: str,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"Run Bundle {description} must contain one row per Operating Step"
        )
    for expected, row in enumerate(rows):
        try:
            observed = int(row["operating_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Run Bundle {description} lacks ordered operating_step values"
            ) from exc
        if observed != expected:
            raise ValueError(
                f"Run Bundle {description} Operating Steps are not contiguous"
            )


def _require_exact_keys(
    value: object,
    expected: set[str],
    description: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Run Bundle {description} schema is incomplete or unknown")
    return value


def _parse_utc_z_text(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"Run Bundle {field_name} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"Run Bundle {field_name} timestamp is invalid") from exc
    if _utc_z(parsed, field_name) != value:
        raise ValueError(f"Run Bundle {field_name} timestamp is not canonical UTC Z")
    return parsed


def _parse_csv_float(value: object, field_name: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Run Bundle {field_name} must be numeric")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Run Bundle {field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Run Bundle {field_name} must be finite")
    return parsed


def _parse_csv_int(value: object, field_name: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"Run Bundle {field_name} must be a canonical nonnegative int")
    return int(value)


def _parse_optional_csv_float(value: object, field_name: str) -> float | None:
    return None if value == "" else _parse_csv_float(value, field_name)


def _parse_optional_csv_int(value: object, field_name: str) -> int | None:
    return None if value == "" else _parse_csv_int(value, field_name)


def _parse_optional_csv_bool(value: object, field_name: str) -> bool | None:
    if value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Run Bundle {field_name} must be true, false, or empty")


def _parse_optional_csv_timestamp(value: object, field_name: str) -> datetime | None:
    return None if value == "" else _parse_utc_z_text(value, field_name)


def _scenario_from_manifest(value: object) -> Scenario:
    content = _require_exact_keys(
        value,
        {
            "name",
            "operating_start_utc",
            "operating_end_utc",
            "forecast_end_utc",
            "forecast_horizon_capacity_steps",
            "step_duration_seconds",
            "operating_step_count",
            "points",
            "provenance",
        },
        "scenario",
    )
    points_value = content["points"]
    provenance_value = content["provenance"]
    if not isinstance(points_value, list) or not isinstance(provenance_value, list):
        raise ValueError("Run Bundle scenario points/provenance schema is invalid")
    points = []
    for index, item in enumerate(points_value):
        point = _require_exact_keys(
            item,
            {
                "timestamp_utc",
                "price_eur_per_kwh",
                "pv_kw",
                "electric_load_kw",
                "outdoor_temperature_c",
                "irradiance_w_per_m2",
            },
            f"scenario.points[{index}]",
        )
        points.append(
            ScenarioPoint(
                timestamp_utc=_parse_utc_z_text(
                    point["timestamp_utc"],
                    f"scenario.points[{index}].timestamp_utc",
                ),
                price_eur_per_kwh=point["price_eur_per_kwh"],
                pv_kw=point["pv_kw"],
                electric_load_kw=point["electric_load_kw"],
                outdoor_temperature_c=point["outdoor_temperature_c"],
                irradiance_w_per_m2=point["irradiance_w_per_m2"],
            )
        )
    provenance = []
    for index, item in enumerate(provenance_value):
        source = _require_exact_keys(
            item,
            {
                "source_name",
                "source_path",
                "sha256",
                "acquisition_parameters",
                "original_timezone",
                "units",
                "transformations",
            },
            f"scenario.provenance[{index}]",
        )
        acquisition = source["acquisition_parameters"]
        units = source["units"]
        transformations = source["transformations"]
        if (
            not isinstance(acquisition, dict)
            or not isinstance(units, dict)
            or not isinstance(transformations, list)
        ):
            raise ValueError("Run Bundle scenario provenance schema is invalid")
        provenance.append(
            SourceProvenance(
                source_name=source["source_name"],
                source_path=source["source_path"],
                sha256=source["sha256"],
                acquisition_parameters=acquisition,
                original_timezone=source["original_timezone"],
                units=units,
                transformations=tuple(transformations),
            )
        )
    duration = content["step_duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("Run Bundle scenario step duration must be numeric")
    scenario = Scenario(
        name=content["name"],
        operating_start=_parse_utc_z_text(
            content["operating_start_utc"], "scenario.operating_start_utc"
        ),
        operating_end=_parse_utc_z_text(
            content["operating_end_utc"], "scenario.operating_end_utc"
        ),
        forecast_end=_parse_utc_z_text(
            content["forecast_end_utc"], "scenario.forecast_end_utc"
        ),
        forecast_horizon_capacity_steps=content["forecast_horizon_capacity_steps"],
        step_duration=timedelta(seconds=float(duration)),
        operating_step_count=content["operating_step_count"],
        points=tuple(points),
        provenance=tuple(provenance),
    )
    if _scenario_content(scenario) != content:
        raise ValueError("Run Bundle scenario is not in its exact canonical schema")
    return scenario


def _hub_configuration_from_manifest(value: object) -> HubConfiguration:
    content = _require_exact_keys(
        value,
        {"battery", "hydrogen", "thermal_store"},
        "asset_capabilities",
    )
    if any(type(content[key]) is not bool for key in content):
        raise ValueError("Run Bundle Asset capabilities must be exact booleans")
    return HubConfiguration(capabilities=AssetCapabilities(**content))


def _policy_from_manifest(value: object) -> EvaluationPolicy:
    content = _require_exact_keys(
        value,
        {
            "name",
            "version",
            "grid_import_fee_eur_per_kwh",
            "wear",
            "sensitivity_multipliers",
            "settlement_rule",
            "comfort_valuation",
        },
        "evaluation_policy",
    )
    wear = _require_exact_keys(
        content["wear"],
        {
            "battery_eur_per_kwh",
            "thermal_store_eur_per_kwh",
            "electrolyser_eur_per_kwh",
            "fuel_cell_eur_per_kwh",
            "coefficient_status",
        },
        "evaluation_policy.wear",
    )
    if wear["coefficient_status"] != PROVISIONAL_COEFFICIENT_STATUS:
        raise ValueError("Run Bundle evaluation coefficient status is unsupported")
    if content["settlement_rule"] != SETTLEMENT_RULE or content["comfort_valuation"] is not None:
        raise ValueError("Run Bundle Evaluation Policy semantics are unsupported")
    multipliers = content["sensitivity_multipliers"]
    if not isinstance(multipliers, list):
        raise ValueError("Run Bundle Evaluation Policy sensitivities must be a list")
    policy = EvaluationPolicy(
        name=content["name"],
        version=content["version"],
        grid_import_fee_eur_per_kwh=content["grid_import_fee_eur_per_kwh"],
        wear=WearCoefficients(
            battery_eur_per_kwh=wear["battery_eur_per_kwh"],
            thermal_store_eur_per_kwh=wear["thermal_store_eur_per_kwh"],
            electrolyser_eur_per_kwh=wear["electrolyser_eur_per_kwh"],
            fuel_cell_eur_per_kwh=wear["fuel_cell_eur_per_kwh"],
        ),
        sensitivity_multipliers=tuple(multipliers),
    )
    if policy.to_serializable_metadata() != content:
        raise ValueError("Run Bundle Evaluation Policy is not canonical")
    return policy


def _validate_identity_graph(
    manifest: dict[str, JSONValue],
    repository_root: str | Path,
) -> tuple[
    Scenario,
    HubConfiguration,
    EvaluationPolicy,
    dict[str, object],
]:
    scenario = _scenario_from_manifest(manifest["scenario"])
    controller = _require_exact_keys(
        manifest["controller"],
        {"name", "configuration", "capability_policy"},
        "controller",
    )
    if (
        not isinstance(controller["name"], str)
        or not controller["name"]
        or not isinstance(controller["configuration"], dict)
        or not isinstance(controller["capability_policy"], dict)
    ):
        raise ValueError("Run Bundle controller schema is invalid")
    _to_json_primitives(controller["configuration"], "controller.configuration")
    _to_json_primitives(controller["capability_policy"], "controller.capability_policy")
    hub_configuration = _hub_configuration_from_manifest(
        manifest["asset_capabilities"]
    )
    policy = _policy_from_manifest(manifest["evaluation_policy"])

    code = _require_exact_keys(
        manifest["code_provenance"],
        {
            "git_revision",
            "executable_source_tree_sha256",
            "executable_path_hashes",
            "committed_executable_path_hashes",
            "publication_eligible",
            "dirty_executable_paths",
            "untracked_executable_paths",
        },
        "code_provenance",
    )
    if (
        code["publication_eligible"] is not True
        or code["dirty_executable_paths"] != []
        or code["untracked_executable_paths"] != []
    ):
        raise ValueError("Run Bundle executable source is not publication-eligible")
    revision = _require_full_git_revision(
        code["git_revision"], "Run Bundle Git revision"
    )
    executable_hashes = _require_hash_mapping(
        code["executable_path_hashes"], "executable_path_hashes"
    )
    committed_hashes = _require_hash_mapping(
        code["committed_executable_path_hashes"],
        "committed_executable_path_hashes",
    )
    if executable_hashes != committed_hashes:
        raise ValueError("Run Bundle executable bytes differ from committed bytes")
    for path in executable_hashes:
        _repository_relative_selector(
            path,
            "Run Bundle executable path selector",
        )
    aggregate = sha256_bytes(canonical_json_bytes(executable_hashes))
    if code["executable_source_tree_sha256"] != aggregate:
        raise ValueError("Run Bundle executable source aggregate digest is invalid")

    runtime = _require_exact_keys(
        manifest["runtime"],
        {"python", "platform", "do_mpc", "casadi", "numpy", "pandas"},
        "runtime",
    )
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("Run Bundle runtime manifest is incomplete")
    if runtime != _actual_runtime_manifest():
        raise ValueError("Run Bundle runtime does not match the actual runtime")

    input_hashes = _require_exact_keys(
        manifest["input_hashes"],
        {"scenario", "sources", "sidecars"},
        "input_hashes",
    )
    _require_full_sha256(input_hashes["scenario"], "input_hashes.scenario")
    _require_hash_mapping(input_hashes["sources"], "input_hashes.sources")
    _require_hash_mapping(input_hashes["sidecars"], "input_hashes.sidecars")
    expected_inputs = _scenario_input_hashes(_scenario_content(scenario))
    if input_hashes != expected_inputs:
        raise ValueError("Run Bundle scenario provenance/input hash graph is invalid")
    source_hashes = _require_hash_mapping(
        input_hashes["sources"], "input_hashes.sources"
    )
    sidecar_hashes = _require_hash_mapping(
        input_hashes["sidecars"], "input_hashes.sidecars"
    )
    _verify_git_hash_map(
        repository_root,
        revision,
        executable_hashes,
        "executable/configuration input",
    )
    _verify_git_hash_map(
        repository_root,
        revision,
        source_hashes,
        "Scenario source input",
    )
    _verify_git_hash_map(
        repository_root,
        revision,
        sidecar_hashes,
        "Scenario sidecar input",
    )

    specification_content: dict[str, JSONValue] = {
        "schema_version": RUN_SPECIFICATION_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "scenario": manifest["scenario"],
        "controller": manifest["controller"],
        "asset_capabilities": manifest["asset_capabilities"],
        "hub_configuration": {
            "capabilities": manifest["asset_capabilities"],
        },
        "evaluation_policy": manifest["evaluation_policy"],
        "code_provenance": manifest["code_provenance"],
        "runtime": manifest["runtime"],
        "input_hashes": manifest["input_hashes"],
    }
    if run_specification_identifier(specification_content) != manifest[
        "run_specification_identifier"
    ]:
        raise ValueError("Run Specification identifier does not match manifest content")
    return scenario, hub_configuration, policy, controller


def _controller_horizon_steps(controller: Mapping[str, object]) -> int:
    """Derive the causal forecast horizon from exact recorded controller metadata."""
    name = controller["name"]
    configuration = controller["configuration"]
    assert isinstance(configuration, Mapping)
    if name == "baseline":
        return 0
    if name != "mpc":
        raise ValueError(f"Run Bundle controller {name!r} is unsupported")
    horizon = configuration.get("horizon_steps")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(
            "Run Bundle MPC controller configuration requires a positive integer "
            "horizon_steps"
        )
    return horizon


def _expected_mpc_terminal_values(
    scenario: Scenario,
    operating_step: int,
    horizon_steps: int,
) -> tuple[float, float]:
    stage_points = scenario.points[
        operating_step : operating_step + horizon_steps
    ]
    if len(stage_points) != horizon_steps:
        raise ValueError("Run Bundle MPC forecast lacks exact recorded horizon coverage")
    electric_value = math.fsum(
        point.price_eur_per_kwh for point in stage_points
    ) / horizon_steps
    heating_prices = [
        point.price_eur_per_kwh
        for point in stage_points
        if point.outdoor_temperature_c < T_SETPOINT_C
    ]
    heat_value = (
        math.fsum(heating_prices) / len(heating_prices) / HP_COP
        if heating_prices
        else 0.0
    )
    return electric_value, heat_value


def _verify_bundle_directory(
    bundle_path: Path,
    expected_identifier: str | None,
    *,
    enforce_path_identifier: bool,
    repository_root: str | Path,
) -> RunBundle:
    from greenhouse_energy_hub.simulation import (
        DecisionDiagnostics,
        OperatingRecord,
        ValidRun,
        ValidationReport,
    )

    if not bundle_path.is_dir() or bundle_path.is_symlink():
        raise ValueError("Run Bundle path must be a real directory")
    expected_names = {"manifest.json", *RUN_BUNDLE_MEMBERS}
    observed_names = {child.name for child in bundle_path.iterdir()}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unknown = sorted(observed_names - expected_names)
        raise ValueError(
            f"Run Bundle member set mismatch; missing={missing}, unknown={unknown}"
        )
    if any(
        not (bundle_path / name).is_file() or (bundle_path / name).is_symlink()
        for name in expected_names
    ):
        raise ValueError("Run Bundle members must be regular files, not symlinks")

    manifest = _read_canonical_json(bundle_path / "manifest.json", "manifest")
    required_manifest_keys = {
        "schema_version",
        "canonicalization_version",
        "run_specification_identifier",
        "scenario",
        "controller",
        "asset_capabilities",
        "evaluation_policy",
        "code_provenance",
        "runtime",
        "input_hashes",
        "member_hashes",
        "valid",
        "run_bundle_identifier",
    }
    if set(manifest) != required_manifest_keys:
        raise ValueError("Run Bundle manifest schema keys are incomplete or unknown")
    if manifest["schema_version"] != RUN_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Run Bundle schema version is unsupported")
    if manifest["canonicalization_version"] != CANONICALIZATION_VERSION:
        raise ValueError("Run Bundle canonicalization version is unsupported")
    if manifest["valid"] is not True:
        raise ValueError("Run Bundle manifest must declare valid=true")
    bundle_identifier = _require_full_sha256(
        manifest["run_bundle_identifier"],
        "Run Bundle identifier",
    )
    specification_identifier = _require_full_sha256(
        manifest["run_specification_identifier"],
        "Run Specification identifier",
    )
    if expected_identifier is not None:
        _require_full_sha256(expected_identifier, "expected Run Bundle identifier")
        if bundle_identifier != expected_identifier:
            raise ValueError("Run Bundle identifier does not match requested full ID")
    if enforce_path_identifier and not bundle_path.name.endswith(
        f"--{bundle_identifier}"
    ):
        raise ValueError("Run Bundle directory must end with its full identifier")

    member_hashes = _require_hash_mapping(
        manifest["member_hashes"],
        "member_hashes",
    )
    if set(member_hashes) != set(RUN_BUNDLE_MEMBERS):
        raise ValueError("Run Bundle member_hashes must cover every exact member")
    for name in RUN_BUNDLE_MEMBERS:
        observed = sha256_bytes((bundle_path / name).read_bytes())
        if observed != member_hashes[name]:
            raise ValueError(f"Run Bundle member hash mismatch: {name}")

    identity_manifest = dict(manifest)
    del identity_manifest["run_bundle_identifier"]
    computed_identifier = sha256_bytes(canonical_json_bytes(identity_manifest))
    if computed_identifier != bundle_identifier:
        raise ValueError("Run Bundle identifier does not match canonical manifest")
    input_hash_preflight = manifest["input_hashes"]
    if isinstance(input_hash_preflight, dict):
        _require_full_sha256(
            input_hash_preflight.get("scenario"),
            "input_hashes.scenario",
        )
    scenario, hub_configuration, policy, controller = _validate_identity_graph(
        manifest,
        repository_root,
    )
    controller_horizon_steps = _controller_horizon_steps(controller)
    if controller_horizon_steps > scenario.forecast_horizon_capacity_steps:
        raise ValueError("Run Bundle controller horizon exceeds Scenario coverage")

    validation = _read_canonical_json(
        bundle_path / "validation.json",
        "validation",
    )
    _require_exact_keys(
        validation,
        {
            "schema_version",
            "complete",
            "valid",
            "checked_operating_steps",
            "issues",
            "steps",
        },
        "validation",
    )
    if (
        validation["schema_version"] != RUN_VALIDATION_SCHEMA_VERSION
        or validation["complete"] is not True
        or validation["valid"] is not True
        or validation["issues"] != []
    ):
        raise ValueError("Run Bundle validation evidence must be complete and valid")
    checked = validation["checked_operating_steps"]
    steps = validation["steps"]
    if (
        isinstance(checked, bool)
        or not isinstance(checked, int)
        or checked != scenario.operating_step_count
        or not isinstance(steps, list)
        or len(steps) != checked
    ):
        raise ValueError("Run Bundle validation step evidence is incomplete")

    trajectory_header, trajectory_rows = _csv_rows(
        bundle_path / "trajectory.csv",
        "trajectory.csv",
    )
    if trajectory_header != list(TRAJECTORY_COLUMNS):
        raise ValueError("Run Bundle trajectory.csv header/columns schema is not exact")
    _verify_operating_steps(trajectory_rows, checked, "trajectory.csv")

    diagnostics_header, diagnostic_rows = _csv_rows(
        bundle_path / "controller_diagnostics.csv",
        "controller_diagnostics.csv",
    )
    if diagnostics_header != list(DIAGNOSTIC_COLUMNS):
        raise ValueError(
            "Run Bundle controller_diagnostics.csv header/columns schema is not exact"
        )
    _verify_operating_steps(
        diagnostic_rows,
        checked,
        "controller_diagnostics.csv",
    )

    records = []
    diagnostics_values = []
    for operating_step, (trajectory, diagnostic) in enumerate(
        zip(trajectory_rows, diagnostic_rows, strict=True)
    ):
        timestamp = _parse_utc_z_text(
            trajectory["timestamp_utc"],
            f"trajectory[{operating_step}].timestamp_utc",
        )
        numeric = {
            key: _parse_csv_float(
                trajectory[key],
                f"trajectory[{operating_step}].{key}",
            )
            for key in TRAJECTORY_COLUMNS
            if key not in {"operating_step", "timestamp_utc"}
        }
        record = OperatingRecord(
            operating_step=operating_step,
            timestamp_utc=timestamp,
            start_state=HubState(
                numeric["start_soc_battery_kwh"],
                numeric["start_soc_hydrogen_kg"],
                numeric["start_soc_thermal_kwh"],
                numeric["start_indoor_temperature_c"],
            ),
            control=HubControl(
                numeric["battery_charge_kw"],
                numeric["battery_discharge_kw"],
                numeric["electrolyser_kw"],
                numeric["fuel_cell_kw"],
                numeric["heat_pump_kw"],
                numeric["electric_boiler_kw"],
                numeric["thermal_charge_kw"],
                numeric["thermal_discharge_kw"],
                numeric["ventilation_fraction"],
            ),
            exogenous=ExogenousInputs(
                numeric["pv_kw"],
                numeric["electric_load_kw"],
                numeric["price_eur_per_kwh"],
                numeric["outdoor_temperature_c"],
                numeric["irradiance_w_per_m2"],
            ),
            reached_state=HubState(
                numeric["reached_soc_battery_kwh"],
                numeric["reached_soc_hydrogen_kg"],
                numeric["reached_soc_thermal_kwh"],
                numeric["reached_indoor_temperature_c"],
            ),
            flows=HubFlows(
                numeric["grid_kw"],
                numeric["generated_heat_kw"],
                numeric["heat_to_air_kw"],
                numeric["thermal_charge_margin_kw"],
                numeric["hydrogen_production_kg_per_h"],
                numeric["hydrogen_consumption_kg_per_h"],
            ),
        )
        diagnostics = DecisionDiagnostics(
            adapter=diagnostic["adapter"],
            decision_status=diagnostic["decision_status"],
            solver_success=_parse_optional_csv_bool(
                diagnostic["solver_success"],
                f"diagnostics[{operating_step}].solver_success",
            ),
            solver_return_status=(
                diagnostic["solver_return_status"] or None
            ),
            solver_iterations=_parse_optional_csv_int(
                diagnostic["solver_iterations"],
                f"diagnostics[{operating_step}].solver_iterations",
            ),
            solver_wall_seconds=_parse_optional_csv_float(
                diagnostic["solver_wall_seconds"],
                f"diagnostics[{operating_step}].solver_wall_seconds",
            ),
            forecast_start_utc=_parse_optional_csv_timestamp(
                diagnostic["forecast_start_utc"],
                f"diagnostics[{operating_step}].forecast_start_utc",
            ),
            forecast_end_utc=_parse_optional_csv_timestamp(
                diagnostic["forecast_end_utc"],
                f"diagnostics[{operating_step}].forecast_end_utc",
            ),
            terminal_electric_value_eur_per_kwh=_parse_optional_csv_float(
                diagnostic["terminal_electric_value_eur_per_kwh"],
                f"diagnostics[{operating_step}].terminal_electric_value",
            ),
            terminal_heat_value_eur_per_kwhth=_parse_optional_csv_float(
                diagnostic["terminal_heat_value_eur_per_kwhth"],
                f"diagnostics[{operating_step}].terminal_heat_value",
            ),
        )
        if diagnostics.adapter != controller["name"]:
            raise ValueError("Run Bundle diagnostics adapter/controller mismatch")
        if diagnostics.decision_status != "success":
            raise ValueError("Run Bundle contains unsuccessful controller status")
        solver_fields = (
            diagnostics.solver_success,
            diagnostics.solver_return_status,
            diagnostics.solver_iterations,
            diagnostics.solver_wall_seconds,
        )
        if controller["name"] == "mpc":
            if (
                diagnostics.solver_success is not True
                or not diagnostics.solver_return_status
                or diagnostics.solver_iterations is None
                or diagnostics.solver_wall_seconds is None
            ):
                raise ValueError("Run Bundle MPC diagnostics lack solver evidence")
        elif controller["name"] == "baseline" and any(
            item is not None for item in solver_fields
        ):
            raise ValueError("Run Bundle Baseline diagnostics claim solver evidence")
        if diagnostics.forecast_start_utc != scenario.points[operating_step].timestamp_utc:
            raise ValueError("Run Bundle diagnostics forecast start is inconsistent")
        expected_forecast_end = (
            diagnostics.forecast_start_utc
            + controller_horizon_steps * scenario.step_duration
        )
        if diagnostics.forecast_end_utc != expected_forecast_end:
            raise ValueError(
                "Run Bundle diagnostics forecast end does not match the recorded "
                "controller horizon"
            )
        if controller["name"] == "baseline":
            if (
                diagnostics.terminal_electric_value_eur_per_kwh is not None
                or diagnostics.terminal_heat_value_eur_per_kwhth is not None
            ):
                raise ValueError(
                    "Run Bundle Baseline diagnostics must not contain terminal "
                    "coefficients"
                )
        else:
            expected_electric, expected_heat = _expected_mpc_terminal_values(
                scenario,
                operating_step,
                controller_horizon_steps,
            )
            observed_electric = diagnostics.terminal_electric_value_eur_per_kwh
            observed_heat = diagnostics.terminal_heat_value_eur_per_kwhth
            if (
                observed_electric is None
                or not math.isclose(
                    observed_electric,
                    expected_electric,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or observed_heat is None
                or not math.isclose(
                    observed_heat,
                    expected_heat,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Run Bundle MPC terminal coefficients do not match the exact "
                    "recorded Scenario stage points"
                )
        records.append(record)
        diagnostics_values.append(diagnostics)

    initial = records[0].start_state
    terminal = records[-1].reached_state
    reconstructed = ValidRun(
        scenario=scenario,
        controller_name=controller["name"],
        controller_configuration=controller["configuration"],
        capability_policy=controller["capability_policy"],
        hub_configuration=hub_configuration,
        initial_state=initial,
        records=tuple(records),
        controller_diagnostics=tuple(diagnostics_values),
        terminal_state=terminal,
        validation=ValidationReport(True, True, checked, ()),
    )
    _validate_publication_run_evidence(reconstructed)
    _validate_record_sequence(reconstructed)
    expected_validation = _validation_content(reconstructed)
    expected_validation_bytes = canonical_json_bytes(
        _to_json_primitives(expected_validation, "recomputed validation")
    )
    if (bundle_path / "validation.json").read_bytes() != expected_validation_bytes:
        raise ValueError(
            "Run Bundle validation types/evidence are not canonical bytes derived "
            "from records and physics"
        )

    for expected, (stored_step, diagnostics) in enumerate(
        zip(steps, diagnostics_values, strict=True)
    ):
        exact_step = _require_exact_keys(
            stored_step,
            {
                "operating_step",
                "decision_status",
                "solver_success",
                "solver_return_status",
                "control_valid",
                "flows_valid",
                "successor_valid",
                "physical_invariants_valid",
            },
            f"validation.steps[{expected}]",
        )
        if (
            exact_step["operating_step"] != expected
            or exact_step["decision_status"] != diagnostics.decision_status
            or exact_step["solver_success"] is not diagnostics.solver_success
            or exact_step["solver_return_status"] != diagnostics.solver_return_status
        ):
            raise ValueError("Run Bundle validation/controller evidence is inconsistent")

    summary = _read_canonical_json(bundle_path / "summary.json", "summary")
    expected_summary = _to_json_primitives(
        _summary_content(evaluate_run(reconstructed, policy)),
        "summary",
    )
    if (bundle_path / "summary.json").read_bytes() != canonical_json_bytes(
        expected_summary
    ):
        raise ValueError(
            "Run Bundle summary types/content do not match canonical recomputed "
            "EvaluationReport bytes"
        )
    return RunBundle(
        identifier=bundle_identifier,
        specification_identifier=specification_identifier,
        path=bundle_path,
        manifest=manifest,
    )


def verify_run_bundle(
    bundle_path: str | Path,
    expected_identifier: str | None = None,
    *,
    repository_root: str | Path,
) -> RunBundle:
    """Verify an authoritative full-ID Run Bundle and return immutable metadata."""
    return _verify_bundle_directory(
        Path(bundle_path),
        expected_identifier,
        enforce_path_identifier=True,
        repository_root=repository_root,
    )


def _slug(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    return slug or "unnamed"


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_noreplace_directory(source: str | Path, destination: str | Path) -> None:
    """Atomically publish a directory only if no directory entry claims the name."""
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    library_name = ctypes.util.find_library("c")
    if not library_name:
        raise RuntimeError("atomic no-replace publication is unavailable: libc not found")
    libc = ctypes.CDLL(library_name, use_errno=True)
    system = platform_module.system()
    if system == "Darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif system == "Linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x1)
    else:
        raise RuntimeError(
            f"atomic no-replace directory rename is unavailable on {system}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    if error_number in {errno.ENOSYS, errno.ENOTSUP, errno.EINVAL}:
        raise RuntimeError(
            "atomic no-replace directory rename is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _remove_owned_temporary(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _assert_fresh_publication_inputs(
    specification: RunSpecification,
    repository_root: str | Path,
) -> None:
    actual_runtime = _actual_runtime_manifest()
    if _to_json_primitives(specification.runtime) != actual_runtime:
        raise ValueError("Run Specification runtime does not match the actual runtime")
    recorded = _to_json_primitives(specification.code_provenance)
    assert isinstance(recorded, dict)
    path_hashes = recorded.get("executable_path_hashes")
    if not isinstance(path_hashes, dict) or not path_hashes:
        raise ValueError("Run Specification executable path map is incomplete")
    current = collect_code_provenance(
        tuple(sorted(path_hashes)),
        repository_root=repository_root,
    )
    if current != recorded or current.get("publication_eligible") is not True:
        raise ValueError(
            "Run Specification executable source changed or became dirty after capture"
        )
    revision = _require_full_git_revision(
        recorded.get("git_revision"),
        "Run Specification Git revision",
    )
    input_hashes = _to_json_primitives(specification.input_hashes)
    if not isinstance(input_hashes, dict):
        raise ValueError("Run Specification input hashes are incomplete")
    for field_name, description in (
        ("sources", "Scenario source input"),
        ("sidecars", "Scenario sidecar input"),
    ):
        hashes = _require_hash_mapping(
            input_hashes.get(field_name),
            f"input_hashes.{field_name}",
        )
        for relative_path, expected_hash in hashes.items():
            normalized = _repository_relative_path(
                relative_path,
                Path(repository_root),
            )
            if normalized != relative_path:
                raise ValueError(f"{description} path selector is not exact")
            working_bytes = (Path(repository_root).resolve() / relative_path).read_bytes()
            committed_bytes = _git_object_bytes(
                repository_root,
                revision,
                relative_path,
                description,
            )
            if (
                sha256_bytes(working_bytes) != expected_hash
                or sha256_bytes(committed_bytes) != expected_hash
                or working_bytes != committed_bytes
            ):
                raise ValueError(
                    f"{description} changed or became dirty after capture: "
                    f"{relative_path}"
                )


def _verified_collision_winner(
    destination: Path,
    bundle_identifier: str,
    all_bytes: Mapping[str, bytes],
    repository_root: str | Path,
) -> RunBundle:
    if destination.is_symlink() or not destination.is_dir():
        raise BundleCollisionError(
            f"Run Bundle destination is claimed by a non-bundle entry: {destination}"
        )
    try:
        existing = verify_run_bundle(
            destination,
            bundle_identifier,
            repository_root=repository_root,
        )
    except (OSError, ValueError) as exc:
        raise BundleCollisionError(
            f"existing Run Bundle {bundle_identifier} failed verification"
        ) from exc
    if any(
        (destination / name).read_bytes() != data
        for name, data in all_bytes.items()
    ):
        raise BundleCollisionError(
            f"Run Bundle identifier collision at {destination}"
        )
    return existing


def _assert_specification_matches_run(
    run: object,
    policy: EvaluationPolicy,
    specification: RunSpecification,
) -> None:
    if run_specification_identifier(specification.canonical_content) != specification.identifier:
        raise ValueError("Run Specification identifier is internally inconsistent")
    expected_scenario = _scenario_content(run.scenario)
    expected_controller = _to_json_primitives(
        {
            "name": run.controller_name,
            "configuration": run.controller_configuration,
            "capability_policy": run.capability_policy,
        }
    )
    expected_capabilities = _to_json_primitives(
        asdict(run.hub_configuration.capabilities)
    )
    expected_hub = _to_json_primitives(asdict(run.hub_configuration))
    expected_policy = _to_json_primitives(policy.to_serializable_metadata())
    comparisons = {
        "scenario": expected_scenario,
        "controller": expected_controller,
        "asset_capabilities": expected_capabilities,
        "hub_configuration": expected_hub,
        "evaluation_policy": expected_policy,
        "input_hashes": _scenario_input_hashes(expected_scenario),
    }
    for key, expected in comparisons.items():
        actual = _to_json_primitives(specification.canonical_content[key])
        if actual != expected:
            raise ValueError(f"Run Specification {key} does not match ValidRun")


def create_run_bundle(
    run: ValidRun,
    report: EvaluationReport,
    specification: RunSpecification,
    target_root: str | Path,
    *,
    repository_root: str | Path,
) -> RunBundle:
    """Persist one verified Valid Run through an owned atomic temporary directory."""
    from greenhouse_energy_hub.simulation import ValidRun

    if not isinstance(run, ValidRun):
        raise TypeError("create_run_bundle accepts only ValidRun")
    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    if not isinstance(specification, RunSpecification):
        raise TypeError("specification must be a RunSpecification")
    if not specification.publication_eligible:
        raise ValueError("Run Specification is not publication-eligible")
    provenance = specification.code_provenance
    if provenance.get("publication_eligible") is not True:
        raise ValueError("Run Specification records dirty executable source")
    _assert_fresh_publication_inputs(specification, repository_root)
    _assert_specification_matches_run(run, report.policy, specification)
    members = serialize_valid_run(run, report)
    member_hashes = {
        name: sha256_bytes(data) for name, data in members.items()
    }
    identity_manifest: dict[str, JSONValue] = {
        "schema_version": RUN_BUNDLE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "run_specification_identifier": specification.identifier,
        "scenario": _to_json_primitives(specification.scenario),
        "controller": _to_json_primitives(specification.controller),
        "asset_capabilities": _to_json_primitives(
            specification.asset_capabilities
        ),
        "evaluation_policy": _to_json_primitives(
            specification.evaluation_policy
        ),
        "code_provenance": _to_json_primitives(specification.code_provenance),
        "runtime": _to_json_primitives(specification.runtime),
        "input_hashes": _to_json_primitives(specification.input_hashes),
        "member_hashes": member_hashes,
        "valid": True,
    }
    bundle_identifier = sha256_bytes(canonical_json_bytes(identity_manifest))
    manifest = {**identity_manifest, "run_bundle_identifier": bundle_identifier}
    all_bytes = {**members, "manifest.json": canonical_json_bytes(manifest)}

    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Run Bundle target root must be a real directory")
    destination = root / (
        f"{_slug(run.scenario.name)}--{_slug(run.controller_name)}--"
        f"{bundle_identifier}"
    )
    temporary = root / f".tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in RUN_BUNDLE_MEMBERS:
            _write_fsynced(temporary / name, all_bytes[name])
        _write_fsynced(temporary / "manifest.json", all_bytes["manifest.json"])
        _fsync_directory(temporary)
        candidate = _verify_bundle_directory(
            temporary,
            bundle_identifier,
            enforce_path_identifier=False,
            repository_root=repository_root,
        )
        assert candidate.identifier == bundle_identifier

        if os.path.lexists(destination):
            return _verified_collision_winner(
                destination,
                bundle_identifier,
                all_bytes,
                repository_root,
            )

        try:
            _atomic_noreplace_directory(temporary, destination)
        except OSError as exc:
            if not os.path.lexists(destination):
                raise
            return _verified_collision_winner(
                destination,
                bundle_identifier,
                all_bytes,
                repository_root,
            )
        _fsync_directory(root)
        return verify_run_bundle(
            destination,
            bundle_identifier,
            repository_root=repository_root,
        )
    finally:
        _remove_owned_temporary(temporary)


def load_run_bundle(
    target_root: str | Path,
    identifier: str,
    *,
    repository_root: str | Path,
) -> RunBundle:
    """Resolve only one exact full authoritative Run Bundle identifier."""
    _require_full_sha256(identifier, "requested Run Bundle identifier")
    root = Path(target_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Run Bundle root does not exist: {root}")
    suffix = f"--{identifier}"
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.endswith(suffix)
    )
    if not candidates:
        raise KeyError(f"unknown Run Bundle identifier: {identifier}")
    if len(candidates) != 1:
        raise BundleCollisionError(
            f"multiple Run Bundle directories claim identifier {identifier}"
        )
    return verify_run_bundle(
        candidates[0],
        identifier,
        repository_root=repository_root,
    )


def write_failure_diagnostics(
    run: InvalidRun,
    specification: RunSpecification,
    diagnostics_root: str | Path,
    *,
    policy: EvaluationPolicy,
) -> Path:
    """Persist InvalidRun evidence outside the authoritative Run Bundle namespace."""
    from greenhouse_energy_hub.simulation import InvalidRun

    if not isinstance(run, InvalidRun):
        raise TypeError("write_failure_diagnostics accepts only InvalidRun")
    if not isinstance(specification, RunSpecification):
        raise TypeError("specification must be a RunSpecification")
    if not isinstance(policy, EvaluationPolicy):
        raise TypeError("policy must be an EvaluationPolicy")
    _assert_specification_matches_run(run, policy, specification)
    root = Path(diagnostics_root)
    specification_root = root / specification.identifier
    specification_root.mkdir(parents=True, exist_ok=True)
    while True:
        attempt_timestamp = datetime.now(timezone.utc)
        attempt = specification_root / attempt_timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            continue
    try:
        failure = {
            "schema_version": "run-failure-diagnostics-v1",
            "run_specification_identifier": specification.identifier,
            "recorded_at_utc": _utc_z(attempt_timestamp),
            "scenario": run.scenario.name,
            "controller": run.controller_name,
            "failed_step": run.failed_step,
            "failure_code": run.failure_code,
            "message": run.message,
        }
        diagnostics = [
            _diagnostic_row(index, item)
            for index, item in enumerate(run.controller_diagnostics)
        ]
        partial_trajectory = [
            _trajectory_row(record) for record in run.partial_records
        ]
        specification_content = {
            "identifier": specification.identifier,
            "publication_eligible": specification.publication_eligible,
            "canonical_content": _to_json_primitives(
                specification.canonical_content
            ),
        }
        _write_fsynced(
            attempt / "failure.json",
            canonical_json_bytes(_to_json_primitives(failure)),
        )
        _write_fsynced(
            attempt / "controller_diagnostics.csv",
            _csv_bytes(DIAGNOSTIC_COLUMNS, diagnostics),
        )
        _write_fsynced(
            attempt / "partial_trajectory.csv",
            _csv_bytes(TRAJECTORY_COLUMNS, partial_trajectory),
        )
        _write_fsynced(
            attempt / "run_specification.json",
            canonical_json_bytes(_to_json_primitives(specification_content)),
        )
        _fsync_directory(attempt)
        _fsync_directory(specification_root)
        return attempt
    except BaseException:
        if attempt.exists():
            shutil.rmtree(attempt)
        raise


# Compatibility wrappers retained until the structural migration removes the
# DataFrame-facing entry points.  New evaluation code uses the stable interfaces.
def stored_equiv_kwh(
    soc_bat: float,
    soc_h2: float,
    soc_tes: float,
    config: HubConfiguration = HubConfiguration(),
) -> float:
    return recoverable_inventory_kwh(
        HubState(soc_bat, soc_h2, soc_tes, 0.0),
        config,
    )


def stored_equiv_from_row(
    row: Mapping[str, object],
    config: HubConfiguration = HubConfiguration(),
) -> float:
    return stored_equiv_kwh(
        float(row["SOC_bat_kWh"]),
        float(row["SOC_h2_kg"]),
        float(row["SOC_tes_kWh"]),
        config,
    )


def saving_pct(baseline_cost: float, comparison_cost: float) -> float:
    return saving_percent(baseline_cost, comparison_cost)


# ---------------------------------------------------------------------------
# Publication evidence
# ---------------------------------------------------------------------------

PUBLICATION_MANIFEST_SCHEMA_VERSION = "publication-manifest-v1"
PUBLICATION_CANDIDATE_KEYS = frozenset(
    {
        "ablation-full",
        "ablation-no-h2",
        "ablation-no-tes",
        "ablation-one-step",
        "summer-baseline",
        "summer-mpc",
        "winter-baseline",
        "winter-mpc",
    }
)
PUBLICATION_REQUIRED_SENSITIVITIES = frozenset({"0x", "1x", "2x"})
PUBLICATION_README_BEGIN = b"<!-- BEGIN GENERATED RESULTS: DO NOT EDIT -->"
PUBLICATION_README_END = b"<!-- END GENERATED RESULTS -->"
PUBLICATION_FIGURE_FILENAMES = frozenset(
    {
        "fig1_cumulative_cost.png",
        "fig2_grid_vs_price.png",
        "fig3_soc_trajectories.png",
        "fig4_temperature.png",
        "fig5_heat_shifting.png",
        "fig6_ablation.png",
    }
)
_PUBLICATION_FULL_ASSET_CAPABILITIES = {
    "battery": True,
    "hydrogen": True,
    "thermal_store": True,
}
_PUBLICATION_WINDOWS = {
    "winter": {
        "name": "winter-2023-14d",
        "operating_start_utc": "2023-01-01T23:00:00Z",
        "operating_end_utc": "2023-01-15T23:00:00Z",
        "forecast_end_utc": "2023-01-16T23:00:00Z",
    },
    "summer": {
        "name": "summer-2023-14d",
        "operating_start_utc": "2023-05-31T22:00:00Z",
        "operating_end_utc": "2023-06-14T22:00:00Z",
        "forecast_end_utc": "2023-06-15T22:00:00Z",
    },
}


@dataclass(frozen=True)
class PublicationEvidence:
    """Verified bundles and immutable publication recipe inputs."""

    candidates: Mapping[str, str]
    bundles: Mapping[str, RunBundle]
    summaries: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))
        object.__setattr__(self, "bundles", MappingProxyType(dict(self.bundles)))
        object.__setattr__(self, "summaries", MappingProxyType(dict(self.summaries)))


def _require_publication_identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a full lowercase SHA-256 identifier")
    return value


def _validated_publication_candidates(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != PUBLICATION_CANDIDATE_KEYS:
        raise ValueError("publication candidates have missing or extra stable keys")
    candidates = {
        key: _require_publication_identifier(
            value[key],
            f"publication candidate {key!r}",
        )
        for key in PUBLICATION_CANDIDATE_KEYS
    }
    if len(set(candidates.values())) != len(candidates):
        raise ValueError("publication candidates must pin distinct Run Bundle identifiers")
    return candidates


def _read_publication_candidate_index(candidate_index: str | Path) -> dict[str, str]:
    path = Path(candidate_index)
    if path.is_symlink() or not path.is_file():
        raise ValueError("publication candidates must be a regular JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication candidates must contain valid UTF-8 JSON") from exc
    return _validated_publication_candidates(payload)


def read_publication_candidates(
    candidate_index: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, str]:
    """Read generated candidates or reconstruct them from a committed recipe.

    Candidate indexes are intentionally ignored developer conveniences.  A clean
    checkout instead derives their exact stable keys and full identifiers from the
    tracked publication manifest.
    """
    path = Path(candidate_index)
    if path.exists() or path.is_symlink():
        return _read_publication_candidate_index(path)
    if manifest_path is None:
        raise ValueError("publication candidate index is absent and no manifest was supplied")
    return publication_candidates_from_manifest(read_publication_manifest(manifest_path))


def _publication_manifest_from_candidates(candidates: Mapping[str, str]) -> dict[str, object]:
    validated = _validated_publication_candidates(candidates)
    return {
        "schema_version": PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "comparisons": {
            "winter": {
                "baseline_bundle_id": validated["winter-baseline"],
                "mpc_bundle_id": validated["winter-mpc"],
            },
            "summer": {
                "baseline_bundle_id": validated["summer-baseline"],
                "mpc_bundle_id": validated["summer-mpc"],
            },
        },
        "ablations": {
            "full": validated["ablation-full"],
            "no-h2": validated["ablation-no-h2"],
            "no-tes": validated["ablation-no-tes"],
            "one-step": validated["ablation-one-step"],
        },
        "figures": {
            "fig1_cumulative_cost.png": [
                validated["winter-baseline"],
                validated["winter-mpc"],
            ],
            "fig2_grid_vs_price.png": [validated["winter-mpc"]],
            "fig3_soc_trajectories.png": [
                validated["winter-baseline"],
                validated["winter-mpc"],
            ],
            "fig4_temperature.png": [
                validated["winter-baseline"],
                validated["winter-mpc"],
            ],
            "fig5_heat_shifting.png": [validated["winter-mpc"]],
            "fig6_ablation.png": [
                validated["ablation-full"],
                validated["ablation-no-h2"],
                validated["ablation-no-tes"],
                validated["ablation-one-step"],
            ],
        },
    }


def publication_candidates_from_manifest(manifest: Mapping[str, object]) -> dict[str, str]:
    """Validate the strict recipe and return its stable candidate mapping."""
    if set(manifest) != {"schema_version", "comparisons", "ablations", "figures"}:
        raise ValueError("publication manifest has missing or extra top-level fields")
    if manifest["schema_version"] != PUBLICATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("publication manifest schema version is unsupported")
    comparisons = manifest["comparisons"]
    ablations = manifest["ablations"]
    figures = manifest["figures"]
    if not isinstance(comparisons, Mapping) or set(comparisons) != {"winter", "summer"}:
        raise ValueError("publication manifest comparisons are incomplete")
    if not isinstance(ablations, Mapping) or set(ablations) != {
        "full",
        "no-h2",
        "no-tes",
        "one-step",
    }:
        raise ValueError("publication manifest ablations are incomplete")
    if not isinstance(figures, Mapping) or set(figures) != PUBLICATION_FIGURE_FILENAMES:
        raise ValueError("publication manifest figures are incomplete")

    candidates: dict[str, str] = {}
    for season in ("winter", "summer"):
        comparison = comparisons[season]
        if not isinstance(comparison, Mapping) or set(comparison) != {
            "baseline_bundle_id",
            "mpc_bundle_id",
        }:
            raise ValueError(f"publication manifest {season} comparison is incomplete")
        candidates[f"{season}-baseline"] = _require_publication_identifier(
            comparison["baseline_bundle_id"],
            f"publication manifest {season} baseline",
        )
        candidates[f"{season}-mpc"] = _require_publication_identifier(
            comparison["mpc_bundle_id"],
            f"publication manifest {season} MPC",
        )
    for key in ("full", "no-h2", "no-tes", "one-step"):
        candidates[f"ablation-{key}"] = _require_publication_identifier(
            ablations[key],
            f"publication manifest ablation {key}",
        )

    candidates = _validated_publication_candidates(candidates)
    if dict(manifest) != _publication_manifest_from_candidates(candidates):
        raise ValueError("publication manifest does not match its pinned recipe")
    return candidates


def read_publication_manifest(path: str | Path) -> dict[str, object]:
    """Read a regular JSON manifest and validate its exact pinned recipe."""
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("publication manifest must be a regular JSON file")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication manifest must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("publication manifest must be a JSON object")
    publication_candidates_from_manifest(value)
    return value


def publication_bundle_ids(manifest: Mapping[str, object]) -> tuple[str, ...]:
    """Return each full bundle ID once, in recipe order, after validation."""
    return tuple(dict.fromkeys(publication_candidates_from_manifest(manifest).values()))


def validate_publication_summary(
    summary: Mapping[str, object],
    *,
    identifier: str,
) -> Mapping[str, object]:
    """Require nominal metrics plus the three policy wear sensitivities."""
    _require_publication_identifier(identifier, "Run Bundle")
    if not isinstance(summary.get("nominal"), Mapping):
        raise ValueError(f"Run Bundle {identifier} summary is missing nominal values")
    sensitivities = summary.get("wear_sensitivities")
    if not isinstance(sensitivities, Mapping) or set(sensitivities) != PUBLICATION_REQUIRED_SENSITIVITIES:
        raise ValueError(
            f"Run Bundle {identifier} is missing required 0x, 1x, and 2x sensitivities"
        )
    return summary


def _publication_summary_for(bundle: RunBundle) -> Mapping[str, object]:
    try:
        summary = json.loads((bundle.path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Run Bundle {bundle.identifier} has no readable summary") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Run Bundle {bundle.identifier} summary must be an object")
    return validate_publication_summary(summary, identifier=bundle.identifier)


def _publication_policy_for(bundle: RunBundle) -> Mapping[str, object]:
    policy = bundle.manifest.get("evaluation_policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"Run Bundle {bundle.identifier} has no Evaluation Policy")
    return policy


def validate_publication_bundles(bundles: Mapping[str, RunBundle]) -> None:
    """Check publication-specific provenance, sensitivity, and policy gates."""
    if not bundles:
        raise ValueError("publication requires at least one verified Run Bundle")
    policies: list[Mapping[str, object]] = []
    for key, bundle in bundles.items():
        _require_publication_identifier(bundle.identifier, f"Run Bundle for {key!r}")
        provenance = bundle.manifest.get("code_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Run Bundle {bundle.identifier} has no provenance")
        if provenance.get("publication_eligible") is not True:
            raise ValueError(f"Run Bundle {bundle.identifier} is not publication-eligible")
        if provenance.get("dirty_executable_paths") or provenance.get(
            "untracked_executable_paths"
        ):
            raise ValueError(f"Run Bundle {bundle.identifier} has dirty provenance")
        if bundle.manifest.get("valid") is not True:
            raise ValueError(f"Run Bundle {bundle.identifier} did not pass validation")
        _publication_summary_for(bundle)
        policies.append(_publication_policy_for(bundle))
    if any(policy != policies[0] for policy in policies[1:]):
        raise ValueError("publication comparisons use different Evaluation Policies")


def _publication_bundle_section(
    bundle: RunBundle,
    section: str,
) -> Mapping[str, object]:
    value = bundle.manifest.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"publication semantic role {bundle.identifier} lacks {section} metadata"
        )
    return value


def _validate_publication_candidate_semantics(
    bundles: Mapping[str, RunBundle],
) -> None:
    """Enforce the exact approved publication-role and causal comparison matrix."""
    if set(bundles) != PUBLICATION_CANDIDATE_KEYS:
        raise ValueError("publication semantic matrix has missing or extra stable roles")

    from greenhouse_energy_hub.controllers.baseline import (
        BASELINE_CAPABILITY_POLICY,
    )

    expected_roles: dict[str, tuple[str, str, int, dict[str, bool]]] = {
        "winter-baseline": (
            "winter",
            "baseline",
            0,
            _PUBLICATION_FULL_ASSET_CAPABILITIES,
        ),
        "winter-mpc": ("winter", "mpc", 24, _PUBLICATION_FULL_ASSET_CAPABILITIES),
        "summer-baseline": (
            "summer",
            "baseline",
            0,
            _PUBLICATION_FULL_ASSET_CAPABILITIES,
        ),
        "summer-mpc": ("summer", "mpc", 24, _PUBLICATION_FULL_ASSET_CAPABILITIES),
        "ablation-full": (
            "winter",
            "mpc",
            24,
            _PUBLICATION_FULL_ASSET_CAPABILITIES,
        ),
        "ablation-no-h2": (
            "winter",
            "mpc",
            24,
            {"battery": True, "hydrogen": False, "thermal_store": True},
        ),
        "ablation-no-tes": (
            "winter",
            "mpc",
            24,
            {"battery": True, "hydrogen": True, "thermal_store": False},
        ),
        "ablation-one-step": (
            "winter",
            "mpc",
            1,
            _PUBLICATION_FULL_ASSET_CAPABILITIES,
        ),
    }

    for key, (season, controller_name, horizon, capabilities) in expected_roles.items():
        bundle = bundles[key]
        scenario = _publication_bundle_section(bundle, "scenario")
        controller = _publication_bundle_section(bundle, "controller")
        assets = _publication_bundle_section(bundle, "asset_capabilities")
        expected_window = _PUBLICATION_WINDOWS[season]
        if any(scenario.get(field) != value for field, value in expected_window.items()):
            raise ValueError(
                f"publication semantic role {key!r} has the wrong Scenario/window"
            )
        if (
            scenario.get("operating_step_count") != 336
            or scenario.get("forecast_horizon_capacity_steps") != 24
            or scenario.get("step_duration_seconds") != 3600.0
        ):
            raise ValueError(
                f"publication semantic role {key!r} must have 336 operating steps "
                "and 24-step Forecast Coverage"
            )
        if controller.get("name") != controller_name:
            raise ValueError(
                f"publication semantic role {key!r} has the wrong controller"
            )
        if _controller_horizon_steps(controller) != horizon:
            raise ValueError(
                f"publication semantic role {key!r} has the wrong controller horizon"
            )
        if dict(assets) != capabilities:
            raise ValueError(
                f"publication semantic role {key!r} has the wrong Asset capabilities"
            )
        capability_policy = controller.get("capability_policy")
        if controller_name == "baseline":
            if capability_policy != BASELINE_CAPABILITY_POLICY:
                raise ValueError(
                    f"publication semantic role {key!r} has the wrong Baseline "
                    "capability policy"
                )
        elif capability_policy != capabilities:
            raise ValueError(
                f"publication semantic role {key!r} has inconsistent MPC capabilities"
            )

    scenarios = {
        key: _publication_bundle_section(bundle, "scenario")
        for key, bundle in bundles.items()
    }
    if scenarios["winter-baseline"] != scenarios["winter-mpc"]:
        raise ValueError("publication winter comparison Scenarios are not identical")
    if scenarios["summer-baseline"] != scenarios["summer-mpc"]:
        raise ValueError("publication summer comparison Scenarios are not identical")
    ablation_scenarios = [
        scenarios[f"ablation-{variant}"]
        for variant in ("full", "no-h2", "no-tes", "one-step")
    ]
    if any(scenario != ablation_scenarios[0] for scenario in ablation_scenarios[1:]):
        raise ValueError("publication ablation Scenarios are not identical")
    if scenarios["winter-mpc"] != scenarios["ablation-full"]:
        raise ValueError(
            "publication direct/full winter MPC Scenarios are not semantically equal"
        )

    full = bundles["ablation-full"].manifest
    full_controller = _publication_bundle_section(bundles["ablation-full"], "controller")
    full_configuration = full_controller["configuration"]
    full_policy = full["evaluation_policy"]
    for variant in ("no-h2", "no-tes"):
        candidate = bundles[f"ablation-{variant}"].manifest
        controller = _publication_bundle_section(
            bundles[f"ablation-{variant}"], "controller"
        )
        if (
            controller["configuration"] != full_configuration
            or candidate["evaluation_policy"] != full_policy
        ):
            raise ValueError(
                f"publication ablation {variant!r} differs from full beyond its "
                "named capability"
            )

    one_step = bundles["ablation-one-step"].manifest
    one_step_controller = _publication_bundle_section(
        bundles["ablation-one-step"], "controller"
    )
    normalized_one_step_configuration = dict(one_step_controller["configuration"])
    normalized_one_step_configuration["horizon_steps"] = 24
    if (
        normalized_one_step_configuration != full_configuration
        or one_step_controller["capability_policy"]
        != full_controller["capability_policy"]
        or one_step["asset_capabilities"] != full["asset_capabilities"]
        or one_step["evaluation_policy"] != full_policy
    ):
        raise ValueError(
            "publication one-step ablation differs from full beyond controller horizon"
        )

    direct = bundles["winter-mpc"].manifest
    if any(
        direct[field] != full[field]
        for field in (
            "scenario",
            "controller",
            "asset_capabilities",
            "evaluation_policy",
        )
    ):
        raise ValueError(
            "publication direct/full winter MPC evidence is not semantically equal"
        )


def _load_verified_publication_candidates(
    candidates: Mapping[str, str],
    *,
    runs_root: str | Path,
    repository_root: str | Path,
) -> dict[str, RunBundle]:
    bundles: dict[str, RunBundle] = {}
    for key, identifier in _validated_publication_candidates(candidates).items():
        loaded = load_run_bundle(runs_root, identifier, repository_root=repository_root)
        verified = verify_run_bundle(
            loaded.path,
            expected_identifier=identifier,
            repository_root=repository_root,
        )
        if verified.identifier != identifier:
            raise ValueError(f"Run Bundle for {key!r} did not retain its requested ID")
        bundles[key] = verified
    return bundles


def build_publication_manifest(
    candidate_index: str | Path,
    *,
    runs_root: str | Path,
    repository_root: str | Path,
) -> dict[str, object]:
    """Build a recipe only after every candidate verifies as a Run Bundle."""
    candidates = _read_publication_candidate_index(candidate_index)
    bundles = _load_verified_publication_candidates(
        candidates,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    _validate_publication_candidate_semantics(bundles)
    validate_publication_bundles(bundles)
    return _publication_manifest_from_candidates(candidates)


def load_verified_publication_evidence(
    manifest: Mapping[str, object],
    *,
    runs_root: str | Path,
    repository_root: str | Path,
) -> PublicationEvidence:
    """Validate the recipe before reading and verifying each pinned bundle."""
    candidates = publication_candidates_from_manifest(manifest)
    bundles = _load_verified_publication_candidates(
        candidates,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    _validate_publication_candidate_semantics(bundles)
    validate_publication_bundles(bundles)
    return PublicationEvidence(
        candidates=candidates,
        bundles=bundles,
        summaries={key: _publication_summary_for(bundle) for key, bundle in bundles.items()},
    )


def validate_committed_publication_bundles(
    manifest: Mapping[str, object],
    *,
    runs_root: str | Path,
    repository_root: str | Path,
) -> None:
    """Require every member of every pinned Run Bundle in the proposed commit."""
    evidence = load_verified_publication_evidence(
        manifest,
        runs_root=runs_root,
        repository_root=repository_root,
    )
    repository = Path(repository_root)
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    )
    required = {
        member.relative_to(repository).as_posix()
        for bundle in evidence.bundles.values()
        for member in bundle.path.iterdir()
    }
    missing = sorted(required - tracked)
    if missing:
        raise ValueError(
            "publication manifest references Run Bundles absent from the proposed commit: "
            + ", ".join(missing)
        )


def publication_metric(summary: Mapping[str, object], key: str) -> float:
    """Return a finite nominal metric after strict publication-summary validation."""
    nominal = summary.get("nominal")
    if not isinstance(nominal, Mapping):
        raise ValueError("summary is missing nominal values")
    value = nominal.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary metric {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"summary metric {key!r} must be finite")
    return result


def publication_cost_difference_percent(full_cost: float, variant_cost: float) -> float:
    """Signed cost difference from full MPC to a variant: (variant-full)/full."""
    if not math.isfinite(full_cost) or not math.isfinite(variant_cost):
        raise ValueError("publication costs must be finite")
    return 100.0 * (variant_cost - full_cost) / abs(full_cost) if full_cost else 0.0


def publication_comparison_rows(
    summaries: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, float | str], ...]:
    """Compute cost-saving and comfort rows for the seasonal controller comparison."""
    rows: list[dict[str, float | str]] = []
    for label, baseline_key, mpc_key in (
        ("Winter", "winter-baseline", "winter-mpc"),
        ("Summer", "summer-baseline", "summer-mpc"),
    ):
        baseline = summaries[baseline_key]
        mpc = summaries[mpc_key]
        baseline_cost = publication_metric(baseline, "inventory_adjusted_cost_eur")
        mpc_cost = publication_metric(mpc, "inventory_adjusted_cost_eur")
        rows.append(
            {
                "window": label,
                "baseline_inventory_adjusted_cost_eur": baseline_cost,
                "mpc_inventory_adjusted_cost_eur": mpc_cost,
                "saving_percent": saving_percent(baseline_cost, mpc_cost),
                "baseline_comfort_violation_c_h": publication_metric(
                    baseline, "comfort_violation_c_h"
                ),
                "mpc_comfort_violation_c_h": publication_metric(
                    mpc, "comfort_violation_c_h"
                ),
            }
        )
    return tuple(rows)


def publication_ablation_rows(
    summaries: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, float | str], ...]:
    """Compute signed ablation cost deltas and separate comfort evidence."""
    full_cost = publication_metric(summaries["ablation-full"], "inventory_adjusted_cost_eur")
    rows: list[dict[str, float | str]] = []
    for label, key in (
        ("MPC (full)", "ablation-full"),
        ("no H₂", "ablation-no-h2"),
        ("no thermal store", "ablation-no-tes"),
        ("one-step horizon", "ablation-one-step"),
    ):
        summary = summaries[key]
        cost = publication_metric(summary, "inventory_adjusted_cost_eur")
        rows.append(
            {
                "variant": label,
                "candidate_key": key,
                "inventory_adjusted_cost_eur": cost,
                "comfort_violation_c_h": publication_metric(
                    summary, "comfort_violation_c_h"
                ),
                "cost_difference_vs_full_percent": publication_cost_difference_percent(
                    full_cost, cost
                ),
            }
        )
    return tuple(rows)


def rewrite_publication_readme_block(
    readme_path: str | Path,
    *,
    candidates: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, object]],
) -> None:
    """Rewrite precisely the generated body, preserving every other README byte."""
    validated = _validated_publication_candidates(candidates)
    rows = publication_comparison_rows(summaries)
    ablations = publication_ablation_rows(summaries)
    comparison_lines = [
        "| Window | Baseline Inventory-Adjusted Cost | MPC Inventory-Adjusted Cost | Saving | Comfort Violation (baseline / MPC) |",
        "|--------|----------------------------------:|-----------------------------:|:------:|:-----------------------------------:|",
        *[
            f"| **{row['window']}** | €{row['baseline_inventory_adjusted_cost_eur']:,.0f} | "
            f"€{row['mpc_inventory_adjusted_cost_eur']:,.0f} | {row['saving_percent']:+.1f} % | "
            f"{row['baseline_comfort_violation_c_h']:.1f} / "
            f"{row['mpc_comfort_violation_c_h']:.1f} °C·h |"
            for row in rows
        ],
        "",
        "Pinned Run Bundle IDs:",
        f"- Winter baseline: `{validated['winter-baseline']}`",
        f"- Winter MPC: `{validated['winter-mpc']}`",
        f"- Summer baseline: `{validated['summer-baseline']}`",
        f"- Summer MPC: `{validated['summer-mpc']}`",
        "",
        "| Winter ablation | Inventory-Adjusted Cost | Comfort Violation | Cost difference vs full |",
        "|-----------------|--------------------------:|------------------:|:-----------------------:|",
        *[
            f"| {row['variant']} | €{row['inventory_adjusted_cost_eur']:,.0f} | "
            f"{row['comfort_violation_c_h']:,.1f} °C·h | "
            f"{row['cost_difference_vs_full_percent']:+.1f} % |"
            for row in ablations
        ],
        "",
        f"Removing hydrogen raises the winter inventory-adjusted cost relative to the full controller (`{validated['ablation-no-h2']}` versus `{validated['ablation-full']}`).",
        f"Removing the thermal store raises the winter inventory-adjusted cost relative to the full controller (`{validated['ablation-no-tes']}` versus `{validated['ablation-full']}`).",
        f"A one-step horizon has substantial comfort violation in this winter run (`{validated['ablation-one-step']}` versus `{validated['ablation-full']}`).",
    ]
    path = Path(readme_path)
    original = path.read_bytes()
    if original.count(PUBLICATION_README_BEGIN) != 1 or original.count(PUBLICATION_README_END) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    begin = original.index(PUBLICATION_README_BEGIN) + len(PUBLICATION_README_BEGIN)
    end = original.index(PUBLICATION_README_END)
    if end < begin:
        raise ValueError("README generated-results markers are out of order")
    replacement = ("\n" + "\n".join(comparison_lines) + "\n").encode("utf-8")
    path.write_bytes(original[:begin] + replacement + original[end:])


def load_publication_trajectory(bundle: RunBundle) -> object:
    """Load a verified trajectory lazily for a renderer or analysis notebook."""
    import pandas as pd

    return pd.read_csv(bundle.path / "trajectory.csv", parse_dates=["timestamp_utc"]).set_index(
        "timestamp_utc"
    )


def render_publication_figures(
    bundles: Mapping[str, RunBundle],
    *,
    figures_root: str | Path,
) -> None:
    """Render the six figures from verified publication evidence on demand only."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    validate_publication_bundles(bundles)
    figures = Path(figures_root)
    figures.mkdir(parents=True, exist_ok=True)
    trajectories = {key: load_publication_trajectory(bundle) for key, bundle in bundles.items()}
    summaries = {key: _publication_summary_for(bundle) for key, bundle in bundles.items()}
    baseline = trajectories["winter-baseline"]
    mpc = trajectories["winter-mpc"]
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    baseline_costs = pd.DataFrame(summaries["winter-baseline"]["step_line_items"])
    mpc_costs = pd.DataFrame(summaries["winter-mpc"]["step_line_items"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(baseline.index, baseline_costs["grid_cost_eur"].cumsum(), label="Baseline", lw=2, color="#b2182b")
    ax.plot(mpc.index, mpc_costs["grid_cost_eur"].cumsum(), label="MPC", lw=2, color="#2166ac")
    ax.set(ylabel="Cumulative grid cost [EUR]", title="Winter cumulative grid cost")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures / "fig1_cumulative_cost.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mpc.index, mpc["grid_kw"], color="#2166ac", lw=1.2, label="MPC grid power [kW]")
    ax.axhline(0, color="black", lw=0.7); ax.set_ylabel("Grid power [kW]")
    price_axis = ax.twinx(); price_axis.plot(mpc.index, mpc["price_eur_per_kwh"] * 100, color="#fdae61", alpha=0.75, label="Day-ahead price [ct/kWh]")
    price_axis.set_ylabel("Price [ct/kWh]"); ax.set_title("MPC grid exchange and day-ahead price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); fig.tight_layout(); fig.savefig(figures / "fig2_grid_vs_price.png"); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for axis, column, label in zip(axes, ("reached_soc_battery_kwh", "reached_soc_hydrogen_kg", "reached_soc_thermal_kwh"), ("Battery SOC [kWh]", "Hydrogen inventory [kg]", "Thermal-store SOC [kWh]"), strict=True):
        axis.plot(baseline.index, baseline[column], color="#b2182b", alpha=0.7, label="Baseline")
        axis.plot(mpc.index, mpc[column], color="#2166ac", label="MPC"); axis.set_ylabel(label); axis.legend(loc="best")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); fig.suptitle("Winter storage trajectories"); fig.tight_layout(); fig.savefig(figures / "fig3_soc_trajectories.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(16, 24, color="#1a9850", alpha=0.08, label="Comfort band")
    ax.plot(mpc.index, mpc["reached_indoor_temperature_c"], color="#2166ac", lw=1.3, label="MPC indoor")
    ax.plot(baseline.index, baseline["reached_indoor_temperature_c"], color="#b2182b", lw=1.0, alpha=0.7, label="Baseline indoor")
    ax.plot(mpc.index, mpc["outdoor_temperature_c"], color="grey", lw=0.9, alpha=0.7, label="Outdoor")
    ax.set(ylabel="Temperature [°C]", title="Winter greenhouse temperature"); ax.legend(loc="best"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); fig.tight_layout(); fig.savefig(figures / "fig4_temperature.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mpc.index, mpc["electric_boiler_kw"], color="#d73027", lw=1.2, label="Electric boiler [kW]")
    ax.plot(mpc.index, mpc["thermal_charge_kw"], color="#1a9850", lw=1.1, label="Thermal-store charge [kWth]")
    ax.set(ylabel="Power [kW]", title="MPC power-to-heat operation"); ax.legend(loc="upper left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); fig.tight_layout(); fig.savefig(figures / "fig5_heat_shifting.png"); plt.close(fig)

    ablations = publication_ablation_rows(summaries)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar([str(row["variant"]).replace(" horizon", "") for row in ablations], [float(row["cost_difference_vs_full_percent"]) for row in ablations], color=["#2166ac", "#7fb3d5", "#7fb3d5", "#7fb3d5"])
    ax.axhline(0, color="black", lw=0.7); ax.set_ylabel("Inventory-adjusted cost difference vs full [%]"); ax.set_title("Winter ablation comparison")
    for bar, row in zip(bars, ablations, strict=True):
        value = float(row["cost_difference_vs_full_percent"])
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.35 if value >= 0 else -1.0), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout(); fig.savefig(figures / "fig6_ablation.png"); plt.close(fig)


def regenerate_publication_artifacts(
    *,
    candidate_index: str | Path,
    manifest_path: str | Path,
    repository_root: str | Path,
    runs_root: str | Path,
    figures_root: str | Path,
    readme_path: str | Path,
) -> dict[str, object]:
    """Write a verified recipe, figures, and marker-limited README evidence."""
    manifest = build_publication_manifest(candidate_index, runs_root=runs_root, repository_root=repository_root)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    evidence = load_verified_publication_evidence(manifest, runs_root=runs_root, repository_root=repository_root)
    render_publication_figures(evidence.bundles, figures_root=figures_root)
    rewrite_publication_readme_block(readme_path, candidates=evidence.candidates, summaries=evidence.summaries)
    return manifest
