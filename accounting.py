"""Named evaluation policy and reproducible Run-level economic scorecards.

Realized evaluation is intentionally independent of the MPC's solver-only
regularization.  Every Controller is scored from immutable Operating Records by
the same named policy; serialized trajectory columns are compatibility output,
not an accounting input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
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

from models.hub_model import (
    E_H2_LHV_KWH_KG,
    ETA_BAT_DIS,
    ETA_FC_E,
    HP_COP,
    T_MAX_C,
    T_MIN_C,
    HubConfiguration,
    HubState,
)

if TYPE_CHECKING:
    from control.rolling_horizon import (
        InvalidRun,
        OperatingRecord,
        ValidRun,
    )


PROVISIONAL_COEFFICIENT_STATUS = "provisional"
SETTLEMENT_RULE = "arithmetic-mean-operating-wholesale-price"
RUN_SPECIFICATION_SCHEMA_VERSION = "run-specification-v1"
RUN_BUNDLE_SCHEMA_VERSION = "run-bundle-v1"
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
    from control.rolling_horizon import ValidRun

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


def _normalized_runtime(
    runtime: Mapping[str, object] | None,
) -> dict[str, JSONValue]:
    actual = _actual_runtime_manifest()
    if runtime is None:
        return actual
    normalized = _to_json_primitives(runtime, "runtime")
    if not isinstance(normalized, dict):
        raise TypeError("runtime must be a JSON object")
    if set(normalized) != set(actual):
        raise ValueError(
            "runtime must contain python, platform, do_mpc, casadi, numpy, and pandas"
        )
    if any(not isinstance(value, str) or not value for value in normalized.values()):
        raise ValueError("runtime versions must be nonempty strings")
    return normalized


def _repository_relative_path(path: str | Path, repository_root: Path) -> str:
    candidate = Path(path)
    resolved_root = repository_root.resolve()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (resolved_root / candidate).resolve()
    )
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("executable paths must remain inside repository_root") from exc
    if not relative.parts:
        raise ValueError("executable path must identify a file")
    return relative.as_posix()


def collect_code_provenance(
    executable_paths: tuple[str | Path, ...],
    *,
    repository_root: str | Path = Path(__file__).resolve().parent,
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
    from control.rolling_horizon import InvalidRun, ValidRun

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
    repository_root: str | Path = Path(__file__).resolve().parent,
    runtime: Mapping[str, object] | None = None,
) -> RunSpecification:
    """Build the exact requested-input identity, independent of output bytes."""
    if not isinstance(policy, EvaluationPolicy):
        raise TypeError("policy must be an EvaluationPolicy")
    _validate_publication_run_evidence(run)
    scenario = _scenario_content(run.scenario)
    input_hashes = _scenario_input_hashes(scenario)
    code_provenance = collect_code_provenance(
        executable_paths,
        repository_root=repository_root,
    )
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
        "runtime": _normalized_runtime(runtime),
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
        "policy": report.policy.to_serializable_metadata(),
        "nominal": asdict(report.nominal),
        "wear_sensitivities": {
            label: asdict(summary)
            for label, summary in report.wear_sensitivities.items()
        },
        "step_line_items": [asdict(item) for item in report.step_line_items],
    }


def _validation_content(run: object) -> dict[str, JSONValue]:
    steps = []
    for record, diagnostics in zip(
        run.records,
        run.controller_diagnostics,
        strict=True,
    ):
        steps.append(
            {
                "operating_step": record.operating_step,
                "decision_status": diagnostics.decision_status,
                "solver_success": diagnostics.solver_success,
                "solver_return_status": diagnostics.solver_return_status,
                "control_valid": True,
                "flows_valid": True,
                "successor_valid": True,
                "physical_invariants_valid": True,
            }
        )
    return {
        "complete": True,
        "valid": True,
        "checked_operating_steps": run.validation.checked_operating_steps,
        "issues": [],
        "steps": steps,
    }


def serialize_valid_run(
    run: ValidRun,
    report: EvaluationReport,
) -> dict[str, bytes]:
    """Serialize authoritative output members without a terminal pseudo-row."""
    from control.rolling_horizon import ValidRun

    if not isinstance(run, ValidRun):
        raise TypeError("serialize_valid_run accepts only ValidRun")
    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    _validate_publication_run_evidence(run)
    expected_report = evaluate_run(run, report.policy)
    if report != expected_report:
        raise ValueError("EvaluationReport does not match the ValidRun records")
    if tuple(record.operating_step for record in run.records) != tuple(
        range(run.scenario.operating_step_count)
    ):
        raise ValueError("Operating Records must be in exact step order")
    trajectory_rows = [_trajectory_row(record) for record in run.records]
    diagnostic_rows = [
        _diagnostic_row(index, diagnostics)
        for index, diagnostics in enumerate(run.controller_diagnostics)
    ]
    summary = _to_json_primitives(_summary_content(report), "summary")
    validation = _to_json_primitives(_validation_content(run), "validation")
    assert isinstance(summary, dict) and isinstance(validation, dict)
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


def _verify_bundle_directory(
    bundle_path: Path,
    expected_identifier: str | None,
    *,
    enforce_path_identifier: bool,
) -> RunBundle:
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

    code_provenance = manifest["code_provenance"]
    if not isinstance(code_provenance, dict):
        raise ValueError("Run Bundle code_provenance must be an object")
    if code_provenance.get("publication_eligible") is not True:
        raise ValueError("Run Bundle executable source is not publication-eligible")
    if code_provenance.get("dirty_executable_paths", []) != []:
        raise ValueError("Run Bundle records dirty executable source")
    if code_provenance.get("untracked_executable_paths", []) != []:
        raise ValueError("Run Bundle records untracked executable source")
    _require_full_sha256(
        code_provenance.get("executable_source_tree_sha256"),
        "executable source-tree digest",
    )
    executable_hashes = _require_hash_mapping(
        code_provenance.get("executable_path_hashes"),
        "executable_path_hashes",
    )
    committed_hashes = _require_hash_mapping(
        code_provenance.get("committed_executable_path_hashes"),
        "committed_executable_path_hashes",
    )
    if executable_hashes != committed_hashes:
        raise ValueError("Run Bundle executable bytes differ from committed bytes")
    revision = code_provenance.get("git_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Run Bundle Git revision must be a full object ID")

    runtime = manifest["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "platform",
        "do_mpc",
        "casadi",
        "numpy",
        "pandas",
    } or any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("Run Bundle runtime manifest is incomplete")

    input_hashes = manifest["input_hashes"]
    if not isinstance(input_hashes, dict) or set(input_hashes) != {
        "scenario",
        "sources",
        "sidecars",
    }:
        raise ValueError("Run Bundle input_hashes manifest is incomplete")
    _require_full_sha256(input_hashes["scenario"], "input_hashes.scenario")
    _require_hash_mapping(input_hashes["sources"], "input_hashes.sources")
    _require_hash_mapping(input_hashes["sidecars"], "input_hashes.sidecars")

    validation = _read_canonical_json(
        bundle_path / "validation.json",
        "validation",
    )
    if validation.get("complete") is not True or validation.get("valid") is not True:
        raise ValueError("Run Bundle validation evidence must be complete and valid")
    checked = validation.get("checked_operating_steps")
    steps = validation.get("steps")
    if (
        isinstance(checked, bool)
        or not isinstance(checked, int)
        or checked <= 0
        or not isinstance(steps, list)
        or len(steps) != checked
    ):
        raise ValueError("Run Bundle validation step evidence is incomplete")
    for expected, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("operating_step") != expected:
            raise ValueError("Run Bundle validation Operating Steps are not contiguous")
        if step.get("physical_invariants_valid") is not True:
            raise ValueError("Run Bundle physical invariant evidence is incomplete")

    trajectory_header, trajectory_rows = _csv_rows(
        bundle_path / "trajectory.csv",
        "trajectory.csv",
    )
    if not {"operating_step", "timestamp_utc"}.issubset(trajectory_header):
        raise ValueError("Run Bundle trajectory schema is incomplete")
    _verify_operating_steps(trajectory_rows, checked, "trajectory.csv")
    for row in trajectory_rows:
        timestamp = row["timestamp_utc"]
        if not timestamp.endswith("Z"):
            raise ValueError("Run Bundle trajectory timestamps must be UTC Z")
        try:
            datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
        except ValueError as exc:
            raise ValueError("Run Bundle trajectory timestamp is invalid") from exc
        for key, value in row.items():
            if key in {"operating_step", "timestamp_utc"}:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Run Bundle trajectory {key} must be numeric"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(f"Run Bundle trajectory {key} must be finite")

    diagnostics_header, diagnostic_rows = _csv_rows(
        bundle_path / "controller_diagnostics.csv",
        "controller_diagnostics.csv",
    )
    required_diagnostics = {
        "operating_step",
        "decision_status",
        "solver_success",
        "solver_return_status",
    }
    if not required_diagnostics.issubset(diagnostics_header):
        raise ValueError("Run Bundle controller diagnostics schema is incomplete")
    _verify_operating_steps(
        diagnostic_rows,
        checked,
        "controller_diagnostics.csv",
    )
    controller = manifest["controller"]
    if not isinstance(controller, dict) or not isinstance(controller.get("name"), str):
        raise ValueError("Run Bundle controller manifest is invalid")
    for row in diagnostic_rows:
        if row["decision_status"] != "success":
            raise ValueError("Run Bundle contains unsuccessful controller diagnostics")
        if controller["name"] == "mpc":
            if row["solver_success"].casefold() != "true":
                raise ValueError("Run Bundle MPC diagnostics lack solver success")
            if not row["solver_return_status"]:
                raise ValueError("Run Bundle MPC diagnostics lack solver return status")

    _read_canonical_json(bundle_path / "summary.json", "summary")
    return RunBundle(
        identifier=bundle_identifier,
        specification_identifier=specification_identifier,
        path=bundle_path,
        manifest=manifest,
    )


def verify_run_bundle(
    bundle_path: str | Path,
    expected_identifier: str | None = None,
) -> RunBundle:
    """Verify an authoritative full-ID Run Bundle and return immutable metadata."""
    return _verify_bundle_directory(
        Path(bundle_path),
        expected_identifier,
        enforce_path_identifier=True,
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


def _assert_specification_matches_run(
    run: object,
    report: EvaluationReport,
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
    expected_policy = _to_json_primitives(report.policy.to_serializable_metadata())
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
) -> RunBundle:
    """Persist one verified Valid Run through an owned atomic temporary directory."""
    from control.rolling_horizon import ValidRun

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
    _assert_specification_matches_run(run, report, specification)
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
        )
        assert candidate.identifier == bundle_identifier

        if destination.exists():
            try:
                existing = verify_run_bundle(destination, bundle_identifier)
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

        try:
            os.replace(temporary, destination)
        except OSError as exc:
            if not destination.exists():
                raise
            try:
                existing = verify_run_bundle(destination, bundle_identifier)
            except (OSError, ValueError) as verification_error:
                raise BundleCollisionError(
                    f"concurrent Run Bundle {bundle_identifier} is inconsistent"
                ) from verification_error
            if any(
                (destination / name).read_bytes() != data
                for name, data in all_bytes.items()
            ):
                raise BundleCollisionError(
                    f"concurrent Run Bundle {bundle_identifier} differs"
                ) from exc
            return existing
        _fsync_directory(root)
        return verify_run_bundle(destination, bundle_identifier)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_run_bundle(target_root: str | Path, identifier: str) -> RunBundle:
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
    return verify_run_bundle(candidates[0], identifier)


def write_failure_diagnostics(
    run: InvalidRun,
    specification: RunSpecification,
    diagnostics_root: str | Path,
) -> Path:
    """Persist InvalidRun evidence outside the authoritative Run Bundle namespace."""
    from control.rolling_horizon import InvalidRun

    if not isinstance(run, InvalidRun):
        raise TypeError("write_failure_diagnostics accepts only InvalidRun")
    if not isinstance(specification, RunSpecification):
        raise TypeError("specification must be a RunSpecification")
    if run_specification_identifier(specification.canonical_content) != specification.identifier:
        raise ValueError("Run Specification identifier is internally inconsistent")
    expected_scenario = _scenario_content(run.scenario)
    if _to_json_primitives(specification.scenario) != expected_scenario:
        raise ValueError("Run Specification scenario does not match InvalidRun")
    root = Path(diagnostics_root)
    specification_root = root / specification.identifier
    specification_root.mkdir(parents=True, exist_ok=True)
    attempt_timestamp = datetime.now(timezone.utc)
    attempt = specification_root / attempt_timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    attempt.mkdir()
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


def _legacy_grid_inventory_adjusted_cost(
    results_df: object,
    init_equiv: float,
    settle_price: float,
    config: HubConfiguration = HubConfiguration(),
) -> float:
    """Characterize committed pre-migration grid-only artifact summaries.

    This private adapter is not a current evaluation API: it intentionally omits
    policy wear so the old committed CSV/summary characterization remains readable.
    Task 14 must delete it with that characterization when artifacts regenerate from
    valid Run Bundles under ``evaluate_run``.
    """
    final_equiv = stored_equiv_from_row(results_df.iloc[-1], config)
    return float(results_df["grid_cost_EUR"].sum()) + settle_price * (
        init_equiv - final_equiv
    )


def saving_pct(baseline_cost: float, comparison_cost: float) -> float:
    return saving_percent(baseline_cost, comparison_cost)
