"""Validated Scenario construction, temporal semantics, and input provenance.

This module is the single owner of the experiment clock.  Acquired price and
PV/weather bytes are validated before alignment; electrical demand is derived
from target timestamps rather than read from the legacy materialized CSV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AMSTERDAM_TIMEZONE = "Europe/Amsterdam"
UTC_TIMEZONE = "UTC"
ONE_HOUR = timedelta(hours=1)

PRICE_PATH = DATA_DIR / "grid_price_signal.csv"
PV_PATH = DATA_DIR / "pv_profile.csv"

PV_CAPACITY_KWP = 500.0
FLOOR_AREA_M2 = 10_000
P_BASE_W_M2 = 12.0
P_LIGHTING_PEAK_W_M2 = 120.0

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ScenarioValidationError(ValueError):
    """A Scenario or one of its sources violates the temporal/data contract."""


class ScenarioCoverageError(ScenarioValidationError):
    """A requested Operating Window lacks exact Forecast Coverage."""


def _freeze_json(value: object) -> object:
    """Copy JSON-like values into a deeply immutable representation."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("provenance mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("provenance numeric values must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"provenance value {value!r} is not JSON-compatible")


def _aware_timestamp(value: object, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError(
            f"{field_name} must be a valid timezone-aware timestamp"
        ) from exc
    if timestamp.tzinfo is None:
        raise ScenarioValidationError(f"{field_name} must be timezone-aware")
    try:
        timestamp.tz_convert(UTC_TIMEZONE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError(
            f"{field_name} timezone must be UTC-convertible"
        ) from exc
    return timestamp


def _utc_datetime(value: object, field_name: str) -> datetime:
    return _aware_timestamp(value, field_name).tz_convert(UTC_TIMEZONE).to_pydatetime()


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScenarioValidationError(f"{field_name} must be finite numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ScenarioValidationError(f"{field_name} must be finite numeric")
    return converted


@dataclass(frozen=True)
class ScenarioPoint:
    timestamp_utc: datetime
    price_eur_per_kwh: float
    pv_kw: float
    electric_load_kw: float
    outdoor_temperature_c: float
    irradiance_w_per_m2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_utc",
            _utc_datetime(self.timestamp_utc, "timestamp_utc"),
        )
        for field_name in (
            "price_eur_per_kwh",
            "pv_kw",
            "electric_load_kw",
            "outdoor_temperature_c",
            "irradiance_w_per_m2",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class SourceProvenance:
    source_name: str
    source_path: str
    sha256: str
    acquisition_parameters: Mapping[str, JSONValue]
    original_timezone: str
    units: Mapping[str, str]
    transformations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ScenarioValidationError("source_name must be nonempty")
        source_path = Path(self.source_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ScenarioValidationError("source_path must be repository-relative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ScenarioValidationError("sha256 must be a full lowercase SHA-256 digest")
        if not isinstance(self.original_timezone, str) or not self.original_timezone:
            raise ScenarioValidationError("original_timezone must be nonempty")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or not value
            for key, value in self.units.items()
        ):
            raise ScenarioValidationError("units must map nonempty strings to strings")
        object.__setattr__(self, "source_path", source_path.as_posix())
        object.__setattr__(
            self,
            "acquisition_parameters",
            _freeze_json(dict(self.acquisition_parameters)),
        )
        object.__setattr__(self, "units", _freeze_json(dict(self.units)))
        object.__setattr__(
            self,
            "transformations",
            tuple(str(item) for item in self.transformations),
        )
        if not self.transformations or any(not item for item in self.transformations):
            raise ScenarioValidationError("transformations must be nonempty strings")


@dataclass(frozen=True)
class Scenario:
    name: str
    operating_start: datetime
    operating_end: datetime
    forecast_end: datetime
    forecast_horizon_capacity_steps: int
    step_duration: timedelta
    operating_step_count: int
    points: tuple[ScenarioPoint, ...]
    provenance: tuple[SourceProvenance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ScenarioValidationError("Scenario name must be nonempty")
        start = _aware_timestamp(self.operating_start, "operating_start")
        end = _aware_timestamp(self.operating_end, "operating_end")
        forecast_end = _aware_timestamp(self.forecast_end, "forecast_end")
        try:
            duration = pd.Timedelta(self.step_duration).to_pytimedelta()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScenarioValidationError("step_duration must be one hour") from exc
        if duration != ONE_HOUR:
            raise ScenarioValidationError("step_duration must be exactly one hour")
        if (
            isinstance(self.forecast_horizon_capacity_steps, bool)
            or not isinstance(self.forecast_horizon_capacity_steps, Integral)
            or self.forecast_horizon_capacity_steps < 0
        ):
            raise ScenarioValidationError(
                "forecast_horizon_capacity_steps must be a nonnegative integer"
            )
        if (
            isinstance(self.operating_step_count, bool)
            or not isinstance(self.operating_step_count, Integral)
            or self.operating_step_count <= 0
        ):
            raise ScenarioValidationError(
                "operating_step_count must be a positive integer"
            )

        start_utc = start.tz_convert(UTC_TIMEZONE)
        end_utc = end.tz_convert(UTC_TIMEZONE)
        forecast_end_utc = forecast_end.tz_convert(UTC_TIMEZONE)
        if end_utc <= start_utc:
            raise ScenarioValidationError("operating_end must follow operating_start")
        expected_operating = pd.date_range(
            start=start_utc,
            end=end_utc,
            freq="h",
            inclusive="left",
        )
        if len(expected_operating) != int(self.operating_step_count):
            raise ScenarioValidationError(
                "operating_step_count does not match the exact UTC Operating Window"
            )
        expected_forecast_end = end_utc + int(
            self.forecast_horizon_capacity_steps
        ) * pd.Timedelta(hours=1)
        if forecast_end_utc != expected_forecast_end:
            raise ScenarioValidationError(
                "forecast_end does not match Forecast Coverage capacity"
            )

        points = tuple(self.points)
        provenance = tuple(self.provenance)
        if not all(isinstance(point, ScenarioPoint) for point in points):
            raise ScenarioValidationError("points must contain only ScenarioPoint values")
        if not all(isinstance(item, SourceProvenance) for item in provenance):
            raise ScenarioValidationError(
                "provenance must contain only SourceProvenance values"
            )
        expected_point_count = int(self.operating_step_count) + int(
            self.forecast_horizon_capacity_steps
        )
        if len(points) != expected_point_count:
            raise ScenarioCoverageError(
                f"{self.name}: Forecast Coverage requires {expected_point_count} "
                f"points; received {len(points)}"
            )
        expected_index = pd.date_range(
            start=start_utc,
            end=forecast_end_utc,
            freq="h",
            inclusive="left",
        )
        observed_index = pd.DatetimeIndex(
            [pd.Timestamp(point.timestamp_utc) for point in points]
        )
        if not observed_index.equals(expected_index):
            raise ScenarioCoverageError(
                f"{self.name}: points do not exactly cover the requested UTC instants"
            )

        object.__setattr__(self, "operating_start", start.to_pydatetime())
        object.__setattr__(self, "operating_end", end.to_pydatetime())
        object.__setattr__(self, "forecast_end", forecast_end.to_pydatetime())
        object.__setattr__(self, "forecast_horizon_capacity_steps", int(self.forecast_horizon_capacity_steps))
        object.__setattr__(self, "step_duration", duration)
        object.__setattr__(self, "operating_step_count", int(self.operating_step_count))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "provenance", provenance)

    def forecast_view(
        self, operating_step: int, horizon_steps: int
    ) -> tuple[ScenarioPoint, ...]:
        if (
            isinstance(operating_step, bool)
            or not isinstance(operating_step, Integral)
            or not 0 <= operating_step < self.operating_step_count
        ):
            raise ScenarioCoverageError(
                f"{self.name}: operating step {operating_step!r} is outside the "
                "Operating Window"
            )
        if (
            isinstance(horizon_steps, bool)
            or not isinstance(horizon_steps, Integral)
            or horizon_steps < 0
        ):
            raise ScenarioCoverageError("horizon_steps must be a nonnegative integer")
        if horizon_steps > self.forecast_horizon_capacity_steps:
            raise ScenarioCoverageError(
                f"{self.name}: requested horizon {horizon_steps} exceeds Scenario "
                f"capacity {self.forecast_horizon_capacity_steps}"
            )
        view = self.points[
            int(operating_step) : int(operating_step) + int(horizon_steps) + 1
        ]
        if len(view) != int(horizon_steps) + 1:
            raise ScenarioCoverageError(
                f"{self.name}: operating step {operating_step} requires "
                f"{int(horizon_steps) + 1} points; received {len(view)}"
            )
        return view


def sha256_file(path: str | Path) -> str:
    """Return the full streaming SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_frame(
    frame: pd.DataFrame,
    source_name: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate an acquired source before any temporal alignment occurs."""
    if not isinstance(frame, pd.DataFrame):
        raise ScenarioValidationError(f"{source_name}: source must be a DataFrame")
    missing_columns = [name for name in required_columns if name not in frame.columns]
    if missing_columns:
        raise ScenarioValidationError(
            f"{source_name}: missing required columns {missing_columns}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ScenarioValidationError(
            f"{source_name}: source index must be a DatetimeIndex"
        )
    if len(frame.index) == 0:
        raise ScenarioValidationError(f"{source_name}: source must not be empty")
    if frame.index.tz is None:
        raise ScenarioValidationError(
            f"{source_name}: source timestamps must be timezone-aware"
        )
    try:
        utc_index = frame.index.tz_convert(UTC_TIMEZONE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError(
            f"{source_name}: source timezone must be UTC-convertible"
        ) from exc
    if not utc_index.is_unique:
        raise ScenarioValidationError(
            f"{source_name}: duplicate source timestamps are forbidden"
        )
    if not utc_index.is_monotonic_increasing:
        raise ScenarioValidationError(
            f"{source_name}: source timestamps must be strictly increasing"
        )
    if any(
        np.asarray(component).any()
        for component in (
            utc_index.minute,
            utc_index.second,
            utc_index.microsecond,
            utc_index.nanosecond,
        )
    ):
        raise ScenarioValidationError(
            f"{source_name}: source timestamps must lie exactly on hourly UTC instants"
        )
    if len(utc_index) > 1:
        deltas = utc_index.to_series(index=range(len(utc_index))).diff().iloc[1:]
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise ScenarioValidationError(
                f"{source_name}: source timestamps must be complete and exactly hourly"
            )

    validated = frame.copy()
    validated.index = utc_index
    for column in required_columns:
        try:
            numeric = pd.to_numeric(validated[column], errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScenarioValidationError(
                f"{source_name}: {column} must contain finite numeric values"
            ) from exc
        if not np.isfinite(numeric).all():
            raise ScenarioValidationError(
                f"{source_name}: {column} must contain only finite values"
            )
        validated[column] = numeric
    return validated


def _read_validated_source(
    path: str | Path,
    source_name: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.is_file():
        raise ScenarioValidationError(f"{source_name}: source file not found: {source_path}")
    try:
        frame = pd.read_csv(source_path, index_col="timestamp", parse_dates=True)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ScenarioValidationError(
            f"{source_name}: could not parse source file {source_path}"
        ) from exc
    return _validate_source_frame(frame, source_name, required_columns)


def _missing_instants_message(
    source_name: str, missing: pd.DatetimeIndex
) -> str:
    preview = ", ".join(timestamp.isoformat() for timestamp in missing[:3])
    suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} missing)"
    return (
        f"{source_name}: Forecast Coverage is missing required UTC instants: "
        f"{preview}{suffix}"
    )


def _direct_utc_mapping(
    source: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    required_columns: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    missing = target_index.difference(source.index)
    if len(missing):
        raise ScenarioCoverageError(_missing_instants_message(source_name, missing))
    aligned = source.reindex(target_index).loc[:, list(required_columns)]
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ScenarioValidationError(
            f"{source_name}: aligned values must remain finite"
        )
    return aligned


def _calendar_transplant(
    source: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    required_columns: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    """Map a source UTC calendar profile onto target UTC month/day/hour keys."""
    if not isinstance(source.index, pd.DatetimeIndex) or source.index.tz is None:
        raise ScenarioValidationError(
            f"{source_name}: transplant source must have aware timestamps"
        )
    source_utc = source.index.tz_convert(UTC_TIMEZONE)
    source_keys = pd.MultiIndex.from_arrays(
        [source_utc.month, source_utc.day, source_utc.hour],
        names=("month", "day", "hour"),
    )
    if source_keys.has_duplicates:
        raise ScenarioValidationError(
            f"{source_name}: duplicate source_utc_calendar_transplant key"
        )
    keyed = source.loc[:, list(required_columns)].copy()
    keyed.index = source_keys

    target_utc = target_index.tz_convert(UTC_TIMEZONE)
    target_keys = pd.MultiIndex.from_arrays(
        [target_utc.month, target_utc.day, target_utc.hour],
        names=("month", "day", "hour"),
    )
    missing_mask = ~target_keys.isin(source_keys)
    if np.asarray(missing_mask).any():
        missing = target_utc[np.asarray(missing_mask)]
        raise ScenarioCoverageError(_missing_instants_message(source_name, missing))
    aligned = keyed.reindex(target_keys)
    aligned.index = target_utc
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ScenarioValidationError(
            f"{source_name}: transplanted values must remain finite"
        )
    return aligned


def _validated_target_index(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ScenarioValidationError("target timestamps must be a DatetimeIndex")
    if timestamps.tz is None:
        raise ScenarioValidationError("target timestamps must be timezone-aware")
    try:
        target_utc = timestamps.tz_convert(UTC_TIMEZONE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError(
            "target timestamps must be UTC-convertible"
        ) from exc
    if not target_utc.is_unique or not target_utc.is_monotonic_increasing:
        raise ScenarioValidationError(
            "target timestamps must be unique and strictly increasing"
        )
    if any(
        np.asarray(component).any()
        for component in (
            target_utc.minute,
            target_utc.second,
            target_utc.microsecond,
            target_utc.nanosecond,
        )
    ):
        raise ScenarioValidationError(
            "target timestamps must lie exactly on hourly UTC instants"
        )
    if len(target_utc) > 1:
        deltas = target_utc.to_series(index=range(len(target_utc))).diff().iloc[1:]
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise ScenarioValidationError("target timestamps must be exactly hourly")
    return target_utc


def lighting_schedule(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return the 06:00-22:00 Europe/Amsterdam seasonal LED schedule [W/m2]."""
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ScenarioValidationError("lighting timestamps must be a DatetimeIndex")
    if timestamps.tz is None:
        raise ScenarioValidationError("lighting timestamps must be timezone-aware")
    try:
        target_utc = timestamps.tz_convert(UTC_TIMEZONE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError(
            "lighting timestamps must be UTC-convertible"
        ) from exc
    local = target_utc.tz_convert(AMSTERDAM_TIMEZONE)
    day_of_year = local.day_of_year.to_numpy(dtype=float)
    hour = local.hour.to_numpy()
    seasonal = 0.5 * (
        1.0 + np.cos(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
    )
    lights_on = ((hour >= 6) & (hour < 22)).astype(float)
    return P_LIGHTING_PEAK_W_M2 * seasonal * lights_on


def derive_electrical_demand(
    target_utc_index: pd.DatetimeIndex,
    aligned_irradiance_w_per_m2: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Derive target-clock electrical demand [kW] from aligned irradiance."""
    target_utc = _validated_target_index(target_utc_index)
    try:
        irradiance = np.asarray(aligned_irradiance_w_per_m2, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError("aligned irradiance must be finite numeric") from exc
    if irradiance.ndim != 1 or len(irradiance) != len(target_utc):
        raise ScenarioValidationError(
            "aligned irradiance must contain exactly one value per target timestamp"
        )
    if not np.isfinite(irradiance).all():
        raise ScenarioValidationError("aligned irradiance must contain only finite values")

    scheduled_lighting = lighting_schedule(target_utc)
    dim_factor = np.clip(1.0 - 0.8 * irradiance / 400.0, 0.2, 1.0)
    demand_w_per_m2 = scheduled_lighting * dim_factor + P_BASE_W_M2
    demand_kw = demand_w_per_m2 * FLOOR_AREA_M2 / 1000.0
    if not np.isfinite(demand_kw).all():
        raise ScenarioValidationError("derived electrical demand must be finite")
    return demand_kw


def _require_amsterdam(value: object, field_name: str) -> pd.Timestamp:
    timestamp = _aware_timestamp(value, field_name)
    if str(timestamp.tz) != AMSTERDAM_TIMEZONE:
        raise ScenarioValidationError(
            f"{field_name} must use the {AMSTERDAM_TIMEZONE} timezone"
        )
    return timestamp


def _is_exact_hour(timestamp: pd.Timestamp) -> bool:
    return not any(
        (timestamp.minute, timestamp.second, timestamp.microsecond, timestamp.nanosecond)
    )


def _iso_utc(value: object) -> str:
    return _aware_timestamp(value, "timestamp").tz_convert(UTC_TIMEZONE).isoformat()


def _utc_coverage(index: pd.DatetimeIndex) -> dict[str, JSONValue]:
    return {
        "start": index[0].isoformat(),
        "end": index[-1].isoformat(),
        "step": "PT1H",
        "row_count": len(index),
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path.name).as_posix()


def _scenario_source_provenance(
    *,
    price_path: Path,
    pv_path: Path,
    prices: pd.DataFrame,
    pv: pd.DataFrame,
    operating_start: pd.Timestamp,
    operating_end: pd.Timestamp,
    forecast_end: pd.Timestamp,
    coverage_index: pd.DatetimeIndex,
    max_horizon_steps: int,
) -> tuple[SourceProvenance, ...]:
    requested_window: dict[str, JSONValue] = {
        "start_local": operating_start.isoformat(),
        "end_local": operating_end.isoformat(),
        "start_utc": operating_start.tz_convert(UTC_TIMEZONE).isoformat(),
        "end_utc": operating_end.tz_convert(UTC_TIMEZONE).isoformat(),
    }
    forecast_coverage: dict[str, JSONValue] = {
        "horizon_capacity_steps": max_horizon_steps,
        "coverage_start_utc": coverage_index[0].isoformat(),
        "coverage_end_utc": forecast_end.tz_convert(UTC_TIMEZONE).isoformat(),
        "last_point_utc": coverage_index[-1].isoformat(),
    }
    common: dict[str, JSONValue] = {
        "provenance_status": "legacy-import",
        "historic_acquisition_time": None,
        "requested_operating_window": requested_window,
        "forecast_coverage": forecast_coverage,
        "target_utc_coverage": _utc_coverage(coverage_index),
    }
    return (
        SourceProvenance(
            source_name="energy-charts.info NL day-ahead price",
            source_path=_repo_relative(price_path),
            sha256=sha256_file(price_path),
            acquisition_parameters={
                **common,
                "source_url": "https://api.energy-charts.info/price",
                "bidding_zone": "NL",
                "source_year": 2023,
                "row_count": len(prices),
                "source_utc_coverage": _utc_coverage(prices.index),
            },
            original_timezone="UTC",
            units={
                "price_EUR_MWh": "EUR/MWh",
                "price_EUR_kWh": "EUR/kWh",
            },
            transformations=("direct_utc_instant_mapping",),
        ),
        SourceProvenance(
            source_name="PVGIS Westland PV and weather profile",
            source_path=_repo_relative(pv_path),
            sha256=sha256_file(pv_path),
            acquisition_parameters={
                **common,
                "source_url": "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc",
                "latitude": 52.0,
                "longitude": 4.25,
                "source_year": 2020,
                "peak_power_kwp": 1.0,
                "tilt_degrees": 30,
                "aspect_degrees": 0,
                "system_loss_percent": 14,
                "radiation_database": "PVGIS-SARAH2",
                "row_count": len(pv),
                "source_utc_coverage": _utc_coverage(pv.index),
            },
            original_timezone="UTC",
            units={
                "P_kW": "kW/kWp",
                "G_Wm2": "W/m2",
                "T2m_C": "degC",
            },
            transformations=(
                "source_utc_calendar_transplant",
                "scale_pv_1_kwp_to_500_kwp",
                "derive_electrical_demand_on_target_clock",
            ),
        ),
    )


def build_scenario(
    *,
    name: str,
    operating_start: datetime | pd.Timestamp,
    max_horizon_steps: int,
    operating_end: datetime | pd.Timestamp | None = None,
    calendar_days: int | None = None,
    step_duration: timedelta = ONE_HOUR,
    price_path: str | Path = PRICE_PATH,
    pv_path: str | Path = PV_PATH,
) -> Scenario:
    """Construct one immutable Scenario with exact Operating/Forecast coverage."""
    start = _require_amsterdam(operating_start, "operating_start")
    try:
        normalized_step = pd.Timedelta(step_duration).to_pytimedelta()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScenarioValidationError("step_duration must be exactly one hour") from exc
    if normalized_step != ONE_HOUR:
        raise ScenarioValidationError("step_duration must be exactly one hour")
    if (
        isinstance(max_horizon_steps, bool)
        or not isinstance(max_horizon_steps, Integral)
        or max_horizon_steps < 0
    ):
        raise ScenarioValidationError(
            "max_horizon_steps must be a nonnegative integer"
        )
    max_horizon_steps = int(max_horizon_steps)
    if not _is_exact_hour(start):
        raise ScenarioValidationError("operating_start must lie exactly on the hour")
    if (operating_end is None) == (calendar_days is None):
        raise ScenarioValidationError(
            "provide exactly one of operating_end or calendar_days"
        )

    if calendar_days is not None:
        if (
            isinstance(calendar_days, bool)
            or not isinstance(calendar_days, Integral)
            or calendar_days <= 0
        ):
            raise ScenarioValidationError("calendar_days must be a positive integer")
        if start.time() != time.min:
            raise ScenarioValidationError(
                "calendar-day requests require a local-midnight operating_start"
            )
        end_date = start.date() + timedelta(days=int(calendar_days))
        end = pd.Timestamp(
            datetime.combine(end_date, time.min),
            tz=AMSTERDAM_TIMEZONE,
        )
    else:
        end = _require_amsterdam(operating_end, "operating_end")
        if not _is_exact_hour(end):
            raise ScenarioValidationError("operating_end must lie exactly on the hour")
    if end <= start:
        raise ScenarioValidationError("operating_end must follow operating_start")

    operating_index = pd.date_range(
        start=start.tz_convert(UTC_TIMEZONE),
        end=end.tz_convert(UTC_TIMEZONE),
        freq="h",
        inclusive="left",
    )
    if len(operating_index) == 0:
        raise ScenarioValidationError("Operating Window must contain at least one step")
    forecast_end = end + max_horizon_steps * pd.Timedelta(hours=1)
    coverage_index = pd.date_range(
        start=operating_index[0],
        end=forecast_end.tz_convert(UTC_TIMEZONE),
        freq="h",
        inclusive="left",
    )

    price_path = Path(price_path)
    pv_path = Path(pv_path)
    prices = _read_validated_source(
        price_path,
        "price source",
        ("price_EUR_MWh", "price_EUR_kWh"),
    )
    pv = _read_validated_source(
        pv_path,
        "PV/weather source",
        ("P_kW", "G_Wm2", "T2m_C"),
    )
    aligned_prices = _direct_utc_mapping(
        prices,
        coverage_index,
        ("price_EUR_MWh", "price_EUR_kWh"),
        "price source",
    )
    aligned_pv = _calendar_transplant(
        pv,
        coverage_index,
        ("P_kW", "G_Wm2", "T2m_C"),
        "PV/weather source",
    )
    demand_kw = derive_electrical_demand(
        coverage_index,
        aligned_pv["G_Wm2"].to_numpy(dtype=float),
    )

    points = tuple(
        ScenarioPoint(
            timestamp_utc=timestamp.to_pydatetime(),
            price_eur_per_kwh=aligned_prices.iloc[position]["price_EUR_kWh"],
            pv_kw=aligned_pv.iloc[position]["P_kW"] * PV_CAPACITY_KWP,
            electric_load_kw=demand_kw[position],
            outdoor_temperature_c=aligned_pv.iloc[position]["T2m_C"],
            irradiance_w_per_m2=aligned_pv.iloc[position]["G_Wm2"],
        )
        for position, timestamp in enumerate(coverage_index)
    )
    provenance = _scenario_source_provenance(
        price_path=price_path,
        pv_path=pv_path,
        prices=prices,
        pv=pv,
        operating_start=start,
        operating_end=end,
        forecast_end=forecast_end,
        coverage_index=coverage_index,
        max_horizon_steps=max_horizon_steps,
    )
    return Scenario(
        name=name,
        operating_start=start.to_pydatetime(),
        operating_end=end.to_pydatetime(),
        forecast_end=forecast_end.to_pydatetime(),
        forecast_horizon_capacity_steps=max_horizon_steps,
        step_duration=normalized_step,
        operating_step_count=len(operating_index),
        points=points,
        provenance=provenance,
    )
