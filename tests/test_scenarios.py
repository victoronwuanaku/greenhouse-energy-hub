from __future__ import annotations

from dataclasses import fields
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_source_bundle(
    repository_root: Path,
    *,
    include_sidecar: bool = True,
) -> tuple[Path, Path, dict[str, object]]:
    data_dir = repository_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_path = data_dir / "replacement.csv"
    frame = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC"),
    )
    frame.index.name = "timestamp"
    frame.to_csv(source_path)
    sidecar_path = source_path.with_suffix(".provenance.json")
    sidecar: dict[str, object] = {
        "schema_version": "source-provenance-v1",
        "provenance_status": "legacy-import",
        "historic_acquisition_time": None,
        "source": {
            "name": "Replacement Test Source",
            "url": "https://replacement.invalid/hourly",
        },
        "parameters": {"dataset": "replacement", "revision": 7},
        "original_timezone": "UTC",
        "units": {"value": "kW"},
        "transformations": ["parse_replacement_utc_timestamps"],
        "row_count": 2,
        "utc_coverage": {
            "start": "2023-01-01T00:00:00+00:00",
            "end": "2023-01-01T01:00:00+00:00",
            "step": "PT1H",
            "row_count": 2,
        },
        "output": {
            "path": "data/replacement.csv",
            "sha256": _sha256(source_path),
        },
    }
    if include_sidecar:
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return source_path, sidecar_path, sidecar


def _thaw(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_scenario_stable_interface_has_exact_fields():
    from greenhouse_energy_hub.scenarios import Scenario, ScenarioPoint, SourceProvenance

    assert tuple(field.name for field in fields(ScenarioPoint)) == (
        "timestamp_utc",
        "price_eur_per_kwh",
        "pv_kw",
        "electric_load_kw",
        "outdoor_temperature_c",
        "irradiance_w_per_m2",
    )
    assert tuple(field.name for field in fields(SourceProvenance)) == (
        "source_name",
        "source_path",
        "sha256",
        "acquisition_parameters",
        "original_timezone",
        "units",
        "transformations",
    )
    assert tuple(field.name for field in fields(Scenario)) == (
        "name",
        "operating_start",
        "operating_end",
        "forecast_end",
        "forecast_horizon_capacity_steps",
        "step_duration",
        "operating_step_count",
        "points",
        "provenance",
    )


@pytest.mark.parametrize(
    ("start_month", "expected_season"),
    [(1, "winter"), (6, "summer")],
)
def test_winter_and_summer_fourteen_day_windows_are_exact(
    start_month, expected_season
):
    from tests.support import load_window
    from greenhouse_energy_hub.scenarios import Scenario

    scenario = load_window(start_month=start_month, n_days=14, forecast_hours=24)

    assert isinstance(scenario, Scenario)
    assert scenario.name == f"{expected_season}_m{start_month:02d}_14d"
    assert scenario.operating_step_count == 14 * 24
    assert len(scenario.points) == 14 * 24 + 24
    assert scenario.forecast_horizon_capacity_steps == 24
    assert scenario.step_duration == timedelta(hours=1)
    assert str(pd.Timestamp(scenario.operating_start).tz) == "Europe/Amsterdam"
    assert str(pd.Timestamp(scenario.operating_end).tz) == "Europe/Amsterdam"
    assert pd.Timestamp(scenario.points[0].timestamp_utc).tz_convert("UTC") == (
        pd.Timestamp(scenario.operating_start).tz_convert("UTC")
    )


def test_december_window_with_available_forecast_coverage_is_complete():
    from tests.support import load_window

    scenario = load_window(start_month=12, n_days=14, forecast_hours=24)

    assert scenario.operating_step_count == 14 * 24
    assert len(scenario.points) == 14 * 24 + 24
    assert pd.Timestamp(scenario.operating_end) == pd.Timestamp(
        "2023-12-15 00:00", tz="Europe/Amsterdam"
    )


def test_cross_month_calendar_window_is_complete():
    from greenhouse_energy_hub.scenarios import build_scenario

    scenario = build_scenario(
        name="cross-month",
        operating_start=pd.Timestamp("2023-01-25 00:00", tz="Europe/Amsterdam"),
        calendar_days=14,
        max_horizon_steps=24,
    )

    assert scenario.operating_step_count == 14 * 24
    assert pd.Timestamp(scenario.operating_end) == pd.Timestamp(
        "2023-02-08 00:00", tz="Europe/Amsterdam"
    )
    assert len(scenario.points) == 14 * 24 + 24


def test_january_first_local_is_rejected_but_first_complete_midnight_succeeds():
    from greenhouse_energy_hub.scenarios import ScenarioCoverageError, build_scenario

    with pytest.raises(ScenarioCoverageError, match="2022-12-31T23:00:00"):
        build_scenario(
            name="missing-pre-year-price",
            operating_start=pd.Timestamp(
                "2023-01-01 00:00", tz="Europe/Amsterdam"
            ),
            calendar_days=1,
            max_horizon_steps=0,
        )

    complete = build_scenario(
        name="first-complete-winter-midnight",
        operating_start=pd.Timestamp("2023-01-02 00:00", tz="Europe/Amsterdam"),
        calendar_days=1,
        max_horizon_steps=0,
    )
    assert complete.operating_step_count == 24
    assert complete.points[0].timestamp_utc == pd.Timestamp(
        "2023-01-01 23:00", tz="UTC"
    ).to_pydatetime()


def test_known_good_winter_source_fixture_alignment_is_stable():
    """Keep acquired price/PV/weather bytes aligned while demand changes owner."""
    from tests.support import load_window
    from greenhouse_energy_hub.scenarios import derive_electrical_demand

    scenario = load_window(start_month=1, n_days=2, forecast_hours=24)
    points_by_time = {
        pd.Timestamp(point.timestamp_utc): point for point in scenario.points
    }
    point = points_by_time[pd.Timestamp("2023-01-02 09:00:00", tz="UTC")]

    assert point.price_eur_per_kwh == pytest.approx(0.14561)
    assert point.pv_kw == pytest.approx(11.3)
    assert point.irradiance_w_per_m2 == pytest.approx(39.48)
    assert point.outdoor_temperature_c == pytest.approx(3.26)
    expected_demand = derive_electrical_demand(
        pd.DatetimeIndex([pd.Timestamp(point.timestamp_utc)]),
        np.array([point.irradiance_w_per_m2]),
    )[0]
    assert point.electric_load_kw == pytest.approx(expected_demand)


def test_december_window_without_final_forecast_coverage_fails_at_construction():
    from greenhouse_energy_hub.scenarios import ScenarioCoverageError, build_scenario

    with pytest.raises(ScenarioCoverageError, match="Forecast Coverage"):
        build_scenario(
            name="year-end-insufficient",
            operating_start=pd.Timestamp(
                "2023-12-31 00:00", tz="Europe/Amsterdam"
            ),
            calendar_days=1,
            max_horizon_steps=2,
        )


@pytest.mark.parametrize(
    ("date", "expected_steps"),
    [("2023-03-26", 23), ("2023-10-29", 25)],
)
def test_dst_local_calendar_day_has_exact_operating_steps(date, expected_steps):
    from greenhouse_energy_hub.scenarios import build_scenario

    scenario = build_scenario(
        name=f"dst-{date}",
        operating_start=pd.Timestamp(f"{date} 00:00", tz="Europe/Amsterdam"),
        calendar_days=1,
        max_horizon_steps=0,
    )

    assert scenario.operating_step_count == expected_steps
    assert len(scenario.points) == expected_steps
    expected_utc = pd.date_range(
        pd.Timestamp(scenario.operating_start).tz_convert("UTC"),
        pd.Timestamp(scenario.operating_end).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    observed_utc = pd.DatetimeIndex(
        [point.timestamp_utc for point in scenario.points]
    )
    assert observed_utc.equals(expected_utc)


def test_lighting_turns_on_at_six_local_in_winter_and_summer():
    from greenhouse_energy_hub.scenarios import lighting_schedule

    local_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2023-01-15 05:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2023-01-15 06:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2023-08-01 05:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2023-08-01 06:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2023-08-01 22:00", tz="Europe/Amsterdam"),
        ]
    )

    values = lighting_schedule(local_times.tz_convert("UTC"))

    assert values[[0, 4]].tolist() == [0.0, 0.0]
    assert values[1] > 0.0
    assert values[3] > 0.0


def _invalid_source_index(case: str) -> pd.DatetimeIndex:
    if case == "missing":
        return pd.DatetimeIndex(
            [
                pd.Timestamp("2023-01-01 00:00", tz="UTC"),
                pd.Timestamp("2023-01-01 02:00", tz="UTC"),
            ]
        )
    if case == "duplicate":
        return pd.DatetimeIndex(
            [
                pd.Timestamp("2023-01-01 00:00", tz="UTC"),
                pd.Timestamp("2023-01-01 00:00", tz="UTC"),
            ]
        )
    if case == "naive":
        return pd.date_range("2023-01-01", periods=2, freq="h")
    if case == "off-grid":
        return pd.DatetimeIndex(
            [
                pd.Timestamp("2023-01-01 00:00", tz="UTC"),
                pd.Timestamp("2023-01-01 01:30", tz="UTC"),
            ]
        )
    raise AssertionError(f"unknown case: {case}")


@pytest.mark.parametrize("case", ["missing", "duplicate", "naive", "off-grid"])
def test_invalid_source_timestamps_raise_scenario_validation_error(case):
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, _validate_source_frame

    source = pd.DataFrame(
        {"value": [1.0, 2.0]}, index=_invalid_source_index(case)
    )
    with pytest.raises(ScenarioValidationError):
        _validate_source_frame(source, "test source", ("value",))


def test_non_finite_required_source_value_is_rejected():
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, _validate_source_frame

    source = pd.DataFrame(
        {"value": [1.0, np.nan]},
        index=pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC"),
    )

    with pytest.raises(ScenarioValidationError, match="finite"):
        _validate_source_frame(source, "test source", ("value",))


@pytest.mark.parametrize(
    ("start", "end", "calendar_days", "step"),
    [
        (
            pd.Timestamp("2023-01-01 00:00"),
            None,
            1,
            timedelta(hours=1),
        ),
        (
            pd.Timestamp("2023-01-01 00:00", tz="UTC"),
            None,
            1,
            timedelta(hours=1),
        ),
        (
            pd.Timestamp("2023-01-01 01:00", tz="Europe/Amsterdam"),
            None,
            1,
            timedelta(hours=1),
        ),
        (
            pd.Timestamp("2023-01-01 00:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2023-01-02 00:00", tz="UTC"),
            None,
            timedelta(hours=1),
        ),
        (
            pd.Timestamp("2023-01-01 00:00", tz="Europe/Amsterdam"),
            None,
            1,
            timedelta(minutes=30),
        ),
    ],
)
def test_build_scenario_rejects_ambiguous_time_semantics(
    start, end, calendar_days, step
):
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, build_scenario

    with pytest.raises(ScenarioValidationError):
        build_scenario(
            name="invalid-time-contract",
            operating_start=start,
            operating_end=end,
            calendar_days=calendar_days,
            max_horizon_steps=0,
            step_duration=step,
        )


def test_source_provenance_is_complete_repo_relative_and_deeply_immutable():
    from tests.support import load_window

    scenario = load_window(start_month=1, n_days=1, forecast_hours=3)

    assert {item.source_path for item in scenario.provenance} == {
        "data/source/grid_price_signal.csv",
        "data/source/pv_profile.csv",
    }
    expected_hashes = {
        "data/source/grid_price_signal.csv": _sha256(
            SOURCE_DATA_DIR / "grid_price_signal.csv"
        ),
        "data/source/pv_profile.csv": _sha256(SOURCE_DATA_DIR / "pv_profile.csv"),
    }
    for item in scenario.provenance:
        assert len(item.sha256) == 64
        assert item.sha256 == expected_hashes[item.source_path]
        assert item.acquisition_parameters["provenance_status"] == "legacy-import"
        assert item.acquisition_parameters["historic_acquisition_time"] is None
        assert "requested_operating_window" in item.acquisition_parameters
        assert "forecast_coverage" in item.acquisition_parameters
        assert "source_utc_coverage" in item.acquisition_parameters
        assert "target_utc_coverage" in item.acquisition_parameters
        with pytest.raises(TypeError):
            item.acquisition_parameters["mutated"] = True
        with pytest.raises(TypeError):
            item.acquisition_parameters["requested_operating_window"][
                "start_utc"
            ] = "mutated"
        with pytest.raises(TypeError):
            item.units["mutated"] = "unit"
    pv_provenance = next(
        item for item in scenario.provenance if item.source_path.endswith("pv_profile.csv")
    )
    assert "source_utc_calendar_transplant" in pv_provenance.transformations


def test_source_resolving_outside_repository_is_rejected(tmp_path):
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, _load_source_bundle

    source_path, _, _ = _write_source_bundle(tmp_path)

    with pytest.raises(ScenarioValidationError, match="outside repository"):
        _load_source_bundle(
            source_path,
            "replacement source",
            ("value",),
            repository_root=ROOT,
        )


def test_missing_companion_provenance_sidecar_is_rejected(tmp_path):
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, _load_source_bundle

    source_path, sidecar_path, _ = _write_source_bundle(
        tmp_path, include_sidecar=False
    )

    with pytest.raises(ScenarioValidationError, match=sidecar_path.name):
        _load_source_bundle(
            source_path,
            "replacement source",
            ("value",),
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "output_path",
        "output_digest",
        "row_count",
        "coverage_row_count",
        "coverage_start",
        "coverage_end",
        "coverage_step",
        "legacy_timestamp",
    ],
)
def test_source_sidecar_must_match_validated_bytes_and_coverage(tmp_path, mutation):
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, _load_source_bundle

    source_path, sidecar_path, sidecar = _write_source_bundle(tmp_path)
    if mutation == "output_path":
        sidecar["output"]["path"] = "data/not-the-source.csv"
    elif mutation == "output_digest":
        sidecar["output"]["sha256"] = "0" * 64
    elif mutation == "row_count":
        sidecar["row_count"] = 3
    elif mutation == "coverage_row_count":
        sidecar["utc_coverage"]["row_count"] = 3
    elif mutation == "coverage_start":
        sidecar["utc_coverage"]["start"] = "2022-12-31T23:00:00+00:00"
    elif mutation == "coverage_end":
        sidecar["utc_coverage"]["end"] = "2023-01-01T02:00:00+00:00"
    elif mutation == "coverage_step":
        sidecar["utc_coverage"]["step"] = "PT30M"
    elif mutation == "legacy_timestamp":
        sidecar["historic_acquisition_time"] = "2023-01-01T00:00:00+00:00"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ScenarioValidationError):
        _load_source_bundle(
            source_path,
            "replacement source",
            ("value",),
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    "mutation",
    ["source_name", "source_url", "parameters", "timezone", "units", "transforms"],
)
def test_source_sidecar_requires_typed_acquisition_metadata(tmp_path, mutation):
    from greenhouse_energy_hub.scenarios import ScenarioValidationError, _load_source_bundle

    source_path, sidecar_path, sidecar = _write_source_bundle(tmp_path)
    if mutation == "source_name":
        sidecar["source"]["name"] = ""
    elif mutation == "source_url":
        sidecar["source"]["url"] = None
    elif mutation == "parameters":
        sidecar["parameters"] = ["not", "a", "mapping"]
    elif mutation == "timezone":
        sidecar["original_timezone"] = ""
    elif mutation == "units":
        sidecar["units"] = {"other": "kW"}
    elif mutation == "transforms":
        sidecar["transformations"] = []
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ScenarioValidationError):
        _load_source_bundle(
            source_path,
            "replacement source",
            ("value",),
            repository_root=tmp_path,
        )


def test_replacement_source_identity_and_metadata_come_only_from_sidecar(tmp_path):
    from greenhouse_energy_hub.scenarios import _load_source_bundle, _source_provenance_from_bundle

    source_path, sidecar_path, sidecar = _write_source_bundle(tmp_path)
    bundle = _load_source_bundle(
        source_path,
        "replacement source",
        ("value",),
        repository_root=tmp_path,
    )
    context = {
        "requested_operating_window": {"start_utc": "2023-01-01T00:00:00+00:00"},
        "forecast_coverage": {"horizon_capacity_steps": 0},
        "target_utc_coverage": sidecar["utc_coverage"],
    }
    provenance = _source_provenance_from_bundle(
        bundle,
        scenario_metadata=context,
        scenario_transformations=("replacement_alignment",),
    )

    assert provenance.source_name == "Replacement Test Source"
    assert provenance.source_path == "data/replacement.csv"
    assert provenance.sha256 == _sha256(source_path)
    assert provenance.original_timezone == "UTC"
    assert provenance.units == {"value": "kW"}
    assert provenance.transformations == (
        "parse_replacement_utc_timestamps",
        "replacement_alignment",
    )
    assert provenance.acquisition_parameters["source_url"] == (
        "https://replacement.invalid/hourly"
    )
    assert _thaw(provenance.acquisition_parameters["parameters"]) == {
        "dataset": "replacement",
        "revision": 7,
    }
    assert provenance.acquisition_parameters["sidecar_path"] == (
        "data/replacement.provenance.json"
    )
    assert provenance.acquisition_parameters["sidecar_sha256"] == _sha256(
        sidecar_path
    )

    first_sidecar_digest = provenance.acquisition_parameters["sidecar_sha256"]
    sidecar["parameters"]["revision"] = 8
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    changed_bundle = _load_source_bundle(
        source_path,
        "replacement source",
        ("value",),
        repository_root=tmp_path,
    )
    changed = _source_provenance_from_bundle(
        changed_bundle,
        scenario_metadata=context,
        scenario_transformations=("replacement_alignment",),
    )
    assert changed.sha256 == provenance.sha256
    assert changed.acquisition_parameters["sidecar_sha256"] != first_sidecar_digest
    assert changed.acquisition_parameters != provenance.acquisition_parameters


def test_normal_scenario_provenance_is_anchored_to_validated_sidecars():
    from tests.support import load_window

    scenario = load_window(start_month=1, n_days=1, forecast_hours=3)
    expected_alignment = {
        "data/source/grid_price_signal.csv": ("direct_utc_instant_mapping",),
        "data/source/pv_profile.csv": (
            "source_utc_calendar_transplant",
            "scale_pv_1_kwp_to_500_kwp",
            "derive_electrical_demand_on_target_clock",
        ),
    }
    for provenance in scenario.provenance:
        source_path = ROOT / provenance.source_path
        sidecar_path = source_path.with_suffix(".provenance.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        acquisition_transforms = tuple(sidecar["transformations"])

        assert provenance.source_name == sidecar["source"]["name"]
        assert provenance.original_timezone == sidecar["original_timezone"]
        assert _thaw(provenance.units) == sidecar["units"]
        assert provenance.transformations == (
            acquisition_transforms + expected_alignment[provenance.source_path]
        )
        assert provenance.acquisition_parameters["source_url"] == (
            sidecar["source"]["url"]
        )
        assert _thaw(provenance.acquisition_parameters["parameters"]) == (
            sidecar["parameters"]
        )
        assert provenance.acquisition_parameters["sidecar_path"] == (
            sidecar_path.relative_to(ROOT).as_posix()
        )
        assert provenance.acquisition_parameters["sidecar_sha256"] == _sha256(
            sidecar_path
        )


def test_provenance_sidecars_match_exact_materialized_bytes_and_schema():
    expected = {
        "source/grid_price_signal": (
            8760,
            "legacy-import",
            "2023-01-01T00:00:00+00:00",
            "2023-12-31T23:00:00+00:00",
        ),
        "source/pv_profile": (
            8784,
            "legacy-import",
            "2020-01-01T00:00:00+00:00",
            "2020-12-31T23:00:00+00:00",
        ),
    }
    for relative_stem, (
        row_count,
        status,
        coverage_start,
        coverage_end,
    ) in expected.items():
        csv_path = DATA_DIR / f"{relative_stem}.csv"
        sidecar_path = DATA_DIR / f"{relative_stem}.provenance.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert sidecar["schema_version"] == "source-provenance-v1"
        assert sidecar["provenance_status"] == status
        assert sidecar["historic_acquisition_time"] is None
        assert sidecar["source"]["name"]
        assert "url" in sidecar["source"]
        assert sidecar["parameters"]
        assert sidecar["original_timezone"] == "UTC"
        assert sidecar["row_count"] == row_count
        assert sidecar["output"]["path"] == f"data/{relative_stem}.csv"
        assert sidecar["output"]["sha256"] == _sha256(csv_path)
        assert sidecar["utc_coverage"]["step"] == "PT1H"
        assert sidecar["utc_coverage"]["start"] == coverage_start
        assert sidecar["utc_coverage"]["end"] == coverage_end
        assert sidecar["utc_coverage"]["row_count"] == row_count
        assert sidecar["units"]
        assert sidecar["transformations"]
