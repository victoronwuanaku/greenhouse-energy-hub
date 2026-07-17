"""Materialize the Scenario-owned electrical-demand derivation for one year.

The generated CSV is a reproducible convenience view, not an authoritative
Scenario input.  Scenario construction calls ``derive_electrical_demand`` directly
on its target UTC index and aligned irradiance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scenarios import (
    FLOOR_AREA_M2,
    P_BASE_W_M2,
    P_LIGHTING_PEAK_W_M2,
    PV_PATH,
    _calendar_transplant,
    _read_validated_source,
    _utc_coverage,
    derive_electrical_demand,
    lighting_schedule,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).parent / "demand_profile.csv"
PROVENANCE_PATH = Path(__file__).parent / "demand_profile.provenance.json"


def materialize_demand(target_year: int) -> pd.DataFrame:
    if isinstance(target_year, bool) or not isinstance(target_year, int):
        raise ValueError("target_year must be an integer")
    target_index = pd.date_range(
        start=pd.Timestamp(f"{target_year}-01-01", tz="UTC"),
        end=pd.Timestamp(f"{target_year + 1}-01-01", tz="UTC"),
        freq="h",
        inclusive="left",
    )
    pv = _read_validated_source(
        PV_PATH,
        "PV/weather source",
        ("G_Wm2",),
    )
    aligned = _calendar_transplant(
        pv,
        target_index,
        ("G_Wm2",),
        "PV/weather source",
    )
    demand_kw = derive_electrical_demand(
        target_index,
        aligned["G_Wm2"].to_numpy(dtype=float),
    )
    frame = pd.DataFrame({"P_elec_kW": demand_kw}, index=target_index)
    frame.index.name = "timestamp"
    return frame


def _write_provenance(frame: pd.DataFrame, target_year: int) -> dict[str, object]:
    pv = _read_validated_source(
        PV_PATH,
        "PV/weather source",
        ("G_Wm2",),
    )
    sidecar: dict[str, object] = {
        "schema_version": "source-provenance-v1",
        "provenance_status": "derived-materialization",
        "historic_acquisition_time": None,
        "source": {
            "name": "scenarios.derive_electrical_demand",
            "url": None,
        },
        "parameters": {
            "target_year": target_year,
            "target_timezone": "UTC",
            "operating_timezone": "Europe/Amsterdam",
            "floor_area_m2": FLOOR_AREA_M2,
            "base_electrical_load_w_per_m2": P_BASE_W_M2,
            "peak_lighting_w_per_m2": P_LIGHTING_PEAK_W_M2,
            "lighting_start_local_hour": 6,
            "lighting_end_local_hour": 22,
            "maximum_daylight_dimming_fraction": 0.8,
        },
        "original_timezone": "UTC",
        "units": {"P_elec_kW": "kW"},
        "transformations": [
            "source_utc_calendar_transplant",
            "target_utc_to_Europe_Amsterdam_for_lighting",
            "seasonal_cosine_envelope",
            "local_06:00_to_22:00_photoperiod",
            "irradiance_daylight_dimming",
            "add_constant_base_load",
        ],
        "inputs": [
            {
                "path": PV_PATH.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(PV_PATH),
                "row_count": len(pv),
                "utc_coverage": _utc_coverage(pv.index),
            }
        ],
        "row_count": len(frame),
        "utc_coverage": _utc_coverage(frame.index),
        "output": {
            "path": OUTPUT_PATH.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(OUTPUT_PATH),
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sidecar


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Scenario-owned greenhouse electrical demand."
    )
    parser.add_argument("--target-year", type=int, default=2023)
    args = parser.parse_args(argv)

    frame = materialize_demand(args.target_year)
    frame.to_csv(OUTPUT_PATH)
    sidecar = _write_provenance(frame, args.target_year)
    print(f"Saved {len(frame)} hourly records -> {OUTPUT_PATH}")
    print(f"Output SHA-256: {sidecar['output']['sha256']}")
    print(f"Annual electricity: {frame['P_elec_kW'].sum()/1000:.1f} MWh")
    print(
        "Per m2 electricity: "
        f"{frame['P_elec_kW'].sum()/FLOOR_AREA_M2:.1f} kWh/m2"
    )


if __name__ == "__main__":
    main()
