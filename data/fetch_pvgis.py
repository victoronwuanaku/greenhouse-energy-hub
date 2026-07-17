"""
Fetch hourly solar PV generation data from the PVGIS API (EU JRC).

Location: Westland glasshouse district, Netherlands (52.0°N, 4.25°E).
System:   1 kWp crystalline silicon, 30° tilt, south-facing, 14% system loss.
Year:     2020 (most recent full year in PVGIS-SARAH2 database; API caps at 2020).
          Price data uses 2023 — datasets are reindexed to a common 8760-hour
          timeline in rolling_horizon.py. Climatically representative for NL.

Output: data/pv_profile.csv
    Columns: timestamp (UTC), P_kW (AC power per kWp), G_Wm2 (irradiance W/m²),
             T2m_C (ambient temperature °C)

Reference: PVGIS — https://re.jrc.ec.europa.eu/pvg_tools/en/
"""

import argparse
from datetime import datetime, timezone
import json
import urllib.request
import pandas as pd
from pathlib import Path

from scenarios import _read_validated_source, _utc_coverage, sha256_file

# --- Site parameters (Westland glasshouse district, NL) ---
# An inland point in the Westland greenhouse district; coastal points are classified
# as sea by PVGIS, so a nearby inland coordinate is used (climatically equivalent).
LAT = 52.0
LON = 4.25
YEAR = 2020  # PVGIS-SARAH2 available 2005–2020
PEAK_POWER_KWP = 1.0   # normalised to 1 kWp; scale in hub_model.py
TILT = 30              # degrees from horizontal
ASPECT = 0             # 0 = south-facing
LOSS = 14              # system losses (%)

OUTPUT_PATH = Path(__file__).parent / "pv_profile.csv"
PROVENANCE_PATH = Path(__file__).parent / "pv_profile.provenance.json"
ROOT = Path(__file__).resolve().parent.parent
SOURCE_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"


def fetch_pvgis() -> pd.DataFrame:
    """Fetch hourly PVGIS time series for one year and return a DataFrame."""
    url = (
        SOURCE_URL
        + f"?lat={LAT}&lon={LON}"
        f"&startyear={YEAR}&endyear={YEAR}"
        f"&pvcalculation=1&peakpower={PEAK_POWER_KWP}&loss={LOSS}"
        f"&angle={TILT}&aspect={ASPECT}"
        f"&mountingplace=free&outputformat=json"
        f"&raddatabase=PVGIS-SARAH2"
    )

    print(f"Fetching PVGIS data for ({LAT}°N, {LON}°E), year {YEAR}...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = json.loads(resp.read().decode())

    hourly = raw["outputs"]["hourly"]
    df = pd.DataFrame(hourly)

    # Parse PVGIS timestamp format "YYYYMMDD:HHMM" and floor to the hour: PVGIS marks
    # hourly values at HH:11 (solar-time offset), so flooring gives clean hourly UTC
    # stamps that align directly with the price series.
    df["timestamp"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", utc=True).floor("h")
    df = df.rename(columns={"P": "P_kW", "G(i)": "G_Wm2", "T2m": "T2m_C"})
    df = df[["timestamp", "P_kW", "G_Wm2", "T2m_C"]].set_index("timestamp")

    # Convert W to kW (PVGIS returns W per kWp)
    df["P_kW"] = df["P_kW"] / 1000.0

    return df


def _write_provenance(
    *,
    provenance_status: str = "legacy-import",
    historic_acquisition_time: str | None = None,
) -> dict[str, object]:
    frame = _read_validated_source(
        OUTPUT_PATH,
        "PV/weather source",
        ("P_kW", "G_Wm2", "T2m_C"),
    )
    sidecar: dict[str, object] = {
        "schema_version": "source-provenance-v1",
        "provenance_status": provenance_status,
        "historic_acquisition_time": historic_acquisition_time,
        "source": {
            "name": "PVGIS Westland PV and weather profile",
            "url": SOURCE_URL,
        },
        "parameters": {
            "latitude": LAT,
            "longitude": LON,
            "year": YEAR,
            "peak_power_kwp": PEAK_POWER_KWP,
            "tilt_degrees": TILT,
            "aspect_degrees": ASPECT,
            "system_loss_percent": LOSS,
            "mounting_place": "free",
            "radiation_database": "PVGIS-SARAH2",
        },
        "original_timezone": "UTC",
        "units": {
            "P_kW": "kW/kWp",
            "G_Wm2": "W/m2",
            "T2m_C": "degC",
        },
        "transformations": [
            "PVGIS_HH11_timestamp_floor_to_UTC_hour",
            "watts_per_kWp_to_kW_per_kWp",
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


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Fetch PVGIS data or describe existing bytes."
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="hash and describe the existing CSV without making a network call",
    )
    args = parser.parse_args(argv)
    if args.provenance_only:
        sidecar = _write_provenance()
        print(
            f"Recorded {sidecar['row_count']} existing PV/weather rows -> "
            f"{PROVENANCE_PATH}"
        )
        return

    df = fetch_pvgis()
    df.to_csv(OUTPUT_PATH)
    print(f"Saved {len(df)} hourly records → {OUTPUT_PATH}")
    print(df.describe().round(3))
    _write_provenance(
        provenance_status="acquired",
        historic_acquisition_time=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    main()
