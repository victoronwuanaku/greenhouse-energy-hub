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

import json
import urllib.request
import pandas as pd
from pathlib import Path

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


def fetch_pvgis() -> pd.DataFrame:
    """Fetch hourly PVGIS time series for one year and return a DataFrame."""
    url = (
        "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
        f"?lat={LAT}&lon={LON}"
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


def main():
    df = fetch_pvgis()
    df.to_csv(OUTPUT_PATH)
    print(f"Saved {len(df)} hourly records → {OUTPUT_PATH}")
    print(df.describe().round(3))


if __name__ == "__main__":
    main()
