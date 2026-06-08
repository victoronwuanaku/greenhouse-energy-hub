"""
Generate the synthetic greenhouse ELECTRICITY demand profile.

Parameterised from published WUR (Wageningen University & Research) data for a
typical Dutch Westland/Monster-area tomato greenhouse (1 ha = 10,000 m2).

Note on heat
------------
Heat is NOT prescribed as a demand series. In the corrected model the greenhouse
indoor temperature is a state governed by a single energy-balance ODE
(`models/hub_model.py`), and the controller supplies heat / opens vents to hold it
in the comfort band. Outdoor temperature and irradiance (drivers of the heat balance)
come directly from the PVGIS dataset, so this script only produces the electrical load.

Key references:
  - Warmenhoven et al. (2023), WUR Greenhouse Energy Study, edepot.wur.nl/158852
  - Verhoeven et al. (2022), Sunergy Greenhouse measurements, edepot.wur.nl/14135

Physical model
--------------
Electricity demand:
  P_elec(t) = P_lighting(t) + P_base
  P_lighting(t): seasonal (cos envelope) x daily (06:00-22:00) x daylight dimming
  P_base: climate control, pumps, miscellaneous = 12 W/m2 constant

Scales:
  Floor area:        10,000 m2  (1 ha)
  Peak lighting:     120 W/m2   (modern LED, partly dimmed by daylight)
  Base electrical:   12 W/m2

Output: data/demand_profile.csv
    Columns: timestamp (UTC), P_elec_kW
"""

import numpy as np
import pandas as pd
from pathlib import Path

FLOOR_AREA_M2 = 10_000          # 1 ha greenhouse
P_BASE_W_M2 = 12.0             # base electrical load (pumps, controls) W/m2
P_LIGHTING_PEAK_W_M2 = 120.0   # peak LED supplemental lighting W/m2

PV_PATH = Path(__file__).parent / "pv_profile.csv"
OUTPUT_PATH = Path(__file__).parent / "demand_profile.csv"


def lighting_schedule(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Supplemental LED lighting profile [W/m2] before daylight dimming.

    - Season: maximum in mid-winter, zero in mid-summer (cosine envelope).
    - Daily: lights ON 06:00-22:00 (16 h photoperiod), OFF at night.
    """
    doy = timestamps.day_of_year.values
    hour = timestamps.hour.values
    # 365.25 keeps the annual phase consistent whether or not the source year is a leap year
    seasonal = 0.5 * (1 + np.cos(2 * np.pi * (doy - 1) / 365.25))  # peak Jan, zero Jun
    lights_on = ((hour >= 6) & (hour < 22)).astype(float)
    return P_LIGHTING_PEAK_W_M2 * seasonal * lights_on


def main():
    if not PV_PATH.exists():
        raise FileNotFoundError(f"{PV_PATH} not found — run fetch_pvgis.py first.")

    pv = pd.read_csv(PV_PATH, index_col="timestamp", parse_dates=True)
    irr = pv["G_Wm2"].values
    timestamps = pv.index

    # Lighting, dimmed proportionally to available daylight (max 80% dimming)
    p_lighting = lighting_schedule(timestamps)
    dim_factor = np.clip(1.0 - 0.8 * irr / 400.0, 0.2, 1.0)
    p_lighting = p_lighting * dim_factor

    p_elec_Wm2 = p_lighting + P_BASE_W_M2
    P_elec_kW = p_elec_Wm2 * FLOOR_AREA_M2 / 1000.0

    df = pd.DataFrame({"P_elec_kW": P_elec_kW}, index=timestamps)
    df.index.name = "timestamp"
    df.to_csv(OUTPUT_PATH)

    print(f"Saved {len(df)} hourly records -> {OUTPUT_PATH}")
    print(df.describe().round(1))
    print(f"\nAnnual electricity: {df['P_elec_kW'].sum()/1000:.1f} MWh")
    print(f"Per m2 electricity: {df['P_elec_kW'].sum()/FLOOR_AREA_M2:.1f} kWh/m2")


if __name__ == "__main__":
    main()
