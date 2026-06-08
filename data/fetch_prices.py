"""
Fetch hourly NL day-ahead electricity prices from energy-charts.info (Fraunhofer ISE).

Data source: ENTSO-E Transparency Platform, via Fraunhofer ISE aggregation.
Bidding zone: NL (Netherlands, 10YNL----------L)
Year: 2023
Unit: EUR/MWh (converted to EUR/kWh for hub model)

Output: data/grid_price_signal.csv
    Columns: timestamp (UTC), price_EUR_MWh, price_EUR_kWh

The NL price signal exhibits:
- Negative prices during high solar / low demand periods (spring/summer afternoons)
- Price spikes during cold windless mornings (winter scarcity)
- Strong intraday pattern: low 01:00–06:00, peak 08:00–10:00 and 17:00–20:00

Reference: https://api.energy-charts.info
"""

import json
import time
import urllib.request
import pandas as pd
from pathlib import Path

YEAR = 2023
BZN = "NL"
OUTPUT_PATH = Path(__file__).parent / "grid_price_signal.csv"

# Fetch in monthly chunks to stay within API limits
MONTHS = [
    ("2023-01-01", "2023-02-01"),
    ("2023-02-01", "2023-03-01"),
    ("2023-03-01", "2023-04-01"),
    ("2023-04-01", "2023-05-01"),
    ("2023-05-01", "2023-06-01"),
    ("2023-06-01", "2023-07-01"),
    ("2023-07-01", "2023-08-01"),
    ("2023-08-01", "2023-09-01"),
    ("2023-09-01", "2023-10-01"),
    ("2023-10-01", "2023-11-01"),
    ("2023-11-01", "2023-12-01"),
    ("2023-12-01", "2024-01-01"),
]


def fetch_month(start: str, end: str) -> pd.DataFrame:
    """Fetch prices for one calendar month."""
    url = (
        f"https://api.energy-charts.info/price"
        f"?bzn={BZN}&start={start}&end={end}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode())

    timestamps = pd.to_datetime(raw["unix_seconds"], unit="s", utc=True)
    prices = raw["price"]  # EUR/MWh
    return pd.DataFrame({"timestamp": timestamps, "price_EUR_MWh": prices})


def main():
    frames = []
    for start, end in MONTHS:
        print(f"  Fetching {start} → {end} ...", end=" ")
        df_m = fetch_month(start, end)
        frames.append(df_m)
        print(f"{len(df_m)} records")
        time.sleep(0.5)  # polite rate limiting

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"].dt.year == YEAR]
    df["price_EUR_kWh"] = df["price_EUR_MWh"] / 1000.0
    df = df.set_index("timestamp")

    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(df)} hourly records → {OUTPUT_PATH}")
    print(df.describe().round(4))
    neg = (df["price_EUR_MWh"] < 0).sum()
    print(f"Negative price hours: {neg} ({100*neg/len(df):.1f}%)")


if __name__ == "__main__":
    main()
