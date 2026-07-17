"""
Fetch hourly NL day-ahead electricity prices from energy-charts.info (Fraunhofer ISE).

Data source: ENTSO-E Transparency Platform, via Fraunhofer ISE aggregation.
Bidding zone: NL (Netherlands, 10YNL----------L)
Year: 2023
Unit: EUR/MWh (converted to EUR/kWh for hub model)

Output: data/source/grid_price_signal.csv
    Columns: timestamp (UTC), price_EUR_MWh, price_EUR_kWh

The NL price signal exhibits:
- Negative prices during high solar / low demand periods (spring/summer afternoons)
- Price spikes during cold windless mornings (winter scarcity)
- Strong intraday pattern: low 01:00–06:00, peak 08:00–10:00 and 17:00–20:00

Reference: https://api.energy-charts.info
"""

import argparse
from datetime import datetime, timezone
import json
import time
import urllib.request
import pandas as pd
from pathlib import Path

from greenhouse_energy_hub.scenarios import (
    _read_validated_source,
    _utc_coverage,
    sha256_file,
)

YEAR = 2023
BZN = "NL"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "source" / "grid_price_signal.csv"
PROVENANCE_PATH = ROOT / "data" / "source" / "grid_price_signal.provenance.json"
SOURCE_URL = "https://api.energy-charts.info/price"

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
    url = SOURCE_URL + f"?bzn={BZN}&start={start}&end={end}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode())

    timestamps = pd.to_datetime(raw["unix_seconds"], unit="s", utc=True)
    prices = raw["price"]  # EUR/MWh
    return pd.DataFrame({"timestamp": timestamps, "price_EUR_MWh": prices})


def _write_provenance(
    *,
    provenance_status: str = "legacy-import",
    historic_acquisition_time: str | None = None,
) -> dict[str, object]:
    frame = _read_validated_source(
        OUTPUT_PATH,
        "price source",
        ("price_EUR_MWh", "price_EUR_kWh"),
    )
    sidecar: dict[str, object] = {
        "schema_version": "source-provenance-v1",
        "provenance_status": provenance_status,
        "historic_acquisition_time": historic_acquisition_time,
        "source": {
            "name": "energy-charts.info NL day-ahead electricity price",
            "url": SOURCE_URL,
        },
        "parameters": {
            "bidding_zone": BZN,
            "year": YEAR,
            "monthly_request_windows": [
                {"start": start, "end": end} for start, end in MONTHS
            ],
        },
        "original_timezone": "UTC",
        "units": {
            "price_EUR_MWh": "EUR/MWh",
            "price_EUR_kWh": "EUR/kWh",
        },
        "transformations": [
            "unix_seconds_to_utc",
            "filter_requested_year",
            "EUR_per_MWh_to_EUR_per_kWh",
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
        description="Fetch NL day-ahead prices or describe existing bytes."
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
            f"Recorded {sidecar['row_count']} existing price rows -> "
            f"{PROVENANCE_PATH}"
        )
        return

    frames = []
    for start, end in MONTHS:
        print(f"  Fetching {start} → {end} ...", end=" ")
        df_m = fetch_month(start, end)
        frames.append(df_m)
        print(f"{len(df_m)} records")
        time.sleep(0.5)  # polite rate limiting

    df = pd.concat(frames, ignore_index=True)
    if df["timestamp"].duplicated().any():
        raise ValueError("price acquisition returned duplicate UTC timestamps")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("price acquisition returned out-of-order UTC timestamps")
    df = df[df["timestamp"].dt.year == YEAR]
    df["price_EUR_kWh"] = df["price_EUR_MWh"] / 1000.0
    df = df.set_index("timestamp")

    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(df)} hourly records → {OUTPUT_PATH}")
    print(df.describe().round(4))
    neg = (df["price_EUR_MWh"] < 0).sum()
    print(f"Negative price hours: {neg} ({100*neg/len(df):.1f}%)")
    _write_provenance(
        provenance_status="acquired",
        historic_acquisition_time=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    main()
