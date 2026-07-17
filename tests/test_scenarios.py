from __future__ import annotations

import pandas as pd
import pytest


def test_known_good_winter_source_fixture_alignment_is_stable():
    """Freeze one non-DST, non-year-end legacy alignment window before it moves."""
    from control.rolling_horizon import load_data

    frame = load_data(start_month=1, n_days=2)

    # Characterize the complete inner UTC date shared by legacy and future local
    # window semantics; the outer window boundaries intentionally remain unfrozen.
    fixture = frame.loc[
        pd.Timestamp("2023-01-01 00:00:00", tz="UTC") :
        pd.Timestamp("2023-01-01 23:00:00", tz="UTC")
    ]
    expected_index = pd.date_range(
        "2023-01-01 00:00:00",
        periods=24,
        freq="h",
        tz="UTC",
    )
    assert len(fixture) == 24
    assert fixture.index.equals(expected_index)

    aligned = fixture.loc[
        pd.Timestamp("2023-01-01 09:00:00", tz="UTC"),
        ["price_EUR_kWh", "P_pv_kW", "G_Wm2", "T_out_C", "P_elec_kW"],
    ]
    assert aligned.to_dict() == pytest.approx(
        {
            "price_EUR_kWh": 0.00099,
            "P_pv_kW": 9.24,
            "G_Wm2": 34.06,
            "T_out_C": 1.88,
            "P_elec_kW": 1238.256,
        }
    )


@pytest.mark.xfail(strict=True, reason="PF-07: insufficient year-end coverage is silently truncated")
def test_december_window_is_complete_or_explicitly_rejected():
    from control.rolling_horizon import load_data

    try:
        frame = load_data(start_month=12, n_days=31)
    except Exception as exc:
        assert type(exc).__name__ == "ScenarioCoverageError"
    else:
        assert len(frame) == 31 * 24


@pytest.mark.xfail(strict=True, reason="PF-07: lighting uses UTC rather than Europe/Amsterdam time")
def test_lighting_turns_on_at_six_local_in_winter_and_summer():
    from data.generate_demand import lighting_schedule

    local_six = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-15 06:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2020-08-01 06:00", tz="Europe/Amsterdam"),
        ]
    )
    source_utc = local_six.tz_convert("UTC")

    assert (lighting_schedule(source_utc) > 0.0).all()


@pytest.mark.xfail(strict=True, reason="PF-07: spring-forward windows use elapsed UTC days")
def test_spring_forward_local_calendar_window_has_23_hour_day():
    from control.rolling_horizon import load_data

    # March 1 through March 27 local contains 25 normal days and one 23-hour day.
    frame = load_data(start_month=3, n_days=26)
    assert len(frame) == 25 * 24 + 23


@pytest.mark.xfail(strict=True, reason="PF-07: fall-back windows use elapsed UTC days")
def test_fall_back_local_calendar_window_has_25_hour_day():
    from control.rolling_horizon import load_data

    # October 1 through October 30 local contains 28 normal days and one 25-hour day.
    frame = load_data(start_month=10, n_days=29)
    assert len(frame) == 28 * 24 + 25


def _invalid_source_index(case: str) -> pd.DatetimeIndex:
    if case == "missing":
        return pd.DatetimeIndex(
            [pd.Timestamp("2023-01-01 00:00", tz="UTC"), pd.Timestamp("2023-01-01 02:00", tz="UTC")]
        )
    if case == "duplicate":
        return pd.DatetimeIndex(
            [pd.Timestamp("2023-01-01 00:00", tz="UTC"), pd.Timestamp("2023-01-01 00:00", tz="UTC")]
        )
    if case == "naive":
        return pd.date_range("2023-01-01", periods=2, freq="h")
    if case == "off-hour":
        return pd.DatetimeIndex(
            [pd.Timestamp("2023-01-01 00:00", tz="UTC"), pd.Timestamp("2023-01-01 01:30", tz="UTC")]
        )
    raise AssertionError(f"unknown case: {case}")


@pytest.mark.xfail(strict=True, reason="PF-07: malformed source timestamps are silently repaired")
@pytest.mark.parametrize("case", ["missing", "duplicate", "naive", "off-hour"])
def test_invalid_source_timestamps_raise_scenario_validation_error(case):
    from control.rolling_horizon import _key_by_hour

    source = pd.DataFrame({"value": [1.0, 2.0]}, index=_invalid_source_index(case))
    with pytest.raises(ValueError) as raised:
        _key_by_hour(source)
    assert type(raised.value).__name__ == "ScenarioValidationError"
