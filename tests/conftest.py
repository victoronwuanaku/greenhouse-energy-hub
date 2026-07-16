from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=49, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "price_EUR_kWh": np.full(49, 0.10),
            "P_pv_kW": np.zeros(49),
            "P_elec_kW": np.full(49, 400.0),
            "T_out_C": np.full(49, 5.0),
            "G_Wm2": np.zeros(49),
        },
        index=index,
    )
