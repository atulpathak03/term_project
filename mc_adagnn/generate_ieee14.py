import numpy as np
import pandas as pd
import pandapower.networks as pn


def generate_ieee14_data(num_days=100):
    """
    Generate hourly active-power load data for the IEEE 14-bus system.

    Parameters
    ----------
    num_days : int
        Number of days to generate (default: 100).

    Returns
    -------
    pd.DataFrame
        Hourly load dataset with shape (num_days * 24, 14).
        Columns: Bus_1, Bus_2, ..., Bus_14.
        Index: Hourly DatetimeIndex.
        Units: MW.
    """
    if not isinstance(num_days, (int, np.integer)) or isinstance(
        num_days, (bool, np.bool_)
    ) or num_days <= 0:
        raise ValueError("num_days must be a positive integer.")

    # 1. Load the standard IEEE 14-bus benchmark network.
    net = pn.case14()

    # 2. Extract the original active power demand for each bus.
    # Bus indices in pandapower are normally zero-based.
    base_load = np.zeros(14, dtype=float)

    if not net.load.empty:
        for bus_id, load_mw in zip(net.load["bus"], net.load["p_mw"]):
            base_load[int(bus_id)] += float(load_mw)

    # 3. Create an hourly DateTime index.
    num_hours = num_days * 24
    timestamps = pd.date_range(
        start="2021-01-01 00:00:00",
        periods=num_hours,
        freq="h"
    )

    rng = np.random.default_rng(seed=42)

    hour_of_day = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    # 4. Daily seasonality:
    # Peak around 3 PM, trough around 3 AM.
    daily_profile = 1.0 + 0.15 * np.sin(
        2.0 * np.pi * (hour_of_day - 9) / 24.0
    )

    # 5. Weekly seasonality:
    # Reduce load by 10% on Saturdays and Sundays.
    weekly_profile = np.where(day_of_week >= 5, 0.90, 1.0)

    # 6. Gaussian noise with a standard deviation of 5%.
    # Each hour and bus receives independent random noise.
    noise = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(num_hours, 14)
    )

    # 7. Generate the hourly load matrix.
    # Multiplicative noise preserves the relative scale of each bus.
    temporal_factor = (
        daily_profile[:, None]
        * weekly_profile[:, None]
    )

    load_data = (
        base_load[None, :]
        * temporal_factor
        * (1.0 + noise)
    )

    # Keep physical loads non-negative and preserve zero-load buses.
    load_data = np.maximum(load_data, 0.0)

    zero_load_buses = base_load == 0.0
    load_data[:, zero_load_buses] = 0.0

    # 8. Build the DataFrame.
    columns = [f"Bus_{i}" for i in range(1, 15)]

    df = pd.DataFrame(
        load_data,
        index=timestamps,
        columns=columns
    )
    df.index.name = "timestamp"

    return df


if __name__ == "__main__":
    # Generate the 100-day dataset.
    df = generate_ieee14_data(num_days=100)

    # Export to CSV.
    df.to_csv("data/ieee14_load_data.csv")

    # Basic validation.
    print("Dataset generated successfully.")
    print("Shape:", df.shape)
    print("Start time:", df.index[0])
    print("End time:", df.index[-1])
    print("\nFirst five rows:")
    print(df.head())

    print("\nBase load and generated load statistics (MW):")
    print(df.describe().round(3))

    assert df.shape == (2400, 14)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2021-01-01 00:00:00")
    assert df.index.freq == pd.tseries.frequencies.to_offset("h")
    assert (df.to_numpy() >= 0).all()

    print("\nSaved to data/ieee14_load_data.csv")
