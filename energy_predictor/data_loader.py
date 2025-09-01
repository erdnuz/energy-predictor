# energy_predictor/data_loader.py
import pandas as pd
from typing import Optional

def load_city_data(consumption_path: str, weather_path: str, city_name: str) -> Optional[pd.DataFrame]:
    """
    Load and combine electricity consumption and weather data for a city.

    Adds temporal and lag features for modeling. Lag features are calculated
    based on datetime differences, so the data does not need to be hourly.

    Parameters
    ----------
    consumption_path : str
        Path to the CSV file containing electricity load data.
    weather_path : str
        Path to the CSV file containing weather data.
    city_name : str
        Name of the city (used for debug prints).

    Returns
    -------
    Optional[pd.DataFrame]
        Combined DataFrame with the following columns:
        ['load', 'Temperature', 'Humidity', 'hour', 'day_of_week',
         'week_of_year', 'prev_load', 'prev_day_load'].
        Returns None if loading fails.

    Notes
    -----
    - Assumes hourly data for lag features.
    - Temporal features: 'hour', 'day_of_week', 'week_of_year'.
    """
    print(f"\nLoading data for {city_name}...")

    # Load consumption data
    try:
        consumption = pd.read_csv(consumption_path, index_col=0, parse_dates=True)
        print(f"  Consumption shape: {consumption.shape}")
    except Exception as e:
        print(f"  Error loading consumption data: {e}")
        return None

    # Load weather data
    try:
        weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
        print(f"  Weather shape: {weather.shape}")
    except Exception as e:
        print(f"  Error loading weather data: {e}")
        return None

    # Merge on datetime index
    try:
        combined = pd.merge(consumption, weather, left_index=True, right_index=True, how="inner")
        print(f"  Combined shape: {combined.shape}")
    except Exception as e:
        print(f"  Error combining datasets: {e}")
        return None

    # Add temporal features
    combined.index = pd.to_datetime(combined.index)
    combined['hour'] = combined.index.hour
    combined['day_of_week'] = combined.index.dayofweek
    combined['week_of_year'] = combined.index.isocalendar().week

    # Add lag features dynamically
    combined['prev_load'] = combined['load'].shift(1)
    combined['prev_day_load'] = combined['load'].shift(24)

    # Drop rows with NaN from lag features
    combined.dropna(inplace=True)

    print(f"  Added temporal and lag features. Combined shape now: {combined.shape}")
    return combined
