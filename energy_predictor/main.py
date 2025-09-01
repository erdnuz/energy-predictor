# energy_predictor/main.py
from typing import Optional
import pandas as pd
import numpy as np
from energy_predictor.data_loader import load_city_data
from energy_predictor.preprocessing import preprocess_city_data
from energy_predictor.models import train_random_forest
from energy_predictor.evaluation import test_model
from sklearn.ensemble import RandomForestRegressor

def run_city_pipeline(consumption_path: str, weather_path: str, city_name: str
                     ) -> Optional[tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series, object]]:
    """
    Load, preprocess, and split city energy data for modeling.

    Parameters
    ----------
    consumption_path : str
        Path to electricity load CSV.
    weather_path : str
        Path to weather CSV.
    city_name : str
        Name of the city (for debug prints).

    Returns
    -------
    Optional[tuple]
        Tuple containing:
        - combined DataFrame
        - X_train, X_test
        - y_train, y_test
        - scaler object
        Returns None if loading fails.
    """
    df = load_city_data(consumption_path, weather_path, city_name)
    if df is None:
        print(f"Failed to load data for {city_name}. Skipping preprocessing.")
        return None
    X_train, X_test, y_train, y_test, scaler = preprocess_city_data(df)
    return df, X_train, X_test, y_train, y_test, scaler


def main() -> None:
    """
    Main pipeline to train and evaluate energy prediction models.
    - Loads and preprocesses data for LA, NY, and Sacramento
    - Trains RandomForest models for each city
    - Evaluates each model on its own test set
    - Evaluates LA model cross-city on NY and Sacramento with scaling
    """
    print("Running energy prediction demo...\n")

    # Load and preprocess all cities
    la_result = run_city_pipeline("./energy_predictor/data/power/la_load.csv",
                                  "./energy_predictor/data/weather/la_weather.csv",
                                  "Los Angeles")
    ny_result = run_city_pipeline("./energy_predictor/data/power/ny_load.csv",
                                  "./energy_predictor/data/weather/ny_weather.csv",
                                  "New York")
    sac_result = run_city_pipeline("./energy_predictor/data/power/sac_load.csv",
                                   "./energy_predictor/data/weather/sac_weather.csv",
                                   "Sacramento")

    if not la_result:
        print("Los Angeles data not available. Exiting pipeline.")
        return

    la_df, X_train_la, X_test_la, y_train_la, y_test_la, scaler_la = la_result

    # Train model on LA
    print("\nTraining RandomForest model on Los Angeles data...")
    model_la: RandomForestRegressor = train_random_forest(X_train_la, y_train_la)

    # Evaluate LA model on LA
    print("\nEvaluating LA model on Los Angeles test set:")
    test_model(X_test_la, y_test_la, model_la, "LA model on Los Angeles")

    # Train model on NY
    if ny_result:
        ny_df, X_train_ny, X_test_ny, y_train_ny, y_test_ny, scaler_ny = ny_result
        print("\nTraining RandomForest model on New York data...")
        model_ny: RandomForestRegressor = train_random_forest(X_train_ny, y_train_ny)

        print("\nEvaluating NY model on New York test set:")
        test_model(X_test_ny, y_test_ny, model_ny, "NY model on New York")

    # Train model on Sacramento
    if sac_result:
        sac_df, X_train_sac, X_test_sac, y_train_sac, y_test_sac, scaler_sac = sac_result
        print("\nTraining RandomForest model on Sacramento data...")
        model_sac: RandomForestRegressor = train_random_forest(X_train_sac, y_train_sac)

        print("\nEvaluating Sacramento model on Sacramento test set:")
        test_model(X_test_sac, y_test_sac, model_sac, "Sacramento model on Sacramento")

    # Cross-evaluate LA model on NY and Sacramento with scaling
    if ny_result:
        factor_ny = y_train_ny.mean() / y_train_la.mean()
        print(f"\nCross-evaluating LA model on New York (scaled) with factor {factor_ny:.3f}:")
        test_model(X_test_ny, y_test_ny, model_la, "LA model on New York (scaled)", factor_ny)

    if sac_result:
        factor_sac = y_train_sac.mean() / y_train_la.mean()
        print(f"\nCross-evaluating LA model on Sacramento (scaled) with factor {factor_sac:.3f}:")
        test_model(X_test_sac, y_test_sac, model_la, "LA model on Sacramento (scaled)", factor_sac)


if __name__ == "__main__":
    main()
