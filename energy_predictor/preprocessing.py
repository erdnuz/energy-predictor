# energy_predictor/preprocessing.py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import Tuple

def preprocess_city_data(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    Preprocess city load data for modeling.

    Steps:
    1. Select features and target.
    2. Apply Z-score scaling to features.
    3. Split into train and test sets (80% train, 20% test, preserving temporal order).

    Parameters
    ----------
    df : pd.DataFrame
        Combined city dataset with features:
        ['Temperature', 'Humidity', 'hour', 'day_of_week', 'week_of_year',
         'prev_hour_load', 'prev_day_same_hour_load', 'load'].

    Returns
    -------
    X_train : np.ndarray
        Scaled feature matrix for training.
    X_test : np.ndarray
        Scaled feature matrix for testing.
    y_train : pd.Series
        Target values for training.
    y_test : pd.Series
        Target values for testing.
    scaler : StandardScaler
        Fitted scaler for transforming new data.
    """
    print("Preprocessing city data...")

    # Features and target
    X = df[['Temperature', 'Humidity', 'hour', 'day_of_week', 'week_of_year',
            'prev_load', 'prev_day_load']]
    y = df['load']

    # Z-score scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (temporal order preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"y_train length={len(y_train)}, y_test length={len(y_test)}")

    return X_train, X_test, y_train, y_test, scaler
