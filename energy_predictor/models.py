# energy_predictor/models.py
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100, 
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a Random Forest regressor on the provided training data.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix for training (2D array: samples x features).
    y_train : np.ndarray
        Target values corresponding to X_train (1D array).
    n_estimators : int, optional
        Number of trees in the forest (default is 100).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    RandomForestRegressor
        Fitted Random Forest model.
    
    Notes
    -----
    This model is used for regression tasks and does not automatically 
    handle temporal dependencies; lag features should be included in X_train 
    if needed.
    """
    print(f"Training RandomForest with {n_estimators} estimators and random_state={random_state}...")
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    print("Training complete. Number of features used:", X_train.shape[1])
    return model
