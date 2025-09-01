# energy_predictor/evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from typing import Tuple
from sklearn.base import RegressorMixin

def test_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RegressorMixin,
    city_name: str = "City",
    factor: float = 1.0
) -> Tuple[float, float]:
    """
    Evaluate a regression model and visualize predictions and residuals.

    Parameters
    ----------
    X_test : np.ndarray
        Feature matrix for testing (2D array: samples x features).
    y_test : np.ndarray
        True target values (1D array).
    model : RegressorMixin
        Trained regression model with a predict() method.
    city_name : str, optional
        Name of the city or dataset for labeling plots (default is "City").
    factor : float, optional
        Scaling factor to adjust predictions (default is 1.0).

    Returns
    -------
    Tuple[float, float]
        RMSE and R² score of the model on the test set.

    Notes
    -----
    This function produces three plots:
    1. Actual vs predicted load
    2. Residuals over time (as % of predicted)
    3. Histogram of residuals (as % of predicted)
    Also prints P10 and P90 of residuals and indicates ~80% coverage.
    """
    print(f"\nEvaluating model for {city_name} with scaling factor={factor:.3f}...")

    y_pred = model.predict(X_test) * factor

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{city_name} model evaluation:")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  R²: {r2:.3f}")

    # Compute residuals as percentage of predicted
    residuals_pct = 100 * (y_test.values / y_pred - 1)
    p10 = np.percentile(residuals_pct, 10)
    p90 = np.percentile(residuals_pct, 90)
    print(f"  Residuals P10: {p10:.2f}%, P90: {p90:.2f}%")
    print(f"  Approximately 80% of residuals are between P10 and P90.")

    # Plot actual vs predicted# energy_predictor/evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from typing import Tuple
from sklearn.base import RegressorMixin

def test_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RegressorMixin,
    city_name: str = "City",
    factor: float = 1.0
) -> Tuple[float, float]:
    """
    Evaluate a regression model and visualize predictions and residuals.

    Parameters
    ----------
    X_test : np.ndarray
        Feature matrix for testing (2D array: samples x features).
    y_test : np.ndarray
        True target values (1D array).
    model : RegressorMixin
        Trained regression model with a predict() method.
    city_name : str, optional
        Name of the city or dataset for labeling plots (default is "City").
    factor : float, optional
        Scaling factor to adjust predictions (default is 1.0).

    Returns
    -------
    Tuple[float, float]
        RMSE and R² score of the model on the test set.

    Notes
    -----
    This function produces three plots:
    1. Actual vs predicted load
    2. Residuals over time (as % of predicted)
    3. Histogram of residuals (as % of predicted)
    Also prints P10 and P90 of residuals and indicates ~80% coverage.
    """
    print(f"\nEvaluating model for {city_name} with scaling factor={factor:.3f}...")

    y_pred = model.predict(X_test) * factor

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{city_name} model evaluation:")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  R²: {r2:.3f}")

    # Compute residuals as percentage of predicted
    residuals_pct = 100 * (y_test.values / y_pred - 1)
    mean_resid = residuals_pct.mean()
    std_resid = residuals_pct.std()

    # z-values for two-sided confidence intervals
    z_90 = 1.645   # 90% CI → 5% in each tail
    z_95 = 1.96  # 95% CI → 2.5% in each tail

    # Compute CI bounds
    ci_90_lower = mean_resid - z_90 * std_resid
    ci_90_upper = mean_resid + z_90 * std_resid

    ci_95_lower = mean_resid - z_95 * std_resid
    ci_95_upper = mean_resid + z_95 * std_resid

    print(f"90% CI for residuals: [{ci_90_lower:.2f}%, {ci_90_upper:.2f}%]")
    print(f"95% CI for residuals: [{ci_95_lower:.2f}%, {ci_95_upper:.2f}%]")


    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual Load", alpha=0.7)
    plt.plot(y_pred, label="Predicted Load", alpha=0.7)
    plt.title(f"{city_name} Load: Actual vs Predicted")
    plt.xlabel("Time index")
    plt.ylabel("Load")
    plt.legend()
    plt.show()

    # Plot residuals
    plt.figure(figsize=(12, 4))
    plt.plot(residuals_pct, label="Residuals (%)", color='red', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"{city_name} Residuals (%)")
    plt.xlabel("Time index")
    plt.ylabel("Residual Load (%)")
    plt.legend()
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals_pct, bins=100, color='orange', alpha=0.7)
    plt.title(f"{city_name} Residual Histogram (%)")
    plt.xlabel("Residual Load (%)")
    plt.ylabel("Frequency")
    plt.show()

    return rmse, r2

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual Load", alpha=0.7)
    plt.plot(y_pred, label="Predicted Load", alpha=0.7)
    plt.title(f"{city_name} Load: Actual vs Predicted")
    plt.xlabel("Time index")
    plt.ylabel("Load")
    plt.legend()
    plt.show()

    # Plot residuals
    plt.figure(figsize=(12, 4))
    plt.plot(residuals_pct, label="Residuals (%)", color='red', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"{city_name} Residuals (%)")
    plt.xlabel("Time index")
    plt.ylabel("Residual Load (%)")
    plt.legend()
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals_pct, bins=100, color='orange', alpha=0.7)
    plt.title(f"{city_name} Residual Histogram (%)")
    plt.xlabel("Residual Load (%)")
    plt.ylabel("Frequency")
    plt.show()

    return rmse, r2
