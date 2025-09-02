# Energy Consumption Prediction: Methodology and Findings

## Introduction

Accurate prediction of city-level energy consumption is critical for energy management, grid planning, and sustainability. Increasing urbanization and the integration of renewable energy sources make it essential to understand temporal patterns and the influence of weather on electricity demand.

Common modeling approaches include:

- **Classical regression & tree-based methods** (e.g., Random Forests): robust to noise, minimal preprocessing.
- **Sequential models** (e.g., LSTMs): capture temporal dependencies but require more data and tuning.

## Data

We use the **US City-Scale Daily Electricity Consumption and Weather Data** dataset from Kaggle, which provides city-level hourly electricity consumption and weather metrics for **Los Angeles, New York, and Sacramento**.

For each city, the dataset contains:

- **Load (Y):** City electricity consumption  
- **Weather features (X):** Temperature, Humidity  
- **Temporal features (X):** Hour of day, Day of week, Week of year  
- **Lagged load features:** Previous hour and same-hour previous day load  

### Lagged Features

Lagged features capture temporal dependencies implicitly. While Random Forests do not account for sequence directly, including lagged values lets the model learn **short-term trends and diurnal patterns**, improving accuracy without sequence models.

## Modeling Approach

We trained **Random Forest Regressors** on each city’s data separately.  

**Key design choices:**

- **Model:** Random Forest (robust, interpretable, less tuning than LSTM).  
- **Features:** 7 total (temporal, weather, lagged load).  
- **Train-test split:** 80% train / 20% test, respecting time order.  
- **Scaling:** Standardization (Z-score) applied to features.  

## Findings

### Within-City Performance

All models achieved high **R² scores** on their test sets, with residuals mostly within ±5–10% of actual load.  

- **Los Angeles:** RMSE = 67.96, R² = 0.989, 90% CI = [-3.92%, 3.29%]  
- **New York:** RMSE = 264.04, R² = 0.993, 90% CI = [-3.01%, 2.15%]  
- **Sacramento:** RMSE = 87.01, R² = 0.975, 90% CI = [-6.52%, 6.19%]  

### Cross-City Evaluation

The LA-trained model was applied to New York and Sacramento with **scaling by average load**. Results show that temporal/weather-load relationships generalize well across cities.  

- **LA → New York (scaled):** RMSE = 740.68, R² = 0.948, 90% CI = [-5.56%, 8.56%]  
- **LA → Sacramento (scaled):** RMSE = 122.03, R² = 0.951, 90% CI = [-8.32%, 8.43%]  

## Conclusion

- Random Forests are **highly effective** for city-level energy prediction with temporal + lagged features.  
- Lagged features enable **short-term temporal dependency modeling** without sequence models.  
- Cross-city scaling suggests **transfer learning potential**.  

## Future Work

Potential directions for improvement:

### Additional Features
- **Industrial & commercial activity**
- **Public holidays / major events**
- **Mobility & traffic patterns**
- **Renewable energy contributions**

### Enhanced Lag Features
- **Rolling averages:** (3–24h windows).  
- **Seasonal lags:** weekly, monthly, yearly patterns.  

### Sequential Models
- **LSTMs / GRUs** for long-term dependencies and seasonal effects.  

### Cross-City Generalization
- Extend to more cities with **population-normalized** predictions.  
- Explore **ensembles / transfer learning**.  

### Real-Time Integration
- Incorporate **weather forecasts, smart meter, IoT** data.  

### Explainability
- Use **SHAP, permutation importance** to analyze feature influence.  

---

**Note:** For setup instructions, running the demo, and detailed model results, see [README.md](./README.md).
