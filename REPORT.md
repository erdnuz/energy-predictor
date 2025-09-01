# Energy Consumption Prediction: Methodology and Findings

## Introduction

Accurate prediction of city-level energy consumption is critical for energy management, grid planning, and sustainability. Increasing urbanization and the integration of renewable energy sources make it essential to understand temporal patterns and the influence of weather on electricity demand.

Common modeling approaches include classical regression, tree-based methods, and sequential models such as Long Short-Term Memory (LSTM) networks. Classical methods, like Random Forest regressors, are robust to noise and require minimal data preprocessing, whereas LSTMs can capture temporal dependencies explicitly but are more sensitive to hyperparameters and data volume.

## Data

We use the US City-Scale Daily Electricity Consumption and Weather Data dataset from Kaggle, which provides city-level hourly electricity consumption and weather metrics for Los Angeles, New York, and Sacramento.

For each city, the dataset contains:

   Load (Y): City electricity consumption

   Weather Features (X): Temperature, Humidity
   Temporal Features (X): Hour of day, Day of week, Week of year

   Lagged load features: Previous hour and previous day same-hour load

### Lagged Features

Lagged features capture temporal dependencies implicitly. Random Forest models do not inherently account for sequential information, but including previous load values allows the model to consider short-term trends and diurnal patterns, improving prediction accuracy without the complexity of sequence models.

## Modeling Approach

We trained Random Forest Regressors on each city's data separately. Key design choices include:

Model choice: Random Forest is robust, interpretable, and requires less tuning compared to LSTMs. Its ensemble nature reduces overfitting and handles non-linear relationships.

Features: Seven features per record, including temporal, weather, and lagged load values.

Train-test split: 80% training, 20% testing, respecting temporal order.

Scaling: Standardization (Z-score) applied to features before training.

## Findings
### Within-City Performance

All models achieved high R² scores on their respective test sets, indicating strong predictive capability. Residuals are mostly within ±5–10% of actual load values.

Los Angeles: RMSE 67.96, R² 0.989, 90% residual CI [-3.92%, 3.29%]

New York: RMSE 264.04, R² 0.993, 90% residual CI [-3.01%, 2.15%]

Sacramento: RMSE 87.01, R² 0.975, 90% residual CI [-6.52%, 6.19%]

### Cross-City Evaluation

The LA-trained model was evaluated on New York and Sacramento using a scaling factor based on the ratio of average loads. After scaling, the LA model performs reasonably well on unseen cities, showing that relationships between temporal/weather features and load are largely consistent across cities.

LA → New York (scaled): RMSE 740.68, R² 0.948, 90% residual CI [-5.56%, 8.56%]

LA → Sacramento (scaled): RMSE 122.03, R² 0.951, 90% residual CI [-8.32%, 8.43%]

## Conclusion

Random Forest models are highly effective for city-level energy prediction when including temporal and lagged load features.

Lagged features allow tree-based models to capture temporal dependencies without using LSTMs.

Cross-city evaluation suggests scaling by average consumption enables generalization, supporting potential for transfer learning.

## Future Work

While the current study demonstrates strong predictive performance using Random Forests and basic temporal/weather features, several directions could further improve accuracy, generalizability, and practical applicability:

Incorporating Additional Features:
   Beyond temperature, humidity, and temporal indicators, other variables could significantly improve predictions. Examples include:

   Industrial and commercial activity: Electricity demand in commercial or industrial zones can drive load patterns.

   Public holidays and major events: Special days often produce anomalous consumption profiles.

   Population mobility and traffic patterns: Changes in population movement can indicate shifts in energy usage.

   Renewable energy contribution: Regions with significant solar or wind generation may show different net load patterns.

Enhanced Lag Features:
   Current lag features capture only the previous hour and same-hour previous day. More sophisticated temporal features could include:

   Rolling averages: Moving averages over the past 3–24 hours to smooth short-term fluctuations.

   Seasonal lags: Weekly, monthly, or yearly lags to capture repeating seasonal patterns.


Sequential Models for Long-Term Dependencies:
   While Random Forests handle short-term dependencies well via lag features, sequential models like LSTMs or GRUs could capture longer-term patterns and trends in the load data, potentially improving performance for cities with complex seasonal or weekly usage cycles.

Cross-City Generalization and Population Normalization:

   Testing models on additional cities with varying population densities and industrial activity could improve generalizability.

   Combining models in an ensemble approach or using transfer learning could exploit similarities across cities while accounting for unique local factors.

Integration with Real-Time Data:
   Incorporating real-time weather forecasts, smart meter data, or IoT sensors could enable near-term load predictions for operational grid management.

Explainability and Feature Importance:
   Investigating the relative importance of features (e.g., via SHAP or permutation importance) could provide insights into what drives energy consumption in each city and inform energy policy or conservation strategies.


Note: For setup instructions, how to run the demo, and example results including all model findings, see the README.md.