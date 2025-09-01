# Energy Predictor

A Python tool to predict city-level energy consumption using temporal and weather data.

## Installation

1. Clone the repository:
    git clone https://github.com/username/energy-predictor.git
    cd energy-predictor


2. Create a virtual environment:
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install the package in editable mode:
    pip install -e .


This will install all required dependencies and make the `energy-predict` command available.

## Running the Application

Once installed, run the app with:
   energy-predict


This executes the `main()` function in `energy_predictor/main.py`, which demonstrates the full application.

## Project Structure
energy-predictor/
├── pyproject.toml         # Project metadata & dependencies
├── README.md              # Setup and usage instructions
├── REPORT.md              # Methodology and findings
├── energy_predictor/      # Main source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   └── main.py            # Entry point

## Data Source

This project uses data from Kaggle: **US City-Scale Daily Electricity Consumption and Weather Data**  
Dataset: https://www.kaggle.com/datasets/shemantosharkar/us-city-scale-daily-electricity-consumption/data

## Citation

Wang, Z., Hong, T., Li, H. and Piette, M.A., 2021. Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models. *Advances in Applied Energy*, p.100025.  
Paper link: https://www.sciencedirect.com/science/article/pii/S2666792421000184?via%3Dihub