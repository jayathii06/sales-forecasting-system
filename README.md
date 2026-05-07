# Sales Forecasting System — Documentation

## Overview

This is a **production-ready end-to-end time series forecasting system** that:
- Processes historical weekly sales data for 43 US states
- Trains 4 ML/DL models per state: SARIMA, Prophet, XGBoost, LSTM
- Automatically selects the best model per state using MAPE
- Serves predictions via a REST API (FastAPI)

---

## Project Structure

```
forecasting_project/
│
├── data/
│   └── sales_data.xlsx          # Raw input data
│
├── utils/
│   └── data_preprocessing.py    # Data loading, cleaning, feature engineering
│
├── models/
│   ├── train_models.py          # Training all 4 models, model selection
│   ├── forecast.py              # Inference / prediction logic
│   └── model_comparison.csv     # Generated after training
│
├── api/
│   └── main.py                  # FastAPI REST API
│
├── run.py                       # Main entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow requires Python 3.8–3.11. Prophet requires `pystan`.

### 2. Run full pipeline (train + API)

```bash
python run.py
```

### 3. Train only

```bash
python run.py --train-only
```

### 4. Start API only (if models already trained)

```bash
python run.py --api-only
```

---

## Data Description

| Column   | Description                         |
|----------|-------------------------------------|
| State    | US State name                       |
| Date     | Observation date (weekly)           |
| Total    | Total sales value (USD)             |
| Category | Product category (Beverages)        |

- **43 states**, weekly frequency
- Date range: Jan 2019 – Oct 2021 (approx)
- Mixed date formats handled automatically

---

## Feature Engineering

| Feature         | Description                                      |
|-----------------|--------------------------------------------------|
| `lag_1`         | Sales from 1 week ago                            |
| `lag_4`         | Sales from 4 weeks ago (~1 month)                |
| `lag_13`        | Sales from 13 weeks ago (~1 quarter)             |
| `roll_mean_4`   | 4-week rolling average (no data leakage)         |
| `roll_std_4`    | 4-week rolling standard deviation                |
| `roll_mean_8`   | 8-week rolling average                           |
| `roll_std_8`    | 8-week rolling standard deviation                |
| `week_of_year`  | ISO week number (1–52)                           |
| `month`         | Month of year (1–12)                             |
| `quarter`       | Quarter (1–4)                                    |
| `day_of_week`   | Day of week (0=Monday)                           |
| `holiday_flag`  | 1 if any US holiday falls in that week           |

**No data leakage:** all lag/rolling features use `.shift(1)` so only past values are used.

---

## Models

### 1. SARIMA
- `order=(1,1,1)`, `seasonal_order=(1,1,1,52)` for yearly seasonality
- Handles trend and seasonality natively
- Fitted with `statsmodels.tsa.statespace.SARIMAX`

### 2. Prophet (Facebook)
- Multiplicative seasonality mode
- Yearly seasonality enabled
- Handles missing data and outliers well

### 3. XGBoost
- Gradient boosted trees on engineered lag/calendar features
- Walk-forward inference at prediction time (no leakage)
- 300 estimators, learning rate 0.05

### 4. LSTM (Deep Learning)
- 2-layer LSTM with Dropout (0.2) for regularization
- Lookback window: 13 weeks
- Trained with early stopping (patience=10)
- Walk-forward prediction for multi-step forecasting

---

## Model Selection

For each state, all 4 models are trained on the training set and evaluated on the **last 8 weeks** (held-out validation split).

Best model selected by **lowest MAPE** (Mean Absolute Percentage Error).

Metrics computed:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Square Error
- **MAPE** — Mean Absolute Percentage Error (primary selector)

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

### `GET /`
Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "Sales Forecasting API",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:00:00"
}
```

---

### `GET /states`
List all states with a trained model.

**Response:**
```json
{
  "total": 43,
  "states": ["Alabama", "Arizona", ...]
}
```

---

### `GET /forecast/{state}?weeks=8`
8-week forecast for a specific state.

**Parameters:**
- `state` — State name (e.g., `California`, `New-York`)
- `weeks` — Number of weeks (1–26, default 8)

**Response:**
```json
{
  "state": "California",
  "model_used": "XGBoost",
  "validation_metrics": {
    "MAE": 5023411.2,
    "RMSE": 6812345.0,
    "MAPE": 1.23
  },
  "forecast": [
    {"forecast_date": "2021-11-08", "predicted_sales": 452000000.0, "week_number": 1},
    ...
  ],
  "generated_at": "2024-01-15T10:00:00"
}
```

---

### `GET /forecast?weeks=8`
8-week forecasts for ALL states.

---

### `GET /model-comparison`
Returns model metrics for all states and all models.

---

### `POST /retrain`
Triggers full retraining in the background.

### `GET /retrain/status`
Check training status.

---

## Example API Calls (curl)

```bash
# Health check
curl http://localhost:8000/

# Get California forecast
curl http://localhost:8000/forecast/California

# Get Texas 4-week forecast
curl http://localhost:8000/forecast/Texas?weeks=4

# All states forecast
curl http://localhost:8000/forecast

# List all states
curl http://localhost:8000/states

# Model comparison table
curl http://localhost:8000/model-comparison
```

---

## Handling Missing Data

- Missing weeks in any state's time series are detected and filled
- Forward-fill (`ffill`) followed by backward-fill (`bfill`) strategy
- Mixed date formats (datetime objects + string 'DD-MM-YYYY') parsed robustly

---

## Assumptions

1. Data is aggregated to **weekly** frequency (Monday-starting weeks)
2. Only **Beverages** category is present in this dataset
3. US public holidays used for holiday flagging
4. Validation set = last 8 weeks of historical data (no leakage)
5. Models retrained from scratch each time `retrain` is called
