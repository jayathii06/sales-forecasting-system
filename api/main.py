"""
FastAPI REST API for Sales Forecasting
Run with: uvicorn api.main:app --reload --port 8000

Endpoints:
  GET  /                          → health check
  GET  /states                    → list all states with trained models
  GET  /forecast/{state}          → 8-week forecast for a state
  GET  /forecast/all              → 8-week forecast for all states
  GET  /model-comparison          → model metrics comparison table
  POST /retrain                   → trigger retraining (async)
"""

import os
import sys
import json
import asyncio
from typing import List, Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_preprocessing import prepare_data
from models.forecast import forecast_state, forecast_all_states

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Sales Forecasting API",
    description="8-week ahead sales forecasting per US state using best ML/DL models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data cache — loaded once on startup
_df_cache: Optional[pd.DataFrame] = None
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sales_data.xlsx")


def get_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = prepare_data(DATA_PATH)
    return _df_cache


# ─────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────

class ForecastPoint(BaseModel):
    forecast_date: str
    predicted_sales: float
    week_number: int


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    validation_metrics: dict
    forecast: List[ForecastPoint]
    generated_at: str


class ModelInfo(BaseModel):
    state: str
    model_used: str
    MAE: float
    RMSE: float
    MAPE: float


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "Sales Forecasting API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/states", tags=["Info"])
def list_states():
    """Return all states that have a trained model."""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_best_model.pkl")]
    states = sorted([f.replace("_best_model.pkl", "").replace("_", " ") for f in model_files])
    return {"total": len(states), "states": states}


@app.get("/model-comparison", tags=["Info"])
def model_comparison():
    """Return model comparison metrics for all states."""
    csv_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="model_comparison.csv not found. Run training first.")
    df = pd.read_csv(csv_path)
    return {
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(
    state: str,
    weeks: int = Query(default=8, ge=1, le=26, description="Number of weeks to forecast"),
):
    """Get n-week sales forecast for a specific state."""
    df = get_data()

    # Normalize state name
    state_title = state.replace("-", " ").title()
    available = df["State"].unique()
    if state_title not in available:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state_title}' not found. Available: {sorted(available)[:5]}...",
        )

    try:
        result_df = forecast_state(state_title, df, MODEL_DIR, n_weeks=weeks)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {e}")

    forecast_points = [
        ForecastPoint(
            forecast_date=str(row["forecast_date"].date()),
            predicted_sales=row["predicted_sales"],
            week_number=i + 1,
        )
        for i, row in result_df.iterrows()
    ]

    return ForecastResponse(
        state=state_title,
        model_used=result_df["model_used"].iloc[0],
        validation_metrics={
            "MAE":  result_df["MAE"].iloc[0],
            "RMSE": result_df["RMSE"].iloc[0],
            "MAPE": result_df["MAPE"].iloc[0],
        },
        forecast=forecast_points,
        generated_at=datetime.utcnow().isoformat(),
    )


@app.get("/forecast", tags=["Forecast"])
def get_all_forecasts(weeks: int = Query(default=8, ge=1, le=26)):
    """Get 8-week forecasts for all states."""
    df = get_data()
    try:
        result_df = forecast_all_states(df, MODEL_DIR, n_weeks=weeks)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    result_df["forecast_date"] = result_df["forecast_date"].astype(str)
    return {
        "total_states": result_df["state"].nunique(),
        "weeks_ahead": weeks,
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": result_df.to_dict(orient="records"),
    }


# Retrain endpoint
_training_status = {"status": "idle", "started_at": None, "message": ""}


def _run_training():
    global _df_cache, _training_status
    try:
        _training_status = {"status": "running", "started_at": datetime.utcnow().isoformat(), "message": "Training started..."}
        from models.train_models import train_all_models
        df = prepare_data(DATA_PATH)
        _df_cache = df
        train_all_models(df, MODEL_DIR)
        _training_status["status"] = "completed"
        _training_status["message"] = "All models retrained successfully."
    except Exception as e:
        _training_status["status"] = "failed"
        _training_status["message"] = str(e)


@app.post("/retrain", tags=["Admin"])
def retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining in the background."""
    if _training_status["status"] == "running":
        return {"message": "Training already in progress.", "status": _training_status}
    background_tasks.add_task(_run_training)
    return {"message": "Retraining started in background. Check /retrain/status for updates."}


@app.get("/retrain/status", tags=["Admin"])
def retrain_status():
    return _training_status


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
