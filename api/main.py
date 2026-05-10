"""
FastAPI REST API for Sales Forecasting
Supports both the default dataset and custom Excel file uploads
"""

import os
import sys
import shutil
import tempfile
import asyncio
from typing import List, Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_preprocessing import prepare_data
from models.forecast import forecast_state, forecast_all_states

# ─────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description="""
## 8-Week Sales Forecasting API

Forecasts next 8 weeks of sales per US state using the best of:
- SARIMA
- Facebook Prophet  
- XGBoost
- LSTM (Deep Learning)

### Usage
- Use `/forecast/{state}` for a specific state
- Use `/forecast` for all states
- Use `/upload-and-forecast` to upload your own Excel file
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sales_data.xlsx")

_df_cache: Optional[pd.DataFrame] = None
_training_status = {"status": "idle", "started_at": None, "message": ""}


def get_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = prepare_data(DATA_PATH)
    return _df_cache


# ─────────────────────────────────────────────
# Schemas
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


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "Sales Forecasting API",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs":              "/docs",
            "list_states":       "/states",
            "forecast_state":    "/forecast/{state}",
            "forecast_all":      "/forecast",
            "upload_forecast":   "/upload-and-forecast",
            "model_comparison":  "/model-comparison",
        }
    }


@app.get("/states", tags=["Info"])
def list_states():
    """List all states that have a trained model."""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_best_model.pkl")]
    states = sorted([f.replace("_best_model.pkl", "").replace("_", " ") for f in model_files])
    return {"total": len(states), "states": states}


@app.get("/model-comparison", tags=["Info"])
def model_comparison():
    """Return model metrics comparison for all states."""
    csv_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Run training first.")
    df = pd.read_csv(csv_path)
    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(
    state: str,
    weeks: int = Query(default=8, ge=1, le=26),
):
    """Get 8-week sales forecast for a specific state."""
    df = get_data()
    state_title = state.replace("-", " ").title()

    if state_title not in df["State"].unique():
        available = sorted(df["State"].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"State '{state_title}' not found. Available states: {available}"
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
            predicted_sales=round(float(row["predicted_sales"]), 2),
            week_number=i + 1,
        )
        for i, (_, row) in enumerate(result_df.iterrows())
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
    """Get forecasts for ALL states."""
    df = get_data()
    try:
        result_df = forecast_all_states(df, MODEL_DIR, n_weeks=weeks)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    result_df["forecast_date"] = result_df["forecast_date"].astype(str)
    return {
        "total_states": result_df["state"].nunique(),
        "weeks_ahead":  weeks,
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts":    result_df.to_dict(orient="records"),
    }


@app.post("/upload-and-forecast", tags=["Custom Data"])
async def upload_and_forecast(
    file: UploadFile = File(..., description="Excel file with columns: State, Date, Total/Sales"),
    weeks: int = Query(default=8, ge=1, le=26),
    retrain: bool = Query(default=False, description="Set True to retrain models on uploaded data"),
):
    """
    Upload your own Excel file and get 8-week forecasts.
    
    Expected Excel columns:
    - **State** — state name
    - **Date** — date of sales record  
    - **Total** or **Sales** — sales amount
    
    Set `retrain=true` to train new models on your data (takes 30-60 mins).
    Set `retrain=false` to use existing trained models (instant).
    """
    # Validate file type
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported.")

    # Save uploaded file to temp location
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Preprocess uploaded file
        try:
            df = prepare_data(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not process file: {e}")

        states_found = sorted(df["State"].unique().tolist())

        if retrain:
            # Train new models on uploaded data
            from models.train_models import train_all_models
            summary = train_all_models(df, MODEL_DIR)
            global _df_cache
            _df_cache = df

        # Generate forecasts using existing or newly trained models
        results = []
        for state in states_found:
            try:
                fc = forecast_state(state, df, MODEL_DIR, n_weeks=weeks)
                fc["forecast_date"] = fc["forecast_date"].astype(str)
                results.append({
                    "state":      state,
                    "model_used": fc["model_used"].iloc[0],
                    "MAPE":       fc["MAPE"].iloc[0],
                    "forecast":   fc[["forecast_date", "predicted_sales"]].to_dict(orient="records"),
                })
            except Exception as e:
                results.append({"state": state, "error": str(e)})

        return {
            "filename":     file.filename,
            "states_found": len(states_found),
            "weeks_ahead":  weeks,
            "retrained":    retrain,
            "generated_at": datetime.utcnow().isoformat(),
            "forecasts":    results,
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Retrain endpoints
def _run_training():
    global _df_cache, _training_status
    try:
        _training_status = {
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "message": "Training started..."
        }
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
    return {"message": "Retraining started. Check /retrain/status for updates."}


@app.get("/retrain/status", tags=["Admin"])
def retrain_status():
    return _training_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)