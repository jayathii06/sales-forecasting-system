"""
Main entry point — runs the full pipeline:
  1. Preprocess data
  2. Train all models (SARIMA, Prophet, XGBoost, LSTM) for each state
  3. Save best model per state
  4. Print a quick forecast summary
  5. Start the FastAPI server

Usage:
    python run.py               # train + start API
    python run.py --train-only  # just train, no server
    python run.py --api-only    # skip training, start API
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Sales Forecasting System")
    parser.add_argument("--train-only", action="store_true", help="Only train models, do not start API")
    parser.add_argument("--api-only",   action="store_true", help="Skip training, just start API server")
    parser.add_argument("--port", type=int, default=8000, help="API port (default 8000)")
    args = parser.parse_args()

    if not args.api_only:
        print("=" * 60)
        print("  STEP 1: Data Preprocessing")
        print("=" * 60)
        from utils.data_preprocessing import prepare_data
        df = prepare_data("data/sales_data.xlsx")

        print("\n" + "=" * 60)
        print("  STEP 2: Model Training")
        print("=" * 60)
        from models.train_models import train_all_models
        summary = train_all_models(df, model_dir="models")

        print("\n" + "=" * 60)
        print("  STEP 3: Best Model Summary")
        print("=" * 60)
        if summary.empty or "best" not in summary.columns:
            print("  No models trained successfully.")
        else:
            best = summary[summary["best"] == True].sort_values("MAPE")
            print(best[["state", "model", "MAE", "RMSE", "MAPE"]].to_string(index=False))

        print("\n" + "=" * 60)
        print("  STEP 4: Sample Forecast (California)")
        print("=" * 60)
        try:
            from models.forecast import forecast_state
            fc = forecast_state("California", df)
            print(fc[["forecast_date", "predicted_sales", "model_used"]].to_string(index=False))
        except Exception as e:
            print(f"  Could not run sample forecast: {e}")

    if not args.train_only:
        print("\n" + "=" * 60)
        print(f"  STEP 5: Starting API on http://0.0.0.0:{args.port}")
        print("  Docs available at: http://localhost:{}/docs".format(args.port))
        print("=" * 60 + "\n")
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()