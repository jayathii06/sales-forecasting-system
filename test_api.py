"""
API Test Script
Tests all endpoints of the forecasting API.
Run AFTER starting the API: python run.py --api-only

Usage:
    python test_api.py                    # test against localhost:8000
    python test_api.py --url http://...   # test against custom URL
"""

import sys
import json
import argparse
import requests

BASE_URL = "http://localhost:8000"


def ok(label: str):
    print(f"  ✅ {label}")


def fail(label: str, detail: str = ""):
    print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))


def test_health(base: str):
    print("\n[1] Health Check")
    r = requests.get(f"{base}/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    ok(f"status=ok, version={data['version']}")


def test_states(base: str):
    print("\n[2] List States")
    r = requests.get(f"{base}/states")
    assert r.status_code == 200
    data = r.json()
    assert "states" in data
    assert data["total"] > 0
    ok(f"{data['total']} states available")
    return data["states"]


def test_forecast_single(base: str, state: str = "California"):
    print(f"\n[3] Forecast — {state}")
    r = requests.get(f"{base}/forecast/{state}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["state"] == state
    assert len(data["forecast"]) == 8
    assert "model_used" in data
    assert "validation_metrics" in data
    ok(f"Model: {data['model_used']}, MAPE: {data['validation_metrics']['MAPE']}%")
    print(f"     Week 1: ${data['forecast'][0]['predicted_sales']:,.0f} ({data['forecast'][0]['forecast_date']})")
    print(f"     Week 8: ${data['forecast'][7]['predicted_sales']:,.0f} ({data['forecast'][7]['forecast_date']})")


def test_forecast_custom_weeks(base: str, state: str = "Texas"):
    print(f"\n[4] Forecast — {state} (4 weeks)")
    r = requests.get(f"{base}/forecast/{state}?weeks=4")
    assert r.status_code == 200
    data = r.json()
    assert len(data["forecast"]) == 4
    ok(f"4-week forecast returned correctly")


def test_forecast_invalid_state(base: str):
    print("\n[5] Forecast — Invalid State (expect 404)")
    r = requests.get(f"{base}/forecast/FakeState123")
    assert r.status_code == 404
    ok("404 returned for unknown state")


def test_all_forecasts(base: str):
    print("\n[6] All States Forecast")
    r = requests.get(f"{base}/forecast")
    assert r.status_code == 200
    data = r.json()
    assert "forecasts" in data
    assert data["total_states"] > 0
    ok(f"Forecasts for {data['total_states']} states, {data['weeks_ahead']} weeks")


def test_model_comparison(base: str):
    print("\n[7] Model Comparison")
    r = requests.get(f"{base}/model-comparison")
    assert r.status_code == 200
    data = r.json()
    assert "data" in data
    assert len(data["data"]) > 0
    models_seen = set(row["model"] for row in data["data"])
    ok(f"{len(data['data'])} rows, models: {models_seen}")


def test_retrain_status(base: str):
    print("\n[8] Retrain Status")
    r = requests.get(f"{base}/retrain/status")
    assert r.status_code == 200
    data = r.json()
    ok(f"Status: {data['status']}")


def run_all(base: str):
    print(f"\n{'='*55}")
    print(f"  Sales Forecasting API Tests")
    print(f"  Base URL: {base}")
    print(f"{'='*55}")

    passed = 0
    failed = 0
    tests = [
        ("Health Check",           lambda: test_health(base)),
        ("List States",            lambda: test_states(base)),
        ("Single State Forecast",  lambda: test_forecast_single(base)),
        ("Custom Weeks Forecast",  lambda: test_forecast_custom_weeks(base)),
        ("Invalid State (404)",    lambda: test_forecast_invalid_state(base)),
        ("All States Forecast",    lambda: test_all_forecasts(base)),
        ("Model Comparison",       lambda: test_model_comparison(base)),
        ("Retrain Status",         lambda: test_retrain_status(base)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            fail(name, str(e))
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    args = parser.parse_args()
    success = run_all(args.url)
    sys.exit(0 if success else 1)
