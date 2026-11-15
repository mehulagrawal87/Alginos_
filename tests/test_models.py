"""
Unit tests for Machine Learning models and inference.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd

from src.demand_forecasting import calculate_wape, MovingAverageBaseline
from src.expiry_risk import predict_batch_risk, RuleBasedExpiryBaseline
from src.data_preprocessing import DEMAND_NUMERIC_FEATURES, DEMAND_CATEGORICAL_FEATURES


def test_model_artifacts_exist():
    """Verify all expected model files exist in models/ directory."""
    expected_files = [
        "models/demand_preprocessor.joblib",
        "models/demand_best_model.joblib",
        "models/expiry_preprocessor.joblib",
        "models/expiry_best_model.joblib",
        "models/model_metadata.json"
    ]
    for file_path in expected_files:
        assert os.path.exists(file_path), f"Missing artifact: {file_path}"


def test_load_models_joblib():
    """Verify Joblib loads saved models and preprocessors without error."""
    demand_model = joblib.load("models/demand_best_model.joblib")
    demand_prep = joblib.load("models/demand_preprocessor.joblib")
    expiry_model = joblib.load("models/expiry_best_model.joblib")
    expiry_prep = joblib.load("models/expiry_preprocessor.joblib")
    
    assert hasattr(demand_model, "predict")
    assert hasattr(demand_prep, "transform")
    assert hasattr(expiry_model, "predict")
    assert hasattr(expiry_prep, "transform")


def test_demand_prediction_inference():
    """Verify demand forecasting model predicts positive demand for valid sample."""
    demand_model = joblib.load("models/demand_best_model.joblib")
    demand_prep = joblib.load("models/demand_preprocessor.joblib")
    
    sample_feat = {
        'lag_1': 1.0, 'lag_2': 0.0, 'lag_3': 0.0, 'lag_7': 1.0,
        'lag_14': 0.0, 'lag_21': 0.0,
        'roll_mean_7': 0.14, 'roll_std_7': 0.35,
        'roll_mean_14': 0.07, 'roll_std_14': 0.25,
        'stock_on_hand': 150, 'unit_price_inr': 5.0,
        'promotion': 0, 'day_of_week_num': 2,
        'day_of_month': 15, 'month': 4, 'is_weekend': 0,
        'category': 'Analgesic'
    }
    df = pd.DataFrame([sample_feat])
    proc_X = demand_prep.transform(df[DEMAND_NUMERIC_FEATURES + DEMAND_CATEGORICAL_FEATURES])
    pred = demand_model.predict(proc_X)
    
    assert len(pred) == 1
    assert pred[0] >= 0.0, "Predicted demand must be non-negative"


def test_expiry_risk_prediction_high_risk():
    """Verify batch with <= 90 days and >= 30% unsold ratio is classified as High risk."""
    expiry_model = joblib.load("models/expiry_best_model.joblib")
    expiry_prep = joblib.load("models/expiry_preprocessor.joblib")
    
    high_risk_batch = {
        'batch_quantity_remaining': 20,
        'average_daily_sales_30d': 0.01,
        'days_to_expiry': 45,
        'projected_demand_before_expiry': 0.45,
        'projected_unsold_quantity': 19.55,
        'projected_unsold_ratio': 0.97,
        'unit_price_inr': 15.0,
        'category': 'Antibiotic'
    }
    
    res = predict_batch_risk(high_risk_batch, expiry_model, expiry_prep)
    assert res['predicted_risk'] == 'High'
    assert 'probabilities' in res
    assert res['probabilities']['High'] > 0.5


def test_expiry_risk_prediction_low_risk():
    """Verify batch with > 180 days to expiry is classified as Low risk."""
    expiry_model = joblib.load("models/expiry_best_model.joblib")
    expiry_prep = joblib.load("models/expiry_preprocessor.joblib")
    
    low_risk_batch = {
        'batch_quantity_remaining': 10,
        'average_daily_sales_30d': 0.0,
        'days_to_expiry': 400,
        'projected_demand_before_expiry': 0.0,
        'projected_unsold_quantity': 10.0,
        'projected_unsold_ratio': 1.0,
        'unit_price_inr': 2.5,
        'category': 'Analgesic'
    }
    
    res = predict_batch_risk(low_risk_batch, expiry_model, expiry_prep)
    assert res['predicted_risk'] == 'Low'


def test_invalid_input_unknown_category():
    """Verify model handles unexpected or unseen medicine category gracefully."""
    expiry_model = joblib.load("models/expiry_best_model.joblib")
    expiry_prep = joblib.load("models/expiry_preprocessor.joblib")
    
    unseen_batch = {
        'batch_quantity_remaining': 10,
        'average_daily_sales_30d': 0.0,
        'days_to_expiry': 60,
        'projected_demand_before_expiry': 0.0,
        'projected_unsold_quantity': 10.0,
        'projected_unsold_ratio': 1.0,
        'unit_price_inr': 10.0,
        'category': 'UnknownExperimentalTherapy'
    }
    
    # Preprocessor should ignore unknown category without crashing
    res = predict_batch_risk(unseen_batch, expiry_model, expiry_prep)
    assert res['predicted_risk'] in ['High', 'Medium', 'Low']


def test_wape_zero_division():
    """Verify WAPE calculation handles all-zero actuals cleanly."""
    y_true = np.array([0, 0, 0])
    y_pred_zero = np.array([0, 0, 0])
    assert calculate_wape(y_true, y_pred_zero) == 0.0
    
    y_pred_nonzero = np.array([1, 0, 0])
    wape_val = calculate_wape(y_true, y_pred_nonzero)
    assert wape_val > 0.0

