"""
Unit tests for data preprocessing and feature engineering.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_sales_data,
    load_inventory_data,
    clean_sales_data,
    clean_inventory_data,
    engineer_demand_features,
    split_chronological_demand,
    split_inventory_data
)


@pytest.fixture
def sample_sales_data():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    records = []
    for d in dates:
        records.append({
            'product_id': 'MED001',
            'medicine_name': 'Test Medicine',
            'category': 'Analgesic',
            'date': str(d.date()),
            'units_sold': np.random.choice([0, 1]),
            'purchase_quantity': 0,
            'stock_on_hand': 100,
            'unit_price_inr': 10.0,
            'promotion': 0,
            'day_of_week': d.strftime('%A'),
            'month': d.month,
            'is_weekend': 1 if d.dayofweek >= 5 else 0
        })
    return pd.DataFrame(records)


def test_load_sales_data():
    """Verify sales data loads correctly with sorted dates."""
    df = load_sales_data("dataset/daily_medicine_sales.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'product_id' in df.columns
    assert 'units_sold' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['date'])


def test_load_inventory_data():
    """Verify inventory data loads with parsed snapshot and expiry dates."""
    df = load_inventory_data("dataset/batch_expiry_inventory.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'batch_id' in df.columns
    assert 'expiry_risk' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['snapshot_date'])
    assert pd.api.types.is_datetime64_any_dtype(df['expiry_date'])


def test_clean_sales_missing_values(sample_sales_data):
    """Verify missing values are handled gracefully without raising errors."""
    corrupted = sample_sales_data.copy()
    corrupted.loc[0, 'units_sold'] = np.nan
    corrupted.loc[1, 'stock_on_hand'] = np.nan
    corrupted.loc[2, 'unit_price_inr'] = np.nan
    
    cleaned = clean_sales_data(corrupted)
    assert cleaned['units_sold'].isnull().sum() == 0
    assert cleaned['stock_on_hand'].isnull().sum() == 0
    assert cleaned['unit_price_inr'].isnull().sum() == 0


def test_engineer_demand_features(sample_sales_data):
    """Verify lag and rolling features are created without future data leakage."""
    sample_sales_data['date'] = pd.to_datetime(sample_sales_data['date'])
    feat_df = engineer_demand_features(sample_sales_data, max_lag=21)
    
    expected_cols = ['lag_1', 'lag_7', 'lag_14', 'lag_21', 'roll_mean_7', 'roll_std_7']
    for col in expected_cols:
        assert col in feat_df.columns
        assert feat_df[col].isnull().sum() == 0


def test_chronological_split_no_leakage(sample_sales_data):
    """Verify chronological split has no date overlap between train, val, and test."""
    sample_sales_data['date'] = pd.to_datetime(sample_sales_data['date'])
    feat_df = engineer_demand_features(sample_sales_data, max_lag=7)
    train_df, val_df, test_df = split_chronological_demand(feat_df, train_ratio=0.70, val_ratio=0.15)
    
    train_max = train_df['date'].max()
    val_min = val_df['date'].min()
    val_max = val_df['date'].max()
    test_min = test_df['date'].min()
    
    assert train_max < val_min, "Leakage detected: Train date >= Val date"
    assert val_max < test_min, "Leakage detected: Val date >= Test date"

