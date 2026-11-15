"""
Data Preprocessing Module for Pharmacy Medicine Demand & Expiry-Risk Forecasting.

This module provides reproducible data loading, cleaning, validation,
feature engineering, and chronological data splitting routines with strict
leakage prevention.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Define default feature columns
DEMAND_NUMERIC_FEATURES = [
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21',
    'roll_mean_7', 'roll_std_7', 'roll_mean_14', 'roll_std_14',
    'stock_on_hand', 'unit_price_inr', 'promotion',
    'day_of_week_num', 'day_of_month', 'month', 'is_weekend'
]

DEMAND_CATEGORICAL_FEATURES = ['category']

EXPIRY_NUMERIC_FEATURES = [
    'batch_quantity_remaining',
    'average_daily_sales_30d',
    'days_to_expiry',
    'projected_demand_before_expiry',
    'projected_unsold_quantity',
    'projected_unsold_ratio',
    'unit_price_inr'
]

EXPIRY_CATEGORICAL_FEATURES = ['category']


def load_sales_data(filepath: str) -> pd.DataFrame:
    """
    Load daily medicine sales dataset from CSV file.
    
    Args:
        filepath: Path to daily_medicine_sales.csv
        
    Returns:
        Sorted pandas DataFrame with parsed date column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sales dataset not found at path: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['product_id', 'date']).reset_index(drop=True)
    return df


def load_inventory_data(filepath: str) -> pd.DataFrame:
    """
    Load batch inventory dataset from CSV file.
    
    Args:
        filepath: Path to batch_expiry_inventory.csv
        
    Returns:
        Sorted pandas DataFrame with parsed dates.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Inventory dataset not found at path: {filepath}")
    
    df = pd.read_csv(filepath)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])
    df = df.sort_values(by=['snapshot_date', 'product_id', 'batch_id']).reset_index(drop=True)
    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate daily sales dataset.
    
    Checks for null values, duplicates, and invalid non-positive stock/price.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    
    # Fill any potential nulls with conservative defaults
    cleaned['units_sold'] = cleaned['units_sold'].fillna(0).astype(int)
    cleaned['stock_on_hand'] = cleaned['stock_on_hand'].fillna(0).clip(lower=0)
    cleaned['promotion'] = cleaned['promotion'].fillna(0).astype(int)
    cleaned['unit_price_inr'] = cleaned['unit_price_inr'].fillna(cleaned['unit_price_inr'].median())
    
    return cleaned


def clean_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate batch inventory dataset.
    """
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    
    # Fill nulls with conservative defaults
    cleaned['batch_quantity_remaining'] = cleaned['batch_quantity_remaining'].fillna(0).clip(lower=0)
    cleaned['average_daily_sales_30d'] = cleaned['average_daily_sales_30d'].fillna(0.0).clip(lower=0)
    cleaned['days_to_expiry'] = cleaned['days_to_expiry'].fillna(0).astype(int)
    cleaned['projected_unsold_ratio'] = cleaned['projected_unsold_ratio'].fillna(1.0).clip(0.0, 1.0)
    
    return cleaned


def engineer_demand_features(df: pd.DataFrame, max_lag: int = 21) -> pd.DataFrame:
    """
    Create time-series lag and rolling features strictly avoiding future data leakage.
    All rolling statistics are computed on shifted series (.shift(1)) so only past observations
    are used.
    
    Args:
        df: Cleaned sales DataFrame sorted by product_id and date.
        max_lag: Maximum lag steps to construct.
        
    Returns:
        DataFrame with added lag, rolling, and calendar features.
    """
    feat_df = df.copy()
    feat_df['date'] = pd.to_datetime(feat_df['date'])
    feat_df = feat_df.sort_values(by=['product_id', 'date']).reset_index(drop=True)
    
    # Calendar features
    feat_df['day_of_week_num'] = feat_df['date'].dt.dayofweek
    feat_df['day_of_month'] = feat_df['date'].dt.day
    feat_df['month'] = feat_df['date'].dt.month
    feat_df['is_weekend'] = feat_df['day_of_week_num'].apply(lambda d: 1 if d >= 5 else 0)
    
    # Lag features per product
    lags = [1, 2, 3, 7, 14, 21]
    for lag in lags:
        feat_df[f'lag_{lag}'] = feat_df.groupby('product_id')['units_sold'].shift(lag)
        
    # Rolling features using shift(1) to avoid target leakage
    feat_df['roll_mean_7'] = feat_df.groupby('product_id')['units_sold'].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )
    feat_df['roll_std_7'] = feat_df.groupby('product_id')['units_sold'].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).std().fillna(0.0)
    )
    feat_df['roll_mean_14'] = feat_df.groupby('product_id')['units_sold'].transform(
        lambda s: s.shift(1).rolling(window=14, min_periods=1).mean()
    )
    feat_df['roll_std_14'] = feat_df.groupby('product_id')['units_sold'].transform(
        lambda s: s.shift(1).rolling(window=14, min_periods=1).std().fillna(0.0)
    )
    
    # Drop initial rows where maximum lag has NaNs
    feat_df = feat_df.dropna(subset=[f'lag_{max_lag}']).reset_index(drop=True)
    return feat_df


def split_chronological_demand(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series chronologically to prevent future data leakage.
    
    Args:
        df: Feature-engineered sales DataFrame.
        train_ratio: Proportion of dates for training.
        val_ratio: Proportion of dates for validation.
        
    Returns:
        (train_df, val_df, test_df)
    """
    unique_dates = np.sort(df['date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    train_df = df[df['date'].isin(train_dates)].copy().reset_index(drop=True)
    val_df = df[df['date'].isin(val_dates)].copy().reset_index(drop=True)
    test_df = df[df['date'].isin(test_dates)].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def build_demand_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for demand features.
    StandardScaler for numeric features, OneHotEncoder for categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), DEMAND_NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), DEMAND_CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def split_inventory_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split batch inventory data chronologically by snapshot_date to reflect
    realistic forward-looking operations without snapshot leakage.
    """
    snapshots = np.sort(df['snapshot_date'].unique())
    n_snaps = len(snapshots)
    train_end = int(n_snaps * train_ratio)
    val_end = int(n_snaps * (train_ratio + val_ratio))
    
    train_snaps = snapshots[:train_end]
    val_snaps = snapshots[train_end:val_end]
    test_snaps = snapshots[val_end:]
    
    train_df = df[df['snapshot_date'].isin(train_snaps)].copy().reset_index(drop=True)
    val_df = df[df['snapshot_date'].isin(val_snaps)].copy().reset_index(drop=True)
    test_df = df[df['snapshot_date'].isin(test_snaps)].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def build_expiry_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for expiry risk features.
    StandardScaler for numeric features, OneHotEncoder for category.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), EXPIRY_NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), EXPIRY_CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

