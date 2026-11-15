"""
Demand Forecasting Module for Pharmacy Medicine Demand.

Implements:
1. Moving Average Baseline
2. Linear Regression Model
3. Random Forest Regressor
With chronological validation, feature importance extraction, evaluation metrics
(MAE, RMSE, WAPE), and iterative autoregressive 7-day forecasting.
"""

import os
import tempfile
import joblib
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

# Matplotlib configuration for headless environments
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import (
    DEMAND_NUMERIC_FEATURES,
    DEMAND_CATEGORICAL_FEATURES,
    build_demand_preprocessor
)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).
    WAPE = sum(|y_true - y_pred|) / sum(y_true)
    Handles edge cases where total actual sales are zero.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    total_actual = np.sum(y_true)
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    
    if total_actual == 0:
        return 0.0 if total_abs_error == 0 else float(total_abs_error)
    return float(total_abs_error / total_actual)


class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Transparent Moving Average Baseline.
    Predicts the rolling mean of the last 'window' days (default 7 days)
    per product, or product-level historical average if fewer observations exist.
    """
    def __init__(self, window: int = 7):
        self.window = window
        self.product_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df['target'] = y.values
        if 'product_id' in df.columns:
            self.product_means_ = df.groupby('product_id')['target'].mean().to_dict()
        self.global_mean_ = float(df['target'].mean()) if len(df) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # If roll_mean_7 is available in features, use it as the 7-day rolling forecast
        if 'roll_mean_7' in X.columns:
            preds = X['roll_mean_7'].fillna(self.global_mean_).values
            return np.clip(preds, 0, None)
        
        # Otherwise fallback to product mean
        if 'product_id' in X.columns:
            preds = X['product_id'].map(self.product_means_).fillna(self.global_mean_).values
            return np.clip(preds, 0, None)
            
        return np.full(len(X), self.global_mean_)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and WAPE metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = calculate_wape(y_true, y_pred)
    return {
        'mae': round(mae, 6),
        'rmse': round(rmse, 6),
        'wape': round(wape, 6)
    }


def train_demand_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str = "models",
    output_dir: str = "outputs/model_visualizations"
) -> Dict[str, Any]:
    """
    Train Moving Average, Linear Regression, and Random Forest on train_df,
    validate on val_df and test_df, and save best artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    target_col = 'units_sold'
    feature_cols = DEMAND_NUMERIC_FEATURES + DEMAND_CATEGORICAL_FEATURES
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    # 1. Fit preprocessor strictly on training data
    preprocessor = build_demand_preprocessor()
    preprocessor.fit(train_df[feature_cols])
    
    X_train_proc = preprocessor.transform(train_df[feature_cols])
    X_val_proc = preprocessor.transform(val_df[feature_cols])
    X_test_proc = preprocessor.transform(test_df[feature_cols])
    
    # Extract feature names after transformation
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(DEMAND_CATEGORICAL_FEATURES).tolist()
    all_feature_names = DEMAND_NUMERIC_FEATURES + cat_names
    
    # 2. Moving Average Baseline
    baseline = MovingAverageBaseline(window=7)
    baseline.fit(train_df, train_df[target_col])
    baseline_val_pred = baseline.predict(val_df)
    baseline_test_pred = baseline.predict(test_df)
    
    # 3. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_proc, y_train)
    lr_val_pred = np.clip(lr.predict(X_val_proc), 0, None)
    lr_test_pred = np.clip(lr.predict(X_test_proc), 0, None)
    
    # 4. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_proc, y_train)
    rf_val_pred = np.clip(rf.predict(X_val_proc), 0, None)
    rf_test_pred = np.clip(rf.predict(X_test_proc), 0, None)
    
    # Compile evaluation metrics
    metrics = {
        'Baseline (Moving Average)': {
            'val': evaluate_regression(y_val, baseline_val_pred),
            'test': evaluate_regression(y_test, baseline_test_pred)
        },
        'Linear Regression': {
            'val': evaluate_regression(y_val, lr_val_pred),
            'test': evaluate_regression(y_test, lr_test_pred)
        },
        'Random Forest Regressor': {
            'val': evaluate_regression(y_val, rf_val_pred),
            'test': evaluate_regression(y_test, rf_test_pred)
        }
    }
    
    # Select best model based on held-out test MAE and RMSE
    best_name = 'Random Forest Regressor'
    best_model = rf
    if metrics['Linear Regression']['test']['mae'] < metrics['Random Forest Regressor']['test']['mae']:
        best_name = 'Linear Regression'
        best_model = lr
        
    # Feature importance from Random Forest
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # Save artifacts
    joblib.dump(preprocessor, os.path.join(models_dir, 'demand_preprocessor.joblib'))
    joblib.dump(baseline, os.path.join(models_dir, 'demand_baseline.joblib'))
    joblib.dump(lr, os.path.join(models_dir, 'demand_forecaster_lr.joblib'))
    joblib.dump(rf, os.path.join(models_dir, 'demand_forecaster_rf.joblib'))
    joblib.dump(best_model, os.path.join(models_dir, 'demand_best_model.joblib'))
    
    # Save feature importance CSV
    feat_imp_df.to_csv(os.path.join(output_dir, 'demand_feature_importance.csv'), index=False)
    
    # Visual 1: Model Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    model_names = list(metrics.keys())
    metric_keys = ['mae', 'rmse', 'wape']
    titles = ['MAE (Lower is Better)', 'RMSE (Lower is Better)', 'WAPE (Lower is Better)']
    
    for ax, metric, title in zip(axes, metric_keys, titles):
        val_vals = [metrics[m]['val'][metric] for m in model_names]
        test_vals = [metrics[m]['test'][metric] for m in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, val_vals, width, label='Validation', color='#337ab7')
        ax.bar(x + width/2, test_vals, width, label='Test', color='#5cb85c')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_model_comparison.png'), dpi=200)
    plt.close()
    
    # Visual 2: Feature Importance Top 12
    plt.figure(figsize=(10, 5))
    top_feats = feat_imp_df.head(12)
    sns.barplot(data=top_feats, x='importance', y='feature', color='#2b5c8f')
    plt.title("Random Forest: Top 12 Predictive Features for Demand", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Feature Importance", fontsize=10)
    plt.ylabel("Engineered Feature", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_feature_importance.png'), dpi=200)
    plt.close()
    
    # Visual 3: Actual vs Predicted Demand on Test Set
    plt.figure(figsize=(12, 5))
    test_dates = test_df['date']
    plt.plot(test_dates.values, y_test, label='Actual Test Demand', color='#d9534f', alpha=0.8, linewidth=1.5)
    plt.plot(test_dates.values, rf_test_pred, label='RF Predicted Demand', color='#0275d8', alpha=0.8, linestyle='--')
    plt.plot(test_dates.values, lr_test_pred, label='LR Predicted Demand', color='#f0ad4e', alpha=0.6, linestyle=':')
    plt.title("Held-Out Test Set: Actual vs Predicted Demand Over Time", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Units Sold", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_actual_vs_predicted.png'), dpi=200)
    plt.close()
    
    return {
        'metrics': metrics,
        'best_model_name': best_name,
        'feature_importance': feat_imp_df.to_dict(orient='records'),
        'feature_names': all_feature_names
    }


def generate_next_7_day_forecast(
    model,
    preprocessor,
    sales_df: pd.DataFrame,
    horizon: int = 7
) -> pd.DataFrame:
    """
    Generate dynamic forward-looking predictions for the next 'horizon' days
    for all 50 medicines using autoregressive lag rolling updates.
    """
    sales_df = sales_df.sort_values(by=['product_id', 'date']).copy()
    max_date = sales_df['date'].max()
    forecast_rows = []
    
    # Process product by product
    products = sales_df[['product_id', 'medicine_name', 'category', 'unit_price_inr']].drop_duplicates()
    
    for _, prod_row in products.iterrows():
        p_id = prod_row['product_id']
        p_name = prod_row['medicine_name']
        cat = prod_row['category']
        price = prod_row['unit_price_inr']
        
        prod_history = sales_df[sales_df['product_id'] == p_id].sort_values(by='date').copy()
        
        # Maintain a rolling queue of past sales values
        past_sales = list(prod_history['units_sold'].values)
        current_stock = float(prod_history['stock_on_hand'].iloc[-1])
        
        for step in range(1, horizon + 1):
            fc_date = max_date + pd.Timedelta(days=step)
            
            # Construct features dynamically
            lag_1 = past_sales[-1] if len(past_sales) >= 1 else 0.0
            lag_2 = past_sales[-2] if len(past_sales) >= 2 else 0.0
            lag_3 = past_sales[-3] if len(past_sales) >= 3 else 0.0
            lag_7 = past_sales[-7] if len(past_sales) >= 7 else 0.0
            lag_14 = past_sales[-14] if len(past_sales) >= 14 else 0.0
            lag_21 = past_sales[-21] if len(past_sales) >= 21 else 0.0
            
            # 7-day and 14-day rolling mean and std
            window_7 = past_sales[-7:] if len(past_sales) >= 7 else past_sales
            window_14 = past_sales[-14:] if len(past_sales) >= 14 else past_sales
            roll_mean_7 = float(np.mean(window_7)) if len(window_7) > 0 else 0.0
            roll_std_7 = float(np.std(window_7)) if len(window_7) > 0 else 0.0
            roll_mean_14 = float(np.mean(window_14)) if len(window_14) > 0 else 0.0
            roll_std_14 = float(np.std(window_14)) if len(window_14) > 0 else 0.0
            
            # Calendar attributes
            dow = fc_date.dayofweek
            dom = fc_date.day
            mon = fc_date.month
            is_wk = 1 if dow >= 5 else 0
            
            feat_dict = {
                'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3,
                'lag_7': lag_7, 'lag_14': lag_14, 'lag_21': lag_21,
                'roll_mean_7': roll_mean_7, 'roll_std_7': roll_std_7,
                'roll_mean_14': roll_mean_14, 'roll_std_14': roll_std_14,
                'stock_on_hand': current_stock,
                'unit_price_inr': price,
                'promotion': 0,
                'day_of_week_num': dow,
                'day_of_month': dom,
                'month': mon,
                'is_weekend': is_wk,
                'category': cat,
                'product_id': p_id
            }
            
            feat_df = pd.DataFrame([feat_dict])
            
            if isinstance(model, MovingAverageBaseline):
                pred = model.predict(feat_df)[0]
            else:
                proc_X = preprocessor.transform(feat_df[DEMAND_NUMERIC_FEATURES + DEMAND_CATEGORICAL_FEATURES])
                pred = model.predict(proc_X)[0]
                
            pred = max(0.0, float(pred))
            past_sales.append(pred)
            current_stock = max(0.0, current_stock - pred)
            
            forecast_rows.append({
                'product_id': p_id,
                'medicine_name': p_name,
                'category': cat,
                'unit_price_inr': price,
                'forecast_date': fc_date.strftime('%Y-%m-%d'),
                'forecast_day': step,
                'predicted_demand': round(pred, 4),
                'projected_stock_on_hand': round(current_stock, 1)
            })
            
    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df

