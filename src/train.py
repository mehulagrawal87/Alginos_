"""
Master Training Pipeline Orchestrator.

Executes the complete reproducible pipeline from raw CSV data to validated,
production-ready model artifacts, evaluation reports, and operational recommendations.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    load_sales_data,
    load_inventory_data,
    clean_sales_data,
    clean_inventory_data,
    engineer_demand_features,
    split_chronological_demand,
    split_inventory_data,
    DEMAND_NUMERIC_FEATURES,
    DEMAND_CATEGORICAL_FEATURES,
    EXPIRY_NUMERIC_FEATURES,
    EXPIRY_CATEGORICAL_FEATURES
)
from src.eda import run_full_eda
from src.demand_forecasting import (
    train_demand_models,
    generate_next_7_day_forecast
)
from src.expiry_risk import (
    train_expiry_models,
    annotate_inventory_predictions
)
from src.recommendation_engine import generate_inventory_recommendations
from src.model_evaluation import generate_evaluation_summary
import joblib


def run_pipeline(
    sales_path: str = "dataset/daily_medicine_sales.csv",
    inventory_path: str = "dataset/batch_expiry_inventory.csv",
    models_dir: str = "models",
    outputs_dir: str = "outputs"
):
    print("=" * 70)
    print("PHARMACY DEMAND & EXPIRY-RISK FORECASTING: END-TO-END PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Stage 1: Load and Clean Datasets
    # ---------------------------------------------------------
    print("\n[Stage 1/6] Loading and validating raw datasets...")
    sales_raw = load_sales_data(sales_path)
    inventory_raw = load_inventory_data(inventory_path)
    print(f"  Loaded Sales records: {sales_raw.shape[0]:,} rows across {sales_raw['product_id'].nunique()} medicines")
    print(f"  Loaded Inventory records: {inventory_raw.shape[0]:,} batch records across {inventory_raw['snapshot_date'].nunique()} snapshot dates")
    
    sales_clean = clean_sales_data(sales_raw)
    inventory_clean = clean_inventory_data(inventory_raw)
    
    # ---------------------------------------------------------
    # Stage 2: Exploratory Data Analysis & Visualization
    # ---------------------------------------------------------
    print("\n[Stage 2/6] Running Exploratory Data Analysis (EDA)...")
    eda_summary = run_full_eda(sales_clean, inventory_clean, output_dir=os.path.join(outputs_dir, "eda_visualizations"))
    print("  Generated 7 publication-ready visual charts in outputs/eda_visualizations/")
    
    # ---------------------------------------------------------
    # Stage 3: Demand Forecasting Model Training & Evaluation
    # ---------------------------------------------------------
    print("\n[Stage 3/6] Engineering time-series features & training Demand Models...")
    sales_feat = engineer_demand_features(sales_clean)
    train_d, val_d, test_d = split_chronological_demand(sales_feat)
    print(f"  Chronological splits: Train={train_d.shape[0]}, Val={val_d.shape[0]}, Test={test_d.shape[0]}")
    
    demand_res = train_demand_models(
        train_d, val_d, test_d,
        models_dir=models_dir,
        output_dir=os.path.join(outputs_dir, "model_visualizations")
    )
    print(f"  Demand Best Model Selected: {demand_res['best_model_name']}")
    for m_name, m_metrics in demand_res['metrics'].items():
        print(f"    - {m_name:25s} | Test MAE: {m_metrics['test']['mae']:.6f} | RMSE: {m_metrics['test']['rmse']:.6f} | WAPE: {m_metrics['test']['wape']:.6f}")
        
    # Generate 7-day forecast
    best_demand_model = joblib.load(os.path.join(models_dir, 'demand_best_model.joblib'))
    demand_preprocessor = joblib.load(os.path.join(models_dir, 'demand_preprocessor.joblib'))
    forecast_df = generate_next_7_day_forecast(best_demand_model, demand_preprocessor, sales_clean, horizon=7)
    forecast_path = os.path.join(outputs_dir, "demand_forecast_results.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"  Generated next 7-day demand forecast for all 50 medicines: {forecast_path}")
    
    # ---------------------------------------------------------
    # Stage 4: Expiry-Risk Classification Training & Evaluation
    # ---------------------------------------------------------
    print("\n[Stage 4/6] Training Expiry-Risk Classification Models...")
    train_i, val_i, test_i = split_inventory_data(inventory_clean)
    print(f"  Chronological snapshot splits: Train={train_i.shape[0]}, Val={val_i.shape[0]}, Test={test_i.shape[0]}")
    
    expiry_res = train_expiry_models(
        train_i, val_i, test_i,
        models_dir=models_dir,
        output_dir=os.path.join(outputs_dir, "model_visualizations")
    )
    print(f"  Expiry Best Model Selected: {expiry_res['best_model_name']}")
    for m_name, m_metrics in expiry_res['metrics'].items():
        print(f"    - {m_name:25s} | Test Acc: {m_metrics['test']['accuracy']:.4f} | Macro F1: {m_metrics['test']['f1_macro']:.4f} | Macro Rec: {m_metrics['test']['recall_macro']:.4f}")
        
    # Annotate full inventory
    best_expiry_model = joblib.load(os.path.join(models_dir, 'expiry_best_model.joblib'))
    expiry_preprocessor = joblib.load(os.path.join(models_dir, 'expiry_preprocessor.joblib'))
    annotated_inv = annotate_inventory_predictions(inventory_clean, best_expiry_model, expiry_preprocessor)
    annotated_inv_path = os.path.join(outputs_dir, "expiry_risk_predictions.csv")
    annotated_inv.to_csv(annotated_inv_path, index=False)
    print(f"  Annotated batch inventory with model predictions: {annotated_inv_path}")
    
    # ---------------------------------------------------------
    # Stage 5: Inventory Recommendation Engine
    # ---------------------------------------------------------
    print("\n[Stage 5/6] Generating Operational Inventory Recommendations...")
    recs_df = generate_inventory_recommendations(annotated_inv, forecast_df, sales_clean)
    recs_path = os.path.join(outputs_dir, "inventory_recommendations.csv")
    recs_df.to_csv(recs_path, index=False)
    print(f"  Generated {len(recs_df)} operational recommendations: {recs_path}")
    print(f"  Priority Breakdown: {recs_df['priority'].value_counts().to_dict()}")
    
    # ---------------------------------------------------------
    # Stage 6: Consolidated Evaluation & Metadata
    # ---------------------------------------------------------
    print("\n[Stage 6/6] Consolidating Evaluation Metrics and Model Metadata...")
    eval_summary = generate_evaluation_summary(demand_res['metrics'], expiry_res['metrics'], output_dir=outputs_dir)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'project': "Pharmacy Medicine Demand & Expiry-Risk Forecasting System",
        'organization': "Algions AI Labs Private Limited",
        'random_seed': 42,
        'demand_model': {
            'best_model': demand_res['best_model_name'],
            'features_numeric': DEMAND_NUMERIC_FEATURES,
            'features_categorical': DEMAND_CATEGORICAL_FEATURES,
            'metrics_test': demand_res['metrics'][demand_res['best_model_name']]['test']
        },
        'expiry_model': {
            'best_model': expiry_res['best_model_name'],
            'features_numeric': EXPIRY_NUMERIC_FEATURES,
            'features_categorical': EXPIRY_CATEGORICAL_FEATURES,
            'metrics_test': expiry_res['metrics'][expiry_res['best_model_name']]['test']
        },
        'artifacts_saved': [
            "models/demand_preprocessor.joblib",
            "models/demand_baseline.joblib",
            "models/demand_forecaster_lr.joblib",
            "models/demand_forecaster_rf.joblib",
            "models/demand_best_model.joblib",
            "models/expiry_preprocessor.joblib",
            "models/expiry_baseline.joblib",
            "models/expiry_classifier_lr.joblib",
            "models/expiry_classifier_rf.joblib",
            "models/expiry_best_model.joblib"
        ]
    }
    
    with open(os.path.join(models_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

