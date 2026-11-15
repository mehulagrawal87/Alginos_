"""
Model Evaluation and Diagnostics Module.

Consolidates performance metrics, residual diagnostics, error analysis,
and model trade-off documentation across both Demand Forecasting and
Expiry-Risk Classification.
"""

import os
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def generate_evaluation_summary(
    demand_metrics: Dict[str, Any],
    expiry_metrics: Dict[str, Any],
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Produce a consolidated evaluation dictionary and save unified performance reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Format Regression Metrics Table
    reg_rows = []
    for model_name, splits in demand_metrics.items():
        reg_rows.append({
            'Model': model_name,
            'Val MAE': splits['val']['mae'],
            'Val RMSE': splits['val']['rmse'],
            'Val WAPE': splits['val']['wape'],
            'Test MAE': splits['test']['mae'],
            'Test RMSE': splits['test']['rmse'],
            'Test WAPE': splits['test']['wape']
        })
    reg_summary_df = pd.DataFrame(reg_rows)
    reg_summary_df.to_csv(os.path.join(output_dir, "demand_evaluation_metrics.csv"), index=False)
    
    # 2. Format Classification Metrics Table
    clf_rows = []
    for model_name, splits in expiry_metrics.items():
        clf_rows.append({
            'Model': model_name,
            'Val Accuracy': splits['val']['accuracy'],
            'Val Macro F1': splits['val']['f1_macro'],
            'Val Macro Recall': splits['val']['recall_macro'],
            'Test Accuracy': splits['test']['accuracy'],
            'Test Macro Precision': splits['test']['precision_macro'],
            'Test Macro Recall': splits['test']['recall_macro'],
            'Test Macro F1': splits['test']['f1_macro'],
            'Test Weighted F1': splits['test']['f1_weighted']
        })
    clf_summary_df = pd.DataFrame(clf_rows)
    clf_summary_df.to_csv(os.path.join(output_dir, "expiry_evaluation_metrics.csv"), index=False)
    
    # 3. Model Limitations and Trade-offs
    limitations = {
        'demand_forecasting': {
            'data_sparsity': (
                "Extreme zero-inflation (99.98% zero sales across historical 200 days). "
                "While models learn low average demand accurately, point forecasts for sudden "
                "demand spikes in rare medicines have inherent uncertainty."
            ),
            'horizon_decay': (
                "Autoregressive multi-step forecasting compounds errors over longer horizons "
                "(> 14 days). Re-training or weekly batch refreshes are recommended."
            ),
            'exogenous_factors': (
                "The synthetic dataset lacks epidemiological disease outbreak feeds, seasonal epidemic "
                "indicators, or doctor prescription flow data that drive real-world pharmacy surges."
            )
        },
        'expiry_risk_classification': {
            'synthetic_rule_proxy': (
                "CRITICAL METHODOLOGY NOTE: The 'expiry_risk' ground-truth label was constructed via "
                "a deterministic heuristic rule (days_to_expiry <= 90 & unsold_ratio >= 0.30). "
                "High model accuracy reflects learning this heuristic surface, NOT predicting "
                "unobserved real-world clinical wastage."
            ),
            'static_snapshots': (
                "Batch inventory records are synthetic snapshots taken every 8 days. In live operations, "
                "real-time barcode scans and FIFO/FEFO dispense events provide continuous state updates."
            ),
            'advisory_guardrail': (
                "Outputs serve strictly as human decision support for pharmacy inventory managers and "
                "must never be used for autonomous disposal or clinical treatment substitution."
            )
        }
    }
    
    summary = {
        'demand_forecasting': demand_metrics,
        'expiry_classification': expiry_metrics,
        'limitations': limitations
    }
    
    with open(os.path.join(output_dir, "evaluation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    return summary

