"""
Inventory Recommendation Engine Module.

Synthesizes demand forecasts, batch expiry risk classifications, and warehouse stock
to generate prioritized, operational inventory recommendations and alerts.

DISCLAIMER:
All generated outputs are strictly operational and logistical supply-chain guidance.
They do NOT provide medical, clinical, prescribing, or treatment advice.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


ADVISORY_DISCLAIMER = (
    "Advisory Notice: These recommendations represent operational inventory "
    "decision-support guidance only. They do not constitute medical, clinical, "
    "or therapeutic advice. All medicine transfers and purchases must comply with "
    "applicable pharmacy regulations and human pharmacist review."
)


def generate_inventory_recommendations(
    inventory_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    latest_sales_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate prioritized inventory operational recommendations by cross-referencing:
    1. Batch-level predicted expiry risks and days to expiry
    2. Product-level projected 7-day demand
    3. Current warehouse stock levels
    
    Args:
        inventory_df: DataFrame of batch records with 'predicted_expiry_risk'
        forecast_df: Next 7-day demand forecasts from demand model
        latest_sales_df: Optional latest sales snapshot for current stock_on_hand
        
    Returns:
        DataFrame of actionable operational recommendations with priority ratings.
    """
    # Aggregate 7-day demand by product
    agg_demand = forecast_df.groupby('product_id').agg(
        total_7d_forecast=('predicted_demand', 'sum'),
        medicine_name=('medicine_name', 'first'),
        category=('category', 'first'),
        unit_price_inr=('unit_price_inr', 'first')
    ).reset_index()
    
    # Get current stock on hand from latest sales date if available
    if latest_sales_df is not None:
        latest_date = latest_sales_df['date'].max()
        current_stock_df = latest_sales_df[latest_sales_df['date'] == latest_date][
            ['product_id', 'stock_on_hand']
        ].drop_duplicates()
        agg_demand = agg_demand.merge(current_stock_df, on='product_id', how='left')
        agg_demand['stock_on_hand'] = agg_demand['stock_on_hand'].fillna(100.0)
    else:
        agg_demand['stock_on_hand'] = 100.0

    recommendations = []
    
    # 1. Product-level Replenishment & Overstock Checks
    for _, prod in agg_demand.iterrows():
        p_id = prod['product_id']
        p_name = prod['medicine_name']
        cat = prod['category']
        stock = prod['stock_on_hand']
        fc_7d = prod['total_7d_forecast']
        price = prod['unit_price_inr']
        
        # Check Replenishment Alert
        # If stock is below 15 units or stock <= fc_7d
        if stock <= max(10, fc_7d * 1.5) and fc_7d > 0:
            recommendations.append({
                'recommendation_id': f"REC-REP-{p_id}",
                'product_id': p_id,
                'medicine_name': p_name,
                'batch_id': 'ALL_BATCHES',
                'category': cat,
                'priority': 'CRITICAL' if stock <= 5 else 'HIGH',
                'alert_type': 'REPLENISHMENT',
                'current_stock': round(stock, 1),
                'predicted_7d_demand': round(fc_7d, 2),
                'days_to_expiry': None,
                'predicted_risk': 'N/A',
                'action_suggestion': (
                    f"Low stock on hand ({int(stock)} units) relative to forecast demand ({fc_7d:.1f} units). "
                    f"Review purchase order to avoid stockout."
                ),
                'value_at_risk_inr': 0.0,
                'advisory_disclaimer': ADVISORY_DISCLAIMER
            })
            
        # Check Overstock / Capital Lockup Alert
        # In this dataset, many medicines have stock ~150-200 with ~0 daily sales
        if stock > 150 and fc_7d < 0.5:
            recommendations.append({
                'recommendation_id': f"REC-OVR-{p_id}",
                'product_id': p_id,
                'medicine_name': p_name,
                'batch_id': 'ALL_BATCHES',
                'category': cat,
                'priority': 'LOW',
                'alert_type': 'OVERSTOCK',
                'current_stock': round(stock, 1),
                'predicted_7d_demand': round(fc_7d, 2),
                'days_to_expiry': None,
                'predicted_risk': 'N/A',
                'action_suggestion': (
                    f"Excess warehouse inventory ({int(stock)} units) with low turnover. "
                    f"Pause future procurement to prevent capital lockup and potential expiry accumulation."
                ),
                'value_at_risk_inr': round(stock * price, 2),
                'advisory_disclaimer': ADVISORY_DISCLAIMER
            })
            
    # 2. Batch-level Expiry & Stock Rotation Checks
    # Look at latest batch records (e.g. from the most recent snapshot date)
    latest_snapshot = inventory_df['snapshot_date'].max()
    latest_batches = inventory_df[inventory_df['snapshot_date'] == latest_snapshot].copy()
    
    risk_col = 'predicted_expiry_risk' if 'predicted_expiry_risk' in latest_batches.columns else 'expiry_risk'
    
    for _, batch in latest_batches.iterrows():
        b_id = batch['batch_id']
        p_id = batch['product_id']
        p_name = batch['medicine_name']
        cat = batch['category']
        b_qty = batch['batch_quantity_remaining']
        days_exp = batch['days_to_expiry']
        risk = batch[risk_col]
        price = batch['unit_price_inr']
        unsold_qty = batch.get('projected_unsold_quantity', b_qty)
        val_at_risk = round(unsold_qty * price, 2)
        
        # Priority 1: Near-Expiry Immediate QA Review (< 30 days)
        if days_exp <= 30:
            recommendations.append({
                'recommendation_id': f"REC-QA-{b_id}",
                'product_id': p_id,
                'medicine_name': p_name,
                'batch_id': b_id,
                'category': cat,
                'priority': 'CRITICAL',
                'alert_type': 'NEAR_EXPIRY_QA',
                'current_stock': int(b_qty),
                'predicted_7d_demand': 0.0,
                'days_to_expiry': int(days_exp),
                'predicted_risk': risk,
                'action_suggestion': (
                    f"Batch expires in {int(days_exp)} days. Perform immediate physical pharmacist inspection, "
                    f"verify packaging integrity, and quarantine or return to distributor if unfeasible to dispense."
                ),
                'value_at_risk_inr': val_at_risk,
                'advisory_disclaimer': ADVISORY_DISCLAIMER
            })
            
        # Priority 2: High Expiry Risk Stock Rotation (31-90 days, High Risk)
        elif risk == 'High' and days_exp <= 90:
            recommendations.append({
                'recommendation_id': f"REC-ROT-{b_id}",
                'product_id': p_id,
                'medicine_name': p_name,
                'batch_id': b_id,
                'category': cat,
                'priority': 'HIGH',
                'alert_type': 'STOCK_ROTATION',
                'current_stock': int(b_qty),
                'predicted_7d_demand': 0.0,
                'days_to_expiry': int(days_exp),
                'predicted_risk': risk,
                'action_suggestion': (
                    f"High expiry risk detected with {int(days_exp)} days remaining ({int(b_qty)} units). "
                    f"Apply First-Expired, First-Out (FEFO) dispensing, evaluate promotional discount, or initiate branch transfer."
                ),
                'value_at_risk_inr': val_at_risk,
                'advisory_disclaimer': ADVISORY_DISCLAIMER
            })
            
        # Priority 3: Medium Expiry Risk Monitoring (91-180 days, Medium Risk)
        elif risk == 'Medium':
            recommendations.append({
                'recommendation_id': f"REC-MON-{b_id}",
                'product_id': p_id,
                'medicine_name': p_name,
                'batch_id': b_id,
                'category': cat,
                'priority': 'MEDIUM',
                'alert_type': 'EXPIRY_MONITORING',
                'current_stock': int(b_qty),
                'predicted_7d_demand': 0.0,
                'days_to_expiry': int(days_exp),
                'predicted_risk': risk,
                'action_suggestion': (
                    f"Medium risk tier: {int(days_exp)} days to expiry with {int(b_qty)} units remaining. "
                    f"Schedule for bi-weekly inventory audit and monitor sell-through velocity."
                ),
                'value_at_risk_inr': val_at_risk,
                'advisory_disclaimer': ADVISORY_DISCLAIMER
            })
            
    rec_df = pd.DataFrame(recommendations)
    
    # Sort by priority order: CRITICAL, HIGH, MEDIUM, LOW
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'NORMAL': 4}
    if not rec_df.empty:
        rec_df['priority_rank'] = rec_df['priority'].map(priority_order)
        rec_df = rec_df.sort_values(by=['priority_rank', 'value_at_risk_inr'], ascending=[True, False]).drop(columns=['priority_rank']).reset_index(drop=True)
        
    return rec_df

