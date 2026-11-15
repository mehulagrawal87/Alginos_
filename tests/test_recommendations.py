"""
Unit tests for the Inventory Recommendation Engine.
"""

import pytest
import pandas as pd
from src.recommendation_engine import generate_inventory_recommendations, ADVISORY_DISCLAIMER


@pytest.fixture
def mock_inventory_and_forecast():
    inv_records = [
        {
            'snapshot_date': pd.Timestamp('2026-07-10'),
            'product_id': 'MED001',
            'medicine_name': 'Paracetamol 500mg',
            'category': 'Analgesic',
            'batch_id': 'BATCH-CRIT-01',
            'batch_quantity_remaining': 15,
            'days_to_expiry': 15,  # <= 30 days -> CRITICAL QA
            'projected_unsold_quantity': 15.0,
            'unit_price_inr': 2.5,
            'predicted_expiry_risk': 'High'
        },
        {
            'snapshot_date': pd.Timestamp('2026-07-10'),
            'product_id': 'MED002',
            'medicine_name': 'Ibuprofen 400mg',
            'category': 'Analgesic',
            'batch_id': 'BATCH-HIGH-01',
            'batch_quantity_remaining': 20,
            'days_to_expiry': 60,  # <= 90 days High Risk -> HIGH ROTATION
            'projected_unsold_quantity': 18.0,
            'unit_price_inr': 4.0,
            'predicted_expiry_risk': 'High'
        }
    ]
    inv_df = pd.DataFrame(inv_records)
    
    fc_records = [
        {'product_id': 'MED001', 'medicine_name': 'Paracetamol 500mg', 'category': 'Analgesic', 'unit_price_inr': 2.5, 'predicted_demand': 1.0},
        {'product_id': 'MED002', 'medicine_name': 'Ibuprofen 400mg', 'category': 'Analgesic', 'unit_price_inr': 4.0, 'predicted_demand': 0.0}
    ]
    fc_df = pd.DataFrame(fc_records)
    
    sales_records = [
        {'product_id': 'MED001', 'date': pd.Timestamp('2026-07-19'), 'stock_on_hand': 100},
        {'product_id': 'MED002', 'date': pd.Timestamp('2026-07-19'), 'stock_on_hand': 180}  # Overstock (>150 and low demand)
    ]
    sales_df = pd.DataFrame(sales_records)
    
    return inv_df, fc_df, sales_df


def test_recommendation_generation(mock_inventory_and_forecast):
    inv_df, fc_df, sales_df = mock_inventory_and_forecast
    recs = generate_inventory_recommendations(inv_df, fc_df, sales_df)
    
    assert isinstance(recs, pd.DataFrame)
    assert not recs.empty
    
    expected_cols = [
        'recommendation_id', 'product_id', 'medicine_name', 'priority',
        'alert_type', 'action_suggestion', 'value_at_risk_inr', 'advisory_disclaimer'
    ]
    for col in expected_cols:
        assert col in recs.columns


def test_near_expiry_qa_alert_trigger(mock_inventory_and_forecast):
    inv_df, fc_df, sales_df = mock_inventory_and_forecast
    recs = generate_inventory_recommendations(inv_df, fc_df, sales_df)
    
    qa_alerts = recs[recs['alert_type'] == 'NEAR_EXPIRY_QA']
    assert len(qa_alerts) >= 1
    assert qa_alerts.iloc[0]['priority'] == 'CRITICAL'
    assert qa_alerts.iloc[0]['days_to_expiry'] <= 30


def test_advisory_disclaimer_present(mock_inventory_and_forecast):
    inv_df, fc_df, sales_df = mock_inventory_and_forecast
    recs = generate_inventory_recommendations(inv_df, fc_df, sales_df)
    
    for disclaimer in recs['advisory_disclaimer']:
        assert "Advisory Notice" in disclaimer
        assert "operational inventory" in disclaimer

