"""
Streamlit Web Dashboard: Pharmacy Medicine Demand & Expiry-Risk Forecasting System
Algions AI Labs Private Limited - Professional Edition
Subtle, elegant healthcare enterprise styling with 100% native Streamlit stability.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.data_preprocessing import (
    load_sales_data,
    load_inventory_data,
    DEMAND_NUMERIC_FEATURES,
    DEMAND_CATEGORICAL_FEATURES,
    EXPIRY_NUMERIC_FEATURES,
    EXPIRY_CATEGORICAL_FEATURES
)
from src.demand_forecasting import generate_next_7_day_forecast
from src.expiry_risk import predict_batch_risk
from src.recommendation_engine import ADVISORY_DISCLAIMER

# -------------------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="PharmaPredict AI | Algions AI Labs",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust display helpers to handle version variations cleanly
def render_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except (TypeError, ValueError):
        st.dataframe(df, use_container_width=True, **kwargs)

def render_chart(fig, **kwargs):
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except (TypeError, ValueError):
        st.plotly_chart(fig, use_container_width=True, **kwargs)

# -------------------------------------------------------------
# Subtle, Tasteful Professional CSS
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Clean base page styling */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0f2742 0%, #1e4265 100%);
        padding: 20px 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 8px rgba(15, 39, 66, 0.08);
    }
    .header-title {
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        color: white;
    }
    .header-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .header-badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #38bdf8;
    }

    /* Native Metric Card Elevation */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Advisory Box */
    .advisory-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 8px;
        color: #92400e;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    /* Recommendation Cards */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .badge-critical { background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-high { background-color: #ffedd5; color: #9a3412; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-medium { background-color: #fef9c3; color: #854d0e; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    .badge-low { background-color: #ecfdf5; color: #065f46; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; }
    
    /* Subtle section titles */
    .section-heading {
        font-size: 15px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Data & Model Loaders (Cached)
# -------------------------------------------------------------
@st.cache_data
def load_all_datasets():
    sales = load_sales_data("dataset/daily_medicine_sales.csv")
    if os.path.exists("outputs/expiry_risk_predictions.csv"):
        inventory = pd.read_csv("outputs/expiry_risk_predictions.csv")
        inventory['snapshot_date'] = pd.to_datetime(inventory['snapshot_date'])
        inventory['expiry_date'] = pd.to_datetime(inventory['expiry_date'])
    else:
        inventory = load_inventory_data("dataset/batch_expiry_inventory.csv")
        
    forecast = pd.read_csv("outputs/demand_forecast_results.csv") if os.path.exists("outputs/demand_forecast_results.csv") else None
    recs = pd.read_csv("outputs/inventory_recommendations.csv") if os.path.exists("outputs/inventory_recommendations.csv") else None
    
    return sales, inventory, forecast, recs


@st.cache_resource
def load_all_models():
    demand_model = joblib.load("models/demand_best_model.joblib")
    demand_prep = joblib.load("models/demand_preprocessor.joblib")
    expiry_model = joblib.load("models/expiry_best_model.joblib")
    expiry_prep = joblib.load("models/expiry_preprocessor.joblib")
    
    metadata = {}
    if os.path.exists("models/model_metadata.json"):
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
            
    return demand_model, demand_prep, expiry_model, expiry_prep, metadata


# Load cached resources
sales_df, inventory_df, forecast_df, recs_df = load_all_datasets()
demand_model, demand_prep, expiry_model, expiry_prep, model_meta = load_all_models()


# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💊 **PharmaPredict AI**")
    st.caption("Pharmacy Decision Support System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        [
            "1. Executive Overview",
            "2. Demand Forecasting",
            "3. Expiry-Risk Analysis",
            "4. Inventory Recommendations",
            "5. Model Performance",
            "6. About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### **System Status**")
    st.write(f"• **Catalog:** {sales_df['product_id'].nunique()} Medicines")
    st.write(f"• **Batches:** {len(inventory_df):,} Tracked")
    st.write(f"• **ML Engine:** Online (Joblib)")
    st.caption("Algions AI Labs Assessment Project")


# Top Header Banner
st.markdown("""
<div class="header-banner">
    <div>
        <div class="header-title">Pharmacy Medicine Demand & Expiry-Risk Forecasting System</div>
        <div class="header-subtitle">Machine Learning Decision Support Platform for Inventory Optimization & Waste Prevention</div>
    </div>
    <div class="header-badge">● Models Active</div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================
if page == "1. Executive Overview":
    st.subheader("Executive Overview")
    st.markdown("High-level key performance indicators across pharmacy sales velocity and warehouse batch expiration risks.")
    
    risk_col = 'predicted_expiry_risk' if 'predicted_expiry_risk' in inventory_df.columns else 'expiry_risk'
    high_risk_count = int((inventory_df[risk_col] == 'High').sum())
    total_stock_units = int(sales_df.groupby('date')['stock_on_hand'].sum().iloc[-1])
    total_predicted_7d = float(forecast_df['predicted_demand'].sum()) if forecast_df is not None else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Catalog Medicines", f"{sales_df['product_id'].nunique()}", "50 SKUs")
    k2.metric("Historical Records", f"{len(sales_df):,}", "200 Days")
    k3.metric("Warehouse Stock", f"{total_stock_units:,}", "Units on Hand")
    k4.metric("Active Batches", f"{len(inventory_df):,}", "Monitored")
    k5.metric("High Expiry Risk", f"{high_risk_count}", "Flagged Batches", delta_color="inverse")
    k6.metric("7-Day Forecast", f"{total_predicted_7d:.1f}", "Units Projected")

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([7, 5])
    
    with chart_col1:
        st.markdown('<div class="section-heading">📈 200-Day Daily Sales Velocity & Stock Evolution</div>', unsafe_allow_html=True)
        daily_agg = sales_df.groupby('date').agg({'units_sold': 'sum', 'stock_on_hand': 'sum'}).reset_index()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_agg['date'],
            y=daily_agg['units_sold'],
            mode='lines+markers',
            name='Daily Units Sold',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=4)
        ))
        fig_trend.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        render_chart(fig_trend)

    with chart_col2:
        st.markdown('<div class="section-heading">🛡️ Batch Expiry-Risk Class Distribution</div>', unsafe_allow_html=True)
        risk_dist = inventory_df[risk_col].value_counts().reset_index()
        risk_dist.columns = ['Risk Tier', 'Batch Count']
        
        fig_donut = px.pie(
            risk_dist,
            names='Risk Tier',
            values='Batch Count',
            hole=0.60,
            color='Risk Tier',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        )
        fig_donut.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        render_chart(fig_donut)

    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns([6, 6])
    
    with row2_col1:
        st.markdown('<div class="section-heading">🏆 Top Medicines by Units Sold</div>', unsafe_allow_html=True)
        top_meds = sales_df.groupby(['medicine_name', 'category'])['units_sold'].sum().reset_index()
        top_meds = top_meds.sort_values(by='units_sold', ascending=False).head(10)
        
        fig_top = px.bar(
            top_meds,
            x='units_sold',
            y='medicine_name',
            orientation='h',
            color='category',
            labels={'units_sold': 'Units Sold', 'medicine_name': 'Medicine'}
        )
        fig_top.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        render_chart(fig_top)
        
    with row2_col2:
        st.markdown('<div class="section-heading">⚠️ Near-Expiry Batches (≤ 30 Days Left)</div>', unsafe_allow_html=True)
        near_exp = inventory_df[inventory_df['days_to_expiry'] <= 30][
            ['batch_id', 'medicine_name', 'category', 'batch_quantity_remaining', 'days_to_expiry', risk_col]
        ].sort_values(by='days_to_expiry').head(7)
        
        render_df(
            near_exp.rename(columns={
                'batch_id': 'Batch ID',
                'medicine_name': 'Medicine',
                'category': 'Category',
                'batch_quantity_remaining': 'Stock Left',
                'days_to_expiry': 'Days Left',
                risk_col: 'Risk Tier'
            }),
            hide_index=True
        )


# =============================================================
# PAGE 2: DEMAND FORECASTING
# =============================================================
elif page == "2. Demand Forecasting":
    st.subheader("Machine Learning Demand Forecasting")
    st.markdown("Forward-looking medicine demand forecasts generated by the trained **Random Forest Regressor**.")
    
    med_list = sales_df[['product_id', 'medicine_name']].drop_duplicates().sort_values(by='medicine_name')
    med_dict = dict(zip(med_list['medicine_name'], med_list['product_id']))
    
    sel_c1, sel_c2 = st.columns([3, 1])
    with sel_c1:
        sel_med_name = st.selectbox("Select Medicine to Forecast", options=list(med_dict.keys()))
    with sel_c2:
        horizon = st.slider("Forecast Horizon (Days)", min_value=1, max_value=14, value=7)
        
    sel_pid = med_dict[sel_med_name]
    med_sales = sales_df[sales_df['product_id'] == sel_pid].sort_values(by='date')
    med_cat = med_sales['category'].iloc[0]
    med_price = med_sales['unit_price_inr'].iloc[0]
    med_cur_stock = med_sales['stock_on_hand'].iloc[-1]
    
    # Real-Time Forecast
    with st.spinner("Generating autoregressive forecast with Random Forest model..."):
        custom_fc = generate_next_7_day_forecast(demand_model, demand_prep, med_sales, horizon=horizon)
        
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Category", med_cat)
    m2.metric("Unit Price", f"₹ {med_price:.2f}")
    m3.metric("Current Stock on Hand", f"{int(med_cur_stock)} units")
    m4.metric(f"{horizon}-Day Projected Demand", f"{custom_fc['predicted_demand'].sum():.2f} units")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Historical Sales vs Forward Forecast</div>', unsafe_allow_html=True)
    recent_history = med_sales.tail(45)
    
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=recent_history['date'],
        y=recent_history['units_sold'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1e293b', width=2),
        marker=dict(size=4)
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.to_datetime(custom_fc['forecast_date']),
        y=custom_fc['predicted_demand'],
        mode='lines+markers',
        name='Random Forest Forecast',
        line=dict(color='#0284c7', width=3, dash='dash'),
        marker=dict(size=7, symbol='diamond')
    ))
    fig_fc.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="Units Demanded",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    render_chart(fig_fc)
    
    f_col1, f_col2 = st.columns([7, 5])
    with f_col1:
        st.markdown(f'<div class="section-heading">📋 {horizon}-Day Forecast Breakdown</div>', unsafe_allow_html=True)
        render_df(
            custom_fc[['forecast_date', 'forecast_day', 'predicted_demand', 'projected_stock_on_hand']].rename(columns={
                'forecast_date': 'Date',
                'forecast_day': 'Day Ahead',
                'predicted_demand': 'Projected Demand (Units)',
                'projected_stock_on_hand': 'Remaining Stock'
            }),
            hide_index=True
        )
        
        csv_bytes = custom_fc.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_bytes,
            file_name=f"demand_forecast_{sel_pid}.csv",
            mime="text/csv"
        )
        
    with f_col2:
        st.markdown('<div class="section-heading">🔬 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        if os.path.exists("outputs/model_visualizations/demand_feature_importance.csv"):
            feat_imp = pd.read_csv("outputs/model_visualizations/demand_feature_importance.csv").head(8)
            fig_imp = px.bar(
                feat_imp,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed"),
                template="plotly_white",
                coloraxis_showscale=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            render_chart(fig_imp)

    # What-If Demand Simulator
    st.markdown("---")
    st.markdown("##### 🧪 Real-Time What-If Demand Simulator")
    st.caption("Adjust operational parameters to test model sensitivity to promotion pricing and stock levels.")
    
    w1, w2, w3 = st.columns(3)
    sim_price = w1.number_input("Test Unit Price (₹)", min_value=1.0, max_value=200.0, value=float(med_price), step=1.0)
    sim_promo = w2.selectbox("Promotion Campaign?", options=[0, 1], format_func=lambda x: "Active (Discounted)" if x == 1 else "Standard")
    sim_stock = w3.number_input("Warehouse Stock on Hand", min_value=0, max_value=500, value=int(med_cur_stock))
    
    sim_dict = {
        'lag_1': float(recent_history['units_sold'].iloc[-1]),
        'lag_2': float(recent_history['units_sold'].iloc[-2]),
        'lag_3': float(recent_history['units_sold'].iloc[-3]),
        'lag_7': float(recent_history['units_sold'].iloc[-7]),
        'lag_14': float(recent_history['units_sold'].iloc[-14]),
        'lag_21': float(recent_history['units_sold'].iloc[-21]),
        'roll_mean_7': float(recent_history['units_sold'].tail(7).mean()),
        'roll_std_7': float(recent_history['units_sold'].tail(7).std()),
        'roll_mean_14': float(recent_history['units_sold'].tail(14).mean()),
        'roll_std_14': float(recent_history['units_sold'].tail(14).std()),
        'stock_on_hand': float(sim_stock),
        'unit_price_inr': float(sim_price),
        'promotion': int(sim_promo),
        'day_of_week_num': datetime.now().weekday(),
        'day_of_month': datetime.now().day,
        'month': datetime.now().month,
        'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
        'category': med_cat
    }
    sim_df = pd.DataFrame([sim_dict])
    proc_sim = demand_prep.transform(sim_df[DEMAND_NUMERIC_FEATURES + DEMAND_CATEGORICAL_FEATURES])
    sim_pred = demand_model.predict(proc_sim)[0]
    
    st.info(f"💡 **Live Inference Result:** Model predicts daily demand of **{max(0.0, float(sim_pred)):.4f} units/day** for {sel_med_name}.")


# =============================================================
# PAGE 3: EXPIRY-RISK ANALYSIS
# =============================================================
elif page == "3. Expiry-Risk Analysis":
    st.subheader("Batch Expiry-Risk Analysis")
    st.markdown("Machine learning identification of medicine batches susceptible to expiring before dispensing.")
    
    st.markdown("""
    <div class="advisory-box">
        <strong>⚠️ Synthetic Heuristic Methodology Notice:</strong> The ground truth label <em>expiry_risk</em> was created using an operational rule:
        (<em>High</em>: days to expiry &le; 90 and projected unsold ratio &ge; 30%; <em>Medium</em>: &le; 180 days and &ge; 15%).
        The Random Forest Classifier captures this risk surface with 100% test accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    fc1, fc2, fc3 = st.columns([3, 3, 4])
    risk_col = 'predicted_expiry_risk' if 'predicted_expiry_risk' in inventory_df.columns else 'expiry_risk'
    
    with fc1:
        risk_filter = st.multiselect("Filter Risk Tier", options=['High', 'Medium', 'Low'], default=['High', 'Medium'])
    with fc2:
        cat_filter = st.multiselect("Filter Category", options=sorted(inventory_df['category'].unique()), default=[])
    with fc3:
        search_kw = st.text_input("🔍 Search Medicine or Batch Code", placeholder="e.g. Paracetamol or MED001-B004")
        
    filtered = inventory_df.copy()
    if risk_filter:
        filtered = filtered[filtered[risk_col].isin(risk_filter)]
    if cat_filter:
        filtered = filtered[filtered['category'].isin(cat_filter)]
    if search_kw:
        filtered = filtered[
            filtered['medicine_name'].str.contains(search_kw, case=False, na=False) |
            filtered['batch_id'].str.contains(search_kw, case=False, na=False)
        ]
        
    st.markdown(f"**Found {len(filtered)} matching batch records:**")
    render_df(
        filtered[['batch_id', 'medicine_name', 'category', 'batch_quantity_remaining', 'days_to_expiry', 'expiry_date', 'projected_unsold_quantity', 'projected_unsold_ratio', risk_col]].rename(columns={
            'batch_id': 'Batch Code',
            'medicine_name': 'Medicine Name',
            'category': 'Category',
            'batch_quantity_remaining': 'Stock Left',
            'days_to_expiry': 'Days Left',
            'expiry_date': 'Expiry Date',
            'projected_unsold_quantity': 'Unsold Qty',
            'projected_unsold_ratio': 'Unsold Ratio',
            risk_col: 'AI Risk Class'
        }),
        hide_index=True
    )
    
    # Interactive Batch Risk Calculator
    st.markdown("---")
    st.markdown("##### 🔬 Interactive Batch Risk Inference Calculator")
    st.caption("Input custom batch parameters to run real-time inference using the saved **Random Forest Classifier**.")
    
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        calc_cat = st.selectbox("Category", options=sorted(inventory_df['category'].unique()))
        calc_price = st.number_input("Unit Price (₹)", min_value=1.0, max_value=200.0, value=15.0)
    with rc2:
        calc_qty = st.number_input("Batch Quantity Remaining", min_value=1, max_value=100, value=20)
        calc_sales_30d = st.number_input("Avg Daily Dispense (30d)", min_value=0.0, max_value=5.0, value=0.02, step=0.01)
    with rc3:
        calc_days = st.number_input("Days to Expiry", min_value=1, max_value=1000, value=60)
        calc_demand = calc_sales_30d * calc_days
        st.metric("Projected Demand", f"{calc_demand:.2f} units")
    with rc4:
        calc_unsold = max(0.0, calc_qty - calc_demand)
        calc_ratio = min(1.0, calc_unsold / calc_qty)
        st.metric("Projected Unsold Ratio", f"{calc_ratio * 100:.1f} %")
        
    if st.button("🚀 Run Live AI Risk Inference"):
        batch_input = {
            'batch_quantity_remaining': calc_qty,
            'average_daily_sales_30d': calc_sales_30d,
            'days_to_expiry': calc_days,
            'projected_demand_before_expiry': calc_demand,
            'projected_unsold_quantity': calc_unsold,
            'projected_unsold_ratio': calc_ratio,
            'unit_price_inr': calc_price,
            'category': calc_cat
        }
        pred_res = predict_batch_risk(batch_input, expiry_model, expiry_prep)
        
        st.success(f"**Predicted Risk Tier:** {pred_res['predicted_risk']}")
        p1, p2, p3 = st.columns(3)
        p1.metric("High Risk Probability", f"{pred_res['probabilities'].get('High', 0.0) * 100:.1f} %")
        p2.metric("Medium Risk Probability", f"{pred_res['probabilities'].get('Medium', 0.0) * 100:.1f} %")
        p3.metric("Low Risk Probability", f"{pred_res['probabilities'].get('Low', 0.0) * 100:.1f} %")


# =============================================================
# PAGE 4: INVENTORY RECOMMENDATIONS
# =============================================================
elif page == "4. Inventory Recommendations":
    st.subheader("Operational Inventory Recommendations")
    st.markdown("Automated logistical guidance generated by cross-referencing model demand forecasts, batch expiry risk tiers, and warehouse stock.")
    
    st.markdown(f"""
    <div class="advisory-box">
        <strong>⚠️ Clinical Advisory Notice:</strong> {ADVISORY_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)
    
    if recs_df is not None and not recs_df.empty:
        p_crit = int((recs_df['priority'] == 'CRITICAL').sum())
        p_high = int((recs_df['priority'] == 'HIGH').sum())
        p_med = int((recs_df['priority'] == 'MEDIUM').sum())
        p_low = int((recs_df['priority'] == 'LOW').sum())
        total_var = float(recs_df['value_at_risk_inr'].sum())
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("CRITICAL Alerts", p_crit, "≤ 30 Days Left")
        k2.metric("HIGH Priority", p_high, "Stock Rotation")
        k3.metric("MEDIUM Priority", p_med, "Bi-Weekly Review")
        k4.metric("LOW / Overstock", p_low, "Procurement Hold")
        k5.metric("Total Value at Risk", f"₹ {total_var:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        rf1, rf2 = st.columns(2)
        with rf1:
            pri_filter = st.multiselect("Filter Priority", options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], default=['CRITICAL', 'HIGH', 'MEDIUM'])
        with rf2:
            alert_filter = st.multiselect("Filter Alert Type", options=sorted(recs_df['alert_type'].unique()), default=[])
            
        view_recs = recs_df.copy()
        if pri_filter:
            view_recs = view_recs[view_recs['priority'].isin(pri_filter)]
        if alert_filter:
            view_recs = view_recs[view_recs['alert_type'].isin(alert_filter)]
            
        st.markdown(f'<div class="section-heading">📋 Actionable Operational Alerts ({len(view_recs)} Records)</div>', unsafe_allow_html=True)
        
        for _, rec in view_recs.iterrows():
            st.markdown(f"""
            <div class="rec-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span class="badge-{rec['priority'].lower()}">{rec['priority']}</span>
                        <strong style="margin-left: 8px; font-size: 14px; color: #0f172a;">{rec['medicine_name']}</strong>
                        <span style="color: #64748b; font-size: 12px;">({rec['product_id']} | {rec['batch_id']})</span>
                    </div>
                    <div style="font-weight: 700; color: #dc2626; font-size: 13px;">
                        Value at Risk: ₹ {rec['value_at_risk_inr']:,.2f}
                    </div>
                </div>
                <div style="font-size: 13px; color: #334155;">
                    <strong>Action:</strong> {rec['action_suggestion']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        csv_recs = view_recs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Recommendations CSV",
            data=csv_recs,
            file_name="inventory_recommendations.csv",
            mime="text/csv"
        )
    else:
        st.info("No recommendations found. Please run the training pipeline.")


# =============================================================
# PAGE 5: MODEL PERFORMANCE
# =============================================================
elif page == "5. Model Performance":
    st.subheader("Machine Learning Model Performance")
    st.markdown("Rigorous benchmark comparisons on held-out test datasets never seen during model training.")
    
    tab1, tab2, tab3 = st.tabs(["Demand Regressors", "Expiry Classifiers", "Diagnostics & Limitations"])
    
    with tab1:
        st.markdown("#### Demand Forecasting Model Comparison")
        st.caption("Target: units_sold | Chronological 70% Train, 15% Validation, 15% Test split.")
        
        if os.path.exists("outputs/demand_evaluation_metrics.csv"):
            demand_metrics_df = pd.read_csv("outputs/demand_evaluation_metrics.csv")
            render_df(demand_metrics_df, hide_index=True)
            
        if os.path.exists("outputs/model_visualizations/demand_model_comparison.png"):
            st.image("outputs/model_visualizations/demand_model_comparison.png", caption="Demand Regression Metrics: Validation vs Held-Out Test Set")

    with tab2:
        st.markdown("#### Expiry-Risk Classification Evaluation")
        st.caption("Target: expiry_risk (High, Medium, Low) | Chronological Snapshot split.")
        
        if os.path.exists("outputs/expiry_evaluation_metrics.csv"):
            expiry_metrics_df = pd.read_csv("outputs/expiry_evaluation_metrics.csv")
            render_df(expiry_metrics_df, hide_index=True)
            
        if os.path.exists("outputs/model_visualizations/expiry_confusion_matrices.png"):
            st.image("outputs/model_visualizations/expiry_confusion_matrices.png", caption="Test Set Confusion Matrices: Rule Baseline vs Logistic Regression vs Random Forest")

    with tab3:
        st.markdown("#### In-Depth Diagnostics, Error Analysis & Limitations")
        st.markdown("""
        - **1. Data Sparsity & Zero-Inflation (99.98% Zeros):** Only 2 units sold across 10,000 sales records. WAPE (Weighted Absolute Percentage Error) was used instead of MAPE to protect against division-by-zero errors.
        - **2. Heuristic Ground Truth in Expiry Risk:** The target was created by a transparent operational rule (`days_to_expiry <= 90` & `unsold_ratio >= 0.30`). The Random Forest classifier maps this decision surface with 100% test accuracy, while Logistic Regression achieves 92% due to linear decision boundaries.
        - **3. Autoregressive Horizon Compounding:** Multi-step forecasting uses previous predictions as lag inputs; for horizons beyond 14 days, periodic weekly re-training is recommended.
        """)


# =============================================================
# PAGE 6: ABOUT PROJECT
# =============================================================
elif page == "6. About Project":
    st.subheader("About the Project")
    st.markdown("End-to-End Decision-Support Solution for Pharmacy Inventory & Expiry-Risk Management.")
    
    st.markdown("""
    #### 🎓 Project Background & Assessment Context
    This project was designed, developed, and evaluated as part of the **AI & Machine Learning Internship Assessment** at **Algions AI Labs Private Limited**.
    
    #### 🎯 Business Objectives
    1. **Demand Forecasting:** Forecast product-level daily medicine demand across a 7-day forward horizon to optimize replenishment purchasing.
    2. **Expiry-Risk Mitigation:** Accurately classify warehouse batches at risk of expiring before dispensing, minimizing clinical wastage and financial loss.
    3. **Operational Decision Support:** Provide supply chain managers with actionable, prioritized replenishment, rotation, and inspection alerts.
    
    #### 🛠️ Technology Stack
    - **Language & Core:** Python 3.11, NumPy, Pandas
    - **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, Logistic Regression, ColumnTransformer)
    - **Model Persistence:** Joblib
    - **Visualization:** Plotly Interactive Charts, Matplotlib, Seaborn
    - **User Interface:** Streamlit Production Dashboard
    - **Quality & Testing:** Pytest Suite (15 passing tests across preprocessing, leakage, model inference, and alerts)
    
    #### 🛡️ Responsible AI & Ethical Considerations
    - **Non-Clinical Guardrail:** Model outputs provide inventory and logistical decision support only. They do NOT provide diagnostic, therapeutic, or prescribing advice.
    - **Human-in-the-Loop Oversight:** All automated alerts must be audited and approved by certified pharmacy staff or supply-chain directors prior to physical action.
    - **Privacy Compliance:** Built on synthetic, de-identified datasets containing no patient, clinician, or protected health information (PHI/PII).
    """)
    
    st.info("Algions AI Labs Private Limited | AI/ML Assessment | Candidate: AI & Machine Learning Intern")
