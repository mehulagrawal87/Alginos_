"""
Exploratory Data Analysis (EDA) Module.

Generates comprehensive data profiling, statistical summaries, outlier checks,
and publication-ready visualization artifacts.
"""

import os
import tempfile

# Set non-interactive matplotlib backend and writable config directory
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any


def run_full_eda(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    output_dir: str = "outputs/eda_visualizations"
) -> Dict[str, Any]:
    """
    Execute end-to-end exploratory data analysis on both sales and inventory datasets.
    Saves visual figures to output_dir and returns structured summary metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
    
    summary = {}
    
    # 1. Dataset Overview
    summary['sales_overview'] = {
        'total_records': len(sales_df),
        'total_features': sales_df.shape[1],
        'unique_medicines': int(sales_df['product_id'].nunique()),
        'unique_categories': int(sales_df['category'].nunique()),
        'date_range_start': str(sales_df['date'].min()),
        'date_range_end': str(sales_df['date'].max()),
        'missing_values': int(sales_df.isnull().sum().sum()),
        'duplicates': int(sales_df.duplicated().sum()),
        'total_units_sold': int(sales_df['units_sold'].sum()),
        'zero_sales_records': int((sales_df['units_sold'] == 0).sum()),
        'zero_sales_percentage': float(round((sales_df['units_sold'] == 0).mean() * 100, 2))
    }
    
    summary['inventory_overview'] = {
        'total_batches': len(inventory_df),
        'total_features': inventory_df.shape[1],
        'unique_products': int(inventory_df['product_id'].nunique()),
        'snapshot_count': int(inventory_df['snapshot_date'].nunique()),
        'snapshot_start': str(inventory_df['snapshot_date'].min()),
        'snapshot_end': str(inventory_df['snapshot_date'].max()),
        'missing_values': int(inventory_df.isnull().sum().sum()),
        'duplicates': int(inventory_df.duplicated().sum()),
        'expiry_risk_counts': inventory_df['expiry_risk'].value_counts().to_dict()
    }
    
    # 2. Outlier Detection using IQR method
    def count_iqr_outliers(series: pd.Series) -> int:
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        return int(((series < lower) | (series > upper)).sum())

    summary['outliers'] = {
        'sales_units_sold': count_iqr_outliers(sales_df['units_sold']),
        'sales_stock_on_hand': count_iqr_outliers(sales_df['stock_on_hand']),
        'inventory_days_to_expiry': count_iqr_outliers(inventory_df['days_to_expiry']),
        'inventory_batch_qty': count_iqr_outliers(inventory_df['batch_quantity_remaining']),
        'inventory_unsold_ratio': count_iqr_outliers(inventory_df['projected_unsold_ratio'])
    }
    
    # Visual 1: Category Distribution
    plt.figure(figsize=(10, 5))
    cat_counts = sales_df.groupby('category')['product_id'].nunique().sort_values(ascending=False)
    ax = sns.barplot(x=cat_counts.values, y=cat_counts.index, color="#2b5c8f")
    plt.title("Medicine Count by Therapeutic Category", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Distinct Medicines", fontsize=11)
    plt.ylabel("Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_category_distribution.png"), dpi=200)
    plt.close()
    
    # Visual 2: Sales Timeline & Aggregate Demand
    daily_sales = sales_df.groupby('date')['units_sold'].sum().reset_index()
    daily_stock = sales_df.groupby('date')['stock_on_hand'].sum().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(daily_sales['date'], daily_sales['units_sold'], color="#d9534f", linewidth=1.5, marker='o', markersize=3)
    ax1.set_title("Aggregated Daily Medicine Demand Across All 50 Medicines", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Units Sold", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(daily_stock['date'], daily_stock['stock_on_hand'], color="#0275d8", linewidth=1.5)
    ax2.set_title("Total Stock on Hand Across All Medicines Over Time", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Total Units in Stock", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_sales_timeline.png"), dpi=200)
    plt.close()
    
    # Visual 3: Top Selling Medicines
    plt.figure(figsize=(10, 5))
    top_meds = sales_df.groupby('medicine_name')['units_sold'].sum().sort_values(ascending=False).head(10)
    # If all or mostly zero, display top 10 products
    sns.barplot(x=top_meds.values, y=top_meds.index, color="#5cb85c")
    plt.title("Top Medicines by Total Units Sold (200-Day History)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Total Units Sold", fontsize=11)
    plt.ylabel("Medicine Name", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_top_medicines.png"), dpi=200)
    plt.close()
    
    # Visual 4: Stock Distribution by Category
    plt.figure(figsize=(11, 6))
    stock_by_cat = sales_df.groupby(['category', 'product_id'])['stock_on_hand'].mean().reset_index()
    sns.boxplot(data=stock_by_cat, x='stock_on_hand', y='category', color="#f0ad4e")
    plt.title("Distribution of Average Stock on Hand per Product Across Categories", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Average Stock on Hand", fontsize=11)
    plt.ylabel("Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_stock_distribution.png"), dpi=200)
    plt.close()
    
    # Visual 5: Expiry Risk Distribution (Donut & Bar)
    risk_counts = inventory_df['expiry_risk'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'Low': '#5cb85c', 'Medium': '#f0ad4e', 'High': '#d9534f'}
    palette = [colors.get(c, '#999999') for c in risk_counts.index]
    
    ax1.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct='%1.1f%%',
        colors=palette,
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )
    ax1.set_title("Batch Expiry-Risk Class Distribution", fontsize=12, fontweight='bold')
    
    sns.barplot(x=risk_counts.index, y=risk_counts.values, hue=risk_counts.index, palette=palette, legend=False, ax=ax2)
    ax2.set_title("Batch Count by Expiry-Risk Tier", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Risk Category", fontsize=10)
    ax2.set_ylabel("Total Batches", fontsize=10)
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_expiry_risk_distribution.png"), dpi=200)
    plt.close()
    
    # Visual 6: Days to Expiry vs Projected Unsold Ratio
    plt.figure(figsize=(10, 6))
    for risk_label in ['Low', 'Medium', 'High']:
        sub = inventory_df[inventory_df['expiry_risk'] == risk_label]
        plt.scatter(
            sub['days_to_expiry'],
            sub['projected_unsold_ratio'],
            label=f"{risk_label} ({len(sub)})",
            color=colors.get(risk_label, '#999999'),
            alpha=0.6,
            edgecolors='w',
            s=45
        )
    # Overlay heuristic threshold lines
    plt.axvline(x=90, color='#d9534f', linestyle='--', linewidth=1.2, label='High Risk Threshold (90d)')
    plt.axvline(x=180, color='#f0ad4e', linestyle='--', linewidth=1.2, label='Medium Risk Threshold (180d)')
    plt.axhline(y=0.30, color='#d9534f', linestyle=':', linewidth=1.2, label='High Unsold Ratio (30%)')
    plt.axhline(y=0.15, color='#f0ad4e', linestyle=':', linewidth=1.2, label='Medium Unsold Ratio (15%)')
    
    plt.title("Heuristic Decision Boundary: Days to Expiry vs Projected Unsold Ratio", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Days to Expiry", fontsize=11)
    plt.ylabel("Projected Unsold Ratio", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_days_to_expiry_vs_unsold.png"), dpi=200)
    plt.close()
    
    # Visual 7: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_cols = [
        'batch_quantity_remaining', 'average_daily_sales_30d', 'days_to_expiry',
        'unit_price_inr', 'projected_demand_before_expiry', 'projected_unsold_quantity',
        'projected_unsold_ratio'
    ]
    corr = inventory_df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
    plt.title("Inventory Risk Features Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=200)
    plt.close()
    
    return summary


if __name__ == "__main__":
    from src.data_preprocessing import load_sales_data, load_inventory_data
    sales = load_sales_data("dataset/daily_medicine_sales.csv")
    inv = load_inventory_data("dataset/batch_expiry_inventory.csv")
    res = run_full_eda(sales, inv)
    print("EDA completed successfully. Summary generated:")
    import pprint
    pprint.pprint(res)
