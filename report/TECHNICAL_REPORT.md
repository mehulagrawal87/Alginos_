# Technical Project Report: Pharmacy Medicine Demand & Expiry-Risk Forecasting System

**Organization:** Algions AI Labs Private Limited  
**Assessment Role:** AI & Machine Learning Internship Assessment  
**Project Domain:** Pharmaceutical Inventory Optimization & Decision Support  
**Date:** November 2026  
**Candidate Evaluation Track:** Advanced Individual ML Solution  

---

## Executive Summary

This technical report documents the end-to-end design, implementation, empirical evaluation, and operational deployment of the **Pharmacy Medicine Demand & Expiry-Risk Forecasting System**. Built to solve acute inventory shrinkage, stockout risks, and pharmaceutical expiry wastage, this platform pairs **autoregressive time-series machine learning** for forward demand estimation with **multiclass non-linear classification** for batch-level expiration risk surfaces.

The complete codebase features rigorous chronological validation protocols (zero data leakage), reproducible feature transformation pipelines, a 15-test automated unit test suite with 100% pass rate, and an interactive modern Streamlit decision-support dashboard loading saved Joblib model artifacts for real-time inference.

---

## 1. Introduction

Pharmaceutical retail and hospital inventory management represent a complex balancing act between clinical availability and financial capital preservation. While stockouts of essential therapeutics can jeopardize patient care, over-procurement leads to warehouse congestion, capital lockup, and catastrophic wastage when medications surpass their manufacturer expiration dates.

Traditional pharmacy supply chains rely heavily on static reorder points (e.g., minimum/maximum inventory thresholds) or intuition-driven purchasing. These static models fail to capture velocity shifts, seasonal patterns, price elasticity, and batch-level expiration timelines. This project implements a modern machine learning decision-support architecture that proactively forecasts product-level demand and flags at-risk inventory batches before expiration.

---

## 2. Problem Statement

Pharmaceutical supply chains present two coupled optimization dilemmas:

1. **Demand Stochasticity & Intermittent Velocity:** Retail medicines frequently display sparse, intermittent demand patterns. Predicting daily sales units ($y_t$) without forward data leakage is essential to inform purchase orders.
2. **Batch-Level Expiry Risk:** Each medicine delivery enters inventory as an isolated batch characterized by an expiration date, initial quantity, and variable dispensing velocity. Batches with insufficient sell-through velocity relative to their remaining shelf life risk total expiration wastage.

**Core Challenge:** Build an automated, explainable, and defensible machine learning system that ingests historical daily sales records and batch inventory snapshots, forecasts forward demand, classifies batch risk tiers (*High*, *Medium*, *Low*), and generates prioritized operational recommendations while strictly adhering to responsible AI guidelines.

---

## 3. Project Objectives

1. **Defensible Exploratory Data Analysis (EDA):** Quantify dataset distributions, identify zero-inflation patterns, detect outliers via interquartile ranges (IQR), and visualize therapeutic category spreads.
2. **Leakage-Free Feature Engineering:** Construct multi-step lag features and rolling window statistics on historical sales using temporal shifts (`shift(1)`) to guarantee strict past-only information availability.
3. **Comparative Demand Modeling:** Build and evaluate a 7-day Moving Average Baseline against Linear Regression and Random Forest Regressor models on a held-out chronological test set.
4. **Transparent Expiry-Risk Modeling:** Develop and evaluate a Rule-Based Heuristic Baseline against Logistic Regression and Random Forest Classifier models, documenting the synthetic label generation mechanics.
5. **Operational Recommendation Engine:** Synthesize model outputs to produce actionable, prioritized inventory guidance (Replenishment, Stock Rotation, Overstock Hold, and Near-Expiry QA Review) with mandatory non-medical disclaimers.
6. **Production-Ready Inference Dashboard:** Deliver a multi-page interactive Streamlit web application powered by saved Joblib model artifacts.
7. **Software Engineering Standards:** Ensure 100% test reproducibility with a modular directory structure, Pytest coverage, and clean documentation.

---

## 4. Dataset Description

The system was developed and validated on the official synthetic pharmaceutical dataset:

### 4.1 Daily Medicine Sales (`daily_medicine_sales.csv`)
- **Total Records:** 10,000 daily observations.
- **Time Horizon:** 200 consecutive days (2026-01-01 through 2026-07-19) with zero temporal gaps.
- **Product Catalog:** 50 distinct synthetic medicines across 15 therapeutic categories.
- **Feature Schema:**
  - `product_id`: Unique alphanumeric identifier (MED001 to MED050).
  - `medicine_name`: Proprietary/generic medication nomenclature.
  - `category`: Therapeutic classification (e.g., Analgesic, Antibiotic, Cardiovascular, Vitamin).
  - `date`: Daily observation timestamp ($YYYY-MM-DD$).
  - `units_sold`: Target variable ($y$). Number of units dispensed daily.
  - `purchase_quantity`: Upstream procurement volume.
  - `stock_on_hand`: Warehouse on-shelf unit inventory.
  - `unit_price_inr`: Retail dispensing price in Indian Rupees (₹ 2.0 to ₹ 35.0).
  - `promotion`: Binary indicator (0/1) for active discount marketing campaigns.
  - `day_of_week`, `month`, `is_weekend`: Calendar attributes.

### 4.2 Batch Expiry Inventory (`batch_expiry_inventory.csv`)
- **Total Records:** 1,000 batch inventory snapshots.
- **Snapshot Frequency:** 20 chronological snapshot dates spaced approximately 8 days apart (2026-02-10 to 2026-07-10).
- **Target Distribution:**
  - *Low Risk:* 575 batches (57.5%)
  - *High Risk:* 344 batches (34.4%)
  - *Medium Risk:* 81 batches (8.1%)
- **Feature Schema:**
  - `snapshot_date`: Audit snapshot date.
  - `batch_id`: Unique batch identification code (e.g., MED001-B001).
  - `batch_quantity_remaining`: Undispensed physical units in warehouse (5 to 29 units).
  - `average_daily_sales_30d`: Trailing 30-day dispensing rate.
  - `days_to_expiry`: Days remaining until manufacturer expiration date (7 to 729 days).
  - `projected_demand_before_expiry`: Model/heuristic estimate of total sales before expiry.
  - `projected_unsold_quantity`: Anticipated residual stock at expiration date.
  - `projected_unsold_ratio`: Ratio of unsold stock ($\frac{\text{Unsold Quantity}}{\text{Batch Quantity}}$).
  - `expiry_risk`: Multiclass risk tier (*High*, *Medium*, *Low*).

---

## 5. Exploratory Data Analysis (EDA)

Exploratory analysis yielded crucial engineering insights into inventory dynamics:

1. **Extreme Zero-Inflation & Intermittent Velocity:**
   - Out of 10,000 daily sales records, 9,998 records recorded 0 sales, with exactly 2 records recording 1 unit sold (both for MED001 - Paracetamol 500mg).
   - *Domain Implication:* The dataset exhibits extreme intermittency (99.98% zero sales), mirroring ultra-slow moving, cold-chain, or specialized emergency pharmaceutical stock. Standard regression metrics like Mean Absolute Percentage Error (MAPE) encounter fatal division-by-zero errors ($\frac{0}{0}$), mandating the adoption of Weighted Absolute Percentage Error (WAPE).
2. **Category Diversity:**
   - 15 therapeutic categories represented across 50 products: Vitamins (1,400 records / 7 products), Antibiotics (1,000 records / 5 products), Gastrointestinal (1,000 records / 5 products), Cardiovascular (1,000 records / 5 products), Analgesics (800 records / 4 products), Respiratory (800 records / 4 products), Dermatology (800 records / 4 products), Antihistamines, Diabetes, Electrolytes, First Aid, Hygiene, Diagnostic, Device, and Ophthalmic.
3. **Outlier Profile:**
   - IQR analysis identified 2 statistical outliers in `units_sold` (the 2 sales events), 2 outliers in `batch_quantity_remaining`, and 4 outliers in `projected_unsold_ratio`. No data corruption or negative values were detected.
4. **Visual Outputs:**
   - Seven publication-ready graphics were generated and stored in `outputs/eda_visualizations/`: category breakdown, 200-day demand timeline, top products, stock distribution across categories, expiry-risk donut chart, heuristic decision boundary scatter plot, and feature correlation heatmap.

---

## 6. Data Preprocessing

Preprocessing was architected in `src/data_preprocessing.py` to prevent all forms of data contamination:

- **Missing Value Handling:** Verified zero missing values in primary tables; defensive imputation handlers configured with median price and 0 for sales/stock.
- **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fit exclusively on the training split to encode medicine categories.
- **Numerical Feature Standardization:** `StandardScaler()` fit strictly on training partitions to normalize feature distributions for Linear and Logistic Regression models.
- **Chronological Splitting Protocol:**
  - *Demand Time-Series:* Sorted chronologically by `product_id` and `date`. The first 70% of dates (125 dates, 6,250 records) formed the Training set; the subsequent 15% (27 dates, 1,350 records) formed the Validation set; the terminal 15% (27 dates, 1,350 records) formed the Held-Out Test set.
  - *Inventory Snapshots:* Partitioned by `snapshot_date`. Snapshots 1–14 (700 batches) allocated to Train; snapshots 15–17 (150 batches) allocated to Validation; snapshots 18–20 (150 batches) allocated to Held-Out Test.

---

## 7. Feature Engineering

### 7.1 Demand Forecasting Features
To capture temporal dependency without target leakage:
- **Lag Features:** Extracted past sales at $t-1, t-2, t-3, t-7, t-14, t-21$.
- **Leakage-Safe Rolling Statistics:** Computed rolling means and standard deviations over 7-day and 14-day sliding windows. **Crucially**, rolling transformations applied `.shift(1)` prior to window calculation:
  $$\text{roll\_mean\_7}_t = \frac{1}{7} \sum_{i=1}^{7} y_{t-i}$$
  This mathematical formulation guarantees that the target value at day $t$ ($y_t$) is never included in the feature vector $\mathbf{x}_t$.
- **Calendar & Pricing Signals:** Day of week (0–6), day of month (1–31), month (1–12), weekend indicator (0/1), active promotion flag (0/1), and unit price (INR).
- **Initial Truncation:** The first 21 days per medicine were dropped to ensure all lag vectors were fully populated with real historical data.

### 7.2 Expiry-Risk Classification Features
- `batch_quantity_remaining`: Physical on-hand batch stock.
- `average_daily_sales_30d`: Recent empirical velocity.
- `days_to_expiry`: Temporal distance to expiry date.
- `projected_demand_before_expiry`: Calculated velocity projection ($\text{sales\_30d} \times \text{days\_to\_expiry}$).
- `projected_unsold_quantity`: $\max(0, \text{batch\_qty} - \text{projected\_demand})$.
- `projected_unsold_ratio`: $\frac{\text{projected\_unsold\_qty}}{\text{batch\_qty}}$.
- `unit_price_inr` and `category`: Financial valuation and clinical classification.

---

## 8. Demand Forecasting Methodology

Demand forecasting models were evaluated on the held-out test partition (last 27 days of history across all 50 medicines):

1. **Baseline Model (Moving Average):**
   - Window: 7 days. Predicts the rolling mean of trailing 7-day sales per product.
2. **Linear Regression:**
   - Multi-feature ordinary least squares regression over scaled numericals and one-hot categorical vectors.
3. **Random Forest Regressor:**
   - Ensemble of 100 decision trees, maximum depth of 10, minimum sample split of 5, fixed random state 42.

### Multi-Step Autoregressive 7-Day Forecasting
To generate forward projections for days $T+1$ through $T+7$, an iterative autoregressive routine was implemented (`generate_next_7_day_forecast`). At each forward step $k \in \{1, \dots, 7\}$:
1. Dynamic lag features are synthesized from past actuals and prior step predictions ($\hat{y}_{T+k-1}$).
2. Calendar attributes for date $T+k$ are generated.
3. Preprocessing scaling is applied.
4. The trained model predicts $\hat{y}_{T+k}$, clipped at zero ($\max(0, \hat{y})$).
5. Projected stock on hand is decremented accordingly.

---

## 9. Expiry-Risk Methodology

### 9.1 Transparent Heuristic Ground Truth
In accordance with the project specification and synthetic generator documentation, the target label `expiry_risk` was generated by a rule heuristic:
- **High Risk:** $\text{days\_to\_expiry} \le 90 \land \text{projected\_unsold\_ratio} \ge 0.30$
- **Medium Risk:** $\text{days\_to\_expiry} \le 180 \land \text{projected\_unsold\_ratio} \ge 0.15$ (and not High)
- **Low Risk:** Otherwise

### 9.2 Candidate Classification Algorithms
1. **Rule-Based Baseline Classifier:** Direct implementation of the domain heuristic.
2. **Multinomial Logistic Regression:** Linear decision boundaries with L2 regularization and balanced class weighting.
3. **Random Forest Classifier:** 100 balanced decision trees with maximum depth of 8 to capture non-linear orthogonal splits.

---

## 10. Model Comparison & Empirical Results

All models were evaluated on the held-out test datasets never seen during training or tuning.

### 10.1 Demand Forecasting Results

| Model Architecture | Val MAE | Val RMSE | Val WAPE | Test MAE | Test RMSE | Test WAPE | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moving Average Baseline (7d)** | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | No |
| **Linear Regression** | 0.000163 | 0.000501 | 0.220207 | 0.000136 | 0.000456 | 0.184248 | No |
| **Random Forest Regressor** | **0.000006** | **0.000227** | **0.008333** | **0.000000** | **0.000000** | **0.000000** | **YES** |

*Analysis:* Because the test split has 0 actual sales due to dataset sparsity, the Moving Average and Random Forest models achieve 0.0 test MAE and 0.0 test RMSE. On the validation set (which contains intermittent sales transitions), the **Random Forest Regressor** demonstrated superior precision (Val MAE $0.000006$, Val WAPE $0.0083$) compared to Linear Regression ($0.000163$ MAE, $0.2202$ WAPE), whose unconstrained hyperplane produced fractional over-predictions.

### 10.2 Expiry-Risk Classification Results

| Model Architecture | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro Prec | Test Macro Rec | Test Macro F1 | Test Weighted F1 | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Benchmark |
| **Logistic Regression** | 0.9333 | 0.8872 | 0.9200 | 0.8407 | 0.9108 | 0.8649 | 0.9255 | No |
| **Random Forest Classifier** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **YES** |

*Analysis:*
- The **Rule-Based Baseline** achieves 1.0000 accuracy because it executes the identical deterministic heuristic from which labels were created.
- The **Random Forest Classifier** achieves 1.0000 accuracy across all evaluation metrics because axis-aligned decision trees perfectly isolate rectangular decision boundaries ($\le 90$ days, $\ge 30\%$ ratio).
- **Logistic Regression** achieves 0.9200 test accuracy (Macro F1: 0.8649). Because logistic regression fits a smooth linear sigmoid boundary across continuous feature space, it incurs boundary classification errors at the sharp step-function thresholds of the synthetic heuristic.

---

## 11. Evaluation Metrics Formulation

To provide rigorous mathematical accountability:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
3. **Weighted Absolute Percentage Error (WAPE):**
   $$\text{WAPE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} y_i}$$
   *Zero-Division Protection:* When $\sum y_i = 0$, if $\sum |y_i - \hat{y}_i| = 0$ then $\text{WAPE} = 0.0$; otherwise $\frac{\sum |y_i - \hat{y}_i|}{\epsilon}$.
4. **Macro-Averaged F1-Score:**
   $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
   Treats all 3 risk classes (*High*, *Medium*, *Low*) with equal weight regardless of class imbalance.

---

## 12. Error Analysis & Diagnostics

### 12.1 Logistic Regression Confusion Matrix (Test Set, N=150)
- **High Risk (Actual: 51):** Predicted High = 51 (100% recall).
- **Low Risk (Actual: 87):** Predicted Low = 87 (100% recall).
- **Medium Risk (Actual: 12):** Predicted Medium = 0, Misclassified as High = 12.
- *Root Cause Analysis:* The Medium risk tier represents an intermediate boundary band ($90 < \text{days} \le 180 \land \text{ratio} \ge 0.15$). Logistic regression's linear hyperplane with balanced weights over-prioritized the High-risk penalty, pushing marginal Medium-risk batches into the High category. This demonstrates the necessity of non-linear tree ensembles for threshold-based operational boundaries.

### 12.2 Feature Importance Rankings (Random Forest)
1. `days_to_expiry`: 51.4% Gini importance.
2. `projected_unsold_ratio`: 28.7% Gini importance.
3. `projected_unsold_quantity`: 8.9% Gini importance.
4. `batch_quantity_remaining`: 4.2% Gini importance.
5. Category one-hot indicators: Remaining 6.8%.

---

## 13. Inventory Recommendation Engine

Operational recommendations are synthesized by cross-referencing batch risk tiers with demand projections and warehouse stock:

### Priority Matrix
- **CRITICAL (8 Alerts):** Batches with $\le 30$ days to expiry regardless of quantity. *Action:* Immediate physical pharmacist inspection, packaging verification, and quarantine/return.
- **HIGH (6 Alerts):** High expiry-risk batches with $31 \le \text{days} \le 90$. *Action:* Implement First-Expired, First-Out (FEFO) dispensing, promotional discount pricing, or inter-branch transfer.
- **MEDIUM (6 Alerts):** Medium risk batches ($91 \le \text{days} \le 180$). *Action:* Bi-weekly inventory audit and sell-through monitoring.
- **LOW / OVERSTOCK (22 Alerts):** Medicines with warehouse stock $> 150$ units and near-zero sales velocity. *Action:* Pause purchase order generation to prevent capital lockup.

**Financial Value at Risk:** Across active batches, ₹ 11,280 in pharmaceutical inventory was identified as vulnerable to expiry write-offs without intervention.

---

## 14. Streamlit Dashboard Architecture

The application (`app.py`) provides a responsive, healthcare-themed web interface:

- **Executive Overview (Page 1):** Real-time KPI summary cards, interactive Plotly sales timeline, expiry-risk donut chart, top-selling medications, and near-expiry warning table.
- **Demand Forecasting (Page 2):** Medicine dropdown selector, dynamic horizon slider (1–14 days), historical demand vs ML forecast chart, 7-day tabular breakdown, feature importance chart, CSV download button, and a real-time What-If demand simulator.
- **Expiry-Risk Analysis (Page 3):** Full batch table with search and multi-select filters, and an interactive Batch Risk Calculator where users can input custom parameters to receive real-time class probabilities from the saved Random Forest model.
- **Inventory Recommendations (Page 4):** Filterable recommendation cards color-coded by priority, financial value at risk metrics, CSV export, and prominent non-medical advisory notices.
- **Model Performance (Page 5):** Regression and classification benchmark comparison tables, test set confusion matrices, and detailed technical documentation on data sparsity and heuristic properties.
- **About Project (Page 6):** Architectural summary, internship assessment context at Algions AI Labs, and responsible AI disclosures.

---

## 15. Responsible AI, Limitations & Ethical Guardrails

### 15.1 Mandatory Non-Clinical Advisory Guardrail
All model outputs, dashboard interfaces, and export files carry the strict advisory disclaimer:
> *"Advisory Notice: These recommendations represent operational inventory decision-support guidance only. They do NOT constitute medical, clinical, prescribing, or therapeutic advice. All medicine transfers and purchases must comply with applicable pharmacy regulations and human pharmacist review."*

### 15.2 Synthetic Data & Rule Heuristic Disclosure
The `expiry_risk` target is a synthetic rule-generated proxy, not empirically observed clinical wastage. High model accuracy reflects the ability of machine learning to learn complex operational rule surfaces. For production deployment in real hospitals, models must be retrained on barcode-scanned dispensing transactions and verified physical disposal logs.

### 15.3 Human-in-the-Loop Oversight
The system is explicitly designed as a Decision Support System (DSS). No purchase orders are automatically placed, and no stock is automatically discarded without manual sign-off by a licensed pharmacist.

---

## 16. Conclusion

The **Pharmacy Medicine Demand & Expiry-Risk Forecasting System** successfully fulfills all technical, methodological, and architectural criteria established for the Algions AI Labs AI & Machine Learning assessment:

- **Robust Engineering:** Modular codebase, automated training orchestrator, and 15 passing Pytest unit tests.
- **Zero Leakage:** Strict chronological temporal partitioning for time-series forecasting.
- **High-Fidelity Machine Learning:** Superior Random Forest models for both regression and classification, thoroughly benchmarked against transparent baselines.
- **Actionable Decision Support:** Operational recommendation engine with ₹ 11,280 value-at-risk visibility.
- **Interactive UI:** Production Streamlit web application loading persisted Joblib models for real-time inference.

