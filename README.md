# Pharmacy Medicine Demand & Expiry-Risk Forecasting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Pytest Passing](https://img.shields.io/badge/Pytest-15%2F15%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![Assessment](https://img.shields.io/badge/Assessment-Algions%20AI%20Labs-teal.svg)](https://algions.com)

An end-to-end, production-grade Artificial Intelligence and Machine Learning decision-support platform designed to forecast retail medicine demand and identify inventory batches susceptible to expiring before dispensing.

Developed as an individual project assessment for the **AI & Machine Learning Internship** at **Algions AI Labs Private Limited**.

---

## 📌 1. Project Overview & Business Problem

Retail and hospital pharmacies face critical inventory inefficiencies:
1. **Stockouts & Shortages:** Under-procurement of essential medicines compromises healthcare delivery and leads to patient dissatisfaction.
2. **Capital Lockup & Expiry Wastage:** Over-procurement leads to warehouse congestion and significant financial losses when medications surpass their manufacturer expiration dates.

**PharmaPredict AI** addresses these challenges through a dual-model machine learning architecture:
- **Demand Forecasting Model:** Multi-step forward-looking demand prediction for 50 medicines across a 7-day horizon using lag features, rolling statistics, and calendar attributes.
- **Expiry-Risk Classification Model:** Multiclass classification of warehouse batches into *High*, *Medium*, and *Low* risk tiers based on remaining shelf life, batch quantity, and historical sales velocity.
- **Inventory Recommendation Engine:** Operational guidance module producing prioritized replenishment, stock rotation, overstock pause, and near-expiry inspection alerts.

---

## 📁 2. Repository Structure

```
Pharmacy-Forecasting/
├── dataset/
│   ├── daily_medicine_sales.csv       # 10,000 daily sales records (50 medicines x 200 days)
│   ├── batch_expiry_inventory.csv     # 1,000 synthetic batch inventory records
│   ├── data_generator.py              # Reproducible dataset generator script
│   └── README.md                      # Original dataset specification & heuristic rules
├── notebooks/
│   └── exploratory_data_analysis.ipynb# Interactive Jupyter Notebook for EDA & profiling
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data ingestion, cleaning, feature engineering, chronological splits
│   ├── eda.py                         # Exploratory data analysis & figure generation
│   ├── demand_forecasting.py          # Baseline, Linear Regression, Random Forest Regressor & 7-day forecasting
│   ├── expiry_risk.py                 # Rule Baseline, Logistic Regression, Random Forest Classifier & inference
│   ├── recommendation_engine.py       # Operational recommendations with priority rankings & disclaimers
│   ├── model_evaluation.py            # Consolidated evaluation reports & diagnostic metrics
│   └── train.py                       # Master automated end-to-end training pipeline
├── models/
│   ├── demand_preprocessor.joblib      # Fitted StandardScaler & OneHotEncoder for demand features
│   ├── demand_baseline.joblib          # Moving Average Baseline estimator
│   ├── demand_forecaster_lr.joblib     # Trained Linear Regression model
│   ├── demand_forecaster_rf.joblib     # Trained Random Forest Regressor
│   ├── demand_best_model.joblib        # Persisted best model for production inference
│   ├── expiry_preprocessor.joblib      # Fitted preprocessor for batch expiry features
│   ├── expiry_baseline.joblib          # Transparent Rule-Based Heuristic classifier
│   ├── expiry_classifier_lr.joblib     # Trained Logistic Regression classifier
│   ├── expiry_classifier_rf.joblib     # Trained Random Forest Classifier
│   ├── expiry_best_model.joblib        # Persisted best model for production inference
│   └── model_metadata.json             # Model training metadata, features, and hyperparameters
├── outputs/
│   ├── eda_visualizations/             # 7 high-resolution EDA figures (PNG)
│   ├── model_visualizations/           # Confusion matrices, actual vs predicted, feature importances
│   ├── demand_forecast_results.csv     # Generated next 7-day demand forecasts for 50 medicines
│   ├── expiry_risk_predictions.csv     # Full inventory dataset annotated with model predictions
│   ├── inventory_recommendations.csv   # Prioritized operational inventory alerts
│   ├── demand_evaluation_metrics.csv   # Regression benchmark metrics table
│   ├── expiry_evaluation_metrics.csv   # Classification benchmark metrics table
│   └── evaluation_summary.json         # Consolidated metrics dictionary
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py      # Tests for data loading, cleaning, feature engineering, and leakage
│   ├── test_models.py                  # Tests for Joblib model loading, inference, edge cases, and WAPE
│   └── test_recommendations.py         # Tests for recommendation generation, triggers, and disclaimers
├── report/
│   └── TECHNICAL_REPORT.md             # Comprehensive 16-section technical engineering report
├── app.py                              # Modern interactive Streamlit web dashboard
├── requirements.txt                    # Production environment dependencies
├── README.md                           # Master project documentation
└── .gitignore                          # Standard git ignore definitions
```

---

## 📊 3. Dataset Description

The system utilizes synthetic pharmaceutical datasets generated with seed 42:
- **`daily_medicine_sales.csv`:** 10,000 product-sales records for 50 medicines across 200 consecutive days (2026-01-01 to 2026-07-19).
- **`batch_expiry_inventory.csv`:** 1,000 synthetic batch-inventory snapshots across 20 snapshot dates (2026-02-10 to 2026-07-10).
- **Therapeutic Categories (15):** Vitamin, Antibiotic, Gastrointestinal, Cardiovascular, Analgesic, Respiratory, Dermatology, Antihistamine, Electrolyte, Diabetes, First Aid, Hygiene, Ophthalmic, Diagnostic, and Device.

### Data Sparsity & Intermittent Demand Note
In the sales history, 9,998 records recorded 0 sales, with 2 records recording 1 unit sold (MED001). This reflects an ultra-slow moving, intermittent demand profile. To prevent division-by-zero errors in standard MAPE, **Weighted Absolute Percentage Error (WAPE)** was implemented with zero-division safety guards.

### Transparent Heuristic Rule for Expiry Risk
The target `expiry_risk` is generated via a transparent operational heuristic:
- **High Risk:** `days_to_expiry <= 90` AND `projected_unsold_ratio >= 0.30`
- **Medium Risk:** `days_to_expiry <= 180` AND `projected_unsold_ratio >= 0.15` (and not High)
- **Low Risk:** Otherwise

---

## 🚀 4. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone or Open the Workspace
```bash
git clone <repository-url>
cd Pharmacy-Forecasting
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ 5. Execution Instructions

### A. Run the Master Training Pipeline
Executes data preprocessing, feature engineering, model training, evaluation, and recommendation generation:
```bash
python -m src.train
```

### B. Run the Automated Test Suite
Executes all 15 Pytest unit tests verifying preprocessing, leakage prevention, model inference, and recommendation logic:
```bash
pytest -v
```

### C. Launch the Streamlit Web Application
Starts the production decision-support dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 6. Machine Learning Methodology & Results

### Strict Leakage-Free Validation Strategy
- **Demand Forecasting:** Chronological partition on dates:
  - **Train Set:** First 70% of dates (125 dates, 6,250 records).
  - **Validation Set:** Next 15% of dates (27 dates, 1,350 records).
  - **Held-Out Test Set:** Final 15% of dates (27 dates, 1,350 records).
  - Rolling features strictly utilize `.shift(1)` so only past observations are ever visible.
- **Expiry Risk:** Chronological partition on `snapshot_date` (14 snapshots Train, 3 snapshots Val, 3 snapshots Test).

### 6.1 Demand Forecasting Performance (Held-Out Test Set)

| Model | Test MAE | Test RMSE | Test WAPE | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Moving Average Baseline (7d)** | 0.000000 | 0.000000 | 0.000000 | Baseline |
| **Linear Regression** | 0.000136 | 0.000456 | 0.184248 | Evaluated |
| **Random Forest Regressor** | **0.000000** | **0.000000** | **0.000000** | **Selected Best** |

*Note on Validation Set:* On the validation set containing intermittent demand, Random Forest achieved an MAE of **0.000006** and WAPE of **0.0083**, outperforming Linear Regression (MAE 0.000163, WAPE 0.2202).

### 6.2 Expiry-Risk Classification Performance (Held-Out Test Set)

| Model | Test Accuracy | Test Macro F1 | Test Macro Precision | Test Macro Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Baseline** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Heuristic Benchmark |
| **Logistic Regression** | 0.9200 | 0.8649 | 0.8407 | 0.9108 | Evaluated |
| **Random Forest Classifier** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **Selected Best** |

*Insights:*
- Random Forest captures orthogonal step-function boundaries perfectly, achieving 1.0000 across all metrics.
- Logistic Regression incurs minor boundary misclassifications on intermediate Medium-risk batches due to its linear decision hyperplane.

---

## 💡 7. Inventory Recommendation Engine

Operational recommendations are automatically synthesized from model outputs:
- **CRITICAL Alerts (8 items):** Batches expiring within 30 days. Action: Immediate physical QA inspection, quarantine, or return.
- **HIGH Priority (6 items):** High-risk batches expiring within 31–90 days. Action: FEFO stock rotation, promotional discounting, or inter-branch transfer.
- **MEDIUM Priority (6 items):** Medium-risk batches expiring within 91–180 days. Action: Bi-weekly inventory audits.
- **LOW / OVERSTOCK (22 items):** Warehouse stock $> 150$ units with low demand velocity. Action: Pause purchase orders.

**Total Financial Value at Risk:** ₹ 11,280 across active inventory batches.

---

## 🖥️ 8. Streamlit Dashboard Features

The web interface (`app.py`) provides 6 interactive pages:
1. **Executive Overview:** High-level KPIs, daily demand timeline, risk donut chart, top-selling medicines, near-expiry table.
2. **Demand Forecasting:** Medicine dropdown selector, dynamic horizon slider (1–14 days), historical vs ML forecast Plotly chart, tabular forecast breakdown, CSV download, and a real-time What-If simulator.
3. **Expiry-Risk Analysis:** Batch table with multi-tier filters and search, plus an interactive Batch Risk Inference Calculator powered by the saved Joblib model.
4. **Inventory Recommendations:** Actionable alerts categorized by priority (Critical, High, Medium, Low), value at risk summary, and CSV export.
5. **Model Performance:** Side-by-side benchmark comparison tables, test set confusion matrices, and error diagnostic documentation.
6. **About Project:** Project background, technology stack, Algions AI Labs assessment details, and responsible AI disclosures.

---

## 🛡️ 9. Responsible AI, Ethics & Limitations

1. **Non-Clinical Advisory Guardrail:**
   > *All outputs represent operational inventory guidance only and do NOT provide medical, clinical, prescribing, or treatment advice. All pharmaceutical handling must be reviewed by a certified pharmacist.*
2. **Synthetic Data Disclosure:**
   The `expiry_risk` ground-truth label is a rule-generated proxy. The models demonstrate high fidelity in learning this operational risk surface, rather than predicting unobserved real-world clinical wastage.
3. **Human-in-the-Loop:**
   The system is an advisory decision-support platform (DSS); automated procurement or physical disposal without human sign-off is intentionally restricted.

---

## 📜 10. License & Assessment Attribution

Developed for the **AI & Machine Learning Internship Assessment** at **Algions AI Labs Private Limited**.
Dataset and source code are provided for academic evaluation and demonstration purposes.
