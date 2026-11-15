"""
Expiry-Risk Prediction Module for Pharmacy Inventory Batches.

Implements:
1. Rule-Based Heuristic Baseline
2. Multinomial Logistic Regression
3. Random Forest Classifier
With chronological validation, classification metrics (Accuracy, Precision, Recall, F1),
confusion matrix visualization, feature importance, and batch risk inference.

CRITICAL METHODOLOGY NOTE:
The expiry_risk label is generated via a transparent heuristic rule in the synthetic data.
These models demonstrate how ML can learn risk surfaces, but MUST be treated as an advisory
decision-support tool rather than an independently observed real-world ground truth.
"""

import os
import tempfile
import joblib
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import (
    EXPIRY_NUMERIC_FEATURES,
    EXPIRY_CATEGORICAL_FEATURES,
    build_expiry_preprocessor
)

RISK_CLASSES = ['High', 'Medium', 'Low']


class RuleBasedExpiryBaseline(BaseEstimator, ClassifierMixin):
    """
    Transparent Rule-Based Baseline reflecting the domain heuristic:
    - High: days_to_expiry <= 90 AND projected_unsold_ratio >= 0.30
    - Medium: days_to_expiry <= 180 AND projected_unsold_ratio >= 0.15 (and not High)
    - Low: otherwise
    """
    def __init__(
        self,
        high_days: int = 90,
        high_ratio: float = 0.30,
        med_days: int = 180,
        med_ratio: float = 0.15
    ):
        self.high_days = high_days
        self.high_ratio = high_ratio
        self.med_days = med_days
        self.med_ratio = med_ratio
        self.classes_ = np.array(RISK_CLASSES)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        days = X['days_to_expiry'].values
        ratio = X['projected_unsold_ratio'].values
        
        preds = []
        for d, r in zip(days, ratio):
            if d <= self.high_days and r >= self.high_ratio:
                preds.append('High')
            elif d <= self.med_days and r >= self.med_ratio:
                preds.append('Medium')
            else:
                preds.append('Low')
        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        probas = []
        for p in preds:
            if p == 'High':
                probas.append([1.0, 0.0, 0.0])
            elif p == 'Medium':
                probas.append([0.0, 1.0, 0.0])
            else:
                probas.append([0.0, 0.0, 1.0])
        return np.array(probas)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = RISK_CLASSES
) -> Dict[str, Any]:
    """Calculate Accuracy, Precision, Recall, F1 (macro & weighted), and Confusion Matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    
    p_weighted = float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    # Per-class metrics
    p_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
    r_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
    f1_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
    
    per_class = {
        label: {
            'precision': round(p_class[i], 4),
            'recall': round(r_class[i], 4),
            'f1': round(f1_class[i], 4)
        } for i, label in enumerate(labels)
    }
    
    return {
        'accuracy': round(acc, 4),
        'precision_macro': round(p_macro, 4),
        'recall_macro': round(r_macro, 4),
        'f1_macro': round(f1_macro, 4),
        'precision_weighted': round(p_weighted, 4),
        'recall_weighted': round(r_weighted, 4),
        'f1_weighted': round(f1_weighted, 4),
        'confusion_matrix': cm,
        'per_class': per_class
    }


def train_expiry_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str = "models",
    output_dir: str = "outputs/model_visualizations"
) -> Dict[str, Any]:
    """
    Train Rule Baseline, Logistic Regression, and Random Forest Classifier.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    target_col = 'expiry_risk'
    feature_cols = EXPIRY_NUMERIC_FEATURES + EXPIRY_CATEGORICAL_FEATURES
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    # 1. Fit Preprocessor strictly on training data
    preprocessor = build_expiry_preprocessor()
    preprocessor.fit(train_df[feature_cols])
    
    X_train_proc = preprocessor.transform(train_df[feature_cols])
    X_val_proc = preprocessor.transform(val_df[feature_cols])
    X_test_proc = preprocessor.transform(test_df[feature_cols])
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(EXPIRY_CATEGORICAL_FEATURES).tolist()
    all_feature_names = EXPIRY_NUMERIC_FEATURES + cat_names
    
    # 2. Rule Baseline
    baseline = RuleBasedExpiryBaseline()
    baseline_val_pred = baseline.predict(val_df)
    baseline_test_pred = baseline.predict(test_df)
    
    # 3. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_proc, y_train)
    lr_val_pred = lr.predict(X_val_proc)
    lr_test_pred = lr.predict(X_test_proc)
    
    # 4. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_proc, y_train)
    rf_val_pred = rf.predict(X_val_proc)
    rf_test_pred = rf.predict(X_test_proc)
    
    # Evaluation Metrics
    metrics = {
        'Rule-Based Baseline': {
            'val': evaluate_classification(y_val, baseline_val_pred),
            'test': evaluate_classification(y_test, baseline_test_pred)
        },
        'Logistic Regression': {
            'val': evaluate_classification(y_val, lr_val_pred),
            'test': evaluate_classification(y_test, lr_test_pred)
        },
        'Random Forest Classifier': {
            'val': evaluate_classification(y_val, rf_val_pred),
            'test': evaluate_classification(y_test, rf_test_pred)
        }
    }
    
    best_name = 'Random Forest Classifier'
    best_model = rf
    
    # Feature Importance from Random Forest
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # Save artifacts
    joblib.dump(preprocessor, os.path.join(models_dir, 'expiry_preprocessor.joblib'))
    joblib.dump(baseline, os.path.join(models_dir, 'expiry_baseline.joblib'))
    joblib.dump(lr, os.path.join(models_dir, 'expiry_classifier_lr.joblib'))
    joblib.dump(rf, os.path.join(models_dir, 'expiry_classifier_rf.joblib'))
    joblib.dump(best_model, os.path.join(models_dir, 'expiry_best_model.joblib'))
    
    feat_imp_df.to_csv(os.path.join(output_dir, 'expiry_feature_importance.csv'), index=False)
    
    # Visual 1: Confusion Matrices Comparison on Test Set
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cm_models = [
        ('Rule Baseline', baseline_test_pred),
        ('Logistic Regression', lr_test_pred),
        ('Random Forest', rf_test_pred)
    ]
    
    for ax, (m_title, preds) in zip(axes, cm_models):
        cm = confusion_matrix(y_test, preds, labels=RISK_CLASSES)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=RISK_CLASSES, yticklabels=RISK_CLASSES,
                    cbar=False)
        ax.set_title(f"{m_title} (Test Set)", fontweight='bold', fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expiry_confusion_matrices.png'), dpi=200)
    plt.close()
    
    # Visual 2: Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    m_names = list(metrics.keys())
    f1_scores = [metrics[m]['test']['f1_macro'] for m in m_names]
    accuracies = [metrics[m]['test']['accuracy'] for m in m_names]
    recalls = [metrics[m]['test']['recall_macro'] for m in m_names]
    
    x = np.arange(len(m_names))
    width = 0.25
    ax.bar(x - width, accuracies, width, label='Accuracy', color='#337ab7')
    ax.bar(x, f1_scores, width, label='Macro F1-Score', color='#5cb85c')
    ax.bar(x + width, recalls, width, label='Macro Recall', color='#f0ad4e')
    ax.set_title("Test Set Classification Performance by Model", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(m_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expiry_model_comparison.png'), dpi=200)
    plt.close()
    
    # Visual 3: Feature Importance Top 10
    plt.figure(figsize=(10, 5))
    top_feats = feat_imp_df.head(10)
    sns.barplot(data=top_feats, x='importance', y='feature', color='#2b5c8f')
    plt.title("Random Forest: Feature Importance for Expiry Risk Classification", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Gini Feature Importance", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expiry_feature_importance.png'), dpi=200)
    plt.close()
    
    return {
        'metrics': metrics,
        'best_model_name': best_name,
        'feature_importance': feat_imp_df.to_dict(orient='records'),
        'feature_names': all_feature_names
    }


def predict_batch_risk(
    batch_dict: Dict[str, Any],
    model,
    preprocessor
) -> Dict[str, Any]:
    """
    Run real-time inference on a batch record or user input.
    Returns predicted risk class and probability distribution.
    """
    df = pd.DataFrame([batch_dict])
    
    if isinstance(model, RuleBasedExpiryBaseline):
        pred_label = model.predict(df)[0]
        probas = model.predict_proba(df)[0]
        prob_dict = {cls: float(probas[i]) for i, cls in enumerate(RISK_CLASSES)}
    else:
        feature_cols = EXPIRY_NUMERIC_FEATURES + EXPIRY_CATEGORICAL_FEATURES
        X_proc = preprocessor.transform(df[feature_cols])
        pred_label = model.predict(X_proc)[0]
        probas = model.predict_proba(X_proc)[0]
        classes = model.classes_.tolist()
        prob_dict = {cls: float(probas[classes.index(cls)]) if cls in classes else 0.0 for cls in RISK_CLASSES}
        
    return {
        'predicted_risk': pred_label,
        'probabilities': prob_dict
    }


def annotate_inventory_predictions(
    inventory_df: pd.DataFrame,
    model,
    preprocessor
) -> pd.DataFrame:
    """
    Apply trained model to annotate the full inventory dataset with model predictions
    and confidence scores for downstream recommendation analysis.
    """
    annotated = inventory_df.copy()
    feature_cols = EXPIRY_NUMERIC_FEATURES + EXPIRY_CATEGORICAL_FEATURES
    
    if isinstance(model, RuleBasedExpiryBaseline):
        preds = model.predict(annotated)
    else:
        X_proc = preprocessor.transform(annotated[feature_cols])
        preds = model.predict(X_proc)
        
    annotated['predicted_expiry_risk'] = preds
    return annotated

