# ML Model Evaluation Reference

## Table of Contents

1. [Classification Metrics](#classification)
2. [Regression Metrics](#regression)
3. [Clustering Metrics](#clustering)
4. [Model Explainability](#explainability)
5. [Evaluation Checklist](#checklist)

---

## 1. Classification Metrics {#classification}

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None
) -> dict:
    """
    Comprehensive classification evaluation.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC, log loss).
        class_names: Optional display names for classes.

    Returns:
        Dictionary of all computed metrics.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "classification_report": report,
        "matthews_corrcoef": mcc,
        "cohen_kappa": kappa,
    }

    if y_prob is not None:
        is_binary = y_prob.ndim == 1 or y_prob.shape[1] == 2
        proba = y_prob[:, 1] if (y_prob.ndim == 2 and is_binary) else y_prob
        metrics["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr" if not is_binary else "raise")
        metrics["log_loss"] = log_loss(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, proba) if is_binary else None

    logger.info("Evaluation complete: AUC=%.4f, MCC=%.4f, Kappa=%.4f",
                metrics.get("roc_auc", 0), mcc, kappa)
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None) -> None:
    """Normalized + raw confusion matrix side by side."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2%"], ["Raw Counts", "Normalized"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob) -> None:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

## 2. Regression Metrics {#regression}

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np
import matplotlib.pyplot as plt


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Comprehensive regression evaluation.

    Returns:
        Dictionary with MAE, RMSE, MAPE, R², adjusted R².
    """
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2,
        "n_samples": n
    }


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Residual plot + distribution of errors."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set(xlabel="Predicted", ylabel="Residuals", title="Residuals vs Predicted")

    import seaborn as sns
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution")

    from scipy import stats
    stats.probplot(residuals, plot=axes[2])
    axes[2].set_title("Q-Q Plot")
    plt.tight_layout()
    plt.show()
```

---

## 3. Clustering Metrics {#clustering}

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evaluate clustering quality using internal validation indices.

    Note: These metrics assume no ground truth is available.
    Higher Calinski-Harabasz = better. Lower Davies-Bouldin = better.
    Silhouette in [-1, 1]; closer to 1 = better.
    """
    return {
        "silhouette_score": silhouette_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
    }


def plot_elbow_curve(inertias: list[float], k_range: range) -> None:
    """Plot elbow curve for K-Means cluster selection."""
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", linewidth=2)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-Cluster SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.show()
```

---

## 4. Model Explainability {#explainability}

```python
import shap
import matplotlib.pyplot as plt
import numpy as np


def shap_summary(model, X_train, X_test=None, model_type: str = "tree") -> None:
    """
    SHAP summary plot for feature importance.

    Args:
        model_type: 'tree' (sklearn/XGBoost), 'linear', or 'kernel' (model-agnostic)
    """
    explainer_map = {
        "tree": shap.TreeExplainer,
        "linear": shap.LinearExplainer,
        "kernel": lambda m: shap.KernelExplainer(m.predict, shap.sample(X_train, 100))
    }
    explainer = explainer_map[model_type](model)
    shap_values = explainer.shap_values(X_test if X_test is not None else X_train)
    shap.summary_plot(shap_values, X_test if X_test is not None else X_train)
    plt.tight_layout()
    plt.show()
```

---

## 5. Evaluation Checklist {#checklist}

Before finalizing any model evaluation, verify:

**Data Leakage**

- [ ] No target-derived features in the feature set
- [ ] Train/test split performed BEFORE any preprocessing fitted on training data only
- [ ] No temporal leakage in time series (always use walk-forward validation)

**Class Imbalance**

- [ ] Check class distribution in train and test sets
- [ ] Report precision, recall, F1 per class — not just accuracy
- [ ] Consider SMOTE, class weighting, or threshold tuning if imbalanced

**Statistical Validity**

- [ ] Cross-validation strategy matches problem type (Stratified K-Fold for classification)
- [ ] Report confidence intervals on key metrics (use bootstrap if needed)
- [ ] Perform paired statistical tests when comparing models (Wilcoxon signed-rank)

**Business Interpretation**

- [ ] Translate metrics into business impact (cost of false positives vs. false negatives)
- [ ] Document model limitations and failure modes
- [ ] Specify monitoring strategy for production deployment
