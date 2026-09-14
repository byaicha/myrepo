import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, confusion_matrix
)


def best_f1_threshold(y_true, proba, start=0.05, stop=0.70, steps=131):
    thresholds = np.linspace(start, stop, steps)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_model(name, model, X_val, y_val, X_test, y_test):
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    test_proba = model.predict_proba(X_test)[:, 1]
    pred = test_proba >= threshold
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "f1": f1_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
