import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    return metrics
def compute_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
def generate_classification_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dict: bool = False
) -> Any:
    y_pred = model.predict(X_test)
    target_names = [f"Digit {i}" for i in range(10)]
    return classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )
def compute_per_class_accuracy(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[int, float]:
    y_pred = model.predict(X_test)
    per_class_acc = {}
    for label in range(10):
        mask = y_test == label
        if mask.sum() > 0:
            per_class_acc[label] = accuracy_score(y_test[mask], y_pred[mask])
        else:
            per_class_acc[label] = 0.0
    return per_class_acc
