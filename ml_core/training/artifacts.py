import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from ml_core.training.evaluate import (
    compute_confusion_matrix,
    generate_classification_report,
    compute_per_class_accuracy,
)

def save_confusion_matrix_plot(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
) -> str:
    cm = compute_confusion_matrix(model, X_test, y_test)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

def save_per_class_accuracy_plot(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
) -> str:
    per_class_acc = compute_per_class_accuracy(model, X_test, y_test)

    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, accuracies, color='steelblue', edgecolor='navy')

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.2%}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(range(10))
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

def save_sample_predictions(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
    n_samples: int = 25,
) -> str:
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

    X_sample = X_test[indices]
    y_true = y_test[indices]
    y_pred = model.predict(X_sample)

    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = axes.flatten()

    for i, (img, true_label, pred_label) in enumerate(zip(X_sample, y_true, y_pred)):
        ax = axes[i]

        img_2d = img.reshape(28, 28)
        ax.imshow(img_2d, cmap='gray')

        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color, fontsize=10)
        ax.axis('off')

    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions', fontsize=14, y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

def save_classification_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
) -> str:
    report = generate_classification_report(model, X_test, y_test, output_dict=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return output_path

def save_training_artifacts(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    run_id: Optional[str] = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {}

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix_plot(model, X_test, y_test, cm_path)
    artifacts["confusion_matrix"] = cm_path

    acc_path = os.path.join(output_dir, "per_class_accuracy.png")
    save_per_class_accuracy_plot(model, X_test, y_test, acc_path)
    artifacts["per_class_accuracy"] = acc_path

    samples_path = os.path.join(output_dir, "sample_predictions.png")
    save_sample_predictions(model, X_test, y_test, samples_path)
    artifacts["sample_predictions"] = samples_path

    report_path = os.path.join(output_dir, "classification_report.json")
    save_classification_report(model, X_test, y_test, report_path)
    artifacts["classification_report"] = report_path

    return artifacts
