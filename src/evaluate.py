"""
Full evaluation of trained bird classification models on the held-out test set.

Generates:
  - Classification report (per-class precision, recall, F1)
  - Confusion matrix heatmap
  - Comparison table (HOG+SVM vs Baseline CNN vs EfficientNetB0)
  - Saves metrics as JSON and CSV

Usage:
    # Evaluate EfficientNetB0 (default, recommended final model)
    python -m src.evaluate

    # Evaluate the baseline CNN
    python -m src.evaluate --model baseline_cnn

    # Evaluate both models and show comparison
    python -m src.evaluate --compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import (
    IMG_SIZE,
    IMG_SIZE_BASELINE,
    build_datasets,
    get_class_names,
    load_class_labels,
)
from src.gpu_utils import setup_gpu

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
CM_DIR = RESULTS_DIR / "confusion_matrix"


# ──────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────
def predict_dataset(model: keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a full dataset; return (y_true, y_pred)."""
    y_true_list, y_pred_list = [], []
    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images, verbose=0)
        y_true_list.append(np.argmax(batch_labels.numpy(), axis=1))
        y_pred_list.append(np.argmax(preds, axis=1))
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


# ──────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    accuracy: float,
) -> None:
    """Save a labeled confusion matrix heatmap."""
    CM_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Normalize to percentages for readability
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm_norm, annot=True, fmt=".0f",
        cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.4, cbar_kws={"label": "% of true class"},
    )
    ax.set_title(f"{model_name} — Confusion Matrix (Test Accuracy: {accuracy:.2%})", fontsize=14)
    ax.set_xlabel("Predicted Species", fontsize=11)
    ax.set_ylabel("True Species", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("-", "_") + "_confusion_matrix.png"
    fig.savefig(CM_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {CM_DIR}/{fname}")


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> None:
    """Bar chart of per-class accuracy."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    per_class_acc = []
    for cls_id in range(len(class_names)):
        mask = y_true == cls_id
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append((y_pred[mask] == cls_id).mean())

    sorted_idx = np.argsort(per_class_acc)
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        [class_names[i] for i in sorted_idx],
        [per_class_acc[i] for i in sorted_idx],
        color="steelblue", edgecolor="white",
    )
    ax.set_xlabel("Per-class Test Accuracy", fontsize=11)
    ax.set_title(f"{model_name} — Per-Class Accuracy", fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=np.mean(per_class_acc), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(per_class_acc):.2%}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("-", "_") + "_per_class_accuracy.png"
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"Per-class accuracy chart saved → {PLOTS_DIR}/{fname}")


def plot_model_comparison(comparison_data: list[dict]) -> None:
    """Bar chart comparing all models' test accuracy."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not comparison_data:
        return
    names = [d["Model"] for d in comparison_data]
    accs = [d.get("Test Accuracy", 0.0) for d in comparison_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728", "#aec7e8", "#1f77b4", "#2ca02c"][:len(names)]
    bars = ax.bar(names, [a * 100 for a in accs], color=colors, edgecolor="white", width=0.55)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison — Test Accuracy", fontsize=13)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Comparison chart saved → {PLOTS_DIR}/model_comparison.png")


# ──────────────────────────────────────────────
# Per-model evaluation
# ──────────────────────────────────────────────
def evaluate_cnn_model(
    model_path: Path,
    img_size: tuple[int, int],
    model_label: str,
) -> dict:
    """Load a Keras model and evaluate on the test set."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {model_label}")
    print(f"  Model file: {model_path}")
    print(f"{'=' * 60}")

    if not model_path.exists():
        print(f"  ✗ Model file not found — skipping.")
        return {}

    model = keras.models.load_model(str(model_path))
    class_names = get_class_names()

    _, _, test_ds, _ = build_datasets(img_size=img_size, batch_size=32)

    print("  Running inference on test set...")
    y_true, y_pred = predict_dataset(model, test_ds)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  Results for {model_label}:")
    print(f"    Test Accuracy    : {acc:.4f}  ({acc:.2%})")
    print(f"    Macro Precision  : {macro_precision:.4f}")
    print(f"    Macro Recall     : {macro_recall:.4f}")
    print(f"    Macro F1         : {macro_f1:.4f}")
    print(f"    Weighted F1      : {weighted_f1:.4f}")

    # Per-class report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Save metrics JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_label.lower().replace(" ", "_").replace("-", "_")
    metrics = {
        "model": model_label,
        "test_accuracy": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
    }
    (METRICS_DIR / f"{safe_name}_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Test metrics saved → {METRICS_DIR}/{safe_name}_test_metrics.json")

    # Save report as CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(METRICS_DIR / f"{safe_name}_classification_report.csv")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, model_label, acc)
    plot_per_class_accuracy(y_true, y_pred, class_names, model_label)

    return {
        "Model": model_label,
        "Test Accuracy": acc,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }


# ──────────────────────────────────────────────
# Training history plots (read from saved JSON)
# ──────────────────────────────────────────────
def plot_history_from_json(metrics_json: Path, model_name: str) -> None:
    """Re-generate training curves from saved metrics JSON."""
    if not metrics_json.exists():
        return
    data = json.loads(metrics_json.read_text())

    def _get_hist(key: str) -> list:
        # Handle both direct and nested history keys
        if "history" in data and key in data["history"]:
            return data["history"][key]
        if key in data:
            return data[key]
        return []

    # For EfficientNet, combine stage1 + stage2 histories
    def _combined(key: str) -> list:
        h1 = data.get("stage1_history", {}).get(key, [])
        h2 = data.get("stage2_history", {}).get(key, [])
        return list(h1) + list(h2) if (h1 or h2) else _get_hist(key)

    acc = _combined("accuracy")
    val_acc = _combined("val_accuracy")
    loss = _combined("loss")
    val_loss = _combined("val_loss")

    if not acc:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(acc) + 1)
    safe = model_name.lower().replace(" ", "_").replace("-", "_")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, acc, label="Train", linewidth=2)
    axes[0].plot(epochs_range, val_acc, label="Validation", linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs_range, loss, label="Train", linewidth=2)
    axes[1].plot(epochs_range, val_loss, label="Validation", linewidth=2)
    axes[1].set_title(f"{model_name} — Loss", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{safe}_training_curves.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained bird classification models")
    parser.add_argument(
        "--model",
        choices=["efficientnet", "baseline_cnn", "both"],
        default="both",
        help="Which CNN model to evaluate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available models and generate comparison charts",
    )
    args = parser.parse_args()
    if args.compare:
        args.model = "both"

    setup_gpu(verbose=True)

    comparison_rows: list[dict] = []

    # ── HOG+SVM (from saved outputs/) ────────────
    svm_metrics_path = Path("outputs/metrics.json")
    if svm_metrics_path.exists():
        svm_data = json.loads(svm_metrics_path.read_text())
        svm_acc = svm_data.get("accuracy", 0.0)
        svm_report = svm_data.get("report", {})
        macro_f1_svm = svm_report.get("macro avg", {}).get("f1-score", 0.0)
        comparison_rows.append({
            "Model": "HOG + Linear SVM",
            "Test Accuracy": svm_acc,
            "Macro Precision": svm_report.get("macro avg", {}).get("precision", 0.0),
            "Macro Recall": svm_report.get("macro avg", {}).get("recall", 0.0),
            "Macro F1": macro_f1_svm,
            "Weighted F1": svm_report.get("weighted avg", {}).get("f1-score", 0.0),
        })
        print(f"\nSVM baseline (from outputs/metrics.json): accuracy={svm_acc:.2%}")

    # ── Baseline CNN ──────────────────────────────
    if args.model in ("baseline_cnn", "both"):
        row = evaluate_cnn_model(
            MODELS_DIR / "baseline_cnn.keras",
            IMG_SIZE_BASELINE,
            "Baseline CNN",
        )
        if row:
            comparison_rows.append(row)
        # Re-plot training curves from saved JSON
        plot_history_from_json(METRICS_DIR / "baseline_cnn_metrics.json", "Baseline CNN")

    # ── EfficientNetB0 ────────────────────────────
    if args.model in ("efficientnet", "both"):
        row = evaluate_cnn_model(
            MODELS_DIR / "efficientnet_finetuned.keras",
            IMG_SIZE,
            "EfficientNetB0 Fine-Tuned",
        )
        if row:
            comparison_rows.append(row)
        plot_history_from_json(METRICS_DIR / "efficientnet_metrics.json", "EfficientNetB0")

    # ── Comparison table ──────────────────────────
    if comparison_rows:
        print("\n" + "=" * 70)
        print("  FINAL MODEL COMPARISON")
        print("=" * 70)
        df = pd.DataFrame(comparison_rows)
        df = df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
        df["Test Accuracy"] = df["Test Accuracy"].map("{:.2%}".format)
        df["Macro Precision"] = df["Macro Precision"].map("{:.4f}".format)
        df["Macro Recall"] = df["Macro Recall"].map("{:.4f}".format)
        df["Macro F1"] = df["Macro F1"].map("{:.4f}".format)
        df["Weighted F1"] = df["Weighted F1"].map("{:.4f}".format)
        print(df.to_string(index=False))

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        # Save with raw floats for the chart
        raw_rows = [r.copy() for r in comparison_rows]
        for r in raw_rows:
            r["Test Accuracy"] = float(r["Test Accuracy"]) if isinstance(r["Test Accuracy"], float) else r["Test Accuracy"]
        df_raw = pd.DataFrame(comparison_rows)
        df_raw.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
        print(f"\nComparison table saved → {METRICS_DIR}/model_comparison.csv")

        plot_model_comparison(comparison_rows)

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
