"""Phase 5: Comprehensive evaluation of the plant disease classifier."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.preprocessing.transforms import build_transforms
from src.training.model import build_mobilenetv3_large


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    # Shorten class names for readability in the matrix
    display_names = [name.replace("___", " ").replace("__", " ").replace("_", " ") for name in class_names]
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True,
    )
    plt.title("Confusion Matrix - Crop Disease Detection (MobileNetV3)", fontsize=14, pad=15)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_confidences: list[float] = []
    inference_times: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            start_time = time.perf_counter()
            logits = model(images)
            batch_time = time.perf_counter() - start_time

            probabilities = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probabilities, dim=1)

            per_sample_time = batch_time / images.size(0)
            inference_times.extend([per_sample_time] * images.size(0))

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            all_confidences.extend(confidences.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    accuracy = float((y_true == y_pred).mean())
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    avg_inference_time_ms = float(np.mean(inference_times) * 1000)
    avg_confidence = float(np.mean(all_confidences))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        if cname in report_dict:
            per_class_metrics[cname] = {
                "precision": report_dict[cname]["precision"],
                "recall": report_dict[cname]["recall"],
                "f1-score": report_dict[cname]["f1-score"],
                "support": report_dict[cname]["support"],
            }

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "average_inference_time_ms": avg_inference_time_ms,
        "average_confidence": avg_confidence,
        "total_test_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_text,
        "per_class_metrics": per_class_metrics,
        "raw_confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-checkpoint", default="models/mobilenetv3_best.pth", help="Checkpoint to evaluate.")
    parser.add_argument("--fallback-checkpoint", default="models/mobilenetv3_head_best.pth")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-per-class", type=int, help="Cap test samples per class if evaluating quickly.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    transforms = build_transforms(config["model"]["image_size"])

    test_data = ImageFolder(args.test_dir, transform=transforms["test"])
    class_names = test_data.classes

    device = choose_device()

    ckpt_path = Path(args.model_checkpoint)
    if not ckpt_path.exists():
        fallback_path = Path(args.fallback_checkpoint)
        if fallback_path.exists():
            print(f"Primary checkpoint {ckpt_path} not found. Using fallback: {fallback_path}")
            ckpt_path = fallback_path
        else:
            smoke_path = Path("models/mobilenetv3_head_smoke_best.pth")
            if smoke_path.exists():
                print(f"Using smoke checkpoint: {smoke_path}")
                ckpt_path = smoke_path
            else:
                raise FileNotFoundError(f"No checkpoint found at {ckpt_path} or {fallback_path}")

    print(f"Loading checkpoint {ckpt_path} on device {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_mobilenetv3_large(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Evaluating model on {len(test_data)} test images across {len(class_names)} classes...")
    start_eval = time.perf_counter()
    metrics = evaluate_model(model, test_loader, device, class_names)
    eval_elapsed = time.perf_counter() - start_eval

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Save confusion matrix plot
    cm_plot_path = outputs_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics["raw_confusion_matrix"], class_names, cm_plot_path)

    # Save classification report text
    report_file = outputs_dir / "classification_report.txt"
    report_file.write_text(metrics["classification_report_text"], encoding="utf-8")

    # Clean metrics for JSON export
    raw_cm = metrics.pop("raw_confusion_matrix")
    metrics["model_checkpoint"] = str(ckpt_path)
    metrics["device"] = str(device)
    metrics["eval_elapsed_seconds"] = eval_elapsed

    metrics_file = outputs_dir / "evaluation.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (ACTUAL MEASURED TEST METRICS)")
    print("=" * 50)
    print(f"Test Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:        {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:           {metrics['recall_macro']:.4f}")
    print(f"Macro F1-Score:         {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score:      {metrics['f1_weighted']:.4f}")
    print(f"Avg Inference Time:     {metrics['average_inference_time_ms']:.2f} ms/image")
    print(f"Average Confidence:     {metrics['average_confidence'] * 100:.2f}%")
    print(f"Saved report to:        {report_file}")
    print(f"Saved confusion matrix: {cm_plot_path}")
    print(f"Saved metrics JSON:     {metrics_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
