"""Phase 2 & 3: Train only the new MobileNetV3-Large classifier head."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.preprocessing.transforms import build_transforms
from src.training.model import build_mobilenetv3_large


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_subset(dataset: ImageFolder, per_class: int, seed: int) -> Subset:
    """Return a reproducible capped subset with equal maximum samples per class."""
    indices_by_class: dict[int, list[int]] = {index: [] for index in range(len(dataset.classes))}
    for index, target in enumerate(dataset.targets):
        indices_by_class[target].append(index)
    selected: list[int] = []
    for class_index, indices in indices_by_class.items():
        rng = random.Random(f"{seed}:{class_index}")
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    return Subset(dataset, selected)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    progress_label: str = "",
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += labels.size(0)
            if batch_index % 25 == 0:
                print(f"{progress_label} batch {batch_index}/{len(loader)}", flush=True)
    return total_loss / samples, correct / samples


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["validation_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="s")
    axes[0].set_title("Head Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, train_acc, label="Train Acc", marker="o")
    axes[1].plot(epochs, val_acc, label="Val Acc", marker="s")
    axes[1].set_title("Head Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-checkpoint", default="models/mobilenetv3_head_best.pth")
    parser.add_argument("--epochs", type=int, help="Override training.head_epochs.")
    parser.add_argument("--lr", type=float, help="Override head learning rate.")
    parser.add_argument("--max-train-per-class", type=int, help="Cap train samples per class for quick/reproducible training.")
    parser.add_argument("--max-validation-per-class", type=int, help="Cap val samples per class.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = config["project"]["random_seed"]
    set_seed(seed)

    dataset_config, training_config = config["dataset"], config["training"]
    transforms = build_transforms(config["model"]["image_size"])

    train_data = ImageFolder(dataset_config["split_dirs"]["train"], transform=transforms["train"])
    validation_data = ImageFolder(dataset_config["split_dirs"]["validation"], transform=transforms["validation"])

    expected_classes = dataset_config["selected_classes"]
    if train_data.classes != expected_classes or validation_data.classes != expected_classes:
        raise ValueError("Split class order does not match config dataset.selected_classes.")

    batch_size = training_config["batch_size"]
    effective_train_data = balanced_subset(train_data, args.max_train_per_class, seed) if args.max_train_per_class else train_data
    effective_validation_data = balanced_subset(validation_data, args.max_validation_per_class, seed) if args.max_validation_per_class else validation_data

    train_loader = DataLoader(effective_train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(effective_validation_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    device = choose_device()
    model = build_mobilenetv3_large(len(train_data.classes), freeze_backbone=True).to(device)

    head_lr = args.lr or training_config.get("head_learning_rate", 0.001)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=head_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    epochs = args.epochs or training_config.get("head_epochs", 5)
    models_dir = Path("models")
    outputs_dir = Path("outputs")
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    output_path = Path(args.output_checkpoint)

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, f"epoch {epoch} train")
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, device, progress_label=f"epoch {epoch} validation")
        scheduler.step(validation_accuracy)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f}, acc {validation_accuracy:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": train_data.classes,
                    "config": config,
                    "phase": "head_training",
                    "best_val_acc": best_accuracy,
                },
                output_path,
            )
            print(f"--> Saved improved head checkpoint to {output_path} (Val Acc: {best_accuracy:.4f})")

    elapsed = time.perf_counter() - start

    results = {
        "phase": "head_training",
        "model": "MobileNetV3-Large",
        "pretrained_weights": "MobileNet_V3_Large_Weights.DEFAULT",
        "backbone_frozen": True,
        "device": str(device),
        "epochs": epochs,
        "train_images": len(effective_train_data),
        "full_train_images": len(train_data),
        "validation_images": len(effective_validation_data),
        "full_validation_images": len(validation_data),
        "best_validation_accuracy": best_accuracy,
        "elapsed_seconds": elapsed,
        "history": history,
    }

    results_file = outputs_dir / "head_training_results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Plot training curves
    plot_path = outputs_dir / "head_training_history.png"
    try:
        plot_history(history, plot_path)
        print(f"Saved head training plot to {plot_path}")
    except Exception as e:
        print(f"Plotting error: {e}")

    # Ensure class_names.json is saved
    (models_dir / "class_names.json").write_text(json.dumps(train_data.classes, indent=2), encoding="utf-8")

    print(f"\nHead training completed in {elapsed:.1f}s. Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
