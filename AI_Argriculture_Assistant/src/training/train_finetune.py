"""Phase 4: Fine-tune MobileNetV3-Large by unfreezing upper backbone layers."""
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.preprocessing.transforms import build_transforms
from src.training.model import build_mobilenetv3_large, unfreeze_upper_layers


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
                # Clip gradient norm to avoid unstable fine-tuning
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, train_acc, label="Train Acc", marker="o")
    axes[1].plot(epochs, val_acc, label="Val Acc", marker="s")
    axes[1].set_title("Accuracy")
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
    parser.add_argument("--head-checkpoint", default="models/mobilenetv3_head_best.pth", help="Pretrained head checkpoint path.")
    parser.add_argument("--output-checkpoint", default="models/mobilenetv3_best.pth", help="Output path for final best model.")
    parser.add_argument("--epochs", type=int, help="Override training.finetune_epochs.")
    parser.add_argument("--lr", type=float, help="Override fine-tuning learning rate.")
    parser.add_argument("--unfreeze-blocks", type=int, default=3, help="Number of upper InvertedResidual blocks to unfreeze.")
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
    num_classes = len(train_data.classes)

    # Initialize model
    model = build_mobilenetv3_large(num_classes=num_classes, freeze_backbone=True)

    # Load Stage 1 weights if available
    head_ckpt_path = Path(args.head_checkpoint)
    if head_ckpt_path.exists():
        print(f"Loading Stage 1 weights from {head_ckpt_path}...")
        checkpoint = torch.load(head_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Warning: {head_ckpt_path} not found. Starting fine-tuning from ImageNet weights.")

    # Unfreeze upper backbone blocks
    model = unfreeze_upper_layers(model, num_blocks=args.unfreeze_blocks).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} / {sum(p.numel() for p in model.parameters()):,}")

    finetune_lr = args.lr or training_config.get("finetune_learning_rate", 0.0001)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(trainable_params, lr=finetune_lr, weight_decay=1e-4)

    epochs = args.epochs or training_config.get("finetune_epochs", 5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
        scheduler.step()

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
                    "phase": "fine_tuning",
                    "best_val_acc": best_accuracy,
                    "unfrozen_blocks": args.unfreeze_blocks,
                },
                output_path,
            )
            print(f"--> Saved improved model checkpoint to {output_path} (Val Acc: {best_accuracy:.4f})")

    elapsed = time.perf_counter() - start

    results = {
        "phase": "fine_tuning",
        "model": "MobileNetV3-Large",
        "unfrozen_blocks": args.unfreeze_blocks,
        "device": str(device),
        "epochs": epochs,
        "learning_rate": finetune_lr,
        "train_images": len(effective_train_data),
        "full_train_images": len(train_data),
        "validation_images": len(effective_validation_data),
        "full_validation_images": len(validation_data),
        "best_validation_accuracy": best_accuracy,
        "elapsed_seconds": elapsed,
        "history": history,
    }

    results_file = outputs_dir / "finetuning_results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Plot training curves
    plot_path = outputs_dir / "finetuning_history.png"
    try:
        plot_history(history, plot_path)
        print(f"Saved fine-tuning plot to {plot_path}")
    except Exception as e:
        print(f"Plotting error: {e}")

    # Ensure class_names.json is up to date
    (models_dir / "class_names.json").write_text(json.dumps(train_data.classes, indent=2), encoding="utf-8")

    print(f"\nFine-tuning completed in {elapsed:.1f}s. Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best model saved to {output_path}")


if __name__ == "__main__":
    main()
