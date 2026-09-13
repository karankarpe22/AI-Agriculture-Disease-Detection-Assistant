"""Generate a concise, human-readable Phase-1 dataset statistics report."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def split_class_counts(manifest: dict[str, list[str]]) -> dict[str, Counter[str]]:
    return {
        split: Counter(Path(relative).parts[0] for relative in files)
        for split, files in manifest.items()
    }


def main() -> None:
    outputs = Path("outputs")
    statistics = json.loads((outputs / "dataset_statistics.json").read_text(encoding="utf-8"))
    manifest = json.loads((outputs / "split_manifest.json").read_text(encoding="utf-8"))
    split_counts = split_class_counts(manifest)
    sets = {name: set(paths) for name, paths in manifest.items()}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    all_duplicates = statistics["exact_duplicate_groups"]
    assignment = {path: split for split, paths in manifest.items() for path in paths}
    duplicate_leaks = sum(
        len({assignment[path] for path in group if path in assignment}) > 1
        for group in all_duplicates
    )
    lines = [
        "# Dataset Statistics Report",
        "",
        "## Source and selection",
        "",
        f"- Class-folder root: `{statistics['class_root']}`",
        f"- Selected classes: {len(statistics['selected_classes'])}",
        f"- Valid selected images: {statistics['total_valid_images']}",
        f"- Corrupt images: {len(statistics['corrupt_images'])}",
        f"- Exact duplicate groups in selected source data: {len(all_duplicates)}",
        "",
        "## Split method",
        "",
        "Deterministic class-stratified 70/15/15 split (seed 42). Exact duplicate groups are assigned as a unit. Near-duplicate detection is not included and remains a limitation.",
        "",
        "| Class | Train | Validation | Test | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in statistics["selected_classes"]:
        values = [split_counts[split][class_name] for split in ("train", "validation", "test")]
        lines.append(f"| {class_name} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")
    totals = [len(manifest[split]) for split in ("train", "validation", "test")]
    lines.extend([
        f"| **Total** | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** | **{sum(totals)}** |",
        "",
        "## Leakage checks",
        "",
        f"- File-path overlap: train/validation {overlaps['train_validation']}, train/test {overlaps['train_test']}, validation/test {overlaps['validation_test']}.",
        f"- Exact-duplicate groups crossing partitions: {duplicate_leaks}.",
        "",
        "## Preprocessing",
        "",
        "All images use RGB tensor conversion and ImageNet normalization (mean 0.485/0.456/0.406; standard deviation 0.229/0.224/0.225) after resizing to 224×224. Training only uses a random crop with scale 0.85–1.00, horizontal flip, ±12° rotation, and modest brightness/contrast jitter. Validation and test transforms are deterministic.",
    ])
    (outputs / "dataset_statistics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote outputs/dataset_statistics_report.md")


if __name__ == "__main__":
    main()
