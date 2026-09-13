"""CLI: create a reproducible, duplicate-aware 70/15/15 split after inspection."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.dataset_tools import find_class_root, grouped_split, inspect_dataset, materialize_split, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="Allow writing into existing split directories.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset, seed = config["dataset"], config["project"]["random_seed"]
    raw_dir, extensions = Path(dataset["raw_dir"]), set(dataset["allowed_extensions"])
    report = inspect_dataset(raw_dir, extensions)
    selected = set(dataset.get("selected_classes") or report["classes"])
    unknown = selected - set(report["classes"])
    if unknown:
        raise ValueError(f"selected_classes not present in dataset: {sorted(unknown)}")
    report["valid_files_by_class"] = {key: value for key, value in report["valid_files_by_class"].items() if key in selected}
    report["classes"] = sorted(selected)
    report["class_counts_valid"] = {key: len(value) for key, value in report["valid_files_by_class"].items()}
    report["total_valid_images"] = sum(report["class_counts_valid"].values())
    destinations = {key: Path(value) for key, value in dataset["split_dirs"].items()}
    if not args.force and any(
        path.exists() and any(item.name != ".gitkeep" for item in path.iterdir())
        for path in destinations.values()
    ):
        raise FileExistsError("Split directory is non-empty. Review it, clear it manually, or rerun with --force.")
    split = grouped_split(report, dataset["split_ratios"], seed)
    class_root = find_class_root(raw_dir, extensions)
    counts = materialize_split(class_root, split, destinations, dataset["copy_mode"])
    report["split_counts"] = counts
    report["selected_classes"] = sorted(selected)
    write_report(report, Path("outputs/dataset_statistics.json"))
    Path("outputs/split_manifest.json").write_text(__import__("json").dumps(split, indent=2), encoding="utf-8")
    print(f"Created split: {counts}. Manifest: outputs/split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
