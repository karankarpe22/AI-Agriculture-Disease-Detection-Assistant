"""CLI: inspect an actual downloaded image dataset without modifying it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.data.dataset_tools import inspect_dataset, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw-dir", help="Override dataset.raw_dir")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = config["dataset"]
    raw_dir = Path(args.raw_dir or dataset["raw_dir"])
    extensions = set(dataset["allowed_extensions"])
    has_images = raw_dir.exists() and any(
        path.is_file() and path.suffix.lower() in extensions for path in raw_dir.rglob("*")
    )
    if not has_images:
        print(f"DATASET NOT FOUND: place the downloaded class-folder dataset under {raw_dir.resolve()}", file=sys.stderr)
        return 2
    report = inspect_dataset(raw_dir, extensions)
    output = Path("outputs/dataset_statistics.json")
    write_report(report, output)
    print(f"Class root: {report['class_root']}")
    print(f"Classes ({len(report['classes'])}): {', '.join(report['classes'])}")
    print(f"Valid images: {report['total_valid_images']}")
    print(f"Class distribution: {report['class_counts_valid']}")
    print(f"Corrupt images: {len(report['corrupt_images'])}")
    print(f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}")
    print(f"Severely underrepresented classes: {report['severely_underrepresented_classes'] or 'none'}")
    print(f"Detailed report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
