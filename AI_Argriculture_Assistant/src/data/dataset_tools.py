"""Safe Phase-1 inspection and split utilities for class-folder image datasets."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)


def find_class_root(raw_dir: Path, extensions: set[str]) -> Path:
    """Locate the shallowest directory whose direct child folders contain images.

    Requiring direct images avoids mistaking a wrapper folder such as
    ``PlantVillage-Dataset/`` for a disease class folder.
    """
    candidates: list[Path] = []
    for directory in [raw_dir, *[p for p in raw_dir.rglob("*") if p.is_dir()]]:
        children = [p for p in directory.iterdir() if p.is_dir()]
        if any(any(f.is_file() and f.suffix.lower() in extensions for f in child.iterdir()) for child in children):
            candidates.append(directory)
    if not candidates:
        raise ValueError(f"No class folders with supported images found below {raw_dir}")
    return min(candidates, key=lambda p: (len(p.parts), str(p)))


def image_files(class_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def validate_image(path: Path) -> str | None:
    """Return an error string for a corrupt image, otherwise None."""
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return str(error)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inspect_dataset(raw_dir: Path, extensions: set[str], check_duplicates: bool = True) -> dict[str, Any]:
    class_root = find_class_root(raw_dir, extensions)
    # A class directory must hold at least one image itself. This deliberately
    # ignores wrapper directories (for example a duplicated nested
    # ``PlantVillage/`` archive folder) that contain only more directories.
    classes = sorted(
        p.name
        for p in class_root.iterdir()
        if p.is_dir() and any(f.is_file() and f.suffix.lower() in extensions for f in p.iterdir())
    )
    valid_by_class: dict[str, list[str]] = {}
    corrupt: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for class_name in classes:
        valid: list[str] = []
        for file_path in image_files(class_root / class_name, extensions):
            error = validate_image(file_path)
            relative = str(file_path.relative_to(class_root))
            if error:
                corrupt.append({"path": relative, "error": error})
                continue
            valid.append(relative)
            if check_duplicates:
                hashes[digest(file_path)].append(relative)
        valid_by_class[class_name] = valid
    counts = {name: len(paths) for name, paths in valid_by_class.items()}
    median = sorted(counts.values())[len(counts) // 2] if counts else 0
    imbalanced = {name: count for name, count in counts.items() if median and count < 0.5 * median}
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "class_root": str(class_root),
        "classes": classes,
        "class_counts_valid": counts,
        "total_valid_images": sum(counts.values()),
        "corrupt_images": corrupt,
        "exact_duplicate_groups": duplicates,
        "severely_underrepresented_classes": imbalanced,
        "valid_files_by_class": valid_by_class,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    public = {key: value for key, value in report.items() if key != "valid_files_by_class"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(public, indent=2), encoding="utf-8")


def grouped_split(report: dict[str, Any], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    if round(sum(ratios.values()), 8) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
    result: dict[str, list[str]] = {key: [] for key in ratios}
    duplicate_groups = [set(group) for group in report["exact_duplicate_groups"]]
    duplicate_map = {item: group for group in duplicate_groups for item in group}
    for class_name, files in report["valid_files_by_class"].items():
        # Exact duplicate sets are assigned together, preventing duplicate leakage.
        units: list[list[str]] = []
        seen: set[str] = set()
        for file in files:
            if file in seen:
                continue
            unit = sorted(duplicate_map.get(file, {file}))
            seen.update(unit)
            units.append(unit)
        rng = random.Random(f"{seed}:{class_name}")
        rng.shuffle(units)
        total = len(files)
        train_target = round(total * ratios["train"])
        val_target = round(total * ratios["validation"])
        for unit in units:
            target = "train" if len([x for x in result["train"] if x.split('/')[0] == class_name]) < train_target else "validation"
            if target == "validation" and len([x for x in result["validation"] if x.split('/')[0] == class_name]) >= val_target:
                target = "test"
            result[target].extend(unit)
    return result


def materialize_split(class_root: Path, split: dict[str, list[str]], destinations: dict[str, Path], mode: str) -> dict[str, int]:
    if mode not in {"copy", "symlink"}:
        raise ValueError("copy_mode must be 'copy' or 'symlink'")
    counts: Counter[str] = Counter()
    for split_name, relative_paths in split.items():
        destination = destinations[split_name]
        destination.mkdir(parents=True, exist_ok=True)
        for relative in relative_paths:
            source = class_root / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            if mode == "symlink":
                target.symlink_to(source.resolve())
            else:
                shutil.copy2(source, target)
            counts[split_name] += 1
    return dict(counts)
