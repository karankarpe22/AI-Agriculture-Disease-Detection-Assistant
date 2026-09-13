from pathlib import Path

from PIL import Image

from src.data.dataset_tools import inspect_dataset
from src.data.generate_statistics_report import split_class_counts


def test_inspection_detects_classes_and_corruption(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "Healthy").mkdir(parents=True)
    (root / "Blight").mkdir()
    Image.new("RGB", (16, 16), "green").save(root / "Healthy" / "leaf.jpg")
    (root / "Blight" / "bad.jpg").write_bytes(b"not an image")
    report = inspect_dataset(root, {".jpg"})
    assert report["classes"] == ["Blight", "Healthy"]
    assert report["total_valid_images"] == 1
    assert len(report["corrupt_images"]) == 1


def test_split_class_counts() -> None:
    counts = split_class_counts({"train": ["healthy/a.jpg", "blight/b.jpg"], "test": ["healthy/c.jpg"]})
    assert counts["train"]["healthy"] == 1
    assert counts["test"]["healthy"] == 1
