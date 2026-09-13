"""Phase 6: Image quality check module for leaf inspection."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


class ImageQualityChecker:
    """Lightweight quality inspection for leaf images before classification."""

    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        blur_threshold: float = 35.0,
        dark_threshold: float = 30.0,
        bright_threshold: float = 230.0,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

        # 3x3 Laplacian kernel for sharpness / blur detection
        self._laplacian_kernel = np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
        )

    def assess_image(self, image_source: str | Path | bytes | BinaryIO | Image.Image) -> dict:
        """Evaluate image usability for disease prediction.

        Returns a dictionary with:
        - is_valid (bool): whether image meets minimum diagnostic criteria
        - issues (list[str]): list of identified issues if any
        - metrics (dict): resolution, blur score, brightness score
        - message (str): user-facing guidance
        """
        # Load image safely
        try:
            if isinstance(image_source, Image.Image):
                img = image_source.convert("RGB")
            elif isinstance(image_source, (str, Path)):
                img = Image.open(image_source).convert("RGB")
            elif isinstance(image_source, bytes):
                img = Image.open(io.BytesIO(image_source)).convert("RGB")
            else:
                img = Image.open(image_source).convert("RGB")
        except Exception as e:
            return {
                "is_valid": False,
                "issues": ["unreadable_or_corrupt"],
                "metrics": {},
                "message": f"Unable to read or decode image. Please ensure the file is an undamaged image (JPG/PNG). Error: {str(e)}",
            }

        width, height = img.size
        issues: list[str] = []
        user_messages: list[str] = []

        # 1. Resolution check
        if width < self.min_width or height < self.min_height:
            issues.append("low_resolution")
            user_messages.append(
                f"Image resolution ({width}x{height}) is too small. Minimum required is {self.min_width}x{self.min_height}."
            )

        # Convert to grayscale numpy array for photometric & blur checks
        gray = np.array(img.convert("L"), dtype=np.float32)

        # 2. Brightness / Exposure check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.dark_threshold:
            issues.append("too_dark")
            user_messages.append("The leaf image is excessively dark or underexposed. Please capture in adequate daylight.")
        elif mean_brightness > self.bright_threshold:
            issues.append("too_bright")
            user_messages.append("The leaf image is excessively bright or overexposed. Please avoid direct harsh glare.")

        # 3. Blur / Sharpness check (Laplacian variance)
        # Downsample large images for fast blur calculation
        if max(width, height) > 600:
            thumb_size = (300, int(300 * height / width)) if width >= height else (int(300 * width / height), 300)
            gray_for_blur = np.array(img.resize(thumb_size, Image.Resampling.BILINEAR).convert("L"), dtype=np.float32)
        else:
            gray_for_blur = gray

        laplacian = convolve2d(gray_for_blur, self._laplacian_kernel, mode="valid")
        blur_score = float(np.var(laplacian))

        if blur_score < self.blur_threshold:
            issues.append("excessive_blur")
            user_messages.append(
                f"The image appears blurry (sharpness score: {blur_score:.1f}). Please hold the camera steady and focus clearly on the leaf blade."
            )

        is_valid = len(issues) == 0
        if is_valid:
            message = "Image quality check passed. Suitable for disease analysis."
        else:
            message = " " .join(user_messages)

        return {
            "is_valid": is_valid,
            "issues": issues,
            "metrics": {
                "width": width,
                "height": height,
                "brightness": round(mean_brightness, 1),
                "blur_score": round(blur_score, 1),
            },
            "message": message,
        }


# Module singleton for convenience
default_checker = ImageQualityChecker()


def check_image_quality(image_source: str | Path | bytes | BinaryIO | Image.Image) -> dict:
    """Convenience helper for image quality checking."""
    return default_checker.assess_image(image_source)
