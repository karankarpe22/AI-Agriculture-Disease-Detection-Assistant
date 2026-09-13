"""Phase 6: Disease prediction module with confidence estimation and Grad-CAM explainability."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.preprocessing.quality_check import check_image_quality
from src.preprocessing.transforms import build_transforms
from src.training.model import build_mobilenetv3_large


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_class_name(raw_class: str) -> tuple[str, str]:
    """Parse dataset folder name into clean (Crop, Disease) display names."""
    # Standard mappings for the 13 classes
    mapping = {
        "Pepper__bell___Bacterial_spot": ("Bell pepper", "Bacterial spot"),
        "Pepper__bell___healthy": ("Bell pepper", "Healthy"),
        "Potato___Early_blight": ("Potato", "Early blight"),
        "Potato___Late_blight": ("Potato", "Late blight"),
        "Tomato_Bacterial_spot": ("Tomato", "Bacterial spot"),
        "Tomato_Early_blight": ("Tomato", "Early blight"),
        "Tomato_Late_blight": ("Tomato", "Late blight"),
        "Tomato_Leaf_Mold": ("Tomato", "Leaf mold"),
        "Tomato_Septoria_leaf_spot": ("Tomato", "Septoria leaf spot"),
        "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider mites (Two-spotted spider mite)"),
        "Tomato__Target_Spot": ("Tomato", "Target spot"),
        "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomato", "Tomato yellow leaf curl virus"),
        "Tomato_healthy": ("Tomato", "Healthy"),
    }
    if raw_class in mapping:
        return mapping[raw_class]

    # Fallback heuristic parser
    clean = raw_class.replace("___", " - ").replace("__", " - ").replace("_", " ")
    parts = [p.strip().title() for p in clean.split("-") if p.strip()]
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return "Crop", clean


class DiseasePredictor:
    """Predicts plant disease from leaf images using MobileNetV3-Large."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path = "config.yaml",
        confidence_threshold: float = 0.60,
    ) -> None:
        self.device = choose_device()
        self.config_path = Path(config_path)
        self.confidence_threshold = confidence_threshold

        # Load config
        if self.config_path.exists():
            self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        else:
            self.config = {"model": {"image_size": 224}}

        # Find best available checkpoint
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.class_names: list[str] = self._load_class_names()
        self.model = self._load_model()

        # Build transform
        image_size = self.config.get("model", {}).get("image_size", 224)
        transform_dict = build_transforms(image_size)
        self.transform = transform_dict["test"]

    def _resolve_checkpoint(self, preferred: str | Path | None) -> Path:
        candidates = [
            Path(preferred) if preferred else None,
            Path("models/mobilenetv3_best.pth"),
            Path("models/mobilenetv3_head_best.pth"),
            Path("models/mobilenetv3_head_smoke_best.pth"),
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        raise FileNotFoundError(
            "No trained model checkpoint found in models/ directory. Please run Phase 2/3 training first."
        )

    def _load_class_names(self) -> list[str]:
        class_names_path = Path("models/class_names.json")
        if class_names_path.exists():
            return json.loads(class_names_path.read_text(encoding="utf-8"))
        # Fallback to config
        return self.config.get("dataset", {}).get("selected_classes", [])

    def _load_model(self) -> torch.nn.Module:
        num_classes = len(self.class_names)
        model = build_mobilenetv3_large(num_classes=num_classes, freeze_backbone=False)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        image_input: str | Path | bytes | BinaryIO | Image.Image,
        top_k: int = 3,
        bypass_quality_check: bool = False,
    ) -> dict:
        """Complete prediction pipeline: Quality Check -> Inference -> Uncertainty."""
        # 1. Image Quality Check
        quality_result = check_image_quality(image_input)
        if not quality_result["is_valid"] and not bypass_quality_check:
            return {
                "success": False,
                "error_type": "quality_check_failed",
                "quality": quality_result,
                "crop": None,
                "disease": None,
                "raw_class": None,
                "confidence": 0.0,
                "top_predictions": [],
                "is_low_confidence": True,
                "warning": quality_result["message"],
            }

        # 2. Convert to PIL Image
        try:
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            elif isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            else:
                img = Image.open(image_input).convert("RGB")
        except Exception as e:
            return {
                "success": False,
                "error_type": "image_decode_error",
                "quality": quality_result,
                "confidence": 0.0,
                "warning": f"Could not decode image: {str(e)}",
            }

        # 3. Preprocessing & Tensor conversion
        tensor_img = self.transform(img).unsqueeze(0).to(self.device)

        # 4. Model Inference
        with torch.no_grad():
            logits = self.model(tensor_img)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # 5. Extract top prediction & top-k
        k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            raw_cname = self.class_names[idx]
            crop, disease = format_class_name(raw_cname)
            top_predictions.append({
                "class_index": idx,
                "raw_class": raw_cname,
                "crop": crop,
                "disease": disease,
                "confidence": round(float(prob), 4),
                "confidence_percentage": round(float(prob) * 100, 2),
            })

        best = top_predictions[0]
        confidence = best["confidence"]
        is_low_confidence = confidence < self.confidence_threshold

        warning = None
        if is_low_confidence:
            warning = (
                f"Prediction confidence is low ({best['confidence_percentage']:.1f}%). "
                "The symptoms may be in an early stage or affected by environmental factors. "
                "Please upload a clearer image or consult an agricultural expert."
            )

        return {
            "success": True,
            "quality": quality_result,
            "crop": best["crop"],
            "disease": best["disease"],
            "raw_class": best["raw_class"],
            "confidence": confidence,
            "confidence_percentage": best["confidence_percentage"],
            "top_predictions": top_predictions,
            "is_low_confidence": is_low_confidence,
            "warning": warning,
            "model_used": "MobileNetV3-Large",
            "checkpoint": str(self.checkpoint_path.name),
        }

    def generate_gradcam(self, image_input: str | Path | bytes | BinaryIO | Image.Image) -> Image.Image | None:
        """Lightweight Grad-CAM visualization for visual explainability."""
        try:
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            elif isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            else:
                img = Image.open(image_input).convert("RGB")

            # Hook into the last convolutional feature block
            target_layer = self.model.features[-1]
            activations: list[torch.Tensor] = []
            gradients: list[torch.Tensor] = []

            def forward_hook(module, input, output):
                activations.append(output)

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            handle_f = target_layer.register_forward_hook(forward_hook)
            handle_b = target_layer.register_full_backward_hook(backward_hook)

            tensor_img = self.transform(img).unsqueeze(0).to(self.device)
            tensor_img.requires_grad = True

            self.model.zero_grad()
            logits = self.model(tensor_img)
            pred_class = logits.argmax(dim=1).item()
            logits[0, pred_class].backward()

            handle_f.remove()
            handle_b.remove()

            if not activations or not gradients:
                return None

            act = activations[0].detach()  # [1, C, H, W]
            grad = gradients[0].detach()   # [1, C, H, W]
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = torch.sum(weights * act, dim=1, keepdim=True)    # [1, 1, H, W]
            cam = torch.relu(cam).squeeze().cpu().numpy()

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            # Resize CAM to original image size
            cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.Resampling.BILINEAR)
            cam_arr = np.array(cam_img, dtype=np.float32) / 255.0

            # Create colored heatmap overlay (red/yellow for high attention)
            orig_arr = np.array(img, dtype=np.float32)
            heatmap = np.zeros_like(orig_arr)
            heatmap[:, :, 0] = cam_arr * 255.0  # Red
            heatmap[:, :, 1] = (1.0 - cam_arr) * 150.0  # Greenish-yellow

            blended = np.clip(0.6 * orig_arr + 0.4 * heatmap, 0, 255).astype(np.uint8)
            return Image.fromarray(blended)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            return None
