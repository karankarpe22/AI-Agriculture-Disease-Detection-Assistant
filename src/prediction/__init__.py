"""Prediction package for crop disease classification."""
from src.prediction.predict import DiseasePredictor, format_class_name
from src.prediction.service import get_predictor

__all__ = ["DiseasePredictor", "format_class_name", "get_predictor"]
