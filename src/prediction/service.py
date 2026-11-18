"""Central thread-safe DiseasePredictor service for Krishi Sahayyak.

Ensures that exactly ONE MobileNetV3-Large model instance is ever instantiated
and shared across FastAPI routes, LangGraph execution nodes, and background tasks,
preventing duplicate model allocations that exceed Render's 512 MB memory boundary.
"""
from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Optional

from src.prediction.predict import DiseasePredictor
from src.utils.memory import log_memory

logger = logging.getLogger("src.prediction.service")

_predictor_lock = threading.Lock()
_shared_predictor: Optional[DiseasePredictor] = None


def get_predictor(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    confidence_threshold: float = 0.60,
) -> DiseasePredictor:
    """Return the shared DiseasePredictor singleton using thread-safe double-checked locking.
    
    Guarantees that concurrent requests or multiple module imports (FastAPI,
    LangGraph nodes) never load duplicate MobileNetV3 model weights into memory.
    """
    global _shared_predictor
    if _shared_predictor is None:
        with _predictor_lock:
            if _shared_predictor is None:
                log_memory("Before MobileNetV3 Initialization", logger)
                logger.info("Initializing shared DiseasePredictor singleton...")
                _shared_predictor = DiseasePredictor(
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    confidence_threshold=confidence_threshold,
                )
                log_memory("After MobileNetV3 Initialization", logger)
                logger.info(
                    "Shared DiseasePredictor loaded successfully (%d classes) on device '%s'.",
                    len(_shared_predictor.class_names),
                    _shared_predictor.device,
                )
    return _shared_predictor


def is_predictor_loaded() -> bool:
    """Check whether the shared DiseasePredictor singleton has been loaded."""
    return _shared_predictor is not None


def reset_predictor() -> None:
    """Reset the shared predictor singleton (primarily used for unit testing)."""
    global _shared_predictor
    with _predictor_lock:
        _shared_predictor = None
        logger.debug("Shared DiseasePredictor singleton reset.")
