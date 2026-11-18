"""Unit and integration tests for memory optimization and single-instance management.

Tests verify:
1. Single shared DiseasePredictor instance reused across backend.main and src.graph.nodes.
2. Concurrent thread-safety for predictor initialization.
3. RAG cached index loading without loading SentenceTransformer.
4. Thread-safe lazy loading of SentenceTransformer encoder.
5. RAG is_ready() reflecting actual memory state.
6. Process memory diagnostics functionality and non-negative reporting.
"""
from __future__ import annotations

import concurrent.futures
import threading
import pytest

from src.prediction.service import get_predictor as get_shared_predictor, reset_predictor, is_predictor_loaded
from backend.main import get_predictor as get_main_predictor
from src.graph.nodes import get_predictor as get_nodes_predictor
from src.rag.vector_store import AgriculturalRAG, get_rag_service, reset_rag_service
from src.utils.memory import get_process_memory_mb, get_peak_memory_mb, log_memory


def test_shared_predictor_identity_across_modules():
    """Verify backend.main and src.graph.nodes access the EXACT same DiseasePredictor singleton."""
    p1 = get_shared_predictor()
    p2 = get_main_predictor()
    p3 = get_nodes_predictor()

    assert p1 is p2, "backend.main must use the shared DiseasePredictor singleton"
    assert p2 is p3, "src.graph.nodes must use the shared DiseasePredictor singleton"
    assert is_predictor_loaded() is True


def test_concurrent_predictor_initialization():
    """Verify that concurrent threads calling get_predictor() receive the same single instance."""
    # Reset predictor to test cold-start concurrent initialization
    reset_predictor()
    assert is_predictor_loaded() is False

    results = []

    def fetch_predictor():
        return get_shared_predictor()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_predictor) for _ in range(16)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert len(results) == 16
    first = results[0]
    for p in results[1:]:
        assert p is first, "Every thread must receive the exact same singleton instance"


def test_rag_cached_index_loading_without_encoder():
    """Verify that load_cached_index() loads FAISS index & chunks without initializing SentenceTransformer."""
    rag = AgriculturalRAG()
    assert rag.encoder is None
    assert rag.index is None
    assert len(rag.chunks) == 0

    success = rag.load_cached_index()
    assert success is True, "Must successfully load cached models/rag_index.faiss and rag_chunks.json"
    assert rag.index is not None, "FAISS index must be loaded"
    assert len(rag.chunks) == 62, "Must load 62 curated research chunks"
    # CRITICAL: SentenceTransformer encoder must remain UNLOADED to conserve RAM
    assert rag.encoder is None, "SentenceTransformer must NOT be loaded when reading cached FAISS index"
    assert rag.is_ready() is True


def test_rag_threadsafe_encoder_lazy_initialization():
    """Verify that multiple concurrent threads calling _ensure_encoder() initialize it safely once."""
    rag = AgriculturalRAG()
    rag.load_cached_index()
    assert rag.encoder is None

    encoders = []

    def fetch_encoder():
        return rag._ensure_encoder()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_encoder) for _ in range(8)]
        for f in concurrent.futures.as_completed(futures):
            encoders.append(f.result())

    assert len(encoders) == 8
    first_enc = encoders[0]
    for enc in encoders[1:]:
        assert enc is first_enc, "All threads must share the exact same encoder instance"


def test_rag_is_ready_reflects_actual_state():
    """Verify is_ready() accurately reflects loaded resources rather than blind directory existence."""
    rag = AgriculturalRAG()
    # Before loading, is_ready() will attempt to load cached index and return True if successful
    assert rag.is_ready() is True
    assert rag.index is not None
    assert len(rag.chunks) > 0

    # Simulate missing index
    rag.index = None
    rag.chunks = []
    # If files exist, is_ready() recovers by loading them
    assert rag.is_ready() is True


def test_memory_diagnostics_utility():
    """Verify memory utility reports positive float values and does not raise exceptions."""
    current_mb = get_process_memory_mb()
    peak_mb = get_peak_memory_mb()

    assert isinstance(current_mb, float)
    assert isinstance(peak_mb, float)
    assert current_mb > 0.0, "Current process memory must be greater than zero"
    assert peak_mb >= current_mb, "Peak memory must be greater than or equal to current RSS"

    logged = log_memory("Unit Test Diagnostic")
    assert logged == current_mb
