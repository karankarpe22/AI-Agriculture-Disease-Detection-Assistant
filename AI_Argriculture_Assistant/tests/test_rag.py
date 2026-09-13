"""Unit tests for RAG vector index and retrieval."""
from __future__ import annotations

from pathlib import Path
from src.rag.vector_store import AgriculturalRAG, get_rag_service


def test_rag_retrieves_relevant_evidence():
    rag = get_rag_service()
    results = rag.retrieve(
        crop="Tomato",
        disease="Late blight",
        question="What fungicide to spray for blight?",
        top_k=3,
    )

    assert len(results) > 0
    assert "content" in results[0]
    assert "source" in results[0]
    assert "title" in results[0]
    assert "heading" in results[0]
    assert results[0]["similarity_score"] > 0.3


def test_rag_sources_summary():
    rag = get_rag_service()
    results = rag.retrieve(crop="Potato", disease="Early blight", top_k=2)
    sources = rag.get_sources_summary(results)

    assert len(sources) > 0
    assert "source" in sources[0]
    assert "title" in sources[0]
