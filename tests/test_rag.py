"""Unit tests for RAG vector index and retrieval with genuine FAISS search and keyword fallback."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.rag.vector_store import AgriculturalRAG, get_rag_service
from src.gemini.guidance_service import get_gemini_service


# 1. Actual FAISS index.search() execution
def test_rag_vector_search_execution():
    """Verify that retrieval executes genuine FAISS index.search() and returns vector results."""
    rag = get_rag_service()
    rag.build_index()
    assert rag.index is not None, "FAISS index must be loaded"

    results = rag.retrieve(
        crop="Tomato",
        disease="Late blight",
        question="What fungicide to spray for blight?",
        top_k=3,
    )

    assert len(results) > 0
    assert results[0]["retrieval_method"] == "vector"
    assert results[0]["similarity_score"] > 0.4
    assert "chunk_id" in results[0]


# 2. Semantic relevance of retrieved chunks
def test_rag_semantic_relevance():
    """Verify that semantic vector search understands natural language intent."""
    rag = get_rag_service()
    results = rag.retrieve(
        crop="Bell pepper",
        disease="Bacterial spot",
        question="What are organic treatments and biological control?",
        top_k=3,
    )

    assert len(results) > 0
    # The organic inquiry should retrieve biological or cultural management chunks
    headings = [r["heading"].lower() for r in results]
    contents = " ".join(r["content"].lower() for r in results)
    assert any("biological" in h or "cultural" in h or "management" in h for h in headings)
    assert "pseudomonas" in contents or "bacillus" in contents or "organic" in contents or "rotation" in contents
    assert results[0]["similarity_score"] >= 0.5


# 3. Correct mapping of FAISS indices to chunks
def test_rag_correct_mapping_of_indices_to_chunks():
    """Verify that the integer indices returned by FAISS map directly to self.chunks."""
    rag = get_rag_service()
    rag.build_index()
    encoder = rag._ensure_encoder()

    query = "Crop: Tomato. Disease: Late blight. Query: What fungicide to spray for blight?"
    q_vec = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = rag.index.search(q_vec.astype("float32"), 3)

    results = rag.retrieve(
        crop="Tomato",
        disease="Late blight",
        question="What fungicide to spray for blight?",
        top_k=3,
    )

    # Top retrieved chunk must match the top FAISS index chunk
    top_faiss_idx = indices[0][0]
    expected_chunk = rag.chunks[top_faiss_idx]
    assert results[0]["content"] == expected_chunk["content"]
    assert results[0]["heading"] == expected_chunk["heading"]
    assert results[0]["file"] == expected_chunk["file"]


# 4. Similarity score presence and valid numeric values
def test_rag_similarity_score_validity():
    """Verify similarity scores are bounded floats between 0.0 and 1.0 (Cosine Similarity)."""
    rag = get_rag_service()
    results = rag.retrieve(crop="Potato", disease="Early blight", top_k=3)

    assert len(results) > 0
    for r in results:
        assert isinstance(r["similarity_score"], float)
        assert 0.0 <= r["similarity_score"] <= 1.0
        # Cosine similarity for relevant agricultural queries is typically above 0.5
        assert r["similarity_score"] >= 0.35


# 5. Metadata preservation
def test_rag_metadata_preservation():
    """Verify all original metadata fields are preserved in retrieved evidence chunks."""
    rag = get_rag_service()
    results = rag.retrieve(crop="Tomato", disease="Late blight", top_k=2)

    assert len(results) > 0
    required_fields = [
        "title",
        "source",
        "crop",
        "disease",
        "reference",
        "heading",
        "content",
        "file",
        "chunk_id",
        "similarity_score",
        "retrieval_method",
    ]
    for r in results:
        for field in required_fields:
            assert field in r, f"Missing required metadata field: {field}"
            assert r[field], f"Field {field} must not be empty"


# 6. Stable chunk IDs
def test_rag_stable_chunk_ids():
    """Verify that chunk IDs are stable, unique, and deterministic across multiple calls."""
    rag = get_rag_service()
    run1 = rag.retrieve(crop="Tomato", disease="Late blight", top_k=3)
    run2 = rag.retrieve(crop="Tomato", disease="Late blight", top_k=3)

    ids1 = [r["chunk_id"] for r in run1]
    ids2 = [r["chunk_id"] for r in run2]

    assert ids1 == ids2
    assert len(ids1) == len(set(ids1)), "Chunk IDs within a result list must be unique"
    for cid in ids1:
        assert cid.startswith("ICAR_")


# 7. Heading deduplication
def test_rag_heading_deduplication():
    """Verify that repeated section headings from the same file are deduplicated."""
    rag = get_rag_service()
    results = rag.retrieve(crop="Tomato", disease="Late blight", top_k=5)

    seen_keys = set()
    for r in results:
        key = f"{r['file']}:{r['heading']}"
        assert key not in seen_keys, f"Duplicate section found: {key}"
        seen_keys.add(key)


# 8. Top-K limit
def test_rag_top_k_limit():
    """Verify retrieval strictly respects the requested top_k parameter."""
    rag = get_rag_service()
    res1 = rag.retrieve(crop="Tomato", disease="Late blight", top_k=1)
    res2 = rag.retrieve(crop="Tomato", disease="Late blight", top_k=2)
    res3 = rag.retrieve(crop="Tomato", disease="Late blight", top_k=3)

    assert len(res1) <= 1
    assert len(res2) <= 2
    assert len(res3) <= 3


# 9. Keyword fallback on index failure
def test_rag_keyword_fallback_on_index_failure():
    """Verify graceful fallback to keyword heuristics when FAISS index search fails."""
    rag = AgriculturalRAG()
    rag.build_index()

    # Simulate FAISS index being unavailable
    rag.index = None

    results = rag.retrieve(
        crop="Tomato",
        disease="Late blight",
        question="What fungicide to spray for blight?",
        top_k=2,
    )

    assert len(results) > 0
    assert results[0]["retrieval_method"] == "keyword_fallback"
    assert "chunk_id" in results[0]
    assert results[0]["similarity_score"] > 0.0


# 10. Keyword fallback on encoder failure
def test_rag_keyword_fallback_on_encoder_failure():
    """Verify graceful fallback to keyword heuristics when SentenceTransformer encoding fails."""
    rag = AgriculturalRAG()
    rag.build_index()

    with patch.object(rag, "_ensure_encoder", side_effect=RuntimeError("CUDA / Model Load Error")):
        results = rag.retrieve(
            crop="Potato",
            disease="Early blight",
            question="How to control early blight?",
            top_k=2,
        )

    assert len(results) > 0
    assert results[0]["retrieval_method"] == "keyword_fallback"


# 11. Empty query handling
def test_rag_empty_query_handling():
    """Verify that completely empty or None queries safely return an empty result list."""
    rag = get_rag_service()

    res_none = rag.retrieve(crop=None, disease=None, question=None)
    assert res_none == []

    res_empty = rag.retrieve(crop="", disease="", question="   ")
    assert res_empty == []


# 12. Irrelevant query handling
def test_rag_irrelevant_query_handling():
    """Verify that completely unrelated queries do not match agricultural chunks."""
    rag = get_rag_service()
    # Completely off-topic question without crop/disease
    results = rag.retrieve(
        crop=None,
        disease=None,
        question="What is the capital of France and European history?",
        min_similarity=0.35,
    )

    # Must return empty list, not unrelated agricultural chunks
    assert results == []


# 13. Source summary compatibility
def test_rag_sources_summary_compatibility():
    """Verify get_sources_summary produces clean citations for API and frontend display."""
    rag = get_rag_service()
    results = rag.retrieve(crop="Potato", disease="Early blight", top_k=2)
    sources = rag.get_sources_summary(results)

    assert len(sources) > 0
    for s in sources:
        assert "source" in s
        assert "title" in s
        assert "crop" in s
        assert "reference" in s


# 14. Gemini evidence compatibility
def test_rag_gemini_evidence_compatibility():
    """Verify vector-retrieved evidence integrates seamlessly into the Gemini guidance service."""
    rag = get_rag_service()
    evidence = rag.retrieve(
        crop="Tomato",
        disease="Late blight",
        question="What fungicide to spray?",
        top_k=2,
    )

    gemini = get_gemini_service()
    guidance = gemini.generate_guidance(
        crop="Tomato",
        disease="Late blight",
        confidence=0.92,
        weather={"location": "Pune", "temperature_c": 22.0, "humidity_percentage": 85, "condition": "Overcast"},
        retrieved_evidence=evidence,
        farmer_question="What fungicide to spray?",
        language="english",
    )

    assert "direct_answer" in guidance
    assert "en" in guidance
    assert "mr" in guidance
    assert guidance["en"]["management_guidance"]
