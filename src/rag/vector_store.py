"""Phase 8: Document-grounded RAG knowledge retriever using SentenceTransformers & FAISS."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import threading
from typing import Any

import numpy as np

from src.utils.memory import log_memory

logger = logging.getLogger("src.rag.vector_store")

# Lazy imports for fast startup
SentenceTransformer = None
faiss = None
_encoder_factory_lock = threading.Lock()
_faiss_import_lock = threading.Lock()
_rag_singleton_lock = threading.Lock()


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global SentenceTransformer
    if SentenceTransformer is None:
        with _encoder_factory_lock:
            if SentenceTransformer is None:
                from sentence_transformers import SentenceTransformer as ST
                SentenceTransformer = ST
    return SentenceTransformer(model_name)


def load_faiss():
    global faiss
    if faiss is None:
        with _faiss_import_lock:
            if faiss is None:
                import faiss as f
                faiss = f
    return faiss


class AgriculturalRAG:
    """Manages indexing and semantic retrieval of curated agricultural documents."""

    def __init__(
        self,
        knowledge_base_dir: str | Path = "knowledge_base",
        index_cache_dir: str | Path = "models",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]

        self.kb_dir = Path(knowledge_base_dir)
        if not self.kb_dir.is_absolute() and not self.kb_dir.exists():
            alt_kb = project_root / knowledge_base_dir
            if alt_kb.exists():
                self.kb_dir = alt_kb

        self.cache_dir = Path(index_cache_dir)
        if not self.cache_dir.is_absolute() and not self.cache_dir.exists():
            alt_cache = project_root / index_cache_dir
            if alt_cache.exists():
                self.cache_dir = alt_cache

        self.model_name = embedding_model_name
        self.encoder = None
        self.index = None
        self.chunks: list[dict[str, Any]] = []

        self._encoder_lock = threading.Lock()
        self._index_lock = threading.Lock()

        self.index_file = self.cache_dir / "rag_index.faiss"
        self.chunks_file = self.cache_dir / "rag_chunks.json"

    def is_ready(self) -> bool:
        """Return True if FAISS index and chunks are loaded and ready in memory."""
        if self.index is not None and len(self.chunks) > 0:
            return True
        return self.load_cached_index()

    def load_cached_index(self) -> bool:
        """Load precomputed FAISS index and chunk metadata from disk if available."""
        if self.index is not None and self.chunks:
            return True
        with self._index_lock:
            if self.index is not None and self.chunks:
                return True
            if self.index_file.exists() and self.chunks_file.exists():
                try:
                    log_memory("Before FAISS Index Load", logger)
                    faiss_module = load_faiss()
                    self.index = faiss_module.read_index(str(self.index_file))
                    self.chunks = json.loads(self.chunks_file.read_text(encoding="utf-8"))
                    log_memory("After FAISS Index Load", logger)
                    logger.info(
                        "Loaded cached FAISS index (%d chunks) from %s",
                        len(self.chunks),
                        self.index_file,
                    )
                    return True
                except Exception as e:
                    logger.warning("Error loading cached index (%s): %s", self.index_file, e)
            return False

    def _ensure_encoder(self):
        if self.encoder is None:
            with self._encoder_lock:
                if self.encoder is None:
                    log_memory("Before SentenceTransformer Load", logger)
                    logger.info("Initializing SentenceTransformer encoder ('%s')...", self.model_name)
                    self.encoder = load_sentence_transformer(self.model_name)
                    log_memory("After SentenceTransformer Load", logger)
                    logger.info("SentenceTransformer encoder ('%s') loaded successfully.", self.model_name)
        return self.encoder

    def parse_markdown_document(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract metadata and chunk content by section headers."""
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        title = file_path.stem.replace("_", " ").title()
        source = "ICAR Agricultural Guide"
        crop = ""
        disease = ""
        reference = ""

        # Extract top metadata lines
        body_start_idx = 0
        for i, line in enumerate(lines[:15]):
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("- **Source**:"):
                source = line.split(":", 1)[1].strip()
            elif line.startswith("- **Crop**:"):
                crop = line.split(":", 1)[1].strip()
            elif line.startswith("- **Disease**:") or line.startswith("- **Topic**:") or line.startswith("- **Pest/Damage**:"):
                disease = line.split(":", 1)[1].strip()
            elif line.startswith("- **Reference**:"):
                reference = line.split(":", 1)[1].strip()
            elif line.startswith("## "):
                body_start_idx = i
                break

        chunks = []
        current_heading = "General"
        current_content: list[str] = []

        for line in lines[body_start_idx:]:
            if line.startswith("## "):
                if current_content:
                    chunk_text = "\n".join(current_content).strip()
                    if len(chunk_text) > 30:
                        chunks.append({
                            "title": title,
                            "source": source,
                            "crop": crop,
                            "disease": disease,
                            "reference": reference,
                            "heading": current_heading,
                            "content": chunk_text,
                            "file": file_path.name,
                        })
                current_heading = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)

        # Append last section
        if current_content:
            chunk_text = "\n".join(current_content).strip()
            if len(chunk_text) > 30:
                chunks.append({
                    "title": title,
                    "source": source,
                    "crop": crop,
                    "disease": disease,
                    "reference": reference,
                    "heading": current_heading,
                    "content": chunk_text,
                    "file": file_path.name,
                })

        return chunks

    def build_index(self, force_rebuild: bool = False) -> None:
        """Parse all markdown files in knowledge_base and build FAISS vector index."""
        if not force_rebuild and self.load_cached_index():
            return

        with self._index_lock:
            if not force_rebuild and self.load_cached_index():
                return

            logger.info("Building agricultural knowledge base index...")
            all_chunks: list[dict[str, Any]] = []
            md_files = list(self.kb_dir.glob("*.md"))
            if not md_files:
                raise FileNotFoundError(f"No markdown documents found in {self.kb_dir}")

            for f in md_files:
                docs = self.parse_markdown_document(f)
                all_chunks.extend(docs)

            logger.info(f"Extracted {len(all_chunks)} sections across {len(md_files)} documents.")

            # Prepare composite text for embedding (incorporates crop & disease context)
            texts_to_embed = [
                f"Crop: {c['crop']}. Disease: {c['disease']}. Topic: {c['heading']}.\n{c['content']}"
                for c in all_chunks
            ]

            encoder = self._ensure_encoder()
            embeddings = encoder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)
            embeddings = embeddings.astype(np.float32)

            dimension = embeddings.shape[1]
            faiss_module = load_faiss()
            # Flat Inner Product on normalized vectors = Cosine Similarity
            self.index = faiss_module.IndexFlatIP(dimension)
            self.index.add(embeddings)
            self.chunks = all_chunks

            # Cache to disk
            self.cache_dir.mkdir(exist_ok=True)
            faiss_module.write_index(self.index, str(self.index_file))
            self.chunks_file.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
            logger.info(f"FAISS index saved to {self.index_file} ({len(self.chunks)} chunks)")

    @staticmethod
    def _build_composite_query(
        crop: str | None = None,
        disease: str | None = None,
        question: str | None = None,
    ) -> str:
        """Formulate composite semantic query from crop, disease context, and farmer question."""
        parts = []
        if crop and crop.strip() and crop.strip().lower() not in ["general", "general / solanaceae"]:
            parts.append(f"Crop: {crop.strip()}")
        if disease and disease.strip() and disease.strip().lower() not in ["general agronomic inquiry", "healthy"]:
            parts.append(f"Disease: {disease.strip()}")
        if question and question.strip():
            parts.append(f"Query: {question.strip()}")
        return ". ".join(parts).strip()

    @staticmethod
    def _generate_chunk_id(chunk: dict[str, Any], index: int) -> str:
        """Generate a stable, unique chunk identifier derived from source file and heading."""
        base_name = Path(chunk.get("file", "chunk")).stem
        heading_slug = re.sub(r"[^a-zA-Z0-9]+", "_", chunk.get("heading", "section")).strip("_").lower()
        return f"ICAR_{base_name}_{heading_slug}_{index:02d}"

    def _score_chunk(
        self,
        chunk: dict,
        crop: str | None,
        disease: str | None,
        question: str | None,
    ) -> float:
        """Keyword-based heuristic scoring (used ONLY as secondary fallback)."""
        score = 0.0
        c_crop = (chunk.get("crop") or "").lower()
        c_dis = (chunk.get("disease") or "").lower()
        c_text = ((chunk.get("heading") or "") + " " + (chunk.get("content") or "")).lower()

        if crop and crop.lower() in c_crop:
            score += 6.0
        if disease:
            for w in disease.lower().split():
                if len(w) > 2 and w in c_dis:
                    score += 5.0
        if question and question.strip():
            q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]
            for w in q_words:
                if w in c_text:
                    score += 1.5

        h = (chunk.get("heading") or "").lower()
        if any(kw in h for kw in ["management", "treatment", "control", "fungicide"]):
            score += 3.0
        elif "symptom" in h:
            score += 1.5
        elif "condition" in h or "environment" in h:
            score += 1.0

        return score

    def retrieve(
        self,
        crop: str | None = None,
        disease: str | None = None,
        question: str | None = None,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant agricultural evidence chunks using FAISS vector search with keyword fallback.
        
        Deterministic Retrieval Flow:
        1. Formulate composite query text from crop, disease, and farmer question.
        2. If query is empty, return empty list (no evidence).
        3. Attempt primary vector similarity search via SentenceTransformer + FAISS index.search().
        4. If vector search succeeds:
           - Filter candidates by min_similarity threshold (default 0.35).
           - Deduplicate sections using source file and heading.
           - Tag each chunk with chunk_id, similarity_score, and retrieval_method='vector'.
           - Return top_k unique chunks.
        5. If FAISS index or encoder is unavailable, or search raises an exception:
           - Log reason for fallback.
           - Execute secondary keyword heuristic fallback via _score_chunk().
           - Tag returned chunks with retrieval_method='keyword_fallback'.
           - Return top_k unique chunks.
        """
        # Ensure index and chunks are loaded
        if not self.chunks:
            try:
                self.build_index()
            except Exception as e:
                logger.warning(f"Could not load or build FAISS index: {e}")

        query_text = self._build_composite_query(crop, disease, question)
        if not query_text:
            logger.debug("Empty query provided to RAG retrieve(); returning empty evidence list.")
            return []

        results: list[dict[str, Any]] = []
        vector_search_attempted = False
        vector_search_succeeded = False

        # --- Primary Path: Vector Similarity Search via FAISS ---
        if self.index is not None and self.chunks:
            vector_search_attempted = True
            try:
                encoder = self._ensure_encoder()
                if encoder is None:
                    raise RuntimeError("SentenceTransformer encoder could not be initialized.")

                query_vec = encoder.encode(
                    [query_text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                query_vec = query_vec.astype(np.float32)

                # Search candidate pool larger than top_k to allow heading deduplication
                search_k = min(max(top_k * 4, 10), len(self.chunks))
                distances, indices = self.index.search(query_vec, search_k)

                seen_headings: set[str] = set()
                for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    score = float(dist)
                    if score < min_similarity:
                        continue

                    raw_chunk = self.chunks[idx]
                    h_key = f"{raw_chunk.get('file', '')}:{raw_chunk.get('heading', '')}"
                    if h_key in seen_headings:
                        continue
                    seen_headings.add(h_key)

                    chunk = raw_chunk.copy()
                    chunk["chunk_id"] = self._generate_chunk_id(chunk, idx)
                    chunk["similarity_score"] = round(score, 4)
                    chunk["retrieval_method"] = "vector"
                    results.append(chunk)

                    if len(results) >= top_k:
                        break

                vector_search_succeeded = True
            except Exception as e:
                logger.warning(f"FAISS vector search failed ({e}); falling back to keyword heuristics.")
                vector_search_succeeded = False

        # --- Secondary Path: Keyword Heuristic Fallback ---
        # Triggered ONLY when vector search could not run or raised an exception
        if not vector_search_succeeded:
            fallback_reason = "FAISS index or encoder unavailable" if not vector_search_attempted else "Vector search execution failed"
            logger.warning(f"Executing keyword fallback ({fallback_reason}).")

            if not self.chunks:
                try:
                    self.build_index()
                except Exception as e:
                    logger.error(f"Cannot load chunks for keyword fallback: {e}")
                    return []

            if self.chunks:
                scored = []
                for idx, chunk in enumerate(self.chunks):
                    s = self._score_chunk(chunk, crop, disease, question)
                    scored.append((s, idx))

                scored.sort(key=lambda x: x[0], reverse=True)

                seen_headings = set()
                for s, idx in scored:
                    if s <= 0.0:
                        continue
                    raw_chunk = self.chunks[idx]
                    h_key = f"{raw_chunk.get('file', '')}:{raw_chunk.get('heading', '')}"
                    if h_key in seen_headings:
                        continue
                    seen_headings.add(h_key)

                    chunk = raw_chunk.copy()
                    chunk["chunk_id"] = self._generate_chunk_id(chunk, idx)
                    chunk["similarity_score"] = round(float(s), 4)
                    chunk["retrieval_method"] = "keyword_fallback"
                    results.append(chunk)

                    if len(results) >= top_k:
                        break

        return results[:top_k]

    @staticmethod
    def get_sources_summary(evidence_list: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Generate deduplicated list of source citations for farmer display."""
        sources = []
        seen_sources = set()
        for item in evidence_list:
            source_name = item.get("source", "Agricultural Extension Authority")
            title = item.get("title", "Crop Advisory")
            key = f"{source_name} - {title}"
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({
                    "title": title,
                    "source": source_name,
                    "crop": item.get("crop", ""),
                    "reference": item.get("reference", ""),
                })
        return sources


_default_rag = None


def get_rag_service() -> AgriculturalRAG:
    """Singleton getter for the Agricultural RAG service with thread-safe double-checked locking."""
    global _default_rag
    if _default_rag is None:
        with _rag_singleton_lock:
            if _default_rag is None:
                _default_rag = AgriculturalRAG()
    return _default_rag


def reset_rag_service() -> None:
    """Reset the RAG singleton (primarily used for unit testing)."""
    global _default_rag
    with _rag_singleton_lock:
        _default_rag = None
        logger.debug("RAG service singleton reset.")
