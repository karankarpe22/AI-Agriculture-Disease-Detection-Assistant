"""Phase 8: Document-grounded RAG knowledge retriever using SentenceTransformers & FAISS."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports for fast startup
SentenceTransformer = None
faiss = None


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer(model_name)


def load_faiss():
    global faiss
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
        self.kb_dir = Path(knowledge_base_dir)
        self.cache_dir = Path(index_cache_dir)
        self.model_name = embedding_model_name
        self.encoder = None
        self.index = None
        self.chunks: list[dict[str, Any]] = []

        self.index_file = self.cache_dir / "rag_index.faiss"
        self.chunks_file = self.cache_dir / "rag_chunks.json"

    def _ensure_encoder(self):
        if self.encoder is None:
            self.encoder = load_sentence_transformer(self.model_name)
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
        if not force_rebuild and self.index_file.exists() and self.chunks_file.exists():
            try:
                faiss_module = load_faiss()
                self.index = faiss_module.read_index(str(self.index_file))
                self.chunks = json.loads(self.chunks_file.read_text(encoding="utf-8"))
                print(f"Loaded existing FAISS index with {len(self.chunks)} chunks.")
                return
            except Exception as e:
                print(f"Error loading cached index ({e}), rebuilding...")

        print("Building agricultural knowledge base index...")
        all_chunks: list[dict[str, Any]] = []
        md_files = list(self.kb_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown documents found in {self.kb_dir}")

        for f in md_files:
            docs = self.parse_markdown_document(f)
            all_chunks.extend(docs)

        print(f"Extracted {len(all_chunks)} sections across {len(md_files)} documents.")

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
        print(f"FAISS index saved to {self.index_file} ({len(self.chunks)} chunks)")

    def _score_chunk(
        self,
        chunk: dict,
        crop: str | None,
        disease: str | None,
        question: str | None,
    ) -> float:
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
    ) -> list[dict[str, Any]]:
        """Retrieve relevant agricultural evidence chunks based on disease and question."""
        if not self.chunks:
            self.build_index()

        if self.chunks:
            scored = []
            for idx, chunk in enumerate(self.chunks):
                s = self._score_chunk(chunk, crop, disease, question)
                scored.append((s, idx))

            scored.sort(key=lambda x: x[0], reverse=True)

            results = []
            seen_headings = set()
            for s, idx in scored:
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = round(float(s), 4)
                h_key = f"{chunk.get('file')}:{chunk.get('heading')}"
                if h_key not in seen_headings:
                    seen_headings.add(h_key)
                    results.append(chunk)
                if len(results) >= top_k:
                    break

            if results:
                return results[:top_k]

        return []

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
    """Singleton getter for the Agricultural RAG service."""
    global _default_rag
    if _default_rag is None:
        _default_rag = AgriculturalRAG()
    return _default_rag
