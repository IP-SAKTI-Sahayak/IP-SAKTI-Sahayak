"""Optional Qdrant storage adapter for the existing chunk and embedding pipeline.

FAISS remains the default runtime store. Set VECTOR_STORE_BACKEND=qdrant only
after indexing a corpus with ``python -m backend.qdrant_store``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


COLLECTION = "ip_sakti_chunks"
VECTOR_SIZE = 384
DOMAIN_BY_CATEGORY = {
    "Patent": "patent", "Ayush Patent Guidelines": "patent",
    "Biodiversity": "abs", "Access and Benefit Sharing": "abs",
    "Traditional Knowledge": "tkdl_prior_art", "Trademark": "trademark",
    "Design": "design", "Intellectual Property": "patent",
}
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("The optional Qdrant backend requires qdrant-client; install backend/requirements-qdrant.txt.") from exc
    url = os.getenv("QDRANT_URL")
    if url:
        _CLIENT = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)
        return _CLIENT
    path = Path(os.getenv("QDRANT_PATH", "vectorstore/qdrant")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _CLIENT = QdrantClient(path=str(path))
    return _CLIENT


def index_chunks(chunks: list[dict], embeddings) -> int:
    """Upsert existing curated chunks/vectors with jurisdiction in each payload."""
    from qdrant_client.http import models

    client = _client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    points = []
    for position, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        jurisdiction = str(chunk.get("jurisdiction", "India")).title()
        identity = "|".join((jurisdiction, str(chunk.get("document", "")), str(chunk.get("page", "")), str(chunk.get("chunk_id", position))))
        payload = dict(chunk)
        payload["jurisdiction"] = jurisdiction
        payload["domain"] = chunk.get("domain") or DOMAIN_BY_CATEGORY.get(chunk.get("category"), "unknown")
        points.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)), vector=vector.tolist(), payload=payload))
    for start in range(0, len(points), 256):
        client.upsert(collection_name=COLLECTION, points=points[start:start + 256], wait=True)
    return len(points)


def search_chunks(vector, jurisdiction: str, limit=5, allowed_categories=None, allowed_domains=None):
    """Retrieve only points whose jurisdiction matches the selected toggle."""
    from qdrant_client.http import models

    client = _client()
    if not client.collection_exists(COLLECTION):
        return []
    must = [models.FieldCondition(key="jurisdiction", match=models.MatchValue(value=jurisdiction.title()))]
    if allowed_categories:
        must.append(models.FieldCondition(key="category", match=models.MatchAny(any=list(allowed_categories))))
    if allowed_domains:
        must.append(models.FieldCondition(key="domain", match=models.MatchAny(any=list(allowed_domains))))
    response = client.query_points(
        collection_name=COLLECTION, query=vector.tolist(), query_filter=models.Filter(must=must),
        limit=limit, with_payload=True,
    )
    results = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload["score"] = float(point.score)
        results.append(payload)
    return results


def build_existing_corpus_index():
    """Reuse the current chunk metadata and embedding model to populate Qdrant."""
    from backend.rag import chunks, international_chunks, model

    all_chunks = chunks + international_chunks
    vectors = model.encode([chunk["text"] for chunk in all_chunks], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    count = index_chunks(all_chunks, vectors)
    print(f"Indexed {count} existing chunks in Qdrant.")


if __name__ == "__main__":
    build_existing_corpus_index()
