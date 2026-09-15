import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/chunks/chunks.json")
VECTORSTORE = Path("vectorstore")

VECTORSTORE.mkdir(exist_ok=True)

# Load chunks
chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
texts = [chunk["text"] for chunk in chunks]

print("Chunks:", len(texts))
print("Loading embedding model...")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(
    index,
    str(VECTORSTORE / "index.faiss")
)

# Save chunk information
(VECTORSTORE / "metadata.json").write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print()
print("================================")
print("EMBEDDINGS CREATED SUCCESSFULLY")
print("================================")
print("Embedding shape:", embeddings.shape)
print("FAISS vectors:", index.ntotal)
print("Saved:", VECTORSTORE / "index.faiss")
print("Saved:", VECTORSTORE / "metadata.json")