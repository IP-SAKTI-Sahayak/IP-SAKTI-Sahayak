import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Files
INDEX_FILE = Path("vectorstore/index.faiss")
METADATA_FILE = Path("vectorstore/metadata.json")


# Load FAISS index
index = faiss.read_index(str(INDEX_FILE))

# Load chunk information
chunks = json.loads(
    METADATA_FILE.read_text(encoding="utf-8")
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Ask a question
query = input("\nEnter your question: ")

# Convert question into embedding
query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)


# Search top 5 relevant chunks
scores, indices = index.search(query_embedding, 5)


print("\n================================")
print("RETRIEVED DOCUMENTS")
print("================================")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]), 1
):

    chunk = chunks[idx]

    print(f"\n--- Result {rank} ---")
    print("Score:", round(float(score), 4))
    print("Document:", chunk["document"])
    print("Category:", chunk["category"])
    print("Page:", chunk["page"])
    print("Source:", chunk["source"])
    print("Text:")
    print(chunk["text"][:1000])