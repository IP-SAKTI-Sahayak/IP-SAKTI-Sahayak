import json
import argparse
from itertools import chain
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/chunks/chunks.json")
INTERNATIONAL_CHUNKS_DIR = Path("data/chunks_international")
VECTORSTORE = Path("vectorstore")

VECTORSTORE.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
build_modes = parser.add_mutually_exclusive_group()
build_modes.add_argument(
    "--international-only",
    action="store_true",
    help="Build the International index without rebuilding the India index.",
)
build_modes.add_argument(
    "--append-india-abs",
    action="store_true",
    help="Append new India ABS chunks to the existing India index only.",
)
args = parser.parse_args()

# Load chunks
existing_chunks = []
existing_index = None
if args.international_only:
    files = sorted(INTERNATIONAL_CHUNKS_DIR.glob("*.json"))
    chunks = list(chain.from_iterable(
        json.loads(path.read_text(encoding="utf-8")) for path in files
    ))
    index_name = "index_international.faiss"
    metadata_name = "metadata_international.json"
elif args.append_india_abs:
    abs_chunks_file = VECTORSTORE.parent / "data" / "chunks" / "abs_chunks.json"
    if not (VECTORSTORE / "index.faiss").exists() or not (VECTORSTORE / "metadata.json").exists():
        raise SystemExit("The existing India index and metadata are required for an incremental append.")
    existing_index = faiss.read_index(str(VECTORSTORE / "index.faiss"))
    existing_chunks = json.loads((VECTORSTORE / "metadata.json").read_text(encoding="utf-8"))
    candidates = json.loads(abs_chunks_file.read_text(encoding="utf-8"))
    existing_keys = {(row["document"], row["page"], row["text"]) for row in existing_chunks}
    chunks = [
        row for row in candidates
        if row.get("domain") == "abs"
        and (row["document"], row["page"], row["text"]) not in existing_keys
    ]
    if not chunks:
        print("No new India ABS chunks to append; the existing index is unchanged.")
        raise SystemExit(0)
    next_chunk_id = max((row.get("chunk_id", 0) for row in existing_chunks), default=0) + 1
    for offset, row in enumerate(chunks):
        row["chunk_id"] = next_chunk_id + offset
    index_name = "index.faiss"
    metadata_name = "metadata.json"
else:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    index_name = "index.faiss"
    metadata_name = "metadata.json"

if not chunks:
    raise SystemExit("No chunks found for the requested jurisdiction index.")
texts = [chunk["text"] for chunk in chunks]

print("Chunks:", len(texts))
print("Loading embedding model...")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

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
if existing_index is not None:
    existing_vectors = existing_index.reconstruct_n(0, existing_index.ntotal)
    if existing_vectors.shape[1] != dimension:
        raise SystemExit("Existing and new embedding dimensions do not match.")
    index.add(existing_vectors)
index.add(embeddings)

# Save FAISS index
faiss.write_index(
    index,
    str(VECTORSTORE / index_name)
)

# Save chunk information
(VECTORSTORE / metadata_name).write_text(
    json.dumps(existing_chunks + chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print()
print("================================")
print("EMBEDDINGS CREATED SUCCESSFULLY")
print("================================")
print("Embedding shape:", embeddings.shape)
print("FAISS vectors:", index.ntotal)
print("Saved:", VECTORSTORE / index_name)
print("Saved:", VECTORSTORE / metadata_name)
