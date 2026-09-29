"""Incrementally ingest curated International PDFs into the FAISS corpus.

Examples:
    python -m backend.ingest_international --list
    python -m backend.ingest_international --all
    python -m backend.ingest_international --pdf data/international/biodiversity/Nagoya_Protocol.pdf
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent.parent
PDF_ROOT = ROOT / "data" / "international"
CHUNK_ROOT = ROOT / "data" / "chunks_international"
INDEX_PATH = ROOT / "vectorstore" / "index_international.faiss"
METADATA_PATH = ROOT / "vectorstore" / "metadata_international.json"
REGISTRY_PATH = ROOT / "data" / "international_sources.json"
FRONTEND_REGISTRY_PATH = ROOT / "frontend" / "src" / "international_sources.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

DOCUMENTS = {
    "Convention_on_Biological_Diversity": ("Convention on Biological Diversity", "Biodiversity", "CBD Secretariat"),
    "Nagoya_Protocol": ("Nagoya Protocol", "Access and Benefit Sharing", "CBD Secretariat"),
    "WIPO_GRATK_Treaty_2024": ("WIPO GRATK Treaty, 2024", "Traditional Knowledge", "WIPO"),
    "Hague_Regulations_2026": ("Hague Regulations, 2026", "Design", "WIPO"),
    "Madrid_Protocol_Regulations_2025": ("Madrid Protocol and Regulations, 2025", "Trademark", "WIPO"),
    "PCT_Treaty": ("PCT Treaty", "Patent", "WIPO"),
    "PCT_Regulations_2026": ("PCT Regulations, 2026", "Patent", "WIPO"),
    "Budapest_Treaty_1980": ("Budapest Treaty, 1980", "Patent", "WIPO"),
    "Budapest_Regulations_2023": ("Budapest Regulations, 2023", "Patent", "WIPO"),
    "TRIPS_Agreement": ("TRIPS Agreement", "Intellectual Property", "WTO"),
}


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def make_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        boundary = text.rfind("\n", start, end)
        if end < len(text) and boundary > start + 400:
            end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def read_pdf(pdf_path):
    records = []
    extracted_pages = []
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text())
            extracted_pages.append(f"--- PAGE {page_number} ---\n{text}\n")
            for chunk in make_chunks(text):
                records.append({"page": page_number, "text": chunk})
    extracted_path = ROOT / "data" / "extracted_international" / f"{pdf_path.stem}.txt"
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.write_text("\n".join(extracted_pages), encoding="utf-8")
    return records


def metadata_record(stem):
    title, category, authority = DOCUMENTS[stem]
    return {
        "document": title,
        "jurisdiction": "International",
        "category": category,
        "authority": authority,
        "source": authority,
        "source_organization": authority,
        "version": "TODO",
        "document_version": "TODO",
        "source_url": None,
        "official_listing_url": None,
        "official_pdf_url": None,
    }


def records_for(stem, extracted):
    source = metadata_record(stem)
    return [
        {
            **source,
            "chunk_id": offset + 1,
            "text": item["text"],
            "page": item["page"],
        }
        for offset, item in enumerate(extracted)
    ]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def list_documents():
    rows = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else []
    indexed = {row.get("document") for row in rows}
    for stem, (title, _, authority) in DOCUMENTS.items():
        candidates = list(PDF_ROOT.rglob(stem + ".pdf"))
        count = sum(row.get("document") == title for row in rows)
        print(f"{'INDEXED' if title in indexed else 'NOT INDEXED':12} {title} | authority={authority} | chunks={count} | pdf={'found' if candidates else 'missing'} | version=TODO | source_url=TODO")


def ingest_records(pdf_path, generated, index, rows, embedder):
    stem = pdf_path.stem
    if stem not in DOCUMENTS:
        raise ValueError(f"No curated source metadata configured for PDF stem {stem!r}.")
    if not generated:
        raise ValueError(f"No extractable text found in {pdf_path.name}.")

    title = DOCUMENTS[stem][0]
    existing_by_key = {(row.get("document"), row.get("page"), row.get("text")): row for row in rows}
    pending = []
    for item in generated:
        key = (title, item["page"], item["text"])
        current = existing_by_key.get(key)
        if current is None:
            pending.append(item)
        else:
            chunk_id = current.get("chunk_id")
            current.update(item)
            current["chunk_id"] = chunk_id

    if pending:
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(MODEL_NAME, local_files_only=True)
        vectors = embedder.encode(
            [row["text"] for row in pending],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if vectors.shape[1] != index.d:
            raise ValueError(f"Embedding dimension {vectors.shape[1]} does not match FAISS index dimension {index.d}.")
        for row in pending:
            row["chunk_id"] = max((item.get("chunk_id", 0) for item in rows), default=0) + 1
            rows.append(row)
        index.add(vectors)
    print(f"[OK] {pdf_path.name}: {len(generated)} chunks, {len(pending)} newly embedded; index now has {index.ntotal} vectors.")
    return embedder


def main():
    parser = argparse.ArgumentParser(description="Ingest official International PDFs into the International FAISS index.")
    parser.add_argument("--pdf", type=Path, help="Ingest one configured PDF file.")
    parser.add_argument("--all", action="store_true", help="Ingest all configured PDFs found under data/international/.")
    parser.add_argument("--list", action="store_true", help="List configured sources and current index status.")
    args = parser.parse_args()

    if args.list:
        list_documents()
        return
    if bool(args.pdf) == bool(args.all):
        parser.error("Choose exactly one of --pdf PATH, --all, or --list.")

    if args.all:
        selected = []
        for stem in DOCUMENTS:
            matches = list(PDF_ROOT.rglob(stem + ".pdf"))
            if not matches:
                print(f"[MISSING] {stem}.pdf")
            selected.extend(matches)
    else:
        pdf = args.pdf if args.pdf.is_absolute() else ROOT / args.pdf
        selected = [pdf.resolve()]
    if not selected:
        raise SystemExit("No configured PDFs found.")
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise SystemExit("The existing International FAISS index and metadata are required; rebuild it with the existing --international-only command first.")

    import faiss

    index = faiss.read_index(str(INDEX_PATH))
    rows = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if index.ntotal != len(rows):
        raise SystemExit(f"International FAISS/metadata mismatch: {index.ntotal} vectors vs {len(rows)} metadata rows.")
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    embedder = None
    for pdf_path in selected:
        if not pdf_path.is_file():
            raise SystemExit(f"PDF not found: {pdf_path}")
        chunks = records_for(pdf_path.stem, read_pdf(pdf_path))
        write_json(CHUNK_ROOT / f"{pdf_path.stem}.json", chunks)
        embedder = ingest_records(pdf_path, chunks, index, rows, embedder)

    VECTORSTORE = ROOT / "vectorstore"
    temporary_index = VECTORSTORE / "index_international.faiss.tmp"
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(INDEX_PATH)
    write_json(METADATA_PATH, rows)
    manifest = [metadata_record(stem) for stem in DOCUMENTS if any(row.get("document") == DOCUMENTS[stem][0] for row in rows)]
    write_json(REGISTRY_PATH, manifest)
    write_json(FRONTEND_REGISTRY_PATH, manifest)
    print(f"Saved International FAISS index ({index.ntotal} vectors) and {len(manifest)} source records.")


if __name__ == "__main__":
    main()
