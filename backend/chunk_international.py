from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "extracted_international"
OUTPUT_DIR = BASE_DIR / "data" / "chunks_international"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DOCUMENT_METADATA = {
    "PCT_Treaty": {
        "document": "PCT Treaty",
        "category": "Patent",
        "source": "WIPO"
    },
    "PCT_Regulations_2026": {
        "document": "PCT Regulations, 2026",
        "category": "Patent",
        "source": "WIPO"
    },
    "Madrid_Protocol_Regulations_2025": {
        "document": "Madrid Protocol and Regulations, 2025",
        "category": "Trademark",
        "source": "WIPO"
    },
    "Hague_Regulations_2026": {
        "document": "Hague Regulations, 2026",
        "category": "Design",
        "source": "WIPO"
    },
    "Budapest_Treaty_1980": {
        "document": "Budapest Treaty, 1980",
        "category": "Patent",
        "source": "WIPO"
    },
    "Budapest_Regulations_2023": {
        "document": "Budapest Regulations, 2023",
        "category": "Patent",
        "source": "WIPO"
    },
    "WIPO_GRATK_Treaty_2024": {
        "document": "WIPO GRATK Treaty, 2024",
        "category": "Traditional Knowledge",
        "source": "WIPO"
    },
    "TRIPS_Agreement": {
        "document": "TRIPS Agreement",
        "category": "Intellectual Property",
        "source": "WTO"
    },
    "Convention_on_Biological_Diversity": {
        "document": "Convention on Biological Diversity",
        "category": "Biodiversity",
        "source": "CBD"
    },
    "Nagoya_Protocol": {
        "document": "Nagoya Protocol",
        "category": "Access and Benefit Sharing",
        "source": "CBD"
    }
}


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_pages(text):
    parts = re.split(r"--- PAGE (\d+) ---", text)

    pages = []

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = clean_text(parts[i + 1])

        if page_text:
            pages.append((page_number, page_text))

    return pages


def create_chunks(page_text):
    chunks = []

    start = 0
    length = len(page_text)

    while start < length:

        end = min(start + CHUNK_SIZE, length)

        if end < length:
            boundary = page_text.rfind("\n", start, end)

            if boundary > start + 400:
                end = boundary

        chunk = page_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


for txt_file in SOURCE_DIR.glob("*.txt"):

    metadata = DOCUMENT_METADATA.get(txt_file.stem)

    if metadata is None:
        print(f"[SKIPPED] No metadata: {txt_file.name}")
        continue

    text = txt_file.read_text(encoding="utf-8")

    pages = split_pages(text)

    records = []
    chunk_id = 1

    for page_number, page_text in pages:

        page_chunks = create_chunks(page_text)

        for chunk in page_chunks:

            records.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "document": metadata["document"],
                "jurisdiction": "International",
                "category": metadata["category"],
                "source": metadata["source"],
                "page": page_number
            })

            chunk_id += 1

    output_file = OUTPUT_DIR / f"{txt_file.stem}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] {txt_file.name} -> {len(records)} chunks")

print("\nInternational chunks created in India-compatible format.")