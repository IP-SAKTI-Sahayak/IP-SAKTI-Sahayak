import json
import re
import argparse
from pathlib import Path

from backend.source_registry import trusted_source_for


# Input and output folders
INPUT_FOLDER = Path("data/extracted")
OUTPUT_FOLDER = Path("data/chunks")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Document information
DOCUMENTS = [
    {
        "file": "Patents_Act_1970.txt",
        "document": "Patents Act, 1970",
        "jurisdiction": "India",
        "category": "Patent",
        "source": "IP India",
        "official_listing_url": trusted_source_for("Patents Act, 1970").get("official_listing_url"),
        "official_pdf_url": trusted_source_for("Patents Act, 1970").get("official_pdf_url"),
    },
    {
        "file": "Patents_Rules_2003.txt",
        "document": "Patents Rules, 2003",
        "jurisdiction": "India",
        "category": "Patent",
        "source": "IP India",
        "official_listing_url": trusted_source_for("Patents Rules, 2003").get("official_listing_url"),
        "official_pdf_url": trusted_source_for("Patents Rules, 2003").get("official_pdf_url"),
    },
    {
        "file": "Guidelines_Ayush_Related_Inventions_2025.txt",
        "document": "Guidelines for Examination of Ayush Related Inventions, 2025",
        "jurisdiction": "India",
        "category": "Ayush Patent Guidelines",
        "source": "IP India",
        "official_listing_url": trusted_source_for(
            "Guidelines for Examination of Ayush Related Inventions, 2025"
        ).get("official_listing_url"),
        "official_pdf_url": trusted_source_for(
            "Guidelines for Examination of Ayush Related Inventions, 2025"
        ).get("official_pdf_url"),
    },
    {
        "file": "Biological_Diversity_Act_2002_amended_2023.txt",
        "document": "Biological Diversity Act, 2002 (as amended in 2023)",
        "jurisdiction": "India",
        "category": "Access and Benefit Sharing",
        "source": "Kerala State Biodiversity Board",
    },
    {
        "file": "Biological_Diversity_Rules_2024.txt",
        "document": "Biological Diversity Rules, 2024",
        "jurisdiction": "India",
        "category": "Access and Benefit Sharing",
        "source": "Government of India Gazette",
    },
    {
        "file": "Biological_Diversity_Amendment_Rules_2025.txt",
        "document": "Biological Diversity (Amendment) Rules, 2025",
        "jurisdiction": "India",
        "category": "Access and Benefit Sharing",
        "source": "Government of India Gazette",
    },
    {
        "file": "Biological_Diversity_ABS_Regulations_2025.txt",
        "document": "Biological Diversity (ABS) Regulations, 2025",
        "jurisdiction": "India",
        "category": "Access and Benefit Sharing",
        "source": "National Biodiversity Authority",
    },
]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--abs-only",
    action="store_true",
    help="Build an ABS-only chunk file for incremental India indexing.",
)
args = parser.parse_args()

# Chunk settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def clean_text(text):
    """Clean unnecessary spaces while preserving page markers."""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_chunks(text):
    """Split text into overlapping chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if end < len(text):
            last_space = chunk.rfind(" ")

            if last_space > CHUNK_SIZE * 0.7:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk.strip())

        start = end - CHUNK_OVERLAP

    return chunks


all_chunks = []
chunk_id = 1


for doc_info in DOCUMENTS:
    source_metadata = trusted_source_for(doc_info["document"])
    is_abs = source_metadata.get("domain") == "abs"
    if args.abs_only and not is_abs:
        continue

    input_file = INPUT_FOLDER / doc_info["file"]

    print(f"\nProcessing: {doc_info['file']}")

    if not input_file.exists():
        print("File not found:", input_file)
        continue

    text = input_file.read_text(encoding="utf-8")

    text = clean_text(text)

    # Split according to page markers
    pages = re.split(r"--- Page (\d+) ---", text)

    current_page = 1

    for i in range(1, len(pages), 2):

        try:
            current_page = int(pages[i])
            page_text = pages[i + 1]
        except IndexError:
            continue

        page_text = page_text.strip()

        if not page_text:
            continue

        chunks = create_chunks(page_text)

        for chunk in chunks:
            record = {
                "chunk_id": chunk_id,
                "text": chunk,
                "document": doc_info["document"],
                "jurisdiction": doc_info["jurisdiction"],
                "domain": source_metadata.get("domain", "patent"),
                "category": doc_info["category"],
                "source": source_metadata.get("source_organization", doc_info["source"]),
                "source_organization": source_metadata.get("source_organization", doc_info["source"]),
                "official_listing_url": source_metadata.get("official_listing_url", doc_info.get("official_listing_url")),
                "official_pdf_url": source_metadata.get("official_pdf_url", doc_info.get("official_pdf_url")),
                "official_url": source_metadata.get("official_pdf_url") or source_metadata.get("official_listing_url") or doc_info.get("official_pdf_url") or doc_info.get("official_listing_url"),
                "source_url": source_metadata.get("official_pdf_url") or source_metadata.get("official_listing_url") or doc_info.get("official_pdf_url") or doc_info.get("official_listing_url"),
                "document_version": source_metadata.get("document_version", doc_info["document"]),
                "document_status": source_metadata.get("document_status"),
                "page": current_page
            }
            section = re.search(r"\bsection\s+(\d+[a-z]?(?:\s*\([^)]*\))?)", chunk, re.I)
            rule = re.search(r"\brule\s+(\d+[a-z]?(?:\s*\([^)]*\))?)", chunk, re.I)
            regulation = re.search(r"\bregulation\s+(\d+[a-z]?(?:\s*\([^)]*\))?)", chunk, re.I)
            if section:
                record["section"] = section.group(1).strip()
            if rule:
                record["rule"] = rule.group(1).strip()
            if regulation:
                record["regulation"] = regulation.group(1).strip()
            all_chunks.append(record)

            chunk_id += 1


# Save chunks
output_file = OUTPUT_FOLDER / ("abs_chunks.json" if args.abs_only else "chunks.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)


print("\n================================")
print("CHUNKING COMPLETED")
print("================================")
print("Total chunks:", len(all_chunks))
print("Saved to:", output_file)
