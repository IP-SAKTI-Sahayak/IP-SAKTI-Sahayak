import json
import re
from pathlib import Path


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
        "source": "IP India"
    },
    {
        "file": "Patents_Rules_2003.txt",
        "document": "Patents Rules, 2003",
        "jurisdiction": "India",
        "category": "Patent",
        "source": "IP India"
    },
    {
        "file": "Guidelines_Ayush_Related_Inventions_2025.txt",
        "document": "Guidelines for Examination of Ayush Related Inventions, 2025",
        "jurisdiction": "India",
        "category": "Ayush Patent Guidelines",
        "source": "IP India"
    }
]


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

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "document": doc_info["document"],
                "jurisdiction": doc_info["jurisdiction"],
                "category": doc_info["category"],
                "source": doc_info["source"],
                "page": current_page
            })

            chunk_id += 1


# Save chunks
output_file = OUTPUT_FOLDER / "chunks.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)


print("\n================================")
print("CHUNKING COMPLETED")
print("================================")
print("Total chunks:", len(all_chunks))
print("Saved to:", output_file)