from pypdf import PdfReader
from pathlib import Path

files = [
    (
        "data/india/patents/Patents_Act_1970.pdf",
        "data/extracted/Patents_Act_1970.txt"
    ),
    (
        "data/india/patents/Patents_Rules_2003.pdf",
        "data/extracted/Patents_Rules_2003.txt"
    ),
    (
        "data/india/ayush/Guidelines_Ayush_Related_Inventions_2025.pdf",
        "data/extracted/Guidelines_Ayush_Related_Inventions_2025.txt"
    )
]

for pdf_file, txt_file in files:

    print(f"\nExtracting: {pdf_file}")

    reader = PdfReader(pdf_file)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        text += f"\n\n--- Page {page_number} ---\n\n"
        text += page_text

    Path(txt_file).write_text(text, encoding="utf-8")

    print(f"Pages: {len(reader.pages)}")
    print(f"Characters: {len(text)}")
    print(f"Saved: {txt_file}")

print("\nALL PDFs EXTRACTED SUCCESSFULLY!")
