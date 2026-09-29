from pathlib import Path
import pymupdf
import re

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "international"
OUTPUT_DIR = BASE_DIR / "data" / "extracted_international"


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdf_files = list(SOURCE_DIR.rglob("*.pdf"))

print(f"International PDFs found: {len(pdf_files)}")

for pdf_path in pdf_files:

    output_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"

    try:
        doc = pymupdf.open(pdf_path)

        with output_file.open("w", encoding="utf-8") as f:

            for page_number, page in enumerate(doc, start=1):
                text = clean_text(page.get_text())

                if text:
                    f.write(f"\n--- PAGE {page_number} ---\n")
                    f.write(text)
                    f.write("\n")

        doc.close()

        print(f"[OK] {pdf_path.name}")

    except Exception as e:
        print(f"[ERROR] {pdf_path.name}: {e}")

print("\nInternational extraction completed.")