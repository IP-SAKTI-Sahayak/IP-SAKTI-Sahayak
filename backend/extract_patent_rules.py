import fitz
from pathlib import Path

pdf_file = "data/india/patents/Patents_Rules_2003.pdf"
txt_file = "data/extracted/Patents_Rules_2003.txt"

doc = fitz.open(pdf_file)

text = ""

for page_number, page in enumerate(doc, start=1):
    page_text = page.get_text()

    text += f"\n\n--- Page {page_number} ---\n\n"
    text += page_text

Path(txt_file).write_text(text, encoding="utf-8")

print("Pages:", len(doc))
print("Characters:", len(text))
print("Saved:", txt_file)