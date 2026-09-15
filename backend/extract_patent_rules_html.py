import requests
from bs4 import BeautifulSoup
from pathlib import Path

url = "https://ipindia.gov.in/pages/patents/rules-patents-2003"

response = requests.get(url, timeout=30)

print("Status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

# Remove unnecessary website elements
for tag in soup(["script", "style", "nav", "header", "footer"]):
    tag.decompose()

text = soup.get_text("\n", strip=True)

output_file = "data/extracted/Patents_Rules_2003.txt"

Path(output_file).write_text(text, encoding="utf-8")

print("Characters:", len(text))
print("Saved:", output_file)