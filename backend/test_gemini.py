import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("Gemini configuration is not loaded.")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Return the word OK."
)

print("Gemini minimal request: passed" if response.output_text.strip().upper() == "OK" else "Gemini responded, but not with the expected word.")
