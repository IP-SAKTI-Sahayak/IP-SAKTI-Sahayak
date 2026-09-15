from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain Ayurveda intellectual property rights in one simple paragraph."
)

print("\nGemini Response:")
print(response.output_text)