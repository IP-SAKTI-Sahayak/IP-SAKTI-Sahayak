"""Pluggable translation boundary for chat input and responses.

The default provider reuses the application's configured Gemini client. A
future Bhashini adapter can implement the same ``translate`` interface.
"""

from typing import Protocol


class TranslationProvider(Protocol):
    def translate(self, text: str, target_language: str, source_language: str = "auto") -> str:
        ...


class GeminiTranslationProvider:
    def translate(self, text: str, target_language: str, source_language: str = "auto") -> str:
        from backend.rag import client

        if not text.strip() or target_language.lower() in {"en", "english"}:
            return text
        language_names = {
            "hi": "Hindi",
            "hindi": "Hindi",
            "ta": "Tamil",
            "tamil": "Tamil",
            "te": "Telugu",
            "telugu": "Telugu",
            "ml": "Malayalam",
            "malayalam": "Malayalam",
        }
        target = language_names.get(target_language.lower(), target_language)
        prompt = (
            f"Translate the following user-facing Sahayak text from {source_language} to {target}. "
            "Return only the translation. Preserve URLs, bracketed citation references, legal section numbers, "
            "and official document titles exactly. Do not add legal advice or explanatory text.\n\n"
            f"TEXT:\n{text}"
        )
        response = client.interactions.create(model="gemini-3.6-flash", input=prompt)
        translated = (response.output_text or "").strip()
        if not translated:
            raise ValueError("The translation provider returned an empty translation.")
        return translated


def translate_text(text: str, target_language: str, source_language: str = "auto", provider: TranslationProvider | None = None) -> str:
    """Translate through an injectable provider; English is a no-op."""
    if target_language.lower() in {"en", "english"} or not text:
        return text
    return (provider or GeminiTranslationProvider()).translate(text, target_language, source_language)
