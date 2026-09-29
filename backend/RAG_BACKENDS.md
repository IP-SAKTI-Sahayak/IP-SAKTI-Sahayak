# Retrieval and translation adapters

## Vector retrieval

The application continues to use its current FAISS indexes by default. The
optional Qdrant adapter reuses the same curated chunk metadata and
`all-MiniLM-L6-v2` 384-dimensional embeddings; it does not replace the FAISS
corpus or change the default route.

To build and enable Qdrant explicitly:

1. Install `backend/requirements-qdrant.txt` in the backend environment.
2. Run `python -m backend.qdrant_store` once to index the existing India and
   International chunk files.
3. Set `VECTOR_STORE_BACKEND=qdrant` for the backend process. The default Qdrant
   path is `vectorstore/qdrant`; set `QDRANT_URL` to use a Qdrant server instead.
4. Keep the existing India/International jurisdiction selection. Qdrant applies
   it as a required payload filter before returning evidence.

FAISS has no additional service and remains the simplest local deployment.
Qdrant adds metadata filtering and a path to a shared vector service, at the cost
of an optional dependency and an additional store to index and operate. Qdrant
mode returns no evidence when its collection is missing; it does not silently
mix in results from the other jurisdiction or switch stores.

## Translation

`backend/translation.py` defines a small `TranslationProvider` interface. Its
current provider uses the existing Gemini client and supports the UI's English
and Hindi choices. The API translates Hindi input to English before query
understanding and retrieval, and translates the user-facing response afterward.
Citation verification runs against the original grounded English response.
Provider failures are reported in `translation_status` and leave an English
fallback response. A future Bhashini adapter can implement the same interface.

## Audit record

Chat and escalation events are appended to local `data/runtime/audit.jsonl`.
Chat entries include the query text, selected jurisdiction, cited document
names, pseudonymous browser actor ID, timestamp, and escalation flag. The
current demo login has no server-side account identity, so `user_id` is a
browser-generated identifier rather than a verified account ID. This local
record is a readiness mechanism, not a complete DPDP governance, access,
retention, or deletion implementation.
