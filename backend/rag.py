import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from backend.source_registry import trusted_source_for


INDEX_FILE = Path("vectorstore/index.faiss")
METADATA_FILE = Path("vectorstore/metadata.json")
INTERNATIONAL_INDEX_FILE = Path("vectorstore/index_international.faiss")
INTERNATIONAL_METADATA_FILE = Path("vectorstore/metadata_international.json")

# Support both FastAPI imports and the module's existing command-line entry point.
load_dotenv()

# Transparent evidence thresholds; they are not legal conclusions.
MIN_RETRIEVAL_SCORE = 0.42
MIN_STRONG_SEMANTIC_SCORE = 0.60
MIN_STRONG_SEMANTIC_RESULTS = 2
# A paraphrased question can have one very strong match and a second supporting
# government chunk just below the top-tier semantic threshold.  These values
# are deliberately used only together, for authoritative corroborated results;
# they are not a replacement for the baseline threshold below.
MIN_CORROBORATED_SEMANTIC_SCORE = 0.52
MIN_CORROBORATING_SCORE = 0.42
MIN_CORROBORATING_RESULTS = 2
MIN_CORROBORATED_AVERAGE_SCORE = 0.46
MIN_CLAIM_SIMILARITY = 0.35
MIN_KEYWORD_OVERLAP = 0.12
# A near-verbatim statutory claim can have a lower embedding score when the
# source chunk is long.  This deliberately high threshold is a transparent
# alternative evidence path, not a relaxation for weak lexical matches.
STRONG_KEYWORD_OVERLAP = 0.60
ABSTENTION_MESSAGE = (
    "The available sources do not provide enough information to answer this."
)

STOP_WORDS = {
    "about", "after", "against", "also", "and", "are", "based", "been",
    "being", "can", "does", "for", "from", "have", "how", "into", "its",
    "may", "not", "only", "our", "page", "that", "the", "their", "these",
    "this", "those", "under", "was", "what", "when", "which", "with", "will",
    "would", "you", "your",
}


index = faiss.read_index(str(INDEX_FILE))
chunks = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
international_index = (
    faiss.read_index(str(INTERNATIONAL_INDEX_FILE))
    if INTERNATIONAL_INDEX_FILE.exists() else None
)
international_chunks = (
    json.loads(INTERNATIONAL_METADATA_FILE.read_text(encoding="utf-8"))
    if INTERNATIONAL_METADATA_FILE.exists() else []
)
# The embedding model is already part of the local runtime; avoid a network
# metadata check when starting the API so retrieval remains available offline.
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def plan_clarification_with_gemini(user_query, conversation_context="", workflow_context="", jurisdiction="india", deterministic_analysis=None):
    """Ask the configured Gemini model for a JSON-only clarification plan."""
    from backend.query_understanding import ALLOWED_DOMAINS, ALLOWED_INTENTS, QUESTION_FIELD_SCHEMA

    prompt = {
        "task": "Plan the minimum useful clarification questions for the Sahayak query. Do not answer the legal question.",
        "user_query": user_query,
        "conversation_context": conversation_context,
        "workflow_context": workflow_context,
        "jurisdiction": jurisdiction,
        "deterministic_findings": deterministic_analysis or {},
        "rules": [
            "Return one JSON object only; no markdown.",
            "Use the latest explicit user correction over older context.",
            "Ask no more than five questions, and do not repeat known or answered facts.",
            "Use only the approved field names, intent values, and domain values supplied below.",
            "Return question_plan items with only field and question.",
            "Do not provide legal conclusions, citations, or claim facts that are not in the inputs.",
            "If no material information is missing, return an empty question_plan and needs_clarification false.",
        ],
        "approved_fields": list(QUESTION_FIELD_SCHEMA),
        "approved_intents": sorted(ALLOWED_INTENTS),
        "approved_domains": sorted(ALLOWED_DOMAINS),
        "json_shape": {"goal": "string", "primary_intent": "approved intent", "secondary_intents": [], "domains": [], "known_facts": {}, "missing_fields": [], "needs_clarification": True, "question_plan": [{"field": "approved field", "question": "short question"}]},
    }
    response = client.interactions.create(model="gemini-3.6-flash", input=json.dumps(prompt, ensure_ascii=False))
    raw = (response.output_text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    return json.loads(raw)


def retrieve(query, top_k=5, jurisdiction="india", allowed_categories=None, allowed_domains=None):
    """Return selected-jurisdiction evidence from the configured vector store.

    FAISS remains the default. Qdrant is an opt-in adapter over the same curated
    chunks and embeddings, selected with VECTOR_STORE_BACKEND=qdrant.
    """
    if os.getenv("VECTOR_STORE_BACKEND", "faiss").lower() == "qdrant":
        if jurisdiction.lower() not in {"india", "international"}:
            return []
        from backend.qdrant_store import search_chunks

        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return search_chunks(
            query_embedding, jurisdiction, top_k,
            allowed_categories=allowed_categories, allowed_domains=allowed_domains,
        )
    if jurisdiction.lower() == "international":
        active_index, active_chunks = international_index, international_chunks
    elif jurisdiction.lower() == "india":
        active_index, active_chunks = index, chunks
    else:
        return []
    # Never fall back to another jurisdiction when its index is unavailable.
    if active_index is None or not active_chunks:
        return []

    exact_matches = []
    direct_subclause_matches = []
    section_match = re.search(
        r"\bsection\s+(\d+)([a-z]?)(?:\(([a-z0-9]+)\))?", query.lower()
    )

    def chunk_domains(chunk):
        if chunk.get("domain"):
            return {chunk["domain"]}
        is_india_patent = chunk.get("jurisdiction", "").lower() == "india"
        patent_domains = {"patent"}
        if is_india_patent and re.search(
            r"\b(?:traditional knowledge|prior art|tkdl)\b", chunk.get("text", ""), re.I
        ):
            patent_domains.add("tkdl_prior_art")
        category_domains = {
            "Patent": patent_domains,
            "Ayush Patent Guidelines": patent_domains,
            "Biodiversity": {"abs"},
            "Access and Benefit Sharing": {"abs"},
            "Traditional Knowledge": {"tkdl_prior_art"},
            "Trademark": {"trademark"},
            "Design": {"design"},
            "Intellectual Property": {"ip"},
        }
        return category_domains.get(chunk.get("category"), set())

    def matches_domain(chunk):
        return not allowed_domains or bool(chunk_domains(chunk) & allowed_domains)

    def source_fields(chunk):
        registered = trusted_source_for(chunk["document"])
        matching_domains = chunk_domains(chunk) & (allowed_domains or chunk_domains(chunk))
        return {
            "domain": chunk.get("domain") or (sorted(matching_domains)[0] if matching_domains else None),
            "source_organization": chunk.get("source_organization") or registered.get("source_organization") or chunk.get("source"),
            "document_version": chunk.get("document_version") or registered.get("document_version"),
            "document_status": chunk.get("document_status") or registered.get("document_status"),
            "official_listing_url": registered.get("official_listing_url"),
            "official_pdf_url": registered.get("official_pdf_url"),
            "official_url": registered.get("official_pdf_url") or registered.get("official_listing_url") or chunk.get("official_url") or chunk.get("source_url"),
        }

    if section_match:
        section_number, section_suffix, subsection = section_match.groups()
        section_pattern = "section " + section_number + section_suffix
        for chunk in active_chunks:
            # The extracted Patents Act places Section 3 subclauses on a
            # continuation page where the "Section 3" heading is absent.
            # Match the requested subclause as an additional, narrow path so
            # that an exact query such as Section 3(p) retrieves the Act text.
            subclause_match = bool(
                section_number == "3"
                and subsection
                and chunk["document"] == "Patents Act, 1970"
                and re.search(
                    rf"^[ \t]*\({re.escape(subsection)}\)[ \t]+",
                    chunk["text"],
                    re.I | re.M,
                )
            )
            if (
                chunk["jurisdiction"].lower() == jurisdiction.lower()
                and matches_domain(chunk)
                and (allowed_categories is None or chunk["category"] in allowed_categories)
                and (section_pattern in chunk["text"].lower() or subclause_match)
            ):
                source_info = source_fields(chunk)
                match = {
                    "score": 1.0,
                    "document": chunk["document"],
                    "jurisdiction": chunk["jurisdiction"],
                    **source_info,
                    "category": chunk["category"],
                    "section": chunk.get("section"),
                    "rule": chunk.get("rule"),
                    "regulation": chunk.get("regulation"),
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                }
                # Place the actual statutory subclause ahead of secondary
                # references that merely mention the same section number.
                (direct_subclause_matches if subclause_match else exact_matches).append(match)

    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    scores, indices = active_index.search(query_embedding, min(15, len(active_chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = active_chunks[idx]
        if chunk["jurisdiction"].lower() != jurisdiction.lower():
            continue
        if not matches_domain(chunk):
            continue
        if allowed_categories is not None and chunk["category"] not in allowed_categories:
            continue
        source_info = source_fields(chunk)
        results.append({
            "score": float(score),
            "document": chunk["document"],
            "jurisdiction": chunk["jurisdiction"],
            **source_info,
            "category": chunk["category"],
            "section": chunk.get("section"),
            "rule": chunk.get("rule"),
            "regulation": chunk.get("regulation"),
            "page": chunk["page"],
            "source": chunk["source"],
            "text": chunk["text"],
        })

    unique_results = []
    seen = set()
    for result in direct_subclause_matches + exact_matches + results:
        key = (result["document"], result["page"], result["text"])
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    return unique_results[:top_k]


def _content_words(text):
    """Return normalized meaningful words for the lexical evidence check."""
    return {
        word for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


def _claim_for_citation(answer, citation_start, citation_end, document):
    """Use the sentence containing a citation as its associated claim."""
    sentence_start = max(
        answer.rfind(".", 0, citation_start),
        answer.rfind("!", 0, citation_start),
        answer.rfind("?", 0, citation_start),
        answer.rfind("\n", 0, citation_start),
    ) + 1
    end_positions = [
        position for position in (
            answer.find(".", citation_end),
            answer.find("!", citation_end),
            answer.find("?", citation_end),
            answer.find("\n", citation_end),
        ) if position != -1
    ]
    sentence_end = min(end_positions) if end_positions else len(answer)
    claim = answer[sentence_start:sentence_end]
    claim = re.sub(re.escape(document), "", claim, flags=re.IGNORECASE)
    claim = re.sub(r"\bdocument\s*:\s*", "", claim, flags=re.I)
    claim = re.sub(r"\b(?:page|p\.)\s*[:#\-]?\s*\d+\b", "", claim, flags=re.I)
    return re.sub(r"\s+", " ", claim).strip(" [](),;:-")


def extract_citations(answer):
    """Extract document/page citations and retain invalid ones for safe flagging."""
    citations = []
    documents = sorted({chunk["document"] for chunk in chunks}, key=len, reverse=True)
    page_pattern = re.compile(r"\b(?:page|p\.)\s*[:#\-]?\s*(\d+)\b", re.I)

    # First parse the structured format requested in the generation prompt.  A
    # non-corpus document is deliberately retained here so the API can expose
    # it as an unsupported citation instead of silently ignoring it.
    canonical_documents = {document.lower(): document for document in documents}
    structured_pattern = re.compile(
        r"\bdocument\s*:\s*([^;\]\n]+?)\s*;\s*page\s*:\s*(\d+)\b",
        re.I,
    )
    for match in structured_pattern.finditer(answer):
        cited_name = match.group(1).strip()
        document = canonical_documents.get(cited_name.lower(), cited_name)
        page = int(match.group(2))
        claim = _claim_for_citation(answer, match.start(), match.end(), cited_name)
        key = (document.lower(), page, claim.lower())
        if any(item["_key"] == key for item in citations):
            continue
        citations.append({
            "_key": key,
            "document": document,
            "page": page,
            "claim": claim,
        })

    for document in documents:
        for document_match in re.finditer(re.escape(document), answer, re.I):
            # A page number must appear near the document name to form a citation.
            window = answer[document_match.start(): min(len(answer), document_match.end() + 80)]
            page_match = page_pattern.search(window)
            if not page_match:
                continue
            page = int(page_match.group(1))
            citation_end = document_match.start() + page_match.end()
            claim = _claim_for_citation(answer, document_match.start(), citation_end, document)
            key = (document.lower(), page, claim.lower())
            if any(item["_key"] == key for item in citations):
                continue
            citations.append({
                "_key": key,
                "document": document,
                "page": page,
                "claim": claim,
            })
    return citations


def _source_metadata(document, page):
    """Get display metadata; its text is never used unless it was retrieved."""
    for chunk in chunks + international_chunks:
        if chunk["document"] == document and chunk["page"] == page:
            return chunk
    return None


def _source_excerpt(results, limit=600):
    """Return a compact, display-safe excerpt from the retrieved evidence."""
    text = " ".join(result["text"] for result in results)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def verify_citation_support(answer, results):
    """Check reference validity and claim support against retrieved page text only."""
    verified_citations = []
    for citation in extract_citations(answer):
        matching_results = [
            result for result in results
            if result["document"] == citation["document"]
            and result["page"] == citation["page"]
        ]
        metadata = matching_results[0] if matching_results else _source_metadata(
            citation["document"], citation["page"]
        )
        reference_valid = bool(matching_results)
        content_supported = False
        similarity = 0.0
        keyword_overlap = 0.0

        if reference_valid and citation["claim"]:
            claim_embedding = model.encode(
                [citation["claim"]], convert_to_numpy=True, normalize_embeddings=True
            )[0]
            source_embeddings = model.encode(
                [result["text"] for result in matching_results],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            similarity = float(np.max(source_embeddings @ claim_embedding))
            claim_words = _content_words(citation["claim"])
            source_words = _content_words(" ".join(
                result["text"] for result in matching_results
            ))
            keyword_overlap = (
                len(claim_words & source_words) / len(claim_words)
                if claim_words else 0.0
            )
            content_supported = (
                (similarity >= MIN_CLAIM_SIMILARITY
                 and keyword_overlap >= MIN_KEYWORD_OVERLAP)
                or keyword_overlap >= STRONG_KEYWORD_OVERLAP
            )

        verified_citations.append({
            "document": citation["document"],
            "page": citation["page"],
            "category": metadata["category"] if metadata else None,
            "source": metadata["source"] if metadata else None,
            "jurisdiction": metadata.get("jurisdiction") if metadata else None,
            "domain": metadata.get("domain") if metadata else None,
            "source_organization": metadata.get("source_organization", metadata.get("source")) if metadata else None,
            "document_version": metadata.get("document_version") if metadata else None,
            "document_status": metadata.get("document_status") if metadata else None,
            "section": metadata.get("section") if metadata else None,
            "rule": metadata.get("rule") if metadata else None,
            "regulation": metadata.get("regulation") if metadata else None,
            "official_listing_url": trusted_source_for(citation["document"]).get("official_listing_url"),
            "official_pdf_url": trusted_source_for(citation["document"]).get("official_pdf_url"),
            "official_url": trusted_source_for(citation["document"]).get("official_pdf_url") or trusted_source_for(citation["document"]).get("official_listing_url") or (metadata or {}).get("official_url"),
            "reference_valid": reference_valid,
            "content_supported": content_supported,
            "citation_verified": content_supported,
            "verification_status": "supported" if content_supported else "unsupported",
            "claim": citation["claim"],
            "semantic_similarity": round(similarity, 3),
            "keyword_overlap": round(keyword_overlap, 3),
            "source_excerpt": _source_excerpt(matching_results)
            if reference_valid else None,
        })
    return verified_citations


def validate_citations(answer, results):
    """Compatibility wrapper retaining the former public helper."""
    return [
        citation for citation in verify_citation_support(answer, results)
        if citation["reference_valid"]
    ]


def has_sufficient_evidence(question, results):
    """Require authoritative semantic or lexical evidence before generation."""
    if not results:
        return False
    if any(result["score"] == 1.0 for result in results):
        return True
    best_score = max(result["score"] for result in results)
    # Both jurisdiction-specific indexes are curated from trusted primary
    # authorities. Their source label is preserved per chunk (IP India, WIPO,
    # WTO, CBD, etc.); require that label before using corroboration thresholds.
    authoritative_results = [result for result in results if result.get("source")]
    strong_semantic_results = [
        result for result in authoritative_results
        if result["score"] >= MIN_STRONG_SEMANTIC_SCORE
    ]

    # A formulation question can use different wording from the governing
    # guideline.  Multiple strong matches in the curated government corpus are
    # sufficient semantic evidence; citation-content verification still guards
    # every generated claim before it can be marked as a successful answer.
    if (
        best_score >= MIN_STRONG_SEMANTIC_SCORE
        and len(strong_semantic_results) >= MIN_STRONG_SEMANTIC_RESULTS
    ):
        return True

    # Preserve safe abstention for a single, merely plausible hit.  This path
    # accepts paraphrased queries only when the curated authority produces a
    # strong lead plus a second independently retrieved corroborating chunk.
    # Citation verification still evaluates every generated claim afterwards.
    corroborating_results = [
        result for result in authoritative_results
        if result["score"] >= MIN_CORROBORATING_SCORE
    ]
    if len(corroborating_results) >= MIN_CORROBORATING_RESULTS:
        top_scores = sorted(
            (result["score"] for result in corroborating_results), reverse=True
        )[:MIN_CORROBORATING_RESULTS]
        if (
            top_scores[0] >= MIN_CORROBORATED_SEMANTIC_SCORE
            and sum(top_scores) / len(top_scores) >= MIN_CORROBORATED_AVERAGE_SCORE
        ):
            return True

    # Retain lexical corroboration for merely moderate retrieval, preserving
    # safe abstention for unrelated questions that happen to have one match.
    question_words = _content_words(question)
    evidence_words = _content_words(" ".join(result["text"] for result in results))
    return best_score >= MIN_RETRIEVAL_SCORE and len(question_words & evidence_words) >= 2


def confidence_from_evidence(results, citations):
    """Produce a qualitative confidence label from retrieval and verification."""
    strong_retrieval = bool(results) and max(
        result["score"] for result in results
    ) >= MIN_RETRIEVAL_SCORE
    if not citations:
        return "low"
    all_references_valid = all(citation["reference_valid"] for citation in citations)
    all_supported = all(citation["content_supported"] for citation in citations)
    any_supported = any(citation["content_supported"] for citation in citations)
    if strong_retrieval and all_references_valid and all_supported:
        return "high"
    if strong_retrieval and any_supported:
        return "medium"
    return "low"


def generate_answer(question, results, domain=None):
    context = ""
    for i, result in enumerate(results, 1):
        context += f"""
SOURCE {i}
Document: {result['document']}
Category: {result['category']}
Page: {result['page']}
Source: {result['source']}

Content:
{result['text']}

--------------------------------
"""

    prior_art_rules = """
11. This is an evidence pointer, not a TKDL database search. The available
    corpus contains public guidance and treaty material, not record-level
    TKDL entries. For a question asking whether a formulation is listed,
    explicitly state that authoritative record-level TKDL evidence could not
    be retrieved and abstain from yes/no conclusions.
12. Do not describe general guidance, a treaty, or a different formulation
    as a match to the user's formulation. A possible prior-art indication
    requires retrieved text that directly matches the supplied ingredients,
    preparation, or stated use. Even then, do not decide novelty or patentability.
13. State that formal patent examination is required whenever discussing a
    possible prior-art indication.
""" if domain == "tkdl_prior_art" else ""

    prompt = f"""
You are IP-SAKTI Sahayak, an Ayurveda Intellectual Property,
Regulatory and ABS guidance assistant.

Answer the user's question using ONLY the authoritative-source context below.

IMPORTANT RULES:
1. Answer using ONLY the supplied government-document context. Do not use
   background knowledge, assumptions, or facts not contained in that context.
2. Give a clear, direct answer first. Do not repeat the user's question.
3. Synthesize relevant passages together when they collectively support a
   qualified answer; do not treat each source passage as an isolated answer.
4. Do not invent laws, sections, rules, cases, dates, authorities, treaty
   provisions, URLs, patent requirements, or facts.
5. Cite every factual or legal claim inline in exactly this format:
   [Document: exact document name; Page: exact page number]
   A citation may only use a document and page supplied in the context.
6. Preserve legal caution. Use qualified language such as "may", "depends on",
   "subject to", or "the retrieved guidance indicates" whenever appropriate.
   Do not guarantee approval, rejection, registration, or legal compliance.
7. If the context genuinely has no sufficient information, reply exactly:
   "{ABSTENTION_MESSAGE}"
8. Do not provide legal advice.
9. Do not abstain solely because the user's wording is broader than the
   document's example or category. When the supplied evidence supports a
   conditional conclusion, explain that condition narrowly and cite the
   supporting document and page. Do not extend that conclusion beyond what
   the supplied evidence says.
10. Use short paragraphs and, where useful, concise bullets headed "Key
    points". Do not force a fixed structure for a short factual answer.
{prior_art_rules}

USER QUESTION:
{question}

AUTHORITATIVE SOURCE CONTEXT:
{context}
"""
    response = client.interactions.create(model="gemini-3.6-flash", input=prompt)
    return response.output_text


if __name__ == "__main__":
    question = input("\nEnter your question: ")
    results = retrieve(question, top_k=5)
    if not has_sufficient_evidence(question, results):
        answer, citations = ABSTENTION_MESSAGE, []
    else:
        answer = generate_answer(question, results)
        citations = verify_citation_support(answer, results)
    print("\nIP-SAKTI SAHAYAK\n")
    print("Answer:", answer)
    print("\nCitation verification:")
    for citation in citations:
        print(f"- {citation['document']} (Page {citation['page']}): "
              f"{citation['verification_status']}")
