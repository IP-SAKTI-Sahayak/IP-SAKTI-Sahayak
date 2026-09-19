import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("vectorstore/index.faiss")
METADATA_FILE = Path("vectorstore/metadata.json")

# Support both FastAPI imports and the module's existing command-line entry point.
load_dotenv()

# Transparent evidence thresholds; they are not legal conclusions.
MIN_RETRIEVAL_SCORE = 0.42
MIN_STRONG_SEMANTIC_SCORE = 0.60
MIN_STRONG_SEMANTIC_RESULTS = 2
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
# The embedding model is already part of the local runtime; avoid a network
# metadata check when starting the API so retrieval remains available offline.
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def retrieve(query, top_k=5, jurisdiction="india", allowed_categories=None):
    """Return the existing exact-section and FAISS evidence results."""
    exact_matches = []
    direct_subclause_matches = []
    section_match = re.search(
        r"\bsection\s+(\d+)([a-z]?)(?:\(([a-z0-9]+)\))?", query.lower()
    )

    if section_match:
        section_number, section_suffix, subsection = section_match.groups()
        section_pattern = "section " + section_number + section_suffix
        for chunk in chunks:
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
                and (not allowed_categories or chunk["category"] in allowed_categories)
                and (section_pattern in chunk["text"].lower() or subclause_match)
            ):
                match = {
                    "score": 1.0,
                    "document": chunk["document"],
                    "category": chunk["category"],
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
    scores, indices = index.search(query_embedding, 15)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        if chunk["jurisdiction"].lower() != jurisdiction.lower():
            continue
        if allowed_categories and chunk["category"] not in allowed_categories:
            continue
        results.append({
            "score": float(score),
            "document": chunk["document"],
            "category": chunk["category"],
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
    for chunk in chunks:
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
            "reference_valid": reference_valid,
            "content_supported": content_supported,
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
    authoritative_results = [
        result for result in results if result["source"] == "IP India"
    ]
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


def generate_answer(question, results):
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

    prompt = f"""
You are IP-SAKTI Sahayak, an Ayurveda Intellectual Property,
Regulatory and ABS guidance assistant.

Answer the user's question using ONLY the government-document context below.

IMPORTANT RULES:
1. Give one clear, direct answer using only the supplied context.
2. Do not invent laws, sections, rules, cases, dates, facts, or authorities.
3. Cite every factual or legal claim inline in exactly this format:
   [Document: exact document name; Page: exact page number]
4. A citation may only use a document and page supplied in the context.
5. Clearly identify a qualified conclusion as a conclusion, rather than as a
   direct statement of the source.
6. If the context genuinely has no sufficient information, reply exactly:
   "{ABSTENTION_MESSAGE}"
7. Do not provide legal advice.

USER QUESTION:
{question}

GOVERNMENT DOCUMENT CONTEXT:
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
