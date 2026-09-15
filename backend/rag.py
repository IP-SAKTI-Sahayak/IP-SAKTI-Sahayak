import json
import os
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from google import genai


# ==============================
# FILE PATHS
# ==============================

INDEX_FILE = Path("vectorstore/index.faiss")
METADATA_FILE = Path("vectorstore/metadata.json")


# ==============================
# LOAD RAG COMPONENTS
# ==============================

index = faiss.read_index(str(INDEX_FILE))

chunks = json.loads(
    METADATA_FILE.read_text(encoding="utf-8")
)

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# GEMINI CLIENT
# ==============================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# ==============================
# RETRIEVAL
# ==============================

def retrieve(query, top_k=5):

    query_lower = query.lower()

    # --------------------------------
    # 1. Generic exact legal section matching
    # --------------------------------

    exact_matches = []

    section_match = re.search(
        r'\bsection\s+(\d+[a-z]?(?:\([a-z0-9]+\))?)',
        query_lower
    )

    if section_match:

        requested_section = section_match.group(1)

        section_pattern = "section " + requested_section

        print(
            f"\nDetected legal section: "
            f"Section {requested_section}"
        )

        # Search all chunks for exact section
        for chunk in chunks:

            chunk_text_lower = chunk["text"].lower()

            if section_pattern in chunk_text_lower:

                exact_matches.append({
                    "score": 1.0,
                    "document": chunk["document"],
                    "category": chunk["category"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "text": chunk["text"]
                })

    # --------------------------------
    # 2. Normal FAISS retrieval
    # --------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        15
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        chunk = chunks[idx]

        results.append({
            "score": float(score),
            "document": chunk["document"],
            "category": chunk["category"],
            "page": chunk["page"],
            "source": chunk["source"],
            "text": chunk["text"]
        })

    # --------------------------------
    # 3. Put exact matches first
    # --------------------------------

    combined = exact_matches + results

    # --------------------------------
    # 4. Remove duplicate chunks
    # --------------------------------

    unique_results = []
    seen = set()

    for result in combined:

        key = (
            result["document"],
            result["page"],
            result["text"]
        )

        if key not in seen:

            seen.add(key)
            unique_results.append(result)

    return unique_results[:top_k]


# ==============================
# CITATION VALIDATION
# ==============================

def validate_citations(answer, results):

    validated_sources = []

    answer_lower = answer.lower()

    for result in results:

        document = result["document"]
        page = str(result["page"])

        document_lower = document.lower()

        # --------------------------------
        # Check whether this exact retrieved
        # document is cited in the answer
        # --------------------------------

        document_found = document_lower in answer_lower

        # --------------------------------
        # Check whether the exact page
        # belonging to that retrieved document
        # is cited
        # --------------------------------

        page_patterns = [
            f"page {page}",
            f"page: {page}",
            f"page-{page}",
            f"page - {page}"
        ]

        page_found = any(
            pattern in answer_lower
            for pattern in page_patterns
        )

        # --------------------------------
        # Validate only when BOTH document
        # and page are present
        # --------------------------------

        if document_found and page_found:

            validated_sources.append({
                "document": document,
                "page": result["page"],
                "category": result["category"],
                "source": result["source"],
                "text": result["text"]
            })

    # --------------------------------
    # Remove duplicate document/page pairs
    # --------------------------------

    unique_sources = []
    seen = set()

    for source in validated_sources:

        key = (
            source["document"],
            source["page"]
        )

        if key not in seen:

            seen.add(key)
            unique_sources.append(source)

    return unique_sources


# ==============================
# GEMINI RAG ANSWER
# ==============================

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

    # ==============================
    # DEBUG: SHOW CONTEXT
    # ==============================

    print("\n========== CONTEXT SENT TO GEMINI ==========")
    print(context)
    print("============================================")

    # ==============================
    # GEMINI PROMPT
    # ==============================

    prompt = f"""
You are IP-SAKTI Sahayak, an Ayurveda Intellectual Property,
Regulatory and ABS guidance assistant.

Answer the user's question using ONLY the government-document
context provided below.

IMPORTANT RULES:

1. Give ONE clear and direct answer.

2. Use the information from the retrieved government documents
   to answer the question.

3. Do not say that information is insufficient if the context
   contains enough information to provide a reasonable,
   qualified answer.

4. Do not invent laws, sections, rules, cases, dates or facts.

5. If the documents provide related information but do not give
   a direct yes/no answer, give a qualified answer based only
   on the available information.

6. Mention the relevant document and page number for the facts
   used in the answer.

7. If the context genuinely contains no useful information
   related to the question, say:

   "The available sources do not provide enough information to answer this."

8. Do not provide legal advice.

9. Keep the answer simple and suitable for a user asking about
   Ayurveda IPR and regulatory matters.

10. Clearly distinguish what is directly stated in the sources
    from any conclusion drawn from those sources.

11. When citing a source, use the document name and the page
    number exactly as provided in the GOVERNMENT DOCUMENT CONTEXT.

USER QUESTION:
{question}

GOVERNMENT DOCUMENT CONTEXT:
{context}
"""

    # ==============================
    # GENERATE ANSWER
    # ==============================

    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return response.output_text


# ==============================
# MAIN PROGRAM
# ==============================

if __name__ == "__main__":

    question = input("\nEnter your question: ")

    print("\nSearching government documents...")

    results = retrieve(
        question,
        top_k=5
    )

    print("\nGenerating RAG answer...")

    answer = generate_answer(
        question,
        results
    )

    # ==============================
    # VALIDATE CITATIONS
    # ==============================

    validated_sources = validate_citations(
        answer,
        results
    )

    # ==============================
    # DISPLAY ANSWER
    # ==============================

    print("\n================================")
    print("IP-SAKTI SAHAYAK")
    print("================================")

    print("\nAnswer:")
    print(answer)

    # ==============================
    # DISPLAY CITATION VALIDATION
    # ==============================

    print("\n================================")
    print("Citation Validation")
    print("================================")

    if validated_sources:

        print(
            f"\nValidated citations: "
            f"{len(validated_sources)}"
        )

        for source in validated_sources:

            print(
                f"- {source['document']} "
                f"(Page {source['page']})"
            )

    else:

        print("\nNo citations could be validated.")

    # ==============================
    # DISPLAY RETRIEVED SOURCES
    # ==============================

    print("\n================================")
    print("Retrieved Sources")
    print("================================")

    for i, result in enumerate(results, 1):

        print(f"\n[{i}] {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Category: {result['category']}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")