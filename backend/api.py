from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import (
    ABSTENTION_MESSAGE,
    confidence_from_evidence,
    generate_answer,
    has_sufficient_evidence,
    retrieve,
    verify_citation_support,
)
from backend.query_understanding import (
    supported_categories_for_domains,
    understand_query,
)
from backend.workflows import (
    abs_readiness,
    append_audit_event,
    escalation_brief,
    formulation_intake,
    tkdl_intake,
)


app = FastAPI(
    title="IP-SAKTI Sahayak API",
    description="Ayurveda IPR, Regulatory and ABS Guidance API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class AssistantRequest(BaseModel):
    question: str
    jurisdiction: str = "india"
    language: str = "en"
    product_category: str = "unknown"


class FormulationIntake(BaseModel):
    product_type: str = ""
    source_text: str = ""
    intended_question: str = ""


class AbsIntake(BaseModel):
    biological_resource: str = ""
    access_origin: str = ""
    commercial_activity: str = ""


class TkdlIntake(BaseModel):
    claim: str = ""
    patent_context: str = ""
    jurisdiction: str = "india"


class EscalationRequest(BaseModel):
    question: str
    reason: str


def display_retrieved_sources(results):
    """Expose the retrieved government evidence for Postman/UI review."""
    sources = []
    for result in results:
        excerpt = " ".join(result["text"].split())
        sources.append({
            "document": result["document"],
            "page": result["page"],
            "category": result["category"],
            "source": result["source"],
            "official_url": None,
            "document_version": result["document"],
            "retrieval_score": round(result["score"], 3),
            "excerpt": excerpt[:600] + ("..." if len(excerpt) > 600 else ""),
        })
    return sources


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "IP-SAKTI Sahayak",
        "version": "1.1.0",
    }


@app.get("/v1/audit/status")
def audit_status():
    return {
        "enabled": True,
        "policy": "Local operational events record timestamps, event types and content hashes only.",
        "storage": "Local deployment storage",
    }


@app.post("/v1/workflows/formulation")
def formulation_workflow(request: FormulationIntake):
    result = formulation_intake(
        request.product_type, request.source_text, request.intended_question
    )
    append_audit_event("formulation_workflow", request.intended_question)
    return result


@app.post("/v1/workflows/abs")
def abs_workflow(request: AbsIntake):
    result = abs_readiness(
        request.biological_resource, request.access_origin, request.commercial_activity
    )
    append_audit_event("abs_workflow", request.commercial_activity)
    return result


@app.post("/v1/workflows/tkdl")
def tkdl_workflow(request: TkdlIntake):
    if request.jurisdiction not in ["india", "international"]:
        raise HTTPException(status_code=400, detail="Invalid jurisdiction")
    result = tkdl_intake(request.claim, request.patent_context, request.jurisdiction)
    append_audit_event("tkdl_workflow", request.claim)
    return result


@app.post("/v1/escalations")
def prepare_escalation(request: EscalationRequest):
    result = escalation_brief(request.question, request.reason)
    append_audit_event("escalation_brief", request.question)
    return result


@app.post("/v1/assistant/query")
def assistant_query(request: AssistantRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    append_audit_event("assistant_query", request.question)

    jurisdiction = request.jurisdiction.lower()
    if jurisdiction not in ["india", "international"]:
        raise HTTPException(
            status_code=400,
            detail="Jurisdiction must be 'india' or 'international'",
        )

    analysis = understand_query(request.question, jurisdiction)
    allowed_categories = supported_categories_for_domains(analysis["domains"])
    results = retrieve(
        request.question,
        top_k=5,
        jurisdiction=jurisdiction,
        allowed_categories=allowed_categories,
    )
    sufficient_evidence = (
        jurisdiction == "india"
        and bool(allowed_categories)
        and not analysis["needs_clarification"]
        and has_sufficient_evidence(request.question, results)
    )

    if analysis["needs_clarification"]:
        answer = analysis["clarification_question"]
        citations = []
        status = "needs_clarification"
        confidence = "low"
        safe_abstention = False
    elif not sufficient_evidence:
        # No model call for weak evidence: safe abstention is deterministic.
        answer = ABSTENTION_MESSAGE
        citations = []
        status = "abstained"
        confidence = "low"
        safe_abstention = True
    else:
        try:
            answer = generate_answer(request.question, results)
        except Exception:
            # Keep the API response safe when the external generator is down;
            # never substitute an ungrounded answer or expose provider details.
            raise HTTPException(
                status_code=503,
                detail=(
                    "The answer-generation service is temporarily unavailable. "
                    "Please try again."
                ),
            )
        citations = verify_citation_support(answer, results)
        if ABSTENTION_MESSAGE.lower() in answer.lower():
            status, confidence, safe_abstention = "abstained", "low", True
        elif not citations:
            # An answer without parseable source evidence cannot be success.
            status, confidence, safe_abstention = "needs_review", "low", False
        elif all(citation["content_supported"] for citation in citations):
            status = "success"
            confidence = confidence_from_evidence(results, citations)
            safe_abstention = False
        else:
            status = "needs_review"
            confidence = confidence_from_evidence(results, citations)
            safe_abstention = False

    return {
        "status": status,
        "question": request.question,
        "jurisdiction": jurisdiction,
        "analysis": analysis,
        "language": request.language.lower(),
        "product_category": request.product_category,
        "answer": answer,
        "confidence": confidence,
        "citations": citations,
        "retrieved_sources": display_retrieved_sources(results),
        "safe_abstention": safe_abstention,
        "disclaimer": (
            "This information is for general guidance only and does not "
            "constitute legal advice."
        ),
    }
