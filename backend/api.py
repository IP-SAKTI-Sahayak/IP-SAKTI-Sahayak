from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import re
import logging
import sqlite3
import os
from backend.translation import translate_text
from backend.delivery import send_escalation_email

from backend.rag import (
    ABSTENTION_MESSAGE,
    confidence_from_evidence,
    generate_answer,
    has_sufficient_evidence,
    retrieve,
    verify_citation_support,
    plan_clarification_with_gemini,
)
from backend.query_understanding import (
    QUESTION_FIELD_SCHEMA,
    supported_categories_for_domains,
    understand_query,
    validate_gemini_plan,
)
from backend.workflows import (
    abs_readiness,
    append_audit_event,
    escalation_brief,
    formulation_intake,
    tkdl_intake,
)
from backend.source_registry import TRUSTED_SOURCES
from backend import db
from backend.auth import get_current_user, require_roles, hash_password, verify_password, issue_token


app = FastAPI(
    title="IP-SAKTI Sahayak API",
    description="Ayurveda IPR, Regulatory and ABS Guidance API",
    version="1.1.0",
)
db.init_db()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)


class AssistantRequest(BaseModel):
    question: str
    jurisdiction: str = "india"
    language: str = "en"
    product_category: str = "unknown"
    conversation_context: str = ""
    workflow_context: str = ""
    user_id: str = "anonymous"


class FormulationIntake(BaseModel):
    product_type: str = ""
    source_text: str = ""
    intended_question: str = ""
    classical_source: str = ""
    newly_developed: str = ""


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
    contact_email: str = ""
    user_id: str = "anonymous"
    jurisdiction: str = "india"
    sources_cited: list[str] = Field(default_factory=list)
    assistant_response: str = ""


class AuthRequest(BaseModel):
    email: str
    password: str


class FacilitatorReplyRequest(BaseModel):
    reply: str


def _validated_email(email):
    normalized = email.strip().lower()
    if len(normalized) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return normalized


@app.post("/v1/auth/register", status_code=201)
def register(request: AuthRequest):
    email = _validated_email(request.email)
    if len(request.password) < 8 or len(request.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters and no more than 72 UTF-8 bytes.")
    try:
        user = db.create_user(email, hash_password(request.password), role="user")
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    return {"notice": "Account created. Please sign in.", "user": user}


@app.post("/v1/auth/login")
def login(request: AuthRequest):
    email = _validated_email(request.email)
    user = db.get_user_by_email(email)
    if user is None or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    public = db.public_user(user)
    return {"access_token": issue_token(public), "token_type": "bearer", "user": public}


@app.get("/v1/auth/me")
def auth_me(user=Depends(get_current_user)):
    return {"user": user}


@app.get("/v1/my-requests")
def my_requests(user=Depends(get_current_user)):
    return {"requests": db.list_user_escalations(user["id"])}


@app.get("/v1/my-requests/{escalation_id}")
def my_request(escalation_id: int, user=Depends(get_current_user)):
    item = db.get_escalation(escalation_id)
    if item is None or (item["user_id"] != user["id"] and user["role"] not in {"facilitator", "admin"}):
        raise HTTPException(status_code=404, detail="Request not found.")
    return {"request": item}


@app.get("/v1/facilitator/escalations")
def facilitator_escalations(user=Depends(require_roles("facilitator", "admin"))):
    items = db.list_escalations()
    return {"new": [item for item in items if item["status"] == "new"],
            "in_progress": [item for item in items if item["status"] == "in_progress"],
            "answered": [item for item in items if item["status"] == "answered"]}


@app.get("/v1/facilitator/escalations/{escalation_id}")
def facilitator_escalation(escalation_id: int, user=Depends(require_roles("facilitator", "admin"))):
    item = db.get_escalation(escalation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Request not found.")
    return {"request": item}


@app.post("/v1/facilitator/escalations/{escalation_id}/claim")
def claim_escalation(escalation_id: int, user=Depends(require_roles("facilitator", "admin"))):
    item = db.get_escalation(escalation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Request not found.")
    if item["status"] == "new":
        item = db.update_escalation_status(escalation_id, "in_progress")
    return {"request": item}


@app.post("/v1/facilitator/escalations/{escalation_id}/reply")
def reply_to_escalation(escalation_id: int, request: FacilitatorReplyRequest,
                        user=Depends(require_roles("facilitator", "admin"))):
    if not request.reply.strip():
        raise HTTPException(status_code=400, detail="Reply cannot be empty.")
    if db.get_escalation(escalation_id) is None:
        raise HTTPException(status_code=404, detail="Request not found.")
    return {"request": db.answer_escalation(escalation_id, request.reply.strip())}


def display_retrieved_sources(results):
    """Expose the retrieved government evidence for Postman/UI review."""
    sources = []
    for result in results:
        excerpt = " ".join(result["text"].split())
        sources.append({
            "document": result["document"],
            "jurisdiction": result.get("jurisdiction"),
            "domain": result.get("domain"),
            "page": result["page"],
            "category": result["category"],
            "source": result["source"],
            "source_organization": result.get("source_organization", result["source"]),
            "official_url": result.get("official_url"),
            "official_listing_url": result.get("official_listing_url"),
            "official_pdf_url": result.get("official_pdf_url"),
            "document_version": result.get("document_version") or result["document"],
            "document_status": result.get("document_status"),
            "section": result.get("section"),
            "rule": result.get("rule"),
            "regulation": result.get("regulation"),
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
        "policy": "Local operational events are stored in local JSONL with query content for audit readiness.",
        "storage": "Local deployment storage",
    }


@app.post("/v1/workflows/formulation")
def formulation_workflow(request: FormulationIntake, user=Depends(get_current_user)):
    result = formulation_intake(
        request.product_type, request.source_text, request.intended_question,
        classical_source=request.classical_source,
        newly_developed=request.newly_developed,
    )
    append_audit_event("formulation_workflow", request.intended_question)
    return result


@app.post("/v1/workflows/abs")
def abs_workflow(request: AbsIntake, user=Depends(get_current_user)):
    result = abs_readiness(
        request.biological_resource, request.access_origin, request.commercial_activity
    )
    append_audit_event("abs_workflow", request.commercial_activity)
    return result


@app.post("/v1/workflows/tkdl")
def tkdl_workflow(request: TkdlIntake, user=Depends(get_current_user)):
    if request.jurisdiction not in ["india", "international"]:
        raise HTTPException(status_code=400, detail="Invalid jurisdiction")
    result = tkdl_intake(request.claim, request.patent_context, request.jurisdiction)
    append_audit_event("tkdl_workflow", request.claim)
    return result


@app.post("/v1/escalations")
def prepare_escalation(request: EscalationRequest, user=Depends(get_current_user)):
    if request.jurisdiction not in {"india", "international"}:
        raise HTTPException(status_code=400, detail="Invalid jurisdiction")
    result = escalation_brief(request.question, request.reason)
    result["contact_provided"] = bool(request.contact_email.strip())
    timestamp = datetime.now(timezone.utc).isoformat()
    delivery_status, delivery_error = send_escalation_email(
        user_id=request.user_id or "anonymous",
        timestamp=timestamp,
        jurisdiction=request.jurisdiction,
        question=request.question,
        assistant_response=request.assistant_response,
        reason=request.reason,
        contact_email=request.contact_email,
    )
    result["delivery_status"] = delivery_status
    result["notice"] = {
        "sent": "Your request was sent to the facilitator.",
        "failed": "Your request could not be sent. Please try again later.",
        "not_configured": "Your request was prepared, but facilitator email delivery is not configured.",
    }[delivery_status]
    if delivery_error:
        result["delivery_error"] = delivery_error
    append_audit_event(
        "escalation_brief", request.question, user_id=request.user_id,
        jurisdiction=request.jurisdiction, sources_cited=request.sources_cited,
        escalation_triggered=True,
        delivery_status=delivery_status,
        delivery_error=delivery_error,
        timestamp=timestamp,
    )
    try:
        db.create_escalation(
            user_id=user["id"], question=request.question, details=request.reason,
            contact_email=request.contact_email, assistant_response=request.assistant_response,
            jurisdiction=request.jurisdiction, timestamp=timestamp,
        )
    except Exception:
        logger.exception("Unable to persist facilitator escalation in SQLite")
    return result


@app.post("/v1/assistant/query")
def assistant_query(request: AssistantRequest, user=Depends(get_current_user)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    jurisdiction = request.jurisdiction.lower()
    if jurisdiction not in ["india", "international"]:
        raise HTTPException(
            status_code=400,
            detail="Jurisdiction must be 'india' or 'international'",
        )

    language = request.language.lower()
    translation_status = "not_needed" if language == "en" else "translated"
    query_text = request.question
    conversation_context = request.conversation_context
    workflow_context = request.workflow_context
    if language != "en":
        try:
            source_language = "auto"
            query_text = translate_text(request.question, "en", source_language=source_language)
            conversation_context = translate_text(conversation_context, "en", source_language=source_language) if conversation_context.strip() else ""
            workflow_context = translate_text(workflow_context, "en", source_language=source_language) if workflow_context.strip() else ""
        except Exception:
            translation_status = "input_translation_unavailable"
            query_text = request.question
            conversation_context = request.conversation_context
            workflow_context = request.workflow_context

    # Context consists only of user-supplied prior messages or deterministic
    # workflow results supplied by the client. Keep the visible question intact.
    routed_question = query_text
    prior_context = "\n\n".join(
        value.strip() for value in (conversation_context, workflow_context)
        if value and value.strip()
    )
    category_aliases = {
        "classical_medicine": "Classical Ayurveda",
        "proprietary_medicine": "Proprietary formulation",
        "new_non_classical_drug": "New / non-classical medicine",
        "phytopharmaceutical": "Phytopharmaceutical",
        "ayurveda_aahar_nutraceutical": "Ayurveda-Aahara / nutraceutical",
        "cosmetic": "Cosmetic",
    }


    supplied_category = request.product_category.strip().lower()
    if supplied_category in category_aliases:
        prior_context = "\n\n".join(filter(None, (
            prior_context, f"formulation_type: {category_aliases[supplied_category]}"
        )))
    if prior_context:
        routed_question = (
            "Relevant prior user-provided context:\n"
            + prior_context
            + "\n\nCurrent user message:\n"
            + query_text
        )
    analysis = understand_query(routed_question, jurisdiction)
    planner_context = "\n\n".join(part for part in (conversation_context.strip(), workflow_context.strip(), routed_question) if part)
    try:
        plan = plan_clarification_with_gemini(
            query_text, conversation_context, workflow_context,
            jurisdiction, analysis,
        )
        analysis = validate_gemini_plan(plan, analysis, planner_context) or analysis
    except Exception:
        # Preserve deterministic query understanding if Gemini is unavailable or malformed.
        pass

    category_names = {
        "classical_medicine": "Classical formulation",
        "proprietary_medicine": "Proprietary formulation",
        "new_non_classical_drug": "New / non-classical medicine",
        "phytopharmaceutical": "Phytopharmaceutical",
        "ayurveda_aahar_nutraceutical": "Ayurveda-Aahara / nutraceutical",
        "cosmetic": "Cosmetic",
    }
    if supplied_category in category_names:
        analysis["formulation_category"] = supplied_category
        analysis.setdefault("known_fields", {})["formulation_type"] = supplied_category
    formulation_category = analysis.get("formulation_category")
    category_label = category_names.get(formulation_category)
    combined_context = "\n".join((request.question, prior_context)).lower()
    formulation_question = bool(re.search(
        r"\b(?:formulation|ayurvedic product|herbal product|medicine|drug|classical ayurveda|proprietary formulation|phytopharmaceutical|nutraceutical|cosmetic)\b",
        combined_context,
    ))
    ip_or_regulatory_question = (
        bool(set(analysis.get("domains", [])) & {"patent", "regulation", "export"})
        or bool(re.search(
            r"\b(?:patent|protect|protection|intellectual property|ip rights?|register|registration|regulat(?:ion|ory))\b",
            combined_context,
        ))
    )
    general_guidance_requested = bool(re.search(
        r"\b(?:just answer generally|answer generally|general guidance|skip this|skip classification|skip reclassification|just skip)\b",
        combined_context,
        re.I,
    )) or bool(re.search(
        r"clarification answer field:\s*formulation_type\s*=\s*(?:not sure|unsure|unknown)\b",
        combined_context,
        re.I,
    ))
    clarification_turns = len(re.findall(
        r"clarification answer field:\s*[a-z_]+\s*=",
        "\n".join((conversation_context, request.question)),
        re.I,
    ))
    clarification_cap_hit = clarification_turns >= 5
    force_answer_caveat = False
    analysis["clarification_turns"] = clarification_turns
    analysis["clarification_cap_hit"] = clarification_cap_hit
    category_gate_required = (
        formulation_question and ip_or_regulatory_question
        and not category_label and not general_guidance_requested
        and not clarification_cap_hit
    )
    if category_gate_required:
        gate_question = (
            "Which best describes it: classical, proprietary, new/non-classical, "
            "phytopharmaceutical, Ayurveda-Aahara/nutraceutical, or cosmetic?"
        )
        analysis.update({
            "formulation_category": "unknown",
            "needs_clarification": True,
            "clarification_questions": [gate_question],
            "clarification_fields": [{"field": "formulation_type", "question": gate_question}],
            "question_plan": [{
                "field": "formulation_type", "question": gate_question,
                "input_type": "single_select",
                "options": [
                    "Classical Ayurveda", "Proprietary formulation",
                    "New / non-classical medicine", "Phytopharmaceutical",
                    "Ayurveda-Aahara / nutraceutical", "Cosmetic", "Not sure",
                ],
                "allow_multiple": False, "required": True,
            }],
            "known_fields": {**analysis.get("known_fields", {}), "formulation_type": None},
            "missing_fields": ["formulation_type"],
            "required_fields": ["formulation_type"],
            "clarification_total": 1,
            "clarification_question": gate_question,
        })
    elif general_guidance_requested and formulation_question:
        analysis.update({
            "needs_clarification": False,
            "clarification_questions": [],
            "clarification_fields": [],
            "question_plan": [],
            "missing_fields": [],
            "required_fields": [],
            "clarification_total": 0,
            "clarification_question": None,
        })

    selected_areas = analysis.get("known_fields", {}).get("guidance_areas") or []
    if isinstance(selected_areas, str):
        selected_areas = [part.strip() for part in re.split(r"\s*[;,|]\s*", selected_areas) if part.strip()]
    area_specs = (
        ("export", "Export requirements", {"export"}, ["destination_country", "product_type"]),
        ("market", "Market approval", {"regulation"}, ["destination_country", "product_type"]),
        ("ip", "IP protection", {"patent"}, ["protection_subject", "invention_feature", "public_disclosure"]),
        ("tkdl", "Traditional knowledge / TKDL", {"tkdl_prior_art"}, ["formulation_name", "ingredients", "traditional_use"]),
    )
    selected_area_specs = []
    for area in selected_areas:
        normalized_area = area.lower()
        spec = next((item for item in area_specs if (
            (item[0] == "export" and "export" in normalized_area)
            or (item[0] == "market" and ("market approval" in normalized_area or "destination-country approval" in normalized_area))
            or (item[0] == "ip" and ("ip protection" in normalized_area or "patent" in normalized_area))
            or (item[0] == "tkdl" and ("traditional knowledge" in normalized_area or "tkdl" in normalized_area))
        )), None)
        if spec and spec not in selected_area_specs:
            selected_area_specs.append(spec)

    guidance_required_fields = []
    for spec in selected_area_specs:
        for field in spec[3]:
            if field not in guidance_required_fields:
                guidance_required_fields.append(field)
    if selected_area_specs and any(spec[0] == "ip" for spec in selected_area_specs) and not category_label:
        guidance_required_fields.insert(0, "formulation_type")

    if selected_area_specs and not general_guidance_requested:
        known_fields = analysis.setdefault("known_fields", {})
        def has_substantive_value(value, field):
            if value is None:
                return False
            if isinstance(value, list):
                return bool(value)
            normalized = str(value).strip().lower()
            if not normalized or normalized in {"unknown", "not provided", "not_applicable", "product"}:
                return False
            return True

        missing_guidance_fields = [
            field for field in guidance_required_fields
            if not has_substantive_value(
                known_fields.get(field) or (formulation_category if field == "formulation_type" else None),
                field,
            )
        ]
        analysis["guidance_required_fields"] = guidance_required_fields
        analysis["guidance_missing_fields"] = missing_guidance_fields
        if missing_guidance_fields and not clarification_cap_hit:
            field = missing_guidance_fields[0]
            question = {
                "destination_country": "Which destination country should I check?",
                "product_type": "What type of product is it?",
                "formulation_type": "Which formulation category applies?",
                "protection_subject": "Are you seeking protection for the product, process, or both?",
                "invention_feature": "What is the main new feature or improvement?",
                "public_disclosure": "Has the product or process already been publicly disclosed or launched?",
                "formulation_name": "What is the formulation name?",
                "ingredients": "What are its main ingredients?",
                "traditional_use": "What traditional use is associated with it?",
            }[field]
            input_type, options = QUESTION_FIELD_SCHEMA[field]
            plan_item = {"field": field, "question": question, "input_type": input_type,
                         "options": options, "allow_multiple": input_type == "multi_select", "required": True}
            analysis.update({
                "needs_clarification": True,
                "question_plan": [plan_item],
                "clarification_fields": [{"field": field, "question": question}],
                "clarification_questions": [question],
                "clarification_question": question,
                "missing_fields": missing_guidance_fields,
                "required_fields": guidance_required_fields,
                "completed_clarifications": clarification_turns,
                "clarification_total": max(clarification_turns + len(missing_guidance_fields), 1),
            })
        else:
            analysis.update({
                "needs_clarification": False,
                "question_plan": [],
                "clarification_fields": [],
                "clarification_questions": [],
                "clarification_question": None,
                "missing_fields": [],
                "required_fields": guidance_required_fields,
                "completed_clarifications": clarification_turns,
                "clarification_total": 0,
            })
            if missing_guidance_fields:
                force_answer_caveat = True

    if clarification_cap_hit and analysis.get("needs_clarification"):
        analysis.update({
            "needs_clarification": False,
            "question_plan": [],
            "clarification_fields": [],
            "clarification_questions": [],
            "clarification_question": None,
            "missing_fields": [],
            "completed_clarifications": clarification_turns,
            "clarification_total": 0,
            "clarification_cap_hit": True,
        })
        force_answer_caveat = True

    selected_domains = set().union(*(spec[2] for spec in selected_area_specs)) if selected_area_specs else set()
    if selected_domains:
        analysis["domains"] = list(dict.fromkeys([*analysis.get("domains", []), *sorted(selected_domains)]))
    analysis["clarification_turns"] = clarification_turns
    retrieval_query = routed_question
    if selected_areas:
        retrieval_query += "\n\nRequested guidance areas: " + "; ".join(selected_areas)
    allowed_categories = supported_categories_for_domains(
        analysis["domains"], jurisdiction
    )
    retrieval_domain = selected_domains or (
        {"tkdl_prior_art"} if analysis["intent"] == "prior_art"
        else {"abs"} if analysis["intent"] == "abs_guidance"
        else None
    )
    results = [] if analysis["needs_clarification"] or not allowed_categories else retrieve(
        retrieval_query,
        top_k=5,
        jurisdiction=jurisdiction,
        allowed_categories=allowed_categories,
        allowed_domains=retrieval_domain,
    )
    route_has_corpus = (
        analysis["domains"] != ["unknown"]
        and allowed_categories is not None
        and bool(allowed_categories)
    )
    sufficient_evidence = (
        route_has_corpus
        and not analysis["needs_clarification"]
        and has_sufficient_evidence(retrieval_query, results)
    )

    if analysis["needs_clarification"]:
        answer = "\n".join(analysis["clarification_questions"])
        citations = []
        status = "needs_clarification"
        confidence = "low"
        safe_abstention = False
    elif not sufficient_evidence:
        # No model call for weak evidence: safe abstention is deterministic.
        if analysis["intent"] == "prior_art":
            answer = (
                f"No sufficient {jurisdiction.title()}-specific public prior-art evidence was retrieved. "
                f"{ABSTENTION_MESSAGE} "
                "Authoritative record-level TKDL evidence could not be retrieved. "
                "I cannot confirm whether this formulation is or is not listed."
            )
        elif analysis["intent"] == "export_regulatory_guidance":
            destination = analysis.get("known_fields", {}).get("destination_country")
            market = f" for {destination}" if destination else " for the requested destination market"
            answer = (
                f"I don't have sufficient authoritative export or destination-market sources{market} in the current knowledge base to answer this reliably. "
                f"{ABSTENTION_MESSAGE}"
            )
        else:
            answer = (
                f"Insufficient {jurisdiction.title()}-specific evidence was retrieved "
                f"for this question. {ABSTENTION_MESSAGE}"
            )
        citations = []
        status = "abstained"
        confidence = "low"
        safe_abstention = True
    else:
        try:
            answer = generate_answer(
                retrieval_query, results,
                domain="tkdl_prior_art" if analysis["intent"] == "prior_art" or "tkdl_prior_art" in selected_domains else None,
            )
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

    record_search_abstained = (
        analysis["intent"] == "prior_art"
        and analysis.get("prior_art_context", {}).get("search_requested")
        and not analysis["needs_clarification"]
    )
    if record_search_abstained:
        verified_evidence = [
            citation for citation in citations
            if citation.get("reference_valid") and citation.get("content_supported")
        ]
        citations = verified_evidence
        if verified_evidence:
            evidence_lines = [
                f'{citation["claim"]} [Document: {citation["document"]}; Page: {citation["page"]}]'
                for citation in verified_evidence
            ]
            answer = (
                "Potentially relevant public evidence was retrieved, but it does not establish a TKDL listing.\n\n"
                + "\n\n".join(evidence_lines)
                + "\n\nAuthoritative record-level TKDL evidence could not be retrieved. "
                "I cannot confirm whether this formulation is or is not listed. "
                "Formal patent examination is required for any patentability conclusion."
            )
        else:
            answer = (
                f"No sufficient {jurisdiction.title()}-specific public prior-art evidence was retrieved. "
                f"{ABSTENTION_MESSAGE} Authoritative record-level TKDL evidence could not be retrieved. "
                "I cannot confirm whether this formulation is or is not listed."
            )
        status = "abstained"
        confidence = "low"
        safe_abstention = True

    prior_art = None
    sources_for_display = results
    if analysis["intent"] == "prior_art":
        relevant_citations = [
            citation for citation in citations
            if citation.get("reference_valid") and citation.get("content_supported")
        ]
        cited_keys = {
            (citation["document"], citation["page"]) for citation in relevant_citations
        }
        sources_for_display = [
            result for result in results
            if (result["document"], result["page"]) in cited_keys
        ]
        if analysis["needs_clarification"]:
            indicator = "clarification_needed"
        elif record_search_abstained and relevant_citations:
            indicator = "record_search_abstained_with_public_evidence"
        elif relevant_citations:
            indicator = "potentially_relevant_public_evidence"
        else:
            indicator = "no_sufficient_evidence"
        prior_art = {
            "indicator": indicator,
            "relevant_evidence": [
                {
                    "document": citation["document"],
                    "page": citation["page"],
                    "source_organization": citation.get("source_organization"),
                    "official_url": citation.get("official_url"),
                    "official_listing_url": citation.get("official_listing_url"),
                    "official_pdf_url": citation.get("official_pdf_url"),
                    "claim": citation.get("claim"),
                    "citation_verified": citation.get("citation_verified", False),
                }
                for citation in relevant_citations
            ],
            "why_it_may_be_relevant": (
                "The cited public text supports the adjacent statement and may inform prior-art review; it does not establish a TKDL record or decide novelty."
                if relevant_citations else None
            ),
            "record_level_tkdl_search_available": False,
            "record_level_search_abstained": record_search_abstained,
            "record_level_note": (
                "This corpus contains public guidance, not record-level TKDL search results. It cannot establish that a formulation is or is not listed."
            ),
            "formal_patent_examination_required": True,
        }

    area_gaps = []
    for area_key, area_label, area_domains, _ in selected_area_specs:
        if area_key == "tkdl":
            covered = any(
                result.get("domain") == "tkdl_prior_art"
                or re.search(r"\b(?:tkdl|traditional knowledge|prior art)\b", result.get("text", ""), re.I)
                for result in results
            )
        else:
            covered = any(
                result.get("domain") in area_domains
                or (area_key == "ip" and result.get("category") == "Patent")
                for result in results
            )
        if not covered:
            area_gaps.append(f"No indexed source found for {area_label} — treat this as a gap, not an answer.")
    if area_gaps:
        answer += "\n\n" + "\n".join(area_gaps)
    if force_answer_caveat:
        answer = (
            "I’m proceeding after five clarification turns with the information available. "
            "Treat this as general guidance; missing details may change the result.\n\n"
            + answer
        )

    disclaimer = "This information is for general guidance only and does not constitute legal advice."
    if safe_abstention:
        answer += "\n\nFor case-specific advice, you can request review from a human IP facilitator."
    if general_guidance_requested and formulation_question:
        answer = (
            "General guidance only: this may not fit your specific formulation type.\n\n"
            + answer
        )
    elif category_label and not analysis.get("needs_clarification"):
        answer = f"Answering for: {category_label}\n\n{answer}"
    if language != "en":
        try:
            answer = translate_text(answer, language, source_language="English")
            disclaimer = translate_text(disclaimer, language, source_language="English")
            for item in analysis.get("question_plan", []):
                item["question"] = translate_text(item["question"], language, source_language="English")
            analysis["clarification_questions"] = [item["question"] for item in analysis.get("question_plan", [])]
            analysis["clarification_question"] = analysis["clarification_questions"][0] if analysis["clarification_questions"] else None
        except Exception:
            translation_status = "translation_unavailable_english_fallback"

    response_payload = {
        "status": status,
        "question": request.question,
        "jurisdiction": jurisdiction,
        "analysis": analysis,
        "language": language,
        "product_category": request.product_category,
        "answer": answer,
        "confidence": confidence,
        "citations": citations,
        "retrieved_sources": display_retrieved_sources(sources_for_display),
        **({"prior_art": prior_art} if prior_art is not None else {}),
        "safe_abstention": safe_abstention,
        "disclaimer": disclaimer,
        "translation_status": translation_status,
    }
    append_audit_event(
        "assistant_query", request.question, user_id=request.user_id,
        jurisdiction=jurisdiction,
        sources_cited=[item.get("document") for item in citations if item.get("reference_valid")],
    )
    return response_payload


@app.get("/v1/source-link")
def source_link(document: str):
    """Open a registered PDF when available, falling back to its listing page."""
    normalized_document = " ".join(document.split()).casefold()
    source = next(
        (metadata for title, metadata in TRUSTED_SOURCES.items()
         if " ".join(title.split()).casefold() == normalized_document),
        None,
    )
    if not source:
        raise HTTPException(status_code=404, detail="Source is not registered.")
    pdf_url = source.get("official_pdf_url")
    listing_url = source.get("official_listing_url")
    if pdf_url:
        return RedirectResponse(pdf_url, status_code=307)
    if listing_url:
        return RedirectResponse(listing_url, status_code=307)
    raise HTTPException(status_code=502, detail="The registered source links are currently unavailable.")
