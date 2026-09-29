"""Conservative structured workflows for IP-SAKTI Sahayak.

These helpers organise facts for review. They intentionally do not make legal,
regulatory or ABS-compliance determinations without authoritative evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re

from backend.query_understanding import FORMULATION_RULES, understand_query


FORMULATION_LABELS = {
    "classical_medicine": "Classical Ayurveda",
    "proprietary_medicine": "Proprietary formulation",
    "new_non_classical_drug": "New / non-classical medicine",
    "phytopharmaceutical": "Phytopharmaceutical",
    "ayurveda_aahar_nutraceutical": "Ayurveda-Aahara / Nutraceutical",
    "cosmetic": "Cosmetic",
}


def formulation_intake(
    product_type: str,
    source_text: str,
    intended_question: str,
    *,
    classical_source: str = "",
    newly_developed: str = "",
):
    """Classify only explicit formulation clues, without making legal conclusions."""
    product_type = product_type.strip()
    source_text = source_text.strip()
    intended_question = intended_question.strip()
    # Query understanding owns the canonical category rules. Keep the user's
    # question out of the classification signal so a question cannot determine
    # the product category by merely mentioning one.
    description = " ".join(value for value in (product_type, source_text) if value)
    product_type_key = understand_query(product_type, "india")["formulation_category"]
    category_key = (
        product_type_key if product_type_key != "not_applicable"
        else understand_query(description, "india")["formulation_category"]
    )
    classical_answer = classical_source.strip().lower()
    developed_answer = newly_developed.strip().lower()
    classical_clue = bool(re.search(r"\bclassical(?: ayurvedic)?\b", description, re.I))
    weak_classical_label = bool(re.search(r"\b(?:classical ayurvedic|traditional ayurvedic)\b", description, re.I)) and not bool(re.search(r"\b(?:based on|made according to|follows?|as described in|according to)\b", description, re.I))
    classification_conflict = product_type_key == "not_applicable" and ((
        classical_answer in {"no", "not sure", "unsure", "unknown"} and classical_clue
    ) or (classical_answer == "yes" and developed_answer == "yes"))
    if classification_conflict:
        category_key = "not_applicable"
    elif category_key == "classical_medicine" and classical_answer != "yes" and product_type_key != "classical_medicine" and weak_classical_label:
        category_key = "not_applicable"
    elif product_type_key == "not_applicable":
        if classical_answer == "yes":
            category_key = "classical_medicine"
        elif developed_answer == "yes" and classical_answer == "no":
            category_key = "new_non_classical_drug"
        elif developed_answer == "yes" and classical_answer in {"not sure", "unsure", "unknown"} and not classical_clue:
            category_key = "new_non_classical_drug"
    category = FORMULATION_LABELS.get(category_key)
    missing = []
    if not product_type:
        missing.append("product type")
    if not source_text:
        missing.append("key source or formulation text")
    if not intended_question:
        missing.append("intended question")
    if not category:
        missing.append("a formulation clue (such as classical, proprietary, non-classical, phytopharmaceutical, Ayurveda-Aahara/nutraceutical, or cosmetic)")
        return {
            "classification": "Needs clarification",
            "category": "needs_clarification",
            "classification_status": "needs_clarification",
            "reason": (
                "The structured answers conflict with the description, so the formulation cannot be classified reliably yet."
                if classification_conflict else
                "The supplied details do not identify a formulation category clearly enough to classify it."
            ),
            "confidence": "low",
            "needs_clarification": True,
            "clarification_question": (
                "Your answer about whether this follows a classical Ayurvedic source conflicts with the description. Does it follow a classical text?"
                if classification_conflict else
                "Could you share the product type and a source or description that indicates whether it is classical, proprietary, non-classical, phytopharmaceutical, Ayurveda-Aahara/nutraceutical, or a cosmetic?"
            ),
            "missing_information": missing,
            "next_step": "Add the requested details and classify again.",
            "disclaimer": "This identifies a formulation type from the supplied information only; it is not legal or regulatory advice.",
        }

    matched_rule = next(
        (keywords for key, keywords in FORMULATION_RULES if key == category_key), ()
    )
    evidence = next((term for term in matched_rule if term in description.lower()), None)
    reason = (
        f'The supplied description includes “{evidence}”, which is an explicit clue for this formulation category.'
        if evidence
        else "The supplied product details explicitly identify this formulation category."
    )
    return {
        "classification": category,
        "category": category_key,
        "classification_status": "provisional",
        "reason": reason,
        "confidence": "high",
        "needs_clarification": bool(missing),
        "clarification_question": None,
        "missing_information": missing,
        "next_step": (
            "Continue with Sahayak for evidence-backed IPR or regulatory guidance."
            if not missing else "The formulation is identified, but provide the missing details before continuing."
        ),
        "disclaimer": "This identifies a formulation type from the supplied information only; it is not legal or regulatory advice.",
    }


def abs_readiness(resource: str, access_origin: str, commercial_activity: str):
    """Report completeness only, avoiding a fabricated ABS assessment."""
    fields = {
        "biological resource": resource,
        "access and origin": access_origin,
        "intended activity": commercial_activity,
    }
    missing = [label for label, value in fields.items() if not value.strip()]
    return {
        "status": "information_incomplete" if missing else "ready_for_evidence_review",
        "missing_information": missing,
        "next_step": (
            "A legal or ABS determination requires the applicable authority, "
            "resource origin and authoritative source evidence. This tool does not "
            "determine whether permission, approval or benefit sharing is required."
        ),
        "disclaimer": "ABS obligations depend on facts, applicable law and authoritative review.",
    }


def tkdl_intake(claim: str, patent_context: str, jurisdiction: str):
    return {
        "status": "ready_for_query" if claim.strip() and patent_context.strip() else "needs_clarification",
        "intent": "prior_art",
        "domain": "tkdl_prior_art",
        "jurisdiction": jurisdiction,
        "query_prompt": (
            f"TKDL / prior-art context: {patent_context.strip()}. "
            f"Claim or traditional-knowledge reference: {claim.strip()}. "
            f"Jurisdiction: {jurisdiction}."
        ).strip(),
        "scope_note": (
            "A TKDL or prior-art conclusion requires the relevant authoritative "
            "record. This corpus contains public guidance, not record-level TKDL "
            "search results, and cannot confirm whether a formulation is or is not listed."
        ),
        "record_level_tkdl_search_available": False,
        "formal_patent_examination_required": True,
    }


def escalation_brief(question: str, reason: str):
    return {
        "status": "brief_prepared",
        "brief": {
            "issue": question.strip(),
            "reason": reason.strip(),
            "recommended_reviewer": "Qualified IP, regulatory or ABS facilitator",
        },
        "delivery_status": "not_sent",
        "notice": "No case was created and no request was transmitted.",
    }


def append_audit_event(
    event: str,
    text: str = "",
    *,
    user_id: str = "anonymous",
    jurisdiction: str | None = None,
    sources_cited: list[str] | None = None,
    escalation_triggered: bool = False,
    delivery_status: str | None = None,
    delivery_error: str | None = None,
    timestamp: str | None = None,
):
    """Append a local JSONL audit record for query/escalation traceability."""
    directory = Path("data/runtime")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user_id": user_id or "anonymous",
        "query": (text or None) if event in {"assistant_query", "escalation_brief"} else None,
        "jurisdiction": jurisdiction,
        "sources_cited": sources_cited or [],
        "escalation_triggered": bool(escalation_triggered),
        "content_hash": sha256(text.encode("utf-8")).hexdigest() if text else None,
    }
    if event == "escalation_brief":
        payload["delivery_status"] = delivery_status or "not_configured"
        payload["delivery_error"] = delivery_error
    with (directory / "audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return payload
