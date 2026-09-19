"""Conservative structured workflows for IP-SAKTI Sahayak.

These helpers organise facts for review. They intentionally do not make legal,
regulatory or ABS-compliance determinations without authoritative evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json


FORMULATION_LABELS = {
    "classical": "Classical Ayurveda medicine",
    "proprietary": "Patent or proprietary Ayurveda medicine",
    "new": "New or non-classical drug",
    "phyto": "Phytopharmaceutical",
    "aahar": "Ayurveda-Aahar / nutraceutical",
    "cosmetic": "Cosmetic",
}


def formulation_intake(product_type: str, source_text: str, intended_question: str):
    """Return a transparent triage result, never a regulatory conclusion."""
    text = f"{product_type} {source_text} {intended_question}".lower()
    category = next(
        (key for key in FORMULATION_LABELS if key in text),
        "unknown",
    )
    missing = []
    if not source_text.strip():
        missing.append("the authoritative formulation text or source reference")
    if not intended_question.strip():
        missing.append("the IPR, regulatory, ABS or prior-art question")
    return {
        "classification": FORMULATION_LABELS.get(category, "Insufficient information"),
        "classification_status": "provisional" if category != "unknown" else "needs_clarification",
        "missing_information": missing,
        "next_step": (
            "Ask Sahayak with the structured product details. The answer will be "
            "limited to evidence available in the active jurisdiction corpus."
        ),
        "disclaimer": "This triage is informational and is not a regulatory classification.",
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
        "query_prompt": (
            f"TKDL / prior-art context: {patent_context.strip()}. "
            f"Claim or traditional-knowledge reference: {claim.strip()}. "
            f"Jurisdiction: {jurisdiction}."
        ).strip(),
        "scope_note": (
            "A TKDL or prior-art conclusion requires the relevant authoritative "
            "record. The current corpus may abstain where that record is unavailable."
        ),
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


def append_audit_event(event: str, text: str = ""):
    """Write a minimum, non-PII operational audit event for local deployments."""
    directory = Path("data/runtime")
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "content_hash": sha256(text.encode("utf-8")).hexdigest() if text else None,
    }
    with (directory / "audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return payload
