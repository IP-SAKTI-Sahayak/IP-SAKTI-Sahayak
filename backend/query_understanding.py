"""Small, explainable query understanding for the curated Phase 4A corpus."""

import re


FORMULATION_RULES = (
    ("classical_medicine", ("classical ayurvedic", "classical ayurveda", "classical formulation")),
    ("proprietary_medicine", ("proprietary medicine", "proprietary ayurvedic")),
    ("new_non_classical_drug", ("new non-classical", "non classical drug", "new drug")),
    ("phytopharmaceutical", ("phytopharmaceutical", "phyto pharmaceutical")),
    ("ayurveda_aahar_nutraceutical", ("ayurveda aahar", "ayurveda-aahar", "nutraceutical")),
    ("cosmetic", ("cosmetic", "cosmeceutical", "beauty product", "skin care")),
)


def understand_query(question, jurisdiction):
    """Return conservative classification without inferring missing product facts."""
    text = question.lower()
    domains = []

    domain_keywords = {
        "patent": ("patent", "patentable", "patents act", "section 3", "invention"),
        "trademark": ("trademark", "trade mark", "brand registration"),
        "geographical_indication": ("geographical indication", "gi registration"),
        "copyright": ("copyright",),
        "design": ("industrial design", "design registration"),
        "plant_variety": ("plant variety", "plant breeder"),
        "regulation": ("regulation", "regulatory", "approval", "ayurveda-aahar", "aahar"),
        "abs": ("access and benefit sharing", "benefit sharing", "biological resource", "abs requirement"),
        "tkdl_prior_art": ("tkdl", "traditional knowledge", "prior art"),
    }
    for domain, keywords in domain_keywords.items():
        if any(keyword in text for keyword in keywords):
            domains.append(domain)

    formulation_category = "not_applicable"
    for category, keywords in FORMULATION_RULES:
        if any(keyword in text for keyword in keywords):
            formulation_category = category
            break

    product_reference = any(word in text for word in (
        "formulation", "product", "medicine", "drug", "ingredient", "composition",
    ))
    ambiguous_protection = (
        any(word in text for word in ("protect", "register", "protection"))
        and not domains
    )
    needs_clarification = ambiguous_protection or (
        product_reference and formulation_category == "not_applicable" and not domains
    )
    if needs_clarification:
        formulation_category = "unknown"

    if not domains:
        intent = "unknown"
    elif "tkdl_prior_art" in domains:
        intent = "prior_art"
    elif "patent" in domains:
        intent = "patentability"
    elif "abs" in domains:
        intent = "abs_guidance"
    elif "regulation" in domains:
        intent = "regulatory_guidance"
    else:
        intent = "information_request"

    clarification_question = None
    if needs_clarification:
        clarification_question = (
            "Are you asking about patent, trademark, regulatory approval, "
            "or another form of protection?"
        )

    return {
        "formulation_category": formulation_category,
        "jurisdiction": jurisdiction,
        "intent": intent,
        "domains": domains or ["unknown"],
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }


def supported_categories_for_domains(domains):
    """Map only corpus-supported domains to current metadata categories."""
    supported = set(domains) & {"patent", "tkdl_prior_art", "regulation"}
    if supported:
        return {"Patent", "Ayush Patent Guidelines"}
    return set()
