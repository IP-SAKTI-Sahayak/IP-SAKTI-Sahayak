"""Small, explainable query understanding for the curated Phase 4A corpus."""

import re


FORMULATION_RULES = (
    ("classical_medicine", (
        "classical ayurvedic", "classical ayurveda", "classical formulation",
        "classical medicine",
    )),
    ("proprietary_medicine", (
        "proprietary medicine", "proprietary ayurvedic", "proprietary formulation",
        "proprietary herbal", "proprietary",
    )),
    ("new_non_classical_drug", (
        "new non-classical", "non classical drug", "new drug", "newly developed",
        "new formulation", "non-classical formulation", "new / non-classical medicine",
    )),
    ("phytopharmaceutical", ("phytopharmaceutical", "phyto pharmaceutical")),
    ("ayurveda_aahar_nutraceutical", ("ayurveda aahar", "ayurveda-aahar", "nutraceutical")),
    ("cosmetic", ("cosmetic", "cosmeceutical", "beauty product", "skin care")),
)

COUNTRY_NAMES = (
    "United States", "United Kingdom", "South Korea", "Saudi Arabia", "United Arab Emirates",
    "Germany", "France", "Japan", "Australia", "Canada", "Brazil", "China", "Italy",
    "Spain", "Netherlands", "Singapore", "Malaysia", "Indonesia", "Thailand", "Nepal",
    "Bangladesh", "Sri Lanka", "Switzerland", "Sweden", "Norway", "Denmark", "Belgium",
    "Austria", "Portugal", "Ireland", "Mexico", "South Africa", "New Zealand", "India",
    "Pakistan", "Kenya", "Nigeria", "Egypt", "Turkey", "Türkiye", "Israel", "Qatar",
    "Oman", "Kuwait", "Russia", "Ukraine", "Poland", "Greece", "Vietnam", "Philippines",
)

QUESTION_FIELD_SCHEMA = {
    "formulation_type": ("single_select", ["Classical Ayurveda", "Proprietary formulation", "New / non-classical medicine", "Phytopharmaceutical", "Ayurveda-Aahara / nutraceutical", "Cosmetic", "Not sure"]),
    "classical_source": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "newly_developed": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "developed_by_org": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "standardized_extracts": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "product_category": ("single_select", ["Classical medicine", "Proprietary medicine", "New / non-classical medicine", "Phytopharmaceutical", "Ayurveda-Aahara / nutraceutical", "Cosmetic", "Not sure"]),
    "intended_product_type": ("single_select", ["Medicine", "Ayurveda-Aahara / Food", "Cosmetic", "Not sure"]),
    "product_type": ("single_select", ["Classical Ayurveda", "Proprietary formulation", "New / non-classical medicine", "Phytopharmaceutical", "Ayurveda-Aahara / nutraceutical", "Cosmetic", "Other", "Not sure"]),
    "intended_use": ("multi_select", ["General wellness", "Therapeutic / medicinal", "Preventive", "Nutritional / food", "Cosmetic", "Other", "Not sure"]),
    "intended_activity": ("single_select", ["Commercial sale", "Research", "Product development", "Trial / sample", "Other"]),
    "biological_resource": ("text", []),
    "associated_traditional_knowledge": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "ipr_planned": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "india_manufacturing_status": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "public_disclosure": ("yes_no_not_sure", ["Yes", "No", "Not sure"]),
    "protection_subject": ("single_select", ["Product", "Process", "Both", "Not sure"]),
    "protection_type": ("multi_select", ["Patent", "Trademark", "GI", "Copyright", "Design", "Trade secret", "Plant variety rights", "Other"]),
    "ip_type": ("multi_select", ["Patent", "Trademark", "GI", "Design", "Copyright", "Trade secret", "Plant variety rights", "Other", "Not sure"]),
    "guidance_areas": ("multi_select", ["Export requirements", "Market approval", "Product registration", "Labeling", "IP protection", "ABS compliance", "Traditional knowledge / TKDL", "Other"]),
    "traditional_use": ("multi_select", ["General wellness", "Digestive use", "Respiratory use", "Stress/sleep-related traditional use", "Skin-related traditional use", "Other", "Not sure"]),
    "origin_country": ("country", list(COUNTRY_NAMES)),
    "destination_country": ("country", list(COUNTRY_NAMES)),
    "jurisdiction": ("single_select", ["India", "International"]),
    "intended_claims": ("textarea", []),
    "regulatory_status": ("yes_no_not_sure", ["Approved in India", "Not approved in India", "Not sure"]),
    "formulation_name": ("text", []),
    "ingredients": ("textarea", []),
    "formulation_details": ("textarea", []),
    "invention_feature": ("textarea", []),
}

QUESTION_FIELD_ALIASES = {
    "resource_origin": "origin_country",
    "development_status": "newly_developed",
    "guidance_scope": "guidance_areas",
    "innovation_description": "invention_feature",
    "ip_scope": "protection_subject",
    "purpose": "intended_activity",
    "entity_context": "formulation_details",
}

ALLOWED_DOMAINS = {
    "patent", "trademark", "geographical_indication", "copyright", "design",
    "plant_variety", "regulation", "export", "abs", "tkdl_prior_art", "unknown",
}
ALLOWED_INTENTS = {
    "unknown", "information_request", "patentability", "abs_guidance", "prior_art",
    "export_regulatory_guidance", "regulatory_guidance",
}


def _countries(text):
    found = []
    for country in sorted(COUNTRY_NAMES, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(country)}\b", text, re.I)
        if match:
            found.append((match.start(), country))
    return sorted(found)


def _last_formulation_category(text):
    matches = []
    for category, keywords in FORMULATION_RULES:
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), text, re.I):
                matches.append((match.start(), category))
    return max(matches, default=(0, "not_applicable"))[1]


def _category_from_clarification(text):
    answers = re.findall(r"clarification answer field:\s*([a-z_]+)\s*=\s*([^\n]+)", text, re.I)
    for field, answer in reversed(answers):
        answer = answer.strip().lower()
        if field == "classical_source" and answer.startswith("yes"):
            return "classical_medicine"
        if field in {"newly_developed", "developed_by_org"} and answer.startswith("yes"):
            return "new_non_classical_drug"
        if field == "standardized_extracts" and answer.startswith("yes"):
            return "phytopharmaceutical"
        if field == "intended_product_type":
            if "cosmetic" in answer:
                return "cosmetic"
            if "aahara" in answer or "nutraceutical" in answer or "food" in answer:
                return "ayurveda_aahar_nutraceutical"
        if field in {"product_category", "formulation_type"}:
            category = _last_formulation_category(answer)
            if category != "not_applicable":
                return category
    return "not_applicable"


def _structured_category_info(text):
    """Prefer explicit guided answers over incidental description keywords."""
    values = {}
    for field in ("classical_source", "newly_developed", "developed_by_org", "product_category", "formulation_type"):
        matches = re.findall(
            rf"(?:clarification answer field:\s*)?{field}\s*[:=]\s*([^\n.;]+)",
            text,
            re.I,
        )
        if matches:
            values[field] = matches[-1].strip()
    selected = _last_formulation_category(
        values.get("formulation_type", values.get("product_category", ""))
    )
    if selected != "not_applicable":
        return selected, False
    classical = values.get("classical_source", "").lower()
    newly = values.get("newly_developed", values.get("developed_by_org", "")).lower()
    has_classical_clue = bool(re.search(r"\bclassical(?: ayurvedic)?\b", text, re.I))
    conflict = (
        classical in {"no", "not sure", "unsure", "unknown"} and has_classical_clue
    ) or (classical.startswith("yes") and newly.startswith("yes"))
    if conflict:
        return "unknown", True
    if classical.startswith("yes"):
        return "classical_medicine", False
    if newly.startswith("yes") and classical.startswith("no"):
        return "new_non_classical_drug", False
    return "not_applicable", False


def _clarification_answer(text, field):
    answers = re.findall(
        rf"clarification answer field:\s*{re.escape(field)}\s*=\s*([^\n]+)",
        text,
        re.I,
    )
    return answers[-1].strip() if answers else None


def _abs_context(question):
    """Expose only ABS facts explicitly mentioned in the user's wording."""
    text = question.lower()
    uses = []
    for label, cues in (
        ("research", ("research",)),
        ("bio-survey / bio-utilisation", ("bio-survey", "biosurvey", "bio-utilisation", "bio-utilization")),
        ("commercial utilisation", ("commercial utilisation", "commercial utilization", "commercial use", "commercially")),
        ("IPR involvement", ("ipr", "intellectual property", "patent", "patents")),
    ):
        if any(cue in text for cue in cues):
            uses.append(label)

    resource_cues = ("biological resource", "medicinal plant", "medicinal herb", "plant", "herb", "species")
    botanical_names = ("neem", "ashwagandha", "tulsi", "turmeric", "ginger", "amla", "triphala", "brahmi", "guduchi", "shatavari")
    resource_names = [name for name in botanical_names if re.search(rf"\b{re.escape(name)}\b", text)]
    resource_field = re.findall(r"(?:clarification answer field:\s*biological_resource\s*=|biological_resource\s*[:=])\s*([^\n.;]+)", text, re.I)
    biological_resource = resource_field[-1].strip() if resource_field else ", ".join(resource_names) or None
    resource_mentioned = bool(biological_resource) or any(cue in text for cue in resource_cues)
    india_origin_mentioned = bool(re.search(r"\bindian?\b|\bindia\b", text)) or bool(re.search(r"(?:clarification answer field:\s*|origin_country\s*[:=]\s*)origin_country\s*=\s*india\b|origin_country\s*[:=]\s*india\b", text))
    source_origin_mentioned = india_origin_mentioned or bool(re.search(r"origin_country\s*[:=]\s*\S", text)) or bool(re.search(r"clarification answer field:\s*origin_country\s*=\s*\S", text)) or bool(re.search(
        r"\b(?:sourced from|obtained from|origin(?:ated)? in|source country|from)\s+[a-z][a-z -]{1,30}",
        text,
    ))
    return {
        "biological_resource_mentioned": resource_mentioned,
        "biological_resource": biological_resource,
        "india_origin_mentioned": india_origin_mentioned,
        "source_origin_mentioned": source_origin_mentioned,
        "associated_traditional_knowledge_mentioned": "traditional knowledge" in text or "associated knowledge" in text,
        "intended_uses_mentioned": uses,
        "ipr_involvement_mentioned": any(cue in text for cue in ("ipr", "intellectual property", "patent", "patents")),
        "clarification_facts": [
            label for present, label in (
                (resource_mentioned, "identify the biological resource"),
                (source_origin_mentioned, "confirm the resource's source or origin"),
                (any(cue in text for cue in ("research", "bio-survey", "biosurvey", "bio-utilisation", "bio-utilization", "commercial", "product", "patent", "ipr")), "describe the intended activity"),
            ) if not present
        ],
    }


def _prior_art_context(question):
    """Describe prior-art search facts without claiming a database result."""
    text = question.lower()
    lookup_cues = (
        "already documented", "already known", "documented as", "documented in",
        "described in", "has this formulation been",
        "check whether", "search for", "prior art for", "listed in tkdl",
        "in tkdl", "traditional use", "traditional formulation",
    )
    known_identifiers = (
        "ashwagandha", "withania", "turmeric", "curcuma", "ginger", "zingiber",
        "tulsi", "ocimum", "neem", "azadirachta", "amla", "emblica", "triphala",
        "dashamula", "chyawanprash", "arogyavardhini", "brahmi", "bacopa",
        "guduchi", "tinospora", "shatavari", "asparagus racemosus",
    )
    formulation_reference = any(term in text for term in (
        "formulation", "herbal product", "ayurvedic preparation", "ayurvedic invention",
    ))
    search_requested = any(cue in text for cue in lookup_cues) or (
        formulation_reference and any(term in text for term in ("is there prior art", "prior-art search"))
    )
    identified_terms = [term for term in known_identifiers if term in text]
    ingredient_marker = re.search(
        r"\b(?:contains?|containing|comprising|ingredients?|composition|prepared from|made from|extract of)\s*[:\-]?\s*([a-z][a-z-]{3,})",
        text,
    )
    generic_terms = {
        "herbs", "herbal", "plants", "plant", "traditional", "traditionally",
        "formulation", "ingredients", "medicine", "product", "ayurvedic",
        "classical", "unknown", "unsure",
    }
    if ingredient_marker and ingredient_marker.group(1) not in generic_terms:
        identified_terms.append(ingredient_marker.group(1))
    details_supplied = bool(identified_terms) or bool(re.search(r"(?:formulation_name|ingredients)\s*[:=]\s*\S", text)) or bool(re.search(r"clarification answer field:\s*formulation_details\s*=\s*\S", text)) or bool(re.search(
        r"clarification answer field:\s*formulation_details\s*=\s*(?!not sure\b|unknown\b|unsure\b)\S",
        text,
    ))
    return {
        "search_requested": search_requested,
        "formulation_details_supplied": details_supplied,
        "identified_terms": sorted(set(identified_terms)),
        "clarification_facts": [] if details_supplied else [
            "provide ingredients or botanical names and the traditional use or description to compare"
        ],
        # The indexed corpus contains public guidance, not searchable TKDL records.
        "record_level_tkdl_search_available": False,
    }


def understand_query(question, jurisdiction):
    """Return conservative classification without inferring missing product facts."""
    text = question.lower()
    current_text = text.split("current user message:")[-1]
    domains = []

    domain_keywords = {
        "patent": ("patent", "patentable", "patents act", "section 3", "invention", "ipr"),
        "trademark": ("trademark", "trade mark", "brand registration"),
        "geographical_indication": ("geographical indication", "gi registration"),
        "copyright": ("copyright",),
        "design": ("industrial design", "design registration"),
        "plant_variety": ("plant variety", "plant breeder"),
        "regulation": ("regulation", "regulatory", "approval", "ayurveda-aahar", "aahar"),
        "export": ("export", "exporting", "exported", "market entry", "sell abroad", "sell it in", "destination country"),
        "abs": (
            "access and benefit sharing", "benefit sharing", "biological resource",
            "abs", "national biodiversity authority", "state biodiversity board",
            "biodiversity management committee", "biological diversity",
        ),
        "tkdl_prior_art": (
            "tkdl", "traditional knowledge", "prior art", "prior-art",
            "already documented", "already known", "documented as", "documented in",
            "traditional use", "traditional ayurvedic literature", "traditional formulation",
            "described in",
            "novelty of", "affect the novelty", "novelty concern",
        ),
    }
    current_domains = [
        domain for domain, keywords in domain_keywords.items()
        if any(keyword in current_text for keyword in keywords)
    ]
    context_text = text.split("current user message:")[0] if "current user message:" in text else ""
    context_domains = [
        domain for domain, keywords in domain_keywords.items()
        if any(keyword in context_text for keyword in keywords)
    ]
    domains = current_domains or context_domains

    formulation_category, structured_conflict = _structured_category_info(current_text)
    if formulation_category == "not_applicable" and not structured_conflict:
        formulation_category = _last_formulation_category(current_text)
    if formulation_category == "not_applicable" and not structured_conflict:
        formulation_category = _category_from_clarification(current_text)
    if formulation_category == "not_applicable" and not structured_conflict:
        formulation_category, structured_conflict = _structured_category_info(context_text)
    if formulation_category == "not_applicable" and not structured_conflict:
        formulation_category = _last_formulation_category(context_text)
    if formulation_category == "not_applicable" and not structured_conflict:
        formulation_category = _category_from_clarification(context_text)
    if structured_conflict:
        formulation_category = "unknown"

    product_reference = any(word in text for word in (
        "formulation", "product", "medicine", "drug", "ingredient", "composition",
        "food product", "herbal product",
    ))
    user_product_reference = bool(re.search(
        r"\b(?:my|our|this)\s+(?:ayurvedic\s+|herbal\s+)?(?:product|formulation|medicine)\b",
        text,
    ))
    protection_intent = any(word in text for word in (
        "protect", "register", "protection",
    ))
    clarification_questions = []
    clarification_fields = []
    explicitly_unsure = bool(re.search(r"clarification answer\s+\d+\s*:\s*(?:not sure|unsure|unknown)\b", current_text))

    def answered(field):
        return bool(re.search(rf"clarification answer field:\s*{re.escape(field)}\s*=", text))

    def ask(field, question):
        clarification_fields.append({"field": field, "question": question})
        clarification_questions.append(question)

    origin_country = None
    origin_match = re.search(r"\b(?:from|origin(?:ating)? in|sourced from)\s+(" + "|".join(re.escape(c) for c in COUNTRY_NAMES) + r")\b", text, re.I)
    if origin_match:
        origin_country = next((c for c in COUNTRY_NAMES if c.lower() == origin_match.group(1).lower()), origin_match.group(1))
    destination_country = None
    destination_match = re.search(r"\b(?:to|sell(?:ing)? (?:it )?in|market(?:ing)? in|destination(?: country)?\s*[:=])\s*(" + "|".join(re.escape(c) for c in COUNTRY_NAMES) + r")\b", text, re.I)
    if destination_match:
        destination_country = next((c for c in COUNTRY_NAMES if c.lower() == destination_match.group(1).lower()), destination_match.group(1))
    if not origin_country:
        origin_match = re.search(r"\bmanufactur(?:ed|ing)\s+in\s+(" + "|".join(re.escape(c) for c in COUNTRY_NAMES) + r")\b", text, re.I)
        if origin_match:
            origin_country = next((c for c in COUNTRY_NAMES if c.lower() == origin_match.group(1).lower()), origin_match.group(1))
    if not origin_country:
        origin_answer = re.findall(r"(?:clarification answer field:\s*origin_country\s*=|origin_country\s*[:=])\s*([^\n;]+)", text, re.I)
        if origin_answer:
            origin_country = next((c for c in COUNTRY_NAMES if c.lower() == origin_answer[-1].strip().lower()), origin_answer[-1].strip())
    if not destination_country:
        field_match = re.findall(r"(?:destination_country|clarification answer field:\s*destination_country\s*=)\s*[:=]?\s*([a-z ]+)", text, re.I)
        if field_match:
            candidate = field_match[-1].strip()
            destination_country = next((c for c in COUNTRY_NAMES if c.lower() == candidate.lower()), candidate[:1].upper() + candidate[1:] if candidate else None)

    export_requested = "export" in domains
    activity_known = bool(re.search(r"\b(?:commercial(?: sale| use)?|research|trial|sample|personal use|sale|sell|selling)\b", text))
    product_known = bool(re.search(r"\b(?:medicine|medicinal|formulation|product|drug|food|cosmetic)\b", text))

    prior_art_context = _prior_art_context(question)
    if (
        "tkdl_prior_art" in domains
        and prior_art_context["search_requested"]
        and not prior_art_context["formulation_details_supplied"]
        and not answered("formulation_details")
        and not explicitly_unsure
    ):
        ask("formulation_details", "What is the formulation name or its main ingredients?")

    # Only request a formulation category when it changes patent or regulatory
    # routing.  Factual statute/TKDL questions bypass this branch completely.
    if (
        product_reference
        and user_product_reference
        and formulation_category == "not_applicable"
        and not explicitly_unsure
        and "regulation" in domains
    ):
        ask("product_category", "What type of product is it: classical medicine, proprietary medicine, Ayurveda-Aahara/nutraceutical, cosmetic, or another category?")

    # Progressively classify a vague product with the shared formulation rules.
    vague_product = any(term in text for term in ("herbal product", "ayurvedic product", "my product"))
    if not explicitly_unsure and not domains and vague_product:
        if formulation_category == "not_applicable":
            if not answered("classical_source") and not re.search(r"\bclassical (?:ayurvedic )?(?:text|source|formulation)\b", text):
                ask("classical_source", "Is the formulation based on a classical Ayurvedic text?")
            if not answered("newly_developed") and not re.search(r"\bnewly developed\b", text):
                ask("newly_developed", "Was it newly developed by your organization?")
        if not answered("intended_product_type") and not re.search(r"\b(?:medicine|food|cosmetic|phytopharmaceutical|nutraceutical)\b", text):
            ask("intended_product_type", "Is it intended as a medicine, Ayurveda-Aahara/food, or cosmetic?")
        if formulation_category == "not_applicable" and not answered("standardized_extracts"):
            ask("standardized_extracts", "Does it use standardized plant or phytochemical extracts?")
        if not answered("intended_use") and not re.search(r"\b(?:used for|intended use|purpose|claim|treat(?:s|ment)?)\b", text):
            ask("intended_use", "What is the intended use or claim?")

    if "abs" in domains and (
        user_product_reference
        or re.search(r"\b(?:using|accessing|obtaining|access to|use of)\b", text)
    ):
        abs_facts = _abs_context(question)
        origin_known = (
            abs_facts["india_origin_mentioned"] if jurisdiction.lower() == "india"
            else abs_facts["source_origin_mentioned"]
        )
        if not abs_facts["biological_resource_mentioned"] and not answered("biological_resource") and not explicitly_unsure:
            ask("biological_resource", "What biological resource are you using?")
        if not origin_known and not answered("origin_country") and not explicitly_unsure:
            ask("origin_country", "Where was the biological resource obtained?" if jurisdiction.lower() == "india" else "Which country was the biological resource obtained from?")
        if not re.search(r"\b(?:research|product development|develop\w*.{0,25}product|commercial use|commercial sale|trial|sample)\b", text) and not answered("intended_activity") and not explicitly_unsure:
            ask("intended_activity", "What is the intended activity: research, product development, or commercial use?")
        if not answered("associated_traditional_knowledge") and not re.search(r"\b(?:associated traditional knowledge|traditional knowledge involved|no associated traditional knowledge)\b", text) and not explicitly_unsure:
            ask("associated_traditional_knowledge", "Is associated traditional knowledge involved?")
        if not answered("ipr_planned") and not re.search(r"\b(?:ipr planned|intellectual property application planned|no ipr|no intellectual property application)\b", text) and not explicitly_unsure:
            ask("ipr_planned", "Is an intellectual property application planned?")

    if "tkdl_prior_art" in domains and not prior_art_context["search_requested"] and not prior_art_context["formulation_details_supplied"] and not answered("formulation_details") and not explicitly_unsure:
        ask("formulation_details", "What is the formulation name or its main ingredients?")

    if "tkdl_prior_art" in domains and prior_art_context["search_requested"] and not re.search(r"\b(?:traditional(?:_| )+(?:use|purpose)|traditionally used|used for|for treating)\b", text) and not answered("traditional_use") and not explicitly_unsure:
        ask("traditional_use", "What traditional use is associated with it, if known?")

    patent_clarification_flow = "patent" in domains and user_product_reference and (
        formulation_category == "not_applicable" or answered("formulation_type") or answered("protection_subject")
    )
    if patent_clarification_flow and formulation_category == "not_applicable" and not answered("formulation_type") and not explicitly_unsure:
        ask("formulation_type", "Is it a classical Ayurvedic formulation, a proprietary formulation, or a newly developed formulation?")
    if patent_clarification_flow and not answered("protection_subject") and not explicitly_unsure:
        ask("protection_subject", "Are you seeking protection for the product, the process, or both?")
    if patent_clarification_flow and not answered("invention_feature") and not re.search(r"\b(?:new feature|novel|inventive step|improvement|new process)\b", text) and not explicitly_unsure:
        ask("invention_feature", "What is the main new feature or improvement you want to protect?")
    if patent_clarification_flow and not answered("public_disclosure") and not re.search(r"\b(?:publicly disclosed|public disclosure|already launched|already published)\b", text) and not explicitly_unsure:
        ask("public_disclosure", "Has this product or process already been publicly disclosed or launched?")

    if "regulation" in domains and user_product_reference and not answered("intended_use") and not re.search(r"\b(?:intended use|used for|purpose|claim)\b", text):
        ask("intended_use", "What is the product intended to be used for, and what claims will you make?")

    if "export" in domains and "patent" not in domains and "abs" not in domains:
        if not destination_country and not answered("destination_country"):
            ask("destination_country", "Which country are you planning to export to?")
        if not product_known and not answered("product_type"):
            ask("product_type", "What type of product or formulation are you exporting?")
        elif product_known and formulation_category == "not_applicable" and not answered("formulation_type"):
            ask("formulation_type", "What type of medicine or Ayurvedic formulation is it?")
        if not activity_known and not answered("intended_activity"):
            ask("intended_activity", "Is this for commercial sale, research, a trial or sample, or another purpose?")
        if product_known and not re.search(r"\b(?:manufactur(?:ed|ing)|already made|approved in India|not yet made)\b", text) and not answered("india_manufacturing_status") and not re.search(r"india_manufacturing_status\s*[:=]\s*\S", text):
            ask("india_manufacturing_status", "Is the product already manufactured or approved in India?")
        if not answered("guidance_areas") and not re.search(r"\b(?:labeling|product registration|destination-country approval|export requirements|market approval)\b", text):
            ask("guidance_areas", "What export or market-entry guidance do you need?")

    # A broad request to "protect" a product needs a legal route, but does not
    # justify asking a generic questionnaire or unrelated regulatory questions.
    if protection_intent and not domains:
        if product_reference and formulation_category == "not_applicable":
            if not clarification_questions:
                ask("formulation_type", "What type of Ayurveda product or formulation is it?")
        ask("protection_type", "What type of protection are you looking for, such as a patent, trademark, GI, copyright, or design?")

    # The label "food product" alone does not identify a regulatory route.
    if "regulation" in domains and "food product" in text and formulation_category == "not_applicable":
        clarification_questions = [
            question for question in clarification_questions
            if not question.startswith("What type of product")
        ]
        clarification_fields = [
            item for item in clarification_fields if item["field"] != "product_category"
        ]
        ask("product_category", "Which food-product category applies (for example, Ayurveda-Aahara, nutraceutical, conventional food, or another category)?")

    clarification_questions = clarification_questions[:5]
    clarification_fields = clarification_fields[:5]
    needs_clarification = bool(clarification_questions)
    if needs_clarification and formulation_category == "not_applicable":
        formulation_category = "unknown"

    explicit_prior_art_question = any(term in text for term in (
        "tkdl", "prior art", "prior-art", "already documented", "already known",
        "documented as", "documented in", "described in", "traditional use",
        "novelty", "affect the novelty",
    ))
    if not domains:
        intent = "unknown"
    elif "tkdl_prior_art" in domains and (
        explicit_prior_art_question or "abs" not in domains
    ):
        intent = "prior_art"
    elif "abs" in domains:
        intent = "abs_guidance"
    elif "tkdl_prior_art" in domains:
        intent = "prior_art"
    elif "patent" in domains:
        intent = "patentability"
    elif "export" in domains:
        intent = "export_regulatory_guidance"
    elif "regulation" in domains:
        intent = "regulatory_guidance"
    else:
        intent = "information_request"

    field_options = {
        "formulation_type": ["Classical Ayurveda", "Proprietary formulation", "New / non-classical medicine", "Phytopharmaceutical", "Ayurveda-Aahara / nutraceutical", "Cosmetic", "Not sure"],
        "classical_source": ["Yes", "No", "Not sure"],
        "newly_developed": ["Yes", "No", "Not sure"],
        "developed_by_org": ["Yes", "No", "Not sure"],
        "standardized_extracts": ["Yes", "No", "Not sure"],
        "product_category": ["Classical medicine", "Proprietary medicine", "New / non-classical medicine", "Phytopharmaceutical", "Ayurveda-Aahara / nutraceutical", "Cosmetic", "Not sure"],
        "intended_product_type": ["Medicine", "Ayurveda-Aahara / Food", "Cosmetic", "Not sure"],
        "product_type": ["Ayurvedic medicine", "Food / Ayurveda-Aahara", "Cosmetic", "Other", "Not sure"],
        "intended_use": ["General wellness", "Therapeutic / medicinal", "Preventive", "Nutritional / food", "Cosmetic", "Other", "Not sure"],
        "intended_activity": ["Commercial sale", "Research", "Trial / sample", "Other"],
        "associated_traditional_knowledge": ["Yes", "No", "Not sure"],
        "ipr_planned": ["Yes", "No", "Not sure"],
        "india_manufacturing_status": ["Yes", "No", "Not sure"],
        "public_disclosure": ["Yes", "No", "Not sure"],
        "protection_subject": ["Product", "Process", "Both", "Not sure"],
        "protection_type": ["Patent", "Trademark", "GI", "Copyright", "Design", "Trade secret", "Plant variety rights", "Other"],
        "guidance_areas": ["Export requirements", "Market approval", "Product registration", "Labeling", "IP protection", "Other"],
        "traditional_use": ["General wellness", "Digestive use", "Respiratory use", "Stress/sleep-related traditional use", "Skin-related traditional use", "Other", "Not sure"],
        "origin_country": list(COUNTRY_NAMES),
        "destination_country": list(COUNTRY_NAMES),
    }
    question_plan = []
    for item in clarification_fields:
        field = item["field"]
        multi_select_fields = {"guidance_areas", "intended_use", "protection_type", "traditional_use"}
        text_fields = {"formulation_details", "invention_feature"}
        yes_no_fields = {
            "classical_source", "newly_developed", "developed_by_org",
            "standardized_extracts", "associated_traditional_knowledge",
            "ipr_planned", "india_manufacturing_status", "public_disclosure",
        }
        input_type = (
            "country" if field in {"destination_country", "origin_country"}
            else "multi_select" if field in multi_select_fields
            else "yes_no_not_sure" if field in yes_no_fields
            else "single_select" if field in field_options
            else "textarea" if field in text_fields
            else "text"
        )
        question_plan.append({
            **item,
            "input_type": input_type,
            "options": field_options.get(field, []),
            "allow_multiple": input_type == "multi_select",
            "required": True,
        })
    answered_fields = sorted(set(re.findall(r"clarification answer field:\s*([a-z_]+)\s*=", current_text)))
    def supplied_field(field):
        values = re.findall(
            rf"(?:clarification answer field:\s*)?{re.escape(field)}\s*[:=]\s*([^\n]+)",
            text,
            re.I,
        )
        return values[-1].strip() if values else _clarification_answer(text, field)

    guidance_areas_value = supplied_field("guidance_areas")
    guidance_areas = (
        [area.strip() for area in re.split(r"\s*[;,|]\s*", guidance_areas_value) if area.strip()]
        if guidance_areas_value else None
    )
    known_fields = {
        "formulation_type": formulation_category if formulation_category not in {"not_applicable", "unknown"} else None,
        "jurisdiction": jurisdiction,
        "origin_country": origin_country,
        "destination_country": destination_country,
        "product_type": supplied_field("product_type") or ("medicine" if re.search(r"\bmedicine\b", text) else "product" if re.search(r"\bproduct\b", text) else None),
        "guidance_areas": guidance_areas,
        "protection_subject": supplied_field("protection_subject"),
        "invention_feature": supplied_field("invention_feature"),
        "public_disclosure": supplied_field("public_disclosure"),
        "intended_activity": supplied_field("intended_activity") or ("commercial" if re.search(r"\b(?:sale|sell|selling|commercial)\b", text) else "research" if re.search(r"\bresearch\b", text) else "product development" if re.search(r"develop\w*.{0,25}product", text) else None),
        "biological_resource": _abs_context(question).get("biological_resource"),
        "ip_type": next((domain for domain in ("patent", "trademark", "design", "copyright") if domain in domains), None),
        "associated_traditional_knowledge": _clarification_answer(text, "associated_traditional_knowledge") or (
            "No" if re.search(r"\bno associated traditional knowledge\b", text)
            else "Yes" if re.search(r"\b(?:associated traditional knowledge|traditional knowledge involved)\b", text)
            else None
        ),
        "ipr_planned": _clarification_answer(text, "ipr_planned") or (
            "No" if re.search(r"\b(?:no ipr|no intellectual property application)\b", text)
            else "Yes" if re.search(r"\b(?:ipr planned|patent planned|intellectual property application planned)\b", text)
            else None
        ),
        "india_manufacturing_status": "Yes" if re.search(r"\b(?:manufactured|approved) in India\b", text) else None,
        "formulation_name": supplied_field("formulation_name") or next(iter(re.findall(r"(?:formulation_name|clarification answer field:\s*formulation_name)\s*[:=]\s*([^\n.;]+)", text, re.I)), None),
        "ingredients": supplied_field("ingredients"),
        "traditional_use": next(iter(re.findall(r"traditional(?:_| )use\s*[:=]\s*([^\n.;]+)", text, re.I)), None),
        "export": export_requested,
        "abs_relevant": "abs" in domains,
        "traditional_knowledge_relevant": "tkdl_prior_art" in domains,
        "regulatory_relevant": "regulation" in domains or export_requested,
        "classification_confidence": next(iter(re.findall(r"classification confidence\s*[:=]\s*(high|medium|low)", text, re.I)), None),
    }
    question_count = len(answered_fields) + len(question_plan)
    result = {
        "formulation_category": formulation_category,
        "jurisdiction": jurisdiction,
        "intent": intent,
        "primary_intent": intent,
        "goal": intent,
        "confidence": "high" if domains and not needs_clarification else "medium" if domains else "low",
        "secondary_intents": [d for d in domains if d != intent],
        "domains": domains or ["unknown"],
        "needs_clarification": needs_clarification,
        "clarification_questions": clarification_questions,
        "clarification_fields": clarification_fields,
        "question_plan": question_plan,
        "known_fields": known_fields,
        "missing_fields": [item["field"] for item in question_plan],
        "required_fields": [item["field"] for item in question_plan],
        "completed_clarifications": len(answered_fields),
        "clarification_total": max(question_count, 1) if needs_clarification else 0,
        # Kept for clients using the original single-question response field.
        "clarification_question": clarification_questions[0] if clarification_questions else None,
    }
    if "abs" in domains:
        result["abs_context"] = _abs_context(question)
    if "tkdl_prior_art" in domains:
        result["prior_art_context"] = prior_art_context
    return result


def supported_categories_for_domains(domains, jurisdiction="india"):
    """Return categories present in the selected corpus for the detected domain.

    ``None`` is reserved for callers that explicitly request unscoped retrieval.
    An empty set means no safe corpus route is available, so retrieval must
    return no evidence instead of broadening.
    """
    domains = set(domains) - {"unknown"}
    if not domains:
        return set()

    if jurisdiction.lower() == "international":
        category_map = {
            "patent": {"Patent", "Intellectual Property"},
            "trademark": {"Trademark", "Intellectual Property"},
            "design": {"Design", "Intellectual Property"},
            "geographical_indication": {"Intellectual Property"},
            "copyright": {"Intellectual Property"},
            "abs": {"Biodiversity", "Access and Benefit Sharing"},
            "tkdl_prior_art": {"Traditional Knowledge"},
        }
    else:
        # Unsupported India domains deliberately return no results.
        category_map = {
            "patent": {"Patent", "Ayush Patent Guidelines"},
            "tkdl_prior_art": {"Patent", "Ayush Patent Guidelines"},
            "regulation": {"Ayush Patent Guidelines"},
            "abs": {"Access and Benefit Sharing"},
        }

    categories = set()
    for domain in domains:
        categories.update(category_map.get(domain, set()))
    return categories


def validate_gemini_plan(payload, base_analysis, source_text):
    """Validate Gemini's plan and attach only application-owned question controls."""
    if not isinstance(payload, dict):
        return None
    intent = payload.get("primary_intent")
    domains = payload.get("domains")
    plan = payload.get("question_plan")
    if intent not in ALLOWED_INTENTS or not isinstance(domains, list) or not domains:
        return None
    aliases_domain = {"international_export": "export", "export_regulatory_guidance": "export"}
    domains = [aliases_domain.get(value, value) for value in domains]
    if any(value not in ALLOWED_DOMAINS for value in domains):
        return None
    required_domain = {"patentability": "patent", "abs_guidance": "abs", "prior_art": "tkdl_prior_art", "export_regulatory_guidance": "export", "regulatory_guidance": "regulation"}.get(intent)
    if required_domain and required_domain not in domains:
        return None
    if not isinstance(plan, list) or len(plan) > 5:
        return None
    source = (source_text or "").lower()
    known = dict(base_analysis.get("known_fields") or {})
    for field, value in re.findall(r"clarification answer field:\s*([a-z_]+)\s*=\s*([^\n]+)", source_text or "", re.I):
        canonical = QUESTION_FIELD_ALIASES.get(field, field)
        if canonical in QUESTION_FIELD_SCHEMA and value.strip():
            known[canonical] = value.strip()
    # Accept new facts only if the value is directly supported by supplied text.
    facts = payload.get("known_facts", {})
    if not isinstance(facts, dict):
        return None
    for key, value in facts.items():
        field = QUESTION_FIELD_ALIASES.get(key, key)
        if field not in QUESTION_FIELD_SCHEMA or (value is not None and not isinstance(value, (str, int, float, bool))):
            return None
        value_text = str(value).strip()
        if value_text and value_text.lower() in source:
            known[field] = value
    normalized = []
    seen = set()
    for item in plan:
        if not isinstance(item, dict):
            return None
        field = QUESTION_FIELD_ALIASES.get(item.get("field"), item.get("field"))
        question = item.get("question")
        if field not in QUESTION_FIELD_SCHEMA or not isinstance(question, str) or not question.strip() or field in seen:
            return None
        seen.add(field)
        value = known.get(field)
        if field == "product_type" and str(value or "").strip().lower() in {"product", "herbal product", "unknown", "not sure"}:
            value = None
        if value is not None and str(value).strip().lower() not in {"", "unknown", "not sure", "unsure", "not_applicable"}:
            continue
        input_type, options = QUESTION_FIELD_SCHEMA[field]
        normalized.append({"field": field, "question": question.strip(), "input_type": input_type, "options": options, "allow_multiple": input_type == "multi_select", "required": True})
    if len(normalized) > 5:
        return None
    result = dict(base_analysis)
    result.update({
        "intent": intent, "primary_intent": intent,
        "goal": payload.get("goal") if isinstance(payload.get("goal"), str) else intent,
        "secondary_intents": [x for x in payload.get("secondary_intents", []) if x in ALLOWED_INTENTS] if isinstance(payload.get("secondary_intents", []), list) else [],
        "domains": domains, "known_fields": known,
        "needs_clarification": bool(normalized), "question_plan": normalized,
        "clarification_fields": [{"field": item["field"], "question": item["question"]} for item in normalized],
        "clarification_questions": [item["question"] for item in normalized],
        "missing_fields": [item["field"] for item in normalized],
        "required_fields": [item["field"] for item in normalized],
        "clarification_question": normalized[0]["question"] if normalized else None,
        "planner": "gemini",
    })
    result["clarification_total"] = max(result.get("completed_clarifications", 0) + len(normalized), 1) if normalized else 0
    return result
