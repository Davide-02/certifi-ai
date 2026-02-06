"""
Document Family Classifier - Updated for Contractor / SOW / LoE
"""

from typing import Dict, Any
from enum import Enum
import re


class DocumentFamily(str, Enum):
    IDENTITY = "identity"
    DRIVING_LICENSE = "driving_license"
    CONTRACT = "contract"
    CERTIFICATE = "certificate"
    FINANCIAL = "financial"
    CORPORATE = "corporate"
    UNKNOWN = "unknown"


class DocumentSubtype(str, Enum):
    """Subtypes for more precise classification"""

    # Contract subtypes
    ENGAGEMENT_LETTER = "engagement_letter"
    STATEMENT_OF_WORK = "statement_of_work"
    INDEPENDENT_CONTRACTOR_AGREEMENT = "independent_contractor_agreement"
    PROFESSIONAL_SERVICES_AGREEMENT = "professional_services_agreement"
    SERVICE_AGREEMENT = "service_agreement"
    NDA = "nda"
    CONTRACT_GENERIC = "contract_generic"

    # Certificate subtypes
    DIPLOMA = "diploma"
    CERTIFICATE_OF_ENGAGEMENT = "certificate_of_engagement"
    CERTIFICATE_GENERIC = "certificate_generic"

    # Financial subtypes
    INVOICE = "invoice"
    PAYSLIP = "payslip"
    BANK_STATEMENT = "bank_statement"

    # Identity subtypes
    ID_CARD = "id_card"
    PASSPORT = "passport"

    # Generic
    UNKNOWN = "unknown"


class FamilyClassifier:
    """
    Classifies documents into families with semantic co-occurrence patterns

    NEW FEATURES:
    1. Semantic co-occurrence: Detects combinations of signals (e.g., "engagement letter" + "service provider" + "fees")
    2. Multi-level classification: Family + Subtype for precise policy application
    3. Adaptive confidence: Adjusts confidence based on signal strength and co-occurrence
    """

    # Semantic co-occurrence patterns - combinations that strongly indicate document type
    # Format: (pattern_regex, required_keywords, boost_score)
    SEMANTIC_CO_OCCURRENCE = {
        DocumentFamily.CONTRACT: [
            # Engagement Letter: "engagement letter" + "service provider" + "fees"
            (
                r"engagement\s+letter.*?(?:service\s+provider|fees\s+charged|services\s+provided)",
                ["engagement letter", "service provider", "fees"],
                0.25,  # Strong boost for this combination
            ),
            # SOW: "statement of work" + "client" + "contractor"
            (
                r"statement\s+of\s+work.*?(?:client|contractor|services)",
                ["statement of work", "client", "contractor"],
                0.20,
            ),
            # Contractor Agreement: "independent contractor" + "client" + "effective date"
            (
                r"independent\s+contractor.*?(?:client|effective\s+date|agreement)",
                ["independent contractor", "client", "effective date"],
                0.20,
            ),
        ],
        DocumentFamily.FINANCIAL: [
            # Invoice: "invoice" + "total" + "vat" or "iva"
            (r"invoice.*?(?:total|vat|iva|amount)", ["invoice", "total", "vat"], 0.20),
        ],
        DocumentFamily.CERTIFICATE: [
            # Diploma: "diploma" + "university" + "cfu" or "credits"
            (
                r"diploma.*?(?:university|universit|cfu|credits)",
                ["diploma", "university", "cfu"],
                0.20,
            ),
        ],
    }

    FAMILY_PATTERNS = {
        DocumentFamily.IDENTITY: {
            "keywords": [
                "carta d'identità",
                "carta di identità",
                "passaporto",
                "passport",
                "documento identità",
                "codice fiscale",
                "data di nascita",
            ],
            "structural": [
                r"[A-Z0-9<]{25,}",  # MRZ
                r"nome\s*:?\s*[A-Z]",
                r"cognome\s*:?\s*[A-Z]",
            ],
            "min_matches": 2,
        },
        DocumentFamily.DRIVING_LICENSE: {
            "keywords": [
                "patente di guida",
                "patente",
                "repubblica italiana",
                "a1",
                "a2",
                "b",
                "c1",
                "c",
                "d1",
                "d",
            ],
            "structural": [
                r"\d+[a-z]?\.\s*",
                r"4a\.\s*",
                r"4b\.\s*",
            ],
            "min_matches": 2,
        },
        DocumentFamily.CONTRACT: {
            "keywords": [
                "contratto",
                "contract",
                "accordo",
                "agreement",
                "clausola",
                "clause",
                "parti contraenti",
                "soggetto",
                "firmato",
                "signed",
                "independent contractor",
                "consulting agreement",
                "freelance contract",
                "service agreement",
                "statement of work",
                "statement of work (sow)",
                "sow",
                "work order",
                "assignment letter",
                "retainer agreement",
                "letter of engagement",
                "engagement letter",  # Both orders
                "certificate of engagement",
                "statement of employment",
                "letter from hr",
                "client:",
                "contractor:",
                "client / contractor",
                "contractor / client",
                "dear",  # "Dear [Name]," pattern
                "this letter",  # "This letter certifies"
                "this engagement letter",
                "this agreement",
                "service provider",  # "Service Provider" in engagement letters
                "services provided",  # "services provided under this Letter"
                "fees charged",  # "fees charged by the Service Provider"
            ],
            "structural": [
                r"contratto\s+di\s+(\w+)",
                r"parti\s+contraenti",
                r"clausola\s+\d+",
                r"firmato\s+il",
                r"independent\s+contractor",
                r"contractor\s+agreement",
                r"statement\s+of\s+work",
                r"statement\s+of\s+work\s*\(sow\)",
                r"effective\s+date",
                r"client:\s*[A-Z]",
                r"contractor:\s*[A-Z]",
                r"reference\s+agreement",
                # Engagement letter patterns
                r"engagement\s+letter",
                r"this\s+engagement\s+letter",
                r"this\s+letter\s+certifies",
                r"dear\s+[A-Z][a-z]+",  # "Dear Franco,"
                r"this\s+letter\s+dated",  # "This letter, dated"
                r"services\s+requested\s+by",  # "services requested by [Client]"
                r"services\.?\s+The\s+services\s+provided",  # "Services. The services provided"
                r"fees\.?\s+The\s+fees\s+charged",  # "Fees. The fees charged"
                r"service\s+provider",  # "Service Provider"
                r"services\s+provided\s+under",  # "services provided under this Letter"
            ],
            "min_matches": 1,  # semantic documents are flexible
        },
        DocumentFamily.CERTIFICATE: {
            "keywords": [
                "certificato",
                "certificate",
                "diploma",
                "attestato",
                "attestation",
                "laurea",
                "università",
                "universita",
                "cfu",
                "crediti",
            ],
            "structural": [
                r"diploma\s+di\s+laurea",
                r"universit[àa]\s+degli?\s+studi",
                r"certificato\s+di",
            ],
            "min_matches": 2,
        },
        DocumentFamily.FINANCIAL: {
            "keywords": [
                "fattura",
                "invoice",
                "busta paga",
                "payslip",
                "estratto conto",
                "iva",
                "totale",
                "importo",
                "pagamento",
            ],
            "structural": [
                r"fattura\s+n[°º]?\s*:?\s*",
                r"iva\s*:?\s*",
                r"totale\s*:?\s*€",
            ],
            "min_matches": 2,
        },
        DocumentFamily.CORPORATE: {
            "keywords": [
                "visura",
                "statuto",
                "bilancio",
                "balance sheet",
                "camera di commercio",
                "partita iva",
                "codice fiscale",
                "società",
                "societa",
            ],
            "structural": [
                r"camera\s+di\s+commercio",
                r"visura\s+camerale",
                r"statuto\s+sociale",
            ],
            "min_matches": 2,
        },
    }

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify document into family with semantic co-occurrence and multi-level classification

        NEW:
        - Semantic co-occurrence patterns (combinations of signals)
        - Subtype classification
        - Adaptive confidence based on signal strength
        """
        if not text or len(text.strip()) < 10:
            return {
                "family": DocumentFamily.UNKNOWN,
                "subtype": DocumentSubtype.UNKNOWN,
                "confidence": 0.0,
                "source": "none",
                "matches": [],
                "co_occurrence_boost": 0.0,
            }

        text_lower = text.lower()
        scores = {}

        for family, config in self.FAMILY_PATTERNS.items():
            score = 0
            matches = []
            co_occurrence_boost = 0.0

            # Keywords
            keyword_matches = sum(
                1 for kw in config["keywords"] if kw.lower() in text_lower
            )
            score += keyword_matches * 3  # increased weight for contract keywords

            # Structural
            structural_matches = []
            for pattern in config["structural"]:
                found = re.search(pattern, text, re.IGNORECASE)
                if found:
                    structural_matches.append(
                        found.group(0) if found.groups() else found.group(0)
                    )
                    # Lower structural weight for driving_license
                    weight = 2 if family == DocumentFamily.DRIVING_LICENSE else 3
                    score += weight
            matches.extend(structural_matches)

            # NEW: Semantic co-occurrence patterns (combinations of signals)
            if family in self.SEMANTIC_CO_OCCURRENCE:
                for co_pattern, required_keywords, boost in self.SEMANTIC_CO_OCCURRENCE[
                    family
                ]:
                    # Check if pattern matches
                    if re.search(co_pattern, text, re.IGNORECASE | re.DOTALL):
                        # Check if required keywords are present
                        keywords_found = sum(
                            1 for kw in required_keywords if kw.lower() in text_lower
                        )
                        if (
                            keywords_found >= len(required_keywords) * 0.7
                        ):  # At least 70% of keywords
                            co_occurrence_boost += boost
                            matches.append(f"co-occurrence: {co_pattern[:50]}")

            # Normalized score
            total_possible = len(config["keywords"]) * 3 + len(config["structural"]) * 3
            normalized_score = (
                min(score / total_possible, 0.98) if total_possible > 0 else 0.0
            )

            # Apply co-occurrence boost
            normalized_score = min(0.98, normalized_score + co_occurrence_boost)

            # Check min_matches
            if keyword_matches + len(structural_matches) >= config["min_matches"]:
                # Boost CONTRACT confidence if semantic
                if family == DocumentFamily.CONTRACT:
                    normalized_score = max(normalized_score, 0.6)
                scores[family] = {
                    "score": normalized_score,
                    "matches": matches,
                    "keyword_matches": keyword_matches,
                    "structural_matches": len(structural_matches),
                    "co_occurrence_boost": co_occurrence_boost,
                }

        if not scores:
            return {
                "family": DocumentFamily.UNKNOWN,
                "subtype": DocumentSubtype.UNKNOWN,
                "confidence": 0.0,
                "source": "none",
                "matches": [],
                "co_occurrence_boost": 0.0,
            }

        # Best family
        best_family = max(scores.items(), key=lambda x: x[1]["score"])
        family_data = best_family[1]

        # NEW: Determine subtype based on family and specific patterns
        subtype = self._classify_subtype(text, best_family[0], text_lower)

        # Determine source
        if family_data["structural_matches"] > 0:
            source = "layout"
        elif family_data["keyword_matches"] > 0:
            source = "keywords"
        elif family_data.get("co_occurrence_boost", 0) > 0:
            source = "semantic_co_occurrence"
        else:
            source = "unknown"

        return {
            "family": best_family[0],
            "subtype": subtype,
            "confidence": family_data["score"],
            "source": source,
            "matches": family_data["matches"],
            "co_occurrence_boost": family_data.get("co_occurrence_boost", 0.0),
        }

    def _classify_subtype(
        self, text: str, family: DocumentFamily, text_lower: str
    ) -> DocumentSubtype:
        """
        Classify document subtype for more precise policy application

        NEW: Multi-level classification with hierarchical priority
        
        Priority order:
        1. Title matching (highest confidence) - check first 500 chars
        2. Structure analysis (payment terms, deliverables) - excludes NDA
        3. Keyword matching (lowest confidence) - only as fallback
        """
        if family == DocumentFamily.CONTRACT:
            # STAGE 1: Title-based classification (highest priority)
            # Extract title from first 500 characters (usually contains main title)
            # Normalize: remove extra whitespace, newlines, convert to uppercase
            title_section = re.sub(r'\s+', ' ', text[:500].upper().strip())
            
            # Professional Services Agreement (highest priority)
            if re.search(r"PROFESSIONAL\s+SERVICES\s+AGREEMENT", title_section):
                return DocumentSubtype.PROFESSIONAL_SERVICES_AGREEMENT
            
            # Independent Contractor Agreement
            if re.search(r"INDEPENDENT\s+CONTRACTOR\s+AGREEMENT", title_section):
                return DocumentSubtype.INDEPENDENT_CONTRACTOR_AGREEMENT
            
            # Statement of Work
            if re.search(r"STATEMENT\s+OF\s+WORK", title_section):
                return DocumentSubtype.STATEMENT_OF_WORK
            
            # Engagement Letter
            if re.search(r"ENGAGEMENT\s+LETTER", title_section) and re.search(
                r"(?:service\s+provider|dear\s+[A-Z])", text, re.IGNORECASE
            ):
                return DocumentSubtype.ENGAGEMENT_LETTER
            
            # Service Agreement
            if re.search(r"SERVICE\s+AGREEMENT", title_section):
                return DocumentSubtype.SERVICE_AGREEMENT
            
            # NDA - ONLY if it's in the title AND no payment terms
            if re.search(r"NON[-\s]?DISCLOSURE\s+AGREEMENT|NDA", title_section):
                # Check if there are payment terms (excludes NDA)
                has_payment_terms = self._has_payment_terms(text)
                if not has_payment_terms:
                    return DocumentSubtype.NDA
            
            # STAGE 2: Structure-based classification (excludes NDA if payment terms exist)
            # Check for payment terms, deliverables, milestones (indicates service agreement, not NDA)
            has_payment_terms = self._has_payment_terms(text)
            has_deliverables = self._has_deliverables(text)
            has_milestones = self._has_milestones(text)
            
            # If payment terms exist, it's NOT an NDA
            if has_payment_terms or has_deliverables or has_milestones:
                # Classify based on other signals
                if re.search(r"independent\s+contractor", text_lower):
                    return DocumentSubtype.INDEPENDENT_CONTRACTOR_AGREEMENT
                elif re.search(r"statement\s+of\s+work", text_lower):
                    return DocumentSubtype.STATEMENT_OF_WORK
                elif re.search(r"professional\s+services", text_lower):
                    return DocumentSubtype.PROFESSIONAL_SERVICES_AGREEMENT
                elif re.search(r"service\s+agreement", text_lower):
                    return DocumentSubtype.SERVICE_AGREEMENT
                else:
                    # Generic service agreement if payment terms exist
                    return DocumentSubtype.SERVICE_AGREEMENT
            
            # STAGE 3: Keyword-based classification (fallback, lowest priority)
            # Only check NDA if no payment terms were found
            if not has_payment_terms:
                # NDA - must be in title or very prominent
                if re.search(r"non[-\s]?disclosure\s+agreement|nda", title_section):
                    return DocumentSubtype.NDA
            
            # Engagement Letter (check full text)
            if re.search(r"engagement\s+letter", text_lower) and re.search(
                r"(?:service\s+provider|dear\s+[A-Z])", text, re.IGNORECASE
            ):
                return DocumentSubtype.ENGAGEMENT_LETTER

            # Statement of Work
            if re.search(r"statement\s+of\s+work", text_lower):
                return DocumentSubtype.STATEMENT_OF_WORK

            # Independent Contractor Agreement
            if re.search(r"independent\s+contractor\s+agreement", text_lower):
                return DocumentSubtype.INDEPENDENT_CONTRACTOR_AGREEMENT

            # Service Agreement
            if re.search(r"service\s+agreement", text_lower):
                return DocumentSubtype.SERVICE_AGREEMENT

            return DocumentSubtype.CONTRACT_GENERIC

        elif family == DocumentFamily.CERTIFICATE:
            # Diploma
            if re.search(r"diploma|laurea", text_lower):
                return DocumentSubtype.DIPLOMA

            # Certificate of Engagement
            if re.search(r"certificate\s+of\s+engagement", text_lower):
                return DocumentSubtype.CERTIFICATE_OF_ENGAGEMENT

            return DocumentSubtype.CERTIFICATE_GENERIC

        elif family == DocumentFamily.FINANCIAL:
            # Invoice
            if re.search(r"invoice|fattura", text_lower):
                return DocumentSubtype.INVOICE

            # Payslip
            if re.search(r"payslip|busta\s+paga", text_lower):
                return DocumentSubtype.PAYSLIP

            # Bank Statement
            if re.search(r"bank\s+statement|estratto\s+conto", text_lower):
                return DocumentSubtype.BANK_STATEMENT

            return DocumentSubtype.UNKNOWN

        elif family == DocumentFamily.IDENTITY:
            # ID Card
            if re.search(r"carta\s+d\'?identit|id\s+card", text_lower):
                return DocumentSubtype.ID_CARD

            # Passport
            if re.search(r"passport|passaporto", text_lower):
                return DocumentSubtype.PASSPORT

            return DocumentSubtype.UNKNOWN

        return DocumentSubtype.UNKNOWN
