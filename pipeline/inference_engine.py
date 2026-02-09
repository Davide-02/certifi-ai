"""
INFERENCE ENGINE

Core deterministic logic that combines structure signals and extracted facts
to make final decisions about document family and subtype.

This is the FINAL DECISION MAKER - implemented in Python code, not LLM.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .structure_analyzer import StructureAnalysis, DocumentFamily
from .semantic_extractor import ExtractedFacts


class DocumentSubtype(str, Enum):
    """Document subtypes - more specific than families"""
    
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
    DEGREE = "degree"  # Laurea / Degree
    CERTIFICATE_OF_COMPLETION = "certificate_of_completion"  # Certificato di completamento corso
    PROFESSIONAL_CERTIFICATE = "professional_certificate"  # Certificato professionale
    COMPETENCE_CERTIFICATE = "competence_certificate"  # Certificato di competenza
    TRAINING_CERTIFICATE = "training_certificate"  # Certificato di formazione
    ISO_CERTIFICATE = "iso_certificate"  # Certificato ISO
    QUALITY_CERTIFICATE = "quality_certificate"  # Certificato di qualità
    COMPLIANCE_CERTIFICATE = "compliance_certificate"  # Certificato di conformità
    SECURITY_CERTIFICATE = "security_certificate"  # Certificato di sicurezza
    LANGUAGE_CERTIFICATE = "language_certificate"  # Certificato linguistico
    SOFTWARE_CERTIFICATE = "software_certificate"  # Certificato software/tecnologico
    PROJECT_CERTIFICATE = "project_certificate"  # Certificato di progetto
    CERTIFICATE_OF_ENGAGEMENT = "certificate_of_engagement"
    CERTIFICATE_OF_ACHIEVEMENT = "certificate_of_achievement"  # Certificato di raggiungimento
    CERTIFICATE_GENERIC = "certificate_generic"
    
    # Financial subtypes
    INVOICE = "invoice"
    PAYSLIP = "payslip"
    BANK_STATEMENT = "bank_statement"
    
    # Identity subtypes
    ID_CARD = "id_card"
    PASSPORT = "passport"
    RESUME = "resume"  # Curriculum Vitae / CV
    
    # Generic
    UNKNOWN = "unknown"


@dataclass
class InferenceDecision:
    """Final inference decision"""
    document_family: DocumentFamily
    document_subtype: DocumentSubtype
    confidence: float
    explanation: str
    signals_used: List[str]
    alternatives_considered: List[Dict[str, Any]]


class InferenceEngine:
    """
    INFERENCE ENGINE - Deterministic decision maker
    
    Combines structure signals and extracted facts to make final decisions.
    This is implemented in Python code - deterministic and explainable.
    """
    
    def __init__(self):
        """Initialize inference engine"""
        pass
    
    def infer(
        self,
        structure: StructureAnalysis,
        facts: ExtractedFacts,
        perception_text: str
    ) -> InferenceDecision:
        """
        Make final inference decision
        
        Args:
            structure: Structure analysis results
            facts: Extracted facts
            perception_text: Raw text for additional pattern matching
            
        Returns:
            InferenceDecision with final classification and explanation
        """
        # Start with structure layer's primary candidate
        primary_family = structure.primary_family or DocumentFamily.UNKNOWN
        base_confidence = structure.primary_confidence
        
        # Refine based on extracted facts
        refined_family, refined_subtype, confidence_boost, explanation = self._refine_with_facts(
            primary_family,
            facts,
            perception_text
        )
        
        # Calculate final confidence
        final_confidence = min(base_confidence + confidence_boost, 0.95)
        
        # Build explanation
        full_explanation = self._build_full_explanation(
            structure,
            facts,
            refined_family,
            refined_subtype,
            final_confidence
        )
        
        # Track signals used
        signals_used = [s.name for s in structure.all_signals if s.detected]
        if facts.fields_extracted:
            signals_used.extend([f"fact_{f}" for f in facts.fields_extracted])
        
        # Consider alternatives
        alternatives = [
            {
                "family": cand.family.value,
                "confidence": cand.confidence,
                "reason": cand.explanation
            }
            for cand in structure.candidates[1:]  # Skip primary
        ]
        
        return InferenceDecision(
            document_family=refined_family,
            document_subtype=refined_subtype,
            confidence=final_confidence,
            explanation=full_explanation,
            signals_used=signals_used,
            alternatives_considered=alternatives
        )
    
    def _refine_with_facts(
        self,
        family: DocumentFamily,
        facts: ExtractedFacts,
        text: str
    ) -> Tuple[DocumentFamily, DocumentSubtype, float, str]:
        """
        Refine family and determine subtype based on extracted facts
        
        Returns:
            (refined_family, subtype, confidence_boost, explanation)
        """
        text_lower = text.lower()
        text_normalized = re.sub(r'\s+', ' ', text[:500].upper().strip())
        
        if family == DocumentFamily.CONTRACT:
            # Determine contract subtype based on facts and text
            
            # Professional Services Agreement
            if (
                "PROFESSIONAL SERVICES AGREEMENT" in text_normalized or
                (facts.amount and facts.amount > 10000 and facts.services)
            ):
                return (
                    DocumentFamily.CONTRACT,
                    DocumentSubtype.PROFESSIONAL_SERVICES_AGREEMENT,
                    0.15,
                    "Professional Services Agreement detected from title and payment terms"
                )
            
            # Independent Contractor Agreement
            if (
                "INDEPENDENT CONTRACTOR AGREEMENT" in text_normalized or
                (facts.party_1_type == "individual" and facts.amount)
            ):
                return (
                    DocumentFamily.CONTRACT,
                    DocumentSubtype.INDEPENDENT_CONTRACTOR_AGREEMENT,
                    0.15,
                    "Independent Contractor Agreement detected from title and party structure"
                )
            
            # Statement of Work
            if "STATEMENT OF WORK" in text_normalized or facts.deliverables:
                return (
                    DocumentFamily.CONTRACT,
                    DocumentSubtype.STATEMENT_OF_WORK,
                    0.15,
                    "Statement of Work detected from title or deliverables"
                )
            
            # Engagement Letter
            if (
                "ENGAGEMENT LETTER" in text_normalized or
                (re.search(r'dear\s+[A-Z][a-z]+', text, re.IGNORECASE) and facts.amount)
            ):
                return (
                    DocumentFamily.CONTRACT,
                    DocumentSubtype.ENGAGEMENT_LETTER,
                    0.15,
                    "Engagement Letter detected from title or salutation pattern"
                )
            
            # NDA - ONLY if no payment terms
            if (
                ("NON-DISCLOSURE AGREEMENT" in text_normalized or "NDA" in text_normalized) and
                not facts.amount and
                not any(kw in text_lower for kw in ["payment", "fee", "compensation", "amount"])
            ):
                return (
                    DocumentFamily.CONTRACT,
                    DocumentSubtype.NDA,
                    0.10,
                    "NDA detected from title, no payment terms found"
                )
            
            # Generic contract
            return (
                DocumentFamily.CONTRACT,
                DocumentSubtype.CONTRACT_GENERIC,
                0.0,
                "Contract detected but subtype unclear"
            )
        
        elif family == DocumentFamily.FINANCIAL:
            # Determine financial subtype
            # IMPORTANT: Exclude Resume/CV from financial classification
            # If document has resume keywords, it should NOT be classified as financial
            has_resume_keywords = any(kw in text_lower for kw in ["curriculum vitae", "curriculum", "cv", "resume", "résumé"])
            has_resume_sections = any(kw in text_lower for kw in ["education", "experience", "work experience", "skills", "formazione", "esperienza"])
            
            if has_resume_keywords or (has_resume_sections and any(kw in text_lower for kw in ["phone", "email", "address"])):
                # This is actually a resume, not financial - return low confidence
                return (
                    DocumentFamily.FINANCIAL,
                    DocumentSubtype.INVOICE,
                    -0.30,  # Negative boost to discourage financial classification
                    "Financial detected but resume keywords found - likely misclassified"
                )
            
            if facts.invoice_number or "invoice" in text_lower:
                return (
                    DocumentFamily.FINANCIAL,
                    DocumentSubtype.INVOICE,
                    0.15,
                    "Invoice detected from invoice number or keywords"
                )
            
            if "payslip" in text_lower or "busta paga" in text_lower:
                return (
                    DocumentFamily.FINANCIAL,
                    DocumentSubtype.PAYSLIP,
                    0.15,
                    "Payslip detected from keywords"
                )
            
            return (
                DocumentFamily.FINANCIAL,
                DocumentSubtype.INVOICE,  # Default to invoice
                0.0,
                "Financial document detected"
            )
        
        elif family == DocumentFamily.CERTIFICATE:
            # Determine certificate subtype with comprehensive pattern matching
            
            # Academic certificates (highest priority)
            if any(kw in text_lower for kw in ["diploma", "laurea", "degree", "bachelor", "master", "phd", "dottorato"]):
                if any(kw in text_lower for kw in ["bachelor", "laurea triennale", "undergraduate"]):
                    return (
                        DocumentFamily.CERTIFICATE,
                        DocumentSubtype.DEGREE,
                        0.20,
                        "Bachelor's degree detected"
                    )
                elif any(kw in text_lower for kw in ["master", "laurea magistrale", "graduate"]):
                    return (
                        DocumentFamily.CERTIFICATE,
                        DocumentSubtype.DEGREE,
                        0.20,
                        "Master's degree detected"
                    )
                else:
                    return (
                        DocumentFamily.CERTIFICATE,
                        DocumentSubtype.DIPLOMA,
                        0.20,
                        "Diploma detected from keywords"
                    )
            
            # ISO and Quality certificates
            if any(kw in text_lower for kw in ["iso", "iso 9001", "iso 14001", "iso 27001", "iso certification"]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.ISO_CERTIFICATE,
                    0.20,
                    "ISO certificate detected"
                )
            
            if any(kw in text_lower for kw in ["quality certificate", "certificato di qualità", "quality management", "qms"]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.QUALITY_CERTIFICATE,
                    0.20,
                    "Quality certificate detected"
                )
            
            # Compliance certificates
            if any(kw in text_lower for kw in ["compliance certificate", "certificato di conformità", "ce marking", "ce mark", "fcc", "fda"]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.COMPLIANCE_CERTIFICATE,
                    0.20,
                    "Compliance certificate detected"
                )
            
            # Security certificates
            if any(kw in text_lower for kw in ["security certificate", "certificato di sicurezza", "ssl certificate", "tls certificate", "pci dss", "gdpr"]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.SECURITY_CERTIFICATE,
                    0.20,
                    "Security certificate detected"
                )
            
            # Language certificates
            if any(kw in text_lower for kw in ["ielts", "toefl", "cambridge", "language certificate", "certificato linguistico", "dele", "delf", "celi", "cils"]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.LANGUAGE_CERTIFICATE,
                    0.20,
                    "Language certificate detected"
                )
            
            # Software/Technology certificates
            if any(kw in text_lower for kw in [
                "microsoft certified", "oracle certified", "cisco certified", "aws certified",
                "google cloud certified", "azure certified", "pmp", "scrum master", "itil"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.SOFTWARE_CERTIFICATE,
                    0.20,
                    "Software/Technology certificate detected"
                )
            
            # Training certificates
            if any(kw in text_lower for kw in [
                "training certificate", "certificato di formazione",
                "course certificate", "certificato di corso",
                "certificate of completion", "certificato di completamento",
                "certificate of attendance", "certificato di partecipazione"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.TRAINING_CERTIFICATE,
                    0.20,
                    "Training certificate detected"
                )
            
            # Achievement certificates
            if any(kw in text_lower for kw in [
                "certificate of achievement", "certificato di raggiungimento",
                "has achieved", "ha raggiunto", "has demonstrated", "ha dimostrato"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.CERTIFICATE_OF_ACHIEVEMENT,
                    0.20,
                    "Achievement certificate detected"
                )
            
            # Professional certificates
            if any(kw in text_lower for kw in [
                "professional certificate", "certificato professionale",
                "professional certification", "certificazione professionale",
                "certified professional", "professionista certificato"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.PROFESSIONAL_CERTIFICATE,
                    0.20,
                    "Professional certificate detected"
                )
            
            # Competence certificates
            if any(kw in text_lower for kw in [
                "competence certificate", "certificato di competenza",
                "skill certificate", "certificato di abilità",
                "certified skills", "competenze certificate"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.COMPETENCE_CERTIFICATE,
                    0.20,
                    "Competence certificate detected"
                )
            
            # Project certificates
            if any(kw in text_lower for kw in [
                "project certificate", "certificato di progetto",
                "project completion", "completamento progetto"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.PROJECT_CERTIFICATE,
                    0.20,
                    "Project certificate detected"
                )
            
            # Completion certificates
            if any(kw in text_lower for kw in [
                "certificate of completion", "certificato di completamento",
                "has successfully completed", "ha completato con successo"
            ]):
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.CERTIFICATE_OF_COMPLETION,
                    0.20,
                    "Completion certificate detected"
                )
            
            # Fallback: Certificate with issuer
            if facts.issuer_name:
                return (
                    DocumentFamily.CERTIFICATE,
                    DocumentSubtype.CERTIFICATE_OF_ENGAGEMENT,
                    0.10,
                    "Certificate with issuer detected"
                )
            
            # Generic certificate
            return (
                DocumentFamily.CERTIFICATE,
                DocumentSubtype.CERTIFICATE_GENERIC,
                0.0,
                "Certificate detected"
            )
        
        elif family == DocumentFamily.IDENTITY:
            # Determine identity subtype
            # Check for Resume/CV first (has priority)
            if (
                any(kw in text_lower for kw in ["curriculum vitae", "curriculum", "cv", "resume", "résumé"]) or
                (any(kw in text_lower for kw in ["education", "experience", "skills", "formazione", "esperienza"]) and
                 any(kw in text_lower for kw in ["phone", "email", "address", "telephone", "e-mail"]))
            ):
                return (
                    DocumentFamily.IDENTITY,
                    DocumentSubtype.RESUME,
                    0.20,
                    "Resume/CV detected from keywords and structure"
                )
            
            if "passport" in text_lower or "passaporto" in text_lower:
                return (
                    DocumentFamily.IDENTITY,
                    DocumentSubtype.PASSPORT,
                    0.15,
                    "Passport detected from keywords"
                )
            
            return (
                DocumentFamily.IDENTITY,
                DocumentSubtype.ID_CARD,
                0.0,
                "ID card detected"
            )
        
        # Unknown or other families
        return (
            family,
            DocumentSubtype.UNKNOWN,
            0.0,
            f"{family.value} detected but subtype unclear"
        )
    
    def _build_full_explanation(
        self,
        structure: StructureAnalysis,
        facts: ExtractedFacts,
        family: DocumentFamily,
        subtype: DocumentSubtype,
        confidence: float
    ) -> str:
        """Build comprehensive explanation"""
        parts = [
            f"Document Family: {family.value}",
            f"Document Subtype: {subtype.value}",
            f"Confidence: {confidence:.2f}",
            "",
            "Structure Analysis:",
            structure.explanation,
            "",
            "Extracted Facts:",
        ]
        
        if facts.fields_extracted:
            parts.append(f"  Extracted fields: {', '.join(facts.fields_extracted)}")
            if facts.amount:
                parts.append(f"  Amount: {facts.amount} {facts.currency or ''}")
            if facts.effective_date:
                parts.append(f"  Effective Date: {facts.effective_date}")
            if facts.subject:
                parts.append(f"  Subject: {facts.subject[:100]}")
        else:
            parts.append("  No facts extracted")
        
        parts.append("")
        parts.append(f"Extraction Method: {facts.extraction_method}")
        
        return "\n".join(parts)
