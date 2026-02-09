"""
STRUCTURE LAYER

Rule-based structure analysis that detects signals without interpretation.

Responsibilities:
- Detect structural patterns (Dear X, numbered sections, etc.)
- Identify candidate document families based on structure
- Provide confidence scores for each candidate
- Explain which signals triggered each candidate

This is DETERMINISTIC and RULE-BASED - no LLM, no guessing.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .perception import PerceptionResult


class DocumentFamily(str, Enum):
    """Document families detected by structure"""
    IDENTITY = "identity"
    DRIVING_LICENSE = "driving_license"
    CONTRACT = "contract"
    CERTIFICATE = "certificate"
    FINANCIAL = "financial"
    CORPORATE = "corporate"
    UNKNOWN = "unknown"


@dataclass
class StructureSignal:
    """A detected structural signal"""
    name: str
    detected: bool
    confidence: float
    location: Optional[str] = None  # Where in document (e.g., "header", "body")
    evidence: Optional[str] = None  # What text/pattern matched


@dataclass
class CandidateFamily:
    """A candidate document family with explanation"""
    family: DocumentFamily
    confidence: float
    signals: List[StructureSignal]
    explanation: str  # Why this family was suggested


@dataclass
class StructureAnalysis:
    """Structure analysis result"""
    candidates: List[CandidateFamily]
    primary_family: Optional[DocumentFamily]
    primary_confidence: float
    all_signals: List[StructureSignal]
    explanation: str  # Overall explanation


class StructureAnalyzer:
    """
    STRUCTURE LAYER - Rule-based structure analysis
    
    Detects structural patterns and suggests candidate families.
    This is deterministic and explainable - no LLM, no guessing.
    """
    
    def __init__(self):
        """Initialize structure analyzer"""
        pass
    
    def analyze(self, perception: PerceptionResult) -> StructureAnalysis:
        """
        Analyze document structure and suggest candidate families
        
        Args:
            perception: Raw perception data
            
        Returns:
            StructureAnalysis with candidates and explanations
        """
        text = perception.raw_text
        text_lower = text.lower()
        
        # Detect all structural signals
        all_signals = self._detect_all_signals(text, text_lower, perception)
        
        # Evaluate candidates based on signals
        candidates = self._evaluate_candidates(all_signals, perception)
        
        # Select primary candidate
        if candidates:
            primary = max(candidates, key=lambda c: c.confidence)
            primary_family = primary.family
            primary_confidence = primary.confidence
        else:
            primary_family = DocumentFamily.UNKNOWN
            primary_confidence = 0.0
        
        # Build explanation
        explanation = self._build_explanation(candidates, all_signals)
        
        return StructureAnalysis(
            candidates=candidates,
            primary_family=primary_family,
            primary_confidence=primary_confidence,
            all_signals=all_signals,
            explanation=explanation
        )
    
    def _detect_all_signals(self, text: str, text_lower: str, perception: PerceptionResult) -> List[StructureSignal]:
        """Detect all structural signals"""
        signals = []
        
        # === IDENTITY SIGNALS ===
        signals.append(StructureSignal(
            name="mrz_pattern",
            detected=bool(re.search(r'[A-Z0-9<]{25,}', text)),
            confidence=0.95 if bool(re.search(r'[A-Z0-9<]{25,}', text)) else 0.0,
            location="body",
            evidence="MRZ pattern detected" if bool(re.search(r'[A-Z0-9<]{25,}', text)) else None
        ))
        
        signals.append(StructureSignal(
            name="id_keywords",
            detected=any(kw in text_lower for kw in ["carta d'identità", "carta di identità", "codice fiscale"]),
            confidence=0.85 if any(kw in text_lower for kw in ["carta d'identità", "carta di identità", "codice fiscale"]) else 0.0,
            location="body"
        ))
        
        # === DRIVING LICENSE SIGNALS ===
        signals.append(StructureSignal(
            name="patente_keywords",
            detected=any(kw in text_lower for kw in ["patente di guida", "repubblica italiana"]),
            confidence=0.90 if any(kw in text_lower for kw in ["patente di guida", "repubblica italiana"]) else 0.0,
            location="body"
        ))
        
        signals.append(StructureSignal(
            name="numbered_fields",
            detected=perception.has_numbered_fields,
            confidence=0.80 if perception.has_numbered_fields else 0.0,
            location="body"
        ))
        
        # === CONTRACT SIGNALS ===
        signals.append(StructureSignal(
            name="dear_pattern",
            detected=bool(re.search(r'dear\s+[A-Z][a-z]+', text, re.IGNORECASE)),
            confidence=0.70 if bool(re.search(r'dear\s+[A-Z][a-z]+', text, re.IGNORECASE)) else 0.0,
            location="header",
            evidence=re.search(r'dear\s+([A-Z][a-z]+)', text, re.IGNORECASE).group(1) if re.search(r'dear\s+[A-Z][a-z]+', text, re.IGNORECASE) else None
        ))
        
        signals.append(StructureSignal(
            name="agreement_keywords",
            detected=any(kw in text_lower for kw in ["agreement", "contract", "engagement letter", "statement of work"]),
            confidence=0.75 if any(kw in text_lower for kw in ["agreement", "contract", "engagement letter", "statement of work"]) else 0.0,
            location="title"
        ))
        
        signals.append(StructureSignal(
            name="payment_terms",
            detected=any(pattern in text_lower for pattern in ["payment schedule", "total project value", "amount (aed)", "compensation"]),
            confidence=0.85 if any(pattern in text_lower for pattern in ["payment schedule", "total project value", "amount (aed)", "compensation"]) else 0.0,
            location="body"
        ))
        
        signals.append(StructureSignal(
            name="service_sections",
            detected=any(pattern in text_lower for pattern in ["scope of services", "scope of work", "deliverables"]),
            confidence=0.80 if any(pattern in text_lower for pattern in ["scope of services", "scope of work", "deliverables"]) else 0.0,
            location="body"
        ))
        
        signals.append(StructureSignal(
            name="signatures",
            detected=perception.has_signatures,
            confidence=0.70 if perception.has_signatures else 0.0,
            location="footer"
        ))
        
        # === FINANCIAL SIGNALS ===
        signals.append(StructureSignal(
            name="invoice_keywords",
            detected=any(kw in text_lower for kw in ["invoice", "fattura", "bill", "invoice number"]),
            confidence=0.90 if any(kw in text_lower for kw in ["invoice", "fattura", "bill", "invoice number"]) else 0.0,
            location="header"
        ))
        
        signals.append(StructureSignal(
            name="vat_tax",
            detected=any(kw in text_lower for kw in ["vat", "iva", "tax", "imposta"]),
            confidence=0.75 if any(kw in text_lower for kw in ["vat", "iva", "tax", "imposta"]) else 0.0,
            location="body"
        ))
        
        signals.append(StructureSignal(
            name="totals",
            detected=any(kw in text_lower for kw in ["total", "totale", "amount due", "balance"]),
            confidence=0.70 if any(kw in text_lower for kw in ["total", "totale", "amount due", "balance"]) else 0.0,
            location="body"
        ))
        
        # === CERTIFICATE SIGNALS ===
        # Generic certificate keywords
        signals.append(StructureSignal(
            name="certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "certificate", "certificato", "certification", "certificazione",
                "diploma", "diplôme", "diplom", "laurea", "degree", "titolo"
            ]),
            confidence=0.85 if any(kw in text_lower for kw in [
                "certificate", "certificato", "certification", "certificazione",
                "diploma", "diplôme", "diplom", "laurea", "degree", "titolo"
            ]) else 0.0,
            location="header"
        ))
        
        # Academic/University certificates
        signals.append(StructureSignal(
            name="university_keywords",
            detected=any(kw in text_lower for kw in [
                "university", "università", "universita", "université", "universität",
                "college", "collegio", "facoltà", "faculty", "faculté",
                "cfu", "credits", "crediti", "ects", "credit hours",
                "bachelor", "master", "phd", "dottorato", "doctorate",
                "thesis", "tesi", "dissertation", "dissertazione"
            ]),
            confidence=0.90 if any(kw in text_lower for kw in [
                "university", "università", "cfu", "credits", "bachelor", "master"
            ]) else 0.0,
            location="body"
        ))
        
        # Professional certificates
        signals.append(StructureSignal(
            name="professional_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "professional certificate", "certificato professionale",
                "professional certification", "certificazione professionale",
                "certified", "certificato", "certified professional",
                "professional qualification", "qualifica professionale",
                "license", "licenza", "licence", "licence professionnelle"
            ]),
            confidence=0.90 if any(kw in text_lower for kw in [
                "professional certificate", "certificato professionale", "certified professional"
            ]) else 0.0,
            location="header"
        ))
        
        # ISO and Quality certificates
        signals.append(StructureSignal(
            name="iso_quality_keywords",
            detected=any(kw in text_lower for kw in [
                "iso", "iso 9001", "iso 14001", "iso 27001", "iso 45001",
                "iso certification", "certificazione iso",
                "quality certificate", "certificato di qualità",
                "quality management", "gestione qualità",
                "quality assurance", "assicurazione qualità",
                "qms", "quality management system"
            ]),
            confidence=0.95 if any(kw in text_lower for kw in [
                "iso", "iso 9001", "iso certification", "certificazione iso"
            ]) else 0.0,
            location="body"
        ))
        
        # Compliance and Conformity certificates
        signals.append(StructureSignal(
            name="compliance_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "compliance certificate", "certificato di conformità",
                "conformity certificate", "certificato di conformità",
                "ce marking", "marcatura ce", "ce mark",
                "fcc", "fda", "rohs", "reach", "warranty", "garanzia",
                "regulatory compliance", "conformità normativa"
            ]),
            confidence=0.90 if any(kw in text_lower for kw in [
                "compliance certificate", "certificato di conformità", "ce marking"
            ]) else 0.0,
            location="body"
        ))
        
        # Security certificates
        signals.append(StructureSignal(
            name="security_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "security certificate", "certificato di sicurezza",
                "ssl certificate", "certificato ssl", "tls certificate",
                "cyber security", "sicurezza informatica",
                "penetration testing", "test di penetrazione",
                "security audit", "audit di sicurezza",
                "pci dss", "gdpr", "hipaa", "sox"
            ]),
            confidence=0.90 if any(kw in text_lower for kw in [
                "security certificate", "certificato di sicurezza", "ssl certificate"
            ]) else 0.0,
            location="body"
        ))
        
        # Training and Course certificates
        signals.append(StructureSignal(
            name="training_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "training certificate", "certificato di formazione",
                "course certificate", "certificato di corso",
                "certificate of completion", "certificato di completamento",
                "certificate of attendance", "certificato di partecipazione",
                "certificate of achievement", "certificato di raggiungimento",
                "workshop", "seminario", "corso", "course", "training",
                "hours completed", "ore completate", "duration", "durata"
            ]),
            confidence=0.85 if any(kw in text_lower for kw in [
                "training certificate", "certificato di formazione", "certificate of completion"
            ]) else 0.0,
            location="body"
        ))
        
        # Language certificates
        signals.append(StructureSignal(
            name="language_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "language certificate", "certificato linguistico",
                "language proficiency", "competenza linguistica",
                "ielts", "toefl", "cambridge", "dele", "delf", "dalf",
                "celi", "cils", "plida", "goethe", "testdaf",
                "english certificate", "certificato inglese",
                "level", "livello", "a1", "a2", "b1", "b2", "c1", "c2"
            ]),
            confidence=0.95 if any(kw in text_lower for kw in [
                "ielts", "toefl", "cambridge", "language certificate", "certificato linguistico"
            ]) else 0.0,
            location="body"
        ))
        
        # Software/Technology certificates
        signals.append(StructureSignal(
            name="software_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "software certificate", "certificato software",
                "technology certificate", "certificato tecnologico",
                "microsoft certified", "oracle certified", "cisco certified",
                "aws certified", "google cloud certified", "azure certified",
                "pmp", "scrum master", "agile", "itil", "prince2",
                "certified developer", "certified engineer", "certified architect"
            ]),
            confidence=0.90 if any(kw in text_lower for kw in [
                "microsoft certified", "oracle certified", "cisco certified", "aws certified"
            ]) else 0.0,
            location="body"
        ))
        
        # Competence and Skill certificates
        signals.append(StructureSignal(
            name="competence_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "competence certificate", "certificato di competenza",
                "skill certificate", "certificato di abilità",
                "certified skills", "competenze certificate",
                "proficiency", "competenza", "skill assessment",
                "competency", "capacità", "ability", "abilità"
            ]),
            confidence=0.85 if any(kw in text_lower for kw in [
                "competence certificate", "certificato di competenza", "certified skills"
            ]) else 0.0,
            location="body"
        ))
        
        # Project certificates
        signals.append(StructureSignal(
            name="project_certificate_keywords",
            detected=any(kw in text_lower for kw in [
                "project certificate", "certificato di progetto",
                "project completion", "completamento progetto",
                "project management", "gestione progetto",
                "deliverable", "consegna", "milestone", "traguardo"
            ]),
            confidence=0.80 if any(kw in text_lower for kw in [
                "project certificate", "certificato di progetto", "project completion"
            ]) else 0.0,
            location="body"
        ))
        
        # Certificate structure patterns
        signals.append(StructureSignal(
            name="certificate_structure",
            detected=any(pattern in text for pattern in [
                "this is to certify", "si certifica che", "certifichiamo che",
                "has successfully completed", "ha completato con successo",
                "has demonstrated", "ha dimostrato", "has achieved", "ha raggiunto",
                "issued on", "rilasciato il", "date of issue", "data di rilascio",
                "valid until", "valido fino al", "expiry date", "data di scadenza",
                "certificate number", "numero certificato", "certificate id", "id certificato"
            ]),
            confidence=0.90 if any(pattern in text_lower for pattern in [
                "this is to certify", "si certifica che", "has successfully completed"
            ]) else 0.0,
            location="body"
        ))
        
        # Issuer/Authority patterns
        signals.append(StructureSignal(
            name="certificate_issuer",
            detected=any(kw in text_lower for kw in [
                "issued by", "rilasciato da", "certified by", "certificato da",
                "authorized by", "autorizzato da", "accredited", "accreditato",
                "certification body", "ente di certificazione",
                "certification authority", "autorità di certificazione"
            ]),
            confidence=0.85 if any(kw in text_lower for kw in [
                "issued by", "rilasciato da", "certified by", "certification body"
            ]) else 0.0,
            location="footer"
        ))
        
        # === RESUME/CV SIGNALS ===
        signals.append(StructureSignal(
            name="resume_keywords",
            detected=any(kw in text_lower for kw in ["curriculum vitae", "curriculum", "cv", "resume", "résumé"]),
            confidence=0.95 if any(kw in text_lower for kw in ["curriculum vitae", "curriculum", "cv", "resume", "résumé"]) else 0.0,
            location="header"
        ))
        
        signals.append(StructureSignal(
            name="resume_sections",
            detected=any(kw in text_lower for kw in ["education", "experience", "work experience", "skills", "competencies", "professional experience", "formazione", "esperienza", "competenze"]),
            confidence=0.90 if any(kw in text_lower for kw in ["education", "experience", "work experience", "skills", "competencies", "professional experience", "formazione", "esperienza", "competenze"]) else 0.0,
            location="body"
        ))
        
        signals.append(StructureSignal(
            name="personal_info",
            detected=any(pattern in text_lower for pattern in ["date of birth", "date of birth:", "born", "nato", "nato il", "phone", "telephone", "email", "e-mail", "address", "indirizzo"]),
            confidence=0.85 if any(pattern in text_lower for pattern in ["date of birth", "date of birth:", "born", "nato", "nato il", "phone", "telephone", "email", "e-mail", "address", "indirizzo"]) else 0.0,
            location="header"
        ))
        
        return signals
    
    def _evaluate_candidates(self, signals: List[StructureSignal], perception: PerceptionResult) -> List[CandidateFamily]:
        """Evaluate candidate families based on signals"""
        candidates = []
        
        # === IDENTITY CANDIDATE ===
        identity_signals = [s for s in signals if s.name in ["mrz_pattern", "id_keywords"]]
        identity_confidence = self._calculate_family_confidence(identity_signals, base_confidence=0.60)
        if identity_confidence > 0.5:
            candidates.append(CandidateFamily(
                family=DocumentFamily.IDENTITY,
                confidence=identity_confidence,
                signals=identity_signals,
                explanation=self._explain_family(identity_signals, "identity")
            ))
        
        # === DRIVING LICENSE CANDIDATE ===
        dl_signals = [s for s in signals if s.name in ["patente_keywords", "numbered_fields"]]
        dl_confidence = self._calculate_family_confidence(dl_signals, base_confidence=0.65)
        if dl_confidence > 0.5:
            candidates.append(CandidateFamily(
                family=DocumentFamily.DRIVING_LICENSE,
                confidence=dl_confidence,
                signals=dl_signals,
                explanation=self._explain_family(dl_signals, "driving_license")
            ))
        
        # === CONTRACT CANDIDATE ===
        contract_signals = [s for s in signals if s.name in [
            "dear_pattern", "agreement_keywords", "payment_terms", 
            "service_sections", "signatures"
        ]]
        contract_confidence = self._calculate_family_confidence(contract_signals, base_confidence=0.50)
        # Boost if multiple contract signals
        if len([s for s in contract_signals if s.detected]) >= 2:
            contract_confidence = min(contract_confidence + 0.15, 0.95)
        if contract_confidence > 0.4:
            candidates.append(CandidateFamily(
                family=DocumentFamily.CONTRACT,
                confidence=contract_confidence,
                signals=contract_signals,
                explanation=self._explain_family(contract_signals, "contract")
            ))
        
        # === FINANCIAL CANDIDATE ===
        financial_signals = [s for s in signals if s.name in ["invoice_keywords", "vat_tax", "totals"]]
        financial_confidence = self._calculate_family_confidence(financial_signals, base_confidence=0.60)
        if financial_confidence > 0.5:
            candidates.append(CandidateFamily(
                family=DocumentFamily.FINANCIAL,
                confidence=financial_confidence,
                signals=financial_signals,
                explanation=self._explain_family(financial_signals, "financial")
            ))
        
        # === CERTIFICATE CANDIDATE ===
        cert_signals = [s for s in signals if s.name in ["certificate_keywords", "university_keywords"]]
        cert_confidence = self._calculate_family_confidence(cert_signals, base_confidence=0.60)
        if cert_confidence > 0.5:
            candidates.append(CandidateFamily(
                family=DocumentFamily.CERTIFICATE,
                confidence=cert_confidence,
                signals=cert_signals,
                explanation=self._explain_family(cert_signals, "certificate")
            ))
        
        # === IDENTITY CANDIDATE (for Resume/CV) ===
        # Resume/CV should be classified as IDENTITY family, not FINANCIAL
        resume_signals = [s for s in signals if s.name in ["resume_keywords", "resume_sections", "personal_info"]]
        resume_confidence = self._calculate_family_confidence(resume_signals, base_confidence=0.70)
        # Boost if multiple resume signals detected
        if len([s for s in resume_signals if s.detected]) >= 2:
            resume_confidence = min(resume_confidence + 0.15, 0.95)
        if resume_confidence > 0.6:
            candidates.append(CandidateFamily(
                family=DocumentFamily.IDENTITY,
                confidence=resume_confidence,
                signals=resume_signals,
                explanation=self._explain_family(resume_signals, "identity (resume/cv)")
            ))
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates
    
    def _calculate_family_confidence(self, signals: List[StructureSignal], base_confidence: float = 0.5) -> float:
        """Calculate confidence for a family based on signals"""
        if not signals:
            return 0.0
        
        detected_signals = [s for s in signals if s.detected]
        if not detected_signals:
            return 0.0
        
        # Average confidence of detected signals
        avg_confidence = sum(s.confidence for s in detected_signals) / len(detected_signals)
        
        # Boost based on number of signals
        signal_count_boost = min(len(detected_signals) * 0.05, 0.15)
        
        return min(base_confidence + avg_confidence * 0.3 + signal_count_boost, 0.95)
    
    def _explain_family(self, signals: List[StructureSignal], family_name: str) -> str:
        """Build explanation for why a family was suggested"""
        detected = [s for s in signals if s.detected]
        if not detected:
            return f"No signals detected for {family_name}"
        
        signal_names = [s.name for s in detected]
        return f"{family_name} suggested by signals: {', '.join(signal_names)}"
    
    def _build_explanation(self, candidates: List[CandidateFamily], all_signals: List[StructureSignal]) -> str:
        """Build overall explanation"""
        if not candidates:
            return "No candidate families detected. Document structure unclear."
        
        primary = candidates[0]
        explanation_parts = [
            f"Primary candidate: {primary.family.value} (confidence: {primary.confidence:.2f})",
            f"Reason: {primary.explanation}"
        ]
        
        if len(candidates) > 1:
            explanation_parts.append(f"\nAlternative candidates:")
            for cand in candidates[1:]:
                explanation_parts.append(f"  - {cand.family.value} (confidence: {cand.confidence:.2f}): {cand.explanation}")
        
        return "\n".join(explanation_parts)
