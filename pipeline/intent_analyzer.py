"""
DOCUMENT INTENT & ONTOLOGY LAYER

Infers the real-world intent of a document, NOT its type.

This layer answers: "What is this document trying to accomplish in the real world?"

Responsibilities:
- Infer primary and secondary intents
- Determine trust role (what kind of trust does this document establish)
- Assess risk profile
- Provide explainable decision path

This is DETERMINISTIC and RULE-BASED - no LLM, no guessing.
Uses explicit ontology mappings and rule sets.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .structure_analyzer import DocumentFamily
from .inference_engine import DocumentSubtype
from .semantic_extractor import ExtractedFacts


class DocumentIntent(str, Enum):
    """Real-world intents that documents can have"""
    
    # Skill/Competence related
    CERTIFY_SKILL = "certify_skill"  # Prove someone has a skill/competence
    CERTIFY_QUALIFICATION = "certify_qualification"  # Prove educational qualification
    
    # Relationship/Responsibility related
    ESTABLISH_RELATIONSHIP = "establish_relationship"  # Create a relationship between parties
    TRANSFER_RESPONSIBILITY = "transfer_responsibility"  # Transfer responsibility/liability
    AUTHORIZE_ACTION = "authorize_action"  # Authorize someone to do something
    
    # Legal/Compliance related
    ASSERT_COMPLIANCE = "assert_compliance"  # Assert compliance with regulations
    LEGAL_COMMITMENT = "legal_commitment"  # Create legal obligation
    DISCLOSE_INFORMATION = "disclose_information"  # Disclose confidential information
    
    # Identity/Verification related
    VERIFY_IDENTITY = "verify_identity"  # Verify someone's identity
    PROVE_ELIGIBILITY = "prove_eligibility"  # Prove eligibility for something
    
    # Financial related
    REQUEST_PAYMENT = "request_payment"  # Request payment for services
    PROVE_INCOME = "prove_income"  # Prove income/earnings
    
    # Generic
    UNKNOWN = "unknown"


class TrustRole(str, Enum):
    """What kind of trust does this document establish"""
    
    PROOF_OF_COMPETENCE = "proof_of_competence"  # Proves skill/competence
    PROOF_OF_IDENTITY = "proof_of_identity"  # Proves identity
    PROOF_OF_QUALIFICATION = "proof_of_qualification"  # Proves qualification
    LEGAL_COMMITMENT = "legal_commitment"  # Creates legal commitment
    COMPLIANCE_EVIDENCE = "compliance_evidence"  # Evidence of compliance
    RELATIONSHIP_EVIDENCE = "relationship_evidence"  # Evidence of relationship
    FINANCIAL_EVIDENCE = "financial_evidence"  # Financial proof
    AUTHORIZATION_EVIDENCE = "authorization_evidence"  # Proof of authorization
    UNKNOWN = "unknown"


class RiskProfile(str, Enum):
    """Risk profile of the document"""
    
    LOW = "low"  # Low risk, standard document
    MEDIUM = "medium"  # Medium risk, requires some verification
    HIGH = "high"  # High risk, requires careful verification


@dataclass
class IntentAnalysis:
    """Intent analysis result"""
    primary_intent: DocumentIntent
    secondary_intents: List[DocumentIntent]
    trust_role: TrustRole
    risk_profile: RiskProfile
    explanation: str
    decision_path: List[str]  # Step-by-step decision path
    confidence: float


class IntentAnalyzer:
    """
    DOCUMENT INTENT & ONTOLOGY LAYER
    
    Infers real-world intent using deterministic rule sets and ontology mappings.
    NO LLM - all logic is explicit Python code.
    """
    
    def __init__(self):
        """Initialize intent analyzer with ontology mappings"""
        # Ontology: Family + Subtype → Primary Intent mapping
        self.family_subtype_to_intent = {
            (DocumentFamily.IDENTITY, DocumentSubtype.ID_CARD): DocumentIntent.VERIFY_IDENTITY,
            (DocumentFamily.IDENTITY, DocumentSubtype.PASSPORT): DocumentIntent.VERIFY_IDENTITY,
            (DocumentFamily.IDENTITY, DocumentSubtype.RESUME): DocumentIntent.CERTIFY_QUALIFICATION,
            (DocumentFamily.DRIVING_LICENSE, None): DocumentIntent.AUTHORIZE_ACTION,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.DIPLOMA): DocumentIntent.CERTIFY_QUALIFICATION,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.DEGREE): DocumentIntent.CERTIFY_QUALIFICATION,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.PROFESSIONAL_CERTIFICATE): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.COMPETENCE_CERTIFICATE): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.TRAINING_CERTIFICATE): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.ISO_CERTIFICATE): DocumentIntent.ASSERT_COMPLIANCE,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.QUALITY_CERTIFICATE): DocumentIntent.ASSERT_COMPLIANCE,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.COMPLIANCE_CERTIFICATE): DocumentIntent.ASSERT_COMPLIANCE,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.SECURITY_CERTIFICATE): DocumentIntent.ASSERT_COMPLIANCE,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.LANGUAGE_CERTIFICATE): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.SOFTWARE_CERTIFICATE): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.PROJECT_CERTIFICATE): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.CERTIFICATE_OF_COMPLETION): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.CERTIFICATE_OF_ACHIEVEMENT): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.CERTIFICATE_OF_ENGAGEMENT): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CERTIFICATE, DocumentSubtype.CERTIFICATE_GENERIC): DocumentIntent.CERTIFY_SKILL,
            (DocumentFamily.CONTRACT, DocumentSubtype.ENGAGEMENT_LETTER): DocumentIntent.ESTABLISH_RELATIONSHIP,
            (DocumentFamily.CONTRACT, DocumentSubtype.STATEMENT_OF_WORK): DocumentIntent.TRANSFER_RESPONSIBILITY,
            (DocumentFamily.CONTRACT, DocumentSubtype.INDEPENDENT_CONTRACTOR_AGREEMENT): DocumentIntent.ESTABLISH_RELATIONSHIP,
            (DocumentFamily.CONTRACT, DocumentSubtype.PROFESSIONAL_SERVICES_AGREEMENT): DocumentIntent.ESTABLISH_RELATIONSHIP,
            (DocumentFamily.CONTRACT, DocumentSubtype.SERVICE_AGREEMENT): DocumentIntent.ESTABLISH_RELATIONSHIP,
            (DocumentFamily.CONTRACT, DocumentSubtype.NDA): DocumentIntent.DISCLOSE_INFORMATION,
            (DocumentFamily.CONTRACT, DocumentSubtype.CONTRACT_GENERIC): DocumentIntent.LEGAL_COMMITMENT,
            (DocumentFamily.FINANCIAL, DocumentSubtype.INVOICE): DocumentIntent.REQUEST_PAYMENT,
            (DocumentFamily.FINANCIAL, DocumentSubtype.PAYSLIP): DocumentIntent.PROVE_INCOME,
            (DocumentFamily.FINANCIAL, DocumentSubtype.BANK_STATEMENT): DocumentIntent.PROVE_INCOME,
        }
        
        # Ontology: Family → Trust Role mapping
        self.family_to_trust_role = {
            DocumentFamily.IDENTITY: TrustRole.PROOF_OF_IDENTITY,
            DocumentFamily.DRIVING_LICENSE: TrustRole.AUTHORIZATION_EVIDENCE,
            DocumentFamily.CERTIFICATE: TrustRole.PROOF_OF_QUALIFICATION,
            DocumentFamily.CONTRACT: TrustRole.LEGAL_COMMITMENT,
            DocumentFamily.FINANCIAL: TrustRole.FINANCIAL_EVIDENCE,
            DocumentFamily.CORPORATE: TrustRole.COMPLIANCE_EVIDENCE,
        }
        
        # Secondary intent patterns (based on facts)
        self.fact_patterns_to_intents = {
            "has_amount": [DocumentIntent.REQUEST_PAYMENT, DocumentIntent.LEGAL_COMMITMENT],
            "has_dates": [DocumentIntent.LEGAL_COMMITMENT, DocumentIntent.ESTABLISH_RELATIONSHIP],
            "has_parties": [DocumentIntent.ESTABLISH_RELATIONSHIP, DocumentIntent.TRANSFER_RESPONSIBILITY],
            "has_services": [DocumentIntent.TRANSFER_RESPONSIBILITY, DocumentIntent.ESTABLISH_RELATIONSHIP],
            "has_issuer": [DocumentIntent.CERTIFY_SKILL, DocumentIntent.CERTIFY_QUALIFICATION],
            "has_confidentiality": [DocumentIntent.DISCLOSE_INFORMATION],
        }
    
    def analyze(
        self,
        family: DocumentFamily,
        subtype: Optional[DocumentSubtype],
        facts: ExtractedFacts,
        structure_signals: Optional[List[str]] = None
    ) -> IntentAnalysis:
        """
        Analyze document intent
        
        Args:
            family: Document family from inference engine
            subtype: Document subtype from inference engine
            facts: Extracted facts from semantic extraction
            structure_signals: Optional list of structure signals
            
        Returns:
            IntentAnalysis with intent, trust role, risk profile, and explanation
        """
        decision_path = []
        confidence = 0.0
        
        # STEP 1: Determine primary intent from family + subtype
        decision_path.append(f"Step 1: Looking up intent for family={family.value}, subtype={subtype.value if subtype else 'None'}")
        
        primary_intent = self._determine_primary_intent(family, subtype, decision_path)
        
        # STEP 2: Determine secondary intents from facts
        decision_path.append("Step 2: Analyzing extracted facts for secondary intents")
        secondary_intents = self._determine_secondary_intents(facts, primary_intent, decision_path)
        
        # STEP 3: Determine trust role
        decision_path.append("Step 3: Determining trust role from family")
        trust_role = self._determine_trust_role(family, subtype, facts, decision_path)
        
        # STEP 4: Assess risk profile
        decision_path.append("Step 4: Assessing risk profile")
        risk_profile = self._assess_risk_profile(family, subtype, facts, primary_intent, decision_path)
        
        # STEP 5: Calculate confidence
        confidence = self._calculate_confidence(family, subtype, facts, primary_intent)
        
        # STEP 6: Build explanation
        explanation = self._build_explanation(
            primary_intent, secondary_intents, trust_role, risk_profile, decision_path
        )
        
        return IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            trust_role=trust_role,
            risk_profile=risk_profile,
            explanation=explanation,
            decision_path=decision_path,
            confidence=confidence
        )
    
    def _determine_primary_intent(
        self,
        family: DocumentFamily,
        subtype: Optional[DocumentSubtype],
        decision_path: List[str]
    ) -> DocumentIntent:
        """Determine primary intent from family and subtype"""
        # Try exact match first
        key = (family, subtype)
        if key in self.family_subtype_to_intent:
            intent = self.family_subtype_to_intent[key]
            decision_path.append(f"  → Found exact match: {intent.value}")
            return intent
        
        # Try with None subtype
        key_with_none = (family, None)
        if key_with_none in self.family_subtype_to_intent:
            intent = self.family_subtype_to_intent[key_with_none]
            decision_path.append(f"  → Found match with None subtype: {intent.value}")
            return intent
        
        # Fallback: use family-based heuristics
        decision_path.append(f"  → No exact match, using family-based heuristics")
        
        if family == DocumentFamily.IDENTITY:
            return DocumentIntent.VERIFY_IDENTITY
        elif family == DocumentFamily.CONTRACT:
            return DocumentIntent.LEGAL_COMMITMENT
        elif family == DocumentFamily.CERTIFICATE:
            return DocumentIntent.CERTIFY_SKILL
        elif family == DocumentFamily.FINANCIAL:
            return DocumentIntent.REQUEST_PAYMENT
        else:
            return DocumentIntent.UNKNOWN
    
    def _determine_secondary_intents(
        self,
        facts: ExtractedFacts,
        primary_intent: DocumentIntent,
        decision_path: List[str]
    ) -> List[DocumentIntent]:
        """Determine secondary intents from extracted facts"""
        secondary = []
        
        # Check fact patterns
        if facts.amount and facts.amount > 0:
            decision_path.append("  → Found amount, adding REQUEST_PAYMENT/LEGAL_COMMITMENT")
            if DocumentIntent.REQUEST_PAYMENT not in secondary and primary_intent != DocumentIntent.REQUEST_PAYMENT:
                secondary.append(DocumentIntent.REQUEST_PAYMENT)
            if DocumentIntent.LEGAL_COMMITMENT not in secondary and primary_intent != DocumentIntent.LEGAL_COMMITMENT:
                secondary.append(DocumentIntent.LEGAL_COMMITMENT)
        
        if facts.party_1_name and facts.party_2_name:
            decision_path.append("  → Found multiple parties, adding ESTABLISH_RELATIONSHIP")
            if DocumentIntent.ESTABLISH_RELATIONSHIP not in secondary and primary_intent != DocumentIntent.ESTABLISH_RELATIONSHIP:
                secondary.append(DocumentIntent.ESTABLISH_RELATIONSHIP)
        
        if facts.services or facts.deliverables:
            decision_path.append("  → Found services/deliverables, adding TRANSFER_RESPONSIBILITY")
            if DocumentIntent.TRANSFER_RESPONSIBILITY not in secondary and primary_intent != DocumentIntent.TRANSFER_RESPONSIBILITY:
                secondary.append(DocumentIntent.TRANSFER_RESPONSIBILITY)
        
        if facts.effective_date and facts.expiration_date:
            decision_path.append("  → Found date range, adding LEGAL_COMMITMENT")
            if DocumentIntent.LEGAL_COMMITMENT not in secondary and primary_intent != DocumentIntent.LEGAL_COMMITMENT:
                secondary.append(DocumentIntent.LEGAL_COMMITMENT)
        
        if facts.issuer_name:
            decision_path.append("  → Found issuer, adding CERTIFY_SKILL")
            if DocumentIntent.CERTIFY_SKILL not in secondary and primary_intent != DocumentIntent.CERTIFY_SKILL:
                secondary.append(DocumentIntent.CERTIFY_SKILL)
        
        # Remove duplicates and limit to top 3
        secondary = list(dict.fromkeys(secondary))[:3]
        
        if secondary:
            decision_path.append(f"  → Secondary intents: {[i.value for i in secondary]}")
        else:
            decision_path.append("  → No secondary intents detected")
        
        return secondary
    
    def _determine_trust_role(
        self,
        family: DocumentFamily,
        subtype: Optional[DocumentSubtype],
        facts: ExtractedFacts,
        decision_path: List[str]
    ) -> TrustRole:
        """Determine trust role from family and context"""
        # Start with family-based mapping
        if family in self.family_to_trust_role:
            trust_role = self.family_to_trust_role[family]
            decision_path.append(f"  → Mapped family to trust role: {trust_role.value}")
            
            # Refine based on subtype
            if subtype == DocumentSubtype.NDA:
                trust_role = TrustRole.COMPLIANCE_EVIDENCE
                decision_path.append(f"  → Refined to COMPLIANCE_EVIDENCE (NDA)")
            elif subtype == DocumentSubtype.DIPLOMA:
                trust_role = TrustRole.PROOF_OF_QUALIFICATION
                decision_path.append(f"  → Refined to PROOF_OF_QUALIFICATION (Diploma)")
            elif subtype == DocumentSubtype.PAYSLIP:
                trust_role = TrustRole.FINANCIAL_EVIDENCE
                decision_path.append(f"  → Refined to FINANCIAL_EVIDENCE (Payslip)")
            
            return trust_role
        
        # Fallback
        decision_path.append("  → No mapping found, using UNKNOWN")
        return TrustRole.UNKNOWN
    
    def _assess_risk_profile(
        self,
        family: DocumentFamily,
        subtype: Optional[DocumentSubtype],
        facts: ExtractedFacts,
        primary_intent: DocumentIntent,
        decision_path: List[str]
    ) -> RiskProfile:
        """Assess risk profile based on document characteristics"""
        risk_score = 0
        
        # Base risk by family
        family_risk = {
            DocumentFamily.IDENTITY: 1,  # Low risk
            DocumentFamily.DRIVING_LICENSE: 1,  # Low risk
            DocumentFamily.CERTIFICATE: 2,  # Medium risk
            DocumentFamily.CONTRACT: 3,  # High risk
            DocumentFamily.FINANCIAL: 3,  # High risk
        }
        risk_score += family_risk.get(family, 2)
        decision_path.append(f"  → Family risk: {family_risk.get(family, 2)}")
        
        # Adjust by intent
        intent_risk = {
            DocumentIntent.VERIFY_IDENTITY: -1,  # Lower risk
            DocumentIntent.CERTIFY_SKILL: 0,  # Neutral
            DocumentIntent.LEGAL_COMMITMENT: +1,  # Higher risk
            DocumentIntent.REQUEST_PAYMENT: +1,  # Higher risk
            DocumentIntent.ESTABLISH_RELATIONSHIP: +1,  # Higher risk
        }
        risk_score += intent_risk.get(primary_intent, 0)
        decision_path.append(f"  → Intent risk adjustment: {intent_risk.get(primary_intent, 0)}")
        
        # Adjust by facts completeness
        fact_completeness = sum([
            1 if facts.amount else 0,
            1 if facts.currency else 0,
            1 if facts.effective_date else 0,
            1 if facts.expiration_date else 0,
            1 if facts.party_1_name else 0,
            1 if facts.party_2_name else 0,
        ])
        
        if fact_completeness < 3:
            risk_score += 1  # Higher risk if incomplete
            decision_path.append(f"  → Low fact completeness ({fact_completeness}/6), +1 risk")
        else:
            decision_path.append(f"  → Good fact completeness ({fact_completeness}/6)")
        
        # Determine final risk profile
        if risk_score <= 2:
            risk_profile = RiskProfile.LOW
        elif risk_score <= 4:
            risk_profile = RiskProfile.MEDIUM
        else:
            risk_profile = RiskProfile.HIGH
        
        decision_path.append(f"  → Final risk score: {risk_score} → {risk_profile.value}")
        
        return risk_profile
    
    def _calculate_confidence(
        self,
        family: DocumentFamily,
        subtype: Optional[DocumentSubtype],
        facts: ExtractedFacts,
        primary_intent: DocumentIntent
    ) -> float:
        """Calculate confidence in intent analysis"""
        confidence = 0.5  # Base confidence
        
        # Boost if we have exact family+subtype mapping
        key = (family, subtype)
        if key in self.family_subtype_to_intent:
            confidence += 0.2
        
        # Boost if we have extracted facts
        fact_count = len(facts.fields_extracted) if facts.fields_extracted else 0
        if fact_count > 0:
            confidence += min(fact_count * 0.05, 0.2)
        
        # Boost if primary intent is not UNKNOWN
        if primary_intent != DocumentIntent.UNKNOWN:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _build_explanation(
        self,
        primary_intent: DocumentIntent,
        secondary_intents: List[DocumentIntent],
        trust_role: TrustRole,
        risk_profile: RiskProfile,
        decision_path: List[str]
    ) -> str:
        """Build human-readable explanation"""
        parts = [
            f"Document Intent Analysis",
            f"=" * 60,
            f"",
            f"Primary Intent: {primary_intent.value}",
            f"  → This document's primary purpose is to {self._intent_description(primary_intent)}",
            f"",
        ]
        
        if secondary_intents:
            parts.append(f"Secondary Intents:")
            for intent in secondary_intents:
                parts.append(f"  • {intent.value}: {self._intent_description(intent)}")
            parts.append("")
        
        parts.extend([
            f"Trust Role: {trust_role.value}",
            f"  → This document serves as {self._trust_role_description(trust_role)}",
            f"",
            f"Risk Profile: {risk_profile.value.upper()}",
            f"  → Risk level: {self._risk_description(risk_profile)}",
            f"",
            f"Decision Path:",
        ])
        
        for step in decision_path:
            parts.append(f"  {step}")
        
        return "\n".join(parts)
    
    def _intent_description(self, intent: DocumentIntent) -> str:
        """Human-readable description of intent"""
        descriptions = {
            DocumentIntent.CERTIFY_SKILL: "certify that someone has a specific skill or competence",
            DocumentIntent.CERTIFY_QUALIFICATION: "certify an educational qualification",
            DocumentIntent.ESTABLISH_RELATIONSHIP: "establish a relationship between parties",
            DocumentIntent.TRANSFER_RESPONSIBILITY: "transfer responsibility or liability",
            DocumentIntent.AUTHORIZE_ACTION: "authorize someone to perform an action",
            DocumentIntent.ASSERT_COMPLIANCE: "assert compliance with regulations or standards",
            DocumentIntent.LEGAL_COMMITMENT: "create a legal obligation or commitment",
            DocumentIntent.DISCLOSE_INFORMATION: "disclose confidential information under terms",
            DocumentIntent.VERIFY_IDENTITY: "verify someone's identity",
            DocumentIntent.PROVE_ELIGIBILITY: "prove eligibility for something",
            DocumentIntent.REQUEST_PAYMENT: "request payment for services or goods",
            DocumentIntent.PROVE_INCOME: "prove income or earnings",
            DocumentIntent.UNKNOWN: "unknown or unclear intent",
        }
        return descriptions.get(intent, "unknown intent")
    
    def _trust_role_description(self, role: TrustRole) -> str:
        """Human-readable description of trust role"""
        descriptions = {
            TrustRole.PROOF_OF_COMPETENCE: "proof of competence or skill",
            TrustRole.PROOF_OF_IDENTITY: "proof of identity",
            TrustRole.PROOF_OF_QUALIFICATION: "proof of qualification",
            TrustRole.LEGAL_COMMITMENT: "evidence of legal commitment",
            TrustRole.COMPLIANCE_EVIDENCE: "evidence of compliance",
            TrustRole.RELATIONSHIP_EVIDENCE: "evidence of relationship",
            TrustRole.FINANCIAL_EVIDENCE: "financial evidence or proof",
            TrustRole.AUTHORIZATION_EVIDENCE: "evidence of authorization",
            TrustRole.UNKNOWN: "unknown trust role",
        }
        return descriptions.get(role, "unknown role")
    
    def _risk_description(self, risk: RiskProfile) -> str:
        """Human-readable description of risk"""
        descriptions = {
            RiskProfile.LOW: "Low risk - standard document with minimal verification needed",
            RiskProfile.MEDIUM: "Medium risk - requires some verification and validation",
            RiskProfile.HIGH: "High risk - requires careful verification and may need human review",
        }
        return descriptions.get(risk, "unknown risk level")
