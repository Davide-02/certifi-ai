"""
PROCESS CONTEXT LAYER

Maps document intent to real-world business processes.

This layer answers: "What business process does this document belong to?"

Responsibilities:
- Map intent to business process
- Determine compliance domain
- Identify lifecycle stage
- Recommend actions
- Provide explainable decision path

This is DETERMINISTIC and RULE-BASED - no LLM, no guessing.
Uses explicit ontology mappings and rule sets.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .intent_analyzer import DocumentIntent, TrustRole, RiskProfile
from .semantic_extractor import ExtractedFacts


class BusinessProcess(str, Enum):
    """Real-world business processes"""
    
    # Onboarding & Qualification
    CONTRACTOR_ONBOARDING = "contractor_onboarding"
    SUPPLIER_QUALIFICATION = "supplier_qualification"
    EMPLOYEE_ONBOARDING = "employee_onboarding"
    VENDOR_REGISTRATION = "vendor_registration"
    
    # Certification & Verification
    EMPLOYEE_CERTIFICATION = "employee_certification"
    SKILL_VERIFICATION = "skill_verification"
    QUALIFICATION_VERIFICATION = "qualification_verification"
    IDENTITY_VERIFICATION = "identity_verification"
    
    # Compliance & Audit
    REGULATORY_AUDIT = "regulatory_audit"
    COMPLIANCE_REVIEW = "compliance_review"
    INTERNAL_AUDIT = "internal_audit"
    RISK_ASSESSMENT = "risk_assessment"
    
    # Financial & Operations
    FINANCIAL_REPORTING = "financial_reporting"
    PAYMENT_PROCESSING = "payment_processing"
    INVOICE_PROCESSING = "invoice_processing"
    CONTRACT_MANAGEMENT = "contract_management"
    
    # Relationship Management
    PARTNERSHIP_ESTABLISHMENT = "partnership_establishment"
    CLIENT_ONBOARDING = "client_onboarding"
    SERVICE_AGREEMENT_PROCESSING = "service_agreement_processing"
    
    # Generic
    UNKNOWN = "unknown"


class ComplianceDomain(str, Enum):
    """Compliance domains"""
    
    HR = "HR"  # Human Resources
    LEGAL = "Legal"
    FINANCE = "Finance"
    OPERATIONS = "Operations"
    COMPLIANCE = "Compliance"
    PROCUREMENT = "Procurement"
    IT = "IT"
    SECURITY = "Security"
    UNKNOWN = "unknown"


class LifecycleStage(str, Enum):
    """Document lifecycle stages"""
    
    CREATION = "creation"  # Document is being created
    VERIFICATION = "verification"  # Document is being verified
    ACTIVE = "active"  # Document is active/in use
    AUDIT = "audit"  # Document is being audited
    RENEWAL = "renewal"  # Document is being renewed
    EXPIRATION = "expiration"  # Document is expiring/expired
    ARCHIVAL = "archival"  # Document is archived


@dataclass
class ProcessContext:
    """Process context analysis result"""
    business_process: BusinessProcess
    compliance_domain: ComplianceDomain
    lifecycle_stage: LifecycleStage
    recommended_actions: List[str]
    explanation: str
    decision_path: List[str]  # Step-by-step decision path
    confidence: float


class ProcessContextAnalyzer:
    """
    PROCESS CONTEXT LAYER
    
    Maps document intent to real-world business processes using deterministic rules.
    NO LLM - all logic is explicit Python code.
    """
    
    def __init__(self):
        """Initialize process context analyzer with ontology mappings"""
        
        # Ontology: Intent → Business Process mapping
        self.intent_to_process = {
            DocumentIntent.ESTABLISH_RELATIONSHIP: [
                BusinessProcess.CONTRACTOR_ONBOARDING,
                BusinessProcess.SUPPLIER_QUALIFICATION,
                BusinessProcess.PARTNERSHIP_ESTABLISHMENT,
                BusinessProcess.CLIENT_ONBOARDING,
            ],
            DocumentIntent.CERTIFY_SKILL: [
                BusinessProcess.EMPLOYEE_CERTIFICATION,
                BusinessProcess.SKILL_VERIFICATION,
            ],
            DocumentIntent.CERTIFY_QUALIFICATION: [
                BusinessProcess.QUALIFICATION_VERIFICATION,
                BusinessProcess.EMPLOYEE_CERTIFICATION,
            ],
            DocumentIntent.VERIFY_IDENTITY: [
                BusinessProcess.IDENTITY_VERIFICATION,
                BusinessProcess.EMPLOYEE_ONBOARDING,
                BusinessProcess.CONTRACTOR_ONBOARDING,
            ],
            DocumentIntent.CERTIFY_QUALIFICATION: [
                BusinessProcess.QUALIFICATION_VERIFICATION,
                BusinessProcess.EMPLOYEE_ONBOARDING,
                BusinessProcess.SKILL_VERIFICATION,
            ],
            DocumentIntent.LEGAL_COMMITMENT: [
                BusinessProcess.CONTRACT_MANAGEMENT,
                BusinessProcess.SERVICE_AGREEMENT_PROCESSING,
            ],
            DocumentIntent.REQUEST_PAYMENT: [
                BusinessProcess.INVOICE_PROCESSING,
                BusinessProcess.PAYMENT_PROCESSING,
                BusinessProcess.FINANCIAL_REPORTING,
            ],
            DocumentIntent.PROVE_INCOME: [
                BusinessProcess.FINANCIAL_REPORTING,
                BusinessProcess.COMPLIANCE_REVIEW,
            ],
            DocumentIntent.ASSERT_COMPLIANCE: [
                BusinessProcess.REGULATORY_AUDIT,
                BusinessProcess.COMPLIANCE_REVIEW,
                BusinessProcess.INTERNAL_AUDIT,
            ],
            DocumentIntent.DISCLOSE_INFORMATION: [
                BusinessProcess.COMPLIANCE_REVIEW,
                BusinessProcess.RISK_ASSESSMENT,
            ],
            DocumentIntent.TRANSFER_RESPONSIBILITY: [
                BusinessProcess.CONTRACT_MANAGEMENT,
                BusinessProcess.SUPPLIER_QUALIFICATION,
            ],
        }
        
        # Ontology: Trust Role → Compliance Domain mapping
        self.trust_role_to_domain = {
            TrustRole.PROOF_OF_COMPETENCE: ComplianceDomain.HR,
            TrustRole.PROOF_OF_QUALIFICATION: ComplianceDomain.HR,
            TrustRole.PROOF_OF_IDENTITY: ComplianceDomain.HR,
            TrustRole.LEGAL_COMMITMENT: ComplianceDomain.LEGAL,
            TrustRole.COMPLIANCE_EVIDENCE: ComplianceDomain.COMPLIANCE,
            TrustRole.RELATIONSHIP_EVIDENCE: ComplianceDomain.OPERATIONS,
            TrustRole.FINANCIAL_EVIDENCE: ComplianceDomain.FINANCE,
            TrustRole.AUTHORIZATION_EVIDENCE: ComplianceDomain.SECURITY,
        }
        
        # Ontology: Business Process → Compliance Domain mapping
        self.process_to_domain = {
            BusinessProcess.CONTRACTOR_ONBOARDING: ComplianceDomain.HR,
            BusinessProcess.EMPLOYEE_ONBOARDING: ComplianceDomain.HR,
            BusinessProcess.EMPLOYEE_CERTIFICATION: ComplianceDomain.HR,
            BusinessProcess.SKILL_VERIFICATION: ComplianceDomain.HR,
            BusinessProcess.QUALIFICATION_VERIFICATION: ComplianceDomain.HR,
            BusinessProcess.IDENTITY_VERIFICATION: ComplianceDomain.HR,
            BusinessProcess.SUPPLIER_QUALIFICATION: ComplianceDomain.PROCUREMENT,
            BusinessProcess.VENDOR_REGISTRATION: ComplianceDomain.PROCUREMENT,
            BusinessProcess.CONTRACT_MANAGEMENT: ComplianceDomain.LEGAL,
            BusinessProcess.SERVICE_AGREEMENT_PROCESSING: ComplianceDomain.LEGAL,
            BusinessProcess.PARTNERSHIP_ESTABLISHMENT: ComplianceDomain.LEGAL,
            BusinessProcess.REGULATORY_AUDIT: ComplianceDomain.COMPLIANCE,
            BusinessProcess.COMPLIANCE_REVIEW: ComplianceDomain.COMPLIANCE,
            BusinessProcess.INTERNAL_AUDIT: ComplianceDomain.COMPLIANCE,
            BusinessProcess.RISK_ASSESSMENT: ComplianceDomain.COMPLIANCE,
            BusinessProcess.FINANCIAL_REPORTING: ComplianceDomain.FINANCE,
            BusinessProcess.PAYMENT_PROCESSING: ComplianceDomain.FINANCE,
            BusinessProcess.INVOICE_PROCESSING: ComplianceDomain.FINANCE,
            BusinessProcess.CLIENT_ONBOARDING: ComplianceDomain.OPERATIONS,
        }
        
        # Lifecycle stage detection rules
        self.lifecycle_rules = {
            LifecycleStage.CREATION: lambda facts: not facts.effective_date or facts.issue_date == facts.effective_date,
            LifecycleStage.VERIFICATION: lambda facts: facts.issue_date and not facts.expiration_date,
            LifecycleStage.ACTIVE: lambda facts: facts.effective_date and facts.expiration_date and self._is_active(facts),
            LifecycleStage.AUDIT: lambda facts: facts.issue_date and self._is_past_midpoint(facts),
            LifecycleStage.EXPIRATION: lambda facts: facts.expiration_date and self._is_expiring_soon(facts),
            LifecycleStage.RENEWAL: lambda facts: facts.expiration_date and self._is_expiring_soon(facts),
        }
        
        # Recommended actions by process
        self.process_actions = {
            BusinessProcess.CONTRACTOR_ONBOARDING: [
                "Verify contractor identity",
                "Check background and references",
                "Review contract terms",
                "Set up payment processing",
                "Schedule orientation",
            ],
            BusinessProcess.SUPPLIER_QUALIFICATION: [
                "Verify supplier credentials",
                "Check compliance certifications",
                "Review financial stability",
                "Assess risk profile",
                "Approve vendor registration",
            ],
            BusinessProcess.EMPLOYEE_CERTIFICATION: [
                "Verify certification validity",
                "Check issuer credentials",
                "Validate skill requirements",
                "Update employee records",
                "Schedule recertification",
            ],
            BusinessProcess.REGULATORY_AUDIT: [
                "Review compliance documentation",
                "Verify regulatory requirements",
                "Assess compliance gaps",
                "Document findings",
                "Create remediation plan",
            ],
            BusinessProcess.FINANCIAL_REPORTING: [
                "Verify financial data",
                "Check calculations",
                "Review supporting documents",
                "Approve for reporting",
                "Archive documentation",
            ],
            BusinessProcess.INVOICE_PROCESSING: [
                "Verify invoice details",
                "Check purchase order match",
                "Validate amounts and taxes",
                "Approve payment",
                "Process payment",
            ],
            BusinessProcess.CONTRACT_MANAGEMENT: [
                "Review contract terms",
                "Verify parties and signatures",
                "Check compliance requirements",
                "Set up monitoring",
                "Schedule renewal review",
            ],
            BusinessProcess.IDENTITY_VERIFICATION: [
                "Verify identity document",
                "Check document authenticity",
                "Validate personal information",
                "Update identity records",
                "Set verification expiry",
            ],
        }
    
    def analyze(
        self,
        intent: DocumentIntent,
        trust_role: TrustRole,
        risk_profile: RiskProfile,
        facts: ExtractedFacts
    ) -> ProcessContext:
        """
        Analyze process context
        
        Args:
            intent: Document intent from Intent Layer
            trust_role: Trust role from Intent Layer
            risk_profile: Risk profile from Intent Layer
            facts: Extracted facts from Semantic Extraction
            
        Returns:
            ProcessContext with business process, domain, lifecycle, and actions
        """
        decision_path = []
        confidence = 0.0
        
        # STEP 1: Determine business process from intent
        decision_path.append(f"Step 1: Mapping intent '{intent.value}' to business process")
        business_process = self._determine_business_process(intent, trust_role, facts, decision_path)
        
        # STEP 2: Determine compliance domain
        decision_path.append("Step 2: Determining compliance domain")
        compliance_domain = self._determine_compliance_domain(business_process, trust_role, decision_path)
        
        # STEP 3: Determine lifecycle stage
        decision_path.append("Step 3: Determining lifecycle stage")
        lifecycle_stage = self._determine_lifecycle_stage(facts, business_process, decision_path)
        
        # STEP 4: Generate recommended actions
        decision_path.append("Step 4: Generating recommended actions")
        recommended_actions = self._generate_recommended_actions(
            business_process, risk_profile, lifecycle_stage, facts, decision_path
        )
        
        # STEP 5: Calculate confidence
        confidence = self._calculate_confidence(business_process, intent, facts)
        
        # STEP 6: Build explanation
        explanation = self._build_explanation(
            business_process, compliance_domain, lifecycle_stage,
            recommended_actions, decision_path
        )
        
        return ProcessContext(
            business_process=business_process,
            compliance_domain=compliance_domain,
            lifecycle_stage=lifecycle_stage,
            recommended_actions=recommended_actions,
            explanation=explanation,
            decision_path=decision_path,
            confidence=confidence
        )
    
    def _determine_business_process(
        self,
        intent: DocumentIntent,
        trust_role: TrustRole,
        facts: ExtractedFacts,
        decision_path: List[str]
    ) -> BusinessProcess:
        """Determine business process from intent and context"""
        # Get candidate processes from intent
        candidates = self.intent_to_process.get(intent, [])
        
        if not candidates:
            decision_path.append(f"  → No mapping found for intent '{intent.value}', using UNKNOWN")
            return BusinessProcess.UNKNOWN
        
        decision_path.append(f"  → Found {len(candidates)} candidate processes: {[p.value for p in candidates]}")
        
        # Refine based on trust role and facts
        if len(candidates) == 1:
            selected = candidates[0]
            decision_path.append(f"  → Single candidate, selected: {selected.value}")
            return selected
        
        # Multiple candidates - refine based on context
        decision_path.append("  → Multiple candidates, refining based on context")
        
        # Refine based on trust role
        if trust_role == TrustRole.PROOF_OF_IDENTITY:
            for proc in candidates:
                if proc == BusinessProcess.IDENTITY_VERIFICATION:
                    decision_path.append(f"  → Selected based on trust role (identity): {proc.value}")
                    return proc
        
        if trust_role == TrustRole.LEGAL_COMMITMENT:
            for proc in candidates:
                if proc in [BusinessProcess.CONTRACT_MANAGEMENT, BusinessProcess.SERVICE_AGREEMENT_PROCESSING]:
                    decision_path.append(f"  → Selected based on trust role (legal): {proc.value}")
                    return proc
        
        if trust_role == TrustRole.FINANCIAL_EVIDENCE:
            for proc in candidates:
                if proc in [BusinessProcess.INVOICE_PROCESSING, BusinessProcess.PAYMENT_PROCESSING]:
                    decision_path.append(f"  → Selected based on trust role (financial): {proc.value}")
                    return proc
        
        # Refine based on facts
        if facts.party_1_type == "individual" and facts.party_2_type == "company":
            for proc in candidates:
                if proc == BusinessProcess.CONTRACTOR_ONBOARDING:
                    decision_path.append(f"  → Selected based on facts (individual+company): {proc.value}")
                    return proc
        
        if facts.invoice_number:
            for proc in candidates:
                if proc == BusinessProcess.INVOICE_PROCESSING:
                    decision_path.append(f"  → Selected based on facts (invoice number): {proc.value}")
                    return proc
        
        # Default to first candidate
        selected = candidates[0]
        decision_path.append(f"  → No refinement match, using first candidate: {selected.value}")
        return selected
    
    def _determine_compliance_domain(
        self,
        business_process: BusinessProcess,
        trust_role: TrustRole,
        decision_path: List[str]
    ) -> ComplianceDomain:
        """Determine compliance domain"""
        # Try process-based mapping first
        if business_process in self.process_to_domain:
            domain = self.process_to_domain[business_process]
            decision_path.append(f"  → Mapped process '{business_process.value}' to domain: {domain.value}")
            return domain
        
        # Fallback to trust role mapping
        if trust_role in self.trust_role_to_domain:
            domain = self.trust_role_to_domain[trust_role]
            decision_path.append(f"  → Mapped trust role '{trust_role.value}' to domain: {domain.value}")
            return domain
        
        # Default
        decision_path.append("  → No mapping found, using UNKNOWN")
        return ComplianceDomain.UNKNOWN
    
    def _determine_lifecycle_stage(
        self,
        facts: ExtractedFacts,
        business_process: BusinessProcess,
        decision_path: List[str]
    ) -> LifecycleStage:
        """Determine lifecycle stage from facts"""
        # Check each lifecycle rule
        for stage, rule_func in self.lifecycle_rules.items():
            try:
                if rule_func(facts):
                    decision_path.append(f"  → Lifecycle stage '{stage.value}' detected based on dates")
                    return stage
            except Exception:
                continue
        
        # Default based on facts availability
        if facts.effective_date and facts.expiration_date:
            decision_path.append("  → Dates present, defaulting to ACTIVE")
            return LifecycleStage.ACTIVE
        elif facts.issue_date:
            decision_path.append("  → Issue date present, defaulting to VERIFICATION")
            return LifecycleStage.VERIFICATION
        else:
            decision_path.append("  → No dates, defaulting to CREATION")
            return LifecycleStage.CREATION
    
    def _generate_recommended_actions(
        self,
        business_process: BusinessProcess,
        risk_profile: RiskProfile,
        lifecycle_stage: LifecycleStage,
        facts: ExtractedFacts,
        decision_path: List[str]
    ) -> List[str]:
        """Generate recommended actions"""
        actions = []
        
        # Get base actions for process
        if business_process in self.process_actions:
            actions.extend(self.process_actions[business_process])
            decision_path.append(f"  → Added {len(self.process_actions[business_process])} base actions for process")
        
        # Add risk-based actions
        if risk_profile == RiskProfile.HIGH:
            actions.extend([
                "Require human review",
                "Verify all supporting documents",
                "Escalate to compliance team",
            ])
            decision_path.append("  → Added high-risk actions")
        elif risk_profile == RiskProfile.MEDIUM:
            actions.extend([
                "Review key fields",
                "Verify critical information",
            ])
            decision_path.append("  → Added medium-risk actions")
        
        # Add lifecycle-based actions
        if lifecycle_stage == LifecycleStage.EXPIRATION:
            actions.extend([
                "Check renewal requirements",
                "Notify stakeholders of expiration",
                "Initiate renewal process",
            ])
            decision_path.append("  → Added expiration-stage actions")
        elif lifecycle_stage == LifecycleStage.AUDIT:
            actions.extend([
                "Review compliance status",
                "Check for updates",
                "Verify ongoing validity",
            ])
            decision_path.append("  → Added audit-stage actions")
        
        # Remove duplicates and limit
        actions = list(dict.fromkeys(actions))[:10]
        
        decision_path.append(f"  → Final action list: {len(actions)} actions")
        
        return actions
    
    def _calculate_confidence(
        self,
        business_process: BusinessProcess,
        intent: DocumentIntent,
        facts: ExtractedFacts
    ) -> float:
        """Calculate confidence in process context analysis"""
        confidence = 0.5  # Base confidence
        
        # Boost if we have exact intent mapping
        if intent in self.intent_to_process:
            confidence += 0.2
        
        # Boost if process has specific actions
        if business_process in self.process_actions:
            confidence += 0.15
        
        # Boost if we have extracted facts
        fact_count = len(facts.fields_extracted) if facts.fields_extracted else 0
        if fact_count > 0:
            confidence += min(fact_count * 0.02, 0.15)
        
        return min(confidence, 0.95)
    
    def _build_explanation(
        self,
        business_process: BusinessProcess,
        compliance_domain: ComplianceDomain,
        lifecycle_stage: LifecycleStage,
        recommended_actions: List[str],
        decision_path: List[str]
    ) -> str:
        """Build human-readable explanation"""
        parts = [
            f"Process Context Analysis",
            f"=" * 60,
            f"",
            f"Business Process: {business_process.value}",
            f"  → This document belongs to the '{self._process_description(business_process)}' process",
            f"",
            f"Compliance Domain: {compliance_domain.value}",
            f"  → Managed by: {self._domain_description(compliance_domain)}",
            f"",
            f"Lifecycle Stage: {lifecycle_stage.value}",
            f"  → Current stage: {self._lifecycle_description(lifecycle_stage)}",
            f"",
            f"Recommended Actions ({len(recommended_actions)}):",
        ]
        
        for i, action in enumerate(recommended_actions, 1):
            parts.append(f"  {i}. {action}")
        
        parts.extend([
            f"",
            f"Decision Path:",
        ])
        
        for step in decision_path:
            parts.append(f"  {step}")
        
        return "\n".join(parts)
    
    def _process_description(self, process: BusinessProcess) -> str:
        """Human-readable description of business process"""
        descriptions = {
            BusinessProcess.CONTRACTOR_ONBOARDING: "contractor onboarding and registration",
            BusinessProcess.SUPPLIER_QUALIFICATION: "supplier qualification and vetting",
            BusinessProcess.EMPLOYEE_CERTIFICATION: "employee certification verification",
            BusinessProcess.REGULATORY_AUDIT: "regulatory compliance audit",
            BusinessProcess.FINANCIAL_REPORTING: "financial reporting and documentation",
            BusinessProcess.INVOICE_PROCESSING: "invoice processing and payment",
            BusinessProcess.CONTRACT_MANAGEMENT: "contract management and administration",
            BusinessProcess.IDENTITY_VERIFICATION: "identity verification and validation",
        }
        return descriptions.get(process, process.value.replace("_", " "))
    
    def _domain_description(self, domain: ComplianceDomain) -> str:
        """Human-readable description of compliance domain"""
        descriptions = {
            ComplianceDomain.HR: "Human Resources department",
            ComplianceDomain.LEGAL: "Legal department",
            ComplianceDomain.FINANCE: "Finance department",
            ComplianceDomain.OPERATIONS: "Operations department",
            ComplianceDomain.COMPLIANCE: "Compliance department",
            ComplianceDomain.PROCUREMENT: "Procurement department",
            ComplianceDomain.IT: "IT department",
            ComplianceDomain.SECURITY: "Security department",
        }
        return descriptions.get(domain, domain.value)
    
    def _lifecycle_description(self, stage: LifecycleStage) -> str:
        """Human-readable description of lifecycle stage"""
        descriptions = {
            LifecycleStage.CREATION: "Document is being created",
            LifecycleStage.VERIFICATION: "Document is being verified",
            LifecycleStage.ACTIVE: "Document is active and in use",
            LifecycleStage.AUDIT: "Document is being audited",
            LifecycleStage.RENEWAL: "Document is being renewed",
            LifecycleStage.EXPIRATION: "Document is expiring or expired",
            LifecycleStage.ARCHIVAL: "Document is archived",
        }
        return descriptions.get(stage, stage.value)
    
    def _is_active(self, facts: ExtractedFacts) -> bool:
        """Check if document is currently active"""
        # Simplified - would need actual date comparison in production
        return True  # Assume active if both dates present
    
    def _is_past_midpoint(self, facts: ExtractedFacts) -> bool:
        """Check if document is past midpoint (audit stage)"""
        # Simplified - would need actual date comparison
        return False
    
    def _is_expiring_soon(self, facts: ExtractedFacts) -> bool:
        """Check if document is expiring soon"""
        # Simplified - would need actual date comparison
        return False
