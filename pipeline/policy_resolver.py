"""
Policy Resolver - Determines certification policy based on document family

This is the CORE business logic: CertiFi certifies POLICIES, not documents
"""

from typing import Dict, Any, Optional
from enum import Enum
from .family_classifier import DocumentFamily


class CertificationPolicy(str, Enum):
    """Certification policies - how to certify each family"""
    HASH_ONLY = "hash_only"  # Just hash + timestamp (contracts, generic docs)
    IDENTITY_MINIMAL = "identity_minimal"  # MRZ/layout fields (ID, passport)
    IDENTITY_STRICT = "identity_strict"  # ID with all fields
    DRIVING_LICENSE_MINIMAL = "driving_license_minimal"  # License basic fields
    FINANCIAL_MINIMAL = "financial_minimal"  # Invoice basic fields
    CERTIFICATE_MINIMAL = "certificate_minimal"  # Certificate basic fields
    CORPORATE_MINIMAL = "corporate_minimal"  # Corporate doc basic fields
    UNKNOWN = "unknown"  # Cannot certify


class PolicyResolver:
    """
    Resolves certification policy based on document family
    
    Key insight: CertiFi certifies the POLICY, not the document itself
    """
    
    # Policy mapping: family → policy
    FAMILY_POLICIES = {
        DocumentFamily.IDENTITY: {
            'default_policy': CertificationPolicy.IDENTITY_MINIMAL,
            'certifiable': True,
            'requires_extraction': True,
            'trusted_sources': ['mrz', 'layout_rules'],
            'min_confidence': 0.50  # Threshold for family classification (extraction will have stricter requirements)
        },
        DocumentFamily.DRIVING_LICENSE: {
            'default_policy': CertificationPolicy.DRIVING_LICENSE_MINIMAL,
            'certifiable': True,
            'requires_extraction': True,
            'trusted_sources': ['layout_rules'],
            'min_confidence': 0.50  # Threshold for family classification
        },
        DocumentFamily.CONTRACT: {
            'default_policy': CertificationPolicy.HASH_ONLY,
            'certifiable': True,
            'requires_extraction': False,  # Contracts: hash + signature verification
            'trusted_sources': ['file_integrity'],
            'min_confidence': 0.60  # Lower threshold - hash is enough, less strict
        },
        DocumentFamily.CERTIFICATE: {
            'default_policy': CertificationPolicy.CERTIFICATE_MINIMAL,
            'certifiable': True,
            'requires_extraction': True,
            'trusted_sources': ['ocr', 'layout_rules'],
            'min_confidence': 0.50  # Threshold for family classification
        },
        DocumentFamily.FINANCIAL: {
            'default_policy': CertificationPolicy.FINANCIAL_MINIMAL,
            'certifiable': True,
            'requires_extraction': True,
            'trusted_sources': ['ocr'],
            'min_confidence': 0.50  # Threshold for family classification
        },
        DocumentFamily.CORPORATE: {
            'default_policy': CertificationPolicy.CORPORATE_MINIMAL,
            'certifiable': True,
            'requires_extraction': True,
            'trusted_sources': ['ocr'],
            'min_confidence': 0.50  # Threshold for family classification
        },
        DocumentFamily.UNKNOWN: {
            'default_policy': CertificationPolicy.UNKNOWN,
            'certifiable': False,
            'requires_extraction': False,
            'trusted_sources': [],
            'min_confidence': 0.0
        }
    }
    
    def resolve(
        self,
        family: DocumentFamily,
        family_confidence: float,
        trusted_source: Optional[str] = None,
        use_claim_based: bool = False
    ) -> Dict[str, Any]:
        """
        Resolve certification policy for document family
        
        KEY INSIGHT: For semantic documents (contracts, SOW, letters),
        we use claim-based certification, not family_confidence gate.
        
        Args:
            family: Document family
            family_confidence: Confidence in family classification
            trusted_source: Source of truth (if already known)
            use_claim_based: If True, ignore family_confidence gate for semantic docs
            
        Returns:
            Policy decision with certifiable, policy, requirements, etc.
        """
        policy_config = self.FAMILY_POLICIES.get(family, self.FAMILY_POLICIES[DocumentFamily.UNKNOWN])
        
        # SEPARATE TWO WORLDS:
        # 1. Structured documents (ID, passport) - require high family_confidence
        # 2. Semantic documents (contract, SOW) - use claim-based, ignore family_confidence gate
        structured_families = [DocumentFamily.IDENTITY, DocumentFamily.DRIVING_LICENSE]
        semantic_families = [DocumentFamily.CONTRACT, DocumentFamily.CERTIFICATE, DocumentFamily.FINANCIAL, DocumentFamily.CORPORATE]
        
        is_structured = family in structured_families
        is_semantic = family in semantic_families
        
        # For structured documents, family_confidence is a hard gate
        if is_structured and family_confidence < policy_config['min_confidence']:
            return {
                'certifiable': False,
                'policy': CertificationPolicy.UNKNOWN,
                'requires_extraction': False,
                'human_review_required': True,
                'reason': 'low_family_confidence',
                'details': {
                    'family_confidence': family_confidence,
                    'required_confidence': policy_config['min_confidence'],
                    'document_type': 'structured'
                }
            }
        
        # For semantic documents, family_confidence is NOT a gate
        # We'll use claim-based certification instead
        if is_semantic or use_claim_based:
            return {
                'certifiable': True,  # Will be decided by claim evaluation
                'policy': policy_config['default_policy'],
                'requires_extraction': policy_config['requires_extraction'],
                'trusted_sources': policy_config['trusted_sources'],
                'min_confidence': policy_config['min_confidence'],
                'human_review_required': False,  # Will be set by claim evaluation
                'reason': f'{family.value}_claim_based',
                'certification_method': 'claim_based',  # NEW: indicates claim-based certification
                'details': {
                    'family': family.value,
                    'family_confidence': family_confidence,
                    'note': 'Family confidence not used as gate for semantic documents'
                }
            }
        
        # Check if trusted source matches expected sources
        if trusted_source and policy_config['trusted_sources']:
            if trusted_source not in policy_config['trusted_sources']:
                # Warning but not blocking
                source_warning = f"Source {trusted_source} not in preferred {policy_config['trusted_sources']}"
            else:
                source_warning = None
        else:
            source_warning = None
        
        # Build policy decision
        decision = {
            'certifiable': policy_config['certifiable'],
            'policy': policy_config['default_policy'],
            'requires_extraction': policy_config['requires_extraction'],
            'trusted_sources': policy_config['trusted_sources'],
            'min_confidence': policy_config['min_confidence'],
            'human_review_required': family_confidence < 0.85,  # Review if uncertain
            'reason': f'{family.value}_policy',
            'details': {
                'family': family.value,
                'family_confidence': family_confidence,
                'source_warning': source_warning
            }
        }
        
        return decision
    
    def get_required_fields(self, policy: CertificationPolicy) -> list[str]:
        """
        Get required fields for a policy
        
        Args:
            policy: Certification policy
            
        Returns:
            List of required field names
        """
        policy_fields = {
            CertificationPolicy.IDENTITY_MINIMAL: ['first_name', 'last_name', 'date_of_birth'],
            CertificationPolicy.IDENTITY_STRICT: ['first_name', 'last_name', 'date_of_birth', 'place_of_birth', 'tax_code'],
            CertificationPolicy.DRIVING_LICENSE_MINIMAL: ['first_name', 'last_name', 'license_number', 'expiry_date'],
            CertificationPolicy.FINANCIAL_MINIMAL: ['invoice_number', 'total_amount', 'invoice_date'],
            CertificationPolicy.CERTIFICATE_MINIMAL: ['student_name', 'university_name', 'degree_type'],
            CertificationPolicy.HASH_ONLY: [],  # No fields required - just hash
            CertificationPolicy.CORPORATE_MINIMAL: [],  # TBD
            CertificationPolicy.UNKNOWN: []
        }
        
        return policy_fields.get(policy, [])
