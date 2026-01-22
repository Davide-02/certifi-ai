"""
Decision Engine - Centralized certification decision logic
This is the core business logic that determines if a document can be certified
"""

from typing import Dict, Any, Optional, Literal
from enum import Enum


class CertificationProfile(str, Enum):
    """Certification profiles with different requirements"""
    IDENTITY_MINIMAL = "identity_minimal"  # MRZ fields only
    IDENTITY_STRICT = "identity_strict"  # MRZ + place_of_birth + tax_code
    INVOICE_MINIMAL = "invoice_minimal"  # Basic invoice fields
    INVOICE_STRICT = "invoice_strict"  # All invoice fields + validation
    DIPLOMA_MINIMAL = "diploma_minimal"  # Basic diploma fields
    DIPLOMA_STRICT = "diploma_strict"  # All diploma fields
    DRIVING_LICENSE_MINIMAL = "driving_license_minimal"  # Basic license fields
    DRIVING_LICENSE_STRICT = "driving_license_strict"  # All license fields


class RiskLevel(str, Enum):
    """Risk levels for certification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionEngine:
    """
    Centralized decision engine for certification
    
    Mental model:
    - Never trust 100% (no confidence = 1.0)
    - Classification confidence must align with extraction
    - Profile determines requirements
    - Risk level determines human review
    """
    
    # Thresholds (never 1.0)
    MIN_CLASSIFICATION_CONFIDENCE = 0.7
    MIN_EXTRACTION_CONFIDENCE = 0.85
    MAX_CONFIDENCE = 0.98  # Never reach 1.0
    
    # Risk factors by source
    RISK_FACTORS = {
        'mrz': 0.95,  # MRZ is highly reliable but not perfect
        'layout_rules': 0.90,  # Structured layout is very reliable
        'ocr': 0.75,  # OCR is less reliable
        'llm': 0.80,  # LLM is good but not perfect
    }
    
    def __init__(self):
        self.profiles = self._define_profiles()
    
    def _define_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Define certification profiles and their requirements"""
        return {
            CertificationProfile.IDENTITY_MINIMAL: {
                'required_fields': ['first_name', 'last_name', 'date_of_birth'],
                'optional_fields': ['place_of_birth', 'tax_code', 'address'],
                'min_confidence': 0.85,
                'preferred_source': 'mrz'
            },
            CertificationProfile.IDENTITY_STRICT: {
                'required_fields': ['first_name', 'last_name', 'date_of_birth', 'place_of_birth', 'tax_code'],
                'optional_fields': ['address', 'city', 'postal_code'],
                'min_confidence': 0.90,
                'preferred_source': 'mrz'
            },
            CertificationProfile.INVOICE_MINIMAL: {
                'required_fields': ['invoice_number', 'total_amount', 'invoice_date'],
                'optional_fields': ['vat_amount', 'seller_name', 'buyer_name'],
                'min_confidence': 0.80,
                'preferred_source': 'ocr'
            },
            CertificationProfile.INVOICE_STRICT: {
                'required_fields': ['invoice_number', 'total_amount', 'invoice_date', 'vat_amount', 'seller_name'],
                'optional_fields': ['buyer_name', 'seller_vat', 'buyer_vat'],
                'min_confidence': 0.85,
                'preferred_source': 'ocr'
            },
            CertificationProfile.DIPLOMA_MINIMAL: {
                'required_fields': ['student_name', 'university_name', 'degree_type'],
                'optional_fields': ['graduation_date', 'final_grade', 'cfu_total'],
                'min_confidence': 0.80,
                'preferred_source': 'ocr'
            },
            CertificationProfile.DIPLOMA_STRICT: {
                'required_fields': ['student_name', 'university_name', 'degree_type', 'graduation_date', 'final_grade'],
                'optional_fields': ['cfu_total', 'thesis_title'],
                'min_confidence': 0.85,
                'preferred_source': 'ocr'
            },
            CertificationProfile.DRIVING_LICENSE_MINIMAL: {
                'required_fields': ['first_name', 'last_name', 'license_number', 'expiry_date'],
                'optional_fields': ['date_of_birth', 'place_of_birth', 'issue_date', 'categories'],
                'min_confidence': 0.85,
                'preferred_source': 'layout_rules'
            },
            CertificationProfile.DRIVING_LICENSE_STRICT: {
                'required_fields': ['first_name', 'last_name', 'license_number', 'expiry_date', 'date_of_birth', 'issue_date'],
                'optional_fields': ['place_of_birth', 'categories', 'address'],
                'min_confidence': 0.90,
                'preferred_source': 'layout_rules'
            }
        }
    
    def decide(
        self,
        document_type: str,
        classification_confidence: float,
        extraction_confidence: float,
        trusted_source: Optional[str],
        field_confidence: Dict[str, float],
        missing_fields: list[str],
        profile: Optional[CertificationProfile] = None
    ) -> Dict[str, Any]:
        """
        Make certification decision
        
        Args:
            document_type: Type of document
            classification_confidence: Confidence in document type classification
            extraction_confidence: Overall extraction confidence
            trusted_source: Source of truth (mrz, ocr, llm)
            field_confidence: Confidence per field
            missing_fields: List of missing fields
            profile: Certification profile to use
            
        Returns:
            Decision dictionary with certification_ready, reason, risk_level, etc.
        """
        
        # Step 1: Determine profile if not provided
        if not profile:
            profile = self._infer_profile(document_type)
        
        profile_config = self.profiles.get(profile, self.profiles[CertificationProfile.IDENTITY_MINIMAL])
        
        # Step 2: Check classification confidence
        # CRITICAL: Classification must be reliable
        if classification_confidence < self.MIN_CLASSIFICATION_CONFIDENCE:
            return {
                'certification_ready': False,
                'human_review_required': True,
                'reason': 'low_classification_confidence',
                'risk_level': RiskLevel.HIGH,
                'confidence': classification_confidence,
                'details': {
                    'classification_confidence': classification_confidence,
                    'threshold': self.MIN_CLASSIFICATION_CONFIDENCE
                }
            }
        
        # Step 3: Align classification confidence with trusted source
        # If MRZ is used, classification should reflect that
        if trusted_source == 'mrz' and classification_confidence < 0.8:
            # MRZ override: MRZ itself confirms document type
            classification_confidence = 0.90  # Boost but never 1.0
        
        # Step 4: Calculate adjusted confidence (never 1.0)
        base_confidence = min(extraction_confidence, classification_confidence)
        
        # Apply risk factor based on source
        risk_factor = self.RISK_FACTORS.get(trusted_source, 0.75)
        adjusted_confidence = min(
            base_confidence * risk_factor,
            self.MAX_CONFIDENCE  # Cap at 0.98
        )
        
        # Step 5: Check required fields
        required_fields = profile_config['required_fields']
        missing_required = [f for f in required_fields if f in missing_fields]
        
        if missing_required:
            return {
                'certification_ready': False,
                'human_review_required': True,
                'reason': 'missing_required_fields',
                'risk_level': RiskLevel.MEDIUM,
                'confidence': adjusted_confidence,
                'details': {
                    'missing_required_fields': missing_required,
                    'profile': profile.value
                }
            }
        
        # Step 6: Check minimum confidence for profile
        min_confidence = profile_config['min_confidence']
        if adjusted_confidence < min_confidence:
            return {
                'certification_ready': False,
                'human_review_required': True,
                'reason': 'below_profile_threshold',
                'risk_level': RiskLevel.MEDIUM,
                'confidence': adjusted_confidence,
                'details': {
                    'current_confidence': adjusted_confidence,
                    'required_confidence': min_confidence,
                    'profile': profile.value
                }
            }
        
        # Step 7: Check field confidence for critical fields
        critical_fields = required_fields[:3]  # First 3 are most critical
        low_confidence_fields = [
            f for f in critical_fields
            if f in field_confidence and field_confidence[f] < 0.7
        ]
        
        if low_confidence_fields:
            return {
                'certification_ready': False,
                'human_review_required': True,
                'reason': 'low_field_confidence',
                'risk_level': RiskLevel.MEDIUM,
                'confidence': adjusted_confidence,
                'details': {
                    'low_confidence_fields': low_confidence_fields,
                    'field_confidence': {f: field_confidence[f] for f in low_confidence_fields}
                }
            }
        
        # Step 8: Determine risk level
        risk_level = self._calculate_risk_level(
            adjusted_confidence,
            trusted_source,
            missing_fields,
            profile_config
        )
        
        # Step 9: Final decision
        # Even if all checks pass, human review might be needed for high risk
        human_review_required = risk_level == RiskLevel.HIGH
        
        return {
            'certification_ready': True,
            'human_review_required': human_review_required,
            'reason': f'{profile.value}_valid',
            'risk_level': risk_level,
            'confidence': adjusted_confidence,
            'profile': profile.value,
            'details': {
                'classification_confidence': classification_confidence,
                'extraction_confidence': extraction_confidence,
                'adjusted_confidence': adjusted_confidence,
                'trusted_source': trusted_source,
                'missing_optional_fields': [
                    f for f in profile_config.get('optional_fields', [])
                    if f in missing_fields
                ]
            }
        }
    
    def _infer_profile(self, document_type: str) -> CertificationProfile:
        """Infer certification profile from document type"""
        if document_type == 'id':
            return CertificationProfile.IDENTITY_MINIMAL
        elif document_type == 'invoice':
            return CertificationProfile.INVOICE_MINIMAL
        elif document_type == 'diploma':
            return CertificationProfile.DIPLOMA_MINIMAL
        elif document_type == 'driving_license':
            return CertificationProfile.DRIVING_LICENSE_MINIMAL
        else:
            return CertificationProfile.IDENTITY_MINIMAL  # Default
    
    def _calculate_risk_level(
        self,
        confidence: float,
        trusted_source: Optional[str],
        missing_fields: list[str],
        profile_config: Dict[str, Any]
    ) -> RiskLevel:
        """
        Calculate risk level for certification
        
        Args:
            confidence: Adjusted confidence
            trusted_source: Source of truth
            missing_fields: Missing fields
            profile_config: Profile configuration
            
        Returns:
            Risk level
        """
        # High risk factors
        if confidence < 0.85:
            return RiskLevel.HIGH
        if trusted_source != profile_config.get('preferred_source'):
            return RiskLevel.MEDIUM
        if len(missing_fields) > 2:
            return RiskLevel.MEDIUM
        
        # Low risk: high confidence, preferred source, minimal missing fields
        if confidence >= 0.90 and trusted_source == profile_config.get('preferred_source'):
            return RiskLevel.LOW
        
        return RiskLevel.MEDIUM
