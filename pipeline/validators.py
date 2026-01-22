"""
Validation module for extracted data
"""

from typing import List, Dict, Any
from .schemas import BaseDocumentSchema


class DocumentValidator:
    """Validates extracted document data"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate(self, schema: BaseDocumentSchema) -> Dict[str, Any]:
        """
        Validate extracted document schema - OPERATIONAL FOCUS
        
        Returns certification readiness, not just "is valid"
        
        Args:
            schema: Extracted document schema
            
        Returns:
            Validation result with certification decision
        """
        self.errors = []
        self.warnings = []
        
        # Basic validation
        if not schema.raw_text or len(schema.raw_text.strip()) < 10:
            self.errors.append("Insufficient text extracted")
            return {
                'certification_ready': False,
                'human_review_required': True,
                'errors': self.errors,
                'warnings': self.warnings,
                'confidence': 0.0,
                'reason': 'insufficient_text'
            }
        
        # Use schema's own certification readiness (calculated during extraction)
        # But add additional validation checks
        
        # Type-specific validation
        if schema.document_type == 'invoice':
            self._validate_invoice(schema)
        elif schema.document_type == 'diploma':
            self._validate_diploma(schema)
        elif schema.document_type == 'id':
            self._validate_id(schema)
        elif schema.document_type == 'driving_license':
            self._validate_driving_license(schema)
        
        # Final certification decision
        # Even if extraction says ready, validator can block
        certification_ready = schema.certification_ready and len(self.errors) == 0
        
        # For ID documents, be extra strict
        if schema.document_type == 'id':
            if schema.confidence < 0.85:
                certification_ready = False
                self.warnings.append(f"Confidence {schema.confidence:.2f} below threshold 0.85 for ID documents")
            
            if schema.missing_fields:
                certification_ready = False
                self.warnings.append(f"Missing required fields: {', '.join(schema.missing_fields)}")
            
            if schema.trusted_source != 'mrz':
                self.warnings.append("No MRZ found - lower reliability")
        
        # Human review required if:
        # - Not certification ready
        # - Has errors
        # - Low confidence
        human_review_required = (
            not certification_ready or
            len(self.errors) > 0 or
            schema.confidence < 0.7
        )
        
        return {
            'certification_ready': certification_ready,
            'human_review_required': human_review_required,
            'errors': self.errors,
            'warnings': self.warnings,
            'confidence': schema.confidence,
            'missing_fields': getattr(schema, 'missing_fields', []),
            'trusted_source': getattr(schema, 'trusted_source', None),
            'reason': 'validation_passed' if certification_ready else 'validation_failed'
        }
    
    def _validate_invoice(self, schema: BaseDocumentSchema):
        """Validate invoice-specific fields"""
        from .schemas import InvoiceSchema
        
        if not isinstance(schema, InvoiceSchema):
            return
        
        # Check critical fields
        if not schema.invoice_number:
            self.warnings.append("Missing invoice number")
        
        if not schema.total_amount:
            self.errors.append("Missing total amount")
        
        if schema.total_amount and schema.total_amount <= 0:
            self.errors.append("Invalid total amount")
        
        # Validate VAT calculation if both amounts present
        if schema.total_amount and schema.vat_amount and schema.net_amount:
            calculated_total = schema.net_amount + schema.vat_amount
            if abs(float(schema.total_amount - calculated_total)) > 0.01:
                self.warnings.append("VAT calculation mismatch")
    
    def _validate_diploma(self, schema: BaseDocumentSchema):
        """Validate diploma-specific fields"""
        from .schemas import DiplomaSchema
        
        if not isinstance(schema, DiplomaSchema):
            return
        
        if not schema.student_name:
            self.warnings.append("Missing student name")
        
        if not schema.university_name:
            self.warnings.append("Missing university name")
        
        if schema.cfu_total and schema.cfu_earned:
            if schema.cfu_earned > schema.cfu_total:
                self.errors.append("CFU earned exceeds total")
    
    def _validate_id(self, schema: BaseDocumentSchema):
        """
        Validate ID document-specific fields - STRICT
        
        For ID documents, missing critical fields = error, not warning
        """
        from .schemas import IDDocumentSchema
        
        if not isinstance(schema, IDDocumentSchema):
            return
        
        # Critical fields - missing = error
        if not schema.first_name:
            self.errors.append("Missing first name (critical)")
        
        if not schema.last_name:
            self.errors.append("Missing last name (critical)")
        
        if not schema.date_of_birth:
            self.errors.append("Missing date of birth (critical)")
        
        # Validate tax code format if present
        if schema.tax_code:
            if len(schema.tax_code) != 16:
                self.errors.append("Invalid tax code format")
            elif not schema.tax_code.isalnum():
                self.errors.append("Tax code contains invalid characters")
        
        # Validate dates
        if schema.issue_date and schema.expiry_date:
            if schema.expiry_date < schema.issue_date:
                self.errors.append("Expiry date before issue date")
    
    def _validate_driving_license(self, schema: BaseDocumentSchema):
        """Validate driving license-specific fields"""
        from .schemas import DrivingLicenseSchema
        
        if not isinstance(schema, DrivingLicenseSchema):
            return
        
        # Critical fields
        if not schema.first_name:
            self.errors.append("Missing first name (critical)")
        
        if not schema.last_name:
            self.errors.append("Missing last name (critical)")
        
        if not schema.license_number:
            self.errors.append("Missing license number (critical)")
        
        if not schema.expiry_date:
            self.errors.append("Missing expiry date (critical)")
        
        # Validate expiry date is in the future (for active licenses)
        if schema.expiry_date:
            from datetime import datetime
            if schema.expiry_date < datetime.now():
                self.warnings.append("License appears to be expired")
        
        # Check field confidence for critical fields
        if hasattr(schema, 'field_confidence') and schema.field_confidence:
            critical_fields = ['first_name', 'last_name', 'license_number', 'expiry_date']
            for field in critical_fields:
                if field in schema.field_confidence:
                    conf = schema.field_confidence[field]
                    if conf < 0.7:
                        self.warnings.append(f"Low confidence for {field}: {conf:.2f}")
        
        # Check field confidence for critical fields
        if hasattr(schema, 'field_confidence') and schema.field_confidence:
            critical_fields = ['first_name', 'last_name', 'date_of_birth']
            for field in critical_fields:
                if field in schema.field_confidence:
                    conf = schema.field_confidence[field]
                    if conf < 0.7:
                        self.warnings.append(f"Low confidence for {field}: {conf:.2f}")
