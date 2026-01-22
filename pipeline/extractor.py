"""
Information extraction module
Uses regex rules first, LLM as fallback
"""

import re
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from dateutil import parser

from .schemas import InvoiceSchema, DiplomaSchema, IDDocumentSchema, DrivingLicenseSchema, BaseDocumentSchema
from .mrz_parser import MRZParser


class InformationExtractor:
    """Extracts structured information from text"""
    
    def __init__(self, use_llm: bool = False, llm_provider: str = "openai"):
        """
        Initialize extractor
        
        Args:
            use_llm: Whether to use LLM for extraction
            llm_provider: LLM provider ('openai' or 'anthropic')
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.mrz_parser = MRZParser()
    
    def extract(self, text: str, document_type: str) -> BaseDocumentSchema:
        """
        Extract information based on document type
        
        Args:
            text: Document text
            document_type: Type of document
            
        Returns:
            Extracted data as Pydantic schema
        """
        if document_type == 'invoice':
            return self._extract_invoice(text)
        elif document_type == 'diploma':
            return self._extract_diploma(text)
        elif document_type == 'id':
            return self._extract_id(text)
        elif document_type == 'driving_license':
            return self._extract_driving_license(text)
        else:
            # Unknown type - return base schema
            return BaseDocumentSchema(
                document_type=document_type,
                confidence=0.0,
                raw_text=text
            )
    
    def _extract_invoice(self, text: str) -> InvoiceSchema:
        """Extract invoice data using regex patterns"""
        
        text_lower = text.lower()
        data = {}
        
        # Invoice number
        invoice_num_match = re.search(
            r'fattura\s+n[°º]?\s*:?\s*([A-Z0-9/-]+)',
            text,
            re.IGNORECASE
        )
        if invoice_num_match:
            data['invoice_number'] = invoice_num_match.group(1).strip()
        
        # Date
        date_match = re.search(
            r'data\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            text,
            re.IGNORECASE
        )
        if date_match:
            try:
                data['invoice_date'] = parser.parse(date_match.group(1), dayfirst=True)
            except:
                pass
        
        # Total amount
        total_match = re.search(
            r'totale\s*:?\s*€?\s*([\d.,]+)',
            text,
            re.IGNORECASE
        )
        if total_match:
            amount_str = total_match.group(1).replace(',', '.')
            try:
                data['total_amount'] = Decimal(amount_str)
            except:
                pass
        
        # VAT
        vat_match = re.search(
            r'iva\s*:?\s*€?\s*([\d.,]+)',
            text,
            re.IGNORECASE
        )
        if vat_match:
            vat_str = vat_match.group(1).replace(',', '.')
            try:
                data['vat_amount'] = Decimal(vat_str)
            except:
                pass
        
        # VAT rate
        vat_rate_match = re.search(
            r'iva\s*(\d+[.,]?\d*)%',
            text,
            re.IGNORECASE
        )
        if vat_rate_match:
            rate_str = vat_rate_match.group(1).replace(',', '.')
            try:
                data['vat_rate'] = float(rate_str)
            except:
                pass
        
        # Seller name (look for common patterns)
        seller_match = re.search(
            r'(?:venditore|fornitore|emittente)\s*:?\s*([A-Z][^,\n]+)',
            text,
            re.IGNORECASE
        )
        if seller_match:
            data['seller_name'] = seller_match.group(1).strip()
        
        # Buyer name
        buyer_match = re.search(
            r'(?:cliente|acquirente|destinatario)\s*:?\s*([A-Z][^,\n]+)',
            text,
            re.IGNORECASE
        )
        if buyer_match:
            data['buyer_name'] = buyer_match.group(1).strip()
        
        # If LLM is enabled and extraction is incomplete, use LLM
        if self.use_llm and len(data) < 3:
            return self._extract_with_llm(text, 'invoice')
        
        return InvoiceSchema(
            **data,
            confidence=self._calculate_confidence(data, 'invoice'),
            raw_text=text
        )
    
    def _extract_diploma(self, text: str) -> DiplomaSchema:
        """Extract diploma data using regex patterns"""
        
        data = {}
        
        # Student name
        name_match = re.search(
            r'(?:studente|candidato)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            text,
            re.IGNORECASE
        )
        if name_match:
            data['student_name'] = name_match.group(1).strip()
        
        # University
        uni_match = re.search(
            r'universit[àa]\s+(?:degli?\s+)?studi\s+di\s+([A-Z][^,\n]+)',
            text,
            re.IGNORECASE
        )
        if uni_match:
            data['university_name'] = uni_match.group(1).strip()
        
        # Degree type
        degree_match = re.search(
            r'(laurea\s+(?:triennale|magistrale|specialistica))',
            text,
            re.IGNORECASE
        )
        if degree_match:
            data['degree_type'] = degree_match.group(1).strip()
        
        # CFU
        cfu_match = re.search(
            r'cfu\s*:?\s*(\d+)',
            text,
            re.IGNORECASE
        )
        if cfu_match:
            try:
                data['cfu_total'] = int(cfu_match.group(1))
            except:
                pass
        
        # Graduation date
        grad_date_match = re.search(
            r'(?:data\s+di\s+laurea|conseguito\s+il)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            text,
            re.IGNORECASE
        )
        if grad_date_match:
            try:
                data['graduation_date'] = parser.parse(grad_date_match.group(1), dayfirst=True)
            except:
                pass
        
        if self.use_llm and len(data) < 3:
            return self._extract_with_llm(text, 'diploma')
        
        return DiplomaSchema(
            **data,
            confidence=self._calculate_confidence(data, 'diploma'),
            raw_text=text
        )
    
    def _extract_id(self, text: str) -> IDDocumentSchema:
        """
        Extract ID document data - MRZ is source of truth
        
        Strategy:
        1. Try MRZ first (most reliable)
        2. Fallback to OCR/regex
        3. Calculate field-level confidence
        4. Determine certification readiness
        """
        
        data = {}
        field_confidence = {}
        trusted_source = None
        missing_fields = []
        
        # STEP 1: Try MRZ first (source of truth)
        mrz_result = self.mrz_parser.parse(text)
        
        if mrz_result['found'] and mrz_result['confidence'] > 0.8:
            trusted_source = 'mrz'
            mrz_data = mrz_result['data']
            
            # Extract from MRZ with high confidence
            if 'surname' in mrz_data:
                data['last_name'] = mrz_data['surname']
                field_confidence['last_name'] = 0.95
            
            if 'given_names' in mrz_data:
                # Split given names - first is usually first_name
                given_names = mrz_data['given_names'].split()
                if given_names:
                    data['first_name'] = given_names[0]
                    field_confidence['first_name'] = 0.95
                    if len(given_names) > 1:
                        data['full_name'] = f"{mrz_data['surname']} {mrz_data['given_names']}"
            
            if 'date_of_birth' in mrz_data:
                data['date_of_birth'] = mrz_data['date_of_birth']
                field_confidence['date_of_birth'] = 0.95
            
            if 'document_number' in mrz_data:
                data['document_number'] = mrz_data['document_number']
                field_confidence['document_number'] = 0.95
            
            if 'nationality' in mrz_data:
                data['nationality'] = mrz_data['nationality']
                field_confidence['nationality'] = 0.95
            
            if 'expiry_date' in mrz_data:
                data['expiry_date'] = mrz_data['expiry_date']
                field_confidence['expiry_date'] = 0.95
        
        # STEP 2: Fallback to OCR/regex for missing fields
        if not data.get('first_name'):
            name_match = re.search(
                r'nome\s*:?\s*([A-Z][a-z]+)',
                text,
                re.IGNORECASE
            )
            if name_match:
                data['first_name'] = name_match.group(1).strip()
                field_confidence['first_name'] = 0.6  # Lower confidence from OCR
        
        if not data.get('last_name'):
            surname_match = re.search(
                r'cognome\s*:?\s*([A-Z][a-z]+)',
                text,
                re.IGNORECASE
            )
            if surname_match:
                data['last_name'] = surname_match.group(1).strip()
                field_confidence['last_name'] = 0.6
        
        if not data.get('date_of_birth'):
            dob_match = re.search(
                r'data\s+di\s+nascita\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                text,
                re.IGNORECASE
            )
            if dob_match:
                try:
                    data['date_of_birth'] = parser.parse(dob_match.group(1), dayfirst=True)
                    field_confidence['date_of_birth'] = 0.7
                except:
                    pass
        
        # Tax code (codice fiscale) - only from OCR
        tax_code_match = re.search(
            r'codice\s+fiscale\s*:?\s*([A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z])',
            text,
            re.IGNORECASE
        )
        if tax_code_match:
            data['tax_code'] = tax_code_match.group(1).upper()
            field_confidence['tax_code'] = 0.8
        
        if not data.get('document_number'):
            doc_num_match = re.search(
                r'(?:numero|n[°º])\s*(?:documento|carta)\s*:?\s*([A-Z0-9]+)',
                text,
                re.IGNORECASE
            )
            if doc_num_match:
                data['document_number'] = doc_num_match.group(1).strip()
                field_confidence['document_number'] = 0.6
        
        # STEP 3: Identify missing required fields
        required_fields = ['first_name', 'last_name', 'date_of_birth']
        for field in required_fields:
            if not data.get(field):
                missing_fields.append(field)
        
        # STEP 4: Calculate overall confidence (NEVER 1.0)
        if field_confidence:
            # For ID documents, we need all critical fields
            critical_fields = ['first_name', 'last_name', 'date_of_birth']
            critical_confidences = [
                field_confidence.get(f, 0.0) 
                for f in critical_fields 
                if f in required_fields
            ]
            
            if critical_confidences:
                # Use minimum of critical fields (pessimistic)
                overall_confidence = min(critical_confidences)
                # Bonus for having additional fields
                if len(data) > len(required_fields):
                    overall_confidence = min(0.98, overall_confidence + 0.1)  # Cap at 0.98
            else:
                overall_confidence = 0.0
        else:
            overall_confidence = 0.0
        
        # NOTE: Certification decision is now made by DecisionEngine
        # We just provide the data and confidence
        
        if not trusted_source:
            trusted_source = 'ocr'
        
        # Add MRZ info to metadata
        metadata = {}
        if mrz_result['found']:
            metadata['mrz_found'] = True
            metadata['mrz_format'] = mrz_result.get('format', 'UNKNOWN')
            metadata['mrz_confidence'] = mrz_result['confidence']
        
        # Certification decision will be made by DecisionEngine
        # Set defaults that will be overridden
        return IDDocumentSchema(
            **data,
            confidence=overall_confidence,
            field_confidence=field_confidence,
            certification_ready=False,  # Will be set by DecisionEngine
            human_review_required=True,  # Will be set by DecisionEngine
            trusted_source=trusted_source,
            missing_fields=missing_fields,
            metadata=metadata,
            raw_text=text
        )
    
    def _extract_driving_license(self, text: str) -> DrivingLicenseSchema:
        """
        Extract driving license data using STRUCTURAL PATTERNS
        
        Strategy:
        - Use numbered fields (1., 2., 3., 4a., 4b., 5.)
        - Layout-based extraction (more reliable than OCR free-form)
        - trusted_source = "layout_rules"
        """
        
        data = {}
        field_confidence = {}
        missing_fields = []
        required_fields = ['first_name', 'last_name', 'license_number', 'expiry_date']
        
        # STEP 1: Extract using structural patterns (numbered fields)
        # Pattern: 1. SURNAME
        pattern_1 = re.search(r'1\.\s*(?P<last_name>[A-Z][A-Z\s]+)', text, re.IGNORECASE)
        if pattern_1:
            data['last_name'] = pattern_1.group('last_name').strip()
            field_confidence['last_name'] = 0.90  # High confidence from structured layout
        
        # Pattern: 2. FIRST NAME
        pattern_2 = re.search(r'2\.\s*(?P<first_name>[A-Z][A-Z\s]+)', text, re.IGNORECASE)
        if pattern_2:
            data['first_name'] = pattern_2.group('first_name').strip()
            field_confidence['first_name'] = 0.90
        
        # Pattern: 3. DATE OF BIRTH AND PLACE
        pattern_3 = re.search(r'3\.\s*(?P<dob_and_place>.+)', text, re.IGNORECASE)
        if pattern_3:
            dob_place_text = pattern_3.group('dob_and_place').strip()
            # Try to extract date
            dob_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', dob_place_text)
            if dob_match:
                try:
                    data['date_of_birth'] = parser.parse(dob_match.group(1), dayfirst=True)
                    field_confidence['date_of_birth'] = 0.85
                except:
                    pass
            # Extract place (usually after date)
            place_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+(.+)', dob_place_text)
            if place_match:
                data['place_of_birth'] = place_match.group(1).strip()
                field_confidence['place_of_birth'] = 0.80
        
        # Pattern: 4a. ISSUE DATE
        pattern_4a = re.search(r'4a\.\s*(?P<issue_date>.+)', text, re.IGNORECASE)
        if pattern_4a:
            issue_text = pattern_4a.group('issue_date').strip()
            issue_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', issue_text)
            if issue_match:
                try:
                    data['issue_date'] = parser.parse(issue_match.group(1), dayfirst=True)
                    field_confidence['issue_date'] = 0.85
                except:
                    pass
        
        # Pattern: 4b. EXPIRY DATE (CRITICAL)
        pattern_4b = re.search(r'4b\.\s*(?P<expiry_date>.+)', text, re.IGNORECASE)
        if pattern_4b:
            expiry_text = pattern_4b.group('expiry_date').strip()
            expiry_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', expiry_text)
            if expiry_match:
                try:
                    data['expiry_date'] = parser.parse(expiry_match.group(1), dayfirst=True)
                    field_confidence['expiry_date'] = 0.90  # Critical field
                except:
                    pass
        
        # Pattern: 5. LICENSE NUMBER (CRITICAL)
        pattern_5 = re.search(r'5\.\s*(?P<license_number>[A-Z0-9]+)', text, re.IGNORECASE)
        if pattern_5:
            data['license_number'] = pattern_5.group('license_number').strip()
            field_confidence['license_number'] = 0.90  # Critical field
        
        # STEP 2: Extract categories (A1, A2, B, C1, C, D1, D)
        categories_match = re.findall(r'\b([A-Z]\d?)\b', text)
        if categories_match:
            # Filter valid categories
            valid_categories = ['A1', 'A2', 'A', 'B', 'C1', 'C', 'D1', 'D', 'BE', 'CE', 'DE']
            found_categories = [c for c in categories_match if c in valid_categories]
            if found_categories:
                data['categories'] = list(set(found_categories))  # Remove duplicates
                field_confidence['categories'] = 0.85
        
        # STEP 3: Fallback to keyword-based extraction if structured patterns fail
        if not data.get('first_name'):
            name_match = re.search(r'nome\s*:?\s*([A-Z][a-z]+)', text, re.IGNORECASE)
            if name_match:
                data['first_name'] = name_match.group(1).strip()
                field_confidence['first_name'] = 0.60  # Lower confidence from OCR
        
        if not data.get('last_name'):
            surname_match = re.search(r'cognome\s*:?\s*([A-Z][a-z]+)', text, re.IGNORECASE)
            if surname_match:
                data['last_name'] = surname_match.group(1).strip()
                field_confidence['last_name'] = 0.60
        
        if not data.get('license_number'):
            # Try alternative patterns
            license_match = re.search(r'(?:numero|n[°º])\s*(?:patente|licenza)\s*:?\s*([A-Z0-9]+)', text, re.IGNORECASE)
            if license_match:
                data['license_number'] = license_match.group(1).strip()
                field_confidence['license_number'] = 0.70
        
        # STEP 4: Calculate confidence (NEVER 1.0)
        if field_confidence:
            critical_fields = ['first_name', 'last_name', 'license_number', 'expiry_date']
            critical_confidences = [
                field_confidence.get(f, 0.0) 
                for f in critical_fields
            ]
            
            if critical_confidences:
                overall_confidence = min(critical_confidences)
                # Bonus for having additional fields
                if len(data) > len(required_fields):
                    overall_confidence = min(0.95, overall_confidence + 0.05)  # Cap at 0.95
            else:
                overall_confidence = 0.0
        else:
            overall_confidence = 0.0
        
        # STEP 5: Identify missing fields
        for field in required_fields:
            if not data.get(field):
                missing_fields.append(field)
        
        # STEP 6: Build metadata
        metadata = {
            'extraction_method': 'layout_rules',
            'structured_patterns_found': len([p for p in [pattern_1, pattern_2, pattern_3, pattern_4a, pattern_4b, pattern_5] if p]) > 0
        }
        
        return DrivingLicenseSchema(
            **data,
            confidence=overall_confidence,
            field_confidence=field_confidence,
            certification_ready=False,  # Will be set by DecisionEngine
            human_review_required=True,  # Will be set by DecisionEngine
            trusted_source='layout_rules',  # Patente uses layout, not MRZ
            missing_fields=missing_fields,
            metadata=metadata,
            raw_text=text
        )
    
    def _extract_with_llm(self, text: str, doc_type: str) -> BaseDocumentSchema:
        """
        Extract using LLM (fallback when regex fails)
        
        Args:
            text: Document text
            doc_type: Document type
            
        Returns:
            Extracted schema
        """
        # TODO: Implement LLM extraction
        # This would use OpenAI/Anthropic to extract structured JSON
        # For now, return base schema with low confidence
        schema_class = {
            'invoice': InvoiceSchema,
            'diploma': DiplomaSchema,
            'id': IDDocumentSchema
        }.get(doc_type, BaseDocumentSchema)
        
        return schema_class(
            confidence=0.3,
            raw_text=text
        )
    
    def _calculate_confidence(self, data: Dict[str, Any], doc_type: str) -> float:
        """
        Calculate confidence score based on extracted fields
        
        Args:
            data: Extracted data dictionary
            doc_type: Document type
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if not data:
            return 0.0
        
        # Define required fields for each type
        required_fields = {
            'invoice': ['invoice_number', 'total_amount', 'invoice_date'],
            'diploma': ['student_name', 'university_name', 'degree_type'],
            'id': ['first_name', 'last_name', 'date_of_birth']
        }
        
        required = required_fields.get(doc_type, [])
        if not required:
            return 0.5
        
        # Count how many required fields are present
        present = sum(1 for field in required if field in data and data[field] is not None)
        base_confidence = present / len(required)
        
        # Bonus for additional fields
        total_fields = len([v for v in data.values() if v is not None])
        bonus = min(0.2, (total_fields - present) * 0.05)
        
        return min(1.0, base_confidence + bonus)
