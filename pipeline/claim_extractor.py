"""
Claim Extractor - Extracts certifiable claims from documents

Key insight: CertiFi certifies CLAIMS, not documents
A claim is a structured statement like:
"X is a contractor for Y from date A to B"

ENHANCED: Now uses NER (spaCy) and advanced semantic extraction
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dateutil import parser
import re

from .role_inference import Role

# Try to import spaCy for NER (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        # Try to load English model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Model not installed, will use regex fallback
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None


class ClaimExtractor:
    """
    Extracts certifiable claims from documents
    
    A claim has:
    - Subject (who)
    - Role (what they are)
    - Entity (for whom)
    - Start date (when it started)
    - End date (when it ends, if applicable)
    - Amount / Currency
    - Evidence (hard/soft)
    
    ENHANCED:
    - Uses spaCy NER for entity extraction (if available)
    - Advanced regex patterns with context
    - Dependency parsing for subject/object relationships
    """
    
    def __init__(self):
        """Initialize with optional NER support"""
        self.use_ner = SPACY_AVAILABLE and nlp is not None
    
    def extract(
        self,
        text: str,
        role: Role,
        document_family: str
    ) -> Dict[str, Any]:
        """
        Extract certifiable claim from document
        
        ENHANCED: Now uses NER and advanced semantic extraction
        
        Args:
            text: Document text
            role: Inferred role
            document_family: Document family
            
        Returns:
            Extracted claim with all components
        """
        claim = {
            'subject': None,
            'role': role.value,
            'entity': None,
            'start_date': None,
            'end_date': None,
            'amount': None,
            'currency': None,
            'services': None,  # NEW: Scope of work / services
            'evidence_type': 'none',
            'confidence': 0.0,
            'raw_text': text[:500],  # Preview
            'extraction_method': 'regex'  # 'ner' or 'regex'
        }
        
        # NEW: Use NER if available for better entity extraction
        if self.use_ner:
            claim = self._extract_with_ner(text, claim, role, document_family)
        else:
            claim = self._extract_with_regex(text, claim, role, document_family)
        
        return claim
    
    def _extract_with_ner(self, text: str, claim: Dict[str, Any], role: Role, document_family: str) -> Dict[str, Any]:
        """
        Extract using spaCy NER for better accuracy
        
        NEW: Named Entity Recognition for automatic extraction
        """
        doc = nlp(text)
        
        # Extract PERSON entities (potential contractor/client names)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        # Extract ORG entities (companies)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        # Extract DATE entities
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        
        # Extract MONEY entities
        money_entities = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
        
        # Use NER results to fill claim
        # Subject: First ORG or PERSON (depending on context)
        if not claim.get('subject'):
            # Look for company names in header or near "Service Provider"
            if orgs:
                claim['subject'] = orgs[0]
                claim['extraction_method'] = 'ner'
        
        # Entity: First PERSON (likely client in engagement letters)
        if not claim.get('entity'):
            if persons:
                claim['entity'] = persons[0]
                claim['extraction_method'] = 'ner'
        
        # Dates: Parse first date as start_date
        if not claim.get('start_date') and dates:
            try:
                claim['start_date'] = parser.parse(dates[0])
                claim['extraction_method'] = 'ner'
            except:
                pass
        
        # Amount: Extract from MONEY entities
        if not claim.get('amount') and money_entities:
            for money in money_entities:
                # Extract number and currency
                amount_match = re.search(r'[\$€£]?\s*([\d,]+\.?\d*)', money)
                if amount_match:
                    try:
                        claim['amount'] = float(amount_match.group(1).replace(',', ''))
                        # Detect currency
                        if '$' in money or 'USD' in money.upper():
                            claim['currency'] = 'USD'
                        elif '€' in money or 'EUR' in money.upper():
                            claim['currency'] = 'EUR'
                        elif '£' in money or 'GBP' in money.upper():
                            claim['currency'] = 'GBP'
                        claim['extraction_method'] = 'ner'
                        break
                    except:
                        pass
        
        # Fallback to regex for anything not found by NER
        return self._extract_with_regex(text, claim, role, document_family)
    
    def _extract_with_regex(self, text: str, claim: Dict[str, Any], role: Role, document_family: str) -> Dict[str, Any]:
        """
        Extract using advanced regex patterns (original method, enhanced)
        
        ENHANCED: Better patterns with context and dynamic regex
        """
        
        # Extract subject (contractor name)
        # For engagement letters, contractor is often the "Service Provider"
        subject_patterns = [
            # Explicit contractor labels
            r'contractor:\s*([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))',
            r'independent\s+contractor:\s*([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))',
            r'contractor\s+name:\s*([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))',
            # Engagement letter patterns - extract company name near "Service Provider"
            r'provided\s+by\s+([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\s+\(["\']?service\s+provider["\']?\)',  # "provided by EXAMPLE COMPANY (Service Provider)"
            r'\(["\']?service\s+provider["\']?\)',  # Look for company name before this
        ]
        
        # First, try explicit patterns
        for pattern in subject_patterns[:4]:  # First 4 patterns
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                subject = match.group(1).strip()
                # Filter out common false positives
                if subject and subject.lower() not in ['service provider', 'contractor', 'client', 'shall be', 'the entire']:
                    claim['subject'] = subject
                    break
        
        # If Service Provider pattern found, extract company name from context
        if not claim['subject'] and re.search(r'\(["\']?service\s+provider["\']?\)', text, re.IGNORECASE):
            # Look for "provided by COMPANY (Service Provider)"
            provider_match = re.search(r'provided\s+by\s+([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\s+\(["\']?service\s+provider["\']?\)', text, re.IGNORECASE)
            if provider_match:
                claim['subject'] = provider_match.group(1).strip()
            else:
                # Fallback: Extract company name from header (usually first few lines)
                header_lines = text.split('\n')[:5]
                for line in header_lines:
                    # Match company name pattern (all caps or title case, not single words)
                    company_match = re.search(r'^([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP)|[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', line)
                    if company_match:
                        potential_name = company_match.group(1).strip()
                        # Filter out common false positives
                        if potential_name.lower() not in ['engagement letter', 'dear', 'services', 'fees']:
                            claim['subject'] = potential_name
                            break
        
        # Extract entity (client/company name)
        entity_patterns = [
            # Explicit client labels
            r'client:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Client: Franco"
            r'company:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Company: ABC Corp"
            # Engagement letter patterns - extract name before "(Client)"
            r'services\s+requested\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\(["\']?client["\']?\)',  # "services requested by Franco (Client)"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\(["\']?client["\']?\)',  # "Franco (Client)"
            r'dear\s+([A-Z][a-z]+)',  # "Dear Franco," - client name (most reliable)
            r'services\s+requested\s+by\s+([A-Z][a-z]+)',  # "services requested by Franco" (without Client label)
        ]
        
        # Try explicit patterns first
        for pattern in entity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                entity = match.group(1).strip()
                # Filter out common false positives
                if entity and entity.lower() not in ['service provider', 'contractor', 'client', 'the entire', 'shall be', 'services', 'fees']:
                    claim['entity'] = entity
                    break
        
        # Fallback: If "(Client)" found but no name extracted, try context extraction
        if not claim['entity'] and re.search(r'\(["\']?client["\']?\)', text, re.IGNORECASE):
            # Look for name immediately before "(Client)" in the same sentence
            client_context = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\(["\']?client["\']?\)', text, re.IGNORECASE)
            if client_context:
                potential_entity = client_context.group(1).strip()
                if potential_entity.lower() not in ['service provider', 'contractor', 'client', 'the entire']:
                    claim['entity'] = potential_entity
        
        # Extract dates
        # Effective date / Start date
        date_patterns = [
            r'effective\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'start\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'commencement\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'from\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'dated\s+(\d{4}-\d{2}-\d{2})',  # "dated 2026-01-21"
            r'dated\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # "dated 01/21/2026"
            r'this\s+letter[,\s]+dated\s+(\d{4}-\d{2}-\d{2})',  # "This letter, dated 2026-01-21"
            r'this\s+engagement\s+letter[,\s]+dated\s+(\d{4}-\d{2}-\d{2})',  # "This Engagement Letter, dated 2026-01-21"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    claim['start_date'] = parser.parse(match.group(1), dayfirst=True)
                    break
                except:
                    pass
        
        # End date / Expiry date
        end_date_patterns = [
            r'end\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'expiry\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'termination\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'until\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        for pattern in end_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    claim['end_date'] = parser.parse(match.group(1), dayfirst=True)
                    break
                except:
                    pass
        
        # Extract amount and currency (for engagement letters, contracts)
        amount_patterns = [
            r'fees?\s+(?:charged|shall be|are)\s+(?:by\s+the\s+)?(?:service\s+provider|contractor)?\s*(?:shall be|are)?\s*\$?([\d,]+\.?\d*)',  # "fees charged by the Service Provider shall be $3000.00"
            r'amount[:\s]+\$?([\d,]+\.?\d*)',  # "Amount: $3000.00"
            r'total[:\s]+\$?([\d,]+\.?\d*)',  # "Total: $3000.00"
            r'\$([\d,]+\.?\d*)',  # "$3000.00"
            r'([\d,]+\.?\d*)\s*(?:usd|eur|gbp|us\s*dollars?|euros?|pounds?)',  # "3000.00 USD"
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    claim['amount'] = float(amount_str)
                    # Extract currency
                    currency_match = re.search(r'(\$|usd|eur|gbp|us\s*dollars?|euros?|pounds?)', text[match.start():match.end()+20], re.IGNORECASE)
                    if currency_match:
                        currency = currency_match.group(1).upper()
                        if currency == '$' or 'USD' in currency or 'DOLLAR' in currency:
                            claim['currency'] = 'USD'
                        elif 'EUR' in currency or 'EURO' in currency:
                            claim['currency'] = 'EUR'
                        elif 'GBP' in currency or 'POUND' in currency:
                            claim['currency'] = 'GBP'
                    else:
                        claim['currency'] = 'USD'  # Default
                    break
                except:
                    pass
        
        # NEW: Extract services / scope of work
        services_patterns = [
            r'services\.?\s+The\s+services\s+provided\s+(?:under\s+this\s+letter|are)\s+(?:as\s+follows|:)\s*(.+?)(?:\.|Fees|fees|$)',
            r'scope\s+of\s+work[:\s]+(.+?)(?:\.|Fees|fees|$)',
            r'services\s+to\s+be\s+provided[:\s]+(.+?)(?:\.|Fees|fees|$)',
        ]
        
        for pattern in services_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                services = match.group(1).strip()
                # Clean up: remove extra whitespace, limit length
                services = ' '.join(services.split())[:200]  # Max 200 chars
                if services and len(services) > 5:  # At least 5 chars
                    claim['services'] = services
                    break
        
        # Calculate confidence based on extracted components
        components_found = sum([
            1 if claim['subject'] else 0,
            1 if claim['entity'] else 0,
            1 if claim['start_date'] else 0,
            0.5 if claim['amount'] else 0,  # Amount is nice to have but not critical
        ])
        
        # Base confidence from components
        if components_found >= 3:
            claim['confidence'] = 0.85
        elif components_found == 2:
            claim['confidence'] = 0.70
        elif components_found == 1:
            claim['confidence'] = 0.50
        else:
            claim['confidence'] = 0.30
        
        # Evidence type
        if claim['subject'] and claim['entity'] and claim['start_date']:
            claim['evidence_type'] = 'hard'
        elif claim['subject'] or claim['entity']:
            claim['evidence_type'] = 'soft'
        else:
            claim['evidence_type'] = 'none'
        
        return claim
    
    def format_claim(self, claim: Dict[str, Any]) -> str:
        """
        Format claim as human-readable statement
        
        Args:
            claim: Extracted claim
            
        Returns:
            Formatted claim statement
        """
        parts = []
        
        if claim['subject']:
            parts.append(claim['subject'])
        else:
            parts.append("[Subject]")
        
        parts.append("is a")
        parts.append(claim['role'])
        
        if claim['entity']:
            parts.append("for")
            parts.append(claim['entity'])
        
        if claim['start_date']:
            parts.append("from")
            parts.append(claim['start_date'].strftime("%Y-%m-%d"))
        
        if claim['end_date']:
            parts.append("until")
            parts.append(claim['end_date'].strftime("%Y-%m-%d"))
        elif claim['start_date']:
            parts.append("(ongoing)")
        
        # Add amount if available
        if claim.get('amount'):
            amount_str = f"{claim['amount']:.2f}" if isinstance(claim['amount'], float) else str(claim['amount'])
            currency = claim.get('currency', 'USD')
            parts.append(f"({currency} {amount_str})")
        
        return " ".join(parts)
