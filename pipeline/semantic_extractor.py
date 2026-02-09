"""
SEMANTIC EXTRACTION LAYER

LLM-assisted fact extraction ONLY - no classification.

Responsibilities:
- Extract factual claims from document text
- Use LLM ONLY to extract facts, NOT to classify
- Return structured JSON with explicit nulls for missing fields
- Fields: parties, dates, amounts, services, issuer, etc.

The LLM prompt explicitly forbids classification and only asks for facts.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json

# Try to import LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ExtractedFacts:
    """Facts extracted from document - explicit nulls for missing fields"""
    
    # Parties
    party_1_name: Optional[str] = None
    party_1_type: Optional[str] = None  # "individual", "company", "organization"
    party_2_name: Optional[str] = None
    party_2_type: Optional[str] = None
    
    # Dates
    effective_date: Optional[str] = None  # ISO format: "2024-03-01"
    expiration_date: Optional[str] = None
    issue_date: Optional[str] = None
    
    # Financial
    amount: Optional[float] = None
    currency: Optional[str] = None
    secondary_amount: Optional[float] = None
    secondary_currency: Optional[str] = None
    
    # Services/Subject
    subject: Optional[str] = None  # What the document is about
    services: Optional[str] = None  # Scope of services
    deliverables: Optional[List[str]] = None
    
    # Issuer (for certificates)
    issuer_name: Optional[str] = None
    issuer_type: Optional[str] = None
    
    # Document-specific
    invoice_number: Optional[str] = None
    total_amount: Optional[float] = None
    vat_amount: Optional[float] = None
    
    # Extraction metadata
    extraction_method: str = "llm"  # "llm" or "rule_based"
    confidence: float = 0.0
    fields_extracted: List[str] = None  # Which fields were successfully extracted


class SemanticExtractor:
    """
    SEMANTIC EXTRACTION LAYER - LLM-assisted fact extraction
    
    Uses LLM ONLY to extract factual claims, NOT to classify documents.
    The prompt explicitly forbids classification and only asks for facts.
    """
    
    def __init__(self, use_llm: bool = False, llm_provider: str = "openai"):
        """
        Initialize semantic extractor
        
        Args:
            use_llm: Whether to use LLM (if False, uses rule-based only)
            llm_provider: "openai" or "anthropic"
        """
        self.use_llm = use_llm and (OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE)
        self.llm_provider = llm_provider if self.use_llm else None
    
    def extract(self, text: str, structure_hints: Optional[Dict[str, Any]] = None) -> ExtractedFacts:
        """
        Extract factual claims from document
        
        Args:
            text: Document text
            structure_hints: Optional hints from structure layer (e.g., detected family)
            
        Returns:
            ExtractedFacts with explicit nulls for missing fields
        """
        if self.use_llm:
            return self._extract_with_llm(text, structure_hints)
        else:
            return self._extract_with_rules(text, structure_hints)
    
    def _extract_with_llm(self, text: str, structure_hints: Optional[Dict[str, Any]] = None) -> ExtractedFacts:
        """
        Extract facts using LLM
        
        CRITICAL: The prompt explicitly forbids classification.
        LLM only extracts facts, does NOT classify the document.
        """
        # Truncate text for LLM (keep first 4000 chars for context)
        text_preview = text[:4000] if len(text) > 4000 else text
        
        prompt = f"""You are a fact extraction system. Your ONLY job is to extract factual claims from a document.

CRITICAL RULES:
1. DO NOT classify the document type
2. DO NOT guess or infer information not explicitly stated
3. Return NULL for any field that is not explicitly stated in the document
4. Return structured JSON only

Extract the following facts from this document:

{text_preview}

Return a JSON object with these fields (use null for missing fields):
{{
  "party_1_name": null or string,
  "party_1_type": null or "individual" or "company" or "organization",
  "party_2_name": null or string,
  "party_2_type": null or "individual" or "company" or "organization",
  "effective_date": null or ISO date string (YYYY-MM-DD),
  "expiration_date": null or ISO date string (YYYY-MM-DD),
  "issue_date": null or ISO date string (YYYY-MM-DD),
  "amount": null or number,
  "currency": null or currency code (AED, USD, EUR, etc.),
  "secondary_amount": null or number,
  "secondary_currency": null or currency code,
  "subject": null or string (what the document is about),
  "services": null or string (scope of services if mentioned),
  "deliverables": null or array of strings,
  "issuer_name": null or string (for certificates),
  "issuer_type": null or "individual" or "company" or "organization",
  "invoice_number": null or string,
  "total_amount": null or number,
  "vat_amount": null or number
}}

IMPORTANT:
- Only extract facts EXPLICITLY stated in the document
- If a field is not mentioned, return null
- Do NOT infer or guess
- Do NOT classify the document type
- Return valid JSON only"""

        try:
            if self.llm_provider == "openai" and OPENAI_AVAILABLE:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a fact extraction system. Extract only explicit facts. Do not classify documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Deterministic
                    max_tokens=1000
                )
                json_str = response.choices[0].message.content.strip()
            elif self.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
                response = anthropic.Anthropic().messages.create(
                    model="claude-3-opus-20240229",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                json_str = response.content[0].text.strip()
            else:
                # Fallback to rule-based
                return self._extract_with_rules(text, structure_hints)
            
            # Parse JSON response
            facts_dict = json.loads(json_str)
            
            # Build ExtractedFacts object
            facts = ExtractedFacts(
                party_1_name=facts_dict.get("party_1_name"),
                party_1_type=facts_dict.get("party_1_type"),
                party_2_name=facts_dict.get("party_2_name"),
                party_2_type=facts_dict.get("party_2_type"),
                effective_date=facts_dict.get("effective_date"),
                expiration_date=facts_dict.get("expiration_date"),
                issue_date=facts_dict.get("issue_date"),
                amount=facts_dict.get("amount"),
                currency=facts_dict.get("currency"),
                secondary_amount=facts_dict.get("secondary_amount"),
                secondary_currency=facts_dict.get("secondary_currency"),
                subject=facts_dict.get("subject"),
                services=facts_dict.get("services"),
                deliverables=facts_dict.get("deliverables"),
                issuer_name=facts_dict.get("issuer_name"),
                issuer_type=facts_dict.get("issuer_type"),
                invoice_number=facts_dict.get("invoice_number"),
                total_amount=facts_dict.get("total_amount"),
                vat_amount=facts_dict.get("vat_amount"),
                extraction_method="llm",
                confidence=0.85,  # LLM extraction confidence
                fields_extracted=[k for k, v in facts_dict.items() if v is not None]
            )
            
            return facts
            
        except Exception as e:
            # Fallback to rule-based on error
            print(f"LLM extraction failed: {e}, falling back to rule-based")
            return self._extract_with_rules(text, structure_hints)
    
    def _extract_with_rules(self, text: str, structure_hints: Optional[Dict[str, Any]] = None) -> ExtractedFacts:
        """
        Extract facts using rule-based patterns (fallback when LLM unavailable)
        
        This is a simplified version - full rule-based extraction is in claim_extractor.py
        """
        import re
        from dateutil import parser as date_parser
        
        # Try to import dateparser
        try:
            import dateparser
            DATEPARSER_AVAILABLE = True
        except ImportError:
            DATEPARSER_AVAILABLE = False
        
        text_normalized = ' '.join(text.split())
        facts = ExtractedFacts(extraction_method="rule_based", confidence=0.70)
        
        # Extract dates
        date_patterns = [
            r'effective\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'expiration\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'issue\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1).strip()
                    if DATEPARSER_AVAILABLE:
                        parsed = dateparser.parse(date_str)
                        if parsed:
                            iso_date = parsed.strftime("%Y-%m-%d")
                            if "effective" in pattern.lower():
                                facts.effective_date = iso_date
                            elif "expiration" in pattern.lower():
                                facts.expiration_date = iso_date
                            elif "issue" in pattern.lower():
                                facts.issue_date = iso_date
                except:
                    pass
        
        # Extract amounts (simplified - full logic in claim_extractor.py)
        amount_patterns = [
            r'total\s+(?:project\s+)?value[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP)',
            r'amount[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1).replace(',', ''))
                    currency = match.group(2).upper() if len(match.groups()) >= 2 else None
                    if amount > 0:
                        facts.amount = amount
                        facts.currency = currency
                        break
                except:
                    pass
        
        # Extract subject/scope
        scope_patterns = [
            r'scope\s+of\s+services[:\s]+(.+?)(?:\.|CONTRACT|$)',
            r'scope\s+of\s+work[:\s]+(.+?)(?:\.|CONTRACT|$)',
        ]
        
        for pattern in scope_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
            if match:
                scope = match.group(1).strip()[:200]
                if len(scope) > 20:
                    facts.subject = scope
                    facts.services = scope
                    break
        
        # Track extracted fields
        facts.fields_extracted = [
            k for k, v in facts.__dict__.items() 
            if k not in ['extraction_method', 'confidence', 'fields_extracted'] and v is not None
        ]
        
        return facts
