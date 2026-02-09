"""
Claim Extractor - Extracts certifiable claims from documents

Key insight: CertiFi certifies CLAIMS, not documents
A claim is a structured statement like:
"X is a contractor for Y from date A to B"

ENHANCED: Now uses NER (spaCy) and advanced semantic extraction
"""

from typing import Dict, Any, Optional, List
from dateutil import parser
import re
import logging

from .role_inference import Role

logger = logging.getLogger(__name__)

# Try to import dateparser (optional, better date parsing)
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    dateparser = None

# Try to import price_parser (optional, better amount/currency extraction)
try:
    from price_parser import Price
    PRICE_PARSER_AVAILABLE = True
except ImportError:
    PRICE_PARSER_AVAILABLE = False
    Price = None

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
        document_family: str,
        compensation_table_data: Optional[Dict[str, Any]] = None,
        document_subtype: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract certifiable claim from document
        
        ENHANCED: Now uses NER and advanced semantic extraction
        NEW: Can use compensation table data from Camelot for accurate amount extraction
        NEW: Can use document_subtype for better extraction (e.g., professional_services_agreement)
        
        Args:
            text: Document text
            role: Inferred role
            document_family: Document family
            compensation_table_data: Optional compensation table data from TableExtractor
            document_subtype: Optional document subtype (e.g., "professional_services_agreement")
            
        Returns:
            Extracted claim with all components
        """
        try:
            print("=" * 80)
            print("🔍 ClaimExtractor.extract() CALLED")
            print(f"📝 Text length: {len(text) if text else 0}")
            print(f"👤 Role: {role.value if role else 'None'}")
            print(f"📄 Document family: {document_family}")
            print(f"📋 Document subtype: {document_subtype}")
            print(f"📊 Compensation table data: {compensation_table_data is not None}")
            print("=" * 80)
            
            claim = {
                'subject': None,
                'role': role.value,
                'entity': None,
                'start_date': None,
                'end_date': None,
                'amount': None,
                'currency': None,
                'secondary_currency': None,  # NEW: Secondary currency (e.g., USD equivalent)
                'secondary_amount': None,  # NEW: Secondary amount
                'services': None,  # NEW: Scope of work / services
                'evidence_type': 'none',
                'confidence': 0.0,
                'raw_text': text[:500] if text else '',  # Preview
                'extraction_method': 'regex'  # 'ner' or 'regex'
            }
            
            # NEW: Use specialized extraction for resume/CV
            if document_subtype == "resume":
                print("📋 Using resume extraction")
                # Use specialized CV extraction
                claim = self._extract_resume_data(text, claim, role)
                # Then enhance with standard regex extraction for anything missed
                claim = self._extract_with_regex(text, claim, role, document_family, document_subtype, compensation_table_data)
            # NEW: Use specialized extraction for professional_services_agreement
            elif document_subtype == "professional_services_agreement":
                print("📋 Using professional_services_agreement extraction")
                # Use specialized extraction first
                claim = self._extract_professional_services_claims(text, claim, role)
                # Then enhance with standard regex extraction for anything missed
                claim = self._extract_with_regex(text, claim, role, document_family, document_subtype, compensation_table_data)
            # NEW: Use NER if available for better entity extraction
            elif self.use_ner:
                print("📋 Using NER extraction")
                claim = self._extract_with_ner(text, claim, role, document_family)
                # Fallback to regex for anything not found by NER
                claim = self._extract_with_regex(text, claim, role, document_family, document_subtype, compensation_table_data)
            else:
                print("📋 Using standard regex extraction")
                claim = self._extract_with_regex(text, claim, role, document_family, document_subtype, compensation_table_data)
            
            # NEW: Override amount/currency with table data if available (more accurate)
            if compensation_table_data:
                print("📊 Applying compensation table data")
                claim = self._apply_compensation_table_data(claim, compensation_table_data)
            
            print(f"✅ Extraction completed. Amount: {claim.get('amount')}, Entity: {claim.get('entity')}, Subject: {claim.get('subject')}")
            print("=" * 80)
            
            return claim
            
        except Exception as e:
            print(f"❌ EXCEPTION in ClaimExtractor.extract: {e}")
            import traceback
            traceback.print_exc()
            # Return empty claim structure instead of None
            return {
                'subject': None,
                'role': role.value if role else 'unknown',
                'entity': None,
                'start_date': None,
                'end_date': None,
                'amount': None,
                'currency': None,
                'secondary_currency': None,
                'secondary_amount': None,
                'services': None,
                'evidence_type': 'none',
                'confidence': 0.0,
                'raw_text': text[:500] if text else '',
                'extraction_method': 'error',
                'error': str(e)
            }
    
    def _apply_compensation_table_data(
        self,
        claim: Dict[str, Any],
        compensation_table_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply compensation table data to claim (overrides text-based extraction)
        
        Priority:
        1. annual_total (highest priority)
        2. monthly_total * 12 (if annual_total not available)
        3. base_fee * 12 (if monthly_total not available)
        """
        # Use annual_total if available (highest priority)
        if compensation_table_data.get('annual_total'):
            claim['amount'] = float(compensation_table_data['annual_total'])
            claim['currency'] = compensation_table_data.get('currency', 'AED')
            claim['extraction_method'] = 'table_annual_total'
        
        # Fallback to monthly_total * 12
        elif compensation_table_data.get('monthly_total'):
            claim['amount'] = float(compensation_table_data['monthly_total']) * 12
            claim['currency'] = compensation_table_data.get('currency', 'AED')
            claim['extraction_method'] = 'table_monthly_total'
        
        # Fallback to base_fee * 12
        elif compensation_table_data.get('base_fee'):
            claim['amount'] = float(compensation_table_data['base_fee']) * 12
            claim['currency'] = compensation_table_data.get('currency', 'AED')
            claim['extraction_method'] = 'table_base_fee'
        
        # Use secondary currency if available
        if compensation_table_data.get('secondary_currency'):
            claim['secondary_currency'] = compensation_table_data['secondary_currency']
            if compensation_table_data.get('secondary_amounts'):
                # Try to get annual secondary amount
                secondary_annual = compensation_table_data['secondary_amounts'].get('annual_total')
                if secondary_annual:
                    claim['secondary_amount'] = float(secondary_annual)
                else:
                    # Fallback to monthly * 12
                    secondary_monthly = compensation_table_data['secondary_amounts'].get('monthly_total')
                    if secondary_monthly:
                        claim['secondary_amount'] = float(secondary_monthly) * 12
        
        return claim
    
    def _extract_amount_with_priority(self, text: str, compensation_table_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Extract amount with correct priority order.
        
        Priority:
        1. Annual Contract Value (from text or table)
        2. Total Monthly Compensation × 12 (annualized)
        3. Monthly Base Fee × 12 (fallback)
        
        Returns:
            {
                'amount': float,
                'currency': str,
                'secondary_amount': float (optional),
                'secondary_currency': str (optional),
                'amount_type': str (for debugging)
            }
        """
        try:
            print("🔍 _extract_amount_with_priority() called")
            
            # ═══════════════════════════════════════════════════════════
            # PRIORITY 1: Annual Contract Value (from text)
            # ═══════════════════════════════════════════════════════════
            print("Priority 1: Searching for Annual Contract Value in text")
            logger.debug("Priority 1: Searching for Annual Contract Value in text")
            
            # 🔧 FIX: Pattern ultra-robusti che ignorano *, spazi multipli, newline
            annual_patterns = [
                # Pattern ultra-robusto: ignora *, \n, spazi multipli tra le parole
                r'Annual[\s*\n]+Contract[\s*\n]+Value[\s*:\n]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                
                # Alternativa: qualsiasi carattere non-digit tra le parole (gestisce **, spazi, newline)
                r'Annual[^0-9]+Contract[^0-9]+Value[^0-9]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                
                # Caso senza bold markers (fallback)
                r'Annual\s+Contract\s+Value[:\s]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                
                # Alternative phrasings (anche con bold markers)
                r'Total[^0-9]+Annual[^0-9]+(?:Value|Compensation)[^0-9]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'Annual[^0-9]+Total[^0-9]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            ]
            
            for i, pattern in enumerate(annual_patterns):
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    aed_amount = self._parse_amount(match.group(1))
                    if aed_amount and aed_amount > 50000:  # Sanity check: > 50K AED/year
                        print(f"✓ Found Annual Contract Value (pattern {i}): {aed_amount} AED")
                        logger.info(f"✓ Found Annual Contract Value (pattern {i}): {aed_amount} AED")
                        
                        result = {
                            'amount': aed_amount,
                            'currency': 'AED',
                            'amount_type': 'annual_from_text'
                        }
                        
                        # Try to extract USD amount (nella stessa riga o contesto)
                        # Cerca il secondo numero dopo "Annual Contract Value"
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(text), match.end() + 200)  # Estendi il contesto per trovare USD
                        context = text[context_start:context_end]
                        
                        # Pattern migliorato: cerca due numeri consecutivi (AED e USD)
                        # Gestisce: "678,000.00 184,572.00" o "**678,000.00** **184,572.00**"
                        usd_patterns = [
                            # Due numeri consecutivi con eventuali asterischi e spazi
                            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)[\s*]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                            # Con eventuali label di valuta
                            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:AED|aed)?[\s*]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|usd)?',
                        ]
                        
                        for usd_pattern in usd_patterns:
                            usd_match = re.search(usd_pattern, context)
                            if usd_match:
                                # Verifica che il primo numero corrisponda all'AED amount trovato
                                first_num = self._parse_amount(usd_match.group(1))
                                if abs(first_num - aed_amount) < 1000:  # Close match (tolleranza per formattazione)
                                    usd_amount = self._parse_amount(usd_match.group(2))
                                    if usd_amount and 10000 < usd_amount < 1000000:  # Reasonable USD range
                                        result['secondary_amount'] = usd_amount
                                        result['secondary_currency'] = 'USD'
                                        print(f"✓ Found secondary USD amount: {usd_amount}")
                                        logger.info(f"✓ Found secondary USD amount: {usd_amount}")
                                        break
                        
                        return result
                else:
                    logger.debug(f"Pattern {i} did not match")
            
            print("⚠ Annual Contract Value not found in text patterns")
            logger.warning("⚠ Annual Contract Value not found in text patterns")
            
            
            # ═══════════════════════════════════════════════════════════
            # PRIORITY 2: Extract from Table (Annual Contract Value row)
            # ═══════════════════════════════════════════════════════════
            if compensation_table_data:
                print("Priority 2: Checking compensation table data")
                if compensation_table_data.get('annual_total'):
                    annual_amount = float(compensation_table_data['annual_total'])
                    result = {
                        'amount': annual_amount,
                        'currency': compensation_table_data.get('currency', 'AED'),
                        'amount_type': 'annual_from_table'
                    }
                    
                    # Check for secondary currency
                    if compensation_table_data.get('secondary_amounts', {}).get('annual_total'):
                        result['secondary_amount'] = float(compensation_table_data['secondary_amounts']['annual_total'])
                        result['secondary_currency'] = compensation_table_data.get('secondary_currency', 'USD')
                    
                    print(f"✓ Found Annual Contract Value (table): {result}")
                    return result
            
            
            # ═══════════════════════════════════════════════════════════
            # PRIORITY 3: Total Monthly Compensation × 12 (annualized)
            # ═══════════════════════════════════════════════════════════
            print("Priority 3: Searching for Total Monthly Compensation")
            logger.debug("Priority 3: Searching for Total Monthly Compensation")
            
            # Pattern robusti che gestiscono bold markers
            monthly_patterns = [
                # Pattern ultra-robusto: ignora *, \n, spazi multipli
                r'Total[\s*\n]+Monthly[\s*\n]+Compensation[\s*:\n]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                # Alternativa: qualsiasi carattere non-digit tra le parole
                r'Total[^0-9]+Monthly[^0-9]+Compensation[^0-9]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                # Pattern senza bold markers (fallback)
                r'Monthly\s+Total[:\s]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'Total\s+Monthly[:\s]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            ]
            
            for i, pattern in enumerate(monthly_patterns):
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    monthly_amount = self._parse_amount(match.group(1))
                    if monthly_amount and 10000 <= monthly_amount <= 200000:  # Sanity check
                        annual_amount = monthly_amount * 12
                        print(
                            f"⚠ Annual Contract Value not found. "
                            f"Using annualized monthly: {monthly_amount} × 12 = {annual_amount}"
                        )
                        
                        result = {
                            'amount': annual_amount,
                            'currency': 'AED',
                            'amount_type': 'monthly_annualized'
                        }
                        
                        # Try to find USD monthly amount
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(text), match.end() + 200)
                        context = text[context_start:context_end]
                        
                        usd_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:AED|aed)?\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|usd)?'
                        usd_match = re.search(usd_pattern, context)
                        if usd_match:
                            first_num = self._parse_amount(usd_match.group(1))
                            if abs(first_num - monthly_amount) < 1000:
                                usd_monthly = self._parse_amount(usd_match.group(2))
                                if usd_monthly:
                                    result['secondary_amount'] = usd_monthly * 12
                                    result['secondary_currency'] = 'USD'
                        
                        return result
            
            
            # ═══════════════════════════════════════════════════════════
            # PRIORITY 4: Monthly Base Fee × 12 (last resort)
            # ═══════════════════════════════════════════════════════════
            print("Priority 4: Searching for Monthly Base Fee")
            logger.debug("Priority 4: Searching for Monthly Base Fee")
            
            # Pattern robusti che gestiscono bold markers
            base_fee_patterns = [
                # Pattern ultra-robusto: ignora *, \n, spazi multipli
                r'Monthly[\s*\n]+Base[\s*\n]+Fee[\s*:\n]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                # Alternativa: qualsiasi carattere non-digit tra le parole
                r'Monthly[^0-9]+Base[^0-9]+Fee[^0-9]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                # Pattern senza bold markers (fallback)
                r'Monthly\s+Base\s+Fee[:\s]+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            ]
            
            for pattern in base_fee_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    base_fee = self._parse_amount(match.group(1))
                    if base_fee:
                        annual_amount = base_fee * 12
                        print(
                            f"⚠ Only found Monthly Base Fee. "
                            f"Using annualized: {base_fee} × 12 = {annual_amount}"
                        )
                        logger.warning(f"⚠ Only found Monthly Base Fee. Using annualized: {base_fee} × 12 = {annual_amount}")
                        return {
                            'amount': annual_amount,
                            'currency': 'AED',
                            'amount_type': 'base_fee_annualized'
                        }
                    break  # Se trovato un match valido, esci dal loop
            
            print("❌ Could not extract any amount from document")
            return None
            
        except Exception as e:
            print(f"❌ EXCEPTION in _extract_amount_with_priority: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_amount(self, text: str) -> Optional[float]:
        """
        Parse amount from text.
        
        Examples:
            "678,000.00" → 678000.0
            "56,500" → 56500.0
            "**678,000**" → 678000.0
        
        Args:
            text: String containing amount
        
        Returns:
            Float amount or None
        """
        if not text:
            return None
        
        # Remove bold markers, currency symbols, spaces
        cleaned = re.sub(r'[\*\$€£]', '', str(text))
        cleaned = cleaned.strip()
        
        # Remove commas (thousand separators)
        cleaned = cleaned.replace(',', '')
        
        # Extract number
        match = re.search(r'(\d+(?:\.\d{1,2})?)', cleaned)
        if match:
            try:
                amount = float(match.group(1))
                
                # Sanity check
                if 0 < amount < 100000000:  # Between 0 and 100M
                    return amount
                else:
                    logger.warning(f"Amount outside expected range: {amount}")
                    return None
            
            except ValueError:
                logger.warning(f"Could not parse amount: {text}")
                return None
        
        return None
    
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
        # Note: document_subtype not passed here as NER is fallback, but we can enhance later if needed
        return self._extract_with_regex(text, claim, role, document_family, None, None)
    
    def _extract_with_regex(self, text: str, claim: Dict[str, Any], role: Role, document_family: str, document_subtype: Optional[str] = None, compensation_table_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract using advanced regex patterns (original method, enhanced)
        
        ENHANCED: Better patterns with context and dynamic regex
        NEW: Can use document_subtype for specialized extraction (e.g., professional_services_agreement)
        """
        
        # If document_subtype is professional_services_agreement, use enhanced patterns
        if document_subtype == "professional_services_agreement":
            claim = self._extract_professional_services_claims(text, claim, role)
        
        # Extract subject (contractor name or service type)
        # IMPROVED: Better patterns for service types and company names
        # Normalize text for better matching
        text_normalized_subject = ' '.join(text.split())
        
        subject_patterns = [
            # HIGHEST PRIORITY: "SCOPE OF SERVICES" section content
            r'scope\s+of\s+services[:\s]+(.+?)(?:\.|CONTRACT|Contract|PERIOD|Period|$)',  # "SCOPE OF SERVICES: Technical consulting..."
            r'scope\s+of\s+work[:\s]+(.+?)(?:\.|CONTRACT|Contract|PERIOD|Period|$)',  # "Scope of Work: Offshore Drilling Operations"
            # Service type patterns (for contracts describing services) - more flexible
            r'services?\s+(?:to\s+be\s+)?(?:provided|rendered|performed)[:\s]+([A-Z][A-Za-z\s]+(?:Services|Operations|Engineering|Consulting|Management|Drilling|Petroleum|Offshore))',  # "Services: Petroleum Engineering Services"
            r'(?:providing|performing|rendering)\s+([A-Z][A-Za-z\s]+(?:Services|Operations|Engineering|Consulting|Management|Drilling|Petroleum|Offshore))',  # "providing Petroleum Engineering Services"
            # Pattern for "Petroleum Engineering Services" or similar
            r'([A-Z][a-z]+\s+(?:Engineering|Operations|Services|Consulting|Management|Drilling|Petroleum|Offshore)\s+Services?)',
            # Extract from "The Contractor agrees to provide the following services"
            r'agrees\s+to\s+provide\s+(?:the\s+following\s+)?services?\s+(?:to\s+the\s+client[:\s]+)?(.+?)(?:\.|CONTRACT|Contract|PERIOD|Period|$)',  # "agrees to provide the following services: Technical consulting..."
            # Explicit contractor labels
            r'contractor:\s*([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))',
            r'independent\s+contractor:\s*([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))',
            r'contractor\s+name:\s*([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))',
            # Engagement letter patterns - extract company name near "Service Provider"
            r'provided\s+by\s+([A-Z][A-Z\s&]+(?:COMPANY|LTD|INC|LLC|AG|CORP|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\s+\(["\']?service\s+provider["\']?\)',  # "provided by EXAMPLE COMPANY (Service Provider)"
        ]
        
        # First, try service type patterns (more specific)
        for pattern in subject_patterns[:6]:  # First 6 patterns (service types and scope)
            match = re.search(pattern, text_normalized_subject, re.IGNORECASE | re.DOTALL)
            if match and match.groups():
                subject = match.group(1).strip()
                # Clean up: take first meaningful sentence
                # Remove list markers like "a)", "b)", etc.
                subject = re.sub(r'^[a-z]\)\s*', '', subject, flags=re.IGNORECASE)
                subject = re.sub(r'^\d+[\.\)]\s*', '', subject)  # Remove "1.", "2)", etc.
                
                # Split by list markers (a), b), c), etc.) to get first item only
                # Also split by periods, newlines, and numbers followed by periods
                parts = re.split(r'\s+[a-z]\)\s+|\s+\d+[\.\)]\s+|\.\s+[A-Z]|\n', subject)
                if parts:
                    # Take first part that's meaningful (at least 15 chars)
                    for part in parts:
                        part = part.strip()
                        # Remove trailing list markers
                        part = re.sub(r'\s+[a-z]\)\s*$', '', part, flags=re.IGNORECASE)
                        part = re.sub(r'\s+\d+[\.\)]\s*$', '', part)
                        if len(part) >= 15 and not part.lower().startswith(('b)', 'c)', 'd)', 'e)', 'f)')):
                            subject = part
                            break
                    else:
                        # Fallback: take first 100 chars and clean
                        subject = subject[:100]
                        # Remove any trailing list markers
                        subject = re.sub(r'\s+[a-z]\)\s*$', '', subject, flags=re.IGNORECASE)
                        subject = re.sub(r'\s+\d+[\.\)]\s*$', '', subject)
                else:
                    subject = subject[:100]
                
                # Final cleanup: limit to 150 chars max, remove trailing numbers/list markers
                subject = subject[:150].strip()
                subject = re.sub(r'\s+[a-z]\)\s*$', '', subject, flags=re.IGNORECASE)
                subject = re.sub(r'\s+\d+[\.\)]\s*$', '', subject)
                subject = subject.strip()
                
                # Filter out common false positives
                if subject and len(subject) > 10:  # At least 10 chars for meaningful subject
                    subject_lower = subject.lower()
                    if subject_lower not in ['service provider', 'contractor', 'client', 'shall be', 'the entire', 'operations', 'the following']:
                        claim['subject'] = subject
                        break
        
        # If no service type found, try contractor/company name patterns
        if not claim['subject']:
            for pattern in subject_patterns[3:]:  # Remaining patterns
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
        # IMPROVED: Better patterns for company names (including LLC, Inc, Ltd, etc.)
        # Normalize text for better matching
        text_normalized_entity = ' '.join(text.split())
        
        entity_patterns = [
            # HIGHEST PRIORITY: "Company Name: Emirates Petroleum Industries LLC" pattern
            r'company\s+name[:\s]+([A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation|Industries|Group|Company))',  # "Company Name: Emirates Petroleum Industries LLC"
            r'client\s*\([^)]+\)[:\s]*company\s+name[:\s]+([A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation|Industries|Group|Company))',  # "CLIENT (Principal): Company Name: ..."
            # Company names with legal suffixes - more flexible
            r'(?:client|company|entity|party|contract\s+with)[:\s]+([A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation|AG|GmbH|S\.A\.|S\.p\.A\.|S\.r\.l\.|Industries|Group|Company))',  # "Client: Emirates Petroleum Industries LLC"
            r'([A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation|AG|GmbH|S\.A\.|S\.p\.A\.|S\.r\.l\.|Industries|Group|Company))',  # Standalone company name
            # Pattern for "Emirates Petroleum Industries LLC" style names
            r'([A-Z][a-z]+\s+(?:Petroleum|Energy|Oil|Gas|Services|Industries|Group|Company)\s+[A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?))',
            # Explicit client labels (simple names)
            r'client:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)',  # "Client: Franco Rossi"
            r'company:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)',  # "Company: ABC Corp"
            # Engagement letter patterns - extract name before "(Client)"
            r'services\s+requested\s+by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s+\(["\']?client["\']?\)',  # "services requested by Franco (Client)"
            r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s+\(["\']?client["\']?\)',  # "Franco (Client)"
            r'dear\s+([A-Z][A-Za-z]+)',  # "Dear Franco," - client name (most reliable)
            r'services\s+requested\s+by\s+([A-Z][A-Za-z]+)',  # "services requested by Franco" (without Client label)
        ]
        
        # Try explicit patterns first (prioritize company names with legal suffixes)
        for pattern in entity_patterns:
            matches = list(re.finditer(pattern, text_normalized_entity, re.IGNORECASE))
            for match in matches:
                if match.groups():
                    entity = match.group(1).strip()
                    # Filter out common false positives and addresses
                    # Must be at least 5 chars and look like a real company name
                    if entity and len(entity) >= 5:
                        entity_lower = entity.lower()
                        # Filter out addresses, partial words, and common false positives
                        # More aggressive filtering for partial phrases
                        invalid_patterns = [
                            r'^(is|may|shall|will|can|must|should)\s+',  # Starts with verbs
                            r'\s+(is|may|shall|will|can|must|should|terminate|engage|provide)\s*$',  # Ends with verbs
                            r'^(the|a|an)\s+',  # Starts with articles
                            r'\s+(the|a|an|this|that)\s*$',  # Ends with articles/determiners
                            r'^(ag|agreement|contract|letter|document)',  # Starts with document words
                            r'(ag|agreement|contract|letter|document)\s*$',  # Ends with document words
                        ]
                        
                        is_invalid = False
                        for invalid_pattern in invalid_patterns:
                            if re.search(invalid_pattern, entity, re.IGNORECASE):
                                is_invalid = True
                                break
                        
                        if (not is_invalid and
                            entity_lower not in ['service provider', 'contractor', 'client', 'the entire', 'shall be', 'services', 'fees', 'p.o. box', 'sheikh', 'road', 'operations', 'is engag', 'is engaged', 'may terminate', 'terminate this'] and
                            not re.search(r'\b(p\.o\.\s*box|road|street|avenue|boulevard|address|location|sheikh\s+zayed|terminate|engage|agreement)\b', entity, re.IGNORECASE) and
                            not entity_lower.startswith('p.o.') and
                            not entity_lower.startswith('sheikh') and
                            not entity_lower.startswith('is ') and
                            not entity_lower.startswith('may ') and
                            not entity_lower.endswith(' engag') and
                            not entity_lower.endswith(' ag') and
                            not entity_lower.endswith(' terminate') and
                            # Must contain at least one capital letter (company names usually do)
                            re.search(r'[A-Z]', entity) and
                            # Must not be a sentence fragment (no verbs in middle)
                            not re.search(r'\b(is|may|shall|will|can|must|should|terminate|engage|provide)\b', entity, re.IGNORECASE)):
                            claim['entity'] = entity
                            break
            if claim['entity']:
                break
        
        # Fallback: If "(Client)" found but no name extracted, try context extraction
        if not claim['entity'] and re.search(r'\(["\']?client["\']?\)', text_normalized_entity, re.IGNORECASE):
            # Look for name immediately before "(Client)" in the same sentence
            client_context = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s+\(["\']?client["\']?\)', text_normalized_entity, re.IGNORECASE)
            if client_context:
                potential_entity = client_context.group(1).strip()
                if potential_entity.lower() not in ['service provider', 'contractor', 'client', 'the entire']:
                    claim['entity'] = potential_entity
        
        # Additional fallback: Look for company names in header (first 20 lines)
        if not claim['entity']:
            header_lines = text.split('\n')[:20]
            for line in header_lines:
                # Look for company name patterns - more specific
                # Try full company name with Industries/Group/Company first
                company_match = re.search(r'([A-Z][A-Za-z\s&]+(?:Petroleum|Energy|Oil|Gas|Industries|Group|Company)\s+[A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation))', line)
                if not company_match:
                    # Fallback to simpler pattern
                    company_match = re.search(r'([A-Z][A-Za-z\s&]+(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation|Industries|Group|Company))', line)
                
                if company_match:
                    potential_entity = company_match.group(1).strip()
                    # More strict validation - filter out sentence fragments
                    invalid_start = ['is ', 'may ', 'shall ', 'will ', 'can ', 'must ', 'should ', 'the ', 'a ', 'an ']
                    invalid_end = [' ag', ' ag.', ' agreement', ' contract', ' terminate', ' engag', ' engage']
                    
                    is_valid = (
                        len(potential_entity) >= 10 and  # At least 10 chars for company name
                        potential_entity.lower() not in ['service provider', 'contractor', 'client', 'the entire', 'operations'] and
                        not re.search(r'\b(p\.o\.\s*box|road|street|address|sheikh|is\s+engag|terminate|agreement|contract)\b', potential_entity, re.IGNORECASE) and
                        not any(potential_entity.lower().startswith(inv) for inv in invalid_start) and
                        not any(potential_entity.lower().endswith(inv) for inv in invalid_end) and
                        # Must have at least 2 words (company names usually do)
                        len(potential_entity.split()) >= 2 and
                        # Must contain capital letters
                        re.search(r'[A-Z]', potential_entity) and
                        # Must not contain verbs or action words
                        not re.search(r'\b(is|may|shall|will|can|must|should|terminate|engage|provide|agreement|contract)\b', potential_entity, re.IGNORECASE)
                    )
                    
                    if is_valid:
                        claim['entity'] = potential_entity
                        break
        
        # Extract dates
        # Effective date / Start date
        # Normalize text for better matching
        text_normalized_start_date = ' '.join(text.split())
        
        date_patterns = [
            r'effective\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format first
            r'start\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format
            r'commencement\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format
            r'from\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format
            r'dated\s+(\d{4}-\d{2}-\d{2})',  # "dated 2026-01-21"
            r'this\s+letter[,\s]+dated\s+(\d{4}-\d{2}-\d{2})',  # "This letter, dated 2026-01-21"
            r'this\s+engagement\s+letter[,\s]+dated\s+(\d{4}-\d{2}-\d{2})',  # "This Engagement Letter, dated 2026-01-21"
            r'effective\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'start\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'commencement\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'from\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'dated\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # "dated 01/21/2026"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text_normalized_start_date, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1).strip()
                    
                    # Try dateparser first (handles any format)
                    if DATEPARSER_AVAILABLE:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date:
                            claim['start_date'] = parsed_date
                            break
                    
                    # Fallback to dateutil parser
                    date_str = date_str.replace('/', '-')
                    claim['start_date'] = parser.parse(date_str, dayfirst=False)  # ISO format first
                    break
                except:
                    try:
                        # Fallback with dayfirst=True
                        if DATEPARSER_AVAILABLE:
                            parsed_date = dateparser.parse(match.group(1))
                            if parsed_date:
                                claim['start_date'] = parsed_date
                                break
                        claim['start_date'] = parser.parse(match.group(1), dayfirst=True)
                        break
                    except:
                        pass
        
        # End date / Expiry date
        # IMPROVED: Better date patterns including ISO format, traditional format, and written format
        # Normalize text for better matching
        text_normalized_dates = ' '.join(text.split())
        
        end_date_patterns = [
            # Written format (highest priority for contracts): "January 14, 2025"
            r'expiration\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "Expiration Date: January 14, 2025"
            r'expiry\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "Expiry Date: January 14, 2025"
            r'end\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "End Date: January 14, 2025"
            r'termination\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "Termination Date: January 14, 2025"
            r'until\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "until January 14, 2025"
            r'continue\s+until\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "continue until January 14, 2025"
            # ISO format patterns
            r'end\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format: "end date: 2025-01-14"
            r'expiry\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format: "expiry date: 2025-01-14"
            r'expiration\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format: "expiration date: 2025-01-14"
            r'termination\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format: "termination date: 2025-01-14"
            r'until\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # ISO format: "until date: 2025-01-14"
            r'expires?\s+(?:on\s+)?(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "expires 2025-01-14" or "expires on 2025-01-14"
            r'valid\s+until\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "valid until 2025-01-14"
            r'contract\s+end[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "contract end: 2025-01-14"
            r'ending\s+(?:on\s+)?(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "ending 2025-01-14"
            r'through\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "through 2025-01-14"
            # Traditional format patterns
            r'end\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format: "end date: 01/14/2025"
            r'expiry\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'expiration\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'termination\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'until\s+date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Traditional format
            r'expires?\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # "expires 01/14/2025"
        ]
        
        for pattern in end_date_patterns:
            match = re.search(pattern, text_normalized_dates, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1).strip()
                    
                    # Try dateparser first (handles any format including "January 14, 2025")
                    if DATEPARSER_AVAILABLE:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date:
                            claim['end_date'] = parsed_date
                            break
                    
                    # Fallback to dateutil parser
                    # Handle written format like "January 14, 2025"
                    if re.search(r'[A-Z][a-z]+\s+\d{1,2}', date_str):
                        # Written format - parse directly
                        claim['end_date'] = parser.parse(date_str)
                    else:
                        # Numeric format
                        date_str = date_str.replace('/', '-')
                        claim['end_date'] = parser.parse(date_str, dayfirst=False)  # ISO format first
                    break
                except:
                    try:
                        # Fallback: try with different parsing options
                        date_str = match.group(1).strip()
                        if DATEPARSER_AVAILABLE:
                            parsed_date = dateparser.parse(date_str)
                            if parsed_date:
                                claim['end_date'] = parsed_date
                                break
                        if re.search(r'[A-Z][a-z]+', date_str):
                            claim['end_date'] = parser.parse(date_str)
                        else:
                            claim['end_date'] = parser.parse(date_str, dayfirst=True)
                        break
                    except:
                        pass
        
        # Extract amount and currency (for engagement letters, contracts)
        # IMPROVED: Use priority-based extraction with correct order
        # Normalize text for better matching (remove extra whitespace, handle newlines)
        text_normalized = ' '.join(text.split())
        
        # Initialize amount_patterns to avoid UnboundLocalError
        amount_patterns = []
        
        # Use new priority-based extraction method FIRST
        try:
            amount_data = self._extract_amount_with_priority(text_normalized, compensation_table_data)
            if amount_data and amount_data.get('amount'):
                print(f"✓ Found amount via priority method: {amount_data.get('amount')} {amount_data.get('currency')}")
                claim['amount'] = amount_data.get('amount')
                claim['currency'] = amount_data.get('currency', 'AED')
                claim['secondary_amount'] = amount_data.get('secondary_amount')
                claim['secondary_currency'] = amount_data.get('secondary_currency')
                # Skip old pattern-based extraction - we got a result from priority method
            else:
                print("⚠ Priority method didn't find amount, falling back to old patterns")
        except Exception as e:
            print(f"⚠ Error in _extract_amount_with_priority: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to old patterns
        
        # Only use old patterns if new method didn't work
        if not claim.get('amount'):
            # Fallback to old patterns if new method didn't work
            print("⚠ Using fallback patterns for amount extraction")
            amount_patterns = [
                    # PRIORITY 0: "Annual Contract Value" - HIGHEST PRIORITY (explicit contract total)
                    r'annual\s+contract\s+value[:\s]*\*?\*?([^\*\n]+?)(?:AED|USD|EUR|GBP|aed|usd|eur|gbp)',  # "Annual Contract Value: 678,000 AED"
                    r'annual\s+contract\s+value[:\s]*\*?\*?(?:AED|USD|EUR|GBP|aed|usd|eur|gbp)\s+([\d,]+(?:\.\d+)?)',  # "Annual Contract Value: AED 678,000"
                    # PRIORITY 0: "Annual Contract Value" with multi-currency
                    r'annual\s+contract\s+value[:\s]*\*?\*?([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)',  # "Annual Contract Value: 678,000 AED (184,572 USD)"
                    # PRIORITY 0: "Total Annual Compensation" or "Total Annual Value"
                    r'total\s+annual\s+(?:compensation|value|amount)[:\s]*\*?\*?([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',
            # Currency BEFORE amount (AED 5,000) - HIGH PRIORITY
            r'(AED|USD|EUR|GBP|aed|usd|eur|gbp)\s+([\d,]+(?:\.\d+)?)',  # "AED 5,000" or "USD 3,000"
            # Annual amounts with parentheses USD equivalent (highest priority)
            r'([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)\s*(?:annual|yearly|per\s+year)',  # "678,000 AED (184,572 USD) annual"
            r'(?:annual|yearly|per\s+year)[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)\s*(?:\(([\d,]+(?:\.\d+)?)\s*(USD|EUR|GBP|usd|eur|gbp)\))?',  # "annual: 678,000 AED (184,572 USD)"
            # Monthly amounts with parentheses USD equivalent
            r'([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)\s*(?:monthly|per\s+month)',  # "56,500 AED (15,381 USD) monthly"
            r'(?:monthly|per\s+month)[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)\s*(?:\(([\d,]+(?:\.\d+)?)\s*(USD|EUR|GBP|usd|eur|gbp)\))?',  # "monthly: 56,500 AED (15,381 USD)"
            # Standalone amounts with currency AFTER (AED priority)
            r'([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*(?:\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\))?',  # "678,000 AED" or "678,000 AED (184,572 USD)"
            r'([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',  # "678,000 AED" or "3000 USD"
            # Fees patterns
            r'fees?\s+(?:charged|shall be|are)\s+(?:by\s+the\s+)?(?:service\s+provider|contractor)?\s*(?:shall be|are)?\s*\$?([\d,]+(?:\.\d+)?)',  # "fees charged by the Service Provider shall be $3000.00"
            # Amount patterns with currency label
            r'amount[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',  # "Amount: 678,000 AED"
            r'total[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',  # "Total: 678,000 AED"
            # Currency symbols
            r'\$([\d,]+(?:\.\d+)?)',  # "$3000.00"
            r'€([\d,]+(?:\.\d+)?)',  # "€3000.00"
            r'£([\d,]+(?:\.\d+)?)',  # "£3000.00"
            # Amount with currency text (fallback)
            r'([\d,]+(?:\.\d+)?)\s*(?:usd|eur|gbp|aed|us\s*dollars?|euros?|pounds?|dirhams?)',  # "3000.00 USD" or "678000 AED"
        ]
        
        # Track all found amounts to pick the largest (usually annual)
        found_amounts = []
        
        # Only process patterns if they were defined (fallback patterns)
        if amount_patterns:
            for pattern in amount_patterns:
                matches = list(re.finditer(pattern, text_normalized, re.IGNORECASE))
                for match in matches:
                    try:
                        # Initialize variables
                        amount_value = None
                        currency = None
                        
                        # Try price_parser first for better extraction (if available)
                        if PRICE_PARSER_AVAILABLE:
                            # Extract text around the match for price_parser
                            context_start = max(0, match.start() - 50)
                            context_end = min(len(text_normalized), match.end() + 50)
                            context_text = text_normalized[context_start:context_end]
                            
                            try:
                                price = Price.fromstring(context_text)
                                if price and price.amount and price.amount > 0:
                                    amount_value = float(price.amount)
                                    currency = price.currency.upper() if price.currency else None
                                    
                                    # Only use if currency is valid
                                    if not currency or currency not in ['AED', 'USD', 'EUR', 'GBP']:
                                        # Invalid currency, fallback to regex
                                        amount_value = None
                                        currency = None
                            except:
                                # Fallback to regex extraction
                                amount_value = None
                                currency = None
                        
                        # Regex extraction (fallback or if price_parser not available)
                        if amount_value is None or currency is None:
                            # Handle pattern where currency comes BEFORE amount (AED 5,000)
                            if pattern.startswith(r'(AED|USD|EUR|GBP|aed|usd|eur|gbp)\s+'):
                                # Currency is group 1, amount is group 2
                                currency_code = match.group(1).upper()
                                amount_str = match.group(2).replace(',', '')
                            else:
                                # Normal pattern: amount is group 1, currency might be group 2
                                amount_str = match.group(1).replace(',', '')
                                currency_code = match.group(2).upper() if len(match.groups()) >= 2 and match.group(2) else None
                            
                            amount_value = float(amount_str)
                            
                            # Extract currency from match groups or context
                            currency = None
                            if currency_code and currency_code in ['AED', 'USD', 'EUR', 'GBP']:
                                currency = currency_code
                        
                        # If no currency in match, search in context
                        if not currency:
                            context_start = max(0, match.start() - 100)
                            context_end = min(len(text_normalized), match.end() + 100)
                            context = text_normalized[context_start:context_end]
                            
                            currency_match = re.search(r'\b(AED|USD|EUR|GBP|aed|usd|eur|gbp|dirhams?|dollars?|euros?|pounds?)\b', context, re.IGNORECASE)
                            if currency_match:
                                curr = currency_match.group(1).upper()
                                if curr in ['AED', 'USD', 'EUR', 'GBP']:
                                    currency = curr
                                elif 'DOLLAR' in curr or curr == 'USD':
                                    currency = 'USD'
                                elif 'EURO' in curr or curr == 'EUR':
                                    currency = 'EUR'
                                elif 'POUND' in curr or curr == 'GBP':
                                    currency = 'GBP'
                                elif 'DIRHAM' in curr or curr == 'AED':
                                    currency = 'AED'
                        
                        if not currency:
                            currency = 'USD'  # Default
                        
                        # Only add if amount is reasonable (not a decimal rate like 3.6725)
                        # For contracts, amounts are usually > 1000
                        # Filter out small decimals that look like exchange rates (e.g., 3.6725)
                        is_likely_rate = amount_value < 10 and '.' in amount_str and len(amount_str.split('.')[1]) >= 3
                        
                        # Also check context for exchange rate indicators
                        # Use wider context to better detect annual/monthly/allowance keywords
                        context_start = max(0, match.start() - 150)
                        context_end = min(len(text_normalized), match.end() + 150)
                        context = text_normalized[context_start:context_end].lower()
                        is_exchange_rate = (
                            'exchange rate' in context or
                            'rate' in context and ('usd' in context or 'aed' in context) and amount_value < 10 or
                            '=' in context and amount_value < 10  # "1 USD = 3.6725 AED"
                        )
                        
                        # Check if this is a specific allowance/budget (not main compensation)
                        # More specific check - must have both allowance/budget term AND be small amount
                        is_allowance_or_budget = (
                            amount_value < 20000 and  # Only small amounts can be allowances
                            any(term in context for term in [
                                'professional development', 'development budget', 'housing allowance',
                                'transportation allowance', 'travel allowance', 'meal allowance',
                                'allowance', 'budget', 'bonus', 'incentive'
                            ])
                        )
                        
                        # Determine priority based on context
                        priority = 5  # Default: medium priority (lower is better, 0 = highest priority)
                        amount_type = 'other'
                        
                        # PRIORITY 0: "Annual Contract Value" (explicit contract total - ABSOLUTE highest priority)
                        if 'annual contract value' in context:
                            priority = 0  # ABSOLUTE highest priority: explicit annual contract value
                            amount_type = 'annual_contract_value'
                        # PRIORITY 0: "Total Annual Compensation" or "Total Annual Value"
                        elif 'total annual' in context and ('compensation' in context or 'value' in context or 'amount' in context):
                            priority = 0  # Highest priority: total annual
                            amount_type = 'annual_total'
                        # Check for annual amounts (high priority) - but NOT if it's monthly
                        elif ('annual' in context or 'yearly' in context or 'per year' in context) and 'monthly' not in context:
                            if 'total' in context or 'compensation' in context or amount_value > 50000:
                                priority = 0  # Highest priority: annual total
                                amount_type = 'annual_total'
                            else:
                                priority = 1  # Annual but not total
                                amount_type = 'annual'
                        # Check for monthly amounts (LOWER priority - only if annual not found)
                        elif 'monthly' in context or 'per month' in context:
                            # CRITICAL: Monthly amounts should NEVER be selected if there's an annual amount
                            # Set lower priority to ensure annual is always preferred
                            if 'total' in context or 'compensation' in context:
                                priority = 5  # Lower priority: monthly total compensation (was 2, now 5)
                                amount_type = 'monthly_total'
                            elif 'base' in context or 'base fee' in context:
                                priority = 6  # Lower priority: base fee monthly (was 3, now 6)
                                amount_type = 'base_fee'
                            else:
                                priority = 7  # Lower priority: other monthly (was 4, now 7)
                                amount_type = 'monthly'
                        # Check for allowances and budgets (lowest priority)
                        elif is_allowance_or_budget:
                            priority = 10  # Lowest priority: allowances and budgets
                            amount_type = 'allowance'
                        # If amount is very large (> 100000), likely annual total even without explicit label
                        elif amount_value > 100000:
                            priority = 0  # Very large amounts are likely annual totals
                            amount_type = 'annual_total'
                        # If amount is medium-large (50000-100000), likely annual
                        elif amount_value > 50000:
                            priority = 1
                            amount_type = 'annual'
                        # If amount is small (< 10000), likely allowance or specific item
                        elif amount_value < 10000:
                            priority = 8  # Lower priority for small amounts
                            amount_type = 'other'
                        
                        if not is_likely_rate and not is_exchange_rate:
                            if amount_value >= 1000 or (amount_value >= 100 and currency == 'AED'):
                                found_amounts.append({
                                    'amount': amount_value,
                                    'currency': currency,
                                    'position': match.start(),
                                    'priority': priority,
                                    'amount_type': amount_type,
                                    'context': context[:100]  # Store context for debugging
                                })
                            # Also track smaller amounts with AED for potential secondary amounts (but not rates)
                            elif amount_value >= 10 and currency == 'AED' and amount_value < 1000 and not is_allowance_or_budget:
                                found_amounts.append({
                                    'amount': amount_value,
                                    'currency': currency,
                                    'position': match.start(),
                                    'is_secondary': True,
                                    'priority': priority,
                                    'amount_type': amount_type
                                })
                    except:
                        pass
        
        # Select the best amount based on priority and type
        if found_amounts:
            # Filter out very small amounts and secondary amounts
            significant_amounts = [a for a in found_amounts if a['amount'] >= 1000 and not a.get('is_secondary', False)]
            
            if significant_amounts:
                # CRITICAL: Filter out monthly amounts if there are ANY annual amounts
                annual_amounts = [a for a in significant_amounts if a.get('priority', 5) == 0 or a.get('amount_type') in ['annual_contract_value', 'annual_total', 'annual']]
                monthly_amounts = [a for a in significant_amounts if a.get('amount_type') in ['monthly_total', 'monthly', 'base_fee']]
                
                # If we have annual amounts, EXCLUDE monthly amounts completely
                if annual_amounts:
                    significant_amounts = annual_amounts
                    # Also filter out monthly from the list
                    significant_amounts = [a for a in significant_amounts if a.get('amount_type') not in ['monthly_total', 'monthly', 'base_fee']]
                
                # Sort by priority (lower = better), then by amount (larger = better)
                # Priority order: annual_contract_value (0) > annual_total (0) > annual (1) > monthly_total (5) > base_fee (6) > monthly (7) > other (5+) > allowance (10)
                # CRITICAL: annual_contract_value gets absolute priority
                significant_amounts.sort(key=lambda x: (
                    x.get('priority', 5),  # Priority (0 = highest)
                    -x['amount'] if x.get('amount_type') != 'annual_contract_value' else -999999,  # Annual Contract Value always first
                    -x['amount']  # Then by amount (larger = better)
                ))
                
                # Debug: print all amounts with their priorities (for troubleshooting)
                # print(f"DEBUG: Found amounts: {[(a['amount'], a.get('priority', 5), a.get('amount_type', 'other')) for a in significant_amounts[:5]]}")
                
                # CRITICAL: Check for "Annual Contract Value" first (absolute priority)
                annual_contract_value_amounts = [a for a in significant_amounts if a.get('amount_type') == 'annual_contract_value']
                if annual_contract_value_amounts:
                    # Annual Contract Value always wins - take the first one (already sorted by priority)
                    best = annual_contract_value_amounts[0]
                # CRITICAL: Also check for annual_total (priority 0) - prefer over monthly
                elif any(a.get('priority', 5) == 0 and a.get('amount_type') == 'annual_total' for a in significant_amounts):
                    annual_total_amounts = [a for a in significant_amounts if a.get('priority', 5) == 0 and a.get('amount_type') == 'annual_total']
                    # Sort by amount (largest first) to get the best annual total
                    annual_total_amounts.sort(key=lambda x: -x['amount'])
                    best = annual_total_amounts[0]
                else:
                    # Prefer AED currency if present
                    aed_amounts = [a for a in significant_amounts if a['currency'] == 'AED']
                    if aed_amounts:
                        # CRITICAL: If there's an amount that's 10x+ larger, it's almost certainly the main compensation
                        # This handles cases where context doesn't clearly indicate "annual total"
                        largest_aed = max(aed_amounts, key=lambda x: x['amount'])
                        second_largest_aed = sorted(aed_amounts, key=lambda x: x['amount'], reverse=True)[1] if len(aed_amounts) > 1 else None
                        
                        if second_largest_aed and largest_aed['amount'] > second_largest_aed['amount'] * 10:
                            # One amount is 10x+ larger - it's definitely the main compensation
                            best = largest_aed
                        else:
                            # Take the highest priority AED amount
                            best = aed_amounts[0]
                            # But if there's a much larger amount (5x+), prefer it even if slightly lower priority
                            if largest_aed['amount'] > best['amount'] * 5 and largest_aed.get('priority', 5) <= 3:
                                # Large amount with reasonable priority - prefer it
                                best = largest_aed
                    else:
                        # Take the highest priority amount regardless of currency
                        best = significant_amounts[0]
                        # Same logic: prefer much larger amounts
                        if len(significant_amounts) > 1:
                            largest = max(significant_amounts, key=lambda x: x['amount'])
                            second_largest = sorted(significant_amounts, key=lambda x: x['amount'], reverse=True)[1]
                            
                            if largest['amount'] > second_largest['amount'] * 10:
                                # One amount is 10x+ larger - it's definitely the main compensation
                                best = largest
                            elif largest['amount'] > best['amount'] * 5 and largest.get('priority', 5) <= 3:
                                best = largest
                
                claim['amount'] = best['amount']
                claim['currency'] = best['currency']
                
                # Try to find secondary currency (USD equivalent) in context
                # Look for patterns like "678,000 AED (184,572 USD)"
                if best.get('context'):
                    secondary_match = re.search(r'\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)', best['context'], re.IGNORECASE)
                    if secondary_match:
                        claim['secondary_amount'] = float(secondary_match.group(1).replace(',', ''))
                        claim['secondary_currency'] = 'USD'
                
                # Also search in wider context around the best amount
                if not claim.get('secondary_currency'):
                    context_start = max(0, best['position'] - 200)
                    context_end = min(len(text_normalized), best['position'] + 200)
                    wider_context = text_normalized[context_start:context_end]
                    secondary_match = re.search(r'\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)', wider_context, re.IGNORECASE)
                    if secondary_match:
                        claim['secondary_amount'] = float(secondary_match.group(1).replace(',', ''))
                        claim['secondary_currency'] = 'USD'
            else:
                # Fallback: try amounts >= 100 (but might be monthly)
                fallback_amounts = [a for a in found_amounts if a['amount'] >= 100 and not a.get('is_secondary', False)]
                if fallback_amounts:
                    # Sort by priority
                    fallback_amounts.sort(key=lambda x: (x.get('priority', 5), -x['amount']))
                    # Prefer AED
                    aed_fallback = [a for a in fallback_amounts if a['currency'] == 'AED']
                    if aed_fallback:
                        best = aed_fallback[0]
                    else:
                        best = fallback_amounts[0]
                    claim['amount'] = best['amount']
                    claim['currency'] = best['currency']
                    
                    # Try to find secondary currency
                    if best.get('context'):
                        secondary_match = re.search(r'\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)', best['context'], re.IGNORECASE)
                        if secondary_match:
                            claim['secondary_amount'] = float(secondary_match.group(1).replace(',', ''))
                            claim['secondary_currency'] = 'USD'
                else:
                    # Last resort: if no significant amounts, check if there's at least one AED amount (excluding allowances)
                    aed_amounts = [a for a in found_amounts if a['currency'] == 'AED' and not a.get('is_secondary', False) and a.get('amount_type') != 'allowance']
                    if aed_amounts:
                        # Sort by priority
                        aed_amounts.sort(key=lambda x: (x.get('priority', 5), -x['amount']))
                        best = aed_amounts[0]
                        claim['amount'] = best['amount']
                        claim['currency'] = best['currency']
            
            # Secondary currency extraction is now handled above in the amount selection logic
            # This is a fallback if not found in context
            if not claim.get('secondary_currency'):
                # Look for patterns like "678,000 AED (184,572 USD)" in the full text
                secondary_match = re.search(r'([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)', text_normalized, re.IGNORECASE)
                if secondary_match:
                    # Check if the AED amount matches our selected amount (or is close)
                    aed_amount = float(secondary_match.group(1).replace(',', ''))
                    if abs(aed_amount - claim.get('amount', 0)) < 1000 or aed_amount > 50000:  # Close match or large amount
                        claim['secondary_currency'] = 'USD'
                        claim['secondary_amount'] = float(secondary_match.group(3).replace(',', ''))
        
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
    
    def _extract_resume_data(self, text: str, claim: Dict[str, Any], role: Role) -> Dict[str, Any]:
        """
        Extract personal and professional data from Resume/CV
        
        Extracts:
        - Name (first name, last name, full name)
        - Email
        - Phone
        - Address
        - Date of birth
        - Nationality
        - Education
        - Work experience
        - Skills
        """
        text_normalized = ' '.join(text.split())
        text_lines = text.split('\n')
        
        # Extract full name (usually at the top of CV, before profession/email)
        # Try to get first 1-3 lines, exclude common profession keywords
        profession_keywords = ['sistemista', 'developer', 'engineer', 'manager', 'analyst', 'consultant', 'designer', 'architect']
        
        name_patterns = [
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # First line with capitalized words
            r'^([A-Z][A-Z\s]{2,30})$',  # All caps name (common in CVs) - limit length
        ]
        
        full_name = None
        # Get first few lines
        first_lines = text.split('\n')[:3]
        first_line = first_lines[0].strip() if first_lines else ""
        
        # Check if first line looks like a name (not a profession)
        if first_line:
            first_line_lower = first_line.lower()
            # Exclude if it contains profession keywords
            if not any(keyword in first_line_lower for keyword in profession_keywords):
                # Check if it matches name pattern (2-3 capitalized words)
                name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})$', first_line)
                if name_match:
                    full_name = name_match.group(1).strip()
        
        # Fallback to other patterns
        if not full_name:
            for pattern in name_patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    candidate = match.group(1).strip()
                    # Exclude if contains profession keywords
                    candidate_lower = candidate.lower()
                    if not any(keyword in candidate_lower for keyword in profession_keywords):
                        # Take first 2-3 words max (first name, middle name, last name)
                        name_parts = candidate.split()[:3]
                        full_name = ' '.join(name_parts)
                        if len(full_name) > 5 and len(full_name) < 50:  # Reasonable name length
                            break
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        email = email_match.group(0) if email_match else None
        
        # Extract phone (improved patterns to avoid dates)
        phone_patterns = [
            r'phone[:\s]+([+\d\s\-\(\)]{8,20})',  # Explicit phone label
            r'telephone[:\s]+([+\d\s\-\(\)]{8,20})',
            r'tel[:\s]+([+\d\s\-\(\)]{8,20})',
            r'mobile[:\s]+([+\d\s\-\(\)]{8,20})',
            r'cell[:\s]+([+\d\s\-\(\)]{8,20})',
            r'\b(\+\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{4,9})\b',  # International format: +39 392 050 6871
            r'\b(\d{3}[\s\-]?\d{3}[\s\-]?\d{4,6})\b',  # Local format: 392 050 6871
        ]
        phone = None
        for pattern in phone_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate = re.sub(r'\s+', ' ', match.group(1).strip())
                # Exclude if it looks like a date (e.g., "05/07/2001")
                if '/' not in candidate and len(candidate) >= 8:  # Phone should be at least 8 chars, no slashes
                    phone = candidate
                    break
            if phone:
                break
        
        # Extract address (improved to catch city, country format)
        address_patterns = [
            r'address[:\s]+(.+?)(?:\n|email|phone|\d{4}|$)',
            r'location[:\s]+(.+?)(?:\n|email|phone|\d{4}|$)',
            r'residence[:\s]+(.+?)(?:\n|email|phone|\d{4}|$)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:Italia|Italy|USA|UK|France|Germany|Spain))',  # City, Country
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[–\-]\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',  # City – Date format
        ]
        address = None
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                # Clean up: remove date if present
                candidate = re.sub(r'\s*[–\-]\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{4}.*$', '', candidate)
                # Remove common profession keywords that might be in the same line
                candidate = re.sub(r'\b(?:Sistemista|Developer|Engineer|Manager|Analyst|Consultant|Designer|Architect|Help|Desk|Informatico|Configurazione|Reti|Automazione|Excel)\b', '', candidate, flags=re.IGNORECASE)
                candidate = re.sub(r'\s+', ' ', candidate).strip()  # Normalize whitespace
                candidate = candidate[:100]  # Limit length
                if len(candidate) > 5 and not candidate.lower().startswith('automazione'):  # At least city name, exclude false positives
                    address = candidate
                    break
        
        # If still no address, try to extract city from context (near date of birth or phone)
        if not address:
            # Look for pattern: "City, Country – Date" or "City Country"
            city_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*(?:Italia|Italy|USA|UK|France|Germany|Spain)', text, re.IGNORECASE)
            if city_match:
                address = city_match.group(0).strip()
        
        # Extract date of birth (improved to catch Italian format)
        dob_patterns = [
            r'date\s+of\s+birth[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'born[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'nato[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'nato\s+il[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'dob[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b',  # Generic date pattern (DD/MM/YYYY or DD-MM-YYYY)
        ]
        date_of_birth = None
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                # Validate it's a reasonable birth date (between 1950 and 2010)
                try:
                    parts = re.split(r'[/\-]', date_str)
                    if len(parts) == 3:
                        year = int(parts[2]) if len(parts[2]) == 4 else int(parts[0]) if len(parts[0]) == 4 else None
                        if year and 1950 <= year <= 2010:  # Reasonable birth year
                            date_of_birth = date_str
                            break
                except:
                    pass
        
        # Extract nationality
        nationality_patterns = [
            r'nationality[:\s]+([A-Z][a-z]+)',
            r'citizenship[:\s]+([A-Z][a-z]+)',
            r'nazionalit[àa][:\s]+([A-Z][a-z]+)',
        ]
        nationality = None
        for pattern in nationality_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                nationality = match.group(1).strip()
                break
        
        # Extract education (simplified - just first degree/university found)
        education_patterns = [
            r'education[:\s]+(.+?)(?:\n\n|experience|work|skills|$)',
            r'university[:\s]+([A-Z][A-Za-z\s]+)',
            r'degree[:\s]+([A-Z][A-Za-z\s]+)',
            r'laurea[:\s]+([A-Z][A-Za-z\s]+)',
        ]
        education = None
        for pattern in education_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                education = match.group(1).strip()[:200]
                if len(education) > 10:
                    break
        
        # Extract work experience (simplified - just first position found)
        experience_patterns = [
            r'experience[:\s]+(.+?)(?:\n\n|education|skills|$)',
            r'work\s+experience[:\s]+(.+?)(?:\n\n|education|skills|$)',
            r'professional\s+experience[:\s]+(.+?)(?:\n\n|education|skills|$)',
        ]
        experience = None
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                experience = match.group(1).strip()[:300]
                if len(experience) > 20:
                    break
        
        # Extract skills (simplified - common skills keywords)
        skills_keywords = [
            'python', 'java', 'javascript', 'sql', 'react', 'angular', 'vue',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'linux', 'agile',
            'project management', 'leadership', 'communication', 'analytics'
        ]
        skills_found = []
        text_lower = text.lower()
        for skill in skills_keywords:
            if skill in text_lower:
                skills_found.append(skill.title())
        
        # Update claim with extracted data
        if full_name:
            claim['entity'] = full_name  # Use entity field for person name
            claim['subject'] = full_name  # Also use subject field
        
        # Store additional CV data in claim (will be extracted separately)
        claim['cv_data'] = {
            'full_name': full_name,
            'first_name': full_name.split()[0] if full_name and len(full_name.split()) > 0 else None,
            'last_name': ' '.join(full_name.split()[1:]) if full_name and len(full_name.split()) > 1 else None,
            'email': email,
            'phone': phone,
            'address': address,
            'date_of_birth': date_of_birth,
            'nationality': nationality,
            'education': education,
            'experience': experience,
            'skills': skills_found[:10] if skills_found else None,  # Limit to 10 skills
        }
        
        return claim
    
    def _extract_professional_services_claims(self, text: str, claim: Dict[str, Any], role: Role) -> Dict[str, Any]:
        """
        Enhanced extraction for Professional Services Agreement
        
        Looks for:
        - Total Project Value (amount)
        - Multi-currency amounts (AED, USD, EUR)
        - Effective Date / Expiration Date
        - Scope of Services
        - Contracting Parties (entity)
        """
        text_normalized = ' '.join(text.split())
        text_lower = text_normalized.lower()
        
        # 1. Extract Annual Contract Value (HIGHEST PRIORITY - explicit contract total)
        # CRITICAL: These patterns must match BEFORE any monthly patterns
        annual_contract_value_patterns = [
            # Pattern 1: "Annual Contract Value: 678,000 AED" (most common)
            r'annual\s+contract\s+value[:\s]*\*?\*?([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',
            # Pattern 2: "Annual Contract Value: AED 678,000"
            r'annual\s+contract\s+value[:\s]*\*?\*?(?:AED|USD|EUR|GBP|aed|usd|eur|gbp)\s+([\d,]+(?:\.\d+)?)',
            # Pattern 3: "Annual Contract Value: 678,000 AED (184,572 USD)" (with secondary currency)
            r'annual\s+contract\s+value[:\s]*\*?\*?([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)',
            # Pattern 4: "Annual Contract Value" on one line, amount on next line
            r'annual\s+contract\s+value[:\s]*\n\s*([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',
            # Pattern 5: More flexible - allows for extra words
            r'annual\s+contract\s+value[:\s]*.*?([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP|aed|usd|eur|gbp)',
        ]
        
        for pattern in annual_contract_value_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    # Extract amount (could be in group 1 or need parsing)
                    amount_str = match.group(1).strip()
                    # Remove bold markers and clean
                    amount_str = re.sub(r'\*+', '', amount_str)
                    amount_str = amount_str.replace(',', '').strip()
                    # Extract number
                    num_match = re.search(r'([\d.]+)', amount_str)
                    if num_match:
                        amount_value = float(num_match.group(1))
                        
                        # Extract currency
                        currency = 'AED'  # Default
                        if len(match.groups()) >= 2 and match.group(2):
                            currency = match.group(2).upper()
                        else:
                            # Search for currency in match
                            currency_match = re.search(r'\b(AED|USD|EUR|GBP|aed|usd|eur|gbp)\b', match.group(0), re.IGNORECASE)
                            if currency_match:
                                currency = currency_match.group(1).upper()
                        
                        # Extract secondary currency (USD) if present
                        if len(match.groups()) >= 4 and match.group(3) and match.group(4):
                            secondary_amount_str = match.group(3).replace(',', '')
                            secondary_amount = float(secondary_amount_str)
                            claim['secondary_amount'] = secondary_amount
                            claim['secondary_currency'] = 'USD'
                        
                        if amount_value > 1000:  # Reasonable threshold
                            claim['amount'] = amount_value
                            claim['currency'] = currency if currency in ['AED', 'USD', 'EUR', 'GBP'] else 'AED'
                            claim['extraction_method'] = 'annual_contract_value'
                            # Skip other patterns if we found Annual Contract Value
                            return claim
                except Exception as e:
                    pass
        
        # 2. Extract Total Project Value (fallback if Annual Contract Value not found)
        total_value_patterns = [
            r'total\s+(?:project\s+)?value[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP)',
            r'total\s+(?:project\s+)?value[:\s]+(?:AED|USD|EUR|GBP)\s+([\d,]+(?:\.\d+)?)',
            r'total\s+contract\s+value[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP)',
            r'total\s+agreement\s+value[:\s]+([\d,]+(?:\.\d+)?)\s*(AED|USD|EUR|GBP)',
        ]
        
        for pattern in total_value_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount_value = float(amount_str)
                    currency = match.group(2).upper() if len(match.groups()) >= 2 else 'AED'
                    
                    if amount_value > 1000:  # Reasonable threshold
                        claim['amount'] = amount_value
                        claim['currency'] = currency if currency in ['AED', 'USD', 'EUR', 'GBP'] else 'AED'
                        break
                except:
                    pass
        
        # 2. Extract multi-currency amounts (look for patterns like "2,940,000 AED (800,350 USD)")
        multi_currency_patterns = [
            r'([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\)',
            r'([\d,]+(?:\.\d+)?)\s*(AED|aed)\s*\(([\d,]+(?:\.\d+)?)\s*(USD|usd)\s*/\s*([\d,]+(?:\.\d+)?)\s*(EUR|eur)\)',  # Triple currency
        ]
        
        for pattern in multi_currency_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    # Primary currency (AED)
                    aed_amount = float(match.group(1).replace(',', ''))
                    if not claim.get('amount') and aed_amount > 1000:
                        claim['amount'] = aed_amount
                        claim['currency'] = 'AED'
                    
                    # Secondary currency (USD)
                    if len(match.groups()) >= 4:
                        usd_amount = float(match.group(3).replace(',', ''))
                        if usd_amount > 1000:
                            claim['secondary_amount'] = usd_amount
                            claim['secondary_currency'] = 'USD'
                    
                    # Tertiary currency (EUR) if present
                    if len(match.groups()) >= 6:
                        eur_amount = float(match.group(5).replace(',', ''))
                        if eur_amount > 1000:
                            claim['tertiary_amount'] = eur_amount
                            claim['tertiary_currency'] = 'EUR'
                    
                    break
                except:
                    pass
        
        # 3. Extract Effective Date (start_date)
        effective_date_patterns = [
            r'effective\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "Effective Date: March 1, 2024"
            r'effective\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "Effective Date: 2024-03-01"
            r'commencement\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'commencement\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        
        for pattern in effective_date_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1).strip()
                    if DATEPARSER_AVAILABLE:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date:
                            claim['start_date'] = parsed_date
                            break
                    else:
                        claim['start_date'] = parser.parse(date_str)
                        break
                except:
                    pass
        
        # 4. Extract Expiration Date (end_date)
        expiration_date_patterns = [
            r'expiration\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',  # "Expiration Date: August 31, 2025"
            r'expiration\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # "Expiration Date: 2025-08-31"
            r'expiry\s+date[:\s]+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'expiry\s+date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        
        for pattern in expiration_date_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1).strip()
                    if DATEPARSER_AVAILABLE:
                        parsed_date = dateparser.parse(date_str)
                        if parsed_date:
                            claim['end_date'] = parsed_date
                            break
                    else:
                        claim['end_date'] = parser.parse(date_str)
                        break
                except:
                    pass
        
        # 5. Extract Scope of Services (subject)
        scope_patterns = [
            r'scope\s+of\s+(?:professional\s+)?services[:\s]+(.+?)(?:\n\n|\d+\.|BACKGROUND|CONTRACT|$)',  # "SCOPE OF SERVICES: IT consulting..."
            r'scope\s+of\s+work[:\s]+(.+?)(?:\n\n|\d+\.|BACKGROUND|CONTRACT|$)',  # "Scope of Work: ..."
            r'background[:\s]+(.+?)(?:\n\n|\d+\.|SCOPE|CONTRACT|$)',  # "BACKGROUND: IT consulting..."
        ]
        
        for pattern in scope_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
            if match:
                scope = match.group(1).strip()
                # Clean up: remove extra whitespace, take first 200 chars
                scope = re.sub(r'\s+', ' ', scope)
                scope = scope[:200]
                if len(scope) > 20:  # Meaningful scope
                    claim['subject'] = scope
                    break
        
        # 6. Extract Entity (Service Provider or Client company name)
        entity_patterns = [
            r'service\s+provider.*?company\s+name[:\s]+([A-Z][A-Za-z\s&]+(?:DMCC|LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation))',
            r'client.*?company\s+name[:\s]+([A-Z][A-Za-z\s&]+(?:DMCC|LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation))',
            r'contracting\s+party.*?company\s+name[:\s]+([A-Z][A-Za-z\s&]+(?:DMCC|LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation))',
        ]
        
        for pattern in entity_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
            if match:
                entity = match.group(1).strip()
                # Filter out false positives
                if len(entity) > 5 and entity.lower() not in ['service provider', 'client', 'contractor']:
                    claim['entity'] = entity
                    break
        
        return claim
