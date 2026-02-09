"""
Roles & Holder Extractor - Deterministic Enterprise-Grade Document Intelligence

This module extracts roles and holders from complex legal documents using a hybrid approach:
1. LLM suggests candidate entities and roles (signals only)
2. Python makes deterministic decisions based on explicit rules
3. Contact information extracted deterministically
4. Fully explainable and deterministic final decisions

Key Features:
- Deterministic decision-making (Python rules, not LLM)
- Multi-entity extraction with contact info
- Semantic role inference with evidence
- Primary holder identification
- Works on long, multi-page documents
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class EntityType(str, Enum):
    """Entity types"""
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"


@dataclass
class ContactInfo:
    """Contact information for an entity"""
    email: Optional[str] = None
    phone: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "email": self.email,
            "phone": self.phone
        }


@dataclass
class RoleAssignment:
    """A role assignment for an entity"""
    role_type: str  # e.g., "contractor", "client", "employer"
    entity_type: EntityType
    name: str
    contact_info: ContactInfo = field(default_factory=ContactInfo)
    confidence: float = 0.0  # 0.0 to 1.0
    evidence: str = ""  # Brief textual explanation with semantic reference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role_type": self.role_type,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "contact_info": self.contact_info.to_dict(),
            "confidence": self.confidence,
            "evidence": self.evidence
        }


@dataclass
class RolesExtractionResult:
    """Result of roles extraction"""
    roles: List[RoleAssignment] = field(default_factory=list)
    primary_holder: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    reasoning: str = ""  # Internal reasoning (not exposed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering low-confidence roles but always returning structured JSON"""
        # Filter roles with confidence >= 0.75
        filtered_roles = [role.to_dict() for role in self.roles if role.confidence >= 0.75]
        
        # Always return structured JSON, even if empty
        result = {
            "roles": filtered_roles if filtered_roles else [],
            "primary_holder": self.primary_holder if self.primary_holder and self.primary_holder.get("confidence", 0) >= 0.70 else None,  # Lower threshold for primary holder
            "confidence": self.confidence
        }
        
        # Ensure primary_holder is always included in structure (even if None)
        if "primary_holder" not in result:
            result["primary_holder"] = None
            
        return result


class RolesExtractor:
    """
    Deterministic Roles & Holder Extractor
    
    Strategy:
    1. LLM suggests candidate entities and roles (signals only)
    2. Python validates and decides using deterministic rules
    3. Extracts contact info deterministically
    4. Calculates confidence deterministically
    5. Identifies primary holder based on explicit rules
    """
    
    def __init__(self, use_llm: bool = True, llm_provider: str = "openai"):
        """
        Initialize roles extractor
        
        Args:
            use_llm: Whether to use LLM for signal suggestion (if False, uses heuristics only)
            llm_provider: "openai" or "anthropic"
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        # Initialize LLM client if needed
        if use_llm:
            if llm_provider == "openai":
                try:
                    import openai
                    self.llm_client = openai.OpenAI()
                except ImportError:
                    print("⚠️  OpenAI not available. Install with: pip install openai")
                    self.use_llm = False
            elif llm_provider == "anthropic":
                try:
                    import anthropic
                    self.llm_client = anthropic.Anthropic()
                except ImportError:
                    print("⚠️  Anthropic not available. Install with: pip install anthropic")
                    self.use_llm = False
        
        # Explicit list of contractual roles to identify
        self.contractual_roles = [
            "client", "principal",
            "contractor", "service provider",
            "signatory",
            "beneficiary",
            "approver",
            "hr manager",
            "data controller", "data subject",  # GDPR
            "payee", "payer",
            "employer", "employee",
            "licensor", "licensee"
        ]
        
        # Deterministic role inference rules with explicit patterns
        # Includes approver, beneficiary, HR manager
        self.role_patterns = {
            "client": {
                "keywords": ["client", "principal", "company", "party a", "receives services", "engages", "hires"],
                "context_patterns": [
                    r"client[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)",
                    r"principal[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)",
                    r"party\s+a[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)",
                    r'\(["\']?client["\']?\)[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)',
                    r'\(["\']?principal["\']?\)[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)',
                ],
                "confidence_base": 0.90
            },
            "contractor": {
                "keywords": ["contractor", "service provider", "party b", "provides services", "independent contractor", "consultant"],
                "context_patterns": [
                    r"contractor[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)",
                    r"service\s+provider[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)",
                    r"party\s+b[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)",
                    r'\(["\']?contractor["\']?\)[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)',
                    r'\(["\']?service\s+provider["\']?\)[:\s]+([A-Z][A-Za-z\s&.,-]+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)',
                ],
                "confidence_base": 0.90
            },
            "signatory": {
                "keywords": ["signatory", "signed by", "witness", "executed by", "authorized signatory"],
                "context_patterns": [
                    r"signatory[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"signed\s+by[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"([A-Z][A-Za-z\s&.,-]+)\s+signed",
                    r"executed\s+by[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                ],
                "confidence_base": 0.85
            },
            "beneficiary": {
                "keywords": ["beneficiary", "benefits", "entitled to"],
                "context_patterns": [
                    r"beneficiary[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                ],
                "confidence_base": 0.80
            },
            "payee": {
                "keywords": ["payee", "receives payment", "paid to"],
                "context_patterns": [
                    r"payee[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                ],
                "confidence_base": 0.80
            },
            "data controller": {
                "keywords": ["data controller", "controller", "processes personal data"],
                "context_patterns": [
                    r"data\s+controller[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                ],
                "confidence_base": 0.75
            },
            "approver": {
                "keywords": ["approver", "approved by", "authorized by", "approval", "must be approved"],
                "context_patterns": [
                    r"approver[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"approved\s+by[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"authorized\s+by[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"([A-Z][A-Za-z\s&.,-]+)\s+approval",
                ],
                "confidence_base": 0.80
            },
            "hr manager": {
                "keywords": ["hr manager", "human resources", "hr department", "hr contact", "personnel manager"],
                "context_patterns": [
                    r"hr\s+manager[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"human\s+resources[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                    r"hr\s+contact[:\s]+([A-Z][A-Za-z\s&.,-]+)",
                ],
                "confidence_base": 0.75
            }
        }
    
    def extract(
        self,
        text: str,
        document_family: Optional[str] = None,
        document_subtype: Optional[str] = None,
        tables: Optional[List[Dict[str, Any]]] = None
    ) -> RolesExtractionResult:
        """
        Extract roles and entities from document
        
        DETERMINISTIC PROCESS:
        1. Combine text with table data
        2. Get LLM signals (suggestions only)
        3. Extract entities deterministically (Python rules)
        4. Assign roles deterministically (Python rules)
        5. Extract contact info deterministically
        6. Identify primary holder deterministically
        """
        """
        Extract roles and holder from document
        
        DETERMINISTIC PROCESS:
        1. Get LLM suggestions (signals only)
        2. Extract entities deterministically
        3. Assign roles using deterministic rules
        4. Extract contact info deterministically
        5. Calculate confidence deterministically
        6. Identify primary holder using deterministic rules
        
        Args:
            text: Document text (can be long, multi-page)
            document_family: Optional document family hint
            document_subtype: Optional document subtype hint
            
        Returns:
            RolesExtractionResult with roles, primary holder, and confidence
        """
        # Ensure we have enough text (check after combining with tables)
        if not text or len(text.strip()) < 50:
            return RolesExtractionResult(
                roles=[],
                primary_holder=None,
                confidence=0.0,
                reasoning="Document text too short or empty"
            )
        
        # Extract entities directly from structured table data (highest priority)
        table_entities = []
        if tables:
            try:
                table_entities = self._extract_entities_from_tables(tables)
                table_text = self._extract_text_from_tables(tables)
                if table_text:
                    text = f"{text}\n\n--- TABLES ---\n{table_text}"
            except Exception as e:
                # If table extraction fails, continue without table entities
                table_entities = []
        
        reasoning_steps = []
        
        # STEP 1: Get LLM suggestions (signals only, not final decision)
        llm_signals = None
        if self.use_llm:
            try:
                llm_signals = self._get_llm_signals(text, document_family, document_subtype)
                reasoning_steps.append("LLM provided candidate entities and role suggestions")
            except Exception as e:
                reasoning_steps.append(f"LLM signal extraction failed: {str(e)}, using deterministic extraction only")
        
        # STEP 2: Extract entities deterministically
        # Start with entities from tables (highest confidence)
        entities = table_entities.copy() if table_entities else []
        seen_table_names = {name for name, _ in entities}
        
        # Then extract from text (add if not already found in tables)
        text_entities = self._extract_entities_deterministic(text, llm_signals)
        for entity_name, entity_type in text_entities:
            if entity_name not in seen_table_names:
                # Check if it's a valid entity name
                if self._is_valid_entity_name(entity_name):
                    entities.append((entity_name, entity_type))
                    seen_table_names.add(entity_name)
        reasoning_steps.append(f"Extracted {len(entities)} candidate entities")
        
        # STEP 3: Assign roles deterministically
        roles = []
        for entity_name, entity_type in entities:
            role_assignment = self._assign_role_deterministic(
                entity_name, entity_type, text, llm_signals
            )
            if role_assignment:
                roles.append(role_assignment)
                reasoning_steps.append(f"Assigned role '{role_assignment.role_type}' to '{entity_name}' (confidence: {role_assignment.confidence:.2f})")
        
        # STEP 4: Extract contact info deterministically
        for role in roles:
            contact_info = self._extract_contact_info(role.name, text)
            role.contact_info = contact_info
        
        # STEP 4.5: Clean duplicate contacts between different roles
        roles = self._clean_duplicate_contacts(roles)
        
        # STEP 5: Identify primary holder deterministically
        primary_holder = self._identify_primary_holder(roles, text, document_family, document_subtype)
        if primary_holder:
            reasoning_steps.append(f"Identified primary holder: {primary_holder['name']} ({primary_holder['role_type']})")
        
        # STEP 6: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(roles, primary_holder)
        
        # STEP 7: Ensure we always return structured JSON (even if empty)
        # Force primary_holder to be included if we have roles but no holder identified
        if not primary_holder and roles:
            # Use highest confidence role as primary holder
            best_role = max(roles, key=lambda r: r.confidence)
            if best_role.confidence >= 0.70:  # Lower threshold for fallback
                primary_holder = {
                    "role_type": best_role.role_type,
                    "entity_type": best_role.entity_type.value,
                    "name": best_role.name,
                    "confidence": best_role.confidence,
                    "justification": f"{best_role.role_type.capitalize()} identified as primary holder based on highest confidence role assignment"
                }
        
        reasoning = " | ".join(reasoning_steps)
        
        return RolesExtractionResult(
            roles=roles,
            primary_holder=primary_holder,
            confidence=overall_confidence,
            reasoning=reasoning
        )
    
    def _get_llm_signals(
        self,
        text: str,
        document_family: Optional[str],
        document_subtype: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get LLM suggestions (signals only, not final decision)
        
        LLM only suggests candidates - Python makes final decision
        """
        # Truncate for LLM (keep first 6000 chars)
        text_for_llm = text[:6000] if len(text) > 6000 else text
        
        prompt = f"""Analyze this legal document and suggest candidate entities and their possible roles.

Document family: {document_family or "unknown"}
Document subtype: {document_subtype or "unknown"}

Document text:
---
{text_for_llm}
---

TASK: Identify all entities (persons and organizations) and map them to legal/contractual roles in the agreement.

ROLES TO IDENTIFY (explicit list):
- client / principal
- contractor / service provider
- signatory
- beneficiary
- data controller / data subject (for GDPR documents)
- payee / payer
- approver
- HR manager
- employer / employee
- licensor / licensee

IMPORTANT: 
- You are only suggesting candidates. The final decision will be made by deterministic Python rules.
- Analyze ALL pages of the document, including tables and bolded headers.
- Do not skip any text. Treat all text as potential source for entity extraction.
- Look for patterns like "ROLE: NAME" in tables and structured sections.

For each candidate entity, suggest:
- name: Entity name as mentioned (exact as in document)
- possible_roles: List of possible roles from the list above
- entity_type_hint: "individual" or "organization"
- context: Brief context where entity appears (include text snippet that proves the role)

Return ONLY valid JSON:
{{
  "candidates": [
    {{
      "name": "Emirates Petroleum Industries LLC",
      "possible_roles": ["client", "principal"],
      "entity_type_hint": "organization",
      "context": "Identified as 'CLIENT (Principal)' in 'Parties to this Agreement' section"
    }}
  ]
}}

Return ONLY the JSON, no other text."""
        
        try:
            if self.llm_provider == "openai":
                response = self._call_openai(prompt)
            elif self.llm_provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                return None
            
            # Extract JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None
            
            return json.loads(json_str)
        except Exception as e:
            return None
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal document analysis assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = self.llm_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _extract_entities_deterministic(
        self,
        text: str,
        llm_signals: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, EntityType]]:
        """
        Extract entities deterministically
        
        Uses:
        1. LLM suggestions as hints
        2. Pattern matching for party definitions
        3. Named entity recognition patterns
        """
        entities = []
        seen_names = set()
        
        # Use LLM suggestions as hints (but validate them)
        if llm_signals and "candidates" in llm_signals:
            for candidate in llm_signals["candidates"]:
                name = candidate.get("name", "").strip()
                if name and len(name) > 2:
                    # CRITICAL: Validate LLM suggestions too
                    if not self._is_valid_entity_name(name):
                        continue
                    entity_type_hint = candidate.get("entity_type_hint", "organization")
                    entity_type = EntityType.INDIVIDUAL if entity_type_hint == "individual" else EntityType.ORGANIZATION
                    if name not in seen_names:
                        entities.append((name, entity_type))
                        seen_names.add(name)
        
        # STEP 1: Extract from explicit "ROLE: NAME" patterns (highest priority)
        # CRITICAL: Extract each entity separately by finding role labels and their associated names
        # Use simpler, more direct patterns that match exactly what we see in the text
        
        # Pattern 1: "Company Name: Entity Name" (for CLIENT/PRINCIPAL - organizations)
        company_name_pattern = r'Company\s+Name:\s*([A-Z][A-Za-z\s&.,-]{2,50}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?=\s*\n|Address|Email|Phone|Registration|Passport|Visa|CONTRACTOR|CLIENT|Full\s+Name|$)'
        company_matches = re.finditer(company_name_pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in company_matches:
            entity_name = match.group(1).strip()
            # Clean up
            entity_name = re.sub(r'<[^>]+>', '', entity_name)
            entity_name = re.sub(r'\s+', ' ', entity_name)
            # Stop at organization suffix
            if any(suffix in entity_name.upper() for suffix in ['LLC', 'INC', 'LTD', 'CORP', 'COMPANY', 'DMCC', 'LIMITED']):
                org_match = re.search(r'^(.+?(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited))', entity_name, re.IGNORECASE)
                if org_match:
                    entity_name = org_match.group(1).strip()
            
            if self._is_valid_entity_name(entity_name) and 3 < len(entity_name) < 100 and entity_name not in seen_names:
                entities.append((entity_name, EntityType.ORGANIZATION))
                seen_names.add(entity_name)
        
        # Pattern 2: "Full Name: Entity Name" (for CONTRACTOR/SERVICE PROVIDER - individuals)
        full_name_pattern = r'Full\s+Name:\s*([A-Z][A-Za-z\s&.,-]{2,50}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?=\s*\n|Address|Email|Phone|Registration|Passport|Visa|CONTRACTOR|CLIENT|Company\s+Name|$)'
        full_name_matches = re.finditer(full_name_pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in full_name_matches:
            entity_name = match.group(1).strip()
            # Clean up
            entity_name = re.sub(r'<[^>]+>', '', entity_name)
            entity_name = re.sub(r'\s+', ' ', entity_name)
            # For "Full Name:", extract person name (stop at org suffix if present)
            if any(suffix in entity_name.upper() for suffix in ['LLC', 'INC', 'LTD', 'CORP', 'COMPANY', 'DMCC', 'LIMITED']):
                # Split: take only the person name part (before org suffix)
                words = entity_name.split()
                person_words = []
                for word in words:
                    if any(suffix in word.upper() for suffix in ['LLC', 'INC', 'LTD', 'CORP', 'COMPANY', 'DMCC', 'LIMITED']):
                        break
                    person_words.append(word)
                if person_words:
                    entity_name = ' '.join(person_words)
            
            if self._is_valid_entity_name(entity_name) and 3 < len(entity_name) < 100 and entity_name not in seen_names:
                # Determine if individual or organization
                entity_type = EntityType.INDIVIDUAL
                if any(word in entity_name.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                    entity_type = EntityType.ORGANIZATION
                elif len(entity_name.split()) > 4:
                    entity_type = EntityType.ORGANIZATION
                entities.append((entity_name, entity_type))
                seen_names.add(entity_name)
        
        # Pattern 3: Look for role labels with HTML tags followed by entity names
        # "CLIENT (Principal):</b> Company Name: Entity" or "CONTRACTOR:</b> Full Name: Entity"
        role_label_patterns = [
            r'(?:<[^>]+>)?(?:CLIENT|PRINCIPAL)[:\s]*(?:\([^)]+\))?(?:</[^>]+>)?[^\n]*Company\s+Name:\s*([A-Z][A-Za-z\s&.,-]{2,50}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?=\s*\n|Address|Email|Phone|Registration|Passport|Visa|CONTRACTOR|CLIENT|Full\s+Name|$)',
            r'(?:<[^>]+>)?(?:CONTRACTOR|SERVICE\s+PROVIDER)[:\s]*(?:\([^)]+\))?(?:</[^>]+>)?[^\n]*Full\s+Name:\s*([A-Z][A-Za-z\s&.,-]{2,50}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?=\s*\n|Address|Email|Phone|Registration|Passport|Visa|CONTRACTOR|CLIENT|Company\s+Name|$)',
        ]
        
        for pattern in role_label_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                entity_name = match.group(1).strip()
                # Clean up
                entity_name = re.sub(r'<[^>]+>', '', entity_name)
                entity_name = re.sub(r'\s+', ' ', entity_name)
                
                # Determine entity type based on pattern
                if r'Company\s+Name' in pattern:
                    # Organization
                    if any(suffix in entity_name.upper() for suffix in ['LLC', 'INC', 'LTD', 'CORP', 'COMPANY', 'DMCC', 'LIMITED']):
                        org_match = re.search(r'^(.+?(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited))', entity_name, re.IGNORECASE)
                        if org_match:
                            entity_name = org_match.group(1).strip()
                    entity_type = EntityType.ORGANIZATION
                else:
                    # Individual
                    if any(suffix in entity_name.upper() for suffix in ['LLC', 'INC', 'LTD', 'CORP', 'COMPANY', 'DMCC', 'LIMITED']):
                        words = entity_name.split()
                        person_words = []
                        for word in words:
                            if any(suffix in word.upper() for suffix in ['LLC', 'INC', 'LTD', 'CORP', 'COMPANY', 'DMCC', 'LIMITED']):
                                break
                            person_words.append(word)
                        if person_words:
                            entity_name = ' '.join(person_words)
                    entity_type = EntityType.INDIVIDUAL
                
                if self._is_valid_entity_name(entity_name) and 3 < len(entity_name) < 100 and entity_name not in seen_names:
                    entities.append((entity_name, entity_type))
                    seen_names.add(entity_name)
        
        # Fallback: Extract using original patterns if nothing found
        if not entities:
            explicit_role_patterns = [
                r'(?:<[^>]+>)?(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER|PARTY\s+[AB])[:\s]*(?:\([^)]+\))?(?:</[^>]+>)?[^\n]*(?:Company\s+Name:|Full\s+Name:)\s*([A-Z][A-Za-z\s&.,-]{2,50}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?=\s*\n|Address|Email|Phone|Registration|Passport|Visa|$)',
                r'(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER|PARTY\s+[AB])[:\s]*(?:\([^)]+\))?\s*\n[^\n]*(?:Company\s+Name:|Full\s+Name:)\s*([A-Z][A-Za-z\s&.,-]{2,50}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?=\s*\n|Address|Email|Phone|Registration|Passport|Visa|$)',
            ]
            
            for pattern in explicit_role_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    fallback_entity_name = match.group(1).strip()
                    # Clean up - remove HTML tags if any
                    fallback_entity_name = re.sub(r'<[^>]+>', '', fallback_entity_name)
                    fallback_entity_name = re.sub(r'\s+', ' ', fallback_entity_name)
                    
                    # Split if contains multiple potential entities
                    potential_entities = []
                    org_suffix_pattern = r'^(.+?\s+(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?:\s|$)'
                    org_match = re.search(org_suffix_pattern, fallback_entity_name)
                    if org_match:
                        potential_entities = [
                            org_match.group(1).strip(),
                            org_match.group(2).strip()
                        ]
                    else:
                        potential_entities = [fallback_entity_name]
                    
                    # Process each potential entity
                    for ent in potential_entities:
                        ent = ent.strip()
                        if self._is_valid_entity_name(ent) and 3 < len(ent) < 100 and ent not in seen_names:
                            entity_type = EntityType.ORGANIZATION
                            if not any(word in ent.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                                if len(ent.split()) <= 4:
                                    entity_type = EntityType.INDIVIDUAL
                            entities.append((ent, entity_type))
                            seen_names.add(ent)
        
        # STEP 2: Extract from "Parties to this Agreement" section
        parties_section = self._extract_parties_section(text)
        if parties_section:
            # Look for entity names in parties section (lines after role labels)
            lines = parties_section.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Detect role labels - if found, next non-label line is entity name
                # Handle HTML tags like <b>CLIENT (Principal):</b>
                role_label_pattern = r'^(?:<[^>]+>)?(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER|PARTY\s+[AB])[:\s]*(?:\(.*?\))?(?:</[^>]+>)?\s*$'
                if re.match(role_label_pattern, line, re.IGNORECASE):
                    # Look for entity name on next lines (skip Address, Email, Phone labels)
                    for j in range(i + 1, min(i + 5, len(lines))):  # Check next 4 lines
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        # Remove HTML tags
                        next_line = re.sub(r'<[^>]+>', '', next_line)
                        # Skip labels like "Company Name:", "Full Name:"
                        if re.match(r'^(?:Company\s+Name|Full\s+Name|Registration\s+Number|Passport\s+Number)[:\s]*', next_line, re.IGNORECASE):
                            # Extract entity name after label
                            name_match = re.search(r':\s*([A-Z][A-Za-z\s&.,-]{3,60}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)', next_line)
                            if name_match:
                                next_line = name_match.group(1).strip()
                            else:
                                continue
                        # Skip labels
                        if any(word in next_line.upper() for word in ['ADDRESS', 'EMAIL', 'PHONE', 'PARTY', 'PARTIES', 'CLIENT', 'CONTRACTOR', 'PRINCIPAL', 'REGISTRATION', 'PASSPORT']):
                            continue
                        # Check if it looks like an entity name (starts with capital, reasonable length)
                        if re.match(r'^[A-Z][A-Za-z\s&.,-]{3,80}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?$', next_line):
                            # STRICT VALIDATION: Must be a real name
                            if self._is_valid_entity_name(next_line) and next_line not in seen_names:
                                entity_type = EntityType.ORGANIZATION
                                if not any(word in next_line.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                                    if len(next_line.split()) <= 4:
                                        entity_type = EntityType.INDIVIDUAL
                                entities.append((next_line, entity_type))
                                seen_names.add(next_line)
                                break  # Found entity, move to next role label
                    continue
                
                # Skip labels
                if any(word in line.upper() for word in ['ADDRESS', 'EMAIL', 'PHONE', 'PARTY', 'PARTIES']):
                    continue
        
        # Fallback: Extract from explicit party definitions in full text
        if not entities:
            party_patterns = [
                # Pattern: "CLIENT (Principal):</b> Company Name: Entity Name" (with HTML tags)
                r'(?:<[^>]+>)?(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER)[:\s]*(?:\([^)]+\))?(?:</[^>]+>)?[:\s]*(?:Company\s+Name:|Full\s+Name:)?\s*([A-Z][A-Za-z\s&.,-]{3,60}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?:\s*\n|Address|Email|Phone|Registration|Passport|$)',
                # Pattern: "CLIENT:\nEntity Name" (entity on next line, no HTML)
                r'(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER)[:\s]*\n\s*(?:Company\s+Name:|Full\s+Name:)?\s*([A-Z][A-Za-z\s&.,-]{3,60}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)(?:\s*\n|Address|Email|Phone|Registration|Passport|$)',
                # Pattern: "Entity Name (CLIENT)" - entity name before role label
                r'([A-Z][A-Za-z\s&.,-]{3,60}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)\s*\(["\']?(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER)["\']?\)',
            ]
            
            for pattern in party_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    name = match.group(1).strip()
                    # Clean up name
                    name = re.sub(r'["\']', '', name)
                    name = re.sub(r'\s+', ' ', name)
                    # STRICT VALIDATION: Must be a real name
                    if not self._is_valid_entity_name(name):
                        continue
                    # Must not be just a single common word
                    if name.upper() in ['CLIENT', 'CONTRACTOR', 'PRINCIPAL', 'PARTY', 'A', 'B', 'AGREEMENT', 'STATUS', 'DATE']:
                        continue
                    
                    if name and 3 < len(name) < 100 and name not in seen_names:
                        # CRITICAL: Validate before adding
                        if not self._is_valid_entity_name(name):
                            continue
                        # Determine entity type
                        entity_type = EntityType.ORGANIZATION
                        if not any(word in name.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                            if len(name.split()) <= 4:
                                entity_type = EntityType.INDIVIDUAL
                        entities.append((name, entity_type))
                        seen_names.add(name)
        
        # CRITICAL: Final validation pass - filter out any invalid entities
        validated_entities = []
        for entity_name, entity_type in entities:
            if self._is_valid_entity_name(entity_name):
                validated_entities.append((entity_name, entity_type))
        
        return validated_entities
    
    def _assign_role_deterministic(
        self,
        entity_name: str,
        entity_type: EntityType,
        text: str,
        llm_signals: Optional[Dict[str, Any]]
    ) -> Optional[RoleAssignment]:
        """
        Assign role deterministically using explicit rules
        
        DETERMINISTIC LOGIC:
        1. Check explicit role definitions in document (highest priority)
        2. Use context patterns around entity name
        3. Use LLM suggestions as hints (but Python decides)
        4. Calculate confidence based on evidence strength
        """
        # CRITICAL: Validate entity name first - reject if not a real name
        if not self._is_valid_entity_name(entity_name):
            return None
        
        # First, check if entity name appears near explicit role labels
        # Pattern: "CLIENT:\nEntity Name" or "Entity Name (CLIENT)"
        explicit_role_match = self._find_explicit_role_for_entity(entity_name, text)
        if explicit_role_match:
            role_type, confidence, evidence = explicit_role_match
            return RoleAssignment(
                role_type=role_type,
                entity_type=entity_type,
                name=entity_name,
                confidence=confidence,
                evidence=evidence
            )
        
        text_lower = text.lower()
        entity_lower = entity_name.lower()
        
        best_role = None
        best_confidence = 0.0
        best_evidence = ""
        
        # Check each role pattern
        for role_type, role_config in self.role_patterns.items():
            confidence = role_config["confidence_base"]
            evidence_parts = []
            
            # Check keywords in context around entity
            entity_context = self._get_entity_context(entity_name, text, window=200)
            context_lower = entity_context.lower()
            
            # Check if entity name appears right after role label
            role_label_patterns = [
                rf'(?:CLIENT|PRINCIPAL)[:\s]+\s*{re.escape(entity_name)}',
                rf'(?:CONTRACTOR|SERVICE\s+PROVIDER)[:\s]+\s*{re.escape(entity_name)}',
            ]
            
            explicit_match = False
            for pattern in role_label_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    explicit_match = True
                    confidence = 0.95
                    if 'CLIENT' in pattern or 'PRINCIPAL' in pattern:
                        best_role = "client"
                    elif 'CONTRACTOR' in pattern or 'SERVICE' in pattern:
                        best_role = "contractor"
                    best_evidence = f"Explicit role label found: entity name appears directly after role definition"
                    break
            
            if explicit_match:
                break
            
            keyword_matches = 0
            for keyword in role_config["keywords"]:
                if keyword in context_lower:
                    keyword_matches += 1
                    evidence_parts.append(f"Keyword '{keyword}' found near entity")
            
            # Check explicit role definitions
            pattern_matches = 0
            for pattern in role_config["context_patterns"]:
                matches = list(re.finditer(pattern, entity_context, re.IGNORECASE))
                if matches:
                    pattern_matches += len(matches)
                    for match in matches:
                        # Check if entity name is actually in the match
                        matched_text = match.group(0).lower()
                        if entity_lower in matched_text or any(word in matched_text for word in entity_name.split()[:2]):
                            # Extract exact text snippet as evidence (limit to 150 chars)
                            evidence_text = match.group(0).strip()
                            if len(evidence_text) > 150:
                                evidence_text = evidence_text[:147] + "..."
                            evidence_parts.append(f"Explicit role definition: '{evidence_text}'")
            
            # Calculate confidence based on evidence
            if keyword_matches > 0 or pattern_matches > 0:
                # Boost confidence for explicit patterns
                if pattern_matches > 0:
                    confidence = min(0.98, confidence + 0.10 * pattern_matches)
                if keyword_matches > 0:
                    confidence = min(0.95, confidence + 0.05 * keyword_matches)
                
                # Use LLM suggestions as hints (small boost if matches)
                if llm_signals and "candidates" in llm_signals:
                    for candidate in llm_signals["candidates"]:
                        candidate_name = candidate.get("name", "").lower()
                        # Check if entity name matches or is similar
                        if candidate_name == entity_lower or entity_lower in candidate_name or candidate_name in entity_lower:
                            possible_roles = candidate.get("possible_roles", [])
                            if role_type in possible_roles:
                                confidence = min(0.98, confidence + 0.03)
                                evidence_parts.append("LLM suggestion matches")
                
                # Only assign if confidence is reasonable
                if confidence > best_confidence and confidence >= 0.75:  # Higher threshold
                    best_role = role_type
                    best_confidence = confidence
                    best_evidence = " | ".join(evidence_parts[:3]) if evidence_parts else f"Identified as '{role_type}' based on document context"
        
        # If no role found, try to infer from document structure
        if not best_role:
            # Check if entity appears in "Parties" section
            parties_section = self._extract_parties_section(text)
            if parties_section:
                if entity_lower in parties_section.lower():
                    # Try to infer role from section structure
                    if "client" in parties_section.lower() and entity_lower in parties_section.lower():
                        best_role = "client"
                        best_confidence = 0.75
                        best_evidence = "Found in 'Parties' section with 'client' context"
                    elif "contractor" in parties_section.lower() and entity_lower in parties_section.lower():
                        best_role = "contractor"
                        best_confidence = 0.75
                        best_evidence = "Found in 'Parties' section with 'contractor' context"
        
        # Return role even if confidence is lower (as per user requirement: "even if confidence is low")
        # But still filter out very low confidence (< 0.60)
        if best_role and best_confidence >= 0.60:
            # Ensure evidence includes exact text snippet
            if not best_evidence or "Explicit role definition" not in best_evidence:
                # Try to find exact text snippet near entity
                entity_context = self._get_entity_context(entity_name, text, window=100)
                if entity_context:
                    # Extract a snippet that shows the role assignment
                    snippet = entity_context[:150].replace('\n', ' ')
                    if len(snippet) > 150:
                        snippet = snippet[:147] + "..."
                    if best_evidence:
                        best_evidence = f"{best_evidence} | Text snippet: '{snippet}'"
                    else:
                        best_evidence = f"Text snippet: '{snippet}'"
            
            return RoleAssignment(
                role_type=best_role,
                entity_type=entity_type,
                name=entity_name,
                confidence=best_confidence,
                evidence=best_evidence or f"Identified as '{best_role}' based on document context"
            )
        
        return None
    
    def _find_explicit_role_for_entity(self, entity_name: str, text: str) -> Optional[Tuple[str, float, str]]:
        """
        Find explicit role assignment for entity (e.g., "CLIENT: Entity Name")
        
        Returns: (role_type, confidence, evidence) or None
        """
        # Look for patterns like "CLIENT:\nEntity Name" or "Entity Name (CLIENT)"
        patterns = [
            (r'CLIENT[:\s]+\s*' + re.escape(entity_name), "client", 0.95),
            (r'PRINCIPAL[:\s]+\s*' + re.escape(entity_name), "client", 0.95),
            (r'CONTRACTOR[:\s]+\s*' + re.escape(entity_name), "contractor", 0.95),
            (r'SERVICE\s+PROVIDER[:\s]+\s*' + re.escape(entity_name), "contractor", 0.95),
            (re.escape(entity_name) + r'\s*\(["\']?CLIENT["\']?\)', "client", 0.90),
            (re.escape(entity_name) + r'\s*\(["\']?CONTRACTOR["\']?\)', "contractor", 0.90),
        ]
        
        for pattern, role_type, confidence in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return (role_type, confidence, f"Explicit role definition found: '{match.group(0)[:100]}'")
        
        return None
    
    def _get_entity_context(self, entity: str, text: str, window: int = 300) -> str:
        """Get context around entity mentions"""
        pattern = re.escape(entity)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Use first match
            match = matches[0]
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end]
        return ""
    
    def _extract_parties_section(self, text: str) -> Optional[str]:
        """Extract 'Parties' section from document"""
        parties_patterns = [
            r'parties?\s+to\s+this\s+(?:agreement|contract)[:\s]*\n(.+?)(?=\nWHEREAS|\nNOW|\n\d+\.|$)',
            r'parties[:\s]*\n(.+?)(?=\nWHEREAS|\nNOW|\n\d+\.|$)',
        ]
        
        for pattern in parties_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if match:
                section = match.group(1).strip()
                # Limit section length to avoid capturing too much
                if 10 < len(section) < 2000:
                    return section
        return None
    
    def _extract_contact_info(self, entity_name: str, text: str) -> ContactInfo:
        """
        Extract contact information deterministically
        
        Looks for email and phone near entity name, including in signature sections
        """
        contact_info = ContactInfo()
        
        # Get context around entity (larger window to catch contact info)
        entity_context = self._get_entity_context(entity_name, text, window=800)
        
        # Also search in signature sections
        signature_sections = self._extract_signature_sections(text)
        for section in signature_sections:
            if entity_name.lower() in section.lower():
                entity_context += "\n" + section
        
        # Extract email - look for patterns near entity name
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = list(re.finditer(email_pattern, entity_context))
        
        # Prefer email that appears in same line or close to entity name
        entity_lower = entity_name.lower()
        for match in email_matches:
            email = match.group(0)
            email_lower = email.lower()
            
            # Check if email domain or username matches entity name
            email_username = email_lower.split('@')[0] if '@' in email_lower else ''
            email_domain = email_lower.split('@')[1].split('.')[0] if '@' in email_lower else ''
            
            # Prefer email that appears close to entity name or matches entity
            if (entity_lower[:15] in email_lower or 
                email_username in entity_lower or 
                email_domain in entity_lower):
                contact_info.email = email
                break
            elif not contact_info.email:  # Use first email found as fallback
                contact_info.email = email
        
        # Extract phone - look for patterns near entity name
        phone_patterns = [
            r'Phone[:\s]+([+\d\s\-\(\)]{10,25})',  # "Phone: +1 234 567 8900"
            r'Tel[:\s]+([+\d\s\-\(\)]{10,25})',  # "Tel: +1 234 567 8900"
            r'Telephone[:\s]+([+\d\s\-\(\)]{10,25})',  # "Telephone: +1 234 567 8900"
            r'Mobile[:\s]+([+\d\s\-\(\)]{10,25})',  # "Mobile: +1 234 567 8900"
            r'\b(\+\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4})',  # International format
            r'\b(\+\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4})',  # International no parentheses
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',  # US format (123) 456-7890
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Standard 123-456-7890
        ]
        
        for pattern in phone_patterns:
            phone_matches = list(re.finditer(pattern, entity_context, re.IGNORECASE))
            for match in phone_matches:
                phone = match.group(1) if len(match.groups()) > 0 else match.group(0)
                phone = phone.strip()
                # Clean up phone - keep + and digits
                phone_clean = re.sub(r'[^\d+]', '', phone)
                if len(phone_clean) >= 10:  # Valid phone length
                    # Prefer phone that appears near "Phone:" or "Tel:" label
                    match_context = entity_context[max(0, match.start()-50):match.end()+50].lower()
                    if 'phone' in match_context or 'tel' in match_context:
                        contact_info.phone = phone_clean
                        break
                    elif not contact_info.phone:  # Use first valid phone as fallback
                        contact_info.phone = phone_clean
            if contact_info.phone:
                break
        
        return contact_info
    
    def _clean_duplicate_contacts(self, roles: List[RoleAssignment]) -> List[RoleAssignment]:
        """
        Remove duplicate contacts between different roles
        
        If multiple roles have the same email/phone, keep it only for the first role
        (usually the primary holder or client with highest confidence)
        """
        seen_contacts = set()
        cleaned_roles = []
        
        # Sort roles by confidence (highest first) to keep contacts for most important roles
        sorted_roles = sorted(roles, key=lambda r: r.confidence, reverse=True)
        
        for role in sorted_roles:
            email = role.contact_info.email
            phone = role.contact_info.phone
            
            # Create contact tuple for deduplication
            contact_tuple = (email, phone)
            
            # If this contact was already seen, remove it from this role
            if contact_tuple in seen_contacts and (email or phone):
                # Keep the role but clear duplicate contacts
                role.contact_info.email = None
                role.contact_info.phone = None
            elif email or phone:
                # Mark this contact as seen
                seen_contacts.add(contact_tuple)
            
            cleaned_roles.append(role)
        
        return cleaned_roles
    
    def _extract_signature_sections(self, text: str) -> List[str]:
        """Extract signature sections from document"""
        signature_patterns = [
            r'(?:FOR\s+THE\s+)?(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER)[:\s]*\n.*?(?=\n\n|\n[A-Z]{3,}|$)',
            r'(?:SIGNED|SIGNATURE)[:\s]*\n.*?(?=\n\n|\n[A-Z]{3,}|$)',
            r'________________+\s*\n.*?(?=\n\n|\n[A-Z]{3,}|$)',
        ]
        
        sections = []
        for pattern in signature_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                section = match.group(0).strip()
                if 20 < len(section) < 500:  # Reasonable length
                    sections.append(section)
        
        return sections
    
    def _identify_primary_holder(
        self,
        roles: List[RoleAssignment],
        text: str,
        document_family: Optional[str],
        document_subtype: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Identify primary holder deterministically
        
        DETERMINISTIC RULES:
        1. In service agreements: contractor is usually holder
        2. In employment contracts: employee is usually holder
        3. Entity with most obligations mentioned is holder
        4. Entity with highest confidence role assignment
        """
        if not roles:
            return None
        
        # Rule 1: Document type-based inference
        if document_subtype:
            subtype_lower = document_subtype.lower()
            if "contractor" in subtype_lower or "service" in subtype_lower:
                # Look for contractor role
                for role in roles:
                    if role.role_type == "contractor":
                        return {
                            "role_type": role.role_type,
                            "entity_type": role.entity_type.value,
                            "name": role.name,
                            "confidence": min(0.95, role.confidence + 0.05),
                            "justification": "Contractor is holder of obligations and right to compensation in service agreements"
                        }
        
        # Rule 2: Entity with highest confidence
        best_role = max(roles, key=lambda r: r.confidence)
        if best_role.confidence >= 0.75:
            return {
                "role_type": best_role.role_type,
                "entity_type": best_role.entity_type.value,
                "name": best_role.name,
                "confidence": best_role.confidence,
                "justification": f"{best_role.role_type.capitalize()} is identified as primary holder based on highest confidence role assignment"
            }
        
        return None
    
    def _calculate_overall_confidence(
        self,
        roles: List[RoleAssignment],
        primary_holder: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence deterministically"""
        if not roles:
            return 0.0
        
        # Average confidence of high-confidence roles
        high_conf_roles = [r for r in roles if r.confidence >= 0.75]
        if not high_conf_roles:
            return 0.0
        
        avg_confidence = sum(r.confidence for r in high_conf_roles) / len(high_conf_roles)
        
        # Boost if primary holder is clear
        if primary_holder and primary_holder.get("confidence", 0) >= 0.75:
            avg_confidence = min(0.98, avg_confidence + 0.05)
        
        return avg_confidence
    
    def _is_valid_entity_name(self, name: str) -> bool:
        """
        Validate that a string is a real entity name, not a sentence or phrase
        
        Rules:
        - Must not contain verbs (agrees, shall, will, must, may, etc.)
        - Must not contain common contract words (hereby, assigns, rights, etc.)
        - Must not start with lowercase (unless it's a known pattern)
        - Must not contain punctuation that suggests it's a sentence (multiple commas, periods)
        - Must be reasonable length (3-100 chars)
        - Must not be a sentence fragment
        """
        if not name or len(name.strip()) < 3:
            return False
        
        name = name.strip()
        name_lower = name.lower()
        
        # Reject if contains verbs (strong indicator of sentence)
        verbs = ['agrees', 'shall', 'will', 'must', 'may', 'can', 'should', 'hereby', 'assigns', 
                 'terminate', 'maintain', 'comply', 'provide', 'receive', 'deliver', 'perform',
                 'maintains', 'complies', 'provides', 'receives', 'delivers', 'performs']
        if any(verb in name_lower for verb in verbs):
            return False
        
        # Reject if contains common contract phrases (strong indicator of sentence)
        contract_phrases = ['all rights', 'title and interest', 'such materials', 'to the', 'including',
                           'applicable to', 'for cause', 'breach of', 'valid certifications',
                           'rights, title', 'interest in', 'materials to', 'the client', 'the contractor']
        if any(phrase in name_lower for phrase in contract_phrases):
            return False
        
        # Reject if contains too many punctuation marks (suggests it's a sentence)
        if name.count(',') > 1 or (name.count('.') > 0 and not name.endswith('.')):
            return False
        
        # Reject if starts with lowercase (unless it's a known pattern like "a.almansouri" or email)
        if len(name) > 0 and name[0].islower() and '@' not in name and '.' not in name[:5]:
            return False
        
        # Reject single words that are common labels (STATUS, DATE, etc.)
        if len(name.split()) == 1 and name.upper() in ['STATUS', 'DATE', 'CLIENT', 'CONTRACTOR', 'PARTY', 'A', 'B']:
            return False
        
        # Reject if it looks like a sentence fragment (starts with lowercase after first word)
        words = name.split()
        if len(words) > 1:
            # Check if second word starts with lowercase (likely a sentence)
            if words[1][0].islower():
                return False
        
        # Must have at least one letter
        if not any(c.isalpha() for c in name):
            return False
        
        # Reject if it's clearly a sentence (ends with period and has multiple words suggesting a sentence)
        if name.endswith('.') and len(words) > 3:
            return False
        
        return True
    
    def _extract_entities_from_tables(self, tables: List[Dict[str, Any]]) -> List[Tuple[str, EntityType]]:
        """
        Extract entity names directly from structured table data
        
        Looks for patterns like:
        - CLIENT: Entity Name
        - CONTRACTOR: Entity Name
        - Tables with role labels in headers and names in rows
        """
        entities = []
        seen_names = set()
        
        for table in tables:
            table_data = table.get('data', [])
            if not table_data:
                continue
            
            # Check if table has role labels (CLIENT, CONTRACTOR, etc.)
            # Look for role labels in first column or header
            for row in table_data:
                if isinstance(row, dict):
                    # Check each cell for role label + entity name pattern
                    for key, value in row.items():
                        if not value:
                            continue
                        value_str = str(value).strip()
                        
                        # Pattern: "CLIENT: Entity Name" or "CLIENT (Principal): Entity Name"
                        role_match = re.match(
                            r'^(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER)[:\s]*(?:\([^)]+\))?\s*(.+)$',
                            value_str,
                            re.IGNORECASE
                        )
                        if role_match:
                            entity_name = role_match.group(1).strip()
                            if self._is_valid_entity_name(entity_name) and entity_name not in seen_names:
                                entity_type = EntityType.ORGANIZATION
                                if not any(word in entity_name.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                                    if len(entity_name.split()) <= 4:
                                        entity_type = EntityType.INDIVIDUAL
                                entities.append((entity_name, entity_type))
                                seen_names.add(entity_name)
                        
                        # Pattern: Entity name in cell, check adjacent cells for role labels
                        if self._is_valid_entity_name(value_str) and len(value_str) > 5:
                            # Check if this looks like an entity name
                            if re.match(r'^[A-Z][A-Za-z\s&.,-]{5,60}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?$', value_str):
                                # Check other cells in same row for role indicators
                                row_values = [str(v).upper() for v in row.values() if v]
                                if any(role in ' '.join(row_values) for role in ['CLIENT', 'CONTRACTOR', 'PRINCIPAL', 'SERVICE PROVIDER']):
                                    if value_str not in seen_names:
                                        entity_type = EntityType.ORGANIZATION
                                        if not any(word in value_str.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                                            if len(value_str.split()) <= 4:
                                                entity_type = EntityType.INDIVIDUAL
                                        entities.append((value_str, entity_type))
                                        seen_names.add(value_str)
                
                elif isinstance(row, list):
                    # Check if row contains role label + entity name
                    row_text = ' '.join(str(v) for v in row if v)
                    # Pattern: "CLIENT: Entity Name"
                    role_match = re.search(
                        r'(?:CLIENT|CONTRACTOR|PRINCIPAL|SERVICE\s+PROVIDER)[:\s]*(?:\([^)]+\))?\s*([A-Z][A-Za-z\s&.,-]{5,60}(?:LLC|Inc|Ltd|Corp|Company|DMCC|Limited)?)',
                        row_text,
                        re.IGNORECASE
                    )
                    if role_match:
                        entity_name = role_match.group(1).strip()
                        if self._is_valid_entity_name(entity_name) and entity_name not in seen_names:
                            entity_type = EntityType.ORGANIZATION
                            if not any(word in entity_name.upper() for word in ["LLC", "INC", "LTD", "CORP", "COMPANY", "DMCC", "LIMITED"]):
                                if len(entity_name.split()) <= 4:
                                    entity_type = EntityType.INDIVIDUAL
                            entities.append((entity_name, entity_type))
                            seen_names.add(entity_name)
        
        return entities
    
    def _extract_text_from_tables(self, tables: List[Dict[str, Any]]) -> str:
        """
        Extract text from tables for entity extraction
        
        Converts table data to text format that can be parsed for entities
        """
        table_texts = []
        for table in tables:
            table_data = table.get('data', [])
            if table_data:
                # Convert table rows to text
                for row in table_data:
                    if isinstance(row, dict):
                        # Join all values in row
                        row_text = ' '.join(str(v) for v in row.values() if v)
                        if row_text.strip():
                            table_texts.append(row_text)
                    elif isinstance(row, list):
                        row_text = ' '.join(str(v) for v in row if v)
                        if row_text.strip():
                            table_texts.append(row_text)
        return '\n'.join(table_texts)
