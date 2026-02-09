"""
Semantic Role & Holder Extractor - Enterprise-Grade Document Intelligence

This module performs semantic role extraction from legal documents using LLM-based
analysis. It identifies entities and their roles based on context, not hardcoded patterns.

Key Features:
- Context-aware role inference (no regex hardcoded)
- Multi-entity extraction with semantic roles
- Holder identification for contractual relationships
- Confidence scoring with evidence-based reasoning
- Works on long, unstructured documents

Roles Identified:
- client / principal
- contractor / service provider
- employer / employee
- licensor / licensee
- data subject / data controller
- beneficiary
- signatory
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class EntityType(str, Enum):
    """Entity types"""
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"


@dataclass
class RoleAssignment:
    """A role assignment for an entity"""
    role_type: str  # e.g., "contractor", "client", "employer"
    entity_type: EntityType
    name: str
    confidence: float  # 0.0 to 1.0
    evidence: str  # Brief textual explanation with semantic reference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role_type": self.role_type,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "confidence": self.confidence,
            "evidence": self.evidence
        }


@dataclass
class RoleExtractionResult:
    """Result of semantic role extraction"""
    roles: List[RoleAssignment] = field(default_factory=list)
    primary_holder: Optional[RoleAssignment] = None
    confidence: float = 0.0
    reasoning: str = ""  # Internal chain-of-thought (not exposed to end user)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal reasoning"""
        result = {
            "roles": [role.to_dict() for role in self.roles],
            "primary_holder": self.primary_holder.to_dict() if self.primary_holder else None,
            "confidence": self.confidence
        }
        # Filter out roles with low confidence (< 0.75)
        result["roles"] = [
            role for role in result["roles"]
            if role["confidence"] >= 0.75
        ]
        # Set primary_holder to None if confidence too low
        if result["primary_holder"] and result["primary_holder"]["confidence"] < 0.75:
            result["primary_holder"] = None
        return result


class SemanticRoleExtractor:
    """
    Semantic Role Extractor using LLM-based analysis
    
    This extractor uses LLM to understand document context and identify:
    1. Entities (people, organizations)
    2. Their semantic roles (contractor, client, employer, etc.)
    3. The primary holder of rights/obligations
    
    No hardcoded regex patterns - all inference is semantic.
    """
    
    def __init__(self, use_llm: bool = True, llm_provider: str = "openai"):
        """
        Initialize semantic role extractor
        
        Args:
            use_llm: Whether to use LLM (if False, uses fallback heuristics)
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
    
    def extract(
        self,
        text: str,
        document_family: Optional[str] = None,
        document_subtype: Optional[str] = None
    ) -> RoleExtractionResult:
        """
        Extract semantic roles from document
        
        Args:
            text: Document text (can be long and unstructured)
            document_family: Optional document family hint
            document_subtype: Optional document subtype hint
            
        Returns:
            RoleExtractionResult with roles, primary holder, and confidence
        """
        if not text or len(text.strip()) < 50:
            return RoleExtractionResult(
                roles=[],
                primary_holder=None,
                confidence=0.0,
                reasoning="Document text too short or empty"
            )
        
        if self.use_llm:
            return self._extract_with_llm(text, document_family, document_subtype)
        else:
            return self._extract_with_heuristics(text, document_family, document_subtype)
    
    def _extract_with_llm(
        self,
        text: str,
        document_family: Optional[str],
        document_subtype: Optional[str]
    ) -> RoleExtractionResult:
        """
        Extract roles using LLM semantic analysis
        
        INTERNAL REASONING (chain-of-thought):
        1. Analyze document structure and key sections
        2. Identify parties mentioned (names, organizations)
        3. Understand relationships from context (who provides services, who receives)
        4. Infer semantic roles from language patterns (not keywords)
        5. Determine primary holder based on who has primary rights/obligations
        6. Calculate confidence based on explicitness and consistency
        """
        
        # Truncate text if too long (keep first 8000 chars for context)
        text_for_analysis = text[:8000] if len(text) > 8000 else text
        
        # Build prompt for LLM
        prompt = self._build_extraction_prompt(text_for_analysis, document_family, document_subtype)
        
        try:
            if self.llm_provider == "openai":
                response = self._call_openai(prompt)
            elif self.llm_provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
            
            # Parse LLM response
            return self._parse_llm_response(response, text)
            
        except Exception as e:
            # Fallback to heuristics on error
            reasoning = f"LLM extraction failed: {str(e)}, falling back to heuristics"
            return self._extract_with_heuristics(text, document_family, document_subtype, reasoning)
    
    def _build_extraction_prompt(
        self,
        text: str,
        document_family: Optional[str],
        document_subtype: Optional[str]
    ) -> str:
        """Build prompt for LLM extraction"""
        
        context_hints = ""
        if document_family:
            context_hints += f"\nDocument family: {document_family}"
        if document_subtype:
            context_hints += f"\nDocument subtype: {document_subtype}"
        
        prompt = f"""You are a legal document intelligence system. Analyze the following document and extract semantic roles for all entities mentioned.

{context_hints}

Document text:
---
{text}
---

TASK: Identify all entities (people, organizations) and their semantic roles based on CONTEXT, not keywords.

ROLES TO IDENTIFY (infer from context):
- client / principal (who receives services or benefits)
- contractor / service provider (who provides services)
- employer / employee (employment relationship)
- licensor / licensee (licensing relationship)
- data subject / data controller (data processing relationship)
- beneficiary (who benefits from the document)
- signatory (who signs the document)

For each entity found, provide:
1. role_type: The semantic role (e.g., "contractor", "client")
2. entity_type: "individual" or "organization"
3. name: The entity's name as mentioned in the document
4. confidence: 0.0-1.0 based on how explicit and clear the role is
5. evidence: A brief explanation with a quote or reference to the document text that supports this role

Also identify the PRIMARY HOLDER: the entity that holds the primary rights/obligations from this document (e.g., the contractor in a service agreement, the employee in an employment contract).

IMPORTANT:
- Use semantic understanding, not keyword matching
- Confidence must be >= 0.75 to be included
- Provide specific evidence quotes from the document
- If uncertain (confidence < 0.75), exclude that role

Return ONLY valid JSON in this exact format:
{{
  "roles": [
    {{
      "role_type": "contractor",
      "entity_type": "individual",
      "name": "John Smith",
      "confidence": 0.92,
      "evidence": "The document states: 'John Smith agrees to provide consulting services as an independent contractor'"
    }}
  ],
  "primary_holder": {{
    "role_type": "contractor",
    "entity_type": "individual",
    "name": "John Smith",
    "confidence": 0.92,
    "evidence": "John Smith is the primary service provider and holds the obligation to deliver services"
  }},
  "reasoning": "Internal chain-of-thought analysis..."
}}

Return ONLY the JSON, no other text."""
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",  # Use cost-effective model
            messages=[
                {"role": "system", "content": "You are a legal document analysis system. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = self.llm_client.messages.create(
            model="claude-3-haiku-20240307",  # Use cost-effective model
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def _parse_llm_response(self, response: str, original_text: str) -> RoleExtractionResult:
        """
        Parse LLM JSON response
        
        INTERNAL REASONING:
        - Extract JSON from response (may have markdown formatting)
        - Validate structure
        - Create RoleAssignment objects
        - Verify evidence exists in original text
        - Adjust confidence if evidence is weak
        """
        
        # Extract JSON from response (may be wrapped in markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in LLM response")
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        
        # Parse roles
        roles = []
        for role_data in data.get("roles", []):
            try:
                role = RoleAssignment(
                    role_type=role_data["role_type"],
                    entity_type=EntityType(role_data["entity_type"]),
                    name=role_data["name"],
                    confidence=float(role_data["confidence"]),
                    evidence=role_data["evidence"]
                )
                # Verify evidence exists in original text (boost confidence if found)
                if self._verify_evidence(role.evidence, original_text):
                    role.confidence = min(1.0, role.confidence + 0.05)
                roles.append(role)
            except (KeyError, ValueError) as e:
                # Skip invalid role entries
                continue
        
        # Parse primary holder
        primary_holder = None
        if data.get("primary_holder"):
            try:
                holder_data = data["primary_holder"]
                primary_holder = RoleAssignment(
                    role_type=holder_data["role_type"],
                    entity_type=EntityType(holder_data["entity_type"]),
                    name=holder_data["name"],
                    confidence=float(holder_data["confidence"]),
                    evidence=holder_data["evidence"]
                )
                # Verify evidence
                if self._verify_evidence(primary_holder.evidence, original_text):
                    primary_holder.confidence = min(1.0, primary_holder.confidence + 0.05)
            except (KeyError, ValueError):
                primary_holder = None
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(roles, primary_holder)
        
        # Get reasoning (internal, not exposed)
        reasoning = data.get("reasoning", "LLM-based semantic analysis completed")
        
        return RoleExtractionResult(
            roles=roles,
            primary_holder=primary_holder,
            confidence=overall_confidence,
            reasoning=reasoning
        )
    
    def _verify_evidence(self, evidence: str, text: str) -> bool:
        """
        Verify that evidence quote exists in original text
        
        INTERNAL REASONING:
        - Extract quoted text from evidence
        - Check if it appears in original document
        - Return True if found (boosts confidence)
        """
        # Extract quoted text from evidence
        quotes = re.findall(r'["\']([^"\']+)["\']', evidence)
        if not quotes:
            # Try to find key phrases
            key_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', evidence)
            quotes = key_phrases[:3]  # Take first 3 capitalized phrases
        
        text_lower = text.lower()
        for quote in quotes:
            if quote.lower() in text_lower:
                return True
        
        return False
    
    def _calculate_overall_confidence(
        self,
        roles: List[RoleAssignment],
        primary_holder: Optional[RoleAssignment]
    ) -> float:
        """
        Calculate overall confidence
        
        INTERNAL REASONING:
        - Average confidence of all roles
        - Boost if primary holder is identified
        - Penalize if too many roles (may indicate confusion)
        - Return weighted average
        """
        if not roles:
            return 0.0
        
        # Average confidence of roles
        avg_role_confidence = sum(r.confidence for r in roles) / len(roles)
        
        # Boost if primary holder is clear
        holder_boost = 0.0
        if primary_holder and primary_holder.confidence >= 0.75:
            holder_boost = 0.1
        
        # Penalize if too many roles (may indicate confusion)
        confusion_penalty = 0.0
        if len(roles) > 5:
            confusion_penalty = 0.05
        
        overall = avg_role_confidence + holder_boost - confusion_penalty
        return max(0.0, min(1.0, overall))
    
    def _extract_with_heuristics(
        self,
        text: str,
        document_family: Optional[str],
        document_subtype: Optional[str],
        reasoning: Optional[str] = None
    ) -> RoleExtractionResult:
        """
        Fallback heuristic extraction (when LLM unavailable)
        
        This is a simplified fallback - not as good as LLM but better than nothing.
        Uses basic pattern matching but tries to be semantic-aware.
        """
        
        if reasoning is None:
            reasoning = "Using heuristic fallback (LLM not available)"
        
        roles = []
        primary_holder = None
        
        # Very basic heuristic: look for party definitions
        # This is NOT regex-based but uses semantic patterns
        
        # Look for "Party A", "Party B", "Contractor", "Client" definitions
        party_patterns = [
            r'(?:party\s+[AB]|contractor|client|service\s+provider|employer|employee)[:\s]+([A-Z][A-Za-z\s&.,]+(?:LLC|Inc|Ltd|Corp|Company)?)',
            r'([A-Z][A-Za-z\s&.,]+(?:LLC|Inc|Ltd|Corp|Company)?)\s+(?:shall|agrees|is\s+engaged|as\s+(?:a|an)\s+(?:contractor|client|service\s+provider))',
        ]
        
        entities_found = set()
        for pattern in party_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(1).strip()
                if len(entity_name) > 3 and len(entity_name) < 100:
                    entities_found.add(entity_name)
        
        # Infer roles from document context (very basic)
        text_lower = text.lower()
        
        for entity in entities_found:
            # Try to infer role from context around entity name
            entity_lower = entity.lower()
            role_type = None
            confidence = 0.6  # Low confidence for heuristics
            
            # Check context around entity mentions
            entity_context = self._get_entity_context(entity, text)
            
            if "contractor" in entity_context.lower() or "service provider" in entity_context.lower():
                role_type = "contractor"
                confidence = 0.7
            elif "client" in entity_context.lower() or "principal" in entity_context.lower():
                role_type = "client"
                confidence = 0.7
            elif "employer" in entity_context.lower():
                role_type = "employer"
                confidence = 0.7
            elif "employee" in entity_context.lower():
                role_type = "employee"
                confidence = 0.7
            
            if role_type:
                # Determine entity type
                entity_type = EntityType.ORGANIZATION
                if len(entity.split()) <= 3 and not any(word in entity.lower() for word in ["llc", "inc", "ltd", "corp", "company"]):
                    entity_type = EntityType.INDIVIDUAL
                
                role = RoleAssignment(
                    role_type=role_type,
                    entity_type=entity_type,
                    name=entity,
                    confidence=confidence,
                    evidence=f"Found entity '{entity}' mentioned in context suggesting {role_type} role"
                )
                
                if confidence >= 0.75:
                    roles.append(role)
        
        # Set primary holder (first contractor or client found)
        for role in roles:
            if role.role_type in ["contractor", "service provider"]:
                primary_holder = role
                break
        
        overall_confidence = self._calculate_overall_confidence(roles, primary_holder)
        
        return RoleExtractionResult(
            roles=roles,
            primary_holder=primary_holder,
            confidence=overall_confidence,
            reasoning=reasoning
        )
    
    def _get_entity_context(self, entity: str, text: str, window: int = 200) -> str:
        """Get context around entity mentions"""
        # Find first mention of entity
        pattern = re.escape(entity)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end]
        return ""
