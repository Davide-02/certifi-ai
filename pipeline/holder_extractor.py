"""
Holder Extractor - Extracts holder information (relationship reference)

Part of the pipeline: OCR → Layout → Vision → LLM → Normalizer → JSON schema
"""

from typing import Dict, Any, Optional
import hashlib
import json


class HolderExtractor:
    """
    Extracts holder information from documents
    
    Holder types:
    - relationship: A relationship claim (e.g., contractor-client)
    - individual: An individual person
    - entity: A company or organization
    """
    
    def extract(self, claim: Dict[str, Any], role: str, document_family: str) -> Optional[Dict[str, Any]]:
        """
        Extract holder information from claim
        
        Args:
            claim: Extracted claim dictionary
            role: Inferred role (contractor, employee, etc.)
            document_family: Document family
            
        Returns:
            Holder information or None
        """
        if not claim:
            return None
        
        # Determine holder type based on role and document family
        holder_type = self._determine_holder_type(role, document_family, claim)
        
        # Generate reference hash from claim
        ref_hash = self._generate_holder_ref(claim)
        
        # Calculate confidence
        confidence = self._calculate_holder_confidence(claim, role)
        
        return {
            "type": holder_type,
            "ref": f"rel:sha256({ref_hash[:16]}...)",
            "confidence": confidence
        }
    
    def _determine_holder_type(self, role: str, document_family: str, claim: Dict[str, Any]) -> str:
        """Determine holder type"""
        # For contracts and semantic documents, holder is usually a relationship
        if document_family in ["contract", "certificate"]:
            return "relationship"
        
        # For identity documents, holder is an individual
        if document_family in ["identity", "driving_license"]:
            return "individual"
        
        # For financial/corporate documents, holder might be an entity
        if document_family in ["financial", "corporate"]:
            return "entity"
        
        # Default based on role
        if role in ["contractor", "employee", "student"]:
            return "individual"
        elif role in ["client", "company"]:
            return "entity"
        else:
            return "relationship"
    
    def _generate_holder_ref(self, claim: Dict[str, Any]) -> str:
        """Generate reference hash from claim"""
        # Create canonical representation
        canonical_data = {
            "subject": claim.get("subject"),
            "entity": claim.get("entity"),
            "role": claim.get("role"),
            "start_date": str(claim.get("start_date")) if claim.get("start_date") else None,
        }
        
        # Sort keys for determinism
        canonical_json = json.dumps(canonical_data, sort_keys=True, default=str)
        
        # Generate hash
        return hashlib.sha256(canonical_json.encode()).hexdigest()
    
    def _calculate_holder_confidence(self, claim: Dict[str, Any], role: str) -> float:
        """Calculate confidence in holder extraction"""
        # Base confidence from claim confidence
        base_confidence = claim.get("confidence", 0.0)
        
        # Boost if we have subject and entity (relationship)
        if claim.get("subject") and claim.get("entity"):
            base_confidence = min(0.95, base_confidence + 0.1)
        
        # Boost if role is clear
        if role and role != "unknown":
            base_confidence = min(0.95, base_confidence + 0.05)
        
        return min(0.98, base_confidence)
