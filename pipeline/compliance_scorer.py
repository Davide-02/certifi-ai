"""
Compliance Scorer - Calculates compliance score for documents

Part of the pipeline: OCR → Layout → Vision → LLM → Normalizer → JSON schema
"""

from typing import Dict, Any


class ComplianceScorer:
    """
    Calculates compliance score for documents
    
    Factors considered:
    - Certification readiness
    - Confidence levels
    - Risk level
    - Missing fields
    - Document quality
    """
    
    def calculate(self, result: Dict[str, Any]) -> float:
        """
        Calculate compliance score (0.0 to 1.0)
        
        Args:
            result: Pipeline result dictionary
            
        Returns:
            Compliance score between 0.0 and 1.0
        """
        # Base score from certification readiness
        base_score = 0.0
        
        if result.get("certification_ready", False):
            base_score = 0.8
        elif result.get("human_review_required", True):
            base_score = 0.5
        else:
            base_score = 0.2
        
        # Adjust based on confidence
        confidence = self._get_confidence(result)
        if confidence > 0:
            base_score = (base_score + confidence) / 2
        
        # Adjust based on risk level
        risk_multiplier = self._get_risk_multiplier(result)
        base_score = base_score * risk_multiplier
        
        # Penalize missing critical fields
        missing_penalty = self._calculate_missing_penalty(result)
        base_score = base_score * (1.0 - missing_penalty)
        
        # Boost for high-quality documents
        quality_boost = self._calculate_quality_boost(result)
        base_score = min(1.0, base_score + quality_boost)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    def _get_confidence(self, result: Dict[str, Any]) -> float:
        """Extract confidence from result"""
        # Try multiple sources
        decision = result.get("metadata", {}).get("decision", {})
        if decision:
            return decision.get("confidence", 0.0)
        
        # Fallback to family confidence
        return result.get("metadata", {}).get("family_confidence", 0.0)
    
    def _get_risk_multiplier(self, result: Dict[str, Any]) -> float:
        """Get risk multiplier based on risk level"""
        risk_level = result.get("risk_level", "high")
        risk_multipliers = {
            "low": 1.0,
            "medium": 0.9,
            "high": 0.7
        }
        return risk_multipliers.get(risk_level.lower(), 0.7)
    
    def _calculate_missing_penalty(self, result: Dict[str, Any]) -> float:
        """Calculate penalty for missing critical fields"""
        missing_fields = result.get("metadata", {}).get("decision", {}).get("missing_fields", [])
        
        if not missing_fields:
            return 0.0
        
        # Penalty: 5% per missing field, max 30%
        penalty = min(0.3, len(missing_fields) * 0.05)
        return penalty
    
    def _calculate_quality_boost(self, result: Dict[str, Any]) -> float:
        """Calculate quality boost from document quality indicators"""
        boost = 0.0
        
        # Boost for trusted source (MRZ, layout rules, etc.)
        trusted_source = result.get("metadata", {}).get("decision", {}).get("trusted_source")
        if trusted_source:
            if trusted_source == "mrz":
                boost += 0.1
            elif trusted_source == "layout_rules":
                boost += 0.05
        
        # Boost for high family confidence
        family_confidence = result.get("metadata", {}).get("family_confidence", 0.0)
        if family_confidence >= 0.9:
            boost += 0.05
        
        # Boost for strong claims
        claims_confidence = result.get("metadata", {}).get("claim_evaluation", {}).get("claims_confidence", 0.0)
        if claims_confidence >= 0.85:
            boost += 0.05
        
        return min(0.2, boost)  # Max 20% boost
