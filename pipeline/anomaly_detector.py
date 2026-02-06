"""
Anomaly Detector - Detects anomalies in document analysis

Part of the pipeline: OCR → Layout → Vision → LLM → Normalizer → JSON schema
"""

from typing import Dict, Any, List


class AnomalyDetector:
    """
    Detects anomalies in document analysis results
    
    Anomalies include:
    - Errors in processing
    - Low confidence scores
    - Missing critical fields
    - High risk levels
    - Inconsistencies in extracted data
    """
    
    def detect(self, result: Dict[str, Any]) -> List[str]:
        """
        Detect anomalies in analysis result
        
        Args:
            result: Pipeline result dictionary
            
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        
        # Check for errors
        errors = result.get("errors", [])
        if errors:
            anomalies.extend([f"Error: {e}" for e in errors])
        
        # Check for low confidence
        confidence = self._get_confidence(result)
        if confidence < 0.5:
            anomalies.append(f"Low confidence score: {confidence:.2f}")
        
        # Check for missing critical fields
        missing_fields = result.get("metadata", {}).get("decision", {}).get("missing_fields", [])
        if missing_fields:
            anomalies.append(f"Missing critical fields: {', '.join(missing_fields)}")
        
        # Check for high risk level
        risk_level = result.get("risk_level", "high")
        if risk_level.lower() == "high":
            anomalies.append("High risk level detected")
        
        # Check for family classification issues
        family_confidence = result.get("metadata", {}).get("family_confidence", 0.0)
        if family_confidence < 0.5:
            anomalies.append(f"Low family classification confidence: {family_confidence:.2f}")
        
        # Check for claim inconsistencies
        claim_anomalies = self._check_claim_consistency(result)
        anomalies.extend(claim_anomalies)
        
        # Check for document quality issues
        quality_anomalies = self._check_quality_issues(result)
        anomalies.extend(quality_anomalies)
        
        return anomalies
    
    def _get_confidence(self, result: Dict[str, Any]) -> float:
        """Extract confidence from result"""
        decision = result.get("metadata", {}).get("decision", {})
        if decision:
            return decision.get("confidence", 0.0)
        return result.get("metadata", {}).get("family_confidence", 0.0)
    
    def _check_claim_consistency(self, result: Dict[str, Any]) -> List[str]:
        """Check for inconsistencies in extracted claims"""
        anomalies = []
        
        claim = result.get("claim", {})
        if not claim:
            return anomalies
        
        # Check if role inference matches claim role
        inferred_role = result.get("inferred_role", "unknown")
        claim_role = claim.get("role", "unknown")
        
        if inferred_role != claim_role and inferred_role != "unknown" and claim_role != "unknown":
            anomalies.append(f"Role mismatch: inferred={inferred_role}, claim={claim_role}")
        
        # Check if dates are logical
        start_date = claim.get("start_date")
        end_date = claim.get("end_date")
        
        if start_date and end_date:
            try:
                from datetime import datetime
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                if end_date < start_date:
                    anomalies.append("End date is before start date")
            except Exception:
                pass
        
        # Check if amount is reasonable
        amount = claim.get("amount")
        if amount is not None:
            if amount < 0:
                anomalies.append("Negative amount detected")
            elif amount > 1000000:  # Suspiciously large
                anomalies.append(f"Unusually large amount: {amount}")
        
        return anomalies
    
    def _check_quality_issues(self, result: Dict[str, Any]) -> List[str]:
        """Check for document quality issues"""
        anomalies = []
        
        # Check text length
        text_length = result.get("metadata", {}).get("text_length", 0)
        if text_length < 50:
            anomalies.append("Very short document text (possible OCR failure)")
        
        # Check for certification readiness without proper data
        if result.get("certification_ready", False):
            # Should have either extracted data or claim
            has_data = result.get("data") is not None
            has_claim = result.get("claim") is not None
            
            if not has_data and not has_claim:
                anomalies.append("Certification ready but no data or claim extracted")
        
        return anomalies
