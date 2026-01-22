"""
Claim Evaluator - Evaluates claims for semantic documents

For semantic documents (contracts, SOW, letters), we don't gate on family_confidence.
Instead, we evaluate CLAIMS to determine if the document demonstrates a certifiable relationship.
"""

from typing import Dict, Any, List
import re


class ClaimEvaluator:
    """
    Evaluates claims for claim-based certification
    
    Key insight: For semantic documents, we certify based on what they DEMONSTRATE,
    not what they ARE structurally.
    """
    
    # Claim patterns - what we're looking for
    CLAIM_PATTERNS = {
        'has_client': [
            r'client:\s*([A-Z][a-z]+)',
            r'client\s+name:\s*([A-Z][a-z]+)',
            r'for\s+client:\s*([A-Z][a-z]+)',
            r'services\s+requested\s+by\s+([A-Z][a-z]+)',  # "services requested by Franco"
            r'dear\s+([A-Z][a-z]+)',  # "Dear Franco,"
            r'\(["\']?client["\']?\)',  # "(Client)" in quotes
            r'client\s+is\s+([A-Z][a-z]+)',
        ],
        'has_contractor': [
            r'contractor:\s*([A-Z][a-z]+)',
            r'contractor\s+name:\s*([A-Z][a-z]+)',
            r'independent\s+contractor:\s*([A-Z][a-z]+)',
            r'engaged\s+as\s+an?\s+independent\s+contractor',
            r'is\s+engaged\s+as\s+an?\s+independent\s+contractor',
            r'contractor\s+details',
            r'this\s+letter\s+certifies\s+that\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',  # "This letter certifies that John Smith"
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+is\s+engaged',  # "John Smith is engaged"
            r'contractor\s+is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ],
        'references_master_agreement': [
            r'reference\s+agreement',
            r'master\s+agreement',
            r'independent\s+contractor\s+agreement',
            r'consulting\s+agreement',
            r'dated\s+\d+',
            r'letter\s+of\s+engagement',
            r'engagement\s+letter',
            r'certificate\s+of\s+engagement',
            r'this\s+engagement\s+letter',
            r'this\s+letter\s+dated',  # "This letter, dated"
        ],
        'defines_scope_of_work': [
            r'statement\s+of\s+work',
            r'scope\s+of\s+work',
            r'work\s+order',
            r'statement\s+of\s+work\s*\(sow\)',
        ],
        'has_effective_date': [
            r'effective\s+date',
            r'effective\s+from',
            r'commencement\s+date',
        ],
        'defines_services': [
            r'services\s+to\s+be\s+provided',
            r'scope\s+of\s+services',
            r'work\s+to\s+be\s+performed',
        ]
    }
    
    def evaluate(self, text: str, document_family: str) -> Dict[str, Any]:
        """
        Evaluate claims in document
        
        Args:
            text: Document text
            document_family: Document family
            
        Returns:
            Claim evaluation with scores and certification readiness
        """
        if not text:
            return {
                'is_contractor_relationship': False,
                'claims_confidence': 0.0,
                'claims_found': {},
                'certifiable': False
            }
        
        text_lower = text.lower()
        claims_found = {}
        claim_scores = {}
        
        # Evaluate each claim pattern
        for claim_name, patterns in self.CLAIM_PATTERNS.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append(pattern)
            
            if matches:
                claims_found[claim_name] = True
                claim_scores[claim_name] = len(matches)
            else:
                claims_found[claim_name] = False
                claim_scores[claim_name] = 0
        
        # Calculate overall claim confidence
        # Critical claims (must have)
        critical_claims = ['has_client', 'has_contractor']
        critical_score = sum(claim_scores.get(c, 0) for c in critical_claims)
        
        # Supporting claims (nice to have)
        supporting_claims = ['references_master_agreement', 'defines_scope_of_work', 'has_effective_date']
        supporting_score = sum(claim_scores.get(c, 0) for c in supporting_claims)
        
        # Base confidence from critical claims
        if critical_score >= 2:  # Both client and contractor found
            base_confidence = 0.85
        elif critical_score == 1:  # One of them found
            # If we have engagement letter pattern, boost confidence
            if claims_found.get('references_master_agreement', False):
                base_confidence = 0.70  # Engagement letter is strong signal
            else:
                base_confidence = 0.50
        else:
            # Even without explicit client/contractor, engagement letter is a strong signal
            if claims_found.get('references_master_agreement', False):
                # Engagement letter found but client/contractor not explicit
                # Check if text contains engagement letter patterns
                if re.search(r'engagement\s+letter|this\s+letter\s+certifies|dear\s+[A-Z]', text, re.IGNORECASE):
                    base_confidence = 0.65  # Engagement letter is certifiable even without explicit labels
            else:
                base_confidence = 0.0
        
        # Boost from supporting claims
        if supporting_score > 0:
            base_confidence = min(0.95, base_confidence + (supporting_score * 0.10))
        
        # Determine if this demonstrates a contractor relationship
        # Multiple ways to demonstrate relationship:
        # 1. Explicit client + contractor labels
        # 2. Engagement letter pattern (strong signal)
        # 3. "Dear [Name]" + engagement letter (common in LoE)
        is_contractor_relationship = (
            (claims_found.get('has_client', False) and claims_found.get('has_contractor', False)) or
            (claims_found.get('references_master_agreement', False) and 
             re.search(r'engagement\s+letter|this\s+letter\s+certifies|dear\s+[A-Z]', text, re.IGNORECASE))
        )
        
        # Additional boost if references master agreement or engagement letter
        if claims_found.get('references_master_agreement', False):
            is_contractor_relationship = True
            base_confidence = min(0.95, base_confidence + 0.20)  # Strong boost for engagement letter
        
        # Certifiable if demonstrates contractor relationship with sufficient confidence
        # Lower threshold for engagement letters (they're inherently certifiable)
        min_confidence = 0.65 if claims_found.get('references_master_agreement', False) else 0.70
        certifiable = is_contractor_relationship and base_confidence >= min_confidence
        
        return {
            'is_contractor_relationship': is_contractor_relationship,
            'claims_confidence': base_confidence,
            'claims_found': claims_found,
            'claim_scores': claim_scores,
            'certifiable': certifiable,
            'critical_claims_present': critical_score >= 2,
            'supporting_claims_count': supporting_score
        }
