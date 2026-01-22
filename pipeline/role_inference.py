"""
Role Inference Engine
Infers roles from semantic patterns, not document titles

Key insight: CertiFi certifies CLAIMS, not documents
A claim is: "X is a contractor for Y from date A to B"
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import re
from datetime import datetime
from dateutil import parser


class Role(str, Enum):
    """Roles that can be inferred from documents"""
    CONTRACTOR = "contractor"
    EMPLOYEE = "employee"
    STUDENT = "student"
    SUPPLIER = "supplier"
    DIRECTOR = "director"
    CLIENT = "client"
    PARTNER = "partner"
    UNKNOWN = "unknown"


class RoleInferenceEngine:
    """
    Infers roles from semantic patterns
    
    Strategy:
    - Look for semantic signals, not document titles
    - Hard evidence (explicit statements) > Soft evidence (implicit)
    - Multiple signals increase confidence
    """
    
    # Hard evidence patterns (explicit statements)
    HARD_EVIDENCE = {
        Role.CONTRACTOR: [
            r'independent\s+contractor',
            r'is\s+not\s+an\s+employee',
            r'contractor\s+agreement',
            r'consulting\s+agreement',
            r'freelance\s+contract',
            r'service\s+agreement',
            r'contractor\s+shall',
            r'contractor\s+will',
            r'as\s+a\s+contractor',
            r'engagement\s+letter',  # Explicit engagement letter
            r'letter\s+of\s+engagement',  # Explicit letter of engagement
            r'service\s+provider',  # "Service Provider" (capitalized, formal)
            r'engaged\s+as\s+an?\s+independent\s+contractor',  # "engaged as an independent contractor"
            r'this\s+engagement\s+letter',  # "This Engagement Letter"
        ],
        Role.EMPLOYEE: [
            r'employment\s+agreement',
            r'employee\s+of',
            r'is\s+an\s+employee',
            r'employment\s+relationship',
            r'wage\s+earner',
        ],
        Role.STUDENT: [
            r'student\s+at',
            r'enrolled\s+student',
            r'student\s+id',
            r'student\s+number',
        ],
        Role.SUPPLIER: [
            r'supplier\s+agreement',
            r'vendor\s+contract',
            r'supplies\s+to',
        ],
        Role.DIRECTOR: [
            r'director\s+of',
            r'board\s+member',
            r'managing\s+director',
        ],
    }
    
    # Soft evidence patterns (implicit signals)
    SOFT_EVIDENCE = {
        Role.CONTRACTOR: [
            r'services\s+rendered',
            r'this\s+agreement',
            r'effective\s+date',
            r'client\s+/\s+contractor',
            r'contractor\s+/\s+client',
            r'statement\s+of\s+work',
            r'work\s+order',
            r'retainer\s+agreement',
            r'letter\s+of\s+engagement',
            r'engagement\s+letter',  # Both orders
            r'certificate\s+of\s+engagement',
            r'service\s+provider',  # "Service Provider" in engagement letters
            r'services\s+provided',  # "services provided by"
            r'services\s+requested\s+by',  # "services requested by [Client]"
            r'this\s+letter\s+certifies',  # "This letter certifies"
            r'dear\s+[A-Z]',  # "Dear [Name]," - client address
            r'fees\s+charged',  # "fees charged by the Service Provider"
        ],
        Role.EMPLOYEE: [
            r'payslip',
            r'salary',
            r'employment\s+letter',
            r'hr\s+letter',
        ],
    }
    
    def infer(self, text: str, document_family: str) -> Dict[str, Any]:
        """
        Infer role from document text
        
        Args:
            text: Document text
            document_family: Document family (contract, certificate, etc.)
            
        Returns:
            Role inference with confidence and evidence
        """
        if not text:
            return {
                'role': Role.UNKNOWN,
                'confidence': 0.0,
                'evidence_type': 'none',
                'signals': []
            }
        
        text_lower = text.lower()
        role_scores = {}
        
        # Score each role
        for role, hard_patterns in self.HARD_EVIDENCE.items():
            hard_matches = []
            for pattern in hard_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    hard_matches.append(pattern)
            
            soft_matches = []
            if role in self.SOFT_EVIDENCE:
                for pattern in self.SOFT_EVIDENCE[role]:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        soft_matches.append(pattern)
            
            # Hard evidence is worth more
            score = len(hard_matches) * 3 + len(soft_matches) * 1
            
            if score > 0:
                role_scores[role] = {
                    'score': score,
                    'hard_evidence': hard_matches,
                    'soft_evidence': soft_matches,
                    'total_signals': len(hard_matches) + len(soft_matches)
                }
        
        # Get best role
        if not role_scores:
            return {
                'role': Role.UNKNOWN,
                'confidence': 0.0,
                'evidence_type': 'none',
                'signals': []
            }
        
        best_role = max(role_scores.items(), key=lambda x: x[1]['score'])
        role_data = best_role[1]
        
        # Calculate confidence
        # Hard evidence = high confidence, soft evidence = lower
        if role_data['hard_evidence']:
            confidence = min(0.95, 0.70 + len(role_data['hard_evidence']) * 0.10)
            evidence_type = 'hard'
        elif role_data['soft_evidence']:
            confidence = min(0.75, 0.50 + len(role_data['soft_evidence']) * 0.05)
            evidence_type = 'soft'
        else:
            confidence = 0.0
            evidence_type = 'none'
        
        # Context boost: if document family matches role
        family_role_map = {
            'contract': Role.CONTRACTOR,
            'certificate': Role.CONTRACTOR,  # Certificate of engagement
            'financial': Role.CONTRACTOR,  # Invoices from contractor
        }
        
        if document_family in family_role_map:
            expected_role = family_role_map[document_family]
            if best_role[0] == expected_role:
                confidence = min(0.98, confidence + 0.10)  # Boost
        
        return {
            'role': best_role[0],
            'confidence': confidence,
            'evidence_type': evidence_type,
            'signals': role_data['hard_evidence'] + role_data['soft_evidence'],
            'hard_evidence_count': len(role_data['hard_evidence']),
            'soft_evidence_count': len(role_data['soft_evidence'])
        }
