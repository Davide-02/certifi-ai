"""
Document classification module
Uses rules + keywords first, LLM as fallback
"""

from typing import Optional, Dict, List
import re


class DocumentClassifier:
    """Classifies document type using rules and patterns"""
    
    # Keyword patterns for each document type
    PATTERNS = {
        'invoice': {
            'keywords': ['fattura', 'invoice', 'numero fattura', 'iva', 'totale', 'importo', 'pagamento'],
            'patterns': [
                r'fattura\s+n[°º]?\s*:?\s*(\w+)',
                r'invoice\s+n[°º]?\s*:?\s*(\w+)',
                r'iva\s*:?\s*(\d+[.,]?\d*)%?',
                r'totale\s*:?\s*€?\s*(\d+[.,]?\d*)',
            ],
            'min_matches': 2
        },
        'diploma': {
            'keywords': ['diploma', 'laurea', 'università', 'universita', 'cfu', 'crediti', 'certificato di laurea'],
            'patterns': [
                r'diploma\s+di\s+laurea',
                r'universit[àa]\s+degli?\s+studi',
                r'cfu\s*:?\s*(\d+)',
                r'crediti\s+formativi',
            ],
            'min_matches': 2
        },
        'id': {
            'keywords': ['carta d\'identità', 'carta di identità', 'documento identità', 'codice fiscale', 'data di nascita'],
            'patterns': [
                r'carta\s+d[ei]\s+identit[àa]',
                r'codice\s+fiscale\s*:?\s*([A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z])',
                r'data\s+di\s+nascita\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                r'nome\s*:?\s*([A-Z][a-z]+)',
                r'cognome\s*:?\s*([A-Z][a-z]+)',
            ],
            'min_matches': 2
        },
        'contract': {
            'keywords': ['contratto', 'contract', 'clausola', 'parti contraenti', 'soggetto'],
            'patterns': [
                r'contratto\s+di\s+(\w+)',
                r'parti\s+contraenti',
                r'clausola\s+(\d+)',
            ],
            'min_matches': 2
        },
        'driving_license': {
            'keywords': ['patente di guida', 'patente', 'republica italiana', 'repubblica italiana', 'a1', 'a2', 'b', 'c1', 'c', 'd1', 'd'],
            'patterns': [
                r'patente\s+di\s+guida',
                r'repubblica\s+italiana',
                r'\b(a1|a2|b|c1|c|d1|d)\b',
                r'\d+[a-z]\.\s*',  # Pattern numerato (1., 2., 3., 4a., 4b., 5.)
                r'4a\.\s*',
                r'4b\.\s*',
            ],
            'min_matches': 2
        },
    }
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize classifier
        
        Args:
            use_llm: Whether to use LLM for classification (requires API key)
        """
        self.use_llm = use_llm
    
    def classify(self, text: str) -> Dict[str, any]:
        """
        Classify document type using MODULAR approach
        
        Multiple classification sources:
        1. Keyword-based classification
        2. Layout-based classification (for structured docs)
        3. LLM classification (fallback)
        
        Takes the best result and explains why
        
        Args:
            text: Extracted document text
            
        Returns:
            Dict with 'type', 'confidence', 'source', and 'matches'
        """
        if not text or len(text.strip()) < 10:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'source': 'none',
                'matches': []
            }
        
        # Multiple classification sources
        classifications = []
        
        # Source 1: Keyword-based classification
        keyword_result = self._classify_by_keywords(text)
        if keyword_result['type'] != 'unknown':
            classifications.append({
                **keyword_result,
                'source': 'keywords'
            })
        
        # Source 2: Layout-based classification (for structured documents)
        layout_result = self._classify_by_layout(text)
        if layout_result['type'] != 'unknown':
            classifications.append({
                **layout_result,
                'source': 'layout'
            })
        
        # Source 3: LLM classification (if enabled and others are uncertain)
        if self.use_llm:
            # Only use LLM if other sources are uncertain
            max_confidence = max([c['confidence'] for c in classifications], default=0.0)
            if max_confidence < 0.5:
                llm_result = self._classify_with_llm(text)
                if llm_result['type'] != 'unknown':
                    classifications.append({
                        **llm_result,
                        'source': 'llm'
                    })
        
        # Get best classification
        if not classifications:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'source': 'none',
                'matches': []
            }
        
        # Take the best (highest confidence)
        best = max(classifications, key=lambda x: x['confidence'])
        
        # Never return confidence = 1.0
        if best['confidence'] >= 0.98:
            best['confidence'] = 0.98
        
        return {
            'type': best['type'],
            'confidence': best['confidence'],
            'source': best.get('source', 'unknown'),
            'matches': best.get('matches', [])
        }
    
    def _classify_by_keywords(self, text: str) -> Dict[str, any]:
        """Keyword-based classification (original method)"""
        text_lower = text.lower()
        scores = {}
        
        # Score each document type
        for doc_type, config in self.PATTERNS.items():
            score = 0
            matches = []
            
            # Check keywords
            keyword_matches = sum(1 for kw in config['keywords'] if kw in text_lower)
            score += keyword_matches * 2
            
            # Check patterns
            pattern_matches = []
            for pattern in config['patterns']:
                found = re.search(pattern, text_lower, re.IGNORECASE)
                if found:
                    pattern_matches.append(found.group(0))
                    score += 3
            
            matches.extend(pattern_matches)
            
            # Normalize score
            total_possible = len(config['keywords']) * 2 + len(config['patterns']) * 3
            normalized_score = min(score / total_possible, 0.98) if total_possible > 0 else 0.0
            
            # Check minimum matches requirement
            if keyword_matches + len(pattern_matches) >= config['min_matches']:
                scores[doc_type] = {
                    'score': normalized_score,
                    'matches': matches,
                    'keyword_matches': keyword_matches
                }
        
        # Get best match
        if not scores:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'matches': []
            }
        
        best_type = max(scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'type': best_type[0],
            'confidence': best_type[1]['score'],
            'matches': best_type[1]['matches']
        }
    
    def _classify_by_layout(self, text: str) -> Dict[str, any]:
        """
        Layout-based classification for structured documents
        
        Detects documents with numbered fields (1., 2., 3., etc.)
        """
        # Check for driving license pattern (numbered fields)
        numbered_fields = re.findall(r'\d+[a-z]?\.\s*', text)
        if len(numbered_fields) >= 3:
            # Check for specific patterns
            if re.search(r'patente|republica italiana|repubblica italiana', text, re.IGNORECASE):
                return {
                    'type': 'driving_license',
                    'confidence': 0.90,  # High confidence from layout
                    'matches': ['numbered_fields', 'patente_keywords']
                }
        
        # Check for MRZ pattern (for ID documents)
        mrz_pattern = re.search(r'[A-Z0-9<]{25,}', text)
        if mrz_pattern:
            return {
                'type': 'id',
                'confidence': 0.85,  # MRZ suggests ID document
                'matches': ['mrz_pattern']
            }
        
        return {
            'type': 'unknown',
            'confidence': 0.0,
            'matches': []
        }
    
    def _classify_with_llm(self, text: str) -> Dict[str, any]:
        """
        Classify using LLM (fallback when rules are uncertain)
        
        Args:
            text: Document text
            
        Returns:
            Classification result
        """
        # This would use OpenAI/Anthropic API
        # For now, return unknown
        # TODO: Implement LLM classification
        return {
            'type': 'unknown',
            'confidence': 0.0,
            'matches': []
        }
