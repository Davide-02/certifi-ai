"""
Main orchestrator for the document processing pipeline
"""

from typing import Union, Dict, Any, Optional
from pathlib import Path
import hashlib
import json

from .ocr import TextExtractor
from .classifier import DocumentClassifier
from .family_classifier import FamilyClassifier, DocumentFamily
from .policy_resolver import PolicyResolver, CertificationPolicy
from .role_inference import RoleInferenceEngine, Role
from .claim_extractor import ClaimExtractor
from .claim_evaluator import ClaimEvaluator
from .extractor import InformationExtractor
from .validators import DocumentValidator
from .decision_engine import DecisionEngine, CertificationProfile, RiskLevel
from .schemas import BaseDocumentSchema


class DocumentPipeline:
    """
    Main pipeline orchestrator - 3 LEVEL ARCHITECTURE
    
    Flow:
    1. Upload → Text Extraction
    2. Family Classification (identity, contract, certificate, etc.)
    3. Policy Resolution (how to certify this family)
    4. Information Extraction (ONLY if policy requires it)
    5. Certification Decision (based on policy)
    6. Hash Generation (if certifiable)
    
    Key insight: CertiFi certifies POLICIES, not documents
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "openai"
    ):
        """
        Initialize pipeline
        
        Args:
            use_llm: Whether to use LLM for classification/extraction
            llm_provider: LLM provider ('openai' or 'anthropic')
        """
        self.text_extractor = TextExtractor()
        self.family_classifier = FamilyClassifier()  # Family classification
        self.classifier = DocumentClassifier(use_llm=use_llm)  # Type classification (optional)
        self.policy_resolver = PolicyResolver()  # Policy resolution
        self.role_inference = RoleInferenceEngine()  # Role inference
        self.claim_extractor = ClaimExtractor()  # Claim extraction
        self.claim_evaluator = ClaimEvaluator()  # NEW: Claim evaluation for semantic docs
        self.extractor = InformationExtractor(use_llm=use_llm, llm_provider=llm_provider)
        self.validator = DocumentValidator()
        self.decision_engine = DecisionEngine()
    
    def process(
        self,
        file_path: Union[str, Path],
        document_type: Optional[str] = None,
        certification_profile: Optional[CertificationProfile] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the full pipeline
        
        Args:
            file_path: Path to document file
            document_type: Optional document type (if known, skips classification)
            
        Returns:
            Complete processing result with extracted data and metadata
        """
        result = {
            'success': False,
            'document_type': None,
            'data': None,
            'validation': None,
            'certification_ready': False,
            'human_review_required': True,
            'metadata': {},
            'errors': []
        }
        
        try:
            # STEP 1: Extract text
            text = self.text_extractor.extract(str(file_path))
            if not text or len(text.strip()) < 10:
                result['errors'].append("Failed to extract text or text too short")
                return result
            
            result['metadata']['text_length'] = len(text)
            result['metadata']['text_preview'] = text[:200] + "..." if len(text) > 200 else text
            
            # STEP 2: Classify FAMILY (LEVEL 1 - KEY CLASSIFIER) + SUBTYPE (LEVEL 2)
            family_result = self.family_classifier.classify(text)
            document_family = family_result['family']
            document_subtype = family_result.get('subtype')  # NEW: Multi-level classification
            family_confidence = family_result['confidence']
            family_source = family_result['source']
            co_occurrence_boost = family_result.get('co_occurrence_boost', 0.0)  # NEW: Semantic boost
            
            result['document_family'] = document_family.value
            result['document_subtype'] = document_subtype.value if document_subtype else None  # NEW
            result['metadata']['family_confidence'] = family_confidence
            result['metadata']['family_source'] = family_source
            result['metadata']['co_occurrence_boost'] = co_occurrence_boost  # NEW
            
            if document_family == DocumentFamily.UNKNOWN:
                result['errors'].append("Could not classify document family")
                result['certification_ready'] = False
                result['human_review_required'] = True
                return result
            
            # STEP 3: Evaluate CLAIMS FIRST (before policy resolution)
            # CRITICAL: Always evaluate claims, regardless of family classification
            # This allows us to override family classification if claims are strong
            claim_evaluation = self.claim_evaluator.evaluate(text, document_family.value)
            result['metadata']['claim_evaluation'] = claim_evaluation
            
            # NEW: Adaptive confidence adjustment based on claims
            # If claims are strong, boost family confidence (reduces false negatives)
            claims_confidence = claim_evaluation.get('claims_confidence', 0.0)
            if claims_confidence >= 0.70:
                # Strong claims → boost family confidence
                adaptive_boost = min(0.15, (claims_confidence - 0.70) * 0.3)  # Max 0.15 boost
                family_confidence = min(0.98, family_confidence + adaptive_boost)
                result['metadata']['adaptive_confidence_boost'] = adaptive_boost
                result['metadata']['family_confidence'] = family_confidence
            
            # If claims indicate contractor relationship, force family to CONTRACT
            # This fixes misclassification issues (e.g., driving_license misclassified as contract)
            if (claim_evaluation.get('is_contractor_relationship', False) and 
                claim_evaluation.get('claims_confidence', 0) >= 0.70):
                # Override family classification based on strong claims
                if document_family != DocumentFamily.CONTRACT:
                    document_family = DocumentFamily.CONTRACT
                    family_confidence = max(family_confidence, 0.70)  # Boost to minimum for semantic docs
                    result['document_family'] = document_family.value
                    result['metadata']['family_override'] = 'claim_based_override'
                    result['metadata']['family_confidence'] = family_confidence
                # Even if already CONTRACT, boost confidence if claims are strong
                elif family_confidence < 0.70:
                    family_confidence = max(family_confidence, 0.70)
                    result['metadata']['family_confidence'] = family_confidence
                    result['metadata']['confidence_boost'] = 'claim_based'
            
            # Determine if this is a semantic document
            is_semantic = document_family in [DocumentFamily.CONTRACT, DocumentFamily.CERTIFICATE, DocumentFamily.FINANCIAL, DocumentFamily.CORPORATE]
            
            # Resolve policy (with claim-based flag for semantic docs)
            policy_decision = self.policy_resolver.resolve(
                family=document_family,
                family_confidence=family_confidence,
                use_claim_based=is_semantic and claim_evaluation and claim_evaluation.get('is_contractor_relationship', False)
            )
            
            result['certification_policy'] = policy_decision['policy'].value
            result['metadata']['policy_decision'] = policy_decision
            
            # For semantic documents, certification decision is based on CLAIMS, not family_confidence
            # CRITICAL: Even if family was misclassified, if claims are valid, certify
            if is_semantic and claim_evaluation:
                # Use claim-based certification
                if claim_evaluation['certifiable']:
                    result['certification_ready'] = True
                    result['human_review_required'] = claim_evaluation['claims_confidence'] < 0.85
                    result['metadata']['certification_method'] = 'claim_based'
                    result['metadata']['claims_confidence'] = claim_evaluation['claims_confidence']
                else:
                    result['certification_ready'] = False
                    result['human_review_required'] = True
                    result['errors'].append(f"Claims evaluation failed: insufficient evidence of contractor relationship")
                    return result
            elif not policy_decision['certifiable']:
                # For structured documents, use policy decision
                # BUT: if claims are strong, still allow certification
                if claim_evaluation.get('is_contractor_relationship', False) and claim_evaluation.get('claims_confidence', 0) >= 0.70:
                    # Override: strong claims override structured document gate
                    result['certification_ready'] = True
                    result['human_review_required'] = claim_evaluation['claims_confidence'] < 0.85
                    result['metadata']['certification_method'] = 'claim_based_override'
                    result['metadata']['claims_confidence'] = claim_evaluation['claims_confidence']
                    # Force policy to hash_only for claim-based
                    policy_decision['policy'] = CertificationPolicy.HASH_ONLY
                    result['certification_policy'] = CertificationPolicy.HASH_ONLY.value
                else:
                    result['certification_ready'] = False
                    result['human_review_required'] = True
                    result['errors'].append(f"Document family {document_family.value} not certifiable: {policy_decision.get('reason', 'unknown')}")
                    return result
            
            # STEP 4: Infer ROLE (NEW - for claim-based certification)
            role_result = self.role_inference.infer(text, document_family.value)
            result['inferred_role'] = role_result['role'].value
            result['metadata']['role_inference'] = {
                'role': role_result['role'].value,
                'confidence': role_result['confidence'],
                'evidence_type': role_result['evidence_type'],
                'signals': role_result['signals']
            }
            
            # STEP 5: Extract CLAIM (NEW - the certifiable statement)
            claim = self.claim_extractor.extract(text, role_result['role'], document_family.value)
            result['claim'] = claim
            result['metadata']['claim_statement'] = self.claim_extractor.format_claim(claim)
            
            # STEP 6: Extract information (ONLY if policy requires it)
            extracted_schema = None
            if policy_decision['requires_extraction']:
                # Map family to document type for extraction
                family_to_type = {
                    DocumentFamily.IDENTITY: 'id',
                    DocumentFamily.DRIVING_LICENSE: 'driving_license',
                    DocumentFamily.CERTIFICATE: 'diploma',
                    DocumentFamily.FINANCIAL: 'invoice',
                    DocumentFamily.CONTRACT: 'contract',
                    DocumentFamily.CORPORATE: 'corporate'
                }
                
                doc_type = family_to_type.get(document_family, 'unknown')
                if doc_type != 'unknown':
                    extracted_schema = self.extractor.extract(text, doc_type)
                    result['data'] = extracted_schema
                    trusted_source = getattr(extracted_schema, 'trusted_source', None)
                else:
                    trusted_source = None
            else:
                # Hash-only policy (e.g., contracts) - no extraction needed
                trusted_source = 'file_integrity'
                result['data'] = None
                result['metadata']['extraction_skipped'] = 'Policy does not require extraction'
            
            # STEP 7: Use DecisionEngine for final certification decision
            # Decision is now based on CLAIM, not just document
            # CRITICAL: Always initialize decision to avoid UnboundLocalError
            decision = {
                'certification_ready': False,
                'human_review_required': True,
                'confidence': 0.0,
                'risk_level': 'high',
                'reason': 'uninitialized',
                'profile': 'unknown'
            }
            
            if extracted_schema:
                # For documents requiring extraction, use DecisionEngine
                decision = self.decision_engine.decide(
                    document_type=document_family.value,
                    classification_confidence=family_confidence,
                    extraction_confidence=extracted_schema.confidence,
                    trusted_source=trusted_source,
                    field_confidence=getattr(extracted_schema, 'field_confidence', {}),
                    missing_fields=getattr(extracted_schema, 'missing_fields', []),
                    profile=certification_profile
                )
                
                # Update schema with decision
                extracted_schema.certification_ready = decision['certification_ready']
                extracted_schema.human_review_required = decision['human_review_required']
                extracted_schema.confidence = decision['confidence']
                
                # Add decision metadata
                extracted_schema.metadata['decision'] = decision
                extracted_schema.metadata['risk_level'] = decision['risk_level'].value
                extracted_schema.metadata['certification_profile'] = decision.get('profile', 'auto')
                
                # Validate
                validation_result = self.validator.validate(extracted_schema)
                result['validation'] = validation_result
                
                result['certification_ready'] = decision['certification_ready']
                result['human_review_required'] = decision['human_review_required']
                result['risk_level'] = decision['risk_level'].value
                result['certification_profile'] = decision.get('profile', 'auto')
            elif is_semantic and claim_evaluation and claim_evaluation.get('certifiable', False):
                # For claim-based semantic documents (hash_only policy)
                # Decision already made in STEP 3, just format it
                decision = {
                    'certification_ready': True,
                    'human_review_required': claim_evaluation['claims_confidence'] < 0.85,
                    'confidence': claim_evaluation['claims_confidence'],
                    'risk_level': RiskLevel.LOW if claim_evaluation['claims_confidence'] >= 0.90 else RiskLevel.MEDIUM,
                    'reason': 'claim_based_certification',
                    'profile': policy_decision['policy'].value
                }
                
                result['certification_ready'] = True
                result['human_review_required'] = decision['human_review_required']
                result['risk_level'] = decision['risk_level'].value if isinstance(decision['risk_level'], RiskLevel) else decision['risk_level']
                result['certification_profile'] = policy_decision['policy'].value
            else:
                # Fallback for other cases
                result['certification_ready'] = False
                result['human_review_required'] = True
                result['risk_level'] = 'high'
                result['certification_profile'] = policy_decision['policy'].value
            
            # STEP 8: Generate hash for CertiFi
            # CRITICAL: Hash is ALWAYS calculated if certifiable
            # Hash is the PROOF, certification is the DECISION to publish it
            if result['certification_ready']:
                # Hash generation based on policy
                certification_method = result.get('metadata', {}).get('certification_method', 'standard')
                
                if (policy_decision['policy'] == CertificationPolicy.HASH_ONLY or 
                    certification_method == 'claim_based'):
                    # For hash-only or claim-based policy, hash the entire file
                    try:
                        with open(str(file_path), 'rb') as f:
                            file_content = f.read()
                        # Use binary hash for files (more reliable)
                        import hashlib
                        file_hash = hashlib.sha256(file_content).hexdigest()
                        result['metadata']['file_hash'] = file_hash
                        result['metadata']['canonical_hash'] = file_hash
                    except Exception as e:
                        result['errors'].append(f"Failed to generate file hash: {str(e)}")
                        result['metadata']['canonical_hash'] = None
                elif extracted_schema and extracted_schema.raw_text:
                    # For extraction-based policies, hash the structured data
                    result['metadata']['text_hash'] = self._generate_hash(extracted_schema.raw_text)
                    result['metadata']['canonical_hash'] = self._generate_canonical_hash(extracted_schema)
                else:
                    # Fallback: hash the text we extracted
                    result['metadata']['text_hash'] = self._generate_hash(text)
                    result['metadata']['canonical_hash'] = result['metadata']['text_hash']
                
                # Always include claim hash if claim exists (for claim-based certification)
                if result.get('claim'):
                    claim_json = json.dumps(result['claim'], default=str, sort_keys=True)
                    result['metadata']['claim_hash'] = self._generate_hash(claim_json)
                    # Canonical hash includes claim if present
                    if result['metadata'].get('canonical_hash'):
                        combined = result['metadata']['canonical_hash'] + result['metadata']['claim_hash']
                        result['metadata']['canonical_hash'] = self._generate_hash(combined)
            else:
                # Even if not ready, we can still calculate hash (but don't use it for certification)
                result['metadata']['canonical_hash'] = None
                result['metadata']['hash_generation_blocked'] = 'Document not certification ready'
            
            # Success = pipeline completed successfully (not certification approved)
            # This is different from certification_ready
            if extracted_schema:
                validation_errors = validation_result.get('errors', []) if 'validation_result' in locals() else []
                result['success'] = len(validation_errors) == 0
            else:
                # For hash-only/claim-based, success = no pipeline errors
                result['success'] = len(result.get('errors', [])) == 0
            
            # Add decision metadata
            result['metadata']['decision'] = {
                'can_certify': result['certification_ready'],
                'needs_human': result['human_review_required'],
                'confidence': decision.get('confidence', 0.0),
                'risk_level': decision['risk_level'].value if isinstance(decision['risk_level'], RiskLevel) else decision.get('risk_level', 'unknown'),
                'reason': decision.get('reason', 'unknown'),
                'trusted_source': getattr(extracted_schema, 'trusted_source', None) if extracted_schema else trusted_source,
                'missing_fields': getattr(extracted_schema, 'missing_fields', []) if extracted_schema else [],
                'details': decision.get('details', {}),
                'certification_method': result.get('metadata', {}).get('certification_method', 'standard')
            }
            
        except Exception as e:
            result['errors'].append(f"Pipeline error: {str(e)}")
            result['success'] = False
        
        return result
    
    def _generate_hash(self, text: str) -> str:
        """Generate SHA256 hash of text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _generate_canonical_hash(self, schema: BaseDocumentSchema) -> str:
        """
        Generate canonical hash for CertiFi on-chain storage
        
        This creates a deterministic hash from the structured data
        """
        # Create canonical JSON representation
        canonical_data = schema.dict(exclude={'raw_text', 'extracted_at', 'metadata'})
        
        # Sort keys for determinism
        canonical_json = json.dumps(canonical_data, sort_keys=True, default=str)
        
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    def to_json(self, result: Dict[str, Any]) -> str:
        """
        Convert result to JSON string
        
        Args:
            result: Pipeline result dictionary
            
        Returns:
            JSON string
        """
        def serialize(obj):
            # Handle Pydantic models
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()  # Pydantic v2
            elif hasattr(obj, 'dict'):
                return obj.dict()  # Pydantic v1
            # Handle datetime
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            # Handle Decimal
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Decimal':
                return float(obj)
            return str(obj)
        
        return json.dumps(result, default=serialize, indent=2, ensure_ascii=False)
