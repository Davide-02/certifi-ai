"""
NEW PRODUCTION-GRADE ORCHESTRATOR

Implements the 6-layer architecture:
1. PERCEPTION LAYER - Raw signal extraction
2. STRUCTURE LAYER - Rule-based structure analysis
3. SEMANTIC EXTRACTION LAYER - LLM-assisted fact extraction
4. INFERENCE ENGINE - Deterministic decision making
5. INTENT & ONTOLOGY LAYER - Real-world intent inference
6. PROCESS CONTEXT LAYER - Business process mapping

This orchestrator ensures:
- Deterministic final decisions (Python code, not LLM)
- Explainable results (traceable signals)
- Modular design (each layer isolated)
- Production-grade error handling
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime

from .perception import PerceptionLayer, PerceptionResult
from .structure_analyzer import StructureAnalyzer, StructureAnalysis
from .semantic_extractor import SemanticExtractor, ExtractedFacts
from .inference_engine import InferenceEngine, InferenceDecision
from .intent_analyzer import IntentAnalyzer, IntentAnalysis
from .process_context import ProcessContextAnalyzer, ProcessContext

# Import existing modules for compatibility
from .claim_extractor import ClaimExtractor
from .role_inference import RoleInferenceEngine, Role
from .semantic_role_extractor import SemanticRoleExtractor  # NEW: Semantic role extraction
from .holder_extractor import HolderExtractor
from .compliance_scorer import ComplianceScorer
from .anomaly_detector import AnomalyDetector


class ProductionPipeline:
    """
    Production-grade document intelligence pipeline
    
    Architecture:
    1. PERCEPTION: Extract raw signals
    2. STRUCTURE: Rule-based structure analysis
    3. SEMANTIC: LLM-assisted fact extraction (optional)
    4. INFERENCE: Deterministic decision making
    5. INTENT & ONTOLOGY: Real-world intent inference
    6. PROCESS CONTEXT: Business process mapping
    """
    
    def __init__(self, use_llm: bool = False, llm_provider: str = "openai"):
        """
        Initialize pipeline
        
        Args:
            use_llm: Whether to use LLM for semantic extraction
            llm_provider: "openai" or "anthropic"
        """
        self.perception_layer = PerceptionLayer()
        self.structure_analyzer = StructureAnalyzer()
        self.semantic_extractor = SemanticExtractor(use_llm=use_llm, llm_provider=llm_provider)
        self.inference_engine = InferenceEngine()
        self.intent_analyzer = IntentAnalyzer()  # Intent & Ontology Layer
        self.process_context_analyzer = ProcessContextAnalyzer()  # Process Context Layer
        
        # Additional modules
        self.role_inference = RoleInferenceEngine()  # Legacy role inference (regex-based)
        self.semantic_role_extractor = SemanticRoleExtractor(use_llm=use_llm, llm_provider=llm_provider)  # Semantic role extraction (LLM-based)
        from .roles_extractor import RolesExtractor  # NEW: Deterministic roles extractor
        self.roles_extractor = RolesExtractor(use_llm=use_llm, llm_provider=llm_provider)  # Deterministic roles extraction
        self.claim_extractor = ClaimExtractor()  # For detailed claim extraction (especially CV)
        self.holder_extractor = HolderExtractor()
        self.compliance_scorer = ComplianceScorer()
        self.anomaly_detector = AnomalyDetector()
    
    def process(
        self,
        file_path: Union[str, Path],
        requested_tasks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process document through 5-layer architecture
        
        Args:
            file_path: Path to document
            requested_tasks: List of tasks (classify, extract, claims, holder, roles, compliance_score, intent)
            
        Returns:
            Complete analysis result with explainability
        """
        requested_tasks = requested_tasks or ["classify", "extract"]
        
        result = {
            "success": False,
            "document_family": "unknown",
            "document_subtype": None,
            "confidence": 0.0,
            "explanation": "",
            "signals_used": [],
            "alternatives_considered": [],
            "errors": [],
            "metadata": {}
        }
        
        try:
            # === LAYER 1: PERCEPTION ===
            perception = self.perception_layer.perceive(file_path)
            result["metadata"]["perception"] = {
                "text_length": perception.text_length,
                "has_tables": perception.has_tables,
                "has_signatures": perception.has_signatures,
                "detected_language": perception.detected_language.value,
                "extraction_methods": perception.extraction_methods
            }
            
            if not perception.raw_text or len(perception.raw_text.strip()) < 10:
                result["errors"].append("Failed to extract text or text too short")
                return result
            
            # === LAYER 2: STRUCTURE ===
            structure = self.structure_analyzer.analyze(perception)
            result["metadata"]["structure_analysis"] = {
                "primary_family": structure.primary_family.value if structure.primary_family else None,
                "primary_confidence": structure.primary_confidence,
                "candidates": [
                    {
                        "family": c.family.value,
                        "confidence": c.confidence,
                        "explanation": c.explanation
                    }
                    for c in structure.candidates
                ],
                "all_signals": [
                    {
                        "name": s.name,
                        "detected": s.detected,
                        "confidence": s.confidence
                    }
                    for s in structure.all_signals
                ]
            }
            
            # === LAYER 3: SEMANTIC EXTRACTION ===
            facts = self.semantic_extractor.extract(
                perception.raw_text,
                structure_hints={"family": structure.primary_family.value if structure.primary_family else None}
            )
            result["metadata"]["extracted_facts"] = {
                "extraction_method": facts.extraction_method,
                "fields_extracted": facts.fields_extracted or [],
                "confidence": facts.confidence
            }
            
            # === LAYER 4: INFERENCE ENGINE ===
            inference = self.inference_engine.infer(
                structure,
                facts,
                perception.raw_text
            )
            
            # Set final results
            result["document_family"] = inference.document_family.value
            result["document_subtype"] = inference.document_subtype.value
            result["confidence"] = inference.confidence
            result["explanation"] = inference.explanation
            result["signals_used"] = inference.signals_used
            result["alternatives_considered"] = inference.alternatives_considered
            
            # Extract detailed claim using ClaimExtractor (especially for CV/resume)
            if "claims" in requested_tasks or "extract" in requested_tasks:
                try:
                    # Infer role
                    role_result = self.role_inference.infer(perception.raw_text, inference.document_family.value)
                    role = role_result.get("role", Role.UNKNOWN)
                    
                    # Extract claim with ClaimExtractor (handles CV data extraction)
                    claim = self.claim_extractor.extract(
                        text=perception.raw_text,
                        role=role,
                        document_family=inference.document_family.value,
                        compensation_table_data=None,
                        document_subtype=inference.document_subtype.value
                    )
                    result["claim"] = claim
                except Exception as e:
                    result["errors"].append(f"Claim extraction error: {str(e)}")
                    result["claim"] = {}
            
            # === LAYER 5: INTENT & ONTOLOGY ===
            # Analyze real-world intent (if requested or always for explainability)
            if "intent" in requested_tasks or "classify" in requested_tasks:
                try:
                    # Import enums for conversion
                    from .structure_analyzer import DocumentFamily
                    from .inference_engine import DocumentSubtype
                    
                    # Convert string values to enum instances
                    family_enum = DocumentFamily(result["document_family"])
                    subtype_enum = None
                    if result["document_subtype"]:
                        try:
                            subtype_enum = DocumentSubtype(result["document_subtype"])
                        except ValueError:
                            subtype_enum = None
                    
                    # Analyze intent
                    intent_analysis = self.intent_analyzer.analyze(
                        family=family_enum,
                        subtype=subtype_enum,
                        facts=facts,
                        structure_signals=inference.signals_used
                    )
                    
                    # Add intent results to response
                    result["intent"] = {
                        "primary_intent": intent_analysis.primary_intent.value,
                        "secondary_intents": [i.value for i in intent_analysis.secondary_intents],
                        "trust_role": intent_analysis.trust_role.value,
                        "risk_profile": intent_analysis.risk_profile.value,
                        "confidence": intent_analysis.confidence,
                        "explanation": intent_analysis.explanation,
                        "decision_path": intent_analysis.decision_path
                    }
                    
                    # Add to metadata
                    result["metadata"]["intent_analysis"] = {
                        "primary_intent": intent_analysis.primary_intent.value,
                        "secondary_intents": [i.value for i in intent_analysis.secondary_intents],
                        "trust_role": intent_analysis.trust_role.value,
                        "risk_profile": intent_analysis.risk_profile.value,
                        "confidence": intent_analysis.confidence
                    }
                    
                except Exception as e:
                    result["errors"].append(f"Intent analysis error: {str(e)}")
                    result["intent"] = {
                        "primary_intent": "unknown",
                        "secondary_intents": [],
                        "trust_role": "unknown",
                        "risk_profile": "medium",
                        "confidence": 0.0,
                        "explanation": f"Intent analysis failed: {str(e)}",
                        "decision_path": []
                    }
            
            # === LAYER 6: PROCESS CONTEXT ===
            # Map intent to business process (if intent analysis succeeded)
            if "process_context" in requested_tasks or ("intent" in requested_tasks and result.get("intent")):
                try:
                    # Get intent data
                    intent_data = result.get("intent", {})
                    if intent_data and intent_data.get("primary_intent") != "unknown":
                        # Import enums for conversion
                        from .intent_analyzer import DocumentIntent, TrustRole, RiskProfile
                        
                        # Convert string values to enum instances
                        try:
                            primary_intent = DocumentIntent(intent_data["primary_intent"])
                        except ValueError:
                            primary_intent = DocumentIntent.UNKNOWN
                        
                        try:
                            trust_role = TrustRole(intent_data["trust_role"])
                        except ValueError:
                            trust_role = TrustRole.UNKNOWN
                        
                        try:
                            risk_profile = RiskProfile(intent_data["risk_profile"])
                        except ValueError:
                            risk_profile = RiskProfile.MEDIUM
                        
                        # Analyze process context
                        process_context = self.process_context_analyzer.analyze(
                            intent=primary_intent,
                            trust_role=trust_role,
                            risk_profile=risk_profile,
                            facts=facts
                        )
                        
                        # Add process context results to response
                        result["process_context"] = {
                            "business_process": process_context.business_process.value,
                            "compliance_domain": process_context.compliance_domain.value,
                            "lifecycle_stage": process_context.lifecycle_stage.value,
                            "recommended_actions": process_context.recommended_actions,
                            "confidence": process_context.confidence,
                            "explanation": process_context.explanation,
                            "decision_path": process_context.decision_path
                        }
                        
                        # Add to metadata
                        result["metadata"]["process_context"] = {
                            "business_process": process_context.business_process.value,
                            "compliance_domain": process_context.compliance_domain.value,
                            "lifecycle_stage": process_context.lifecycle_stage.value,
                            "recommended_actions_count": len(process_context.recommended_actions),
                            "confidence": process_context.confidence
                        }
                    else:
                        result["process_context"] = {
                            "business_process": "unknown",
                            "compliance_domain": "unknown",
                            "lifecycle_stage": "creation",
                            "recommended_actions": [],
                            "confidence": 0.0,
                            "explanation": "Process context analysis skipped - intent analysis unavailable",
                            "decision_path": []
                        }
                        
                except Exception as e:
                    result["errors"].append(f"Process context analysis error: {str(e)}")
                    result["process_context"] = {
                        "business_process": "unknown",
                        "compliance_domain": "unknown",
                        "lifecycle_stage": "creation",
                        "recommended_actions": [],
                        "confidence": 0.0,
                        "explanation": f"Process context analysis failed: {str(e)}",
                        "decision_path": []
                    }
            
            # === ADDITIONAL PROCESSING ===
            
            # NEW: Deterministic Roles & Holder Extraction (separate step from classification)
            if "roles" in requested_tasks or "holder" in requested_tasks:
                try:
                    # Combine full text with table data for better extraction
                    full_text = perception.raw_text
                    
                    # Add table data to text for entity extraction
                    if perception.extracted_tables:
                        table_text = self._extract_text_from_tables(perception.extracted_tables)
                        if table_text:
                            full_text = f"{full_text}\n\n--- TABLES ---\n{table_text}"
                    
                    roles_extraction = self.roles_extractor.extract(
                        text=full_text,  # Use full text including tables
                        document_family=inference.document_family.value,
                        document_subtype=inference.document_subtype.value,
                        tables=perception.extracted_tables  # Pass tables for better extraction
                    )
                    
                    # Store roles extraction result
                    result["roles"] = roles_extraction.to_dict()
                    
                    # Use primary holder from roles extraction if available
                    if roles_extraction.primary_holder and roles_extraction.primary_holder.get("confidence", 0) >= 0.75:
                        result["holder"] = {
                            "type": "relationship" if roles_extraction.primary_holder.get("entity_type") == "organization" else "individual",
                            "name": roles_extraction.primary_holder.get("name"),
                            "role": roles_extraction.primary_holder.get("role_type"),
                            "confidence": roles_extraction.primary_holder.get("confidence", 0.0),
                            "justification": roles_extraction.primary_holder.get("justification", "")
                        }
                except Exception as e:
                    result["errors"].append(f"Roles extraction error: {str(e)}")
                    result["roles"] = {"roles": [], "primary_holder": None, "confidence": 0.0}
            
            # Initialize claim_dict for use across multiple tasks
            claim_dict = {
                "role": "contractor" if facts.party_1_type == "individual" else "company",
                "entity": facts.party_2_name or facts.party_1_name,
                "amount": facts.amount,
                "currency": facts.currency,
                "start_date": facts.effective_date,
                "end_date": facts.expiration_date,
                "subject": facts.subject or facts.services,
                "confidence": inference.confidence
            }
            
            # Extract holder if requested (fallback to legacy method if semantic extraction didn't work)
            if "holder" in requested_tasks and "holder" not in result:
                try:
                    # HolderExtractor.extract() signature: extract(claim, role, document_family)
                    role_str = claim_dict.get("role", "unknown")
                    holder = self.holder_extractor.extract(claim_dict, role_str, inference.document_family.value)
                    if holder:
                        result["holder"] = holder
                except Exception as e:
                    result["errors"].append(f"Legacy holder extraction error: {str(e)}")
            
            # Extract claims if requested
            if "claims" in requested_tasks:
                # Use claim from ClaimExtractor if available (has CV data for resume)
                claim = result.get("claim", {})
                
                if claim:
                    # Extract from detailed claim
                    claims_info = {
                        "is_contractor": claim.get("role") == "contractor" if claim.get("role") else None,
                        "amount": claim.get("amount"),
                        "currency": claim.get("currency"),
                        "secondary_currency": claim.get("secondary_currency"),
                        "secondary_amount": claim.get("secondary_amount"),
                        "subject": claim.get("subject") or claim.get("services"),
                        "entity": claim.get("entity"),
                        "start_date": claim.get("start_date").isoformat() if isinstance(claim.get("start_date"), datetime) else claim.get("start_date"),
                        "end_date": claim.get("end_date").isoformat() if isinstance(claim.get("end_date"), datetime) else claim.get("end_date"),
                    }
                else:
                    # Fallback to facts from semantic extraction
                    claims_info = {
                        "is_contractor": facts.party_1_type == "individual" if facts.party_1_type else None,
                        "amount": facts.amount,
                        "currency": facts.currency,
                        "secondary_currency": facts.secondary_currency,
                        "secondary_amount": facts.secondary_amount,
                        "subject": facts.subject or facts.services,
                        "entity": facts.party_2_name or facts.party_1_name,
                        "start_date": facts.effective_date,
                        "end_date": facts.expiration_date,
                    }
                
                result["claims"] = claims_info
            
            # Calculate compliance score if requested
            if "compliance_score" in requested_tasks:
                try:
                    # ComplianceScorer.calculate() takes only result dict as argument
                    compliance_score = self.compliance_scorer.calculate(result)
                    result["compliance_score"] = compliance_score
                except Exception as e:
                    result["errors"].append(f"Compliance score calculation error: {str(e)}")
                    result["compliance_score"] = None
            
            # Detect anomalies (always run, passes full result)
            try:
                # AnomalyDetector.detect() expects the full result dictionary
                anomalies = self.anomaly_detector.detect(result)
            except Exception as e:
                # Fallback: return empty anomalies list
                anomalies = []
                result["errors"].append(f"Anomaly detection error: {str(e)}")
            
            result["anomalies"] = anomalies
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(f"Pipeline error: {str(e)}")
            import traceback
            result["metadata"]["error_traceback"] = traceback.format_exc()
        
        return result
    
    def _extract_text_from_tables(self, extracted_tables: List[Dict[str, Any]]) -> str:
        """
        Extract text from tables for entity extraction
        
        Converts table data to text format that can be parsed for entities
        """
        table_texts = []
        for table in extracted_tables:
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
