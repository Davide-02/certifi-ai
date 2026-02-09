"""
CertiFi AI External API

POST /analyze endpoint for document analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import tempfile
import os
import json
import traceback
from pathlib import Path

from pipeline.new_orchestrator import ProductionPipeline

app = FastAPI(title="CertiFi AI API", version="1.0.0")


class HolderInfo(BaseModel):
    """Holder information"""
    type: str = Field(..., description="Type: 'relationship', 'individual', 'entity'")
    ref: Optional[str] = Field(None, description="Reference hash or identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ClaimsInfo(BaseModel):
    """Claims information"""
    is_contractor: Optional[bool] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    secondary_currency: Optional[str] = None  # Secondary currency (e.g., USD equivalent)
    secondary_amount: Optional[float] = None  # Secondary amount
    subject: Optional[str] = None
    entity: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class IntentInfo(BaseModel):
    """Intent analysis information"""
    primary_intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    trust_role: str
    risk_profile: str
    confidence: float
    explanation: Optional[str] = None


class ProcessContextInfo(BaseModel):
    """Process context information"""
    business_process: str
    compliance_domain: str
    lifecycle_stage: str
    recommended_actions: List[str] = Field(default_factory=list)
    confidence: float
    explanation: Optional[str] = None


class SemanticRoleInfo(BaseModel):
    """Semantic Role information"""
    role_type: str
    entity_type: str  # "individual" or "organization"
    name: str
    confidence: float
    evidence: str


class RolesInfo(BaseModel):
    """Roles extraction result"""
    roles: List[Dict[str, Any]]  # Flexible structure
    primary_holder: Optional[Dict[str, Any]] = None
    confidence: float


class AnalyzeResponse(BaseModel):
    """Response payload for /analyze endpoint"""
    model_config = ConfigDict(extra="allow")  # Permetti campi extra per flessibilità
    
    document_family: str
    document_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    holder: Optional[HolderInfo] = None
    claims: Optional[ClaimsInfo] = None
    compliance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    intent: Optional[IntentInfo] = None  # NEW: Intent & Ontology Layer
    process_context: Optional[ProcessContextInfo] = None  # NEW: Process Context Layer
    roles: Optional[RolesInfo] = None  # NEW: Deterministic Roles Extraction
    anomalies: List[str] = Field(default_factory=list)


@app.post("/analyze")
async def analyze_document(
    document_id: str = Form(...),
    hash: str = Form(default=""),  # Hash parameter (ignored, not used)
    requested_tasks: str = Form(default="classify,extract,claims"),
    ai_version: str = Form(default="v1.0"),
    file: UploadFile = File(...)
):
    """
    Analyze a document through the CertiFi AI pipeline
    
    Pipeline stages:
    OCR → Layout → Vision → LLM → Normalizer → JSON schema
    
    Args:
        document_id: Unique document identifier
        document_hash: SHA256 hash of the document (sent as 'hash' in form data)
        requested_tasks: Comma-separated list of tasks (classify, extract, claims, holder, roles, compliance_score, intent, process_context)
        ai_version: AI version identifier
        file: Document file (PDF, image, etc.)
    
    Returns:
        Analysis result with document family, type, holder, claims, compliance score, and anomalies
    """
    try:
        # Parse requested tasks
        tasks = [t.strip() for t in requested_tasks.split(",")]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Initialize pipeline (using new ProductionPipeline with all 6 layers)
            pipeline = ProductionPipeline(use_llm="llm" in tasks or "vision" in tasks)
            
            # Process document with requested tasks
            # Automatically includes intent and process_context if "classify" is requested
            if "classify" in tasks:
                # Ensure intent and process_context are included
                if "intent" not in tasks:
                    tasks.append("intent")
                if "process_context" not in tasks:
                    tasks.append("process_context")
            
            result = pipeline.process(
                file_path=tmp_path,
                requested_tasks=tasks
            )
            
            # Build response based on requested tasks
            # Map document_subtype to document_type (e.g., "engagement_letter" from subtype)
            document_subtype = result.get("document_subtype")
            if document_subtype:
                document_type = document_subtype
            else:
                # Fallback to family if no subtype
                document_type = result.get("document_family", "unknown")
            
            # STRUCTURED JSON RESPONSE - Build exact structure as requested
            perception_meta = result.get("metadata", {}).get("perception", {})
            text_length = perception_meta.get("text_length", 0)
            has_tables = perception_meta.get("has_tables", False)
            detected_language = perception_meta.get("detected_language", "unknown")
            
            # Extract fields that were successfully extracted
            extracted_fields = []
            claims_data = _extract_claims(result)
            if claims_data.get("amount") is not None:
                extracted_fields.append("amount")
            if claims_data.get("currency"):
                extracted_fields.append("currency")
            if claims_data.get("subject"):
                extracted_fields.append("subject")
            if claims_data.get("entity"):
                extracted_fields.append("entity")
            if claims_data.get("start_date"):
                extracted_fields.append("start_date")
            if claims_data.get("end_date"):
                extracted_fields.append("end_date")
            
            # Build structured response
            response_data = {
                "document_id": document_id,
                "file_name": file.filename or "unknown",
                "document_family": result.get("document_family", "unknown"),
                "document_type": document_type,
            }
            
            # Extract holder with required structure { type, name, role_type, confidence }
            holder_data = None
            if "holder" in tasks or "roles" in tasks:
                # Try to get holder from roles extraction first
                roles_data = result.get("roles", {})
                primary_holder = roles_data.get("primary_holder") if isinstance(roles_data, dict) else None
                
                if primary_holder:
                    holder_data = {
                        "type": primary_holder.get("entity_type", "unknown"),
                        "name": primary_holder.get("name", ""),
                        "role_type": primary_holder.get("role_type", "unknown"),
                        "confidence": primary_holder.get("confidence", 0.0)
                    }
                elif result.get("holder"):
                    # Fallback to legacy holder extraction
                    legacy_holder = result.get("holder")
                    holder_data = {
                        "type": legacy_holder.get("type", "unknown"),
                        "name": legacy_holder.get("name", ""),
                        "role_type": legacy_holder.get("role", "unknown"),
                        "confidence": legacy_holder.get("confidence", 0.0)
                    }
            
            response_data["holder"] = holder_data
            
            # Extract roles with structure: { roles: [...], primary_holder: {...}, confidence: 0.95 }
            roles_structure = {"roles": [], "primary_holder": None, "confidence": 0.0}
            if "roles" in tasks or "holder" in tasks:
                roles_data = result.get("roles", {})
                if isinstance(roles_data, dict) and roles_data.get("roles"):
                    roles_list = roles_data["roles"]
                    primary_holder_info = roles_data.get("primary_holder")
                    
                    # Build roles array
                    roles_array = []
                    for role in roles_list:
                        role_obj = {
                            "role_type": role.get("role_type", "unknown"),
                            "entity_type": role.get("entity_type", "unknown"),
                            "name": role.get("name", ""),
                            "contact_info": {
                                "email": role.get("contact_info", {}).get("email") if isinstance(role.get("contact_info"), dict) else None,
                                "phone": role.get("contact_info", {}).get("phone") if isinstance(role.get("contact_info"), dict) else None
                            },
                            "confidence": role.get("confidence", 0.0)
                        }
                        roles_array.append(role_obj)
                    
                    # Extract primary_holder (from roles_data or from holder_data)
                    primary_holder_obj = None
                    if primary_holder_info:
                        primary_holder_obj = {
                            "role_type": primary_holder_info.get("role_type", "unknown"),
                            "entity_type": primary_holder_info.get("entity_type", "unknown"),
                            "name": primary_holder_info.get("name", ""),
                            "confidence": primary_holder_info.get("confidence", 0.0)
                        }
                    elif holder_data:
                        # Fallback to holder_data if primary_holder not in roles_data
                        primary_holder_obj = {
                            "role_type": holder_data.get("role_type", "unknown"),
                            "entity_type": holder_data.get("type", "unknown"),
                            "name": holder_data.get("name", ""),
                            "confidence": holder_data.get("confidence", 0.0)
                        }
                    
                    # Get overall confidence from roles_data or calculate from roles
                    overall_confidence = roles_data.get("confidence", 0.0)
                    if overall_confidence == 0.0 and roles_array:
                        # Calculate average confidence if not provided
                        confidences = [r.get("confidence", 0.0) for r in roles_array]
                        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    roles_structure = {
                        "roles": roles_array,
                        "primary_holder": primary_holder_obj,
                        "confidence": overall_confidence
                    }
            
            response_data["roles"] = roles_structure
            
            # Extract claims as single object: { claim_type, is_contractor, amount, currency, subject, entity, start_date, end_date }
            claims_object = None
            if "claims" in tasks:
                claims_info = _extract_claims(result)
                # Determine claim_type based on document type
                claim_type = "service_agreement"
                if result.get("document_subtype") == "resume":
                    claim_type = "resume"
                elif result.get("document_family") == "invoice":
                    claim_type = "invoice"
                elif result.get("document_family") == "certificate":
                    claim_type = "certificate"
                
                # Always create claims object if claims task was requested
                claims_object = {
                    "claim_type": claim_type,
                    "is_contractor": claims_info.get("is_contractor"),
                    "amount": claims_info.get("amount"),
                    "currency": claims_info.get("currency"),
                    "subject": claims_info.get("subject"),
                    "entity": claims_info.get("entity"),
                    "start_date": claims_info.get("start_date"),
                    "end_date": claims_info.get("end_date")
                }
            
            response_data["claims"] = claims_object
            
            # Compliance score
            response_data["compliance_score"] = result.get("compliance_score") if "compliance_score" in tasks else None
            
            # Risk profile (extract from intent)
            risk_profile = "medium"
            if "intent" in tasks and result.get("intent"):
                risk_profile = result.get("intent", {}).get("risk_profile", "medium")
            response_data["risk_profile"] = risk_profile
            
            # Intent: { primary_intent, secondary_intents, trust_role, confidence, risk_profile }
            intent_data = None
            if "intent" in tasks and result.get("intent"):
                intent_raw = result.get("intent", {})
                intent_data = {
                    "primary_intent": intent_raw.get("primary_intent", "unknown"),
                    "secondary_intents": intent_raw.get("secondary_intents", []),
                    "trust_role": intent_raw.get("trust_role", "unknown"),
                    "confidence": intent_raw.get("confidence", 0.0),
                    "risk_profile": intent_raw.get("risk_profile", "medium")
                }
            response_data["intent"] = intent_data
            
            # Process context: { business_process, compliance_domain, lifecycle_stage, recommended_actions, confidence }
            process_context_data = None
            if "process_context" in tasks and result.get("process_context"):
                process_raw = result.get("process_context", {})
                process_context_data = {
                    "business_process": process_raw.get("business_process", "unknown"),
                    "compliance_domain": process_raw.get("compliance_domain", "unknown"),
                    "lifecycle_stage": process_raw.get("lifecycle_stage", "creation"),
                    "recommended_actions": process_raw.get("recommended_actions", []),
                    "confidence": process_raw.get("confidence", 0.0)
                }
            response_data["process_context"] = process_context_data
            
            # Metadata: { text_length, has_tables, detected_language, extracted_fields }
            response_data["metadata"] = {
                "text_length": text_length,
                "has_tables": has_tables,
                "detected_language": detected_language,
                "extracted_fields": extracted_fields
            }
            
            # Log response in console
            print("\n" + "="*80)
            print("📤 RISPOSTA API INVIATA")
            print("="*80)
            print(json.dumps(response_data, indent=2, default=str))
            print("="*80 + "\n")
            
            # Validazione flessibile della risposta
            try:
                # Crea l'oggetto response per validazione
                response_obj = AnalyzeResponse(**response_data)
                # Converti in dict includendo tutti i campi (anche None)
                response_dict = response_obj.model_dump(exclude_none=False, mode='json')
                # Assicurati che roles sia incluso anche se None o vuoto
                if "roles" in response_data:
                    response_dict["roles"] = response_data["roles"]
                return JSONResponse(
                    status_code=200,
                    content=response_dict
                )
            except Exception as validation_error:
                # Se la validazione fallisce, restituisci comunque i dati con campi opzionali
                error_response = {
                    **response_data,
                    "_validation_warning": str(validation_error)
                }
                print("\n" + "="*80)
                print("⚠️  RISPOSTA CON WARNING DI VALIDAZIONE")
                print("="*80)
                print(json.dumps(error_response, indent=2, default=str))
                print("="*80 + "\n")
                return JSONResponse(
                    status_code=200,
                    content=error_response
                )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=error_detail)


def _extract_claims(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract claims information from pipeline result
    
    Always returns a dict (never None) so client can see which fields were extracted
    """
    claim = result.get("claim")
    
    # If claim is None or not a dict, return empty structure
    if claim is None or not isinstance(claim, dict):
        return {
            "is_contractor": None,
            "amount": None,
            "currency": None,
            "subject": None,
            "entity": None,
            "start_date": None,
            "end_date": None,
            "secondary_currency": None,
            "secondary_amount": None,
        }
    
    # Extract claims_info from claim dict (even if empty)
    claims_info = {
        "is_contractor": claim.get("role") == "contractor" if claim.get("role") else None,
        "amount": claim.get("amount"),
        "currency": claim.get("currency"),
        "subject": claim.get("subject"),
        "entity": claim.get("entity"),
        "start_date": claim.get("start_date").isoformat() if claim.get("start_date") else None,
        "end_date": claim.get("end_date").isoformat() if claim.get("end_date") else None,
        "secondary_currency": claim.get("secondary_currency"),
        "secondary_amount": claim.get("secondary_amount"),
    }
    
    return claims_info


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
