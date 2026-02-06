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
from pathlib import Path

from pipeline.orchestrator import DocumentPipeline

app = FastAPI(title="CertiFi AI API", version="1.0.0")


class AnalyzeRequest(BaseModel):
    """Request payload for /analyze endpoint"""
    document_id: str = Field(..., description="Unique document identifier")
    hash: str = Field(..., description="SHA256 hash of the document")
    requested_tasks: List[str] = Field(
        default=["classify", "extract", "claims"],
        description="List of tasks to perform: classify, extract, claims, holder, compliance_score"
    )
    ai_version: str = Field(default="v1.0", description="AI version identifier")


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


class AnalyzeResponse(BaseModel):
    """Response payload for /analyze endpoint"""
    model_config = ConfigDict(extra="allow")  # Permetti campi extra per flessibilità
    
    document_family: str
    document_type: Optional[str] = None
    holder: Optional[HolderInfo] = None
    claims: Optional[ClaimsInfo] = None
    compliance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    anomalies: List[str] = Field(default_factory=list)


@app.post("/analyze")
async def analyze_document(
    document_id: str = Form(...),
    hash: str = Form(...),  # Nome del campo form, non conflitto con Python built-in qui
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
        requested_tasks: Comma-separated list of tasks (classify, extract, claims, holder, compliance_score)
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
            # Initialize pipeline
            pipeline = DocumentPipeline(use_llm="llm" in tasks or "vision" in tasks)
            
            # Process document with requested tasks
            result = pipeline.process(
                file_path=tmp_path,
                certification_profile=None,
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
            
            response_data = {
                "document_family": result.get("document_family", "unknown"),
                "document_type": document_type,
            }
            
            # Extract holder information if requested (already extracted by pipeline)
            if "holder" in tasks and result.get("holder"):
                response_data["holder"] = result["holder"]
            
            # Extract claims if requested (already extracted by pipeline)
            if "claims" in tasks:
                claims_info = _extract_claims(result)
                # Always include claims_info (even if empty/None fields) so client can see what was extracted
                response_data["claims"] = claims_info
            
            # Calculate compliance score if requested (already calculated by pipeline)
            if "compliance_score" in tasks and result.get("compliance_score") is not None:
                response_data["compliance_score"] = result["compliance_score"]
            
            # Detect anomalies (always included, already detected by pipeline)
            if result.get("anomalies"):
                response_data["anomalies"] = result["anomalies"]
            else:
                response_data["anomalies"] = []
            
            # Validazione flessibile della risposta
            try:
                return AnalyzeResponse(**response_data)
            except Exception as validation_error:
                # Se la validazione fallisce, restituisci comunque i dati con campi opzionali
                return JSONResponse(
                    status_code=200,
                    content={
                        **response_data,
                        "_validation_warning": str(validation_error)
                    }
                )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=error_detail)


def _extract_holder(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract holder information from pipeline result"""
    claim = result.get("claim", {})
    role = result.get("inferred_role", "unknown")
    
    if not claim or role == "unknown":
        return None
    
    # Determine holder type based on role
    holder_type = "relationship"
    if role in ["contractor", "employee", "student"]:
        holder_type = "individual"
    elif role in ["client", "company"]:
        holder_type = "entity"
    
    # Generate reference hash from claim
    import hashlib
    import json
    claim_ref = json.dumps(claim, default=str, sort_keys=True)
    ref_hash = hashlib.sha256(claim_ref.encode()).hexdigest()
    
    # Calculate confidence from claim confidence
    confidence = claim.get("confidence", 0.0)
    
    return {
        "type": holder_type,
        "ref": f"rel:sha256({ref_hash[:16]}...)",
        "confidence": confidence
    }


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


def _calculate_compliance_score(result: Dict[str, Any]) -> float:
    """Calculate compliance score from pipeline result"""
    # Base score from certification readiness
    base_score = 0.0
    
    if result.get("certification_ready", False):
        base_score = 0.8
    elif result.get("human_review_required", True):
        base_score = 0.5
    
    # Adjust based on confidence
    confidence = result.get("metadata", {}).get("decision", {}).get("confidence", 0.0)
    if confidence > 0:
        base_score = (base_score + confidence) / 2
    
    # Adjust based on risk level
    risk_level = result.get("risk_level", "high")
    risk_multiplier = {
        "low": 1.0,
        "medium": 0.9,
        "high": 0.7
    }.get(risk_level.lower(), 0.7)
    
    final_score = base_score * risk_multiplier
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, final_score))


def _detect_anomalies(result: Dict[str, Any]) -> List[str]:
    """Detect anomalies in the analysis result"""
    anomalies = []
    
    # Check for errors
    errors = result.get("errors", [])
    if errors:
        anomalies.extend([f"Error: {e}" for e in errors])
    
    # Check for low confidence
    confidence = result.get("metadata", {}).get("decision", {}).get("confidence", 0.0)
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
    
    return anomalies


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
