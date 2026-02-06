# CertiFi AI API Documentation

## Overview

The CertiFi AI API provides a RESTful interface for document analysis and certification. The API processes documents through a modular pipeline:

```
OCR → Layout → Vision → LLM → Normalizer → JSON schema
```

## Endpoints

### POST `/analyze`

Analyzes a document and returns structured information including classification, claims, holder information, compliance score, and anomalies.

#### Request

**Content-Type:** `multipart/form-data`

**Parameters:**

- `document_id` (required): Unique document identifier
- `hash` (required): SHA256 hash of the document
- `requested_tasks` (optional): Comma-separated list of tasks to perform. Default: `"classify,extract,claims"`
  - `classify`: Document family and type classification
  - `extract`: Extract structured information from document
  - `claims`: Extract certifiable claims
  - `holder`: Extract holder information (relationship reference)
  - `compliance_score`: Calculate compliance score
- `ai_version` (optional): AI version identifier. Default: `"v1.0"`
- `file` (required): Document file (PDF, image, etc.)

#### Example Request

```bash
curl -X POST "http://certifi-ai/analyze" \
  -F "document_id=doc_123" \
  -F "hash=sha256:abc123..." \
  -F "requested_tasks=classify,extract,claims,holder,compliance_score" \
  -F "ai_version=v1.0" \
  -F "file=@document.pdf"
```

#### Example Response

```json
{
  "document_family": "contract",
  "document_type": "engagement_letter",
  "holder": {
    "type": "relationship",
    "ref": "rel:sha256(abc123def456...)",
    "confidence": 0.93
  },
  "claims": {
    "is_contractor": true,
    "amount": 3000.0,
    "currency": "USD",
    "subject": "EXAMPLE COMPANY",
    "entity": "Franco",
    "start_date": "2026-01-21T00:00:00",
    "end_date": null
  },
  "compliance_score": 0.91,
  "anomalies": []
}
```

#### Response Fields

- `document_family`: Document family classification (`identity`, `contract`, `certificate`, `financial`, `corporate`, `driving_license`, `unknown`)
- `document_type`: Document subtype (e.g., `engagement_letter`, `statement_of_work`, `id_card`, `passport`)
- `holder` (optional): Holder information
  - `type`: Type of holder (`relationship`, `individual`, `entity`)
  - `ref`: Reference hash or identifier
  - `confidence`: Confidence score (0.0 to 1.0)
- `claims` (optional): Extracted claims
  - `is_contractor`: Whether the claim indicates a contractor relationship
  - `amount`: Monetary amount (if applicable)
  - `currency`: Currency code (e.g., `USD`, `EUR`)
  - `subject`: Subject of the claim (contractor/individual)
  - `entity`: Entity (client/company)
  - `start_date`: Start date (ISO format)
  - `end_date`: End date (ISO format, if applicable)
- `compliance_score` (optional): Compliance score (0.0 to 1.0)
- `anomalies`: List of detected anomalies (always present)

### GET `/health`

Health check endpoint.

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Pipeline Stages

### 1. OCR (Optical Character Recognition)
Extracts text from PDFs and images using `pdfplumber`, `PyMuPDF`, and `pytesseract`.

### 2. Layout Analysis
Analyzes document structure:
- Detects sections (header, body, footer)
- Identifies structured fields (numbered lists, form fields, MRZ)
- Determines layout type (form, table, unstructured)

### 3. Vision Analysis
Analyzes document images:
- Detects orientation
- Assesses image quality
- Detects signatures and stamps
- Identifies text regions

### 4. LLM Processing (Optional)
Uses Large Language Models for advanced classification and extraction when enabled.

### 5. Normalization
Normalizes extracted data:
- Text normalization (whitespace, OCR errors)
- Date unification
- Field standardization

### 6. JSON Schema
Outputs structured JSON following Pydantic schemas.

## Task-Based Processing

The API supports task-based processing, allowing you to request only the analysis you need:

- **Minimal**: `classify` - Only classification
- **Standard**: `classify,extract,claims` - Classification, extraction, and claims
- **Full**: `classify,extract,claims,holder,compliance_score` - All available tasks

## Error Handling

The API returns HTTP status codes:

- `200`: Success
- `400`: Bad request (missing required parameters)
- `500`: Internal server error

Errors are also included in the `anomalies` field of the response.

## Running the API

### Development

```bash
python api.py
```

### Production

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Example Usage

See `example_api_usage.py` for Python examples.

## Notes

- The API accepts PDFs and images (PNG, JPG, TIFF, etc.)
- Large files may take longer to process
- The `hash` parameter should be the SHA256 hash of the document file
- The `document_id` should be unique per document
