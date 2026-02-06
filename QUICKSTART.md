# Quick Start - CertiFi AI API

## Comandi Essenziali

### 1. Setup Iniziale (una volta)

```bash
# Crea ambiente virtuale
python3 -m venv venv

# Attiva ambiente virtuale
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Avvia API

```bash
# Assicurati che venv sia attivato
source venv/bin/activate

# Avvia API
python api.py
```

Oppure con uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 3. Test API

Apri nel browser: http://localhost:8000/docs

Oppure con curl:

```bash
curl http://localhost:8000/health
```

## Analizzare un documento da terminale

### Metodo 1: Script automatico (consigliato)

```bash
# Analizza un documento
./analyze_document.sh /path/to/document.pdf

# Con document_id personalizzato
./analyze_document.sh document.pdf doc_123

# Con tasks specifici
./analyze_document.sh document.pdf doc_123 classify,extract,claims
```

### Metodo 2: Curl diretto

```bash
# Calcola hash del file (macOS)
FILE_HASH=$(shasum -a 256 document.pdf | awk '{print $1}')

# Oppure (Linux)
FILE_HASH=$(sha256sum document.pdf | awk '{print $1}')

# Invia richiesta
curl -X POST "http://localhost:8000/analyze" \
  -F "document_id=doc_123" \
  -F "hash=sha256:$FILE_HASH" \
  -F "requested_tasks=classify,extract,claims,holder,compliance_score" \
  -F "ai_version=v1.0" \
  -F "file=@document.pdf" | python3 -m json.tool
```

### Metodo 3: Python script

```bash
python test_api.py /path/to/document.pdf
```

## Script Automatici

```bash
# Setup completo
./setup.sh

# Avvia API
./start_api.sh
```

## Troubleshooting

**Problema**: `python: command not found`
**Soluzione**: Usa `python3` invece di `python`

**Problema**: `ModuleNotFoundError: No module named 'fastapi'`
**Soluzione**: 
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Problema**: Porta 8000 già in uso
**Soluzione**: Cambia porta in `api.py` o usa:
```bash
uvicorn api:app --port 8001
```
