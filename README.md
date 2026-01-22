# CertiFi AI Pipeline

Pipeline modulare per l'estrazione di informazioni da documenti (fatture, diplomi, documenti di identità, ecc.)

## 🎯 Filosofia

**Non un'AI che fa tutto, ma una pipeline modulare** dove ogni componente è sostituibile.

```
Upload → Pre-processing → Classification → Text Extraction → Information Extraction → Normalization → Validation
```

## 📁 Struttura

```
certifi-ai/
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py      # Normalizzazione testo
│   ├── ocr.py               # Estrazione testo (PDF/immagini)
│   ├── classifier.py        # Classificazione documenti (regole + LLM)
│   ├── extractor.py         # Estrazione informazioni (regex + LLM)
│   ├── validators.py        # Validazione dati estratti
│   ├── orchestrator.py      # Orchestratore principale
│   └── schemas/
│       ├── base.py
│       ├── invoice.py
│       ├── diploma.py
│       └── id_document.py
├── main.py                  # Esempio di utilizzo
├── requirements.txt
└── README.md
```

## 🚀 Installazione

1. **Installa dipendenze:**
```bash
pip install -r requirements.txt
```

2. **Installa Tesseract OCR:**
```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ita

# Windows
# Scarica da: https://github.com/UB-Mannheim/tesseract/wiki
```

3. **Configura variabili d'ambiente (opzionale, per LLM):**
```bash
# Copia il file di esempio
cp .env.example .env

# Modifica .env con le tue chiavi API
# oppure esporta direttamente:
export OPENAI_API_KEY="your-key-here"
export USE_LLM="true"  # Per abilitare LLM
```

4. **Testa l'installazione:**
```bash
python test_pipeline.py
```

## 💻 Utilizzo

### Esempio base

```python
from pipeline.orchestrator import DocumentPipeline

# Inizializza pipeline
pipeline = DocumentPipeline(use_llm=False)

# Processa documento
result = pipeline.process("path/to/document.pdf")

# Risultato
print(result['document_type'])  # 'invoice', 'diploma', 'id', etc.
print(result['data'])           # Schema Pydantic con dati estratti
print(result['validation'])     # Risultato validazione
```

### Con LLM (opzionale)

```python
pipeline = DocumentPipeline(use_llm=True, llm_provider="openai")
result = pipeline.process("document.pdf")
```

## 📋 Tipi di Documenti Supportati

### 1. Fatture (Invoice)
- Numero fattura
- Data
- Venditore/Cliente
- Importi (totale, IVA, netto)
- Partita IVA

### 2. Diplomi (Diploma)
- Nome studente
- Università
- Tipo laurea
- CFU
- Data conseguimento
- Voto finale

### 3. Documenti Identità (ID)
- Nome/Cognome
- Data di nascita
- Codice fiscale
- Numero documento
- Indirizzo

## 🔧 Componenti

### 1. Text Extraction (`ocr.py`)
- PDF con testo: `pdfplumber`, `PyMuPDF`
- PDF scannerizzati: OCR con `pytesseract`
- Immagini: OCR con preprocessing

### 2. Classification (`classifier.py`)
- **Livello 1**: Regole + keyword matching (80% dei casi)
- **Livello 2**: LLM fallback (quando incerto)

### 3. Information Extraction (`extractor.py`)
- **Livello 1**: Regex patterns (veloce, robusto)
- **Livello 2**: LLM extraction (quando regex fallisce)

### 4. Validation (`validators.py`)
- Validazione campi obbligatori
- Controllo coerenza dati
- Confidence scoring

## 🎯 Strategia di Sviluppo

### ✅ Fase 1 (Settimana 1)
- [x] OCR + PDF extraction
- [x] Dump testo pulito

### ✅ Fase 2 (Settimana 2)
- [x] Estrazione rule-based
- [x] JSON schema (Pydantic)

### 🔄 Fase 3 (Settimana 3)
- [ ] Fallback LLM
- [ ] Validazione output avanzata

### 📅 Fase 4 (Settimana 4)
- [ ] Confidence score migliorato
- [ ] Integrazione CertiFi on-chain

## 🔐 Hash Canonico

Per CertiFi, ogni documento genera un hash canonico basato sui dati strutturati:

```python
result['metadata']['canonical_hash']  # SHA256 del JSON canonico
```

Questo hash può essere usato per:
- Verifica on-chain
- Deduplicazione
- Integrità dati

## ⚠️ Note Importanti

1. **Non partire da ML avanzato**: Inizia con regole, poi aggiungi LLM
2. **Un tipo documento alla volta**: Diventa eccellente su un dominio prima di generalizzare
3. **Pipeline > Modello**: Ogni blocco è sostituibile (leva tecnica)
4. **Validazione critica**: Per CertiFi, i dati devono essere validati prima dell'on-chain

## 🛠️ Estendere la Pipeline

### Aggiungere un nuovo tipo documento

1. Crea schema in `schemas/`:
```python
class NewDocumentSchema(BaseDocumentSchema):
    document_type: str = Field(default="new_type", const=True)
    field1: Optional[str] = None
    # ...
```

2. Aggiungi pattern in `classifier.py`:
```python
'new_type': {
    'keywords': ['keyword1', 'keyword2'],
    'patterns': [r'pattern1', r'pattern2'],
    'min_matches': 2
}
```

3. Aggiungi estrazione in `extractor.py`:
```python
def _extract_new_type(self, text: str) -> NewDocumentSchema:
    # Regex patterns
    # ...
```

## 📝 Esempi

Vedi `main.py` per un esempio completo di utilizzo.

## 🔗 Integrazione CertiFi

La pipeline è progettata per integrarsi con CertiFi:

1. **Estrazione** → Dati strutturati
2. **Validazione** → Controllo qualità
3. **Hash canonico** → Per on-chain storage
4. **Schema standardizzato** → Per smart contracts

## 📄 Licenza

[Da definire]
