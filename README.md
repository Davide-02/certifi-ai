# CertiFi AI Pipeline

**Claim Verification Engine** per CertiFi - Sistema modulare per la certificazione di documenti e affermazioni verificabili.

## 🎯 Filosofia

**CertiFi non certifica documenti, certifica CLAIMS (affermazioni verificabili).**

Il sistema è progettato come una **pipeline modulare** dove ogni componente è sostituibile, seguendo il principio: **sistemi > magia**.

```
Upload → OCR → Family Classification → Policy Resolution → Claim Extraction → Claim Evaluation → Certification Decision → Hash Generation
```

## 🏗️ Architettura a 3 Livelli

### Livello 1: Family Classification
Classifica documenti in **famiglie** (identity, contract, certificate, financial, corporate) per determinare la policy di certificazione.

### Livello 2: Policy Resolution
Mappa ogni famiglia a una **policy specifica** (hash_only, identity_minimal, claim_based, etc.).

### Livello 3: Claim Verification
Per documenti semantici (contratti, SOW, lettere), estrae e verifica **claim verificabili** invece di affidarsi solo alla struttura del documento.

## 📁 Struttura

```
certifi-ai/
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py          # Normalizzazione testo e immagini
│   ├── ocr.py                   # Estrazione testo (PDF/immagini)
│   ├── classifier.py            # Classificazione tipo documento (opzionale)
│   ├── family_classifier.py     # ⭐ Classificazione famiglia + subtype
│   ├── policy_resolver.py       # ⭐ Risoluzione policy di certificazione
│   ├── role_inference.py        # ⭐ Inferenza ruolo (contractor, employee, etc.)
│   ├── claim_extractor.py       # ⭐ Estrazione claim verificabili
│   ├── claim_evaluator.py       # ⭐ Valutazione certifiability claim
│   ├── extractor.py            # Estrazione informazioni strutturate
│   ├── mrz_parser.py           # Parsing MRZ per documenti identità
│   ├── decision_engine.py      # ⭐ Motore decisionale centralizzato
│   ├── validators.py           # Validazione dati estratti
│   ├── orchestrator.py         # Orchestratore principale
│   └── schemas/
│       ├── base.py
│       ├── invoice.py
│       ├── diploma.py
│       ├── id_document.py
│       └── driving_license.py
├── main.py                     # Esempio di utilizzo
├── example_usage.py            # Esempi avanzati
├── test_pipeline.py            # Test suite
├── requirements.txt
└── README.md
```

## 🚀 Installazione

### 1. Dipendenze Base

```bash
pip install -r requirements.txt
```

### 2. Tesseract OCR

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng

# Windows
# Scarica da: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. spaCy (Opzionale - per NER avanzato)

```bash
# Installa spaCy
pip install spacy

# Scarica modello inglese
python -m spacy download en_core_web_sm

# Il sistema funziona anche senza spaCy (usa regex fallback)
```

### 4. Configurazione LLM (Opzionale)

```bash
export OPENAI_API_KEY="your-key-here"
# oppure
export ANTHROPIC_API_KEY="your-key-here"
```

### 5. Test Installazione

```bash
python test_pipeline.py
```

## 💻 Utilizzo

### Esempio Base

```python
from pipeline.orchestrator import DocumentPipeline

# Inizializza pipeline
pipeline = DocumentPipeline(use_llm=False)

# Processa documento
result = pipeline.process("path/to/document.pdf")

# Risultato
print(result['document_family'])        # 'contract', 'identity', etc.
print(result['document_subtype'])       # 'engagement_letter', 'invoice', etc.
print(result['certification_ready'])     # True/False
print(result['claim'])                  # Claim estratto (se presente)
print(result['metadata']['canonical_hash'])  # Hash per on-chain
```

### Output Completo

```json
{
  "success": true,
  "document_family": "contract",
  "document_subtype": "engagement_letter",
  "certification_ready": true,
  "human_review_required": false,
  "certification_policy": "hash_only",
  "claim": {
    "subject": "EXAMPLE COMPANY",
    "role": "contractor",
    "entity": "Franco",
    "start_date": "2026-01-21",
    "amount": 3000.0,
    "currency": "USD",
    "services": "Pulizia caldaia"
  },
  "claim_statement": "EXAMPLE COMPANY is a contractor for Franco from 2026-01-21 (ongoing) (USD 3000.00)",
  "metadata": {
    "family_confidence": 0.85,
    "co_occurrence_boost": 0.25,
    "adaptive_confidence_boost": 0.10,
    "claims_confidence": 0.95,
    "canonical_hash": "...",
    "claim_hash": "..."
  }
}
```

## 🆕 Funzionalità Avanzate

### 1. Co-occorrenze Semantiche

Il sistema rileva **combinazioni di segnali** per classificazione più precisa:

- **Engagement Letter**: "engagement letter" + "service provider" + "fees" → boost +0.25
- **SOW**: "statement of work" + "client" + "contractor" → boost +0.20
- **Invoice**: "invoice" + "total" + "vat" → boost +0.20

```python
result['metadata']['co_occurrence_boost']  # Boost applicato
```

### 2. Classificazione Multi-Livello

Ogni documento è classificato in **2 livelli**:

- **Family**: Categoria generale (contract, identity, financial, etc.)
- **Subtype**: Tipo specifico (engagement_letter, invoice, id_card, etc.)

```python
result['document_family']   # 'contract'
result['document_subtype']  # 'engagement_letter'
```

### 3. Confidence Adattativa

La confidence si adatta automaticamente:

- **Boost da claim**: Se `claims_confidence >= 0.70`, boost fino a +0.15
- **Penalità campi mancanti**: Confidence ridotta se mancano campi critici
- **Co-occurrence boost**: Combinazioni semantiche aumentano confidence

```python
result['metadata']['adaptive_confidence_boost']  # Boost adattativo
```

### 4. NER (Named Entity Recognition)

Estrazione automatica con **spaCy** (opzionale):

- **PERSON**: Nomi di persone (client, contractor)
- **ORG**: Organizzazioni (aziende)
- **DATE**: Date (start, end, issue)
- **MONEY**: Importi e valute

```python
claim['extraction_method']  # 'ner' o 'regex'
```

### 5. Estrazione Semantica Avanzata

Pattern regex migliorati con contesto:

- Estrazione **servizi/scope of work**
- Estrazione **importi e valute** con pattern multipli
- Estrazione **date** in vari formati
- **Dependency parsing** per relazioni soggetto/oggetto (con spaCy)

## 📋 Tipi di Documenti Supportati

### 1. Documenti Identità (Identity)
- Carta d'identità
- Passaporto
- **Subtypes**: `id_card`, `passport`

**Policy**: `IDENTITY_MINIMAL` - Richiede MRZ o layout rules, confidence >= 0.85

### 2. Patenti di Guida (Driving License)
- Estrazione da pattern strutturati (1., 2., 3., 4a., 4b., 5.)
- **Subtypes**: `driving_license`

**Policy**: `DRIVING_LICENSE_MINIMAL` - Layout rules, confidence >= 0.85

### 3. Contratti e Documenti Legali (Contract)
- Engagement Letter
- Statement of Work (SOW)
- Independent Contractor Agreement
- Service Agreement
- NDA
- **Subtypes**: `engagement_letter`, `statement_of_work`, `independent_contractor_agreement`, `service_agreement`, `nda`, `contract_generic`

**Policy**: `HASH_ONLY` - Certificazione basata su claim, non su struttura

### 4. Certificati (Certificate)
- Diplomi
- Certificate of Engagement
- **Subtypes**: `diploma`, `certificate_of_engagement`, `certificate_generic`

**Policy**: `CERTIFICATE_MINIMAL` - Estrazione campi chiave

### 5. Documenti Finanziari (Financial)
- Fatture
- Buste paga
- Estratti conto
- **Subtypes**: `invoice`, `payslip`, `bank_statement`

**Policy**: `FINANCIAL_MINIMAL` - Estrazione importi e date

### 6. Documenti Aziendali (Corporate)
- Visure camerali
- Statuti
- Bilanci
- **Subtypes**: Vari

**Policy**: `CORPORATE_MINIMAL` - Estrazione dati aziendali

## 🔧 Componenti Principali

### 1. Family Classifier (`family_classifier.py`)

Classifica documenti in **famiglie** usando:
- **Keywords**: Parole chiave specifiche
- **Pattern strutturali**: Regex per layout
- **Co-occorrenze semantiche**: Combinazioni di segnali
- **Subtype classification**: Classificazione di secondo livello

```python
from pipeline.family_classifier import FamilyClassifier

classifier = FamilyClassifier()
result = classifier.classify(text)
# result['family'] = DocumentFamily.CONTRACT
# result['subtype'] = DocumentSubtype.ENGAGEMENT_LETTER
# result['co_occurrence_boost'] = 0.25
```

### 2. Policy Resolver (`policy_resolver.py`)

Mappa famiglie a **policy di certificazione**:

- `HASH_ONLY`: Solo hash del file (contratti)
- `IDENTITY_MINIMAL`: Estrazione campi minimi (ID)
- `CLAIM_BASED`: Certificazione basata su claim (documenti semantici)

```python
from pipeline.policy_resolver import PolicyResolver

resolver = PolicyResolver()
policy = resolver.resolve(family, confidence, use_claim_based=True)
```

### 3. Claim Extractor (`claim_extractor.py`)

Estrae **claim verificabili** da documenti:

- **Subject**: Chi (contractor, employee)
- **Role**: Ruolo (contractor, employee, student)
- **Entity**: Per chi (client, company)
- **Dates**: Quando (start, end)
- **Amount**: Importo e valuta
- **Services**: Servizi/scope of work

**Metodi di estrazione**:
- **NER (spaCy)**: Estrazione automatica entità
- **Regex avanzato**: Pattern con contesto

```python
from pipeline.claim_extractor import ClaimExtractor
from pipeline.role_inference import Role

extractor = ClaimExtractor()
claim = extractor.extract(text, Role.CONTRACTOR, 'contract')
# claim['subject'] = "EXAMPLE COMPANY"
# claim['entity'] = "Franco"
# claim['extraction_method'] = 'ner' o 'regex'
```

### 4. Claim Evaluator (`claim_evaluator.py`)

Valuta la **certifiability** dei claim:

- **Critical claims**: `has_client`, `has_contractor`
- **Supporting claims**: `references_master_agreement`, `defines_scope_of_work`
- **Confidence calculation**: Basata su evidenze hard/soft

```python
from pipeline.claim_evaluator import ClaimEvaluator

evaluator = ClaimEvaluator()
result = evaluator.evaluate(text, 'contract')
# result['certifiable'] = True
# result['claims_confidence'] = 0.95
# result['is_contractor_relationship'] = True
```

### 5. Decision Engine (`decision_engine.py`)

**Motore decisionale centralizzato** per certificazione:

- Calcola confidence finale
- Determina `certification_ready`
- Assegna `risk_level` (LOW, MEDIUM, HIGH)
- Seleziona `certification_profile`

```python
from pipeline.decision_engine import DecisionEngine, CertificationProfile

engine = DecisionEngine()
decision = engine.decide(
    document_family='contract',
    classification_confidence=0.85,
    field_confidence={'subject': 0.9, 'entity': 0.9},
    missing_fields=[],
    profile=CertificationProfile.CONTRACT_HASH_ONLY
)
```

### 6. Role Inference (`role_inference.py`)

Inferisce **ruoli** da pattern semantici:

- `CONTRACTOR`: Independent contractor, service provider
- `EMPLOYEE`: Employee, wage earner
- `STUDENT`: Student, enrolled
- `SUPPLIER`: Supplier, vendor
- `DIRECTOR`: Director, board member

```python
from pipeline.role_inference import RoleInferenceEngine

engine = RoleInferenceEngine()
result = engine.infer(text, 'contract')
# result['role'] = Role.CONTRACTOR
# result['confidence'] = 0.98
# result['evidence_type'] = 'hard'
```

## 🔐 Certificazione e Hash

### Hash Canonico

Ogni documento certificabile genera un **hash canonico**:

```python
result['metadata']['canonical_hash']  # SHA256 del JSON canonico
```

**Composizione hash**:
- Dati strutturati estratti (se presenti)
- Claim hash (se presente)
- File hash (per policy `HASH_ONLY`)

### Claim Hash

Per documenti semantici, viene generato anche un **claim hash**:

```python
result['metadata']['claim_hash']  # SHA256 del claim statement
```

### Certificazione Ready

Un documento è `certification_ready` se:

1. **Documenti strutturati** (ID, passport):
   - `family_confidence >= threshold` (es. 0.85)
   - Campi critici estratti
   - `trusted_source` presente (MRZ, layout_rules)

2. **Documenti semantici** (contract, SOW):
   - `claims_confidence >= 0.70` (o threshold policy)
   - `is_contractor_relationship = True`
   - Claim critici presenti

```python
result['certification_ready']      # True/False
result['human_review_required']    # True se confidence < 0.85
result['risk_level']              # 'low', 'medium', 'high'
```

## 🎯 Strategia di Sviluppo

### ✅ Fase 1: Foundation
- [x] OCR + PDF extraction
- [x] Preprocessing e normalizzazione
- [x] Schema Pydantic

### ✅ Fase 2: Classification
- [x] Family classification
- [x] Subtype classification
- [x] Policy resolution

### ✅ Fase 3: Claim Verification
- [x] Role inference
- [x] Claim extraction
- [x] Claim evaluation
- [x] Claim-based certification

### ✅ Fase 4: Advanced Features
- [x] Co-occurrence semantiche
- [x] Confidence adattativa
- [x] NER integration (spaCy)
- [x] Estrazione semantica avanzata
- [x] Multi-level classification

### 🔄 Fase 5: Future Enhancements
- [ ] Pattern learning da documenti certificati
- [ ] Dependency parsing avanzato
- [ ] Regex dinamici generati automaticamente
- [ ] LLM fine-tuning per estrazione

## 🛠️ Estendere la Pipeline

### Aggiungere un nuovo Document Family

1. **Aggiungi enum in `family_classifier.py`**:
```python
class DocumentFamily(str, Enum):
    NEW_FAMILY = "new_family"
```

2. **Aggiungi pattern**:
```python
FAMILY_PATTERNS = {
    DocumentFamily.NEW_FAMILY: {
        "keywords": ["keyword1", "keyword2"],
        "structural": [r"pattern1", r"pattern2"],
        "min_matches": 2,
    }
}
```

3. **Aggiungi policy in `policy_resolver.py`**:
```python
FAMILY_POLICIES = {
    DocumentFamily.NEW_FAMILY: {
        'default_policy': CertificationPolicy.NEW_POLICY,
        'certifiable': True,
        'requires_extraction': True,
        'min_confidence': 0.70,
    }
}
```

### Aggiungere un nuovo Subtype

1. **Aggiungi enum**:
```python
class DocumentSubtype(str, Enum):
    NEW_SUBTYPE = "new_subtype"
```

2. **Aggiungi logica in `_classify_subtype()`**:
```python
if family == DocumentFamily.RELEVANT_FAMILY:
    if re.search(r'pattern', text_lower):
        return DocumentSubtype.NEW_SUBTYPE
```

### Aggiungere Co-occurrence Pattern

```python
SEMANTIC_CO_OCCURRENCE = {
    DocumentFamily.NEW_FAMILY: [
        (
            r'pattern.*?(?:keyword1|keyword2)',
            ['keyword1', 'keyword2', 'keyword3'],
            0.25  # Boost score
        ),
    ]
}
```

## 📝 Esempi

### Esempio 1: Engagement Letter

```python
from pipeline.orchestrator import DocumentPipeline

pipeline = DocumentPipeline(use_llm=False)
result = pipeline.process("engagement_letter.pdf")

# Output
assert result['document_family'] == 'contract'
assert result['document_subtype'] == 'engagement_letter'
assert result['certification_ready'] == True
assert result['claim']['subject'] == "EXAMPLE COMPANY"
assert result['claim']['entity'] == "Franco"
```

### Esempio 2: ID Document

```python
result = pipeline.process("id_card.pdf")

# Output
assert result['document_family'] == 'identity'
assert result['document_subtype'] == 'id_card'
assert result['data']['first_name'] == "DAVIDE"
assert result['data']['trusted_source'] == "mrz"
```

### Esempio 3: Driving License

```python
result = pipeline.process("driving_license.pdf")

# Output
assert result['document_family'] == 'driving_license'
assert result['data']['license_number'] == "TA5418408X"
assert result['data']['trusted_source'] == "layout_rules"
```

## 🔗 Integrazione CertiFi

La pipeline è progettata per integrarsi con CertiFi on-chain:

1. **Estrazione** → Dati strutturati + Claim
2. **Validazione** → Controllo qualità e confidence
3. **Decision** → `certification_ready`, `risk_level`
4. **Hash Generation** → `canonical_hash` + `claim_hash`
5. **On-chain Storage** → Smart contract verification

### Flow Completo

```
Document → OCR → Family Classification → Policy Resolution
    ↓
Claim Extraction → Claim Evaluation → Decision Engine
    ↓
Certification Ready? → Hash Generation → On-chain
```

## ⚠️ Note Importanti

1. **Claim-based vs Structure-based**: 
   - Documenti semantici (contratti) usano **claim confidence**
   - Documenti strutturati (ID) usano **family confidence**

2. **Confidence Realistica**: 
   - Maximum confidence = 0.98 (mai 1.0)
   - Confidence adattativa basata su evidenze

3. **Human Review**: 
   - `human_review_required = True` se confidence < 0.85
   - Sempre per documenti ad alto rischio

4. **Pattern Learning**: 
   - Sistema estendibile con pattern da documenti certificati
   - Co-occurrence patterns migliorano nel tempo

## 📊 Metriche e Performance

### Confidence Scoring

- **Field-level confidence**: Per ogni campo estratto
- **Overall confidence**: Minimo ponderato dei campi
- **Claims confidence**: Basata su evidenze hard/soft
- **Family confidence**: Basata su keyword + pattern + co-occurrence

### Risk Levels

- **LOW**: Confidence >= 0.90, tutti i campi critici presenti
- **MEDIUM**: Confidence 0.70-0.90, alcuni campi opzionali mancanti
- **HIGH**: Confidence < 0.70, campi critici mancanti

## 🐛 Troubleshooting

### Problema: spaCy non trova entità

**Soluzione**: Il sistema usa automaticamente regex fallback. Per abilitare NER:
```bash
python -m spacy download en_core_web_sm
```

### Problema: Documento non classificato correttamente

**Soluzione**: 
1. Verifica pattern in `family_classifier.py`
2. Aggiungi co-occurrence pattern se necessario
3. Controlla `co_occurrence_boost` nell'output

### Problema: Claim non estratti

**Soluzione**:
1. Verifica pattern in `claim_extractor.py`
2. Abilita NER se disponibile
3. Controlla `extraction_method` nell'output

## 📄 Licenza

[Da definire]

## 🤝 Contribuire

1. Fork del repository
2. Crea branch per feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

## 📚 Riferimenti

- **CertiFi**: Sistema di certificazione on-chain
- **spaCy**: NLP library per NER
- **Pydantic**: Data validation
- **Tesseract**: OCR engine

---

**Built with ❤️ for CertiFi**
