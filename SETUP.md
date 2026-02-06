# Setup CertiFi AI

Guida completa per installare e avviare l'API CertiFi AI.

## Prerequisiti

- Python 3.8 o superiore
- pip (gestore pacchetti Python)
- Sistema operativo: macOS, Linux, o Windows (con WSL)

## Installazione Rapida

### Opzione 1: Script Automatico (Consigliato)

```bash
# Rendi eseguibili gli script
chmod +x setup.sh start_api.sh

# Esegui setup
./setup.sh

# Avvia l'API
./start_api.sh
```

### Opzione 2: Manuale

#### 1. Crea ambiente virtuale

```bash
python3 -m venv venv
```

#### 2. Attiva ambiente virtuale

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

#### 3. Aggiorna pip

```bash
pip install --upgrade pip setuptools wheel
```

#### 4. Installa dipendenze

```bash
pip install -r requirements.txt
```

#### 5. (Opzionale) Installa modello spaCy per NER

```bash
python -m spacy download en_core_web_sm
```

## Avvio API

### Metodo 1: Script Python diretto

```bash
# Assicurati che venv sia attivato
source venv/bin/activate  # macOS/Linux
# oppure
venv\Scripts\activate     # Windows

# Avvia API
python api.py
```

### Metodo 2: Uvicorn (Consigliato per produzione)

```bash
# Assicurati che venv sia attivato
source venv/bin/activate

# Avvia con uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000

# Con reload automatico (sviluppo)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Metodo 3: Script bash

```bash
./start_api.sh
```

## Verifica Installazione

### Test rapido

```bash
# Attiva venv
source venv/bin/activate

# Verifica import
python -c "import fastapi; import uvicorn; print('OK')"
```

### Test API

Dopo aver avviato l'API, apri nel browser:

- **Documentazione interattiva**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Schema OpenAPI**: http://localhost:8000/openapi.json

## Risoluzione Problemi

### Errore: "python: command not found"

Usa `python3` invece di `python`:

```bash
python3 -m venv venv
python3 api.py
```

### Errore: "ModuleNotFoundError: No module named 'fastapi'"

Assicurati che:
1. L'ambiente virtuale sia attivato (`source venv/bin/activate`)
2. Le dipendenze siano installate (`pip install -r requirements.txt`)

### Errore: "Port 8000 already in use"

Cambia la porta:

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

Oppure modifica `api.py` alla fine del file:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Cambia porta
```

### Errore con Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Scarica e installa da: https://github.com/UB-Mannheim/tesseract/wiki

### Errore con OpenCV

Se `opencv-python` non si installa:

```bash
pip install opencv-python-headless
```

## Struttura Comandi Completa

```bash
# 1. Setup iniziale (una volta)
./setup.sh

# 2. Ogni volta che vuoi avviare l'API
./start_api.sh

# Oppure manualmente:
source venv/bin/activate
python api.py
```

## Variabili d'Ambiente (Opzionale)

Crea un file `.env` per configurazioni:

```bash
# .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
TESSERACT_CMD=/usr/local/bin/tesseract
```

## Test API

Dopo aver avviato l'API, testa con curl:

```bash
curl http://localhost:8000/health
```

Dovresti ricevere:
```json
{"status":"healthy","version":"1.0.0"}
```

## Prossimi Passi

- Leggi `API_README.md` per la documentazione completa dell'API
- Vedi `example_api_usage.py` per esempi di utilizzo
- Consulta `README.md` per la documentazione della pipeline
