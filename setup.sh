#!/bin/bash

# CertiFi AI - Setup Script
# Crea ambiente virtuale, installa dipendenze e configura il sistema

set -e  # Exit on error

echo "🚀 CertiFi AI - Setup"
echo "===================="

# Colori per output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Creare ambiente virtuale
echo -e "\n${YELLOW}Step 1: Creazione ambiente virtuale...${NC}"
if [ -d "venv" ]; then
    echo "⚠️  Ambiente virtuale 'venv' già esistente. Vuoi ricrearlo? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Ambiente virtuale ricreato${NC}"
    else
        echo "✓ Usando ambiente virtuale esistente"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Ambiente virtuale creato${NC}"
fi

# Step 2: Attivare ambiente virtuale
echo -e "\n${YELLOW}Step 2: Attivazione ambiente virtuale...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Ambiente virtuale attivato${NC}"

# Step 3: Aggiornare pip
echo -e "\n${YELLOW}Step 3: Aggiornamento pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip aggiornato${NC}"

# Step 4: Installare dipendenze
echo -e "\n${YELLOW}Step 4: Installazione dipendenze...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dipendenze installate${NC}"

# Step 5: Download modello spaCy (opzionale ma consigliato)
echo -e "\n${YELLOW}Step 5: Download modello spaCy (opzionale)...${NC}"
echo "Vuoi scaricare il modello spaCy per NER? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    python -m spacy download en_core_web_sm || echo "⚠️  Errore nel download del modello spaCy (non critico)"
    echo -e "${GREEN}✓ Modello spaCy installato${NC}"
else
    echo "⏭️  Salto installazione modello spaCy"
fi

# Step 6: Verifica installazione
echo -e "\n${YELLOW}Step 6: Verifica installazione...${NC}"
python -c "import fastapi; import uvicorn; print('✓ FastAPI e Uvicorn OK')" || {
    echo "❌ Errore: FastAPI o Uvicorn non installati correttamente"
    exit 1
}

echo -e "\n${GREEN}✅ Setup completato!${NC}"
echo -e "\nPer avviare l'API, esegui:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  ${YELLOW}python api.py${NC}"
echo -e "\nOppure:"
echo -e "  ${YELLOW}uvicorn api:app --host 0.0.0.0 --port 8000${NC}"
