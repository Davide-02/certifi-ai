#!/bin/bash

# Script per avviare l'API CertiFi AI

set -e

# Colori
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Avvio CertiFi AI API${NC}"
echo "===================="

# Verifica quale venv esiste (.venv ha priorità su venv)
if [ -d ".venv" ]; then
    VENV_PATH=".venv"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
else
    echo -e "${RED}❌ Ambiente virtuale non trovato!${NC}"
    echo "Esegui prima: ./setup.sh"
    exit 1
fi

# Attiva ambiente virtuale
echo -e "\n${YELLOW}Attivazione ambiente virtuale (${VENV_PATH})...${NC}"
source ${VENV_PATH}/bin/activate

# Verifica che le dipendenze siano installate
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}❌ FastAPI non installato!${NC}"
    echo "Esegui: pip install -r requirements.txt"
    exit 1
fi

# Avvia l'API
echo -e "\n${GREEN}✓ Avvio API su http://0.0.0.0:8000${NC}"
echo -e "${YELLOW}Documentazione API: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}Health check: http://localhost:8000/health${NC}"
echo -e "\nPremi Ctrl+C per fermare l'API\n"

python api.py
