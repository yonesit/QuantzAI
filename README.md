# QuantzAI

Automatisiertes ML-basiertes Trading-System fuer MetaTrader 5.

## Setup

```bash
# 1. Repository klonen
git clone https://github.com/itgnf/QuantzAI.git
cd QuantzAI

# 2. Virtuelle Umgebung erstellen
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Abhaengigkeiten installieren
pip install -e ".[dev]"

# 4. Umgebungsvariablen konfigurieren
cp .env.example .env
# .env mit eigenen Werten befuellen

# 5. Tests ausfuehren
pytest tests/
```

## Projektstruktur

```
src/data/        - Datenbeschaffung & Qualitaet
src/models/      - ML-Modelle & FreqAI
src/execution/   - MT5-Bridge & Orders
src/risk/        - Risikomanagement
src/monitoring/  - Drift-Erkennung & Alerts
```

## Development

Branches: `feature/`, `fix/`, `data/`
Commits: `feat:`, `fix:`, `data:`, `test:`, `docs:`
