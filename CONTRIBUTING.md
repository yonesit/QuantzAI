# Contributing

## Branch-Naming
- `feature/beschreibung` - neue Funktionen
- `fix/beschreibung`     - Bugfixes
- `data/beschreibung`    - Daten-Pipeline-Aenderungen

## Commit-Konventionen
- `feat: kurze Beschreibung`
- `fix: kurze Beschreibung`
- `data: kurze Beschreibung`
- `test: kurze Beschreibung`
- `docs: kurze Beschreibung`

## Vor jedem Commit
```bash
ruff check .
black .
pytest tests/
```
