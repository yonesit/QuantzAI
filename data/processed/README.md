# data/processed — Datensätze (Phase 1)

Dieses Verzeichnis enthält die aufbereiteten Marktdaten. **Parquet-Dateien sind
git-ignoriert** (`*.parquet`) und liegen nur lokal auf der VM — versioniert wird
ausschließlich die reproduzierbare Tooling-Pipeline (`scripts/`, `src/data/`).

## 1. Dukascopy-Langhistorie (Training/Labeling) — PRIMÄR

Lange, tick-abgeleitete M15-Historie inkl. **echtem historischem Bid-Ask-Spread**
pro Bar. Grundlage für das kostenbewusste Triple-Barrier-Labeling (Phase 1).

| Datei | Symbol | Zeitraum | Inhalt |
|-------|--------|----------|--------|
| `EURUSD_M15_<von>-<bis>.parquet` | EURUSD | ab ~2003 | OHLC (Mid) + Spread |
| `XAUUSD_M15_<von>-<bis>.parquet` | XAUUSD | ab ~2010 | OHLC (Mid) + Spread |

Spalten: `timestamp` (UTC, Bar-Open), `open/high/low/close` (Mid-Preis),
`volume`, `spread_mean`, `spread_median` (Preis-Einheiten), `tick_count`.

Erzeugen / fortsetzen (resume-fähig):
```
python scripts/fetch_dukascopy.py --symbol EURUSD
python scripts/fetch_dukascopy.py --symbol XAUUSD
```
Tages-Cache: `dukascopy_cache/<SYMBOL>/<YYYY-MM-DD>.parquet`. Ein Abbruch verwirft
nur den laufenden Tag; erneuter Start überspringt bereits geladene Tage.

## 2. Fusion-Markets / MT5-Referenz — RESERVIERT FÜR PHASE 6

Verzeichnis `fusion_ref/`. Enthält die **maximal bei Fusion Markets / MT5
verfügbare M15-Historie (~4 Jahre**, begrenzt durch das Terminal-Limit von
100.000 Bars/Chart; der Broker hält serverseitig ohnehin nichts vor ~2022 vor).

> ⚠️ **Dieser Datensatz ist NICHT für Training/Labeling gedacht** (zu kurz) und
> darf **nicht überschrieben oder gelöscht** werden. Er ist reserviert als
> **Broker-Abgleichsdatensatz für Phase 6 (Demo-Validierung):** reale
> Fusion-Fills / Bar-Preise gegen die Backtest-Annahmen abgleichen.

| Datei | Symbol | Quelle |
|-------|--------|--------|
| `fusion_ref/EURUSD_M15_<von>-<bis>_fusion_ref.parquet` | EURUSD | MT5/Fusion |
| `fusion_ref/XAUUSD_M15_<von>-<bis>_fusion_ref.parquet` | XAUUSD | MT5/Fusion |

Erzeugen:
```
python scripts/fetch_fusion_ref.py
```

## Qualitätsberichte

`quality_reports/<SYMBOL>_M15_dukascopy_quality.json` — pro Datensatz: Zeitraum,
Bar-Anzahl, Lücken (Wochenende vs. Intra-Session), Duplikate, Ausreißer und
Spread-Statistik (min/median/mean/max in Pips, getrennt nach Handelssession).
