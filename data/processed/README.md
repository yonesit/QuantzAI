# data/processed — Datensätze (Phase 1)

Dieses Verzeichnis enthält die aufbereiteten Marktdaten. **Parquet-Dateien sind
git-ignoriert** (`*.parquet`) und liegen nur lokal auf der VM — versioniert wird
ausschließlich die reproduzierbare Tooling-Pipeline (`scripts/`, `src/data/`).

## 1. Dukascopy-Langhistorie (Training/Labeling) — PRIMÄR

Lange, tick-abgeleitete M15-Historie inkl. **echtem historischem Bid-Ask-Spread**
pro Bar. Grundlage für das kostenbewusste Triple-Barrier-Labeling (Phase 1).

Spalten: `timestamp` (UTC, Bar-Open), `open/high/low/close` (Mid-Preis),
`volume`, `spread_mean`, `spread_median` (Preis-Einheiten), `tick_count`.

### Baseline-Datensatz (10 Jahre, identisches Fenster) — AKTUELL

Für die erste Baseline werden **beide Symbole über den IDENTISCHEN Zeitraum
`2016-01-01 .. heute`** gezogen, damit der spätere Portfolio-Sharpe über dieselben
Marktregime / dieselbe Länge gemessen wird (nicht unterschiedliche Historien-Längen).

| Datei | Symbol | Zeitraum |
|-------|--------|----------|
| `EURUSD_M15_2016-2026.parquet` | EURUSD | 2016-01-01 .. heute |
| `XAUUSD_M15_2016-2026.parquet` | XAUUSD | 2016-01-01 .. heute |

Erzeugen / fortsetzen (resume-fähig, gleicher Zeitraum für beide):
```
python scripts/fetch_dukascopy.py --symbol EURUSD --start 2016-01-01
python scripts/fetch_dukascopy.py --symbol XAUUSD --start 2016-01-01
```

### Vollhistorie — ZIEL FÜR PHASE 5 (Robustheit)

Die **volle Historie (EURUSD ~2003, XAUUSD ~2010)** bleibt das Ziel: sie wird für
**Phase 5 (Robustheits-Validierung / ungesehener Out-of-Sample-Härtetest)** per
**Resume** nachgezogen — ohne `--start` lädt das Skript die maximale Historie und
nutzt die bereits im Tages-Cache liegenden Tage weiter. Das Output-Parquet wird
dabei exakt auf das angeforderte `[start, end]`-Fenster geklemmt; der Cache behält
immer alle geladenen Tage.

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
