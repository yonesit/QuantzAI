# QuantzAI — Projektkontext für Claude

## Projektziel

QuantzAI ist ein vollständig autonomes, ML-basiertes Handelssystem für den Forex- und Rohstoffmarkt.
Das übergeordnete Ziel ist **echter, nachhaltiger Profit** im Live-Markt.

Konkret bedeutet das:
- **Gemessenes, kostenbereinigtes P&L-Walk-Forward-Sharpe ≥ 1.0** ist das EINZIGE
  Erfolgskriterium vor dem Live-Einstieg — keine Proxy-Metrik (Accuracy/F1/
  Klassifikations-„Sharpe") mehr.
- Paper/Demo-Betrieb ohne Abstürze, mit autonomen SL/TP-Schließungen
- RiskGuard schützt das Kapital (max. 5 % Tagesverlust, max. 15 % Drawdown)
- Vollständige Marktanalyse: Trend, Mean-Reversion, Multi-Timeframe, Sentiment, Kalender-Filter

> **Wichtige Korrektur zur Historie:** Der früher genannte „OOS-Sharpe ~0.41" war
> ein **Klassifikations-Proxy** (±1-Returns, OHNE Kosten). Der echte P&L-Sharpe des
> alten H4-Systems lag nach voller Kostenkette bei **~−1.33** — also kein Edge. Diese
> Erkenntnis hat den kompletten Strategie-Neuaufbau ausgelöst.

---

## Portfolio (Strategie-Neuaufbau, M15)

| Symbol  | Timeframe | Datenbasis                        | Status Daten            |
|---------|-----------|-----------------------------------|-------------------------|
| EURUSD  | M15       | Dukascopy 2016–2026 (10 J, echter Spread) | fertig + gelabelt |
| XAUUSD  | M15       | Dukascopy 2010–2026 (Download läuft)      | Download läuft    |

Finale Baseline erst, wenn **beide** Symbole im **gleichen Fenster** vorliegen
(Portfolio-Sharpe). Broker: FusionMarkets-Demo (Raw-/Zero-Konto) über MetaTrader 5.

---

## Architektur

```
scripts/run_gui_bot.py          ← Einstiegspunkt (GUI + Bot-Thread)
src/orchestrator.py             ← MultiSymbolOrchestrator, pro Symbol ein Zyklus
src/data/pipeline.py            ← DataPipeline: fetch → validate → features → parquet
src/data/feature_builder.py     ← Technische Features + MTF (H4/D1 merge in H1)
src/models/signal_model.py      ← LightGBM TrendFollow (Triple-Barrier Labels)
src/models/mean_reversion_model.py ← LightGBM Mean-Reversion
src/models/retraining_scheduler.py ← Automatisches Retraining nach Drift
src/risk/risk_guard.py          ← Tagesverlust + Max-Drawdown Schutz
src/risk/position_sizer.py      ← Kelly / ATR-basierte Lotgröße
src/execution/order_executor.py ← Paper- und Live-Orders, check_paper_sl_tp()
src/data/calendar.py            ← Wirtschaftskalender (News-Filter)
src/monitoring/audit_log.py     ← SQLite-Audit-Log aller Entscheidungen
gui/                            ← PySide6 GUI (Cockpit, Dashboard, Backtest, Risk)
```

**Walk-Forward-Validation** mit OOS-Sharpe-Annualisierung per Timeframe:
- M15 = 35.040, H1 = 8.760, H4 = 2.190, D1 = 252

---

## Aktueller Status — Strategie-Neuaufbau auf M15

Kompletter Neuaufbau, weil das alte H4-System nach Kosten kein Edge hatte
(P&L-Sharpe ~−1.33). Neuer Ansatz: 10 J Dukascopy-M15-Daten, **kostenbewusstes
Triple-Barrier-Labeling** mit **gemessenem** Fusion-Kostenmodell.

Gemessenes Kostenmodell EURUSD (Raw-/Zero-Konto, `config/cost_model_EURUSD.yaml`):
- **Kommission 0.464 Pips Round-Turn** (aus 68 echten MT5-Deals gemessen)
- **Effektiver Spread ≈ Dukascopy-Niveau** (~0.3 Pips Median), Faktor 0.667
- Slippage 0.2 Pips/Seite, Swap nur bei Overnight

Fortschritt der Phasen (Issues #73–#78):
1. **Fundament** (#73): 10 J Daten + kostenbewusstes Labeling — EURUSD fertig
   (`EURUSD_M15_2016-2026_labeled.parquet`), XAUUSD-Download läuft. **← hier**
2. Regime-Filter (#74) · 3. Meta-Labeling (#75) · 4. Sizing (#76)
5. Robustheit (#77) · 6. Demo-Validierung (#78)

Validierung: **Purged Walk-Forward mit Embargo**, Erfolgsmaß ausschließlich
gemessenes kostenbereinigtes P&L-OOS-Sharpe. **Staying-on-Demo bleibt bewusste
Entscheidung, bis der Sharpe stabil ≥ 1.0 ist.**

---

## Tech-Stack

- **Python 3.11**, LightGBM, scikit-learn, pandas, pyarrow
- **MetaTrader 5** (MT5Connector) für Marktdaten und Orderausführung
- **PySide6** für die Desktop-GUI
- **Loguru** für Logging, **SQLite** (audit.db) für Trade-Protokoll
- **pytest** für alle Tests — vor jedem Commit vollständiger Testlauf

---

## Arbeitsregeln mit Claude

1. **Profitabilität vor Features** — nichts implementieren, das die Sharpe-Ratio nicht verbessert
2. Nach jeder Code-Änderung: `python -m pytest tests/ -x -q`, dann commit, dann push
3. Kein Mock der echten Daten in Tests, die `paper_trades.json` berühren — immer `tmp_path`
4. Commits auf Deutsch, prägnant, Konventionalcommit-Format (`feat:`, `fix:`, `refactor:`)
5. PowerShell-Commits mit Heredoc-Syntax (`git commit -m @'...'@`)
6. Keine `{expression}` in Loguru-Format-Strings — immer als `kwargs` übergeben

---

## Wichtige Dateipfade

| Datei | Inhalt |
|-------|--------|
| `data/processed/paper_trades.json` | Alle Paper-Trades (offen + geschlossen) |
| `data/processed/risk_state.json` | RiskGuard-Zustand (Balance, Drawdown) |
| `data/processed/audit.db` | SQLite-Audit-Log aller Bot-Entscheidungen |
| `models/*.joblib` | Trainierte Modelle |
| `config/config.yaml` | Zentrale Konfiguration |
| `.env` | MT5-Zugangsdaten (nicht im Repo) |

---

## Roadmap (Phasen 1–6, Issues #73–#78)

1. **Fundament** (#73): 10–16 J M15-Daten (Dukascopy) + kostenbewusstes
   Triple-Barrier-Labeling mit gemessenem Kostenmodell. — läuft
2. **Regime-Filter** (#74): Marktphasen erkennen, Strategie danach filtern
3. **Meta-Labeling** (#75): zweites Modell entscheidet, ob ein Signal gehandelt wird
4. **Sizing** (#76): realistisches Risiko/Trade (kein All-in), Kelly/Vol-Targeting
5. **Robustheit** (#77): Monte-Carlo, Parameter-Stabilität, Overfitting-Checks
6. **Demo-Validierung** (#78): Live-Demo, bis Sharpe stabil ≥ 1.0 → dann Live

Jeder Schritt wird ausschließlich am gemessenen, kostenbereinigten
P&L-Walk-Forward-Sharpe gemessen.
