# QuantzAI – Strategie-Forschungslog

Zentrale, lückenlose Dokumentation aller Profitabilitäts-Experimente. Jeder Test wird hier
eingetragen, BEVOR er als "vielversprechend" gilt – kein Ergebnis wird vorzeitig gefeiert.

**Regel:** Eine Kombination gilt erst als ernstzunehmender Kandidat, wenn sie über mehrere
Jahre UND mehrere Walk-Forward-Fenster konsistent positiv ist (Ø OOS-Sharpe > 0,
profitable Fenster-Quote > 50%, keine extremen Einzel-Ausreißer die das Ergebnis tragen).

---

## Referenz-Baseline (bereits abgeschlossen)

| # | Symbol | TF | Strategie | Zeitraum | Ø OOS-Sharpe | Std | Profitable Fenster | Status |
|---|--------|----|-----------|----------|--------------|-----|---------------------|--------|
| 0a | EURUSD | H1 | Trendfolge (23 Feat) | 2020-2024 | -0.484 | 2.177 | 16/41 (39%) | Baseline |
| 0b | EURUSD | H1 | Trendfolge + MTF (25 Feat, ungefiltert) | 2020-2024 | -0.561 | 2.161 | 20/41 (49%) | Verworfen |
| 0c | EURUSD | H1 | Trendfolge + MTF-Gate | 2020-2024 | -0.507 | 2.133 | 19/41 (46%) | Teilverbesserung, kein Edge |

**Erkenntnis aus der Baseline-Phase:** Reine technische Preis-Indikatoren auf EURUSD H1 liefern
keinen stabilen Edge. Feature-Engineering allein (mehr/weniger Features, Multi-Timeframe-Kontext)
verbessert die Situation nicht grundlegend. Nächster Hebel: andere Symbole, Timeframes, Strategie-Typen.

---

## Priorisierte Testmatrix (aktuell)

| # | Symbol | TF | Strategie | Begründung | Status |
|---|--------|----|-----------|------------|--------|
| 1 | USDJPY | H4 | Trendfolge | BoJ-Politik erzeugt historisch längere, klarere Trends als EURUSD | Verworfen |
| 2 | USDJPY | D1 | Trendfolge | Test ob noch längerer Timeframe noch robuster | ⏳ offen |
| 3 | XAUUSD | H4 | Trendfolge | Andere Asset-Klasse, andere Treiber (Inflation/Risk-Off), oft trendstärker | ⏳ offen (Datenverfügbarkeit prüfen) |
| 4 | EURUSD | H4 | Mean-Reversion | Testet ob MR auf höherem Timeframe als H1 besser performt | ⏳ offen |

## Erweiterte Testmatrix (nur falls Priorisierte Matrix keinen klaren Kandidaten liefert)

| Symbol | H1 | H4 | D1 |
|--------|----|----|----|
| EURUSD | Trend ✅ (0a-c) / MR ⏳(4) | MR ⏳(4) | – |
| GBPUSD | ⏳ | ⏳ | ⏳ |
| USDJPY | ⏳ | ⏳(1) | ⏳(2) |
| XAUUSD | ⏳ | ⏳(3) | ⏳ |

---

## Testprotokoll-Vorlage (für jeden neuen Eintrag)

```
### Test #N: [Symbol] [TF] [Strategie]
- Datum: YYYY-MM-DD
- Zeitraum: 4 Jahre (oder begründete Abweichung)
- Walk-Forward: 6M Training / 1M Test, rollierend
- Ø OOS-Sharpe: X.XXX
- Std OOS-Sharpe: X.XXX
- Profitable Fenster: X/Y (Z%)
- Anzahl Trades gesamt: X
- SHAP Top-3 Features: ...
- Auffälligkeiten/Extremwerte: ...
- Urteil: [Kandidat / Verworfen / Unklar - weitere Tests nötig]
- Begründung des Urteils: ...
```

---

## Ergebnisse (wird laufend ergänzt)

### Test #1: USDJPY H4 Trendfolge
- Datum: 2026-06-22
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Walk-Forward: 6M Training / 1M Test, rollierend
- Ø OOS-Sharpe: -0.671
- Std OOS-Sharpe: 4.480
- Profitable Fenster: 15/40 (38%)
- Anzahl Trades gesamt: 5199
- SHAP Top-3 Features: atr_14, ema_200, obv
- Auffälligkeiten/Extremwerte: Extremer Ausreisser oben: Fenster 19 OOS-Sharpe=11.52; Extremer Ausreisser unten: Fenster 39 OOS-Sharpe=-7.84
- Urteil: Verworfen
- Begründung des Urteils: Ø OOS-Sharpe -0.671 <= 0 und nur 38% profitable Fenster. Kein stabiler Edge nachweisbar.
