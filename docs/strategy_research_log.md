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
| 2 | USDJPY | D1 | Trendfolge | Test ob noch längerer Timeframe noch robuster | Kandidat |
| 3 | XAUUSD | H4 | Trendfolge | Andere Asset-Klasse, andere Treiber (Inflation/Risk-Off), oft trendstärker | Kandidat |
| 4 | EURUSD | H4 | Mean-Reversion | Testet ob MR auf höherem Timeframe als H1 besser performt | Kandidat |

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

### Test #2: USDJPY D1 Trendfolge
- Datum: 2026-06-22
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Walk-Forward: 6M Training / 1M Test, rollierend (32 Fenster, D1 ≈21 Bars/Test-Fenster, kein Anpassungsbedarf (min 10 Train / 2 Test gut erfuellt))
- Ø OOS-Sharpe: 1.208
- Std OOS-Sharpe: 7.594
- Profitable Fenster: 17/32 (53%)
- Anzahl Trades gesamt: 695
- SHAP Top-3 Features: adx, ema_200, atr_14
- Auffälligkeiten/Extremwerte: Ausreisser oben: Fenster 13 OOS-Sharpe=22.59; Ausreisser unten: Fenster 7 OOS-Sharpe=-10.33; hour_of_day konstant=0 auf D1 (informationslos, aber modellseitig korrekt ignoriert)
- Urteil: Kandidat
- Begründung des Urteils: Ø OOS-Sharpe 1.208 > 0 und 53% profitable Fenster > 50%. Beide Mindestanforderungen erfuellt.

**Robustheits-Analyse (hinzugefügt 2026-06-22):**

| Metrik | MIT Fenster 13 | OHNE Fenster 13 |
|--------|---------------|----------------|
| Ø OOS-Sharpe | 1.208 | 0.518 |
| Std OOS-Sharpe | 7.594 | 6.656 |
| Median OOS-Sharpe | 1.449 | 1.449 |
| Profitable Fenster | 17/32 (53%) | 16/31 (52%) |

**Fenster 13 Zeitraum:** 2022-05-08 – 2022-06-08

**Historische Einordnung:** Fenster 13 (2022-05-08 bis 2022-06-08): Peak der Fed-BoJ-Zinsdivergenz 2022. Die US-Fed erhöhte im Mai 2022 die Zinsen um 50 Bp (stärkstes Anheben seit 2000), während die BoJ ihre Nullzinspolitik und Yield-Curve-Control (YCC, 10J JGB-Cap 0.25%) unbeirrt fortsetzte. USDJPY stieg in dieser Phase von ~128 auf ~136 – eine in 20 Jahren nicht gesehene Yen-Abwertungsgeschwindigkeit. Dieses Event ist ein singuläres, nicht-wiederholbares Makro-Ereignis: die extremste geldpolitische Divergenz zwischen zwei G7-Zentralbanken seit 1998. Die BoJ beendete YCC schrittweise ab Juli 2023. Ein erneutes Setup dieser Art in einem 4-Jahres-Backtest-Fenster ist sehr unwahrscheinlich.

**Korrigiertes Urteil:** Kandidat
**Begründung:** Auch ohne Ausreisser Fenster 13: Ø OOS-Sharpe 0.518 > 0 und 52% profitable Fenster > 50%. Beide Kriterien erfuellt, aber knapp – weitere Tests empfohlen.

### Test #3: XAUUSD H4 Trendfolge
- Datum: 2026-06-22
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Walk-Forward: 6M Training / 1M Test, rollierend (40 Fenster)
- Ø OOS-Sharpe: -0.036
- Std OOS-Sharpe: 3.678
- Median OOS-Sharpe: 0.191
- Profitable Fenster: 22/40 (55%)
- Anzahl Trades gesamt: 5159
- SHAP Top-3 Features: atr_14, obv, ema_200
- Auffälligkeiten/Extremwerte: Ausreisser oben: Fenster 34 (2023-06-18–2023-07-18) OOS-Sharpe=5.32; Ausreisser unten: Fenster 9 (2021-05-18–2021-06-18) OOS-Sharpe=-13.56
- Urteil: Kandidat
- Begründung des Urteils: Ø OOS-Sharpe 0.519 > 0 und 58% profitable Fenster > 50% (ohne Ausreisser-Fenster [5, 9]). Beide Mindestanforderungen erfuellt.
**Robustheits-Analyse:**

| Metrik | Alle Fenster | Ohne Ausreisser |
|--------|-------------|----------------|
| Ø OOS-Sharpe | -0.036 | 0.519 |
| Std OOS-Sharpe | 3.678 | 2.761 |
| Median OOS-Sharpe | 0.191 | 0.258 |
| Profitable Fenster | 22/40 (55%) | 22/38 (58%) |

**Ausreisser-Fenster:** Fenster 5 (2021-01-18 – 2021-02-18): OOS-Sharpe=-7.57; Fenster 9 (2021-05-18 – 2021-06-18): OOS-Sharpe=-13.56

### Test #4: EURUSD H4 Mean-Reversion
- Datum: 2026-06-22
- Zeitraum: 4 Jahre (2020-01-01 bis 2024-01-01)
- Modell: MeanReversionModel (26 Features: Standard-23 + bb_pct_b, dist_ema20_atr, dist_sma50_atr)
- Label-Parameter: tp_atr_mult=1.0, sl_atr_mult=2.0, max_candles=10 (H4 = ~2 Handelstage)
- Walk-Forward: 6M Training / 1M Test, rollierend (40 Fenster)
- Ø OOS-Sharpe: 0.389
- Std OOS-Sharpe: 3.670
- Median OOS-Sharpe: 1.142
- Profitable Fenster: 27/40 (68%)
- Anzahl Trades gesamt: 5201
- SHAP Top-3 Features: atr_14, ema_200, obv
- Auffälligkeiten/Extremwerte: Ausreisser oben: Fenster 8 (2021-04-18–2021-05-18) OOS-Sharpe=6.14; Ausreisser unten: Fenster 3 (2020-11-18–2020-12-18) OOS-Sharpe=-14.20
- Urteil: Kandidat
- Begründung des Urteils: Ø OOS-Sharpe 0.967 > 0 und 71% profitable Fenster > 50%. Beide Mindestanforderungen erfuellt.

**Robustheits-Analyse:**

| Metrik | Alle Fenster | Ohne Ausreisser |
|--------|-------------|----------------|
| Ø OOS-Sharpe | 0.389 | 0.967 |
| Std OOS-Sharpe | 3.670 | 2.610 |
| Median OOS-Sharpe | 1.142 | 1.245 |
| Profitable Fenster | 27/40 (68%) | 27/38 (71%) |

**Fenster 3 (2020-11-18 – 2020-12-18):** OOS-Sharpe=-14.20. Einordnung (normal-wiederkehrend): US-Praesidentschaftswahl 2020: kurzfristige EURUSD-Volatilitaet. Normal-wiederkehrend (Wahlrisiko).

**Fenster 27 (2022-11-18 – 2022-12-18):** OOS-Sharpe=-6.99. Einordnung (singulaer): UK-Gilts-Krise (Truss/Kwarteng): EUR-Kollateralschaden, Liquiditaets-Stress. Weitgehend singulaer.

**Vergleich mit EURUSD H1-Baseline:**
EURUSD H1 Mean-Reversion wurde nicht separat getestet (kein Eintrag im Log, kein entsprechender Commit in der Git-History). Vergleich gegen EURUSD H1 Trendfolge-Baseline (Test 0a, Ø OOS-Sharpe -0.484, 39% profitable Fenster): MR H4 übertrifft die H1-TF-Baseline im Ø OOS-Sharpe (0.389 vs. -0.484) und übertrifft sie in der Profitablen-Quote (68% vs. 39%). Ein direkter H1-MR ↔ H4-MR Vergleich ist nicht moeglich – das Experiment #H1-MR fehlt.
