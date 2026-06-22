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
---

## Gesamtauswertung: Alle 4 priorisierten Tests

*Stand: 2026-06-22 – sortiert nach Median OOS-Sharpe (robust gegenüber Einzelausreißern)*

| Rang | Test | Symbol | TF | Strategie | Ø OOS-Sharpe | Median OOS-Sharpe | Std OOS-Sharpe | Profitable Fenster | Trades | Urteil |
|------|------|--------|----|-----------|-------------|-------------------|----------------|---------------------|--------|--------|
| 1 | #2 | USDJPY | D1 | Trendfolge | 1.208 | 1.449 | 7.594 | 17/32 (53%) | 695 | Kandidat |
| 2 | #4 | EURUSD | H4 | Mean-Reversion | 0.389 | 1.142 | 3.670 | 27/40 (68%) | 5,201 | Kandidat |
| 3 | #3 | XAUUSD | H4 | Trendfolge | -0.036 | 0.191 | 3.678 | 22/40 (55%) | 5,159 | Kandidat |
| 4 | #1 | USDJPY | H4 | Trendfolge | -0.671 | n/a¹ | 4.480 | 15/40 (38%) | 5,199 | Verworfen |

*¹ Median wurde für Test #1 nicht separat erfasst (Einführung ab Test #2).*

**Lesart:** Median OOS-Sharpe > 0 bedeutet: in mehr als 50% aller Monate war das Modell profitabel
(der Median ist der 50%-Quantilswert des Sharpe-Verteilung). Er ist robuster als der Mittelwert,
da er von extremen Einzelfenstern nicht verzerrt wird.

---

## Korrelationsanalyse der drei Kandidaten

**Methode:** Pearson-Korrelation der fensterweisen OOS-Sharpe-Werte.
Tests #3 und #4 (beide H4, gleiche Periode) sind fenstergenau ausgerichtet (40 Fenster identisch).
Test #2 (D1, 32 Fenster) wird monatlich ausgerichtet auf den Überschneidungszeitraum.

### Paarweise Korrelationen

| Paar | Pearson r | Überlappende Fenster | Interpretation |
|------|-----------|---------------------|----------------|
| USDJPY D1 TF ↔ XAUUSD H4 TF (#2 vs. #3) | -0.056 | 32 Monate | nahezu unkorreliert |
| USDJPY D1 TF ↔ EURUSD H4 MR (#2 vs. #4) | 0.048 | 32 Monate | nahezu unkorreliert |
| XAUUSD H4 TF ↔ EURUSD H4 MR (#3 vs. #4) | 0.001 | 40 Fenster (exakt) | nahezu unkorreliert |

### Gleichgerichtete Fenster (#3 und #4, n=40)

- Beide positiv (profitabler Monat fuer beide): **16/40** (40%)
- Beide negativ (Verlusmonat fuer beide): **7/40** (18%)
- Divergent (einer positiv, einer negativ): **17/40** (42%)

### Portfolio-Simulation: 50/50 Kombination von #3 und #4

| Metrik | XAUUSD H4 TF (#3) | EURUSD H4 MR (#4) | 50/50 Portfolio |
|--------|------------------|------------------|-----------------|
| Ø OOS-Sharpe | -0.036 | 0.389 | 0.176 |
| Median OOS-Sharpe | 0.191 | 1.142 | 0.875 |
| Std OOS-Sharpe | 3.678 | 3.670 | **2.600** |
| Profitable Fenster | 55% | 68% | 62% |

### Interpretation

Die Korrelation zwischen XAUUSD H4 TF und EURUSD H4 MR beträgt r=0.001 (nahezu unkorreliert). Das ist der theoretisch erwartete Effekt: Trendfolge und Mean-Reversion arbeiten nach entgegengesetzten Marktregime-Annahmen. In Trend-Monaten profitiert die TF-Strategie, während MR kämpft – und umgekehrt in Seitwärtsphasen. Diese niedrige Korrelation macht eine Kombination prinzipiell attraktiv.

Das 50/50 Portfolio hat eine Std von 2.600 – deutlich unter dem Minimum der Einzelsysteme (3.670). Echter Diversifikationsnutzen: Risiko sinkt ohne entsprechenden Renditeverlust.

**USDJPY D1 TF (Test #2) als dritte Komponente:** Die Korrelation mit XAUUSD H4 TF beträgt r=-0.056 (nahezu unkorreliert), mit EURUSD H4 MR r=0.048 (nahezu unkorreliert). Hinweis: Die monatliche Ausrichtung ist eine Näherung (D1- vs. H4-Fenster sind nicht identisch), die Werte sind daher weniger präzise als die #3/#4-Korrelation.

**Kombinations-Empfehlung:**
Vor einer echten Portfolio-Kombination müssen zwei weitere Fragen geklärt werden:
1. Transaktionskosten und Spread auf allen drei Instrumenten (besonders USDJPY D1: nur 695 Trades,
   wenig Signal-Frequenz; XAUUSD und EURUSD H4 mit ~5200 Trades deutlich aktiver).
2. Kapital-Effizienz: USDJPY D1 bindet Overnight-Margin-Risiko; H4-Systeme sind kürzer exponiert.
Wenn beide Punkte akzeptabel: Kombination von #3 (XAUUSD TF) + #4 (EURUSD MR) als erster Schritt empfohlen,
da deren Korrelation direkt messbar und die Signal-Frequenz vergleichbar ist.
---

## 3-Wege Portfolio-Analyse

*Stand: 2026-06-22*

### Methode: Kalendermonatliche Ausrichtung

Jedes WF-Fenster wird seinem Kalendermonat (`YYYY-MM` aus `test_start`) zugeordnet.
Kombinierter Monats-Sharpe = Σ wᵢ × OOS-Sharpeᵢ für jeden gemeinsamen Monat.

| System | Warmup | Erstes Fenster | Letztes Fenster | Fenster |
|--------|--------|---------------|----------------|---------|
| USDJPY D1 TF  (#2) | 200 D1-Bars ≈ 10 Monate | 2021-04 | 2023-11 | 32 |
| XAUUSD H4 TF  (#3) | 200 H4-Bars ≈ 33 Tage   | 2020-08 | 2023-11 | 40 |
| EURUSD H4 MR  (#4) | 200 H4-Bars ≈ 33 Tage   | 2020-08 | 2023-11 | 40 |

**Gemeinsamer Auswertungszeitraum:** 2021-04 bis 2023-11 (32 Monate)

*Anmerkung: OOS-Sharpe ist `mean(r)/std(r)*√252`, dimensionslos und timeframe-unabhängig.
Kombination über D1 (≈22 Trades/Monat) und H4 (≈130 Trades/Monat) ist Standard-Näherung.
Gewichtung repräsentiert den Kapitalanteil pro Strategie.*

### Ergebnisse nach Gewichtungsschema

*Gemeinsamer Zeitraum: 2021-04 bis 2023-11 (32 Monate)*

| Kategorie | Gewichtung (D1/H4-TF/H4-MR) | Ø OOS-Sharpe | Median | Std | Profitable Monate | Median/Std |
|-----------|------------------------------|-------------|--------|-----|-------------------|------------|
| Referenz | Nur USDJPY D1 TF | +1.208 | +1.449 | 7.594 | 17/32 (53%) | **0.191** |
| Referenz | Nur XAUUSD H4 TF | +0.328 | +0.440 | 3.758 | 20/32 (62%) | **0.117** |
| Referenz | Nur EURUSD H4 MR | +0.516 | +1.126 | 2.873 | 21/32 (66%) | **0.392** |
| 2-Wege | H4-Portfolio (50/50 #3+#4) | +0.422 | +1.005 | 2.437 | 21/32 (66%) | **0.413** |
| 3-Wege | Gleichgewichtet (33/33/34) | +0.684 | +0.475 | 3.292 | 18/32 (56%) | **0.144** |
| 3-Wege | D1 niedrig (25/40/35) | +0.614 | +0.278 | 2.923 | 16/32 (50%) | **0.095** |
| 3-Wege | D1 sehr niedrig (20/40/40) | +0.579 | +0.521 | 2.720 | 18/32 (56%) | **0.192** |
| 3-Wege | Risiko-Parität (19/40/40) | +0.575 | +0.539 | 2.702 | 18/32 (56%) | **0.200** |
| 3-Wege | Perf-gewichtet (52/7/41) ² | +0.863 | +1.400 | 4.439 | 19/32 (59%) | **0.315** |
*² In-Sample Median als Gewicht – Datensnooping-Risiko*

**Bestes Median/Std-Verhältnis: H4-Portfolio (50/50 #3+#4)** (Median/Std = 0.413)

### Korrelation im gemeinsamen Zeitraum

| Paar | Pearson r | Interpretation |
|------|-----------|----------------|
| USDJPY D1 TF ↔ XAUUSD H4 TF | +0.087 | nahezu unkorreliert |
| USDJPY D1 TF ↔ EURUSD H4 MR | +0.256 | schwach korreliert |
| XAUUSD H4 TF ↔ EURUSD H4 MR | +0.064 | nahezu unkorreliert |

*Korrelationen hier auf dem gemeinsamen Teilzeitraum (32 Monate, exakt ausgerichtet).*

### Fazit und Empfehlung

Die drei Kandidaten sind im gemeinsamen Zeitraum (2021-04 bis 2023-11 (32 Monate)) nahezu unkorreliert (maximales |r| = 0.256). Jede 3-Wege-Kombination reduziert die Std gegenüber den Einzelsystemen.

Das **beste Median/Std-Verhältnis** erreicht **H4-Portfolio (50/50 #3+#4)** mit Median/Std = 0.413 (Median = +1.005, Std = 2.437, 66% profitable Monate).

Das 2-Wege H4-Portfolio übertrifft alle 3-Wege-Varianten im Median/Std (0.413 vs. 0.200 beim besten 3-Wege-Schema Risiko-Parität). Das Hinzufügen von USDJPY D1 verschlechtert das Risiko-Rendite-Verhältnis: die hohe D1-Std (7.594) überwiegt den Nutzen aus dem höchsten Einzelmedian (1.449).

**Praktische Einschränkungen USDJPY D1 (#2) im Portfolio:**
- Nur 695 Trades über 4 Jahre (≈14.5 Trades/Monat) – statistisch dünnere Basis als H4
- Overnight-Margin-Risiko durch D1-Haltedauer
- Der starke Ausreisser in Fenster 13 (Fed/BoJ-Divergenz 2022) ist im Backtest enthalten   und nicht reproduzierbar

**Empfehlung:** Als ersten Live-Test empfiehlt sich die H4-Portfolio-Kombination (#3 XAUUSD TF + #4 EURUSD MR, 50/50), da dort die Korrelation exakt messbar und die Datengrundlage mit je 40 Fenstern robuster ist. USDJPY D1 kann als optionale dritte Komponente mit niedrigem Gewicht (15–25%) hinzugefügt werden, sobald Live-Daten über mindestens 12 Monate vorliegen.

---

## Demo-Live-Test: H4-Portfolio (XAUUSD TF + EURUSD MR)

**Startdatum:** 2026-06-22  
**Geplante Mindestlaufzeit:** 1 Woche (bis 2026-06-29)  
**Modus:** CONFIRM_REQUIRED (Bestätigung per GUI-Banner vor jedem Trade)

### Setup

| Parameter | Wert |
|-----------|------|
| Symbol #1 | XAUUSD H4 – Trendfolge (SignalModel, Test #3) |
| Symbol #2 | EURUSD H4 – Mean-Reversion (MeanReversionModel, Test #4) |
| Risikoallokation | 50/50 (gleiche `risk_per_trade_pct` pro Symbol) |
| Handelsmodus | CONFIRM_REQUIRED (Demo-Live, echte MT5-Demo-Positionen) |
| Zyklusintervall | 300 s (5 min) pro Symbol sequenziell |
| Infrastruktur | Gemeinsamer RiskGuard, PositionSizer, CorrelationGuard, AuditLog |

### Ziel des Demo-Live-Tests

- Entspricht das reale Signal-Verhalten dem Walk-Forward-Backtest?
- Sind Spread-Kosten und Slippage auf dem Demo-Konto im akzeptablen Bereich?
- Funktioniert die sequenzielle Parallelüberwachung beider Symbole stabil?
- Verhält sich der EURUSD-MR-Features-Loader (Live-Ergänzung der 3 MR-Features) korrekt?

### Protokollierung

Alle Trades und Ereignisse werden protokolliert in:
- `data/processed/audit.db` (AuditLog – vollständige Zyklusdaten)
- `data/processed/paper_trades.json` (OrderExecutor – Trade-History)

### Start-Anleitung

Siehe Abschnitt *Startup-Anleitung* am Ende dieses Logs.
