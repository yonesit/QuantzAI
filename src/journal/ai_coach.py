"""
src/journal/ai_coach.py
AICoach – KI-Coach fuer die eigene Handelshistorie.

Natuerlichsprachige Fragen zur Handelshistorie werden auf Basis
strukturierter Daten aus TradeJournal, TradingDNA und PsychologyTracker
beantwortet. Das Sprachmodell (Anthropic API) erhaelt die strukturierten
Daten als Kontext-Prompt; kein Feintuning, kein eigenes Modell.

Sicherheitsprinzip:
  - Antworten basieren ausschliesslich auf den uebergebenen Daten.
  - Verweise auf konkrete Trade-IDs wo moeglich.
  - Bei wenig Daten ehrliche Rueckmeldung statt erfundener Aussagen.

Testbarkeit:
  _llm_fn injizierbar: (system_prompt: str, user_message: str) -> str.
  Standardmaessig wird der Anthropic Python-Client verwendet.
  Wenn anthropic nicht installiert und kein _llm_fn gesetzt:
  ask() gibt eine erklaerende Fehlermeldung zurueck.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL          = "claude-opus-4-8"
_DEFAULT_MAX_TRADES     = 50
_STATS_LOOKBACK_DAYS    = 30
_MIN_DATA_WARNING_TRADES = 10

_SYSTEM_INTRO = """\
Du bist ein erfahrener Trading-Coach, der einem Trader hilft, aus seiner \
eigenen Handelshistorie zu lernen.

Antworte AUSSCHLIESSLICH auf Basis der unten bereitgestellten Trade-Daten. \
Erfinde keine Daten und keine Trends. Verweise wenn moeglich auf konkrete \
Trade-IDs (Format: #ID). Wenn die Datenlage fuer eine Aussage nicht ausreicht, \
sage das ehrlich und klar. Sprich den Nutzer direkt an (du-Form).\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  AICoach
# ─────────────────────────────────────────────────────────────────────────────

class AICoach:
    """
    KI-Coach fuer die eigene Handelshistorie.

    Parameters
    ----------
    journal              : TradeJournal-Instanz. Pflicht fuer Tradeverlauf
                           und Statistiken. Wenn None: nur DNA/Psychologie.
    dna                  : TradingDNA-Instanz (optional).
    psychology_tracker   : PsychologyTracker-Instanz (optional).
    model                : Anthropic-Modell (Standard: claude-opus-4-8).
    max_trades_in_context: Maximale Anzahl Trades im Kontext-Prompt.
    _llm_fn              : Injizierbare LLM-Funktion fuer Tests.
                           Signatur: (system_prompt: str, user_msg: str) -> str.
                           Wenn None: Anthropic-Client wird erstellt.
    """

    def __init__(
        self,
        journal: Any                                        = None,
        dna: Any                                            = None,
        psychology_tracker: Any                             = None,
        model: str                                          = _DEFAULT_MODEL,
        max_trades_in_context: int                          = _DEFAULT_MAX_TRADES,
        _llm_fn: Optional[Callable[[str, str], str]]        = None,
    ) -> None:
        self._journal     = journal
        self._dna         = dna
        self._psychology  = psychology_tracker
        self._model       = model
        self._max_trades  = max_trades_in_context
        self._llm_fn      = _llm_fn

    # ── Oeffentliche Schnittstelle ────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Beantwortet eine natuerlichsprachige Frage zur Handelshistorie.

        Der Kontext (Trades, Statistiken, DNA-Profil, Psychologie) wird
        automatisch aus den uebergebenen Modulen zusammengestellt und
        als System-Prompt an das LLM uebergeben.

        Parameters
        ----------
        question : Natuerlichsprachige Frage des Traders.

        Returns
        -------
        str: Antwort des LLM oder Fehlermeldung wenn kein LLM verfuegbar.
        """
        if not question or not question.strip():
            return "Bitte stelle eine Frage."

        context = self.build_context()
        system  = f"{_SYSTEM_INTRO}\n\n{context}"

        llm_fn = self._llm_fn or self._default_llm_fn

        try:
            answer = llm_fn(system, question.strip())
            logger.info("AICoach: Frage beantwortet | len={n}", n=len(answer))
            return answer
        except Exception as exc:  # noqa: BLE001
            logger.error("AICoach: LLM-Aufruf fehlgeschlagen: {exc}", exc=exc)
            return f"LLM-Fehler: {exc}"

    def build_context(self) -> str:
        """
        Baut den strukturierten Daten-Kontext fuer den System-Prompt.

        Gibt einen Markdown-formatierten String mit allen verfuegbaren
        Datenquellen zurueck. Oeffentlich damit Datenaufbereitung isoliert
        getestet werden kann.
        """
        sections: list[str] = []

        # Letzte N Trades
        trades = self._get_recent_trades(self._max_trades)
        sections.append(self._format_trades_section(trades))

        # Performance-Statistiken
        sections.append(self._format_stats_section())

        # TradingDNA
        if self._dna is not None:
            sections.append(self._format_dna_section())

        # Psychologie-Muster
        if self._psychology is not None:
            sections.append(self._format_psychology_section())

        return "\n\n".join(s for s in sections if s.strip())

    # ── Datenaufbereitung (isoliert testbar) ──────────────────────────────────

    def _get_recent_trades(self, n: int) -> list[dict]:
        """Laedt die n juengsten Trades aus dem Journal."""
        if self._journal is None:
            return []
        try:
            with self._journal._lock:
                cur = self._journal._conn.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return list(reversed(rows))  # chronologisch aufsteigend
        except Exception as exc:  # noqa: BLE001
            logger.warning("AICoach: Fehler beim Laden der Trades: {exc}", exc=exc)
            return []

    def _format_trades_section(self, trades: list[dict]) -> str:
        """Formatiert die Trade-Liste als Markdown-Abschnitt."""
        n = len(trades)
        header = f"## Handelshistorie (letzte {n} Trades)"

        if not trades:
            return f"{header}\n\n*(Keine Trades vorhanden)*"

        if n < _MIN_DATA_WARNING_TRADES:
            header += f"\n\n> Hinweis: Nur {n} Trade(s) vorhanden – Aussagen sind statistisch wenig belastbar."

        return header + "\n\n" + self._format_trades_table(trades)

    def _format_trades_table(self, trades: list[dict]) -> str:
        """Formatiert eine Liste von Trade-Dicts als Markdown-Tabelle."""
        if not trades:
            return "*(Keine Trades)*"

        header = "| ID | Symbol | Richtung | Lots | Einstieg | Ausstieg | P&L | Status | Setup |"
        sep    = "|----|--------|----------|------|----------|----------|-----|--------|-------|"
        rows   = []
        for t in trades:
            pnl    = t.get("pnl")
            pnl_s  = f"{pnl:+.2f}" if pnl is not None else "offen"
            entry  = str(t.get("entry_price") or "–")
            exit_p = str(t.get("exit_price") or "–")
            rows.append(
                f"| #{t['id']} | {t.get('symbol','')} | {t.get('direction','')} "
                f"| {t.get('lot_size') or '–'} | {entry} | {exit_p} "
                f"| {pnl_s} | {t.get('status','')} | {t.get('setup') or '–'} |"
            )
        return "\n".join([header, sep] + rows)

    def _format_stats_section(self) -> str:
        """Erstellt Performance-Statistiken der letzten 30 Tage."""
        header = f"## Performance-Statistiken (letzte {_STATS_LOOKBACK_DAYS} Tage)"
        if self._journal is None:
            return f"{header}\n\n*(Kein Journal verfuegbar)*"
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=_STATS_LOOKBACK_DAYS)
            stats = self._journal.calculate_stats(start, end)
            return header + "\n\n" + self._format_stats(stats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AICoach: Statistikfehler: {exc}", exc=exc)
            return f"{header}\n\n*(Statistikfehler: {exc})*"

    def _format_stats(self, stats: dict) -> str:
        """Formatiert ein Statistik-Dict als Markdown-Text."""
        if stats.get("n_trades", 0) == 0:
            return "Keine abgeschlossenen Trades im Zeitraum."

        pf = stats.get("profit_factor", 0.0)
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        best  = stats.get("best_trade")
        worst = stats.get("worst_trade")
        return (
            f"- Trades: **{stats['n_trades']}**\n"
            f"- Win-Rate: **{stats['win_rate']:.1%}**\n"
            f"- Profit-Faktor: **{pf_s}**\n"
            f"- Durchschn. Gewinn: **{stats['avg_win']:.2f}**\n"
            f"- Durchschn. Verlust: **{stats['avg_loss']:.2f}**\n"
            f"- Gesamt-P&L: **{stats['total_pnl']:+.2f}**\n"
            f"- Bester Trade: **{f'+{best:.2f}' if best is not None else '–'}**\n"
            f"- Schlechtester Trade: **{f'{worst:.2f}' if worst is not None else '–'}**"
        )

    def _format_dna_section(self) -> str:
        """Erstellt einen TradingDNA-Profilabschnitt."""
        header = "## TradingDNA-Profil"
        try:
            profile = self._dna.generate_profile()
            return header + "\n\n" + self._format_dna_summary(profile)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AICoach: DNA-Fehler: {exc}", exc=exc)
            return f"{header}\n\n*(Profil nicht verfuegbar: {exc})*"

    def _format_dna_summary(self, profile: dict) -> str:
        """Formatiert ein TradingDNA-Profil-Dict als kompakten Text."""
        if profile.get("status") != "ready":
            n       = profile.get("n_trades", 0)
            minimum = profile.get("min_trades_required", 500)
            return (
                f"Profil noch nicht verfuegbar: {n}/{minimum} Trades vorhanden. "
                "Noch keine statistisch belastbaren Aussagen moeglich."
            )

        lines: list[str] = []

        # Beste/schlechteste Handelsstunden
        hours = profile.get("trading_hours", {})
        best_h  = hours.get("best",  [])
        worst_h = hours.get("worst", [])
        if best_h:
            best_h_str = ", ".join(f"{h['hour']}:00 Uhr (WR {h['win_rate']:.0%})" for h in best_h)
            lines.append(f"**Beste Handelsstunden:** {best_h_str}")
        if worst_h:
            worst_h_str = ", ".join(f"{h['hour']}:00 Uhr (WR {h['win_rate']:.0%})" for h in worst_h)
            lines.append(f"**Schlechteste Handelsstunden:** {worst_h_str}")

        # Beste/schlechteste Wochentage
        days = profile.get("trading_weekdays", {})
        best_d  = days.get("best",  [])
        worst_d = days.get("worst", [])
        if best_d:
            best_d_str = ", ".join(f"{d['weekday']} (WR {d['win_rate']:.0%})" for d in best_d)
            lines.append(f"**Beste Wochentage:** {best_d_str}")
        if worst_d:
            worst_d_str = ", ".join(f"{d['weekday']} (WR {d['win_rate']:.0%})" for d in worst_d)
            lines.append(f"**Schlechteste Wochentage:** {worst_d_str}")

        # Beste/schlechteste Symbole
        syms = profile.get("symbols", {})
        best_s  = syms.get("best",  [])
        worst_s = syms.get("worst", [])
        if best_s:
            lines.append("**Beste Symbole:** " + ", ".join(
                f"{s['symbol']} (P&L {s['total_pnl']:+.2f}, WR {s['win_rate']:.0%})" for s in best_s
            ))
        if worst_s:
            lines.append("**Schlechteste Symbole:** " + ", ".join(
                f"{s['symbol']} (P&L {s['total_pnl']:+.2f}, WR {s['win_rate']:.0%})" for s in worst_s
            ))

        # Psychologische Schwaechen aus DNA
        weaknesses = profile.get("psychological_weaknesses", [])
        if weaknesses:
            lines.append("**Psychologische Schwaechen:** " + "; ".join(weaknesses))

        # Optimale Lot-Groesse
        sizing = profile.get("position_sizing", {})
        opt_label = sizing.get("optimal_label")
        opt_wr    = sizing.get("optimal_win_rate")
        if opt_label and opt_wr is not None:
            lines.append(f"**Optimale Lot-Groesse:** {opt_label} (WR {opt_wr:.0%})")

        return "\n".join(lines) if lines else "Kein DNA-Profil erstellt."

    def _format_psychology_section(self) -> str:
        """Erstellt einen Psychologie-Muster-Abschnitt."""
        header = "## Psychologie-Muster"
        try:
            patterns = self._psychology.analyze_mood_patterns()
            return header + "\n\n" + self._format_psychology_summary(patterns)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AICoach: Psychologie-Fehler: {exc}", exc=exc)
            return f"{header}\n\n*(Psychologie-Daten nicht verfuegbar: {exc})*"

    def _format_psychology_summary(self, patterns: dict) -> str:
        """Formatiert Mood-Muster als Markdown-Text."""
        if not patterns:
            return (
                "Noch nicht genug Trades fuer eine Musteranalyse "
                "(mindestens 30 abgeschlossene Trades benoetigt)."
            )

        lines: list[str] = ["| Stimmung | Trades | Win-Rate |", "|---------|--------|----------|"]
        for mood, stats in sorted(patterns.items(), key=lambda x: -x[1]["win_rate"]):
            mood_label = mood.value.capitalize() if hasattr(mood, "value") else str(mood)
            lines.append(
                f"| {mood_label} | {stats['n_trades']} | {stats['win_rate']:.0%} |"
            )
        return "\n".join(lines)

    # ── LLM-Aufruf ────────────────────────────────────────────────────────────

    def _default_llm_fn(self, system_prompt: str, user_message: str) -> str:
        """Ruft den Anthropic-Client auf. Wirft ImportError wenn nicht installiert."""
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Das 'anthropic'-Paket ist nicht installiert. "
                "Installiere es mit: pip install anthropic"
            ) from exc

        client   = anthropic.Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
