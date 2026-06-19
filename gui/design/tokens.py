"""
gui/design/tokens.py
Design-Tokens fuer das QuantzAI Design-System.

Einmalig definiert, ueberall genutzt – kein view-spezifischer Stil.
Keine Qt-Abhaengigkeiten; rein Python.

Farb-Semantik (unveraenderlich):
  PROFIT  -> Gruen      – ausschliesslich fuer Gewinne/positive Werte
  LOSS    -> Rot        – ausschliesslich fuer Verluste/negative Werte
  WARNING -> Amber      – nicht-kritische Warnungen, Bot pausiert
  DANGER  -> Dunkelrot  – irreversible / kritische Aktionen
  INFO    -> Blau       – informative Hinweise
  NEUTRAL -> Grau       – inaktive / neutrale Elemente
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
#  Color Tokens
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColorTokens:
    """Vollstaendige Farbpalette fuer einen Modus (Dark oder Light)."""

    # Hintergrund-Hierarchie
    bg_base:     str  # dunkelster Hintergrund (App-Root)
    bg_surface:  str  # Karten, Panels
    bg_elevated: str  # Dropdowns, Tooltips

    # Rahmen
    border:        str  # sichtbare Trennlinien
    border_subtle: str  # kaum sichtbare Trennlinien

    # Text
    text_primary:   str  # Haupttext
    text_secondary: str  # Beschriftungen, Sekundaertext
    text_disabled:  str  # deaktivierte Elemente
    text_inverse:   str  # Text auf dunklem Hintergrund (bei Light-Mode)

    # Semantische Farben (modus-unabhaengig – gleich in Dark und Light)
    # WICHTIG: niemals profit fuer Verluste oder loss fuer Gewinne verwenden!
    profit:  str = "#22c55e"   # Gruen  – nur fuer Gewinne / positive Werte
    loss:    str = "#ef4444"   # Rot    – nur fuer Verluste / negative Werte
    warning: str = "#f59e0b"   # Amber  – nicht-kritische Warnungen
    danger:  str = "#b91c1c"   # Dunkelrot – irreversible / kritische Aktionen
    info:    str = "#3b82f6"   # Blau   – informative Hinweise
    neutral: str = "#6b7280"   # Grau   – inaktiv / neutral

    # Interaktive Elemente
    accent:        str = "#6366f1"  # Haupt-Akzentfarbe (Indigo)
    accent_hover:  str = "#818cf8"
    accent_active: str = "#4f46e5"


DARK = ColorTokens(
    bg_base="#0f0f11",
    bg_surface="#1a1a1f",
    bg_elevated="#252530",
    border="#2e2e3a",
    border_subtle="#1e1e28",
    text_primary="#f1f1f5",
    text_secondary="#9494a8",
    text_disabled="#4a4a5a",
    text_inverse="#0f0f11",
)

LIGHT = ColorTokens(
    bg_base="#f8f8fc",
    bg_surface="#ffffff",
    bg_elevated="#f0f0f8",
    border="#d0d0e0",
    border_subtle="#e8e8f0",
    text_primary="#111118",
    text_secondary="#555568",
    text_disabled="#b0b0c0",
    text_inverse="#f1f1f5",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Typography Tokens
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TypographyTokens:
    """Typografie-Skala fuer konsistente Schriftgroessen und -gewichte."""

    family_ui:   str = "Inter, Segoe UI, system-ui, sans-serif"
    family_mono: str = "JetBrains Mono, Cascadia Code, Consolas, monospace"

    # Groessen in pt
    size_xs:  int = 9
    size_sm:  int = 11
    size_md:  int = 13
    size_lg:  int = 15
    size_xl:  int = 17
    size_2xl: int = 21
    size_3xl: int = 27

    # Gewichte (CSS-Werte)
    weight_normal:   int = 400
    weight_medium:   int = 500
    weight_semibold: int = 600
    weight_bold:     int = 700


# ─────────────────────────────────────────────────────────────────────────────
#  Spacing Tokens
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpacingTokens:
    """Abstand-System auf 4-px-Basis fuer konsistente Layouts."""

    xs:    int = 4
    sm:    int = 8
    md:    int = 12
    lg:    int = 16
    xl:    int = 20
    xxl:   int = 24
    xxxl:  int = 32
    huge:  int = 48
    giant: int = 64


# ─────────────────────────────────────────────────────────────────────────────
#  Border Radius Tokens
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RadiusTokens:
    """Eckenradien fuer Buttons, Karten und Eingabefelder."""

    sm:   int = 2
    md:   int = 4
    lg:   int = 8
    xl:   int = 12
    full: int = 9999


# ─────────────────────────────────────────────────────────────────────────────
#  Modul-Level-Instanzen (Konstanten)
# ─────────────────────────────────────────────────────────────────────────────

TYPOGRAPHY = TypographyTokens()
SPACING    = SpacingTokens()
RADIUS     = RadiusTokens()
