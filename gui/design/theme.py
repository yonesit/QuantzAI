"""
gui/design/theme.py
Theme-System: Dark-/Light-Modus, QSS-Generierung, ThemeManager.

ThemeManager ist kein Qt-Objekt – reines Python mit Callback-Pattern.
Dadurch testbar ohne laufende QApplication.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable

from gui.design.tokens import DARK, LIGHT, ColorTokens, TYPOGRAPHY, SPACING, RADIUS


class ThemeMode(Enum):
    DARK  = auto()
    LIGHT = auto()


# ─────────────────────────────────────────────────────────────────────────────
#  QSS-Generierung
# ─────────────────────────────────────────────────────────────────────────────

def build_stylesheet(colors: ColorTokens) -> str:
    """Generiert globales QSS-Stylesheet aus ColorTokens."""
    t = TYPOGRAPHY
    s = SPACING
    r = RADIUS
    return f"""
/* QuantzAI Global Stylesheet */
QWidget {{
    background-color: {colors.bg_base};
    color: {colors.text_primary};
    font-family: "{t.family_ui}";
    font-size: {t.size_md}pt;
    border: none;
    outline: none;
}}
QMainWindow {{
    background-color: {colors.bg_base};
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: {colors.bg_surface};
    width: 8px;
    border-radius: {r.sm}px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {colors.border};
    border-radius: {r.sm}px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QPushButton {{
    background-color: {colors.accent};
    color: {colors.text_inverse};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.lg}px;
    font-size: {t.size_sm}pt;
    font-weight: {t.weight_medium};
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: {colors.accent_hover};
}}
QPushButton:pressed {{
    background-color: {colors.accent_active};
}}
QPushButton:disabled {{
    background-color: {colors.bg_elevated};
    color: {colors.text_disabled};
}}
QPushButton[secondary="true"] {{
    background-color: {colors.bg_elevated};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
}}
QPushButton[secondary="true"]:hover {{
    background-color: {colors.border};
}}
QPushButton[danger="true"] {{
    background-color: {colors.danger};
    color: #ffffff;
}}
QPushButton[danger="true"]:hover {{
    background-color: #dc2626;
}}
QPushButton[nav="true"] {{
    background-color: transparent;
    color: {colors.text_secondary};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.lg}px;
    text-align: left;
    font-size: {t.size_sm}pt;
}}
QPushButton[nav="true"]:hover {{
    background-color: {colors.bg_elevated};
    color: {colors.text_primary};
}}
QPushButton[nav="true"]:checked {{
    background-color: {colors.accent};
    color: #ffffff;
    font-weight: {t.weight_medium};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {colors.bg_surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    border-radius: {r.md}px;
    padding: {s.sm}px {s.md}px;
    font-size: {t.size_md}pt;
    selection-background-color: {colors.accent};
    min-height: 26px;
}}
QTextEdit, QPlainTextEdit {{
    background-color: {colors.bg_surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    border-radius: {r.md}px;
    padding: {s.sm}px;
    font-family: "{t.family_mono}";
    font-size: {t.size_sm}pt;
    selection-background-color: {colors.accent};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {colors.accent};
}}
QComboBox::drop-down {{
    border: none;
    width: {s.xl}px;
}}
QComboBox QAbstractItemView {{
    background-color: {colors.bg_elevated};
    border: 1px solid {colors.border};
    border-radius: {r.md}px;
    selection-background-color: {colors.accent};
    outline: none;
}}
QTableWidget, QTableView {{
    background-color: {colors.bg_surface};
    border: 1px solid {colors.border};
    border-radius: {r.md}px;
    gridline-color: {colors.border_subtle};
    selection-background-color: {colors.accent};
    alternate-background-color: {colors.bg_elevated};
    outline: none;
}}
QHeaderView::section {{
    background-color: {colors.bg_elevated};
    color: {colors.text_secondary};
    border: none;
    border-bottom: 1px solid {colors.border};
    border-right: 1px solid {colors.border_subtle};
    padding: {s.sm}px {s.md}px;
    font-size: {t.size_sm}pt;
    font-weight: {t.weight_semibold};
}}
QStatusBar {{
    background-color: {colors.bg_surface};
    border-top: 1px solid {colors.border};
    color: {colors.text_secondary};
    font-size: {t.size_sm}pt;
}}
QToolTip {{
    background-color: {colors.bg_elevated};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    border-radius: {r.sm}px;
    padding: {s.xs}px {s.sm}px;
    font-size: {t.size_sm}pt;
}}
QDialog {{
    background-color: {colors.bg_surface};
}}
QFrame[sidebar_sep="true"] {{
    background-color: {colors.border};
    max-width: 1px;
}}
QLabel {{
    background-color: transparent;
    color: {colors.text_primary};
}}
QLabel[secondary="true"] {{
    color: {colors.text_secondary};
    font-size: {t.size_sm}pt;
}}
QCheckBox, QRadioButton {{
    color: {colors.text_primary};
    spacing: {s.sm}px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {colors.border};
    border-radius: {r.sm}px;
    background-color: {colors.bg_surface};
}}
QCheckBox::indicator:checked {{
    background-color: {colors.accent};
    border-color: {colors.accent};
}}
QSplitter::handle {{
    background-color: {colors.border};
}}
QTabWidget::pane {{
    border: 1px solid {colors.border};
    border-radius: {r.md}px;
    background-color: {colors.bg_surface};
}}
QTabBar::tab {{
    background-color: {colors.bg_elevated};
    color: {colors.text_secondary};
    padding: {s.sm}px {s.lg}px;
    border-top-left-radius: {r.md}px;
    border-top-right-radius: {r.md}px;
    font-size: {t.size_sm}pt;
}}
QTabBar::tab:selected {{
    background-color: {colors.accent};
    color: #ffffff;
}}
QTabBar::tab:hover {{
    background-color: {colors.border};
    color: {colors.text_primary};
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  ThemeManager
# ─────────────────────────────────────────────────────────────────────────────

class ThemeManager:
    """
    Verwaltet den aktuellen Dark-/Light-Modus.

    Kein QObject – reines Python mit Callback-Pattern.
    Konsumenten registrieren sich mit `on_theme_changed(callback)`.
    """

    def __init__(self, mode: ThemeMode = ThemeMode.DARK) -> None:
        self._mode = mode
        self._callbacks: list[Callable[[str], None]] = []

    # ── Eigenschaften ─────────────────────────────────────────────────────────

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def colors(self) -> ColorTokens:
        return DARK if self._mode is ThemeMode.DARK else LIGHT

    def stylesheet(self) -> str:
        return build_stylesheet(self.colors)

    # ── Steuerung ─────────────────────────────────────────────────────────────

    def set_mode(self, mode: ThemeMode) -> None:
        """Setzt den Modus und benachrichtigt alle Konsumenten."""
        if mode != self._mode:
            self._mode = mode
            self._notify()

    def toggle(self) -> None:
        """Wechselt zwischen Dark und Light."""
        new = ThemeMode.LIGHT if self._mode is ThemeMode.DARK else ThemeMode.DARK
        self.set_mode(new)

    # ── Observer-Pattern ──────────────────────────────────────────────────────

    def on_theme_changed(self, callback: Callable[[str], None]) -> None:
        """Registriert einen Callback der bei Theme-Wechsel aufgerufen wird."""
        self._callbacks.append(callback)

    def _notify(self) -> None:
        qss = self.stylesheet()
        for cb in self._callbacks:
            cb(qss)


# ─────────────────────────────────────────────────────────────────────────────
#  Globale Instanz (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_default_manager: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    """Gibt die globale ThemeManager-Instanz zurueck (lazy erstellt)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ThemeManager()
    return _default_manager


def reset_theme_manager() -> None:
    """Fuer Tests: globalen ThemeManager zuruecksetzen."""
    global _default_manager
    _default_manager = None
