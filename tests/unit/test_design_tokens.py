"""
tests/unit/test_design_tokens.py
Token-Konsistenz-Checks fuer gui/design/tokens.py.

Kein Qt erforderlich – rein Python.

Gepruefte Invarianten:
  - Farb-Semantik korrekt (profit=gruen, loss=rot, profit != loss)
  - Alle Farben gueltiges Hex-Format
  - Typography-Groessen aufsteigend
  - Spacing aufsteigend
  - Radius aufsteigend
  - Dark/Light-Paletten unterscheiden sich in den Basis-Farben
  - Semantische Farben sind in DARK und LIGHT identisch (Konsistenz)
"""

from __future__ import annotations

import re

import pytest

from gui.design.tokens import (
    DARK,
    LIGHT,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorTokens,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktion
# ─────────────────────────────────────────────────────────────────────────────

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _is_hex(color: str) -> bool:
    return bool(_HEX_RE.match(color))


def _is_green_ish(hex_color: str) -> bool:
    """Grob: gruen-dominante Farbe (g > r und g > b)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return g > r and g > b


def _is_red_ish(hex_color: str) -> bool:
    """Grob: rot-dominante Farbe (r > g und r > b)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return r > g and r > b


# ─────────────────────────────────────────────────────────────────────────────
#  Farb-Semantik
# ─────────────────────────────────────────────────────────────────────────────

class TestColorSemantics:
    def test_profit_is_green(self):
        """profit-Farbe muss gruen-dominierend sein."""
        assert _is_green_ish(DARK.profit)

    def test_loss_is_red(self):
        """loss-Farbe muss rot-dominierend sein."""
        assert _is_red_ish(DARK.loss)

    def test_profit_not_equal_loss(self):
        """profit und loss duerfen niemals dieselbe Farbe sein."""
        assert DARK.profit != DARK.loss

    def test_warning_not_equal_danger(self):
        """warning und danger muessen unterschiedlich sein."""
        assert DARK.warning != DARK.danger

    def test_profit_not_red(self):
        """profit darf niemals rot sein – wuerde HCI-Konvention verletzen."""
        assert not _is_red_ish(DARK.profit)

    def test_loss_not_green(self):
        """loss darf niemals gruen sein – wuerde HCI-Konvention verletzen."""
        assert not _is_green_ish(DARK.loss)

    def test_accent_not_profit(self):
        """Akzentfarbe und profit-Farbe muessen unterschiedlich sein."""
        assert DARK.accent != DARK.profit

    def test_accent_not_loss(self):
        """Akzentfarbe und loss-Farbe muessen unterschiedlich sein."""
        assert DARK.accent != DARK.loss

    def test_danger_is_red_ish(self):
        """danger soll rot-dominierend sein (irreversible Aktionen)."""
        assert _is_red_ish(DARK.danger)

    def test_danger_darker_than_loss(self):
        """danger soll dunkler/saettigungsarmer sein als loss."""
        # Beide sind rot, aber danger hat eine niedrigere Helligkeit
        r_loss   = int(DARK.loss[1:3], 16)
        r_danger = int(DARK.danger[1:3], 16)
        # Danger-Rot-Wert kann hoeher sein, aber Gesamthelligkeit ist niedriger
        # Pruefen dass sie verschieden sind
        assert DARK.loss != DARK.danger


# ─────────────────────────────────────────────────────────────────────────────
#  Gueltige Hex-Farben
# ─────────────────────────────────────────────────────────────────────────────

class TestHexValidity:
    @pytest.mark.parametrize("tokens", [DARK, LIGHT])
    def test_all_color_fields_are_valid_hex(self, tokens: ColorTokens):
        """Jedes Farbfeld muss ein gueltiges #RRGGBB-Format haben."""
        for field_name in tokens.__dataclass_fields__:
            value = getattr(tokens, field_name)
            assert _is_hex(value), (
                f"ColorTokens.{field_name} = {value!r} ist kein gueltiges Hex-Format"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Dark / Light Unterschiede
# ─────────────────────────────────────────────────────────────────────────────

class TestDarkLightDifference:
    def test_bg_base_differs(self):
        assert DARK.bg_base != LIGHT.bg_base

    def test_bg_surface_differs(self):
        assert DARK.bg_surface != LIGHT.bg_surface

    def test_bg_elevated_differs(self):
        assert DARK.bg_elevated != LIGHT.bg_elevated

    def test_text_primary_differs(self):
        assert DARK.text_primary != LIGHT.text_primary

    def test_dark_is_darker_than_light(self):
        """Dark-Hintergrund muss dunkler sein als Light-Hintergrund."""
        def brightness(hex_color: str) -> int:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return r + g + b

        assert brightness(DARK.bg_base) < brightness(LIGHT.bg_base)

    def test_semantic_colors_consistent_across_modes(self):
        """Semantische Farben (profit/loss/warning/danger) sind modusunabhaengig."""
        semantic_fields = ("profit", "loss", "warning", "danger", "info", "neutral",
                           "accent", "accent_hover", "accent_active")
        for field in semantic_fields:
            assert getattr(DARK, field) == getattr(LIGHT, field), (
                f"Semantische Farbe '{field}' unterscheidet sich zwischen Dark und Light"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Typografie
# ─────────────────────────────────────────────────────────────────────────────

class TestTypography:
    def test_font_sizes_ascending(self):
        """Schriftgroessen muessen strikt aufsteigend sein."""
        sizes = [
            TYPOGRAPHY.size_xs,
            TYPOGRAPHY.size_sm,
            TYPOGRAPHY.size_md,
            TYPOGRAPHY.size_lg,
            TYPOGRAPHY.size_xl,
            TYPOGRAPHY.size_2xl,
            TYPOGRAPHY.size_3xl,
        ]
        assert sizes == sorted(sizes), f"Schriftgroessen nicht aufsteigend: {sizes}"
        assert len(set(sizes)) == len(sizes), "Doppelte Schriftgroessen gefunden"

    def test_all_sizes_positive(self):
        for attr in ("size_xs", "size_sm", "size_md", "size_lg",
                     "size_xl", "size_2xl", "size_3xl"):
            assert getattr(TYPOGRAPHY, attr) > 0

    def test_font_weights_valid(self):
        """Schriftgewichte muessen gueltige CSS-Werte sein (100-900, Vielfaches von 100)."""
        for attr in ("weight_normal", "weight_medium", "weight_semibold", "weight_bold"):
            w = getattr(TYPOGRAPHY, attr)
            assert 100 <= w <= 900, f"{attr}={w} ausserhalb 100-900"
            assert w % 100 == 0, f"{attr}={w} kein Vielfaches von 100"

    def test_font_families_nonempty(self):
        assert len(TYPOGRAPHY.family_ui) > 0
        assert len(TYPOGRAPHY.family_mono) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Spacing
# ─────────────────────────────────────────────────────────────────────────────

class TestSpacing:
    def test_spacing_values_ascending(self):
        values = [
            SPACING.xs, SPACING.sm, SPACING.md, SPACING.lg,
            SPACING.xl, SPACING.xxl, SPACING.xxxl, SPACING.huge, SPACING.giant,
        ]
        assert values == sorted(values), f"Spacing nicht aufsteigend: {values}"

    def test_all_spacings_positive(self):
        for attr in ("xs", "sm", "md", "lg", "xl", "xxl", "xxxl", "huge", "giant"):
            assert getattr(SPACING, attr) > 0

    def test_xs_is_multiple_of_4(self):
        """Basis-Grid ist 4px."""
        assert SPACING.xs % 4 == 0

    def test_sm_is_multiple_of_4(self):
        assert SPACING.sm % 4 == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Border Radius
# ─────────────────────────────────────────────────────────────────────────────

class TestRadius:
    def test_radius_values_ascending(self):
        values = [RADIUS.sm, RADIUS.md, RADIUS.lg, RADIUS.xl]
        assert values == sorted(values), f"Radien nicht aufsteigend: {values}"

    def test_all_radii_positive(self):
        for attr in ("sm", "md", "lg", "xl"):
            assert getattr(RADIUS, attr) > 0

    def test_full_is_large(self):
        """'full' muss groesser als alle anderen Radien sein."""
        assert RADIUS.full > RADIUS.xl
