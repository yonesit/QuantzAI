"""
scripts/reset_paper_state.py
Setzt den Paper-Trading-Status fuer einen sauberen Teststart zurueck.

Was wird zurueckgesetzt (mit --confirm):
  data/processed/paper_trades.json  → leere Liste []
  data/processed/audit.db           → alle Tabellen geleert (Struktur bleibt erhalten)
  data/processed/quality_reports/   → alle *.json Berichte geloescht

Was NICHT angefasst wird:
  models/*.joblib      – trainierte Modelle
  data/features/       – berechnete Feature-Parquets
  data/processed/calendar/ – Wirtschaftskalender-Cache
  config/              – Konfiguration
  .env                 – Zugangsdaten

Verwendung:
  python scripts/reset_paper_state.py           # Dry-Run: zeigt was geloescht wuerde
  python scripts/reset_paper_state.py --confirm # Fuehrt den Reset tatsaechlich aus
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_PAPER_TRADES = _ROOT / "data" / "processed" / "paper_trades.json"
_AUDIT_DB     = _ROOT / "data" / "processed" / "audit.db"
_QUALITY_DIR  = _ROOT / "data" / "processed" / "quality_reports"

_AUDIT_TABLES = ["orders", "shadow_trades", "errors", "emergencies"]


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse-Funktionen (kein I/O ausser Lesen)
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_paper_trades() -> dict:
    if not _PAPER_TRADES.exists():
        return {"exists": False, "total": 0, "open": 0, "closed": 0, "trades": []}
    try:
        trades = json.loads(_PAPER_TRADES.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "total": 0, "open": 0, "closed": 0, "trades": []}
    open_t   = [t for t in trades if t.get("status") == "open"]
    closed_t = [t for t in trades if t.get("status") == "closed"]
    return {
        "exists":  True,
        "total":   len(trades),
        "open":    len(open_t),
        "closed":  len(closed_t),
        "trades":  trades,
    }


def _analyse_audit_db() -> dict:
    if not _AUDIT_DB.exists():
        return {"exists": False, "counts": {}}
    try:
        conn   = sqlite3.connect(str(_AUDIT_DB))
        counts = {}
        for tbl in _AUDIT_TABLES:
            try:
                counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[tbl] = 0
        conn.close()
        return {"exists": True, "counts": counts}
    except Exception:
        return {"exists": True, "counts": {}}


def _analyse_quality_reports() -> dict:
    if not _QUALITY_DIR.exists():
        return {"exists": False, "count": 0}
    files = list(_QUALITY_DIR.glob("*.json"))
    return {"exists": True, "count": len(files)}


# ─────────────────────────────────────────────────────────────────────────────
#  Reset-Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def _reset_paper_trades() -> None:
    """Schreibt eine leere Liste in paper_trades.json."""
    _PAPER_TRADES.parent.mkdir(parents=True, exist_ok=True)
    _PAPER_TRADES.write_text("[]", encoding="utf-8")
    print(f"  OK {_PAPER_TRADES.relative_to(_ROOT)} --> []")


def _reset_audit_db() -> None:
    """Leert alle Tabellen in audit.db (Tabellenstruktur bleibt erhalten)."""
    if not _AUDIT_DB.exists():
        print(f"  -- {_AUDIT_DB.relative_to(_ROOT)} existiert nicht, uebersprungen")
        return
    conn = sqlite3.connect(str(_AUDIT_DB))
    for tbl in _AUDIT_TABLES:
        try:
            conn.execute(f"DELETE FROM {tbl}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()
    # VACUUM muss ausserhalb einer Transaktion laufen
    conn2 = sqlite3.connect(str(_AUDIT_DB))
    conn2.execute("VACUUM")
    conn2.close()
    print(f"  OK {_AUDIT_DB.relative_to(_ROOT)} --> alle Tabellen geleert")


def _reset_quality_reports() -> None:
    """Loescht alle quality_report JSON-Dateien."""
    if not _QUALITY_DIR.exists():
        return
    files = list(_QUALITY_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    print(f"  OK {_QUALITY_DIR.relative_to(_ROOT)}/ --> {len(files)} Dateien geloescht")


# ─────────────────────────────────────────────────────────────────────────────
#  Hauptlogik
# ─────────────────────────────────────────────────────────────────────────────

def _print_preview() -> None:
    """Zeigt eine Zusammenfassung dessen was zurueckgesetzt wuerden."""
    paper  = _analyse_paper_trades()
    audit  = _analyse_audit_db()
    qr     = _analyse_quality_reports()

    print("\n" + "=" * 60)
    print("  PAPER-STATE RESET -- VORSCHAU (Dry-Run)")
    print("=" * 60)

    print("\n[paper_trades.json]")
    if paper["exists"]:
        print(f"   {paper['total']} Trades gesamt | {paper['open']} offen | {paper['closed']} geschlossen")
        for t in paper["trades"]:
            status = t.get("status", "?")
            ticket = t.get("ticket", "?")
            sym    = t.get("symbol", "?")
            opened = t.get("open_time", "?")[:19]
            pnl    = t.get("pnl")
            pnl_s  = f"  PnL={pnl:+.2f}" if pnl is not None else ""
            print(f"   #{ticket} {sym} {status} | Eroeffnet: {opened}{pnl_s}")
        print("   --> wird auf [] zurueckgesetzt")
    else:
        print("   (Datei existiert nicht)")

    print("\n[audit.db]")
    if audit["exists"]:
        for tbl, cnt in audit["counts"].items():
            print(f"   {tbl}: {cnt} Eintraege")
        print("   --> alle Tabellen werden geleert (Struktur bleibt)")
    else:
        print("   (Datei existiert nicht)")

    print("\n[quality_reports/]")
    if qr["exists"]:
        print(f"   {qr['count']} JSON-Dateien --> werden geloescht")
    else:
        print("   (Verzeichnis existiert nicht)")

    print("\n[NICHT angefasst]")
    print("   models/*.joblib | data/features/ | data/processed/calendar/ | config/ | .env")
    print("\n" + "-" * 60)
    print("  Starte mit --confirm um den Reset tatsaechlich auszufuehren.")
    print("-" * 60 + "\n")


def _run_reset() -> None:
    """Fuehrt den Reset tatsaechlich aus."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'=' * 60}")
    print(f"  PAPER-STATE RESET -- AUSGEFUEHRT ({ts})")
    print(f"{'=' * 60}\n")

    _reset_paper_trades()
    _reset_audit_db()
    _reset_quality_reports()

    print(f"\n{'-' * 60}")
    print("  Reset abgeschlossen. Bot kann jetzt sauber starten.")
    print(f"{'-' * 60}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Setzt den Paper-Trading-Status zurueck fuer einen sauberen Teststart."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Fuehrt den Reset tatsaechlich aus (Standard: nur Dry-Run/Vorschau).",
    )
    args = parser.parse_args(argv)

    if args.confirm:
        _print_preview()
        _run_reset()
    else:
        _print_preview()

    return 0


if __name__ == "__main__":
    sys.exit(main())
