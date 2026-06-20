"""
gui/widgets/bot_controls_widget.py
BotControlsWidget – GUI-Steuerung des TradingOrchestrators.

Ermoeglicht Start / Stop / Pause / Resume des Orchestrators aus der GUI heraus.
Der Bot laeuft in einem QThread – die GUI bleibt stets bedienbar.

Komponenten:
  BotState           – Enum: STOPPED | RUNNING | PAUSED | STOPPING
  BotWorker          – QObject: fuehrt orchestrator.run_loop() im Hintergrund aus
  BotControlsWidget  – Oeffentliches Widget: Buttons + Status-Indicator + Modus-Auswahl

Sicherheiten:
  - Stop signalisiert dem Orchestrator sauber (stop_event); laufende Zyklen enden
    nach Abschluss des aktuellen Symbols (kein Abbruch mitten in einer Order)
  - STOPPING-Zustand verhindert Doppelstart waehrend Thread noch laeuft
  - cleanup() wartet max 3 s auf Thread-Ende – kein Zombie-Prozess beim App-Close
  - Mode-Wechsel zu AUTONOMOUS nur wenn CONFIRM_AUTONOMOUS=yes gesetzt;
    bei EnvironmentError wird Combo automatisch zurueckgesetzt
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from loguru import logger

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
#  BotState
# ─────────────────────────────────────────────────────────────────────────────

class BotState(Enum):
    STOPPED  = "Gestoppt"
    RUNNING  = "Aktiv"
    PAUSED   = "Pausiert"
    STOPPING = "Stoppt..."


# ─────────────────────────────────────────────────────────────────────────────
#  Modus-Mapping (GUI-Label -> src.modes.TradingMode-Name)
# ─────────────────────────────────────────────────────────────────────────────

_MODE_LABELS: list[tuple[str, str]] = [
    ("Vorschlag",    "SUGGEST_ONLY"),
    ("Bestätigung",  "CONFIRM_REQUIRED"),
    ("Autonom",      "AUTONOMOUS"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  BotWorker
# ─────────────────────────────────────────────────────────────────────────────

class BotWorker(QObject):
    """
    Fuehrt orchestrator.run_loop() im Hintergrund-Thread aus.

    Alle Signale werden via Qt-Queued-Connection im Hauptthread empfangen,
    sodass UI-Updates thread-sicher sind.
    """

    stopped          = Signal()
    error_occurred   = Signal(str)
    cycle_completed  = Signal(object)   # dict aus run_cycle()

    def __init__(
        self,
        orchestrator,
        symbols: list[str],
        interval_seconds: int = 300,
    ) -> None:
        super().__init__()
        self._orchestrator  = orchestrator
        self._symbols       = symbols
        self._interval      = interval_seconds

    @Slot()
    def run(self) -> None:
        """Blockierender Aufruf von run_loop – laeuft im Hintergrund-Thread."""
        if hasattr(self._orchestrator, "set_activity_callback"):
            self._orchestrator.set_activity_callback(self.cycle_completed.emit)
        try:
            self._orchestrator.run_loop(
                self._symbols,
                interval_seconds=self._interval,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("BotWorker: unbehandelte Exception: {e}", e=exc)
            self.error_occurred.emit(str(exc))
        finally:
            if hasattr(self._orchestrator, "set_activity_callback"):
                self._orchestrator.set_activity_callback(None)
            self.stopped.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  BotControlsWidget
# ─────────────────────────────────────────────────────────────────────────────

class BotControlsWidget(QWidget):
    """
    Widget zur Bot-Steuerung aus der GUI heraus.

    Zustandsmaschine:
      STOPPED  --[Start]--> RUNNING
      RUNNING  --[Pause]--> PAUSED
      RUNNING  --[Stop]-->  STOPPING --> (Thread endet) --> STOPPED
      PAUSED   --[Resume]-> RUNNING
      PAUSED   --[Stop]-->  STOPPING --> (Thread endet) --> STOPPED

    Signale
    -------
    state_changed(BotState)  – bei jedem Zustandsuebergang
    error_occurred(str)      – wenn der BotWorker eine unbehandelte Exception wirft

    Parameters
    ----------
    orchestrator     : TradingOrchestrator-Instanz oder None.
                       Wenn None, sind alle Buttons deaktiviert.
    symbols          : Liste der zu handelnden Symbole (Standard: []).
    interval_seconds : Wartezeit zwischen Zyklen in Sekunden (Standard: 300).
    parent           : Qt-Elternobjekt.
    """

    state_changed    = Signal(object)   # BotState
    error_occurred   = Signal(str)
    cycle_completed  = Signal(object)   # dict aus run_cycle() – weitergleitet von BotWorker

    def __init__(
        self,
        orchestrator=None,
        symbols: Optional[list[str]] = None,
        interval_seconds: int = 300,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("bot_controls_widget")

        self._orchestrator  = orchestrator
        self._symbols       = list(symbols) if symbols else []
        self._interval      = interval_seconds
        self._state         = BotState.STOPPED
        self._thread: Optional[QThread]   = None
        self._worker: Optional[BotWorker] = None

        self._build()
        self._refresh_status_indicator()
        self._update_buttons()

    # ── Builder ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel("🤖  Bot-Steuerung")
        title.setObjectName("bot_controls_title")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        outer.addWidget(title)

        # Status-Indicator
        self._status_dot   = QLabel("●")
        self._status_label = QLabel(self._state.value)
        self._status_dot.setObjectName("bot_status_dot")
        self._status_label.setObjectName("bot_status_label")

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        outer.addLayout(status_row)

        # Modus-Auswahl
        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("bot_mode_combo")
        for label, _ in _MODE_LABELS:
            self._mode_combo.addItem(label)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        outer.addWidget(self._mode_combo)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._start_btn        = QPushButton("▶ Start")
        self._stop_btn         = QPushButton("■ Stop")
        self._pause_resume_btn = QPushButton("⏸ Pause")

        self._start_btn.setObjectName("bot_start_btn")
        self._stop_btn.setObjectName("bot_stop_btn")
        self._pause_resume_btn.setObjectName("bot_pause_resume_btn")

        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._pause_resume_btn.clicked.connect(self._on_pause_resume)

        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._pause_resume_btn)
        outer.addLayout(btn_row)

    # ── Oeffentliche API ──────────────────────────────────────────────────────

    def set_orchestrator(
        self,
        orchestrator,
        symbols: list[str],
        interval_seconds: int = 300,
    ) -> None:
        """
        Setzt/wechselt den Orchestrator (nur im STOPPED-Zustand erlaubt).

        Raises RuntimeError wenn der Bot noch laeuft.
        """
        if self._state != BotState.STOPPED:
            raise RuntimeError(
                "Orchestrator kann nicht gewechselt werden waehrend Bot laeuft."
            )
        self._orchestrator = orchestrator
        self._symbols      = list(symbols)
        self._interval     = interval_seconds
        self._update_buttons()

    def cleanup(self) -> None:
        """
        Beendet den Hintergrund-Thread sauber.
        Muss im closeEvent des MainWindow aufgerufen werden.
        """
        if (
            self._orchestrator is not None
            and self._state not in (BotState.STOPPED, BotState.STOPPING)
        ):
            if self._state == BotState.PAUSED:
                self._orchestrator.resume()
            self._orchestrator.stop()

        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(3000):
                logger.warning(
                    "BotWorker: Thread hat nicht innerhalb von 3 s beendet – terminate()."
                )
                self._thread.terminate()

        self._thread = None
        self._worker = None
        self._set_state(BotState.STOPPED)

    @property
    def bot_state(self) -> BotState:
        """Aktueller Zustand des Bots."""
        return self._state

    def start(self) -> None:
        """Startet den Bot programmatisch (z. B. durch WatchdogService)."""
        self._on_start()

    # ── Button-Slots ──────────────────────────────────────────────────────────

    @Slot()
    def _on_start(self) -> None:
        if self._orchestrator is None or self._state != BotState.STOPPED:
            return

        logger.info(
            "BotControlsWidget: Start | Symbole={s} | Intervall={i}s",
            s=self._symbols, i=self._interval,
        )

        self._thread = QThread()
        self._worker = BotWorker(
            self._orchestrator,
            self._symbols,
            self._interval,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._worker.stopped.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.cycle_completed.connect(self.cycle_completed)

        self._thread.start()
        self._set_state(BotState.RUNNING)

    @Slot()
    def _on_stop(self) -> None:
        if self._orchestrator is None or self._state in (
            BotState.STOPPED, BotState.STOPPING
        ):
            return

        logger.info("BotControlsWidget: Stop")
        if self._state == BotState.PAUSED:
            self._orchestrator.resume()
        self._orchestrator.stop()
        self._set_state(BotState.STOPPING)

    @Slot()
    def _on_pause_resume(self) -> None:
        if self._orchestrator is None:
            return

        if self._state == BotState.RUNNING:
            self._orchestrator.pause("GUI")
            logger.info("BotControlsWidget: Pause")
            self._set_state(BotState.PAUSED)

        elif self._state == BotState.PAUSED:
            self._orchestrator.resume()
            logger.info("BotControlsWidget: Resume")
            self._set_state(BotState.RUNNING)

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        if self._orchestrator is None:
            return

        _, mode_name = _MODE_LABELS[index]
        try:
            from src.modes import TradingMode
            new_mode = TradingMode[mode_name]
            self._orchestrator.set_mode(new_mode)
            logger.info("Bot-Modus geaendert: {m}", m=new_mode.value)
        except EnvironmentError as exc:
            QMessageBox.warning(self, "Modus nicht erlaubt", str(exc))
            self._reset_mode_combo()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Modus-Wechsel fehlgeschlagen: {e}", e=exc)
            self._reset_mode_combo()

    # ── Worker-Slots ──────────────────────────────────────────────────────────

    @Slot()
    def _on_worker_stopped(self) -> None:
        logger.info("BotWorker: gestoppt")
        self._set_state(BotState.STOPPED)

    @Slot()
    def _on_thread_finished(self) -> None:
        # Fires after the OS thread has terminated – safe to drop Python refs now.
        # Releasing self._thread before finished would destroy QThread while its
        # event loop is still running, causing a Qt abort/crash.
        self._thread = None
        self._worker = None

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        logger.error("BotWorker: Fehler: {m}", m=message)
        self.error_occurred.emit(message)

    # ── Interna ───────────────────────────────────────────────────────────────

    def _set_state(self, new_state: BotState) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        logger.debug(
            "BotControlsWidget: {old} -> {new}",
            old=old.name, new=new_state.name,
        )
        self._refresh_status_indicator()
        self._update_buttons()
        self.state_changed.emit(new_state)

    def _refresh_status_indicator(self) -> None:
        colors = {
            BotState.STOPPED:  "#6b7280",
            BotState.RUNNING:  "#22c55e",
            BotState.PAUSED:   "#f59e0b",
            BotState.STOPPING: "#ef4444",
        }
        color = colors[self._state]
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 9pt;")
        self._status_label.setText(self._state.value)

    def _update_buttons(self) -> None:
        has_orc  = self._orchestrator is not None
        stopped  = self._state == BotState.STOPPED
        running  = self._state == BotState.RUNNING
        paused   = self._state == BotState.PAUSED
        stopping = self._state == BotState.STOPPING

        self._start_btn.setEnabled(has_orc and stopped)
        self._stop_btn.setEnabled(has_orc and (running or paused))
        self._pause_resume_btn.setEnabled(has_orc and (running or paused))
        self._mode_combo.setEnabled(has_orc and not stopping)

        if paused:
            self._pause_resume_btn.setText("▶ Weiter")
        else:
            self._pause_resume_btn.setText("⏸ Pause")

    def _reset_mode_combo(self) -> None:
        """Setzt Combo auf den aktuellen Orchestrator-Modus zurueck."""
        if self._orchestrator is None:
            return
        try:
            current_name = self._orchestrator.mode.name
            idx = next(
                (i for i, (_, n) in enumerate(_MODE_LABELS) if n == current_name),
                0,
            )
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(idx)
            self._mode_combo.blockSignals(False)
        except Exception:  # noqa: BLE001
            pass
