# app_v1_minimal.py
import sys
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QGroupBox, QStatusBar, QStyle
)

# ---------- Worker Thread (idle skeleton) ----------
@dataclass
class ProcParams:
    mode: str = "Both"
    black: int = 16
    white: int = 235
    scale: float = 0.5
    phase_step: int = 2

class VideoWorker(QThread):
    frameReady = pyqtSignal(object)  # will be QImage later
    fpsInfo    = pyqtSignal(float)
    opened     = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self._running = False

    def play(self):  # placeholder
        pass

    def pause(self):  # placeholder
        pass

    def stop(self):
        self._running = False

    def run(self):
        # Intentionally idle for now
        self._running = True
        while self._running:
            self.msleep(50)

# ---------- Main Window ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zebra Tool — Minimal Init")
        self.setMinimumSize(800, 450)

        # Video display (placeholder)
        self.label = QLabel("Ready. (No video logic yet)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("QLabel{background:#111;color:#bbb;}")

        # Basic controls (wired but no logic)
        self.playBtn  = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), " Play")
        self.pauseBtn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause), " Pause")

        ctlRow = QHBoxLayout()
        ctlRow.addWidget(self.playBtn)
        ctlRow.addWidget(self.pauseBtn)
        ctlBox = QGroupBox("Controls")
        v = QVBoxLayout(ctlBox)
        v.addLayout(ctlRow)

        self.status = QStatusBar()
        self.status.showMessage("Initialized")

        root = QVBoxLayout(self)
        root.addWidget(self.label, stretch=1)
        root.addWidget(ctlBox)
        root.addWidget(self.status)

        # Worker thread (idle)
        self.worker = VideoWorker()
        self.worker.start()

        # Wire buttons to placeholders
        self.playBtn.clicked.connect(self.worker.play)
        self.pauseBtn.clicked.connect(self.worker.pause)

    def closeEvent(self, e):
        self.worker.stop()
        self.worker.wait(500)
        return super().closeEvent(e)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
