# app_v2_simple_player.py
import sys, os, time
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout,
    QVBoxLayout, QSlider, QSpinBox, QGroupBox, QFormLayout, QStatusBar, QStyle
)

# ---------- Worker Thread (basic playback) ----------
class VideoWorker(QThread):
    frameReady = pyqtSignal(QImage)
    fpsInfo    = pyqtSignal(float)
    opened     = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self._cap = None
        self._running = False
        self._pause = True
        self._target_fps = 30.0
        self.scale = 0.5

    def open_video(self, path: str):
        if self._cap is not None:
            try: self._cap.release()
            except: pass
            self._cap = None

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self.opened.emit(False, "Cannot open video.")
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._target_fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self.opened.emit(True, f"Opened: {os.path.basename(path)} ({self._target_fps:.1f} fps)")

    def play(self):
        self._pause = False

    def pause(self):
        self._pause = True

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        last_ts = time.time()
        frames = 0
        fps_emit_last = time.time()

        while self._running:
            if self._cap is None or not self._cap.isOpened() or self._pause:
                self.msleep(15)
                continue

            ok, frame = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Ensure 3-channel BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Scale preview
            if 0.25 <= self.scale <= 1.0:
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (int(w0 * self.scale), int(h0 * self.scale)),
                                   interpolation=cv2.INTER_AREA)

            # BGR -> RGB -> QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.frameReady.emit(qimg)

            # pacing
            frame_period = 1.0 / max(1.0, self._target_fps)
            dt = time.time() - last_ts
            if dt < frame_period:
                self.msleep(int((frame_period - dt) * 1000))
            last_ts = time.time()

            # fps
            frames += 1
            if (time.time() - fps_emit_last) >= 0.5:
                self.fpsInfo.emit(frames / (time.time() - fps_emit_last))
                fps_emit_last = time.time()
                frames = 0

        if self._cap is not None:
            try: self._cap.release()
            except: pass
            self._cap = None

# ---------- Main Window ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Video Preview")
        self.setMinimumSize(960, 540)

        # Display
        self.label = QLabel("Open a video to start…")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("QLabel{background:#111;color:#bbb;}")
        self._last_qimg = None

        # Controls
        self.openBtn  = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), " Open")
        self.playBtn  = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), " Play")
        self.pauseBtn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause), " Pause")

        self.scaleSlider = QSlider(Qt.Orientation.Horizontal)
        self.scaleSlider.setRange(25, 100); self.scaleSlider.setValue(50)
        self.scaleSpin   = QSpinBox(); self.scaleSpin.setRange(25, 100); self.scaleSpin.setValue(50)

        ctlRow = QHBoxLayout()
        ctlRow.addWidget(self.openBtn); ctlRow.addWidget(self.playBtn); ctlRow.addWidget(self.pauseBtn)
        ctlRow.addStretch(1)

        form = QFormLayout()
        sRow = QHBoxLayout(); sRow.addWidget(self.scaleSlider); sRow.addWidget(self.scaleSpin)
        form.addRow("Scale %", QWidget())
        form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(sRow)

        gb = QGroupBox("Controls")
        gbl = QVBoxLayout(gb)
        gbl.addLayout(ctlRow)
        gbl.addLayout(form)

        self.status = QStatusBar()

        root = QVBoxLayout(self)
        root.addWidget(self.label, stretch=1)
        root.addWidget(gb)
        root.addWidget(self.status)

        # Worker
        self.worker = VideoWorker()
        self.worker.frameReady.connect(self.on_frame)
        self.worker.fpsInfo.connect(self.on_fps)
        self.worker.opened.connect(self.on_opened)
        self.worker.start()

        # Wiring
        self.openBtn.clicked.connect(self.choose_file)
        self.playBtn.clicked.connect(self.worker.play)
        self.pauseBtn.clicked.connect(self.worker.pause)

        self.scaleSlider.valueChanged.connect(self.scaleSpin.setValue)
        self.scaleSpin.valueChanged.connect(self.scaleSlider.setValue)
        self.scaleSlider.valueChanged.connect(self._push_scale)

        self._paused = True

    def on_frame(self, qimg: QImage):
        self._last_qimg = qimg
        self._set_pixmap_scaled()

    def resizeEvent(self, e):
        self._set_pixmap_scaled()
        return super().resizeEvent(e)

    def on_fps(self, fps: float):
        self.status.showMessage(f"~{fps:.1f} FPS")

    def on_opened(self, ok: bool, msg: str):
        self.status.showMessage(msg)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "",
                                              "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if not path:
            return
        self.worker.open_video(path)
        self.worker.play()

    def _push_scale(self, *_):
        self.worker.scale = max(0.25, min(1.0, self.scaleSlider.value()/100.0))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._paused = not self._paused
            if self._paused: self.worker.pause()
            else: self.worker.play()
            e.accept()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.worker.stop()
        self.worker.wait(1000)
        return super().closeEvent(e)

    def _set_pixmap_scaled(self):
        if self._last_qimg is None:
            return
        target = self.label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        pix = QPixmap.fromImage(self._last_qimg).scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(pix)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
