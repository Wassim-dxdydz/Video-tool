import sys, os, time
from dataclasses import dataclass
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout,
    QVBoxLayout, QSlider, QComboBox, QGroupBox, QFormLayout, QMessageBox, QStatusBar, QSizePolicy
)

from processing import zebra_overlay, safe_frame_bgr

@dataclass
class ProcParams:
    mode: str = "Both"
    black: int = 16
    white: int = 235
    scale: float = 0.5
    phase_step: int = 2

class VideoWorker(QThread):
    framePairReady = pyqtSignal(QImage, QImage)
    fpsInfo        = pyqtSignal(float)
    opened         = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self._cap = None
        self._running = False
        self._pause = True
        self._phase = 0
        self._params = ProcParams()
        self._mutex = QMutex()
        self._target_fps = 30.0
        self._path = None

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
        self._path = path
        self._phase = 0
        self.opened.emit(True, f"Opened: {os.path.basename(path)} ({self._target_fps:.2f} fps)")

    def set_params(self, **kwargs):
        with QMutexLocker(self._mutex):
            for k, v in kwargs.items():
                if hasattr(self._params, k):
                    setattr(self._params, k, v)

    def play(self):  self._pause = False
    def pause(self): self._pause = True
    def stop(self):  self._running = False

    def run(self):
        self._running = True
        last_ts = time.time()
        frames = 0
        fps_emit_last = time.time()
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self.msleep(10); continue
            if self._pause:
                self.msleep(10); continue

            ok, frame = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            frame = safe_frame_bgr(frame)

            with QMutexLocker(self._mutex):
                p = ProcParams(**self._params.__dict__)

            if 0.25 <= p.scale < 1.0:
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (int(w0 * p.scale), int(h0 * p.scale)), interpolation=cv2.INTER_AREA)

            before_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            after_bgr  = zebra_overlay(frame, p.mode, p.black, p.white, phase=self._phase)
            self._phase = (self._phase + p.phase_step) % 10000
            after_rgb  = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)

            h, w = before_rgb.shape[:2]
            q_before = QImage(before_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            q_after  = QImage(after_rgb.data,  w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.framePairReady.emit(q_before, q_after)

            frame_period = 1.0 / max(1.0, self._target_fps)
            dt = time.time() - last_ts
            if dt < frame_period:
                self.msleep(int((frame_period - dt) * 1000))
            last_ts = time.time()

            frames += 1
            if (time.time() - fps_emit_last) >= 0.5:
                self.fpsInfo.emit(frames / (time.time() - fps_emit_last))
                fps_emit_last = time.time(); frames = 0

        if self._cap is not None:
            try: self._cap.release()
            except: pass
            self._cap = None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zebra Tool")
        self.setStyleSheet("""
            QWidget{background:#0f0f0f;color:#ddd;}
            QLabel#view{background:#111;border:1px solid #222;}
            QGroupBox{border:1px solid #333;margin-top:6px;}
            QGroupBox::title{subcontrol-origin: margin; left:8px; padding:0 4px;}
        """)

        self.leftLabel  = QLabel("Before"); self.leftLabel.setObjectName("view")
        self.rightLabel = QLabel("After (Zebra)"); self.rightLabel.setObjectName("view")
        for lab in (self.leftLabel, self.rightLabel):
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setScaledContents(False)
            lab.setMinimumSize(1, 1)                                   # don't push window min-size
            lab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            lab.setFixedHeight(max(240, int(self.screen().availableGeometry().height() * 0.50)))  # 50% screen

        self._left_img  = None
        self._right_img = None

        self.openBtn = QPushButton(" Open");  self.openBtn.setIcon(QIcon.fromTheme("document-open"))
        self.playBtn = QPushButton(" Play");  self.playBtn.setIcon(QIcon.fromTheme("media-playback-start"))
        self._is_playing = False

        self.modeBox = QComboBox(); self.modeBox.addItems(["Both", "Over", "Under"])
        same_h = max(self.openBtn.sizeHint().height() + 1, self.playBtn.sizeHint().height()+1)
        self.openBtn.setFixedHeight(same_h)
        self.playBtn.setFixedHeight(same_h)
        self.modeBox.setFixedHeight(same_h)
        self.modeBox.setMinimumWidth(80)

        topRow = QHBoxLayout()
        topRow.addWidget(self.openBtn)
        topRow.addWidget(self.playBtn)
        topRow.addWidget(self.modeBox)
        topRow.addStretch(1)

        centerRow = QHBoxLayout()
        centerRow.addWidget(self.leftLabel, 1)
        centerRow.addWidget(self.rightLabel, 1)

        self.blackVal   = QLabel("16")
        self.whiteVal   = QLabel("235")
        self.scaleVal   = QLabel("50%")

        self.blackSlider = QSlider(Qt.Orientation.Horizontal); self.blackSlider.setRange(0, 254); self.blackSlider.setValue(16)
        self.whiteSlider = QSlider(Qt.Orientation.Horizontal); self.whiteSlider.setRange(1, 255); self.whiteSlider.setValue(235)
        self.scaleSlider = QSlider(Qt.Orientation.Horizontal); self.scaleSlider.setRange(25, 100); self.scaleSlider.setValue(50)

        bRow = QHBoxLayout(); bRow.addWidget(self.blackVal, 0); bRow.addWidget(self.blackSlider, 1)
        wRow = QHBoxLayout(); wRow.addWidget(self.whiteVal, 0); wRow.addWidget(self.whiteSlider, 1)
        sRow = QHBoxLayout(); sRow.addWidget(self.scaleVal, 0); sRow.addWidget(self.scaleSlider, 1)

        form = QFormLayout()
        form.addRow("Black", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(bRow)
        form.addRow("White", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(wRow)
        form.addRow("Scale %", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(sRow)

        bottomGB = QGroupBox("Thresholds & Scale")
        gbl = QVBoxLayout(bottomGB); gbl.addLayout(form)
        
        self.status = QStatusBar()
        self._update_status(0.0)

        bottomSplit = QHBoxLayout()
        leftBottom = QWidget(); lbLayout = QVBoxLayout(leftBottom); lbLayout.setContentsMargins(0,0,0,0)
        lbLayout.addWidget(bottomGB)
        rightBottom = QWidget()
        bottomSplit.addWidget(leftBottom, 1)
        bottomSplit.addWidget(rightBottom, 1)
        root = QVBoxLayout(self)
        root.addLayout(topRow)
        root.addLayout(centerRow, 1)
        root.addLayout(bottomSplit)
        root.addWidget(self.status)

        self.worker = VideoWorker()
        self.worker.framePairReady.connect(self.on_frames)
        self.worker.fpsInfo.connect(self.on_fps)
        self.worker.opened.connect(self.on_opened)
        self.worker.start()

        self.openBtn.clicked.connect(self.choose_file)
        self.playBtn.clicked.connect(self.toggle_play_pause)
        self.modeBox.currentTextChanged.connect(lambda m: (self.worker.set_params(mode=m), self._update_status()))

        self.blackSlider.valueChanged.connect(self._push_thresholds)
        self.whiteSlider.valueChanged.connect(self._push_thresholds)
        self.scaleSlider.valueChanged.connect(self._push_scale)

        self.showMaximized()

    def _scaled_pixmap(self, qimg: QImage, target_label: QLabel) -> QPixmap:
        r = target_label.contentsRect()
        if qimg is None or r.width() <= 0 or r.height() <= 0:
            return QPixmap()
        return QPixmap.fromImage(qimg).scaled(
            r.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    def on_frames(self, q_before: QImage, q_after: QImage):
        self._left_img, self._right_img = q_before, q_after
        self.leftLabel.setPixmap(self._scaled_pixmap(self._left_img, self.leftLabel))
        self.rightLabel.setPixmap(self._scaled_pixmap(self._right_img, self.rightLabel))

    def resizeEvent(self, e):
        if self._left_img is not None:
            self.leftLabel.setPixmap(self._scaled_pixmap(self._left_img, self.leftLabel))
        if self._right_img is not None:
            self.rightLabel.setPixmap(self._scaled_pixmap(self._right_img, self.rightLabel))
        super().resizeEvent(e)

    def on_fps(self, fps: float):
        self._update_status(fps)

    def _update_status(self, fps: float = None):
        b = self.blackSlider.value()
        w = self.whiteSlider.value()
        s = self.scaleSlider.value()
        mode = self.modeBox.currentText()
        msg = f"Mode: {mode}   Black: {b}   White: {w}   Scale: {s}%"
        if fps is not None:
            msg += f"   ~{fps:.1f} FPS"
        self.status.showMessage(msg)

    def on_opened(self, ok: bool, msg: str):
        self.status.showMessage(msg)
        if not ok:
            QMessageBox.warning(self, "Open video", msg)
        else:
            self._set_playing(True)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)")
        if not path:
            return
        self.worker.open_video(path)

    def _set_playing(self, playing: bool):
        self._is_playing = playing
        if playing:
            self.worker.play()
            self.playBtn.setText(" Pause")
            self.playBtn.setIcon(QIcon.fromTheme("media-playback-pause"))
        else:
            self.worker.pause()
            self.playBtn.setText(" Play")
            self.playBtn.setIcon(QIcon.fromTheme("media-playback-start"))

    def toggle_play_pause(self):
        self._set_playing(not self._is_playing)

    def _push_thresholds(self, *_):
        b = int(self.blackSlider.value())
        w = int(self.whiteSlider.value())
        if b >= w:
            sender = self.sender()
            if sender is self.blackSlider:
                w = min(255, b + 1)
                self.whiteSlider.setValue(w)
            else:
                b = max(0, w - 1)
                self.blackSlider.setValue(b)

        self.blackVal.setText(f"{b}")
        self.whiteVal.setText(f"{w}")

        self.worker.set_params(black=b, white=w)
        self._update_status()

    def _push_scale(self, *_):
        s = int(self.scaleSlider.value())
        self.scaleVal.setText(f"{s}%")
        scale = max(0.25, min(1.0, s / 100.0))
        self.worker.set_params(scale=scale)
        self._update_status()


    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self.toggle_play_pause(); e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal(); e.accept()
            else:
                super().keyPressEvent(e)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.worker.stop()
        self.worker.wait(1500)
        return super().closeEvent(e)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
