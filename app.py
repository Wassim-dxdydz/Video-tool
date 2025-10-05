import sys, os, time
from dataclasses import dataclass
import cv2
import numpy as np

from collections import deque
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QRect
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout, QProgressBar, QCheckBox,
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
    statsReady = pyqtSignal(int, float, float, float)
    
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
        self._seek_to = None
        self._frame_idx = 0

    def open_video(self, path: str):
        if self._cap is not None:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.opened.emit(False, "Cannot open video.")
            return
        self._cap = cap

        fps_raw = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._target_fps = fps_raw if 1.0 <= fps_raw <= 240.0 else 30.0

        self._path = path
        self._phase = 0
        self._frame_idx = 0
        self._frame_count = count
        self._seek_to = None

        self._loop_enabled = False
        self._loop_a = None
        self._loop_b = None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        dur = (count / self._target_fps) if (self._target_fps > 0 and count > 0) else 0.0
        name = os.path.basename(path)
        self.opened.emit(True, f"Opened: {name}  {w}x{h}  {self._target_fps:.2f} fps  {count} frames (~{dur:.1f}s)")


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
            # wait for a valid, playing capture
            if self._cap is None or not self._cap.isOpened():
                self.msleep(10)
                continue
            if self._pause:
                self.msleep(10)
                continue

            # --- consume any pending seek (thread-safe) ---
            with QMutexLocker(self._mutex):
                st = self._seek_to
                self._seek_to = None
            if st is not None:
                # clamp to [0, frame_count-1] if we know the length
                if getattr(self, "_frame_count", 0) > 0:
                    st = max(0, min(int(st), self._frame_count - 1))
                else:
                    st = max(0, int(st))
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, st)
                self._frame_idx = st
                self._phase = 0  # keep zebra animation continuous after seeks

            # --- read next frame (wrap to start at EOF) ---
            ok, frame = self._cap.read()
            if not ok:
                # rewind if we know length; otherwise just try frame 0
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                continue

            frame = safe_frame_bgr(frame)

            # snapshot current params
            with QMutexLocker(self._mutex):
                p = ProcParams(**self._params.__dict__)

            # optional preview scale
            if 0.25 <= p.scale < 1.0:
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (int(w0 * p.scale), int(h0 * p.scale)),
                    interpolation=cv2.INTER_AREA
                )

            # --- live stats (avg luma, over/under %) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avgY = float(gray.mean())                       # 0..255
            over_pct = float((gray >= p.white).mean())      # 0..1
            under_pct = float((gray <= p.black).mean())     # 0..1
            self.statsReady.emit(int(self._frame_idx), avgY, over_pct, under_pct)

            # --- make QImages for UI ---
            before_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            after_bgr  = zebra_overlay(frame, p.mode, p.black, p.white, phase=self._phase)
            self._phase = (self._phase + p.phase_step) % 10000
            after_rgb  = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)

            h, w = before_rgb.shape[:2]
            q_before = QImage(before_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            q_after  = QImage(after_rgb.data,  w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.framePairReady.emit(q_before, q_after)

            # --- pacing to target FPS ---
            frame_period = 1.0 / max(1.0, self._target_fps)
            dt = time.time() - last_ts
            if dt < frame_period:
                self.msleep(int((frame_period - dt) * 1000))
            last_ts = time.time()

            # --- FPS meter ---
            frames += 1
            if (time.time() - fps_emit_last) >= 0.5:
                self.fpsInfo.emit(frames / (time.time() - fps_emit_last))
                fps_emit_last = time.time()
                frames = 0

            # advance logical index
            self._frame_idx += 1
            if getattr(self, "_frame_count", 0) > 0 and self._frame_idx >= self._frame_count:
                self._frame_idx = 0  # keep it in range for stats/seek UI

        # cleanup
        if self._cap is not None:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None


    def seek_to(self, frame_idx: int):
        """Thread-safe: queue a seek that run() will consume."""
        with QMutexLocker(self._mutex):
            if getattr(self, "_frame_count", 0):
                frame_idx = max(0, min(int(frame_idx), self._frame_count - 1))
            else:
                frame_idx = max(0, int(frame_idx))

            self._seek_to = frame_idx
            self._frame_idx = frame_idx
            self._phase = 0
            
    def seek_to_time(self, t_sec: float, *, pause: bool = False):
        fps = self._target_fps if self._target_fps > 0 else 30.0
        self.seek_to(int(round(max(0.0, t_sec) * fps)), pause=pause)

    def total_frames(self) -> int:
        return int(self._frame_count)

    def current_frame(self) -> int:
        return int(self._frame_idx)

    def step_frames(self, delta: int):
        # Pause and step
        self.pause()
        self.seek_to(self.current_frame() + int(delta))

    # A/B loop controls
    def set_loop_a(self, frame_index: int | None):
        self._loop_a = int(frame_index) if frame_index is not None else None

    def set_loop_b(self, frame_index: int | None):
        self._loop_b = int(frame_index) if frame_index is not None else None

    def enable_loop(self, enabled: bool):
        self._loop_enabled = bool(enabled)



class TimelineWidget(QWidget):
    seekRequested = pyqtSignal(int)  # frame index to jump to

    def __init__(self, max_points=1200, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.max_points = max_points
        self.frames   = deque(maxlen=max_points)  # absolute frame indices
        self.avgY     = deque(maxlen=max_points)  # 0..255
        self.overpct  = deque(maxlen=max_points)  # 0..1
        self.underpct = deque(maxlen=max_points)  # 0..1
        self._last_frame = -1

    def on_stat(self, frame_idx: int, avgY: float, over: float, under: float):
        self.frames.append(frame_idx)
        self.avgY.append(avgY)
        self.overpct.append(over)
        self.underpct.append(under)
        self._last_frame = frame_idx
        self.update()

    def paintEvent(self, ev):
        if not self.frames:
            return
        p = QPainter(self)
        rect = self.contentsRect()
        p.fillRect(rect, QColor(22,22,22))

        # axes
        p.setPen(QPen(QColor(70,70,70), 1))
        p.drawRect(rect.adjusted(0,0,-1,-1))

        w = rect.width()
        h = rect.height()
        n = len(self.frames)
        if n < 2:
            return

        # map helpers
        def x_at(i):
            return rect.left() + int((i/(n-1)) * (w-1))
        def y_luma(v):   # 0..255 -> top..bottom
            return rect.bottom() - int((v/255.0) * (h-20)) - 10
        def y_pct(v):    # 0..1 -> top..bottom
            return rect.bottom() - int(v * (h-20)) - 10

        # grid lines
        for yy in (0.25, 0.5, 0.75):
            y = y_luma(yy*255)
            p.setPen(QPen(QColor(55,55,55), 1, Qt.PenStyle.DotLine))
            p.drawLine(rect.left(), y, rect.right(), y)

        # draw average luma
        p.setPen(QPen(QColor(200,200,200), 2))
        prev = None
        for i, v in enumerate(self.avgY):
            pt = (x_at(i), y_luma(v))
            if prev: p.drawLine(prev[0], prev[1], pt[0], pt[1])
            prev = pt

        # draw over/under percentages
        p.setPen(QPen(QColor(200,60,60), 2))    # over – red
        prev = None
        for i, v in enumerate(self.overpct):
            pt = (x_at(i), y_pct(v))
            if prev: p.drawLine(prev[0], prev[1], pt[0], pt[1])
            prev = pt

        p.setPen(QPen(QColor(60,120,200), 2))   # under – blue
        prev = None
        for i, v in enumerate(self.underpct):
            pt = (x_at(i), y_pct(v))
            if prev: p.drawLine(prev[0], prev[1], pt[0], pt[1])
            prev = pt

        # playhead at last frame
        p.setPen(QPen(QColor(180,180,0), 1))
        x = x_at(n-1)
        p.drawLine(x, rect.top(), x, rect.bottom())

        # legend
        p.setPen(QPen(QColor(180,180,180)))
        p.drawText(rect.adjusted(6,4,-6,-6),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                   "Luma (gray) • Over% (red) • Under% (blue)")

    def mousePressEvent(self, e):
        if not self.frames:
            return
        rect = self.contentsRect()
        n = len(self.frames)
        t = (e.position().x() - rect.left()) / max(1, rect.width()-1)
        t = min(1.0, max(0.0, t))
        idx_in_window = int(t * (n-1))
        target_frame = self.frames[0] + idx_in_window
        self.seekRequested.emit(int(target_frame))

class ExportWorker(QThread):
    progress = pyqtSignal(int)            # 0..100
    finished = pyqtSignal(bool, str)      # ok, message

    def __init__(self, in_path, out_path, mode, black, white, phase_step=2):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.mode = mode
        self.black = int(black)
        self.white = int(white)
        self.phase_step = int(phase_step)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        cap = cv2.VideoCapture(self.in_path)
        if not cap.isOpened():
            self.finished.emit(False, "Cannot open input video.")
            return

        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(self.out_path, fourcc, fps, (w, h))
        if not wr.isOpened():
            cap.release()
            self.finished.emit(False, "Cannot open output for writing.")
            return

        phase = 0
        i = 0
        last_pct = -1
        ok_all = True

        while True:
            if self._cancel:
                ok_all = False
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame = safe_frame_bgr(frame)
            out = zebra_overlay(frame, self.mode, self.black, self.white, phase=phase)
            phase = (phase + self.phase_step) % 10000
            wr.write(out)

            i += 1
            if count > 0:
                pct = int((i * 100) / count)
                if pct != last_pct:
                    self.progress.emit(pct)
                    last_pct = pct

        wr.release()
        cap.release()

        if self._cancel:
            try: os.remove(self.out_path)
            except: pass
            self.finished.emit(False, "Export canceled.")
        else:
            self.progress.emit(100)
            self.finished.emit(ok_all, f"Export complete: {os.path.basename(self.out_path)}")


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

        # --- Video views (Before / After) ---
        self.leftLabel  = QLabel("Before"); self.leftLabel.setObjectName("view")
        self.rightLabel = QLabel("After (Zebra)"); self.rightLabel.setObjectName("view")
        for lab in (self.leftLabel, self.rightLabel):
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setScaledContents(False)
            lab.setMinimumSize(1, 1)
            lab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            lab.setFixedHeight(max(240, int(self.screen().availableGeometry().height() * 0.50)))  # ~50% of screen

        self._left_img  = None
        self._right_img = None
        self._frame_idx = 0
        self._frame_count = 0
        self._seek_to = None
        self._loop_enabled = False
        self._loop_a = None
        self._loop_b = None

        # --- Top row: Open | Play/Pause | Export… | Mode ---
        self.openBtn   = QPushButton(" Open");  self.openBtn.setIcon(QIcon.fromTheme("document-open"))
        self.playBtn   = QPushButton(" Play");  self.playBtn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.stepBackBtn = QPushButton(" -1f ")
        self.stepFwdBtn  = QPushButton(" +1f ")
        self.setABtn     = QPushButton(" Set A ")
        self.setBBtn     = QPushButton(" Set B ")
        self.loopChk     = QCheckBox("Loop A-B")
        self.exportBtn = QPushButton(" Export…")
        self._is_playing = False

        self.modeBox = QComboBox(); self.modeBox.addItems(["Both", "Over", "Under"])  # exact item text

        # same heights
        same_h = max(self.openBtn.sizeHint().height() + 1, self.playBtn.sizeHint().height()+1)
        for w in (self.openBtn, self.playBtn, self.exportBtn, self.modeBox, self.stepBackBtn, self.stepFwdBtn, self.setABtn, self.setBBtn, self.loopChk):
            w.setFixedHeight(same_h)
        self.modeBox.setMinimumWidth(80)

        # same widths for buttons (looks tidy)
        same_w = max(self.openBtn.sizeHint().width(),
                     self.playBtn.sizeHint().width(),
                     self.exportBtn.sizeHint().width())
        for b in (self.openBtn, self.playBtn, self.exportBtn):
            b.setFixedWidth(same_w)

        topRow = QHBoxLayout()
        topRow.addWidget(self.openBtn)
        topRow.addWidget(self.playBtn)
        topRow.addWidget(self.stepBackBtn)
        topRow.addWidget(self.stepFwdBtn)
        topRow.addWidget(self.setABtn)
        topRow.addWidget(self.setBBtn)
        topRow.addWidget(self.loopChk)
        topRow.addWidget(self.exportBtn)
        topRow.addWidget(self.modeBox)
        topRow.addStretch(1)

        # --- Center: side-by-side previews ---
        centerRow = QHBoxLayout()
        centerRow.addWidget(self.leftLabel, 1)
        centerRow.addWidget(self.rightLabel, 1)

        # --- Bottom: sliders (left) + timeline (right) ---
        self.blackVal = QLabel("16")
        self.whiteVal = QLabel("235")
        self.scaleVal = QLabel("50%")

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

        # Right: live timeline
        self.timeline = TimelineWidget(max_points=1200)
        tlGroup = QGroupBox("Luma Timeline")
        tlLayout = QVBoxLayout(tlGroup); tlLayout.setContentsMargins(8,8,8,8)
        tlLayout.addWidget(self.timeline)

        # Split bottom area (left sliders, right timeline)
        bottomSplit = QHBoxLayout()
        leftBottom = QWidget(); lbLayout = QVBoxLayout(leftBottom); lbLayout.setContentsMargins(0,0,0,0)
        lbLayout.addWidget(bottomGB)
        rightBottom = QWidget(); rbLayout = QVBoxLayout(rightBottom); rbLayout.setContentsMargins(0,0,0,0)
        rbLayout.addWidget(tlGroup)
        bottomSplit.addWidget(leftBottom, 1)
        bottomSplit.addWidget(rightBottom, 1)

        # Status bar + export progress
        self.status = QStatusBar()
        self.exportBar = QProgressBar()
        self.exportBar.setFixedWidth(180)
        self.exportBar.setRange(0, 100)
        self.exportBar.setValue(0)
        self.exportBar.setVisible(False)
        self.status.addPermanentWidget(self.exportBar)
        self._update_status(0.0)

        # Root layout
        root = QVBoxLayout(self)
        root.addLayout(topRow)
        root.addLayout(centerRow, 1)
        root.addLayout(bottomSplit)
        root.addWidget(self.status)

        # --- Worker & connections ---
        self.worker = VideoWorker()
        self.worker.framePairReady.connect(self.on_frames)
        self.worker.fpsInfo.connect(self.on_fps)
        self.worker.opened.connect(self.on_opened)
        self.worker.statsReady.connect(self.timeline.on_stat)
        self.timeline.seekRequested.connect(self.worker.seek_to)
        self.stepBackBtn.clicked.connect(lambda: self.worker.step_frames(-1))
        self.stepFwdBtn.clicked.connect(lambda: self.worker.step_frames(+1))
        self.setABtn.clicked.connect(self._mark_a)
        self.setBBtn.clicked.connect(self._mark_b)
        self.loopChk.toggled.connect(self._toggle_loop)
        self.worker.start()

        # Controls
        self.openBtn.clicked.connect(self.choose_file)
        self.playBtn.clicked.connect(self.toggle_play_pause)
        self.exportBtn.clicked.connect(self.on_export)
        self.modeBox.currentTextChanged.connect(lambda m: (self.worker.set_params(mode=m.strip()), self._update_status()))
        self.blackSlider.valueChanged.connect(self._push_thresholds)
        self.whiteSlider.valueChanged.connect(self._push_thresholds)
        self.scaleSlider.valueChanged.connect(self._push_scale)

        # exporter state
        self._exporter = None
        self._was_playing_before_export = False

        self.showMaximized()

    # ---------- helpers / slots ----------
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
        if self._loop_a is not None or self._loop_b is not None:
            a = "-" if self._loop_a is None else str(self._loop_a)
            bb = "-" if self._loop_b is None else str(self._loop_b)
            loop = " ON" if self.loopChk.isChecked() else " off"
            msg += f"   A:{a}  B:{bb}  Loop:{loop}"
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
        key = e.key()
        mods = e.modifiers()

        if key == Qt.Key.Key_Space:
            self.toggle_play_pause(); e.accept(); return

        # frame stepping
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = -1 if key == Qt.Key.Key_Left else +1
            if mods & Qt.KeyboardModifier.ShiftModifier:
                step *= 10
            self.worker.step_frames(step); e.accept(); return

        # set A / B
        if key == Qt.Key.Key_A:
            self._mark_a(); e.accept(); return
        if key == Qt.Key.Key_B:
            self._mark_b(); e.accept(); return

        # toggle loop
        if key == Qt.Key.Key_L:
            self.loopChk.toggle(); e.accept(); return

        if key == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal(); e.accept(); return

        super().keyPressEvent(e)


    def closeEvent(self, e):
        # cancel any running export cleanly
        if getattr(self, "_exporter", None) and self._exporter.isRunning():
            try:
                self._exporter.cancel()
                self._exporter.wait(2000)
            except Exception:
                pass
        self.worker.stop()
        self.worker.wait(1500)
        return super().closeEvent(e)

    # ---------- export UI handlers ----------
    def on_export(self):
        # must have an open video
        in_path = getattr(self.worker, "_path", None)
        if not in_path:
            QMessageBox.information(self, "Export", "Open a video first.")
            return

        # choose output path
        base = os.path.splitext(os.path.basename(in_path))[0]
        suggested = f"{base}_zebra.mp4"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export with Zebra", suggested, "MP4 Video (*.mp4);;All Files (*)"
        )
        if not out_path:
            return

        # freeze current parameters
        mode  = self.modeBox.currentText().strip()
        black = int(self.blackSlider.value())
        white = int(self.whiteSlider.value())

        # pause preview while exporting; remember state
        self._was_playing_before_export = getattr(self, "_is_playing", False)
        if self._was_playing_before_export:
            self._set_playing(False)

        # disable controls during export
        for w in (self.openBtn, self.playBtn, self.modeBox,
                  self.blackSlider, self.whiteSlider, self.scaleSlider, self.exportBtn):
            w.setEnabled(False)

        # show progress
        self.exportBar.setVisible(True)
        self.exportBar.setValue(0)

        # start export worker
        self._exporter = ExportWorker(in_path, out_path, mode, black, white, phase_step=2)
        self._exporter.progress.connect(self.exportBar.setValue)
        self._exporter.finished.connect(self.on_export_finished)
        self._exporter.start()

    def on_export_finished(self, ok: bool, msg: str):
        # re-enable controls
        for w in (self.openBtn, self.playBtn, self.modeBox,
                  self.blackSlider, self.whiteSlider, self.scaleSlider, self.exportBtn):
            w.setEnabled(True)

        # hide progress
        self.exportBar.setVisible(False)

        # restore playback if it was playing
        if self._was_playing_before_export:
            self._set_playing(True)

        self.status.showMessage(msg, 5000)
        (QMessageBox.information if ok else QMessageBox.warning)(self, "Export", msg)

        # cleanup
        self._exporter = None
        
    def _mark_a(self):
        fi = self.worker.current_frame()
        self._loop_a = fi
        self.worker.set_loop_a(fi)
        self._update_status()

    def _mark_b(self):
        fi = self.worker.current_frame()
        self._loop_b = fi
        self.worker.set_loop_b(fi)
        self._update_status()

    def _toggle_loop(self, on: bool):
        # sanity: ensure A/B are valid and ordered
        if on and (self._loop_a is None or self._loop_b is None or self._loop_a > self._loop_b):
            QMessageBox.information(self, "Loop A–B", "Set A then B (A ≤ B) before enabling loop.")
            self.loopChk.blockSignals(True)
            self.loopChk.setChecked(False)
            self.loopChk.blockSignals(False)
            self.worker.enable_loop(False)
            return
        self.worker.enable_loop(on)
        # if enabling and we're past B, jump to A
        if on and self.worker.current_frame() > self._loop_b:
            self.worker.seek_to(self._loop_a)
        self._update_status()
    
        


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
