# """
# Trash Detector — YOLOv8 + PyQt5
# =================================
# Install:
#     pip install PyQt5 ultralytics opencv-python numpy

# Jalankan:
#     python trash_detector.py
# """

# import sys
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton,
#     QHBoxLayout, QVBoxLayout, QFrame, QSlider, QSizePolicy
# )
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
# from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush
# from ultralytics import YOLO
# from collections import defaultdict
# import time

# # ─── Config ──────────────────────────────────────────────────────────────────

# MODEL_PATH  = "yolov8n.pt"
# CAMERA_ID   = 0
# CONF_THRESH = 0.45

# HIDDEN_CLASSES = {"person", "tv", "chair", "couch", "bed",
#                   "dining table", "toilet", "potted plant"}

# LABEL_COLORS = {
#     "bottle":     "#4ECB8D",
#     "cup":        "#4ECB8D",
#     "bowl":       "#4ECB8D",
#     "toothbrush": "#4ECB8D",
#     "banana":     "#F5A623",
#     "apple":      "#F5A623",
#     "orange":     "#F5A623",
#     "pizza":      "#F5A623",
#     "sandwich":   "#F5A623",
#     "book":       "#5B9CF6",
#     "laptop":     "#A78BFA",
#     "cell phone": "#A78BFA",
#     "keyboard":   "#A78BFA",
#     "mouse":      "#A78BFA",
#     "remote":     "#A78BFA",
#     "scissors":   "#94A3B8",
#     "knife":      "#94A3B8",
#     "fork":       "#94A3B8",
#     "spoon":      "#94A3B8",
#     "wine glass": "#34D399",
#     "vase":       "#34D399",
#     "backpack":   "#F472B6",
#     "handbag":    "#F472B6",
#     "suitcase":   "#F472B6",
#     "umbrella":   "#FB923C",
#     "clock":      "#FB923C",
# }
# DEFAULT_COLOR = "#64748B"

# # BGR versions for OpenCV drawing
# def hex_to_bgr(h):
#     h = h.lstrip("#")
#     r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
#     return (b, g, r)

# # ─── Detection Thread ─────────────────────────────────────────────────────────

# class DetectionThread(QThread):
#     frame_ready = pyqtSignal(np.ndarray, list, float)

#     def __init__(self, model, conf):
#         super().__init__()
#         self.model   = model
#         self.conf    = conf
#         self.running = False
#         self.hidden  = set(HIDDEN_CLASSES)

#     def run(self):
#         cap = cv2.VideoCapture(CAMERA_ID)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         fps_t = time.time()
#         fps_n = 0
#         fps   = 0.0

#         while self.running:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             results = self.model(frame, conf=self.conf, verbose=False)[0]
#             dets    = []

#             for box in results.boxes:
#                 cls_name = self.model.names[int(box.cls[0])]
#                 if cls_name in self.hidden:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 score  = float(box.conf[0])
#                 color  = LABEL_COLORS.get(cls_name, DEFAULT_COLOR)
#                 bgr    = hex_to_bgr(color)

#                 dets.append({
#                     "cls":   cls_name,
#                     "conf":  score,
#                     "hex":   color,
#                     "bbox":  (x1, y1, x2, y2),
#                 })

#                 # Draw box on frame
#                 self._draw_box(frame, x1, y1, x2, y2, cls_name, bgr)

#             fps_n += 1
#             now = time.time()
#             if now - fps_t >= 1.0:
#                 fps   = fps_n / (now - fps_t)
#                 fps_n = 0
#                 fps_t = now

#             self.frame_ready.emit(frame, dets, fps)

#         cap.release()

#     def _draw_box(self, frame, x1, y1, x2, y2, label, bgr):
#         # Transparent fill
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
#         cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
#         # Border
#         cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#         # Pill
#         fs = 0.55
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
#         px, py = 10, 6
#         py1 = max(0, y1 - th - py*2)
#         cv2.rectangle(frame, (x1, py1), (x1+tw+px*2, y1), bgr, -1)
#         cv2.putText(frame, label, (x1+px, y1-py),
#                     cv2.FONT_HERSHEY_DUPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

# # ─── Label Row Widget ─────────────────────────────────────────────────────────

# class LabelRow(QWidget):
#     def __init__(self, cls_name, count, hex_color, parent=None):
#         super().__init__(parent)
#         self.setFixedHeight(48)
#         self.setAttribute(Qt.WA_StyledBackground, True)
#         self.setStyleSheet("background: #1E1E24; border-radius: 8px;")

#         layout = QHBoxLayout(self)
#         layout.setContentsMargins(12, 0, 12, 0)
#         layout.setSpacing(10)

#         # Color dot
#         dot = QLabel()
#         dot.setFixedSize(10, 10)
#         dot.setStyleSheet(f"""
#             background: {hex_color};
#             border-radius: 5px;
#         """)
#         layout.addWidget(dot)

#         # Label name
#         name_lbl = QLabel(cls_name)
#         name_lbl.setStyleSheet("color: #E2E8F0; font-size: 13px; background: transparent;")
#         name_lbl.setFont(QFont("Helvetica", 11))
#         layout.addWidget(name_lbl)

#         layout.addStretch()

#         # Count badge
#         badge = QLabel(str(count))
#         badge.setFixedSize(28, 22)
#         badge.setAlignment(Qt.AlignCenter)
#         badge.setStyleSheet(f"""
#             background: transparent;
#             color: {hex_color};
#             font-size: 13px;
#             font-weight: bold;
#             border: 1px solid {hex_color}44;
#             border-radius: 4px;
#         """)
#         layout.addWidget(badge)

# # ─── Main Window ─────────────────────────────────────────────────────────────

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Trash Detector — YOLOv8")
#         self.setMinimumSize(1100, 650)
#         self.setStyleSheet("background: #0F0F13;")

#         self.model   = None
#         self.thread  = None
#         self.running = False
#         self.conf    = CONF_THRESH
#         self.shot_n  = 0

#         self._build_ui()
#         self._load_model()

#     def _build_ui(self):
#         central = QWidget()
#         self.setCentralWidget(central)
#         root = QVBoxLayout(central)
#         root.setContentsMargins(0, 0, 0, 0)
#         root.setSpacing(0)

#         # ── Topbar ──
#         topbar = QWidget()
#         topbar.setFixedHeight(58)
#         topbar.setStyleSheet("background: #16161C; border-bottom: 1px solid #2A2A35;")
#         tl = QHBoxLayout(topbar)
#         tl.setContentsMargins(20, 0, 20, 0)
#         tl.setSpacing(16)

#         # App title
#         title = QLabel("TRASH <span style='color:#4ECB8D'>DETECTOR</span>")
#         title.setTextFormat(Qt.RichText)
#         title.setStyleSheet("color: #E2E8F0; font-size: 15px; font-weight: bold; background: transparent;")
#         tl.addWidget(title)

#         sep = QFrame()
#         sep.setFrameShape(QFrame.VLine)
#         sep.setStyleSheet("color: #2A2A35;")
#         tl.addWidget(sep)

#         # Conf label
#         tl.addWidget(self._muted_label("Confidence"))
#         self.conf_val = QLabel(f"{self.conf:.0%}")
#         self.conf_val.setStyleSheet("color: #4ECB8D; font-size: 12px; font-weight: bold; background: transparent; min-width: 36px;")
#         tl.addWidget(self.conf_val)

#         slider = QSlider(Qt.Horizontal)
#         slider.setRange(10, 90)
#         slider.setValue(int(self.conf * 100))
#         slider.setFixedWidth(100)
#         slider.setStyleSheet("""
#             QSlider::groove:horizontal { height: 4px; background: #2A2A35; border-radius: 2px; }
#             QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0;
#                 background: #4ECB8D; border-radius: 7px; }
#             QSlider::sub-page:horizontal { background: #4ECB8D; border-radius: 2px; }
#         """)
#         slider.valueChanged.connect(lambda v: (
#             setattr(self, 'conf', v/100),
#             self.conf_val.config(text=f"{v}%") if hasattr(self.conf_val, 'config')
#             else self.conf_val.setText(f"{v/100:.0%}"),
#             setattr(self.thread, 'conf', v/100) if self.thread else None
#         ))
#         tl.addWidget(slider)

#         tl.addStretch()

#         # Status
#         self.status_dot = QLabel("●")
#         self.status_dot.setStyleSheet("color: #2A2A35; font-size: 18px; background: transparent;")
#         tl.addWidget(self.status_dot)

#         self.status_lbl = QLabel("Memuat model...")
#         self.status_lbl.setStyleSheet("color: #64748B; font-size: 12px; background: transparent;")
#         tl.addWidget(self.status_lbl)

#         self.fps_lbl = QLabel("")
#         self.fps_lbl.setStyleSheet("color: #4ECB8D; font-size: 12px; font-weight: bold; background: transparent; min-width: 60px;")
#         tl.addWidget(self.fps_lbl)

#         sep2 = QFrame()
#         sep2.setFrameShape(QFrame.VLine)
#         sep2.setStyleSheet("color: #2A2A35;")
#         tl.addWidget(sep2)

#         # Buttons
#         self.btn_start = self._btn("▶  Start", "#4ECB8D", "#0F0F13", enabled=False)
#         self.btn_start.clicked.connect(self._start)
#         tl.addWidget(self.btn_start)

#         self.btn_stop = self._btn("■  Stop", "#1E1E24", "#64748B", enabled=False)
#         self.btn_stop.clicked.connect(self._stop)
#         tl.addWidget(self.btn_stop)

#         self.btn_shot = self._btn("⬤  Screenshot", "#1E1E24", "#64748B")
#         self.btn_shot.clicked.connect(self._screenshot)
#         tl.addWidget(self.btn_shot)

#         root.addWidget(topbar)

#         # ── Body ──
#         body = QWidget()
#         bl   = QHBoxLayout(body)
#         bl.setContentsMargins(0, 0, 0, 0)
#         bl.setSpacing(0)

#         # Video area
#         self.video_lbl = QLabel()
#         self.video_lbl.setAlignment(Qt.AlignCenter)
#         self.video_lbl.setStyleSheet("background: #0A0A0E;")
#         self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.video_lbl.setText("Tekan  ▶ Start  untuk mulai\nArahkan kamera ke sampah")
#         self.video_lbl.setStyleSheet("""
#             background: #0A0A0E;
#             color: #2A2A35;
#             font-size: 16px;
#         """)
#         bl.addWidget(self.video_lbl, stretch=1)

#         # Sidebar
#         sidebar = QWidget()
#         sidebar.setFixedWidth(260)
#         sidebar.setStyleSheet("background: #16161C; border-left: 1px solid #2A2A35;")
#         sl = QVBoxLayout(sidebar)
#         sl.setContentsMargins(0, 0, 0, 0)
#         sl.setSpacing(0)

#         # Tab bar
#         tab_bar = QWidget()
#         tab_bar.setFixedHeight(44)
#         tab_bar.setStyleSheet("background: #16161C; border-bottom: 1px solid #2A2A35;")
#         tbl = QHBoxLayout(tab_bar)
#         tbl.setContentsMargins(0, 0, 0, 0)
#         tbl.setSpacing(0)

#         self.tab_labels = QPushButton("Labels")
#         self.tab_labels.setCheckable(True)
#         self.tab_labels.setChecked(True)
#         self.tab_class  = QPushButton("Classifications")
#         self.tab_class.setCheckable(True)

#         tab_style = """
#             QPushButton {
#                 background: transparent;
#                 color: #64748B;
#                 font-size: 12px;
#                 border: none;
#                 border-bottom: 2px solid transparent;
#                 padding: 0 8px;
#             }
#             QPushButton:checked {
#                 color: #E2E8F0;
#                 border-bottom: 2px solid #4ECB8D;
#                 font-weight: bold;
#             }
#             QPushButton:hover { color: #94A3B8; }
#         """
#         self.tab_labels.setStyleSheet(tab_style)
#         self.tab_class.setStyleSheet(tab_style)
#         self.tab_labels.clicked.connect(lambda: (self.tab_labels.setChecked(True), self.tab_class.setChecked(False)))
#         self.tab_class.clicked.connect(lambda: (self.tab_class.setChecked(True), self.tab_labels.setChecked(False)))

#         tbl.addWidget(self.tab_labels)
#         tbl.addWidget(self.tab_class)
#         sl.addWidget(tab_bar)

#         # Label list scroll area
#         self.list_container = QWidget()
#         self.list_container.setStyleSheet("background: #16161C;")
#         self.list_layout = QVBoxLayout(self.list_container)
#         self.list_layout.setContentsMargins(12, 12, 12, 12)
#         self.list_layout.setSpacing(6)
#         self.list_layout.setAlignment(Qt.AlignTop)

#         self.empty_lbl = QLabel("Belum ada deteksi.\nTunjukkan sampah ke kamera.")
#         self.empty_lbl.setStyleSheet("color: #2A2A35; font-size: 12px; background: transparent;")
#         self.empty_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
#         self.list_layout.addWidget(self.empty_lbl)

#         sl.addWidget(self.list_container, stretch=1)

#         # Footer
#         footer = QWidget()
#         footer.setFixedHeight(52)
#         footer.setStyleSheet("background: #16161C; border-top: 1px solid #2A2A35;")
#         fl = QHBoxLayout(footer)
#         fl.setContentsMargins(16, 0, 16, 0)

#         self.total_lbl = QLabel("Total terdeteksi: 0")
#         self.total_lbl.setStyleSheet("color: #64748B; font-size: 11px; background: transparent;")
#         fl.addWidget(self.total_lbl)
#         fl.addStretch()

#         btn_person = QPushButton("Toggle person")
#         btn_person.setStyleSheet("""
#             QPushButton {
#                 background: #1E1E24;
#                 color: #64748B;
#                 font-size: 10px;
#                 border: 1px solid #2A2A35;
#                 border-radius: 4px;
#                 padding: 4px 10px;
#             }
#             QPushButton:hover { background: #2A2A35; color: #94A3B8; }
#         """)
#         btn_person.clicked.connect(self._toggle_person)
#         fl.addWidget(btn_person)

#         sl.addWidget(footer)
#         bl.addWidget(sidebar)

#         root.addWidget(body, stretch=1)

#         self._label_rows = []
#         self._show_person = False

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def _muted_label(self, text):
#         lbl = QLabel(text)
#         lbl.setStyleSheet("color: #64748B; font-size: 11px; background: transparent;")
#         return lbl

#     def _btn(self, text, bg, fg, enabled=True):
#         btn = QPushButton(text)
#         btn.setEnabled(enabled)
#         btn.setFixedHeight(32)
#         btn.setCursor(Qt.PointingHandCursor)
#         btn.setStyleSheet(f"""
#             QPushButton {{
#                 background: {bg};
#                 color: {fg};
#                 font-size: 12px;
#                 border: 1px solid #2A2A35;
#                 border-radius: 6px;
#                 padding: 0 16px;
#             }}
#             QPushButton:hover {{ opacity: 0.85; background: #2A2A35; }}
#             QPushButton:disabled {{ opacity: 0.4; }}
#         """)
#         return btn

#     # ── Model ─────────────────────────────────────────────────────────────────

#     def _load_model(self):
#         from PyQt5.QtCore import QThread, pyqtSignal

#         class Loader(QThread):
#             done = pyqtSignal(object)
#             def run(self):
#                 self.done.emit(YOLO(MODEL_PATH))

#         self._loader = Loader()
#         self._loader.done.connect(self._on_model_ready)
#         self._loader.start()

#     def _on_model_ready(self, model):
#         self.model = model
#         self.status_lbl.setText("Model siap")
#         self.status_lbl.setStyleSheet("color: #4ECB8D; font-size: 12px; background: transparent;")
#         self.status_dot.setStyleSheet("color: #4ECB8D; font-size: 18px; background: transparent;")
#         self.btn_start.setEnabled(True)

#     # ── Camera ────────────────────────────────────────────────────────────────

#     def _start(self):
#         self.thread         = DetectionThread(self.model, self.conf)
#         self.thread.hidden  = set(HIDDEN_CLASSES) if not self._show_person else set(HIDDEN_CLASSES - {"person"})
#         self.thread.running = True
#         self.thread.frame_ready.connect(self._on_frame)
#         self.thread.start()
#         self.btn_start.setEnabled(False)
#         self.btn_stop.setEnabled(True)
#         self.status_lbl.setText("Deteksi aktif")
#         self.status_dot.setStyleSheet("color: #4ECB8D; font-size: 18px; background: transparent;")
#         self.video_lbl.setStyleSheet("background: #0A0A0E;")
#         self.video_lbl.setText("")

#     def _stop(self):
#         if self.thread:
#             self.thread.running = False
#             self.thread.wait()
#         self.btn_start.setEnabled(True)
#         self.btn_stop.setEnabled(False)
#         self.status_lbl.setText("Kamera mati")
#         self.status_lbl.setStyleSheet("color: #64748B; font-size: 12px; background: transparent;")
#         self.status_dot.setStyleSheet("color: #2A2A35; font-size: 18px; background: transparent;")
#         self.fps_lbl.setText("")
#         self.video_lbl.setStyleSheet("""
#             background: #0A0A0E;
#             color: #2A2A35;
#             font-size: 16px;
#         """)
#         self.video_lbl.setText("Tekan  ▶ Start  untuk mulai\nArahkan kamera ke sampah")
#         self._update_sidebar([])

#     def _screenshot(self):
#         px = self.video_lbl.pixmap()
#         if px:
#             fname = f"screenshot_{self.shot_n:03d}.jpg"
#             px.save(fname)
#             print(f"Saved: {fname}")
#             self.shot_n += 1

#     def _toggle_person(self):
#         self._show_person = not self._show_person
#         if self.thread:
#             if self._show_person:
#                 self.thread.hidden.discard("person")
#             else:
#                 self.thread.hidden.add("person")

#     # ── Frame update ──────────────────────────────────────────────────────────

#     def _on_frame(self, frame, dets, fps):
#         self.fps_lbl.setText(f"FPS {fps:.1f}")

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#         px   = QPixmap.fromImage(qimg)

#         lw = self.video_lbl.width()
#         lh = self.video_lbl.height()
#         self.video_lbl.setPixmap(
#             px.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         )
#         self._update_sidebar(dets)

#     def _update_sidebar(self, dets):
#         # Clear rows
#         for row in self._label_rows:
#             row.setParent(None)
#         self._label_rows = []

#         if not dets:
#             self.empty_lbl.show()
#             self.total_lbl.setText("Total terdeteksi: 0")
#             return

#         self.empty_lbl.hide()

#         agg = defaultdict(lambda: {"count": 0, "hex": DEFAULT_COLOR})
#         for d in dets:
#             k = d["cls"]
#             agg[k]["count"] += 1
#             agg[k]["hex"]    = d["hex"]

#         for cls_name, info in sorted(agg.items(), key=lambda x: -x[1]["count"]):
#             row = LabelRow(cls_name, info["count"], info["hex"])
#             self.list_layout.addWidget(row)
#             self._label_rows.append(row)

#         total = sum(i["count"] for i in agg.values())
#         self.total_lbl.setText(f"Total terdeteksi: {total}")

#     def closeEvent(self, e):
#         if self.thread:
#             self.thread.running = False
#             self.thread.wait()
#         e.accept()

# # ─── Entry ───────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app    = QApplication(sys.argv)
#     app.setFont(QFont("Helvetica", 10))
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())