"""
Trash Detector — YOLOv8 + OpenCV
=================================
Deteksi sampah real-time pakai webcam laptop.
Tampilan mirip screenshot: video feed + sidebar label kanan.

Install dulu:
    pip install ultralytics opencv-python numpy

Jalankan:
    python trash_detector.py

Kontrol:
    Q / ESC  → keluar
    S        → screenshot
    SPACE    → pause/resume
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os

# ─── Konfigurasi ──────────────────────────────────────────────────────────────

MODEL_PATH = "yolov8n.pt"   # Auto-download kalau belum ada (~6MB)
CAMERA_ID  = 0              # 0 = webcam depan laptop
CONF_THRESH = 0.45          # Threshold confidence (0-1)
SIDEBAR_W  = 280            # Lebar sidebar kanan (pixel)
WINDOW_NAME = "Trash Detector — YOLOv8"

# Mapping: label COCO → info sampah
TRASH_MAP = {
    "bottle":      ("Botol plastik",   "Plastik",  (29, 158, 117)),
    "cup":         ("Gelas/cup",       "Plastik",  (29, 158, 117)),
    "bowl":        ("Mangkok",         "Organik",  (186, 117, 23)),
    "banana":      ("Kulit pisang",    "Organik",  (239, 159, 39)),
    "apple":       ("Sisa buah",       "Organik",  (239, 159, 39)),
    "orange":      ("Kulit jeruk",     "Organik",  (239, 159, 39)),
    "pizza":       ("Sisa makanan",    "Organik",  (186, 117, 23)),
    "sandwich":    ("Sisa makanan",    "Organik",  (186, 117, 23)),
    "book":        ("Kertas/buku",     "Kertas",   (55, 138, 221)),
    "laptop":      ("Laptop bekas",    "Elektronik",(83, 74, 183)),
    "cell phone":  ("Handphone",       "Elektronik",(83, 74, 183)),
    "keyboard":    ("Keyboard",        "Elektronik",(83, 74, 183)),
    "mouse":       ("Mouse komputer",  "Elektronik",(83, 74, 183)),
    "scissors":    ("Gunting",         "Logam",    (136, 135, 128)),
    "knife":       ("Pisau",           "Logam",    (136, 135, 128)),
    "fork":        ("Garpu",           "Logam",    (136, 135, 128)),
    "spoon":       ("Sendok",          "Logam",    (136, 135, 128)),
    "wine glass":  ("Gelas kaca",      "Kaca",     (93, 202, 165)),
    "vase":        ("Vas/kaca",        "Kaca",     (93, 202, 165)),
    "backpack":    ("Tas bekas",       "Tekstil",  (212, 83, 126)),
    "handbag":     ("Tas tangan",      "Tekstil",  (212, 83, 126)),
    "suitcase":    ("Koper",           "Tekstil",  (212, 83, 126)),
    "tie":         ("Kain/dasi",       "Tekstil",  (212, 83, 126)),
    "umbrella":    ("Payung",          "Campuran", (216, 90, 48)),
    "clock":       ("Jam bekas",       "Campuran", (216, 90, 48)),
    "toothbrush":  ("Sikat gigi",      "Plastik",  (29, 158, 117)),
    "remote":      ("Remote",          "Elektronik",(83, 74, 183)),
    "can":         ("Kaleng",          "Logam",    (136, 135, 128)),
    "paper":       ("Kertas",          "Kertas",   (55, 138, 221)),
}

# Warna per kategori (BGR)
CAT_COLOR = {
    "Plastik":    (29, 158, 117),
    "Organik":    (239, 159, 39),
    "Kertas":     (55, 138, 221),
    "Elektronik": (83, 74, 183),
    "Logam":      (136, 135, 128),
    "Kaca":       (93, 202, 165),
    "Tekstil":    (212, 83, 126),
    "Campuran":   (216, 90, 48),
    "Lainnya":    (100, 100, 100),
}

# ─── Helper draw ──────────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    """Gambar bounding box + label pill di atas box."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}  {int(conf*100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.52, 1
    (tw, th), bl = cv2.getTextSize(text, font, fs, thick)

    pad = 6
    bx1, by1 = x1, max(0, y1 - th - pad*2)
    bx2, by2 = x1 + tw + pad*2, y1

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(frame, text, (bx1+pad, by2-pad//2), font, fs, (255,255,255), thick, cv2.LINE_AA)


def draw_sidebar(sidebar, detections):
    """Render sidebar putih mirip screenshot: label + count + conf bar."""
    sidebar[:] = (250, 250, 250)   # background putih

    H, W = sidebar.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    pad   = 18

    # ── Header ──
    cv2.putText(sidebar, "Labels", (pad, 36), font, 0.65, (30,30,30), 1, cv2.LINE_AA)
    cv2.line(sidebar, (pad, 44), (W-pad, 44), (200,200,200), 1)

    if not detections:
        cv2.putText(sidebar, "Belum ada deteksi.", (pad, 80), font, 0.42, (160,160,160), 1, cv2.LINE_AA)
        cv2.putText(sidebar, "Tunjukkan sampah ke kamera.", (pad, 100), font, 0.42, (160,160,160), 1, cv2.LINE_AA)
        return

    # Agregasi label
    agg = defaultdict(lambda: {"count":0, "conf":0.0, "color":(100,100,100)})
    for d in detections:
        k = d["display"]
        agg[k]["count"] += 1
        agg[k]["conf"]   = max(agg[k]["conf"], d["conf"])
        agg[k]["color"]  = d["color"]

    y = 68
    for label, info in sorted(agg.items(), key=lambda x: -x[1]["count"]):
        if y > H - 20:
            break

        color = info["color"]      # BGR
        count = info["count"]
        conf  = info["conf"]

        # ── Dot warna ──
        dot_r = 6
        cv2.circle(sidebar, (pad + dot_r, y + 2), dot_r, color, -1)

        # ── Nama label ──
        cv2.putText(sidebar, label, (pad + dot_r*2 + 8, y + 7),
                    font, 0.46, (30,30,30), 1, cv2.LINE_AA)

        # ── Count di kanan ──
        cnt_str = str(count)
        (cw, _), _ = cv2.getTextSize(cnt_str, font, 0.52, 1)
        cv2.putText(sidebar, cnt_str, (W - pad - cw, y + 7),
                    font, 0.52, (80,80,80), 1, cv2.LINE_AA)

        # ── Confidence bar ──
        bar_y  = y + 14
        bar_x1 = pad + dot_r*2 + 8
        bar_x2 = W - pad - 30
        bar_w  = bar_x2 - bar_x1
        cv2.rectangle(sidebar, (bar_x1, bar_y), (bar_x2, bar_y+3), (220,220,220), -1)
        cv2.rectangle(sidebar, (bar_x1, bar_y), (bar_x1 + int(bar_w*conf), bar_y+3), color, -1)

        # ── Separator ──
        y += 38
        cv2.line(sidebar, (pad, y - 4), (W - pad, y - 4), (230,230,230), 1)

    # ── Footer: total deteksi ──
    total = sum(v["count"] for v in agg.values())
    footer = f"Total terdeteksi: {total}"
    cv2.line(sidebar, (0, H-36), (W, H-36), (200,200,200), 1)
    cv2.putText(sidebar, footer, (pad, H-14), font, 0.42, (120,120,120), 1, cv2.LINE_AA)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa buka kamera ID={CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused     = False
    screenshot_n = 0
    fps_timer  = time.time()
    fps_count  = 0
    fps_disp   = 0.0

    print("=" * 50)
    print("Trash Detector berjalan!")
    print("  Q / ESC  → keluar")
    print("  S        → screenshot")
    print("  SPACE    → pause/resume")
    print("=" * 50)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('s'):
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"Screenshot disimpan: {fname}")
            screenshot_n += 1

        if paused:
            cv2.imshow(WINDOW_NAME, canvas if 'canvas' in dir() else np.zeros((480,640+SIDEBAR_W,3), np.uint8))
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # ── Inferensi YOLOv8 ──
        results = model(frame, conf=CONF_THRESH, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id  = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            info = TRASH_MAP.get(cls_name)
            if info:
                display, category, color = info
            else:
                display  = cls_name
                category = "Lainnya"
                color    = (100, 100, 100)

            detections.append({
                "cls":     cls_name,
                "display": display,
                "category": category,
                "conf":    conf,
                "color":   color,
                "bbox":    (x1, y1, x2, y2),
            })

            draw_box(frame, x1, y1, x2, y2, display, conf, color)

        # ── FPS overlay ──
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_disp  = fps_count / (now - fps_timer)
            fps_count = 0
            fps_timer = now

        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,200,120), 1, cv2.LINE_AA)

        # ── Sidebar ──
        sidebar = np.zeros((frame.shape[0], SIDEBAR_W, 3), dtype=np.uint8)
        draw_sidebar(sidebar, detections)

        canvas = np.hstack([frame, sidebar])
        cv2.imshow(WINDOW_NAME, canvas)

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    main()
