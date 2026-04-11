"""
Trash Detector — Flask + YOLOv8
=================================
Install:
    pip install flask ultralytics opencv-python numpy

Jalankan:
    python app.py

Buka di browser:
    http://localhost:5000
"""

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
from collections import defaultdict
import threading
import time
import base64

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = "yolov8n.pt"
CAMERA_ID   = 0
CONF_THRESH = 0.45

HIDDEN_CLASSES = {"person", "tv", "chair", "couch", "bed",
                  "dining table", "toilet", "potted plant"}

LABEL_COLORS = {
    "bottle":     "#4ECB8D",
    "cup":        "#4ECB8D",
    "bowl":       "#4ECB8D",
    "toothbrush": "#4ECB8D",
    "banana":     "#F5A623",
    "apple":      "#F5A623",
    "orange":     "#F5A623",
    "pizza":      "#F5A623",
    "sandwich":   "#F5A623",
    "book":       "#5B9CF6",
    "laptop":     "#A78BFA",
    "cell phone": "#A78BFA",
    "keyboard":   "#A78BFA",
    "mouse":      "#A78BFA",
    "remote":     "#A78BFA",
    "scissors":   "#94A3B8",
    "knife":      "#94A3B8",
    "fork":       "#94A3B8",
    "spoon":      "#94A3B8",
    "wine glass": "#34D399",
    "vase":       "#34D399",
    "backpack":   "#F472B6",
    "handbag":    "#F472B6",
    "suitcase":   "#F472B6",
    "umbrella":   "#FB923C",
    "clock":      "#FB923C",
}
DEFAULT_COLOR = "#64748B"

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

# ─── App state ───────────────────────────────────────────────────────────────

app      = Flask(__name__)
model    = YOLO(MODEL_PATH)
cap      = None
lock     = threading.Lock()
state    = {
    "running":      False,
    "conf":         CONF_THRESH,
    "show_person":  False,
    "detections":   [],
    "fps":          0.0,
}

# ─── Camera thread ───────────────────────────────────────────────────────────

latest_frame = None

def camera_loop():
    global cap, latest_frame

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_t = time.time()
    fps_n = 0

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            break

        hidden = set(HIDDEN_CLASSES)
        if state["show_person"]:
            hidden.discard("person")

        results = model(frame, conf=state["conf"], verbose=False)[0]
        dets    = []

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in hidden:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score  = float(box.conf[0])
            color  = LABEL_COLORS.get(cls_name, DEFAULT_COLOR)
            bgr    = hex_to_bgr(color)

            dets.append({
                "cls":   cls_name,
                "conf":  round(score * 100),
                "color": color,
            })

            # Draw box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), bgr, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)

            # Pill label
            fs = 0.55
            (tw, th), _ = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
            px, py = 10, 6
            py1 = max(0, y1 - th - py*2)
            cv2.rectangle(frame, (x1, py1), (x1+tw+px*2, y1), bgr, -1)
            cv2.putText(frame, cls_name, (x1+px, y1-py),
                        cv2.FONT_HERSHEY_DUPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

        fps_n += 1
        now = time.time()
        if now - fps_t >= 1.0:
            state["fps"] = round(fps_n / (now - fps_t), 1)
            fps_n = 0
            fps_t = now

        state["detections"] = dets

        with lock:
            latest_frame = frame.copy()

    cap.release()

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    if not state["running"]:
        state["running"] = True
        threading.Thread(target=camera_loop, daemon=True).start()
    return jsonify({"status": "ok"})

@app.route("/stop", methods=["POST"])
def stop():
    state["running"] = False
    return jsonify({"status": "ok"})

@app.route("/conf/<float:val>", methods=["POST"])
def set_conf(val):
    state["conf"] = max(0.1, min(0.9, val))
    return jsonify({"status": "ok"})

@app.route("/toggle_person", methods=["POST"])
def toggle_person():
    state["show_person"] = not state["show_person"]
    return jsonify({"show_person": state["show_person"]})

@app.route("/detections")
def detections():
    return jsonify({
        "detections": state["detections"],
        "fps":        state["fps"],
        "running":    state["running"],
    })

def gen_frames():
    while True:
        with lock:
            frame = latest_frame
        if frame is None or not state["running"]:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Trash Detector running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
