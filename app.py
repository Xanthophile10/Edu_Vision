import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from insightface.app import FaceAnalysis
import os
import sqlite3
from datetime import datetime
import time
import threading
import torch
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# =========================
# FLASK + FACE DB PART
# =========================
app = Flask(__name__)

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "face_db.pkl")
LOG_DB = os.path.join(BASE_DIR, "face_logs.db")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "face_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# -------------------------
# LOAD FACE DATABASE
# -------------------------
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# -------------------------
# INIT LOG DATABASE
# -------------------------
def init_logs_db():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            course TEXT,
            time_in TEXT,
            score REAL,
            image TEXT
        )
    """)
    conn.commit()
    conn.close()

init_logs_db()

# -------------------------
# INSIGHTFACE FOR FLASK
# -------------------------
face_app_flask = FaceAnalysis(name="buffalo_l")
face_app_flask.prepare(ctx_id=-1, det_size=(640, 640))

current_student = {}
latest_face = None

# -------------------------
# LOG COOLDOWN
# -------------------------
last_logged = {}
LOG_COOLDOWN = 10  # seconds

def should_log(student_id):
    now = time.time()
    if student_id not in last_logged or now - last_logged[student_id] >= LOG_COOLDOWN:
        last_logged[student_id] = now
        return True
    return False

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def register_stream():
    global latest_face
    cap = cv2.VideoCapture(0)  # Use default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect faces using InsightFace
        faces = face_app_flask.get(frame)
        if faces:
            latest_face = faces[0]  # Save latest detected face
            x1, y1, x2, y2 = map(int, latest_face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# =========================
# DASHBOARD ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html", total_faces=len(face_db))

@app.route("/register", methods=["GET", "POST"])
def register():
    global current_student
    if request.method == "POST":
        student_id = request.form["student_id"].strip()
        name = request.form["name"].strip()
        course = request.form["course"].strip()
        current_student = {"student_id": student_id, "name": name, "course": course}
        return redirect(url_for("register_camera"))
    return render_template("register.html")

@app.route("/register_camera")
def register_camera():
    return render_template("register_camera.html")

@app.route("/register_feed")
def register_feed():
    return Response(register_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------------
# SAVE FACE
# -------------------------
@app.route("/save_face", methods=["POST"])
def save_face():
    global latest_face, current_student
    if latest_face is not None and current_student:
        face_crop = latest_face.normed_face if hasattr(latest_face, "normed_face") and latest_face.normed_face.size != 0 else \
                    latest_face.image[int(latest_face.bbox[1]):int(latest_face.bbox[3]),
                                      int(latest_face.bbox[0]):int(latest_face.bbox[2])]
        if face_crop.size != 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"{current_student['student_id']}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)
            print(f"[INFO] Saved face snapshot: {image_name}")

            # Save to logs
            conn = sqlite3.connect(LOG_DB)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (student_id, name, course, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?)",
                (current_student["student_id"], current_student["name"],
                 current_student["course"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 1.0, image_name)
            )
            conn.commit()
            conn.close()

        # Save embedding
        face_db[current_student["student_id"]] = {
            "student_id": current_student["student_id"],
            "name": current_student["name"],
            "course": current_student["course"],
            "embedding": latest_face.embedding
        }
        with open(DB_PATH, "wb") as f:
            pickle.dump(face_db, f)

    return redirect(url_for("index"))

@app.route("/quit")
def quit_camera():
    return redirect(url_for("index"))

@app.route("/logs")
def logs():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("logs.html", logs=data)

@app.route("/face_snapshots/<filename>")
def face_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

@app.route("/manage_faces", methods=["GET", "POST"])
def manage_faces():
    global face_db
    if request.method == "POST":
        edit_id = request.form.get("edit_id")
        new_student_id = request.form.get("new_student_id")
        new_name = request.form.get("new_name")
        new_course = request.form.get("new_course")
        if edit_id and edit_id in face_db:
            face_db[new_student_id.strip()] = {
                "student_id": new_student_id.strip(),
                "name": new_name.strip(),
                "course": new_course.strip(),
                "embedding": face_db.pop(edit_id)["embedding"]
            }
        delete_id = request.form.get("delete_id")
        if delete_id and delete_id in face_db:
            del face_db[delete_id]
        with open(DB_PATH, "wb") as f:
            pickle.dump(face_db, f)
        return redirect(url_for("manage_faces"))
    return render_template("manage_faces.html", faces=list(face_db.values()))

# =========================
# RTSP + YOLO + MEDIAPIPE
# =========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("[INFO] Using device:", DEVICE)

CAMERA_SOURCE = "rtsp://tapoc260:gerald123@192.168.254.136:554/stream1"
frame_buffer = deque(maxlen=2)
frame_lock = threading.Lock()

face_app_rtsp = FaceAnalysis(name="buffalo_l")
face_app_rtsp.prepare(ctx_id=0, det_size=(640, 640))

pose_model = YOLO("yolo26l-pose.pt").to(DEVICE)
phone_model = YOLO("yolo26l.pt").to(DEVICE)

dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
with torch.no_grad():
    pose_model.predict(dummy_frame, verbose=False)
    phone_model.predict(dummy_frame, verbose=False)
print("[INFO] YOLO models warmed up")
yolo_lock = threading.Lock()

MODEL_PATH = os.path.join(BASE_DIR, "C:\\Users\\RENT ACCOUNT\\Downloads\\face_landmarker.task")
YAWN_THRESHOLD = 0.25
EYE_CLOSED_THRESHOLD = 0.27

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def mouth_open_ratio(landmarks, w, h):
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]
    vertical = np.linalg.norm([(upper.x - lower.x) * w, (upper.y - lower.y) * h])
    horizontal = np.linalg.norm([(left.x - right.x) * w, (left.y - right.y) * h])
    return vertical / horizontal

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

# -------------------------
# RTSP CAMERA THREAD
# -------------------------
def camera_reader():
    while True:
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        print("[INFO] Connected to RTSP stream")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] RTSP frame failed, reconnecting...")
                break
            with frame_lock:
                frame_buffer.append(frame)
        cap.release()
        time.sleep(1)

# -------------------------
# RECOGNIZE STREAM
# -------------------------
def recognize_stream():
    yawn_counter = 0
    while True:
        # Grab the latest frame safely
        with frame_lock:
            if not frame_buffer:
                time.sleep(0.05)
                continue
            frame = frame_buffer[-1].copy()

        h, w, _ = frame.shape

        # ---------- FACE RECOGNITION ----------
        faces = face_app_rtsp.get(frame)
        for face in faces:
            embedding = face.embedding
            student_id = "Unknown"
            name = "Unknown"
            course = ""
            max_score = 0
            for sid, info in face_db.items():
                score = cosine_similarity(embedding, info["embedding"])
                if score > max_score and score > 0.6:
                    max_score = score
                    student_id = info["student_id"]
                    name = info["name"]
                    course = info["course"]

            # Safe bbox coordinates
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Draw rectangle and label
            color = (0, 255, 0) if student_id != "Unknown" else (0, 0, 255)
            display_name = name if student_id != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_name} ({max_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ---------- FACE CROP ----------
            face_crop = None
            if hasattr(face, "normed_face") and face.normed_face is not None and face.normed_face.size != 0:
                face_crop = face.normed_face
            else:
                # fallback
                face_crop = frame[y1:y2, x1:x2].copy()

            # ---------- LOG RECOGNITION ----------
            if student_id != "Unknown" and should_log(student_id):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_name = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                if face_crop.size != 0:
                    cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO logs (student_id, name, course, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?)",
                    (student_id, name, course, ts, float(max_score), image_name)
                )
                conn.commit()
                conn.close()

            # ---------- YAWN & EYE DETECTION ----------
            if face_crop.size != 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_image)
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    mor = mouth_open_ratio(landmarks, face_crop.shape[1], face_crop.shape[0])
                    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, face_crop.shape[1], face_crop.shape[0])
                    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, face_crop.shape[1], face_crop.shape[0])
                    ear = (ear_left + ear_right) / 2

                    label = "NO YAWN"
                    color_yawn = (0, 255, 0)
                    if mor > YAWN_THRESHOLD and ear < EYE_CLOSED_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter > 10:
                            label = "YAWNING"
                            color_yawn = (0, 0, 255)
                    else:
                        yawn_counter = 0
                    cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_yawn, 2)

        # ---------- POSE & PHONE DETECTION ----------
        with yolo_lock, torch.inference_mode():
            pose_results = pose_model.predict(frame, device=DEVICE, verbose=False)
            annotated = pose_results[0].plot()

            phone_results = phone_model.predict(frame, device=DEVICE, verbose=False)[0]
            if phone_results.boxes is not None:
                for box, cls, conf in zip(phone_results.boxes.xyxy,
                                          phone_results.boxes.cls,
                                          phone_results.boxes.conf):
                    if int(cls) == 67 and conf > 0.4:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, "PHONE", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ---------- STREAM TO FLASK ----------
        ret, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/recognize")
def recognize():
    return render_template("recognize.html")

@app.route("/recognize_feed")
def recognize_feed():
    return Response(recognize_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# START THREADS
# =========================
threading.Thread(target=camera_reader, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True)
