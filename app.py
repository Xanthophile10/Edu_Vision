import cv2
import pickle
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, send_file
from insightface.app import FaceAnalysis
import os
import sqlite3
from datetime import datetime, timedelta
import time
import serial
import threading
import json
import pandas as pd
from collections import deque
from collections import defaultdict
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import torch

app = Flask(__name__)

# Extra safety
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
# ARDUINO SERIAL CONFIG
# =========================
SERIAL_PORT = "COM3"   # change to your Arduino port
BAUD_RATE = 9600

DEVICE = "cuda"  # use "cpu" if no GPU available

arduino_id = None
serial_lock = threading.Lock()

registration_embeddings = defaultdict(list)

# =========================
# FRAME BUFFERS PER CAMERA
# =========================
frame_buffers = {}
frame_locks = {}

def camera_reader(cam_id, source):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        with frame_locks[cam_id]:
            frame_buffers[cam_id].append(frame)

def yolo_inference_thread(cam_id):
    while True:
        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                time.sleep(0.05)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        with yolo_lock, torch.inference_mode():
            # Pose
            pose_results = pose_model.predict(frame, device=DEVICE, verbose=False)
            annotated = pose_results[0].plot()

            # Phone detection
            phone_results = phone_model.predict(frame, device=DEVICE, verbose=False)[0]
            if phone_results.boxes is not None:
                for box, cls, conf in zip(phone_results.boxes.xyxy, phone_results.boxes.cls, phone_results.boxes.conf):
                    if int(cls) == 67 and conf > 0.4:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, "PHONE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

            # Replace latest frame with annotated version
            frame_buffers[cam_id].append(annotated)
        time.sleep(0.01)

def read_arduino():
    global arduino_id

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("[INFO] Arduino connected")
    except Exception as e:
        print(f"[ERROR] Arduino not found: {e}")
        return

    while True:
        try:
            if ser.in_waiting > 0:
                raw = ser.readline().decode(errors='ignore').strip()
                if raw:
                    with serial_lock:
                        arduino_id = raw  # Set global variable
                    print(f"[RFID READ] Arduino ID updated: {arduino_id}")
            time.sleep(0.1)  # avoid busy loop
        except Exception as e:
            print(f"[ERROR] Serial read failed: {e}")
            time.sleep(1)



DB_PATH = "face_db.pkl"
LOG_DB = "face_logs.db"
SNAPSHOT_DIR = "face_snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# =========================
# LOAD FACE DATABASE
# =========================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# =========================
# INIT LOG DATABASE
# =========================
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
    # Attendance table
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            course TEXT,
            time_in TEXT,
            score REAL,
            status TEXT,
            image TEXT
        )
    """)
    conn.commit()
    conn.close()

init_logs_db()

def init_visitor_db():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Drop old table if exists
    #c.execute("DROP TABLE IF EXISTS visitors")

    # Create new visitors table with proper columns
    c.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id TEXT UNIQUE,
            name TEXT,
            visitor_type TEXT,
            purpose TEXT,
            expiration_date TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

init_visitor_db()

def auto_delete_expired_visitors():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get expired visitors (for debug/logging)
    c.execute(
        "SELECT visitor_id FROM visitors WHERE expiration_date <= ?",
        (now,)
    )
    expired = c.fetchall()

    if expired:
        print("[AUTO-DELETE] Expired visitors:", [v[0] for v in expired])

    # Delete expired visitors
    c.execute(
        "DELETE FROM visitors WHERE expiration_date <= ?",
        (now,)
    )

    conn.commit()
    conn.close()

def calculate_expiration(visitor_type):
    now = datetime.now()
    if visitor_type == "Temporary":
        # Expires today at 23:59:59
        return now.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        # ~1 semester (4 months) safely
        month = now.month + 4
        year = now.year
        if month > 12:
            month -= 12
            year += 1
        day = min(now.day, 28)  # prevent invalid day
        return datetime(year, month, day, now.hour, now.minute, now.second)

# =========================
# INSIGHTFACE
# =========================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

def detect_camera(max_index=3):
    for i in range(max_index):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    return cv2.VideoCapture(0)

#camera = detect_camera()

TAPO_RTSP1 = "rtsp://tapoc200:gerald123@192.168.254.121:554/stream1" 
TAPO_RTSP2 = "rtsp://tapoc260:gerald123@192.168.254.136:554/stream1"
TAPO_RTSP3 = "https://192.168.254.105:8080/video"

CAMERA_SOURCES = {
    0: TAPO_RTSP1,
    1: TAPO_RTSP2,
    2: TAPO_RTSP3
}

# =========================
# YOLO MODELS
# =========================
pose_model = YOLO("yolo26l-pose.pt").to(DEVICE)   # Pose detection
phone_model = YOLO("yolo26l.pt").to(DEVICE)       # COCO model for phone detection

yolo_lock = threading.Lock()

DUMMY = np.zeros((640, 640, 3), dtype=np.uint8)

with torch.no_grad():
    pose_model.predict(DUMMY, verbose=False)
    phone_model.predict(DUMMY, verbose=False)

print("[INFO] YOLO models warmed up on GPU")

# =========================
# MEDIA PIPE YAWN DETECTION
# =========================
MODEL_PATH = r"C:\Users\RENT ACCOUNT\Downloads\face_landmarker.task"
YAWN_THRESHOLD = 0.40
EYE_CLOSED_THRESHOLD = 0.27
YAWN_FRAMES = 10

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

yawn_counter = 0

def mouth_open_ratio(landmarks, w, h):
    upper = landmarks[UPPER_LIP]; lower = landmarks[LOWER_LIP]
    left = landmarks[LEFT_MOUTH]; right = landmarks[RIGHT_MOUTH]
    vertical = np.linalg.norm([(upper.x - lower.x)*w, (upper.y - lower.y)*h])
    horizontal = np.linalg.norm([(left.x - right.x)*w, (left.y - right.y)*h])
    return vertical / horizontal

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    p = [(landmarks[i].x*w, landmarks[i].y*h) for i in eye_idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

def yawning_detector():
    global yawn_counter
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    cap = cv2.VideoCapture(TAPO_RTSP2, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (1280, 720))
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        label = "NO YAWN"
        color = (0, 255, 0)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            mor = mouth_open_ratio(landmarks, w, h)
            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (ear_left + ear_right) / 2.0

            if mor > YAWN_THRESHOLD and ear < EYE_CLOSED_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if yawn_counter >= YAWN_FRAMES:
                label = "YAWNING"
                color = (0, 0, 255)

            cv2.putText(frame, f"MOR: {mor:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, label, (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Yawning Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

for cam_id in CAMERA_SOURCES.keys():
    frame_buffers[cam_id] = deque(maxlen=2)
    frame_locks[cam_id] = threading.Lock()

def generate_frames(cam_id):
    auto_delete_expired_visitors()
    source = CAMERA_SOURCES.get(cam_id)
    if source is None:
        print(f"[ERROR] Camera ID {cam_id} not configured")
        return
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")
        return

    while True:
        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                time.sleep(0.05)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        faces = face_app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            embedding = face.embedding

            student_id = "Unknown"
            name = "Unknown"
            course = ""
            role_label = "Unknown"
            max_score = 0

            # Compare with known faces
            for sid, info in face_db.items():
                score = cosine_similarity(embedding, info["embedding"])
                if score > max_score and score > 0.6:
                    max_score = score
                    student_id = info["student_id"]
                    name = info["name"]
                    course = info["course"]
                    role_label = "Student"

            # ---------- Crop face for MediaPipe ----------
            face_crop = frame[y1:y2, x1:x2].copy()
            if face_crop.size != 0:
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                result = landmarker.detect(mp_image)

                yawn_label = "NO YAWN"
                yawn_color = (0, 255, 0)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    mor = mouth_open_ratio(landmarks, face_crop.shape[1], face_crop.shape[0])
                    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, face_crop.shape[1], face_crop.shape[0])
                    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, face_crop.shape[1], face_crop.shape[0])
                    ear = (ear_left + ear_right) / 2.0

                    # Yawn detection condition
                    if mor > YAWN_THRESHOLD and ear < EYE_CLOSED_THRESHOLD:
                        yawn_label = "YAWNING"
                        yawn_color = (0, 0, 255)

                # Draw Yawn/Eye label above the face
                cv2.putText(frame, yawn_label, (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

            # Check visitors if not found
            if student_id == "Unknown":
                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()
                c.execute("SELECT visitor_id, name, visitor_type, expiration_date, embedding FROM visitors")
                for row in c.fetchall():
                    v_id, v_name, v_type, exp, v_embedding = row
                    v_embedding = pickle.loads(v_embedding)
                    score = cosine_similarity(embedding, v_embedding)
                    if score > max_score and score > 0.6:
                        # Check expiration
                        if datetime.now() <= datetime.strptime(exp, "%Y-%m-%d %H:%M:%S"):
                            max_score = score
                            student_id = v_id
                            name = v_name
                            course = f"Visitor ({v_type})"
                            role_label = f"Visitor ({v_type})"

                conn.close()

            # Save snapshot
            face_crop = frame[y1:y2, x1:x2]
            image_name = None
            if face_crop.size != 0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                image_name = f"{student_id if student_id != 'Unknown' else 'unknown'}_{ts}.jpg"
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

            print("RFID:", arduino_id, "Face:", student_id)
            #  Logging + attendance
            if should_log(student_id):
                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()

                #  ALWAYS log
                c.execute(
                    "INSERT INTO logs (student_id, name, course, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        student_id,
                        name,
                        course,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        float(max_score),
                        image_name
                    )
                )

                #  Attendance ONLY if Known + RFID match
                if student_id != "Unknown" and arduino_id is not None and arduino_id == student_id:
                    c.execute(
                        "INSERT INTO attendance (student_id, name, course, time_in, score, status, image) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            student_id,
                            name,
                            course,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            float(max_score),
                            "Present",
                            image_name
                        )
                    )

                conn.commit()
                conn.close()

            color = (0, 255, 0) if student_id != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Role label (TOP)
            cv2.putText(
                frame,
                role_label,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            cv2.putText(frame, f"{name} ({max_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --------------------------
        # YOLO POSE & PHONE DETECTION
        # --------------------------
        with yolo_lock, torch.inference_mode():

            pose_results = pose_model.predict(
                frame,
                device=DEVICE,
                verbose=False
            )
            annotated = pose_results[0].plot()

            phone_results = phone_model.predict(
                frame,
                device=DEVICE,
                verbose=False
            )[0]

            if phone_results.boxes is not None:
                for box, cls, conf in zip(
                    phone_results.boxes.xyxy,
                    phone_results.boxes.cls,
                    phone_results.boxes.conf
                ):
                    if int(cls) == 67 and conf > 0.4:  # cellphone
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            "PHONE",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
        ret, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

current_student = {} 

latest_face = None

# =========================
# LOG COOLDOWN (ANTI-SPAM)
# =========================
last_logged = {}
LOG_COOLDOWN = 10  # seconds

def should_log(student_id):
    now = time.time()
    if student_id not in last_logged:
        last_logged[student_id] = now
        return True
    if now - last_logged[student_id] >= LOG_COOLDOWN:
        last_logged[student_id] = now
        return True
    return False

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# DASHBOARD
# =========================
@app.route("/")
def index():
    auto_delete_expired_visitors()  # 🔴 AUTO DELETE HERE
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM visitors")
    total_visitors = c.fetchone()[0]

    conn.close()

    return render_template(
        "index.html",
        total_faces=len(face_db),
        total_visitors=total_visitors,
        attendance_dates=json.dumps(["Mon","Tue","Wed","Thu","Fri"]),
        attendance_counts=json.dumps([20,35,30,40,25])
    )

# =========================
# Registration Stream (Camera Feed)
# =========================
def register_stream(cam_id=1):
    global latest_face
    registration_embeddings["current"] = []

    while True:
        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                time.sleep(0.05)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        faces = face_app.get(frame)

        # Skip if no face or multiple faces
        if len(faces) != 1:
            latest_face = None
        else:
            latest_face = faces[0]

            # Append embedding to buffer
            registration_embeddings["current"].append(latest_face.embedding)

            # Keep only last 5 embeddings
            if len(registration_embeddings["current"]) > 5:
                registration_embeddings["current"].pop(0)

        # Draw bounding boxes
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face Registration",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# =========================
# REGISTER STUDENT
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    global current_student
    if request.method == "POST":
        student_id = request.form["student_id"].strip()
        name = request.form["name"].strip()
        course = request.form["course"].strip()

        # ===== Prevent duplicate student IDs =====
        if student_id in face_db:
            return render_template("register.html", error=f"Student ID {student_id} already exists!")

        current_student = {"student_id": student_id, "name": name, "course": course}
        return redirect(url_for("register_camera"))

    return render_template("register.html")

@app.route("/register_camera")
def register_camera():
    return render_template("register_camera.html")

@app.route("/register_feed")
def register_feed():
    return Response(register_stream(cam_id=1),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# TEMP STORAGE FOR MULTIPLE CAPTURES
# =========================
registration_embeddings = {"current": []}  # store embeddings before saving

# =========================
# Save Face Route
# =========================
@app.route("/save_face", methods=["POST"])
def save_face():
    global latest_face, current_student, registration_embeddings

    if latest_face is None or not current_student:
        return redirect(url_for("register_camera"))

    student_id = current_student["student_id"]

    # Initialize buffer if not exists
    if "current" not in registration_embeddings:
        registration_embeddings["current"] = []

    # Append current embedding
    registration_embeddings["current"].append(latest_face.embedding)
    print(f"[INFO] Captured {len(registration_embeddings['current'])}/5 frames for {student_id}")

    # Only save after 10 captures
    if len(registration_embeddings["current"]) >= 10:
        avg_embedding = np.mean(registration_embeddings["current"], axis=0)

        # Save in face_db
        face_db[student_id] = {
            "student_id": student_id,
            "name": current_student["name"],
            "course": current_student["course"],
            "embedding": avg_embedding
        }

        # Save face snapshot
        x1, y1, x2, y2 = map(int, latest_face.bbox)
        face_crop = latest_face.normed_face if hasattr(latest_face, "normed_face") else None
        if face_crop is not None and face_crop.size != 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"{student_id}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

            # Log registration
            conn = sqlite3.connect(LOG_DB)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (student_id, name, course, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    current_student["name"],
                    current_student["course"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    1.0,
                    image_name
                )
            )
            # Attendance
            c.execute(
                "INSERT INTO attendance (student_id, name, course, time_in, score, status, image) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    student_id,
                    current_student["name"],
                    current_student["course"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    1.0,
                    "Present",
                    image_name
                )
            )
            conn.commit()
            conn.close()

        # Save DB
        with open(DB_PATH, "wb") as f:
            pickle.dump(face_db, f)

        # Clear embeddings buffer
        registration_embeddings["current"].clear()
        print(f"[SUCCESS] Saved stable embedding for {student_id}")

    return redirect(url_for("register_camera"))


@app.route("/quit")
def quit_camera():
    return redirect(url_for("index"))

# =========================
# FACE RECOGNITION
# =========================
def recognize_stream():
    auto_delete_expired_visitors()
    while True:
        with frame_locks[cam_id]:
            if not frame_buffers[cam_id]:
                time.sleep(0.05)
                continue
            frame = frame_buffers[cam_id][-1].copy()

        faces = face_app.get(frame)
        for face in faces:
            embedding = face.embedding
            student_id = "Unknown"
            name = "Unknown"
            course = ""
            role_label = "Unknown"
            max_score = 0

            # Compare with known faces
            for sid, info in face_db.items():
                score = cosine_similarity(embedding, info["embedding"])
                if score > max_score and score > 0.6:
                    max_score = score
                    student_id = info["student_id"]
                    name = info["name"]
                    course = info["course"]
                    role_label = "Student"

            # Check visitors if not found
            if student_id == "Unknown":
                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()
                c.execute("SELECT visitor_id, name, visitor_type, expiration_date, embedding FROM visitors")
                for row in c.fetchall():
                    v_id, v_name, v_type, exp, v_embedding = row
                    v_embedding = pickle.loads(v_embedding)
                    score = cosine_similarity(embedding, v_embedding)
                    if score > max_score and score > 0.6:
                        # Check expiration
                        if datetime.now() <= datetime.strptime(exp, "%Y-%m-%d %H:%M:%S"):
                            max_score = score
                            student_id = v_id
                            name = v_name
                            course = f"Visitor ({v_type})"
                            role_label = f"Visitor ({v_type})"
                conn.close()

            # Save snapshot for both known and unknown faces
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = frame[y1:y2, x1:x2]
            image_name = None
            if face_crop.size != 0:
                ts = time.strftime("%Y%m%d_%H%M%S")
                if student_id != "Unknown":
                    image_name = f"{student_id}_{ts}.jpg"
                else:
                    image_name = f"unknown_{ts}.jpg"
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

            # Single log entry
            if should_log(student_id):
                conn = sqlite3.connect(LOG_DB)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO logs (student_id, name, course, time_in, score, image) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        student_id,
                        name,
                        course,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        float(max_score),
                        image_name
                    )
                )
                # Also insert into attendance
                if student_id != "Unknown":
                    c.execute(
                        "INSERT INTO attendance (student_id, name, course, time_in, score, status, image) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            student_id,
                            name,
                            course,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            float(max_score),
                            "Present",
                            image_name
                        )
                    )
                conn.commit()
                conn.close()

            color = (0, 255, 0) if student_id != "Unknown" else (0, 0, 255)
            display_name = name if student_id != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Role label (TOP)
            cv2.putText(
                frame,
                role_label,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            # 🔹 Draw role label (Student or Visitor)
            role_label = "Student" if "Visitor" not in course else course
            role_color = (255, 255, 0)  # yellow
            cv2.putText(
                frame,
                role_label,
                (x1, y1 - 30),          # above the name
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                role_color,
                2
            )            
            cv2.putText(
                frame,
                f"{display_name} ({max_score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/recognize")
def recognize():
    return render_template("recognize.html")

@app.route('/recognize_feed/<int:cam_id>')
def recognize_feed(cam_id):
    return Response(
        generate_frames(cam_id=cam_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# LOGS & ATTENDANCE
# =========================
@app.route("/logs")
def logs():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("logs.html", logs=data)

@app.route("/attendance")
def attendance():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY time_in DESC")
    data = c.fetchall()
    conn.close()
    return render_template("attendance.html", attendance=data)

# =========================
# SERVE SNAPSHOTS
# =========================
@app.route("/face_snapshots/<filename>")
def face_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

# =========================
# MANAGE / EDIT / DELETE REGISTERED FACES
# =========================
@app.route("/manage_registered_students", methods=["GET", "POST"])
def manage_registered_students():
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
        return redirect(url_for("manage_registered_students"))

    return render_template("manage_registered_students.html", faces=list(face_db.values()))

current_visitor = {}

@app.route("/register_visitor", methods=["GET", "POST"])
def register_visitor():
    global current_visitor
    if request.method == "POST":
        visitor_id = request.form["visitor_id"].strip()
        name = request.form["name"].strip()
        visitor_type = request.form["visitor_type"]
        purpose = request.form["purpose"].strip()

        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM visitors WHERE visitor_id=?", (visitor_id,))
        if c.fetchone():
            conn.close()
            return render_template("register_visitor.html", error="Visitor ID already exists!")
        conn.close()

        current_visitor = {
            "visitor_id": visitor_id,
            "name": name,
            "visitor_type": visitor_type,
            "purpose": purpose
        }
        return redirect(url_for("register_visitor_camera"))
    return render_template("register_visitor.html")

@app.route("/register_visitor_camera")
def register_visitor_camera():
    return render_template("register_visitor_camera.html")

@app.route("/register_visitor_feed")
def register_visitor_feed():
    return Response(register_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# Visitor multi-frame embedding
# =========================
visitor_embeddings = {"current": []}  # temporary buffer for current visitor

@app.route("/save_visitor_face", methods=["POST"])
def save_visitor_face():
    global latest_face, current_visitor

    if latest_face is not None and current_visitor:
        # Add latest embedding to buffer
        visitor_embeddings["current"].append(latest_face.embedding)

        # Wait until we have 10 frames
        if len(visitor_embeddings["current"]) < 10:
            return "Frame captured, not yet enough frames.", 200

        # Average embeddings for stability
        avg_embedding = np.mean(visitor_embeddings["current"], axis=0)

        visitor_id = current_visitor["visitor_id"]
        expiration = calculate_expiration(current_visitor["visitor_type"])

        # Save visitor in DB
        conn = sqlite3.connect(LOG_DB)
        c = conn.cursor()
        c.execute("""
            INSERT INTO visitors (visitor_id, name, visitor_type, purpose, expiration_date, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            visitor_id,
            current_visitor["name"],
            current_visitor["visitor_type"],
            current_visitor["purpose"],
            expiration.strftime("%Y-%m-%d %H:%M:%S"),
            pickle.dumps(avg_embedding)
        ))
        conn.commit()
        conn.close()

        # Save snapshot of latest face
        x1, y1, x2, y2 = map(int, latest_face.bbox)
        face_crop = latest_face.normed_face if hasattr(latest_face, "normed_face") else None
        if face_crop is not None and face_crop.size != 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"visitor_{visitor_id}_{ts}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, image_name), face_crop)

        # Clear buffer for next visitor
        visitor_embeddings["current"].clear()

        return "Visitor stable embedding saved ", 200

    return "No face detected", 400

@app.route("/manage_visitors", methods=["GET", "POST"])
def manage_visitors():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()

    # Edit visitor
    if request.method == "POST":
        edit_id = request.form.get("edit_id")
        new_name = request.form.get("new_name")
        new_type = request.form.get("new_type")
        new_purpose = request.form.get("new_purpose")

        if edit_id:
            c.execute("""
                UPDATE visitors SET name=?, visitor_type=?, purpose=? WHERE visitor_id=?
            """, (new_name, new_type, new_purpose, edit_id))

        delete_id = request.form.get("delete_id")
        if delete_id:
            c.execute("DELETE FROM visitors WHERE visitor_id=?", (delete_id,))
        conn.commit()

    c.execute("SELECT * FROM visitors")
    visitors = c.fetchall()
    conn.close()
    return render_template("manage_visitors.html", visitors=visitors)

@app.route("/download/attendance_csv")
def download_attendance_csv():
    conn = sqlite3.connect(LOG_DB)

    query = """
        SELECT
            student_id,
            name,
            course,
            time_in,
            status,
            score
        FROM attendance
        ORDER BY time_in DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return "No attendance data available", 204

    os.makedirs("reports", exist_ok=True)
    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join("reports", filename)

    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True)

@app.route("/download/visitor_logs_csv")
def download_visitor_logs_csv():
    conn = sqlite3.connect(LOG_DB)

    query = """
        SELECT
            visitor_id,
            name,
            visitor_type,
            purpose,
            expiration_date
        FROM visitors
        ORDER BY expiration_date DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return "No visitor data available", 204

    os.makedirs("reports", exist_ok=True)
    filename = f"visitors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join("reports", filename)

    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    # Start Arduino thread
    arduino_thread = threading.Thread(target=read_arduino, daemon=True)
    arduino_thread.start()

    # Start Camera threads
    for cam_id, src in CAMERA_SOURCES.items():
        threading.Thread(target=camera_reader, args=(cam_id, src), daemon=True).start()

    # Optional: Start YOLO inference threads for all cameras
    for cam_id in CAMERA_SOURCES.keys():
        threading.Thread(target=yolo_inference_thread, args=(cam_id,), daemon=True).start()

    # Optional: Start yawning detection thread
    # threading.Thread(target=yawning_detector, daemon=True).start()

    # Run Flask app
    app.run(debug=True, use_reloader=False)


