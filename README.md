# Smart Face Recognition & Monitor System

## Overview
This project is a **real-time smart monitoring system** that combines facial recognition, yawn detection, eye closure detection, pose estimation, and phone detection. It uses **RTSP camera streams** and integrates:

- **Flask** – for the web dashboard and logs.
- **InsightFace (Buffalo_L)** – for offline facial recognition.
- **YOLO (Ultralytics)** – for pose and phone detection.
- **MediaPipe FaceLandmarker** – for yawn and eye-closure detection.
- **SQLite3** – for logging recognized faces and snapshots.

The system captures live video, recognizes students, logs their attendance, detects drowsiness (yawning or eye closure), and monitors phone usage.

---

## Features

### 1. Facial Recognition
- Uses **Buffalo_L InsightFace model**.
- Maintains a **face database (`face_db.pkl`)** for registered users.
- Recognizes faces in real-time and labels them on the video stream.
- Unknown faces are labeled as `Unknown`.

### 2. Attendance Logs
- Logs are stored in **SQLite3 database (`face_logs.db`)**.
- Each entry includes:
  - `student_id`
  - `name`
  - `course`
  - `time_in`
  - `recognition score`
  - `snapshot image`
- Snapshots are saved in `face_snapshots/`.

### 3. Yawn & Eye Closure Detection
- Uses **MediaPipe FaceLandmarker**.
- **Yawn Threshold:** Adjustable (default 0.25 for higher sensitivity).  
- **Eye Closed Threshold:** 0.27.
- Detects and labels "YAWNING" or "NO YAWN" in real-time.

### 4. Pose & Phone Detection
- Uses **YOLOv8 models**:
  - `yolo26l-pose.pt` for pose detection
  - `yolo26l.pt` for phone detection (class 67)
- Draws bounding boxes and labels detected objects.

### 5. Web Interface
- Dashboard at `/` showing total registered faces.
- **Register students** via `/register` → `/register_camera`.
- **Live camera feed** for registration at `/register_feed`.
- **Recognize live feed** at `/recognize` → `/recognize_feed`.
- **Attendance logs** at `/logs`.
- **Manage registered faces** at `/manage_faces`.

---

## Installation

### Prerequisites
- Python 3.10+
- GPU recommended for YOLO/InsightFace acceleration.
- Libraries:
```bash
pip install flask opencv-python numpy torch torchvision ultralytics mediapipe insightface
```

### RTSP Camera
- Ensure RTSP camera is accessible:
```python
CAMERA_SOURCE = "rtsp://<username>:<password>@<IP>:554/stream1"
```

### YOLO Models
- Place models in your project folder:
  - `yolo26l-pose.pt` → pose detection
  - `yolo26l.pt` → phone detection

### MediaPipe FaceLandmarker
- Download your `.task` file and update the path:
```python
MODEL_PATH = "C:\\path\\to\\face_landmarker.task"
```

---

## Running the Project

1. Start Flask:
```bash
python app.py
```

2. Access the dashboard:
```
http://127.0.0.1:5000/
```

3. Register a student:
- Navigate `/register`
- Fill in student info → `/register_camera`
- Capture face → Save

4. Start recognition:
- Navigate `/recognize`
- Real-time recognition, yawn, and phone detection will display.

5. View logs:
- Navigate `/logs` for attendance records.
- Snapshots are clickable links stored in `face_snapshots/`.

---

## Configuration

### Yawn Sensitivity
- **Lower threshold → more sensitive** (detects smaller mouth openings).
```python
YAWN_THRESHOLD = 0.25  # default 0.25
EYE_CLOSED_THRESHOLD = 0.27
```

### Cooldown for logging
- Prevents repeated log entries for the same face:
```python
LOG_COOLDOWN = 10  # seconds
```

### RTSP Frame Buffer
- Ensures smoother feed:
```python
frame_buffer = deque(maxlen=2)
```

---

## File Structure
```
Project_1/
│
├─ app.py                   # Main Flask + recognition code
├─ face_db.pkl              # Pickle file for registered faces
├─ face_logs.db             # SQLite3 attendance logs
├─ face_snapshots/          # Captured snapshots
├─ yolo26l-pose.pt          # Pose detection model
├─ yolo26l.pt               # Phone detection model
├─ face_landmarker.task     # MediaPipe FaceLandmarker model
└─ templates/               # HTML templates (index, register, logs, etc.)
```

---

## Notes
- Ensure GPU is available for YOLO and InsightFace for real-time performance.
- For local testing without GPU, set:
```python
DEVICE = torch.device("cpu")
```
- Adjust **YAWN_THRESHOLD** for drowsiness detection sensitivity.
- Make sure snapshots directory is writable:
```python
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
```

