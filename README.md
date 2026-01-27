# Face Recognition & Monitoring System

## Overview

This project is an advanced facial recognition and monitoring system built with Python, Flask, InsightFace, YOLO, and MediaPipe. It integrates multiple functionalities for real-time monitoring, attendance logging, visitor management, fatigue detection, and behavioral monitoring.

The system is designed for educational or workplace environments, providing an efficient way to track attendance, detect yawning, monitor human poses, and detect mobile phone usage.

---

## Key Features

1. **Face Recognition**
   - Detects and recognizes faces using InsightFace (buffalo_l model).
   - Maintains a database of students and visitors.
   - Multi-frame embedding for stable recognition.
   - Logs attendance automatically when a recognized student’s RFID matches.

2. **Yawn Detection**
   - Detects yawning using MediaPipe FaceLandmarker.
   - Calculates Mouth Open Ratio (MOR) and Eye Aspect Ratio (EAR).
   - Visual alerts for fatigue detection.

3. **Pose Detection**
   - Real-time human pose estimation using YOLOv26 pose model.
   - Highlights human skeleton and keypoints.

4. **Phone Detection**
   - Detects mobile phones using YOLOv26 COCO model.
   - Alerts displayed on video frames.

5. **RFID Integration**
   - Reads RFID cards via Arduino serial communication.
   - Links recognized students to their RFID for accurate attendance logging.

6. **Visitor Management**
   - Register temporary or permanent visitors.
   - Auto-delete expired visitor records.
   - Save embeddings and snapshots for visitors.

7. **Web Dashboard**
   - Built with Flask.
   - Features:
     - Live camera feeds
     - Attendance and visitor logs
     - Downloadable CSV reports
     - Registration and management of students and visitors

8. **Multi-Camera Support**
   - Supports multiple RTSP or local cameras.
   - Parallel threading for simultaneous camera feeds.

---

## System Requirements

- Python >= 3.10
- CUDA-enabled GPU (optional, for faster YOLO inference)
- Arduino (optional, for RFID integration)
- Libraries:

```
pip install opencv-python flask insightface mediapipe ultralytics torch pandas numpy
```

- YOLOv26 models:
  - yolo26l-pose.pt (pose detection)
  - yolo26l.pt (phone detection)
- MediaPipe FaceLandmarker:
  - face_landmarker.task (yawn detection)

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd face-recognition-monitoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place required models in the project directory:
   - yolo26l-pose.pt  
   - yolo26l.pt  
   - face_landmarker.task  

4. Configure cameras:
   - Edit CAMERA_SOURCES in the Python file with RTSP URLs or local camera indices.

5. Configure Arduino (optional):
   - Update SERIAL_PORT in the Python file with your Arduino COM port.

---

## Usage

Run the main application:
```bash
python app.py
```

Open your browser and go to:
```
http://localhost:5000
```

### Available Pages

- `/` – Dashboard (attendance summary, live feeds)  
- `/register` – Student registration  
- `/register_camera` – Camera feed for registration  
- `/recognize` – Live recognition feed  
- `/logs` – Attendance logs  
- `/attendance` – Attendance reports  
- `/manage_registered_students` – Edit/delete students  
- `/register_visitor` – Visitor registration  
- `/manage_visitors` – Edit/delete visitors  
- `/download/attendance_csv` – Download CSV  
- `/download/visitor_logs_csv` – Download CSV  

---

## Configuration

- Yawn Detection Thresholds:
```python
YAWN_THRESHOLD = 0.40
EYE_CLOSED_THRESHOLD = 0.27
YAWN_FRAMES = 10
```

- Face Recognition:
  - Embeddings stored in `face_db.pkl`
  - Snapshots saved in `face_snapshots/`

- RFID Integration:
  - Reads card ID from Arduino
  - Matches with registered students for automatic attendance

---

## Notes

- Uses threading for real-time multi-camera processing.
- GPU support improves YOLO inference speed.
- Logs and snapshots are automatically saved.
- Ensure RTSP URLs are valid and accessible.
- Intended for educational or monitoring purposes—use ethically.

---

## License

MIT License – free to use, modify, and distribute.

