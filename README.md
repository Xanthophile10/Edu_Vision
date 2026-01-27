# Edu_Vision
Face Recognition & Monitoring System
Overview

This project is an advanced facial recognition and monitoring system built with Python, Flask, InsightFace, YOLO, and MediaPipe. It integrates multiple functionalities for real-time monitoring, attendance logging, visitor management, fatigue detection, and behavioral monitoring.

The system is designed for educational or workplace environments, providing an efficient way to track attendance, detect yawning, monitor human poses, and detect mobile phone usage.

Key Features

Face Recognition

Detects and recognizes faces using InsightFace (buffalo_l model).

Maintains a database of students and visitors.

Multi-frame embedding for stable recognition.

Logs attendance automatically when a recognized student’s RFID matches.

Yawn Detection

Detects yawning using MediaPipe FaceLandmarker.

Calculates Mouth Open Ratio (MOR) and Eye Aspect Ratio (EAR).

Visual alerts for fatigue detection.

Pose Detection

Real-time human pose estimation using YOLOv26 pose model.

Highlights human skeleton and keypoints.

Phone Detection

Detects mobile phones using YOLOv26 COCO model.

Alerts displayed on video frames.

RFID Integration

Reads RFID cards via Arduino serial communication.

Links recognized students to their RFID for accurate attendance logging.

Visitor Management

Register temporary or permanent visitors.

Auto-delete expired visitor records.

Save embeddings and snapshots for visitors.

Web Dashboard

Built with Flask.

Features:

Live camera feeds

Attendance and visitor logs

Downloadable CSV reports

Registration and management of students and visitors

Multi-Camera Support

Supports multiple RTSP or local cameras.

Parallel threading for simultaneous camera feeds.

System Requirements

Python >= 3.10

CUDA-enabled GPU (optional, for faster YOLO inference)

Arduino (optional, for RFID integration)

Libraries:

pip install opencv-python flask insightface mediapipe ultralytics torch pandas numpy


YOLOv26 models:

yolo26l-pose.pt (pose detection)

yolo26l.pt (phone detection)

MediaPipe FaceLandmarker:

face_landmarker.task (yawn detection)

Installation

Clone the repository:

git clone <your-repo-url>
cd face-recognition-monitoring


Install dependencies:

pip install -r requirements.txt


Place required models in the project directory:

yolo26l-pose.pt

yolo26l.pt

face_landmarker.task

Configure cameras:

Edit CAMERA_SOURCES in the Python file with RTSP URLs or local camera indices.

Configure Arduino (optional):

Update SERIAL_PORT in the Python file with your Arduino COM port.

Usage

Run the main application:

python app.py


Open your browser and go to:

http://localhost:5000

Available Pages

/ – Dashboard (attendance summary, live feeds)

/register – Student registration

/register_camera – Camera feed for registration

/recognize – Live recognition feed

/logs – Attendance logs

/attendance – Attendance reports

/manage_registered_students – Edit/delete students

/register_visitor – Visitor registration

/manage_visitors – Edit/delete visitors

/download/attendance_csv – Download CSV

/download/visitor_logs_csv – Download CSV

Configuration

Yawn Detection Thresholds:

YAWN_THRESHOLD = 0.40
EYE_CLOSED_THRESHOLD = 0.27
YAWN_FRAMES = 10


Face Recognition:

Embeddings stored in face_db.pkl

Snapshots saved in face_snapshots/

RFID Integration:

Reads card ID from Arduino

Matches with registered students for automatic attendance

Notes

Uses threading for real-time multi-camera processing.

GPU support improves YOLO inference speed.

Logs and snapshots are automatically saved.

Ensure RTSP URLs are valid and accessible.

Intended for educational or monitoring purposes—use ethically.
