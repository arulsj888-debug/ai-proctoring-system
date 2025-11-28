# AI Automated Proctoring System

An advanced, AI-powered remote proctoring solution designed to ensure the integrity of online assessments. This system replaces human invigilators with an intelligent agent that monitors student behavior in real-time using computer vision.

## 🚀 Features

*   **Multi-Model AI Pipeline:** Runs **YOLOv8/v12** (Object Detection) and **MediaPipe** (Face Mesh) concurrently to detect violations.
*   **Real-Time Violation Detection:**
    *   **Mobile Phone Detection:** Instantly flags unauthorized devices.
    *   **Multiple Person Detection:** Detects if more than one person is in the frame.
    *   **Gaze Tracking:** Calculates Eye Aspect Ratio (EAR) to detect looking away or sleeping.
*   **Live Proctor Dashboard:** A WebSocket-powered interface for human supervisors to view live feeds and alerts.
*   **Automated Reporting:** Logs violations to MongoDB and emails evidence snapshots to administrators.

## 🛠️ Tech Stack

*   **Backend:** Python 3.12.4, Flask, Socket.IO
*   **AI/ML:** Ultralytics YOLO, MediaPipe, OpenCV
*   **Database:** MongoDB
*   **Frontend:** HTML5, JavaScript (WebSockets)

## 📋 Prerequisites

*   **Python 3.12.4** (Strictly recommended)
*   **MongoDB** (Running locally on default port 27017 or a cloud URI)
*   **Webcam** (For testing the student client)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/ai-proctoring-system.git
    cd ai-proctoring-system
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Configuration:**
    *   Open `config.py` and update `MONGO_HOST` if needed.
    *   If using email alerts, configure your SMTP settings in `app.py` (Environment variables recommended).

5.  **Download YOLO Model:**
    *   Ensure `yolo12n.pt` is present in the root directory. If not included due to size limits, download it from Ultralytics and place it here.

## 🚀 Usage

1.  **Start the Server:**
    ```bash
    python src/app.py
    ```
    *The server will start on `http://0.0.0.0:YOUR_PORT`*

2.  **Student View:**
    *   Open `http://localhost:YOUR_PORT` in your browser.
    *   Enter a User ID and Email to start the exam session.
    *   Grant camera permissions.

3.  **Proctor Dashboard:**
    *   Open `http://localhost:YOUR_PORT/dashboard` in a separate tab/window.
    *   You will see live feeds of active students and receive real-time violation alerts.

## 🛡️ Logic & Violations

*   **Mobile Detected:** Immediate Kick-out.
*   **Multiple People:** 2 Warnings -> Kick-out.
*   **Looking Away/No Face:** Warning logged.

---
**Note:** This project was built for educational and demonstration purposes.