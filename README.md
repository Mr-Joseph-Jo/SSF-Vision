# SSF-Vision 👁️🛡️

**SSF-Vision** is an intelligent surveillance system built for real-time monitoring and threat detection. It leverages computer vision and AI to enable smart camera feeds, automated alerts, and event logging — designed for scalable, self-hosted deployment.

## ✨ Features

* **Real-Time Monitoring:** Processes video feeds on the fly using advanced computer vision models (powered by YOLOv8).
* **Threat & Anomaly Detection:** Intelligent algorithms designed to detect unusual behavior or unauthorized access (`anomaly.py`).
* **Person Re-Identification (ReID):** Tracks and re-identifies individuals across multiple frames or camera feeds (`reid.py`, `suspect_finder.py`).
* **Automated Event Logging:** Keeps track of security events automatically, saving structured data to CSV (`security_logs.csv`) for later auditing and analysis.
* **Hardware Acceleration:** Includes experimental support for DirectML (`test_directml.py`) to leverage GPU acceleration on supported hardware.

## 📂 Project Structure

* `main.py` - The main entry point to run the surveillance system.
* `anomaly.py` - Core logic for detecting anomalous behaviors or events.
* `reid.py` / `reid1.0.py` - Scripts handling Person Re-Identification.
* `suspect_finder.py` - Module for finding and tracking specific individuals.
* `surveillance_*.py` - Specialized/modular surveillance configurations.
* `yolov8n.pt` - Pre-trained YOLOv8 Nano model weights used for fast object detection.
* `requirements.txt` - Python dependencies required to run the project.
* `analytics_output/` - Directory storing generated analytics and system outputs.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mr-Joseph-Jo/SSF-Vision.git](https://github.com/Mr-Joseph-Jo/SSF-Vision.git)
   cd SSF-Vision