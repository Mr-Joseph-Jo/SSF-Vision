import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import torch
import numpy as np
import threading
from PIL import Image, ImageTk
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import supervision as sv
from torchreid.utils import FeatureExtractor

# ==================== CONFIGURATION ====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_WEIGHTS = './log/osnet_duke/model/model.pth.tar-300' 
CAM_SOURCE_1 = "../videos/4PTF1_short.mp4"
CAM_SOURCE_2 = "../videos/4PTF2_short.mp4"
MATCH_THRESHOLD = 0.75 

# ==================== RE-ID ENGINE ====================

class OSNetReIDEngine:
    def __init__(self):
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=MODEL_WEIGHTS,
            device=DEVICE
        )
        self.tracker = sv.ByteTrack()
        self.detector = YOLO('yolov8n.pt')

    def get_features(self, crop):
        if crop is None or crop.size == 0: return None
        feat = self.extractor([crop])[0].cpu().numpy()
        return feat / np.linalg.norm(feat)

class GlobalReIDManager:
    def __init__(self, max_gallery_size=15):
        self.target_gallery = []
        self.max_gallery_size = max_gallery_size
        self.trail_history = deque(maxlen=50)

    def add_to_gallery(self, feature):
        if len(self.target_gallery) < self.max_gallery_size:
            self.target_gallery.append(feature)

    def compute_similarity(self, candidate_feature):
        if not self.target_gallery: return 0.0
        similarities = np.dot(self.target_gallery, candidate_feature)
        return np.max(similarities)

# ==================== MAIN DASHBOARD ====================

class SentinelVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentinel Vision - Integrated Security Suite")
        self.geometry("1200x900")
        self.configure(bg="#ecf0f1")

        # Initialize Engines
        self.engine = OSNetReIDEngine()
        self.manager = GlobalReIDManager()
        self.is_running = False
        self.selected_tid = None
        self.current_detections = []

        self.setup_ui()
        self.show_frame("Tracking") # Start on Re-ID page

    def setup_ui(self):
        # Sidebar
        self.sidebar = tk.Frame(self, width=250, bg="#2c3e50")
        self.sidebar.pack(side="left", fill="y")
        tk.Label(self.sidebar, text="SENTINEL AI", font=("Arial", 18, "bold"), fg="white", bg="#2c3e50", pady=20).pack()

        # Main Container
        self.container = tk.Frame(self, bg="#ecf0f1")
        self.container.pack(side="right", fill="both", expand=True)

        self.frames = {}
        for name, label in [("Dashboard", "🏠 Overview"), ("Tracking", "👤 Person Trail")]:
            frame = tk.Frame(self.container, bg="#ecf0f1")
            self.frames[name] = frame
            tk.Button(self.sidebar, text=label, command=lambda n=name: self.show_frame(n),
                      relief="flat", bg="#34495e", fg="white", font=("Arial", 12),
                      anchor="w", padx=20, pady=10).pack(fill="x", pady=2)

        self.setup_reid_ui(self.frames["Tracking"])

    def setup_reid_ui(self, frame):
        # Top: Video Feed
        self.video_label = tk.Label(frame, bg="black")
        self.video_label.pack(pady=10, fill="both", expand=True)
        self.video_label.bind("<Button-1>", self.on_video_click)

        # Middle: Controls
        ctrl_frame = tk.Frame(frame, bg="#ecf0f1")
        ctrl_frame.pack(fill="x", padx=20)
        self.status_lbl = tk.Label(ctrl_frame, text="Status: Ready. Click 'Start Phase 1'", bg="#ecf0f1")
        self.status_lbl.pack(side="left")
        
        tk.Button(ctrl_frame, text="Start Selection", command=lambda: self.start_thread(self.run_phase1)).pack(side="right", padx=5)
        tk.Button(ctrl_frame, text="Stop All", command=self.stop_processing, bg="#e74c3c", fg="white").pack(side="right")

        # Bottom: Scrollable Gallery
        tk.Label(frame, text="Timeline Matches", font=("Arial", 10, "bold"), bg="#ecf0f1").pack(anchor="w", padx=20)
        self.gal_canvas = tk.Canvas(frame, height=180, bg="#dfe6e9")
        self.gal_scroll = ttk.Scrollbar(frame, orient="horizontal", command=self.gal_canvas.xview)
        self.gal_inner = tk.Frame(self.gal_canvas, bg="#dfe6e9")

        self.gal_canvas.create_window((0, 0), window=self.gal_inner, anchor="nw")
        self.gal_canvas.configure(xscrollcommand=self.gal_scroll.set)
        self.gal_canvas.pack(fill="x", padx=20)
        self.gal_scroll.pack(fill="x", padx=20, pady=5)

    def show_frame(self, name):
        for f in self.frames.values(): f.pack_forget()
        self.frames[name].pack(fill="both", expand=True)

    def start_thread(self, func):
        self.is_running = True
        threading.Thread(target=func, daemon=True).start()

    def stop_processing(self):
        self.is_running = False

    def on_video_click(self, event):
        # Convert click coordinates to frame coordinates
        vx, vy = event.x, event.y
        for (xyxy, tid) in self.current_detections:
            x1, y1, x2, y2 = xyxy
            if x1 <= vx <= x2 and y1 <= vy <= y2:
                self.selected_tid = tid
                self.manager.target_gallery = [] 
                self.status_lbl.config(text=f"Target Locked: ID {tid}")

    def run_phase1(self):
        cap = cv2.VideoCapture(CAM_SOURCE_1)
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret or len(self.manager.target_gallery) >= self.manager.max_gallery_size: break

            results = self.engine.detector(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.engine.tracker.update_with_detections(detections)

            self.current_detections = []
            for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                self.current_detections.append((xyxy, tid))
                x1, y1, x2, y2 = map(int, xyxy)
                color = (0, 255, 255) if tid == self.selected_tid else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if tid == self.selected_tid:
                    crop = frame[y1:y2, x1:x2]
                    self.manager.add_to_gallery(self.engine.get_features(crop))

            self.update_video_label(frame)
        
        cap.release()
        if self.is_running: self.run_phase2()

    def run_phase2(self):
        self.status_lbl.config(text="Phase 2: Searching Cross-Camera...")
        cap = cv2.VideoCapture(CAM_SOURCE_2)
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break

            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = f"{int(msec//60000)}:{int((msec//1000)%60):02d}"

            results = self.engine.detector(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.engine.tracker.update_with_detections(detections)

            for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)
                feat = self.engine.get_features(frame[y1:y2, x1:x2])
                score = self.manager.compute_similarity(feat)

                if score > MATCH_THRESHOLD:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    self.add_to_gallery_ui(frame[y1:y2, x1:x2], "Cam_02", timestamp)

            self.update_video_label(frame)
        cap.release()

    def update_video_label(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((800, 450))
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)

    def add_to_gallery_ui(self, cv_crop, cam_name, time_str):
        # Convert crop for UI
        crop_pil = Image.fromarray(cv2.cvtColor(cv_crop, cv2.COLOR_BGR2RGB)).resize((100, 130))
        crop_tk = ImageTk.PhotoImage(crop_pil)

        thumb = tk.Frame(self.gal_inner, bg="white", bd=1, relief="solid")
        thumb.pack(side="left", padx=5, pady=5)
        
        lbl = tk.Label(thumb, image=crop_tk, bg="white")
        lbl.image = crop_tk
        lbl.pack()
        tk.Label(thumb, text=f"{cam_name}\n{time_str}", font=("Arial", 7), bg="white").pack()

        # Auto-scroll
        self.gal_inner.update_idletasks()
        self.gal_canvas.config(scrollregion=self.gal_canvas.bbox("all"))
        self.gal_canvas.xview_moveto(1.0)

if __name__ == "__main__":
    app = SentinelVisionApp()
    app.mainloop()