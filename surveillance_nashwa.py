"""
Smart Surveillance System - Zone Monitoring
Complete implementation: Draw polygon, detect people, alert on intrusion
"""

import cv2
import numpy as np
import os
from collections import defaultdict, deque
from typing import List, Tuple, Optional

VIDEO_PATH="./videos/cam1.1.mp4"

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings."""
    # YOLO settings
    MODEL_PATH = "yolov8n.pt"  # or "yolov8s.pt", "yolov8m.pt"
    CONFIDENCE_THRESHOLD = 0.5
    CLASS_IDS = [0]  # 0 = person in COCO dataset
    
    # Display settings
    POLYGON_COLOR = (0, 255, 0)  # Green
    POLYGON_THICKNESS = 2
    POLYGON_ALPHA = 0.3  # Transparency
    PERSON_BOX_COLOR = (255, 0, 0)  # Blue
    INTRUDER_COLOR = (0, 0, 255)  # Red
    TEXT_COLOR = (0, 0, 0)  # black
    
    # Alert settings
    ALERT_COOLDOWN = 5  # seconds between alerts for same person
    
    # Tracking settings
    TRACK_BUFFER = 30  # frames to keep track history

# ============================================================================
# POLYGON DRAWING CLASS
# ============================================================================

class PolygonDrawer:
    """Interactive polygon drawing on video."""
    
    def __init__(self, window_name="Draw Polygon Zone"):
        self.window_name = window_name
        self.points = []  # List of (x, y) points
        self.polygon_complete = False
        self.frame = None
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.polygon_complete:
                # Add point
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                # Check if we have at least 4 points
                if len(self.points) >= 4:
                    print("Minimum 4 points reached. Right-click to complete polygon.")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) >= 4 and not self.polygon_complete:
                # Complete the polygon
                self.polygon_complete = True
                print(f"Polygon completed with {len(self.points)} points")
                print("Press 's' to save and continue, 'r' to reset, 'q' to quit")
    
    def draw_polygon(self, frame):
        """Draw the polygon on the frame."""
        display_frame = frame.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(display_frame, point, 5, (0, 255, 255), -1)  # Yellow points
            cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points)):
                start_point = self.points[i]
                end_point = self.points[(i + 1) % len(self.points)]
                if i < len(self.points) - 1 or self.polygon_complete:
                    cv2.line(display_frame, start_point, end_point, 
                            (0, 255, 255), 2)  # Yellow lines
        
        # Draw filled polygon if complete
        if self.polygon_complete and len(self.points) >= 3:
            overlay = display_frame.copy()
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], Config.POLYGON_COLOR)
            cv2.addWeighted(overlay, Config.POLYGON_ALPHA, 
                           display_frame, 1 - Config.POLYGON_ALPHA, 0, display_frame)
            
            # Draw polygon outline
            cv2.polylines(display_frame, [pts], True, 
                         (0, 255, 0), Config.POLYGON_THICKNESS)
            
            # Add polygon name
            center_x = sum(p[0] for p in self.points) // len(self.points)
            center_y = sum(p[1] for p in self.points) // len(self.points)
            cv2.putText(display_frame, "RESTRICTED ZONE", (center_x-60, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.TEXT_COLOR, 2)
        
        # Draw instructions
        instructions = [
            "INSTRUCTIONS:",
            "1. Left-click: Add point",
            "2. Right-click: Complete polygon (need ≥4 points)",
            "3. 'r': Reset polygon",
            "4. 's': Save and start surveillance",
            "5. 'q': Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(display_frame, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.TEXT_COLOR, 2)
        
        # Show point count
        cv2.putText(display_frame, f"Points: {len(self.points)}/4 minimum", 
                   (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display_frame
    
    def get_polygon(self):
        """Get the current polygon points."""
        return self.points.copy() if self.polygon_complete else []
    
    def reset(self):
        """Reset the polygon."""
        self.points = []
        self.polygon_complete = False
        print("Polygon reset")

# ============================================================================
# PERSON DETECTOR
# ============================================================================

class PersonDetector:
    """Detect and track people using YOLO."""
    
    def __init__(self):
        if YOLO_AVAILABLE:
            print(f"Loading YOLO model: {Config.MODEL_PATH}")
            self.model = YOLO(Config.MODEL_PATH)
            print("YOLO model loaded successfully")
        else:
            print("Running in DEMO mode (no actual detection)")
            self.model = None
        
        # Simple tracking using IoU
        self.track_history = defaultdict(lambda: deque(maxlen=Config.TRACK_BUFFER))
        self.next_track_id = 0
        self.tracks = {}  # track_id: {'bbox': bbox, 'disappeared': frames}
        
    def detect_people(self, frame):
        """Detect people in the frame."""
        if self.model is None:
            # Demo mode - create dummy detections for testing
            h, w = frame.shape[:2]
            dummy_boxes = []
            # Create some dummy person boxes
            for i in range(2):
                x1 = w // 4 + i * w // 3
                y1 = h // 4
                x2 = x1 + 100
                y2 = y1 + 200
                dummy_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.9,
                    'class_id': 0
                })
            return dummy_boxes
        
        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf)
                class_id = int(box.cls)
                
                # Filter for people only with confidence threshold
                if (confidence > Config.CONFIDENCE_THRESHOLD and 
                    class_id in Config.CLASS_IDS):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id
                    })
        
        return detections
    
    def update_tracks(self, detections):
        """Simple tracking using IoU matching."""
        updated_detections = []
        
        # If no detections, mark all tracks as disappeared
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > 30:  # Remove after 30 frames
                    del self.tracks[track_id]
            return []
        
        # Calculate IoU matrix between tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(
                    self.tracks[track_id]['bbox'], det['bbox'])
        
        # Match tracks to detections
        matched_tracks = set()
        matched_detections = set()
        
        # Greedy matching based on IoU
        for i in range(len(track_ids)):
            if i in matched_tracks:
                continue
                
            best_iou = 0.3  # Threshold
            best_j = -1
            
            for j in range(len(detections)):
                if j in matched_detections:
                    continue
                    
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            
            if best_j != -1:
                # Update track
                track_id = track_ids[i]
                det = detections[best_j]
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['disappeared'] = 0
                det['track_id'] = track_id
                matched_tracks.add(i)
                matched_detections.add(best_j)
                updated_detections.append(det)
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'disappeared': 0
                }
                det['track_id'] = track_id
                updated_detections.append(det)
        
        # Mark unmatched tracks as disappeared
        for i, track_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > 30:
                    del self.tracks[track_id]
        
        return updated_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

# ============================================================================
# ZONE MONITORING SYSTEM
# ============================================================================

class ZoneMonitoringSystem:
    """Main system for monitoring polygon zones."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.polygon_points = []
        self.detector = PersonDetector()
        self.alert_cooldown = {}  # track_id: last_alert_time
        self.frame_count = 0
        
    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon."""
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_bottom_center(self, bbox):
        """Get bottom center point of bounding box (feet position)."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)
    
    def check_intrusion(self, detection, polygon_points):
        """Check if a person has entered the polygon zone."""
        if not polygon_points or len(polygon_points) < 3:
            return False
        
        # Get person's feet position (bottom center of bounding box)
        feet_position = self.get_bottom_center(detection['bbox'])
        
        # Check if feet are inside polygon
        return self.is_point_in_polygon(feet_position, polygon_points)
    
    def draw_person(self, frame, detection, is_intruder=False):
        """Draw person bounding box and info."""
        x1, y1, x2, y2 = detection['bbox']
        
        # Choose color based on intrusion status
        color = Config.INTRUDER_COLOR if is_intruder else Config.PERSON_BOX_COLOR
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with track ID and confidence
        label = f"Person"
        if 'track_id' in detection:
            label = f"ID: {detection['track_id']}"
        
        if 'confidence' in detection:
            label += f" ({detection['confidence']:.2f})"
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10),
                     (x1 + text_width, y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.TEXT_COLOR, 2)
        
        # Mark feet position
        feet_pos = self.get_bottom_center(detection['bbox'])
        cv2.circle(frame, feet_pos, 5, color, -1)
        
        return frame
    
    def draw_polygon_zone(self, frame, polygon_points):
        """Draw the polygon zone on frame."""
        if len(polygon_points) < 3:
            return frame
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Convert points to numpy array
        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], Config.POLYGON_COLOR)
        
        # Draw polygon outline
        cv2.polylines(overlay, [pts], True, 
                     Config.POLYGON_COLOR, Config.POLYGON_THICKNESS)
        
        # Apply transparency
        cv2.addWeighted(overlay, Config.POLYGON_ALPHA, 
                       frame, 1 - Config.POLYGON_ALPHA, 0, frame)
        
        # Add zone label
        if len(polygon_points) >= 4:
            center_x = sum(p[0] for p in polygon_points) // len(polygon_points)
            center_y = sum(p[1] for p in polygon_points) // len(polygon_points)
            cv2.putText(frame, "RESTRICTED ZONE", (center_x - 70, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.TEXT_COLOR, 2)
        
        return frame
    
    def run_surveillance(self, polygon_points):
        """Run the main surveillance loop."""
        if not polygon_points or len(polygon_points) < 4:
            print("Error: Need at least 4 points for polygon")
            return
        
        self.polygon_points = polygon_points
        
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  Path: {self.video_path}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Polygon points: {len(polygon_points)}")
        print("\nStarting surveillance...")
        print("Press 'q' to quit, 'p' to pause/resume")
        # Make the surveillance display window resizable and set to video resolution
        cv2.namedWindow("Smart Surveillance - Zone Monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Surveillance - Zone Monitoring", frame_width, frame_height)
        
        paused = False
        self.frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                self.frame_count += 1
                
                # Detect people
                raw_detections = self.detector.detect_people(frame)
                
                # Update tracks
                detections = self.detector.update_tracks(raw_detections)
                
                # Check each person for intrusion
                intrusion_detected = False
                for detection in detections:
                    is_intruder = self.check_intrusion(detection, polygon_points)
                    
                    if is_intruder and 'track_id' in detection:
                        track_id = detection['track_id']
                        current_time = self.frame_count / fps  # Simulated time
                        
                        # Check cooldown
                        if (track_id not in self.alert_cooldown or 
                            current_time - self.alert_cooldown[track_id] > Config.ALERT_COOLDOWN):
                            
                            # Trigger alert
                            print(f"\n🚨 ALERT! Person ID {track_id} entered restricted zone!")
                            print(f"   Frame: {self.frame_count}")
                            print(f"   Position: {self.get_bottom_center(detection['bbox'])}")
                            
                            # Update cooldown
                            self.alert_cooldown[track_id] = current_time
                        
                        intrusion_detected = True
                    
                    # Draw person
                    frame = self.draw_person(frame, detection, is_intruder)
                
                # Draw polygon zone
                frame = self.draw_polygon_zone(frame, polygon_points)
                
                # Draw statistics
                cv2.putText(frame, f"Frame: {self.frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.TEXT_COLOR, 2)
                cv2.putText(frame, f"People: {len(detections)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.TEXT_COLOR, 2)
                cv2.putText(frame, f"Polygon Points: {len(polygon_points)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.TEXT_COLOR, 2)
                
                # Draw alert banner if intrusion detected
                if intrusion_detected:
                    cv2.putText(frame, "INTRUSION DETECTED!", 
                               (frame_width // 2 - 100, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, Config.INTRUDER_COLOR, 3)
                
                # Show frame
                cv2.imshow("Smart Surveillance - Zone Monitoring", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nSurveillance stopped by user")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\nSurveillance Summary:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Total alerts triggered: {len(self.alert_cooldown)}")
        print(f"  Polygon zone monitored: {len(polygon_points)} points")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the surveillance system."""
    print("\n" + "="*60)
    print("SMART SURVEILLANCE SYSTEM - POLYGON ZONE MONITORING")
    print("="*60)
    
    # Ask for video path
    video_path = VIDEO_PATH
    
    if not video_path:
        # Check for sample video
        sample_videos = ["sample.mp4", "test.mp4", "video.mp4", "people.mp4"]
        for sample in sample_videos:
            if os.path.exists(sample):
                video_path = sample
                print(f"Using sample video: {sample}")
                break
        
        if not video_path:
            print("No video file specified and no sample found.")
            print("Please create a 'sample.mp4' file or specify a video path.")
            return
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Step 1: Draw polygon
    print("\n" + "="*60)
    print("STEP 1: DRAW POLYGON ZONE")
    print("="*60)
    
    # Open video for drawing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        cap.release()
        return
    
    cap.release()
    
    # Create polygon drawer
    drawer = PolygonDrawer("Draw Polygon Zone")
    # Make the drawing window resizable and set initial size to the frame resolution
    cv2.namedWindow(drawer.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(drawer.window_name, first_frame.shape[1], first_frame.shape[0])
    cv2.setMouseCallback(drawer.window_name, drawer.mouse_callback)
    
    print("\nDraw a polygon zone on the video frame:")
    print("1. Click at least 4 points to define the polygon")
    print("2. Right-click to complete the polygon")
    print("3. Press 's' to save and start surveillance")
    print("4. Press 'r' to reset the polygon")
    print("5. Press 'q' to quit")
    
    while True:
        # Draw polygon on frame
        display_frame = drawer.draw_polygon(first_frame.copy())
        cv2.imshow(drawer.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r') or key == ord('R'):
            drawer.reset()
        
        elif key == ord('s') or key == ord('S'):
            polygon_points = drawer.get_polygon()
            if polygon_points:
                print(f"\nPolygon saved with {len(polygon_points)} points")
                break
            else:
                print("Please complete the polygon first (right-click)")
        
        elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
            print("\nExiting without surveillance")
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()
    
    # Step 2: Run surveillance
    print("\n" + "="*60)
    print("STEP 2: SURVEILLANCE MODE")
    print("="*60)
    
    # Create and run surveillance system
    surveillance = ZoneMonitoringSystem(video_path)
    surveillance.run_surveillance(polygon_points)
    
    print("\n" + "="*60)
    print("SYSTEM SHUTDOWN")
    print("="*60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if OpenCV is installed
    try:
        cv2.__version__
    except:
        print("Error: OpenCV not installed. Install with: pip install opencv-python")
        exit(1)
    
    # Run the system
    main()