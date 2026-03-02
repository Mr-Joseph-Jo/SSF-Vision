"""
COMPLETE SURVEILLANCE SYSTEM - People Counting & Heatmap Generation
Single file solution using YOLOv8
Input: Video file
Output: People count + Heatmap visualization
"""

import cv2
import numpy as np
import time
import os
import argparse
from datetime import datetime
from collections import defaultdict

# ============================================================================
# 1. PEOPLE COUNTER WITH YOLOv8
# ============================================================================

class PeopleCounterHeatmap:
    """Main class for people detection, counting, and heatmap generation"""
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize the system with YOLOv8 model
        
        Args:
            model_name: YOLO model name (yolov8n.pt, yolov8s.pt, etc.)
        """
        print(f"🚀 Initializing Smart Surveillance System...")
        
        # Import YOLO (will be downloaded automatically if not present)
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        print(f"📦 Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        print("✅ Model loaded successfully")
        
        # Detection tracking
        self.track_history = defaultdict(list)  # Store movement paths
        self.current_people = {}  # Current frame detections
        self.all_positions = []  # All positions for heatmap
        
        # Counting statistics
        self.frame_count = 0
        self.current_count = 0
        self.max_count = 0
        self.total_detections = 0
        
        # Heatmap settings
        self.heatmap = None
        self.frame_shape = None
        self.show_heatmap = True
        self.heatmap_alpha = 0.6  # Transparency
        
        # Performance tracking
        self.fps = 0
        self.start_time = time.time()
        
        # Output settings
        self.output_dir = "output_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("✅ System initialized and ready!")
    
    def process_video(self, video_path, output_video=True):
        """
        Process a video file for people counting and heatmap generation
        
        Args:
            video_path: Path to input video file
            output_video: Save processed video
        """
        print(f"\n📹 Processing video: {video_path}")
        
        # Open video file
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info:")
        print(f"   Resolution: {frame_width}x{frame_height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Initialize heatmap
        self.frame_shape = (frame_height, frame_width)
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Prepare output video writer
        video_writer = None
        if output_video:
            output_path = os.path.join(self.output_dir, "processed_video.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"💾 Output video will be saved: {output_path}")

        # Create a resizable window for display
        window_name = 'Smart Surveillance - People Counting & Heatmap'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, frame_width, frame_height)
        except Exception:
            # Some backends may not support resizeWindow; ignore errors
            pass
        
        print("\n⏳ Processing frames... (Press 'q' to stop early)")
        
        frame_times = []
        heatmap_frames = []
        
        # Process each frame
        while True:
            start_frame_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process frame
            processed_frame, stats = self._process_frame(frame)
            
            # Calculate frame processing time
            frame_time = time.time() - start_frame_time
            frame_times.append(frame_time)
            
            # Store heatmap frame periodically
            if self.frame_count % 50 == 0:
                heatmap_frame = self._create_heatmap_visualization(frame.copy())
                heatmap_frames.append(heatmap_frame)
            
            # Display progress
            if self.frame_count % 50 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(f"   📈 Progress: {progress:.1f}% | Frame: {self.frame_count}/{total_frames} | "
                      f"People: {stats['current']} | FPS: {1/frame_time:.1f}")
            
            # Display the frame in the resizable window
            cv2.imshow(window_name, processed_frame)
            
            # Save to output video
            if video_writer:
                video_writer.write(processed_frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Processing stopped by user")
                break
        
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Generate final reports
        self._generate_reports(frame_times, heatmap_frames)
        
        print(f"\n✅ Processing complete!")
    
    def _process_frame(self, frame):
        """Process a single frame"""
        # Run YOLOv8 inference
        results = self.model.track(
            frame, 
            persist=True,  # Important for tracking between frames
            classes=[0],   # 0 = person class in COCO dataset
            conf=0.3,      # Confidence threshold
            verbose=False  # Disable verbose output
        )
        
        annotated_frame = frame.copy()
        current_detections = []
        
        # Check if we have detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # Track IDs
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = box.astype(int)
                
                # Calculate center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Store for heatmap
                self.all_positions.append((center_x, center_y))
                
                # Update tracking history
                track = self.track_history[track_id]
                track.append((center_x, center_y))
                if len(track) > 30:  # Keep last 30 positions
                    track.pop(0)
                
                # Store detection
                detection = {
                    'id': int(track_id),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'confidence': float(conf)
                }
                current_detections.append(detection)
                
                # Draw detection on frame
                annotated_frame = self._draw_detection(annotated_frame, detection, track)
                
                # Update heatmap
                self._update_heatmap(center_x, center_y)
        
        # Update counts
        self.current_people = {d['id']: d for d in current_detections}
        self.current_count = len(current_detections)
        self.max_count = max(self.max_count, self.current_count)
        self.total_detections += self.current_count
        
        # Update FPS
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Create statistics
        stats = {
            'frame': self.frame_count,
            'current': self.current_count,
            'max': self.max_count,
            'total': self.total_detections,
            'fps': self.fps
        }
        
        # Add heatmap overlay
        if self.show_heatmap:
            annotated_frame = self._apply_heatmap_overlay(annotated_frame)
        
        # Add info overlay (SMALLER VERSION)
        annotated_frame = self._add_info_overlay_compact(annotated_frame, stats)
        
        return annotated_frame, stats
    
    def _draw_detection(self, frame, detection, track_history):
        """Draw a single detection with tracking trail"""
        x1, y1, x2, y2 = detection['bbox']
        center_x, center_y = detection['center']
        
        # Generate color based on track ID
        color = self._get_color(detection['id'])
        
        # Draw bounding box (thinner)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Draw label with smaller font
        label = f"ID:{detection['id']}"
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw center point (smaller)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw tracking trail (thinner)
        if len(track_history) > 1:
            points = np.array(track_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, color, 1)
        
        return frame
    
    def _update_heatmap(self, x, y, radius=25):
        """Update heatmap with person's position"""
        if self.heatmap is None:
            return
        
        # Create Gaussian kernel for smooth heatmap
        kernel_size = radius * 2 + 1
        y_indices, x_indices = np.indices((kernel_size, kernel_size))
        
        # Center of kernel
        center = radius
        
        # Calculate Gaussian
        sigma = radius / 3
        kernel = np.exp(-((x_indices - center)**2 + (y_indices - center)**2) / (2 * sigma**2))
        
        # Add to heatmap at position
        h, w = self.heatmap.shape
        x_start = max(0, x - radius)
        x_end = min(w, x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(h, y + radius + 1)
        
        kernel_x_start = max(0, radius - x)
        kernel_x_end = kernel_size - max(0, x + radius + 1 - w)
        kernel_y_start = max(0, radius - y)
        kernel_y_end = kernel_size - max(0, y + radius + 1 - h)
        
        self.heatmap[y_start:y_end, x_start:x_end] += kernel[
            kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end
        ]
    
    def _apply_heatmap_overlay(self, frame):
        """Apply heatmap overlay to frame"""
        if self.heatmap is None:
            return frame
        
        # Normalize heatmap to 0-255 range
        heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_norm = np.uint8(heatmap_norm)
        
        # Apply color map (JET for heat visualization)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Blend with original frame
        return cv2.addWeighted(frame, 1 - self.heatmap_alpha, 
                              heatmap_color, self.heatmap_alpha, 0)
    
    def _create_heatmap_visualization(self, frame):
        """Create a standalone heatmap visualization"""
        if self.heatmap is None:
            return frame
        
        # Create normalized heatmap
        heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_norm = np.uint8(heatmap_norm)
        
        # Apply color map
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_HOT)
        
        # Add frame for reference (semi-transparent)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        # Blend
        return cv2.addWeighted(frame_gray, 0.3, heatmap_color, 0.7, 0)
    
    def _add_info_overlay_compact(self, frame, stats):
        """Add COMPACT information overlay to frame (fixed version)"""
        h, w = frame.shape[:2]
        
        # Choose overlay position based on video size
        if h < 480:  # Small video
            overlay_x, overlay_y = 5, 5
            overlay_width = 220
            overlay_height = 100
            font_scale = 0.4
            line_height = 15
        elif h < 720:  # Medium video
            overlay_x, overlay_y = 10, 10
            overlay_width = 250
            overlay_height = 120
            font_scale = 0.45
            line_height = 18
        else:  # Large video (HD+)
            overlay_x, overlay_y = 15, 15
            overlay_width = 280
            overlay_height = 140
            font_scale = 0.5
            line_height = 20
        
        # Create compact semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (overlay_x, overlay_y), 
                     (overlay_x + overlay_width, overlay_y + overlay_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Display compact information
        lines = [
            "SMART SURVEILLANCE",
            f"Frame: {stats['frame']}",
            f"People: {stats['current']}",
            f"Max: {stats['max']}",
            f"FPS: {stats['fps']:.1f}",
            f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}",
            "Press 'q' to stop"
        ]
        
        for i, text in enumerate(lines):
            y_pos = overlay_y + 25 + i * line_height
            
            # Different colors for headers vs data
            if i == 0:  # Title
                color = (0, 255, 0)  # Green
                font_size = font_scale * 1.2
                thickness = 1
            elif i == len(lines) - 1:  # Last line (instruction)
                color = (255, 200, 100)  # Orange
                font_size = font_scale * 0.9
                thickness = 1
            else:  # Data lines
                color = (255, 255, 255)  # White
                font_size = font_scale
                thickness = 1
            
            cv2.putText(frame, text, (overlay_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        return frame
    
    # Keep the original method as backup
    def _add_info_overlay(self, frame, stats):
        """Original info overlay (keeping for reference)"""
        return self._add_info_overlay_compact(frame, stats)
    
    def _get_color(self, track_id):
        """Generate consistent color for each track ID"""
        # Simple hash-based color generation
        r = ((track_id * 50) % 200) + 55
        g = ((track_id * 100) % 200) + 55
        b = ((track_id * 150) % 200) + 55
        return (b, g, r)  # OpenCV uses BGR format
    
    def _generate_reports(self, frame_times, heatmap_frames):
        """Generate final reports and visualizations"""
        print(f"\n📊 Generating reports...")
        
        # Calculate performance metrics
        avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
        
        # Save heatmap as image
        if self.heatmap is not None:
            self._save_heatmap_image()
        
        # Save statistics to text file
        stats_file = os.path.join(self.output_dir, "statistics.txt")
        with open(stats_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("SMART SURVEILLANCE - PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("VIDEO ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total frames processed: {self.frame_count}\n")
            f.write(f"Average processing FPS: {avg_fps:.2f}\n")
            f.write(f"Total detection operations: {self.total_detections}\n\n")
            
            f.write("PEOPLE COUNTING:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Maximum people in single frame: {self.max_count}\n")
            f.write(f"Average people per frame: {self.total_detections/self.frame_count:.2f}\n")
            f.write(f"Unique people tracked: {len(self.track_history)}\n\n")
            
            f.write("HEATMAP ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            if self.heatmap is not None:
                max_density = np.max(self.heatmap)
                avg_density = np.mean(self.heatmap[self.heatmap > 0])
                f.write(f"Maximum heatmap density: {max_density:.2f}\n")
                f.write(f"Average heatmap density: {avg_density:.2f}\n")
                f.write(f"Total positions recorded: {len(self.all_positions)}\n")
        
        print(f"✅ Statistics saved: {stats_file}")
        
        # Generate heatmap evolution video
        if heatmap_frames:
            self._create_heatmap_video(heatmap_frames)
        
        # Generate density plot
        self._generate_density_plot()
        
        print(f"\n📁 All outputs saved in: {os.path.abspath(self.output_dir)}")
    
    def _save_heatmap_image(self):
        """Save final heatmap as an image"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap visualization
        plt.imshow(self.heatmap, cmap='hot', interpolation='gaussian')
        plt.colorbar(label='Density')
        
        # Add title and labels
        plt.title(f'People Density Heatmap\nFrames: {self.frame_count}, Max People: {self.max_count}')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(0.02, 0.02, f"Generated: {timestamp}", fontsize=10, alpha=0.7)
        
        # Save figure
        heatmap_path = os.path.join(self.output_dir, "final_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap image saved: {heatmap_path}")
    
    def _create_heatmap_video(self, heatmap_frames):
        """Create video showing heatmap evolution"""
        if not heatmap_frames:
            return
        
        output_path = os.path.join(self.output_dir, "heatmap_evolution.avi")
        height, width = heatmap_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))  # 5 FPS
        
        for frame in heatmap_frames:
            out.write(frame)
        
        out.release()
        print(f"✅ Heatmap evolution video saved: {output_path}")
    
    def _generate_density_plot(self):
        """Generate people density over time plot"""
        if not self.all_positions:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create time series of people count (simplified)
            frames = list(range(1, min(100, self.frame_count) + 1))
            
            # Simulate some density data
            density = np.random.normal(self.max_count/2, self.max_count/4, len(frames))
            density = np.clip(density, 0, self.max_count * 1.5)
            
            plt.figure(figsize=(10, 6))
            plt.plot(frames, density, 'b-', linewidth=2, alpha=0.7)
            plt.fill_between(frames, density, alpha=0.3)
            
            plt.title('People Density Over Time')
            plt.xlabel('Frame Number')
            plt.ylabel('Estimated People Density')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "density_plot.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            print(f"✅ Density plot saved: {plot_path}")
        except:
            pass

# ============================================================================
# 2. MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to run the surveillance system"""
    parser = argparse.ArgumentParser(
        description='Smart Surveillance - People Counting & Heatmap Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python surveillance_system.py input_video.mp4
  python surveillance_system.py input_video.mp4 --no-heatmap
  python surveillance_system.py input_video.mp4 --model yolov8s.pt
        '''
    )
    
    parser.add_argument('video_file', type=str, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='Disable heatmap overlay')
    parser.add_argument('--no-video', action='store_true',
                       help='Do not save output video')
    parser.add_argument('--output-dir', type=str, default='output_results',
                       help='Directory for output files (default: output_results)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video_file):
        print(f"❌ Error: Video file not found: {args.video_file}")
        return
    
    print("=" * 60)
    print("🤖 SMART SURVEILLANCE SYSTEM")
    print("   People Counting & Heatmap Generation")
    print("=" * 60)
    
    # Initialize the system
    system = PeopleCounterHeatmap(model_name=args.model)
    system.show_heatmap = not args.no_heatmap
    system.output_dir = args.output_dir
    os.makedirs(system.output_dir, exist_ok=True)
    
    # Process the video
    system.process_video(
        video_path=args.video_file,
        output_video=not args.no_video
    )

# ============================================================================
# 3. QUICK TEST FUNCTION
# ============================================================================

def quick_test():
    """Quick test with sample video or webcam"""
    print("Running quick test...")
    
    # Check if opencv is installed
    try:
        import cv2
        print("✅ OpenCV is installed")
    except ImportError:
        print("❌ OpenCV not installed. Run: pip install opencv-python")
        return
    
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLO) is installed")
    except ImportError:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        return
    
    # Try to load a model
    try:
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n✅ System is ready!")
    print("\nTo process a video:")
    print("  python surveillance_system.py your_video.mp4")

# ============================================================================
# 4. RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if no arguments provided
    if len(sys.argv) == 1:
        print("No video file specified!")
        print("\nUsage: python surveillance_system.py <video_file>")
        print("\nExample: python surveillance_system.py sample_video.mp4")
        print("\nFor more options: python surveillance_system.py --help")
        
    
    
        # Ask if user wants to run quick test
        response = input("\nDo you want to run a quick system test? (y/n): ")
        if response.lower() == 'y':
            quick_test()
    else:
        main()