import cv2
import os
import numpy as np
import csv
import threading
import queue
import torch
from abc import ABC, abstractmethod
from ultralytics import YOLO
from collections import deque
from datetime import datetime

# =============================================================================
# UNIFIED SECURITY DETECTION SYSTEM
# Combines: Fall Detection | Loitering Detection | Panic/Crowd Run Detection
# =============================================================================

# --- VIDEO PATH ---
#VIDEO_PATH = './videos/crowd.mp4'      # for panic / crowd detection
VIDEO_PATH = './videos/loitering.avi' # for loitering detection
#VIDEO_PATH = './videos/test.mp4'      # for fall / fight detection

# --- MODEL & OUTPUT ---
MODEL_PATH    = 'yolov8n-pose.pt'
SAVE_DIR      = 'evidence_clips'
FALL_SAVE_DIR = os.path.join(SAVE_DIR, 'fall')

# --- LOITERING THRESHOLDS (seconds — auto-converted to frames at runtime) ---
LIMIT_STATIONARY_S      = 15
LIMIT_PACING_S          = 25
LIMIT_CROUCHING_S       = 5
LIMIT_RUNNING_S         = 2
MOVEMENT_THRESHOLD      = 40
RUNNING_SPEED_THRESHOLD = 15

# --- BEHAVIOUR STABILISATION ---
BEHAVIOR_CONFIRM_FRAMES = 15

# --- PANIC / CROWD THRESHOLDS ---
PANIC_RUNNING_SPEED = 20
CROWD_DISTANCE      = 150
CROWD_MIN_COUNT     = 3

# --- FALL / FIGHT THRESHOLDS ---
MOVE_THRESHOLD          = 0.0015
SMOOTHING_WINDOW        = 5
BUFFER_SECONDS          = 5
ASPECT_RATIO_FALL_THRESHOLD = 0.8
FALL_HIP_DROP_THRESHOLD = 0.04
FIGHT_WRIST_THRESHOLD   = 0.35
FIGHT_HEAD_THRESHOLD    = 0.12
ALERT_COOLDOWN_SECONDS  = 5

# --- GENERAL ---
CONF_THRESHOLD = 0.50
BLUR_FACES     = True


# =============================================================================
# DEVICE SELECTION  (CUDA → MPS → CPU)
# =============================================================================

def select_device() -> str:
    """
    Returns the best available device string for PyTorch / Ultralytics.
      - 'cuda'  : NVIDIA GPU (fastest)
      - 'mps'   : Apple Silicon GPU
      - 'cpu'   : fallback
    half-precision (float16) is only safe on CUDA.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# ASYNC CSV LOGGER  (persistent handle, queue-based, flush-safe)
# =============================================================================

class CSVLogger:
    """
    Thread-safe, async CSV writer.
    Writes happen in a background thread so the main loop never blocks on I/O.
    Calling shutdown() flushes all pending rows and closes the file cleanly.
    """
    def __init__(self, path: str):
        self._q      = queue.Queue()
        self._path   = path
        self._thread = threading.Thread(target=self._worker, daemon=True)
        with open(path, mode='w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Track_ID", "Event", "Status"])
        self._thread.start()

    def log(self, tid: int, event: str, status: str = "High"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._q.put((ts, tid, event, status))

    def shutdown(self):
        self._q.put(None)           # sentinel
        self._thread.join(timeout=5)

    def _worker(self):
        with open(self._path, mode='a', newline='', buffering=1) as f:
            writer = csv.writer(f)
            while True:
                row = self._q.get()
                if row is None:
                    break
                writer.writerow(row)
                f.flush()


# =============================================================================
# UI HELPERS
# =============================================================================

class UI:
    BG_PANEL     = (20,  20,  28)
    NORMAL       = (80,  200, 100)
    WATCHING     = (0,   210, 230)
    ALERT        = (40,  60,  230)
    CROWD        = (30,  140, 255)
    FIGHT        = (0,   140, 255)
    FALL         = (180,  60, 255)
    STATIC       = (160, 100, 60)
    TEXT_PRIMARY = (240, 240, 245)
    TEXT_DIM     = (140, 145, 160)
    FONT_BOLD    = cv2.FONT_HERSHEY_DUPLEX
    FONT_LIGHT   = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.55, radius=6):
        overlay = frame.copy()
        fh, fw  = frame.shape[:2]
        x1, y1  = max(0, x1), max(0, y1)
        x2, y2  = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return
        r = radius
        cv2.rectangle(overlay, (x1+r, y1),   (x2-r, y2),   color, -1)
        cv2.rectangle(overlay, (x1,   y1+r), (x2,   y2-r), color, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    @staticmethod
    def border_rect(frame, x1, y1, x2, y2, color, thickness=2, radius=6):
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1),    max(0, y1)
        x2, y2 = min(fw-1, x2), min(fh-1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        r = radius
        cv2.line(frame, (x1+r,y1),   (x2-r,y1),   color, thickness)
        cv2.line(frame, (x1+r,y2),   (x2-r,y2),   color, thickness)
        cv2.line(frame, (x1,  y1+r), (x1,  y2-r), color, thickness)
        cv2.line(frame, (x2,  y1+r), (x2,  y2-r), color, thickness)
        cv2.ellipse(frame,(x1+r,y1+r),(r,r),180, 0,90,color,thickness)
        cv2.ellipse(frame,(x2-r,y1+r),(r,r),270, 0,90,color,thickness)
        cv2.ellipse(frame,(x1+r,y2-r),(r,r), 90, 0,90,color,thickness)
        cv2.ellipse(frame,(x2-r,y2-r),(r,r),  0, 0,90,color,thickness)

    @staticmethod
    def label_pill(frame, text, x, y, color, scale=0.52, thickness=1):
        font = UI.FONT_LIGHT
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        px, py = 8, 5
        bx1, by1 = x - px,      y - th - py
        bx2, by2 = x + tw + px, y + bl + py
        UI.alpha_rect(frame, bx1, by1, bx2, by2, UI.BG_PANEL, alpha=0.72, radius=4)
        fh = frame.shape[0]
        cv2.line(frame, (bx1, max(0,by1)), (bx1, min(fh-1,by2)), color, 2)
        cv2.putText(frame, text, (x, y), font, scale,
                    UI.TEXT_PRIMARY, thickness, cv2.LINE_AA)

    @staticmethod
    def alert_badge(frame, text, x, y, color, scale=0.58, thickness=2):
        font = UI.FONT_BOLD
        (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
        px, py = 10, 6
        bx1, by1 = x - px,      y - th - py
        bx2, by2 = x + tw + px, y + bl + py
        UI.alpha_rect(frame, bx1, by1, bx2, by2, color, alpha=0.30, radius=5)
        UI.border_rect(frame, bx1, by1, bx2, by2, color, thickness=1, radius=5)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def corner_mark(frame, x1, y1, x2, y2, color, length=14, thickness=2):
        if x2 <= x1 or y2 <= y1:
            return
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1),    max(0, y1)
        x2, y2 = min(fw-1, x2), min(fh-1, y2)
        l = min(length, (x2-x1)//2, (y2-y1)//2)
        for origin, h_pt, v_pt in [
            ((x1,y1),(x1+l,y1),(x1,y1+l)),
            ((x2,y1),(x2-l,y1),(x2,y1+l)),
            ((x1,y2),(x1+l,y2),(x1,y2-l)),
            ((x2,y2),(x2-l,y2),(x2,y2-l)),
        ]:
            cv2.line(frame, origin, h_pt, color, thickness, cv2.LINE_AA)
            cv2.line(frame, origin, v_pt, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_hud(frame, frame_count, fps, active_alerts, device: str = "cpu"):
        fh, fw  = frame.shape[:2]
        now_str = datetime.now().strftime("%H:%M:%S")
        UI.alpha_rect(frame, 10, 10, 320, 66, UI.BG_PANEL, alpha=0.75, radius=6)
        cv2.putText(frame, "UNIFIED SECURITY SYSTEM",
                    (20, 30), UI.FONT_BOLD, 0.45, UI.TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"LIVE  |  {now_str}  |  FRM {frame_count:06d}  |  {device.upper()}",
                    (20, 52), UI.FONT_LIGHT, 0.40, UI.TEXT_PRIMARY, 1, cv2.LINE_AA)
        fps_txt = f"SRC {fps:.0f} FPS"
        (tw, _), _ = cv2.getTextSize(fps_txt, UI.FONT_LIGHT, 0.42, 1)
        cv2.putText(frame, fps_txt, (fw - tw - 14, fh - 12),
                    UI.FONT_LIGHT, 0.42, UI.TEXT_DIM, 1, cv2.LINE_AA)
        if active_alerts:
            seen, unique = set(), []
            for a in active_alerts:
                if a not in seen:
                    seen.add(a); unique.append(a)
            ticker = "  ●  ".join(unique)
            (tw, _), _ = cv2.getTextSize(ticker, UI.FONT_LIGHT, 0.42, 1)
            UI.alpha_rect(frame, 10, fh-36, tw + 26, fh-10,
                          UI.BG_PANEL, alpha=0.70, radius=4)
            cv2.putText(frame, ticker, (16, fh-18),
                        UI.FONT_LIGHT, 0.42, (40, 200, 120), 1, cv2.LINE_AA)


# =============================================================================
# BASE DETECTOR
# =============================================================================

class BaseDetector(ABC):
    """Abstract base — every module inherits shared FPS / threshold helpers."""

    def __init__(self, fps: float, csv_logger: CSVLogger):
        self.fps        = max(fps, 1.0)
        self.logger     = csv_logger
        self.frame_count = 0

    def update_fps(self, fps: float):
        self.fps = max(fps, 1.0)
        self._recalc_thresholds()

    def frames_to_sec(self, frames: int) -> int:
        return int(frames / self.fps)

    def _recalc_thresholds(self):
        """Override in subclasses that have frame-based thresholds."""
        pass

    @abstractmethod
    def compute(self, frame, boxes, keypoints_xy, keypoints_xyn, frame_count: int, **kwargs):
        """Run detection logic. Must return a state dict keyed by track ID."""
        ...

    def cleanup_stale(self, current_ids: list, frame_count: int):
        """Override to prune old track state."""
        pass


# =============================================================================
# MODULE 1 — PANIC / CROWD DETECTOR
# =============================================================================

class PanicCrowdDetector(BaseDetector):

    def __init__(self, fps, csv_logger, roi_polygon, save_dir, W, H):
        super().__init__(fps, csv_logger)
        self.roi_polygon      = roi_polygon
        self.save_dir         = save_dir
        self.W, self.H        = W, H
        self.movement_path    = {}
        self.active_writers   = {}
        self._last_seen_frame = {}

    def _in_zone(self, feet) -> bool:
        return cv2.pointPolygonTest(self.roi_polygon, feet, False) >= 0

    def _detect_crowds(self, current_ids, feet_positions) -> set:
        if len(feet_positions) < CROWD_MIN_COUNT:
            return set()
        coords = np.array(feet_positions, dtype=np.float32)
        dist   = np.sqrt(
            ((coords[:, np.newaxis] - coords[np.newaxis]) ** 2).sum(axis=2))
        adj = (dist < CROWD_DISTANCE).astype(np.uint8)
        crowd_ids, visited = set(), set()
        for i in range(len(current_ids)):
            if i in visited:
                continue
            cluster, stack = [], [i]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(current_ids[node])
                stack.extend(int(x) for x in np.where(adj[node] == 1)[0])
            if len(cluster) >= CROWD_MIN_COUNT:
                crowd_ids.update(cluster)
        return crowd_ids

    def compute(self, frame, boxes, keypoints_xy, keypoints_xyn,
                frame_count: int, active_alerts: list, **kwargs):
        self.frame_count = frame_count
        panic_state, crowd_members = {}, set()

        if boxes is None:
            return panic_state, crowd_members

        zone_ids, zone_feet = [], []
        for box in boxes:
            tid          = int(box[4])
            x1,y1,x2,y2 = map(int, box[:4])
            feet         = ((x1 + x2) // 2, y2)
            self._last_seen_frame[tid] = frame_count
            if self._in_zone(feet):
                zone_ids.append(tid)
                zone_feet.append(feet)
                if tid not in self.movement_path:
                    self.movement_path[tid] = deque(maxlen=20)
                self.movement_path[tid].append(feet)

        crowd_members = self._detect_crowds(zone_ids, zone_feet)
        if crowd_members:
            active_alerts.append(f"CROWD({len(crowd_members)})")

        for box in boxes:
            tid          = int(box[4])
            x1,y1,x2,y2 = map(int, box[:4])
            feet         = ((x1 + x2) // 2, y2)
            if not self._in_zone(feet):
                continue

            speed, is_running = 0.0, False
            path = self.movement_path.get(tid, deque())
            if len(path) > 4:
                speed = float(
                    np.linalg.norm(np.array(path[-1]) - np.array(path[-5])) / 5.0)
                is_running = speed > PANIC_RUNNING_SPEED

            in_crowd = tid in crowd_members
            panic_state[tid] = {
                'is_running':  is_running,
                'is_in_crowd': in_crowd,
                'speed':       speed
            }

            if is_running or in_crowd:
                ev_tag = "RUNNING" if is_running else "CROWD"
                if tid not in self.active_writers:
                    fn = f"ID{tid}_{ev_tag}_{frame_count}.avi"
                    self.active_writers[tid] = cv2.VideoWriter(
                        os.path.join(self.save_dir, fn),
                        cv2.VideoWriter_fourcc(*'XVID'),
                        self.fps, (self.W, self.H))
                self.active_writers[tid].write(frame)
                active_alerts.append(
                    ("RUNNING" if is_running else "IN CROWD") + f" ID{tid}")

        return panic_state, crowd_members

    def cleanup_stale(self, current_ids: list, frame_count: int):
        stale_frames = int(3.0 * self.fps)
        for tid in list(self.movement_path.keys()):
            if tid not in current_ids:
                if frame_count - self._last_seen_frame.get(tid, 0) > stale_frames:
                    if tid in self.active_writers:
                        self.active_writers[tid].release()
                        del self.active_writers[tid]
                    self.movement_path.pop(tid, None)
                    self._last_seen_frame.pop(tid, None)

    def release(self):
        for w in self.active_writers.values():
            w.release()


# =============================================================================
# MODULE 2 — FALL / FIGHT DETECTOR
# =============================================================================

class FallFightDetector(BaseDetector):

    def __init__(self, fps, csv_logger, save_dir):
        super().__init__(fps, csv_logger)
        self._recalc_thresholds()
        self.save_dir         = save_dir
        self.pose_history     = {}
        self.pose_last_frame  = {}
        self.frame_buffer     = deque(maxlen=self.BUFFER_FRAMES)
        self.is_recording     = False
        self.video_writer     = None
        self.record_counter   = 0
        self._cooldown        = {}

    def _recalc_thresholds(self):
        self.BUFFER_FRAMES      = int(BUFFER_SECONDS * self.fps)
        self.COOLDOWN_FRAMES    = int(ALERT_COOLDOWN_SECONDS * self.fps)

    def _in_cooldown(self, tid: int, event: str) -> bool:
        return (self.frame_count
                - self._cooldown.get(tid, {}).get(event, -99999)
                < self.COOLDOWN_FRAMES)

    def _set_cooldown(self, tid: int, event: str):
        self._cooldown.setdefault(tid, {})[event] = self.frame_count

    def _get_smoothed_kpts(self, tid, new_kpts):
        self.pose_history[tid]['kpts_buffer'].append(new_kpts)
        return np.mean(self.pose_history[tid]['kpts_buffer'], axis=0)

    def _start_recording(self, frame_shape, tid, event):
        if self.is_recording:
            return
        # Only start a clip if we already have at least 1 second of buffered
        # frames — this eliminates sub-second spam clips from transient detections.
        if len(self.frame_buffer) < int(self.fps):
            return
        fh, fw = frame_shape[:2]
        fname  = os.path.join(
            self.save_dir,
            f"{event}_{tid}_{self.frame_count}.mp4")
        self.video_writer = cv2.VideoWriter(
            fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (fw, fh))
        for bf in self.frame_buffer:
            self.video_writer.write(bf)
        self.is_recording   = True
        self.record_counter = self.BUFFER_FRAMES
        print(f"[INFO] Recording {event} | ID {tid} | frame {self.frame_count}")

    def _log_and_record(self, tid, event, frame_shape):
        if self._in_cooldown(tid, event):
            return
        self._set_cooldown(tid, event)
        self.logger.log(tid, event)
        self._start_recording(frame_shape, tid, event)

    def _check_fall_keypoints(self, avg_kpts, prev_kpts, torso_size) -> bool:
        """
        Keypoint-confidence-gated fall check.
        Requires both torso tilt > 45° AND downward hip velocity.
        Returns False immediately if keypoints lack a confidence column or are too small.
        """
        # Need at least 13 keypoints and a confidence column (dim >= 3)
        if avg_kpts.shape[0] < 13 or avg_kpts.ndim < 2 or avg_kpts.shape[-1] < 3:
            return False

        # --- Confidence gate (index 2 = confidence in keypoints_xy format) ---
        critical = [5, 6, 11, 12]   # left/right shoulder, left/right hip
        if any(avg_kpts[j][2] < 0.3 for j in critical):
            return False

        hip_y_now  = (avg_kpts[11][1]  + avg_kpts[12][1])  / 2.0
        hip_y_prev = (prev_kpts[11][1] + prev_kpts[12][1]) / 2.0
        hip_drop   = (hip_y_now - hip_y_prev) / (torso_size + 1e-6)

        mid_hip      = (avg_kpts[11][:2] + avg_kpts[12][:2]) / 2.0
        mid_shoulder = (avg_kpts[5][:2]  + avg_kpts[6][:2])  / 2.0
        torso_vec    = mid_shoulder - mid_hip
        if np.linalg.norm(torso_vec) > 1e-6:
            angle_deg = abs(np.degrees(
                np.arctan2(torso_vec[0], torso_vec[1] + 1e-9)))
        else:
            angle_deg = 90.0

        torso_horizontal = angle_deg > 45.0
        hip_falling      = hip_drop > FALL_HIP_DROP_THRESHOLD

        return torso_horizontal and hip_falling

    def compute(self, frame, boxes, keypoints_xy, keypoints_xyn,
                frame_count: int, active_alerts: list, all_box_count: int = 0, **kwargs):
        self.frame_count = frame_count

        if self.is_recording:
            self.video_writer.write(frame)
            self.record_counter -= 1
            if self.record_counter <= 0:
                self.is_recording = False
                self.video_writer.release()
                self.video_writer = None
        else:
            self.frame_buffer.append(frame.copy())

        fall_fight_data = {}

        # keypoints_xy  → shape [N, 17, 3]  pixel x, y, confidence  ← required here
        # keypoints_xyn → shape [N, 17, 2]  normalised x, y only (no confidence col)
        # _check_fall_keypoints reads index [j][2] for confidence, so we must use
        # keypoints_xy throughout this detector.
        if boxes is None or keypoints_xy is None:
            return fall_fight_data

        for i, box in enumerate(boxes):
            x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            tid          = int(box[4])
            if i >= len(keypoints_xy):
                continue
            raw_kpts     = keypoints_xy[i]   # shape [17, 3] — x, y, confidence

            if tid not in self.pose_history:
                self.pose_history[tid] = {
                    'kpts_buffer':    deque(maxlen=SMOOTHING_WINDOW),
                    'prev_avg_kpts':  None,
                    'total_movement': 0.0,
                    'is_static':      True
                }

            self.pose_last_frame[tid] = frame_count
            state    = self.pose_history[tid]
            avg_kpts = self._get_smoothed_kpts(tid, raw_kpts)

            if state['prev_avg_kpts'] is not None:
                state['total_movement'] += float(np.linalg.norm(
                    avg_kpts[0] - state['prev_avg_kpts'][0]))
                if state['total_movement'] > MOVE_THRESHOLD:
                    state['is_static'] = False

            if state['is_static']:
                state['prev_avg_kpts'] = avg_kpts
                fall_fight_data[tid] = {'is_static': True,
                                        'has_fall': False, 'has_fight': False}
                continue

            torso_size = float(np.linalg.norm(avg_kpts[5] - avg_kpts[11])) + 1e-6
            prev_k     = state['prev_avg_kpts']
            has_fight  = False
            has_fall   = False

            if prev_k is not None:
                # --- Fight detection (visual alert only — no evidence clip saved) ---
                lw = float(np.linalg.norm(avg_kpts[9]  - prev_k[9]))  / torso_size
                rw = float(np.linalg.norm(avg_kpts[10] - prev_k[10])) / torso_size
                hd = float(np.linalg.norm(avg_kpts[0]  - prev_k[0]))  / torso_size
                if ((lw > FIGHT_WRIST_THRESHOLD or rw > FIGHT_WRIST_THRESHOLD)
                        and hd > FIGHT_HEAD_THRESHOLD
                        and all_box_count > 1):
                    has_fight = True
                    # Log to CSV only — no clip recording for fight events
                    if not self._in_cooldown(tid, "FIGHT"):
                        self._set_cooldown(tid, "FIGHT")
                        self.logger.log(tid, "FIGHT")
                    active_alerts.append(f"FIGHT ID{tid}")

                # --- Fall detection: keypoint-based + aspect ratio ---
                aspect_fallen   = (y2 - y1) < (x2 - x1) * ASPECT_RATIO_FALL_THRESHOLD
                keypoint_fallen = self._check_fall_keypoints(avg_kpts, prev_k, torso_size)
                if aspect_fallen and keypoint_fallen:
                    has_fall = True
                    self._log_and_record(tid, "FALL", frame.shape)
                    active_alerts.append(f"FALL ID{tid}")

            fall_fight_data[tid] = {'is_static': False,
                                    'has_fall': has_fall, 'has_fight': has_fight}
            state['prev_avg_kpts'] = avg_kpts

        return fall_fight_data

    def cleanup_stale(self, current_ids: list, frame_count: int):
        stale_frames = int(3.0 * self.fps)
        for tid in list(self.pose_history.keys()):
            if tid not in current_ids:
                if frame_count - self.pose_last_frame.get(tid, 0) > stale_frames:
                    self.pose_history.pop(tid, None)
                    self.pose_last_frame.pop(tid, None)
                    self._cooldown.pop(tid, None)

    def release(self):
        if self.video_writer:
            self.video_writer.release()


# =============================================================================
# MODULE 3 — LOITERING DETECTOR
# =============================================================================

class LoiteringDetector(BaseDetector):

    def __init__(self, fps, csv_logger, roi_polygon, save_dir, W, H):
        super().__init__(fps, csv_logger)
        self._recalc_thresholds()
        self.roi_polygon        = roi_polygon
        self.save_dir           = save_dir
        self.W, self.H          = W, H
        self.zone_entry_frame   = {}
        self.in_zone_prev       = {}
        self.loiter_last_frame  = {}
        self.movement_path      = {}
        self.active_writers     = {}
        self.behavior_history   = {}
        self.confirmed_behavior = {}

    def _recalc_thresholds(self):
        self.LIMIT_STATIONARY_F = int(LIMIT_STATIONARY_S * self.fps)
        self.LIMIT_PACING_F     = int(LIMIT_PACING_S     * self.fps)
        self.LIMIT_CROUCHING_F  = int(LIMIT_CROUCHING_S  * self.fps)
        self.LIMIT_RUNNING_F    = int(LIMIT_RUNNING_S     * self.fps)

    def _in_zone(self, feet) -> bool:
        return cv2.pointPolygonTest(self.roi_polygon, feet, False) >= 0

    def _get_gaze(self, kpts) -> str:
        nose, l_ear, r_ear = kpts[0], kpts[3], kpts[4]
        if nose[1] == 0 or l_ear[1] == 0 or r_ear[1] == 0:
            return "UNK"
        ratio = (np.linalg.norm(nose[:2] - l_ear[:2]) /
                 (np.linalg.norm(nose[:2] - r_ear[:2]) + 1e-6))
        if   0.8 < ratio < 1.2: return "FWD"
        elif ratio <= 0.8:      return "L"
        else:                   return "R"

    def _get_confirmed_behavior(self, tid, raw_behavior) -> str:
        if tid not in self.behavior_history:
            self.behavior_history[tid]   = deque(maxlen=BEHAVIOR_CONFIRM_FRAMES)
            self.confirmed_behavior[tid] = raw_behavior
        self.behavior_history[tid].append(raw_behavior)
        h = self.behavior_history[tid]
        if (len(h) == BEHAVIOR_CONFIRM_FRAMES
                and h.count(raw_behavior) == BEHAVIOR_CONFIRM_FRAMES):
            self.confirmed_behavior[tid] = raw_behavior
        return self.confirmed_behavior[tid]

    def _path_arc_length(self, path) -> float:
        pts = list(path)
        if len(pts) < 2:
            return 0.0
        return float(sum(
            np.linalg.norm(np.array(pts[j]) - np.array(pts[j-1]))
            for j in range(1, len(pts))
        ))

    def compute(self, frame, boxes, keypoints_xy,
                frame_count: int, active_alerts: list, **kwargs):
        self.frame_count = frame_count
        loiter_data = {}

        if boxes is None or keypoints_xy is None:
            return loiter_data

        for i, box in enumerate(boxes):
            x1,y1,x2,y2,tid = map(int, box[:5])
            feet    = ((x1 + x2) // 2, y2)
            in_zone = self._in_zone(feet)

            was_in_zone = self.in_zone_prev.get(tid, False)
            if in_zone and not was_in_zone:
                self.zone_entry_frame[tid] = frame_count
                self.movement_path[tid]    = deque(maxlen=64)
            self.in_zone_prev[tid]      = in_zone
            self.loiter_last_frame[tid] = frame_count

            if tid not in self.movement_path:
                self.movement_path[tid] = deque(maxlen=64)
            self.movement_path[tid].append(feet)

            kpts = keypoints_xy[i]

            speed = 0.0
            if len(self.movement_path[tid]) > 2:
                speed = float(np.linalg.norm(
                    np.array(self.movement_path[tid][-1]) -
                    np.array(self.movement_path[tid][-2])))

            if not in_zone:
                loiter_data[tid] = {
                    'in_zone': False, 'is_alert': False,
                    'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                    'kpts': kpts, 'path_pts': []
                }
                continue

            dwell_frames = frame_count - self.zone_entry_frame.get(tid, frame_count)
            dwell_sec    = self.frames_to_sec(dwell_frames)
            path_pts     = list(self.movement_path[tid])

            raw_behavior = "Stationary"
            if speed > RUNNING_SPEED_THRESHOLD:
                raw_behavior = "RUNNING"
            elif len(self.movement_path[tid]) >= 30:
                arc = self._path_arc_length(self.movement_path[tid])
                if arc > MOVEMENT_THRESHOLD * 2:
                    raw_behavior = "Pacing"

            behavior = self._get_confirmed_behavior(tid, raw_behavior)

            posture = "Upright"
            if kpts[11][1] > 0 and kpts[15][1] > 0:
                if (kpts[15][1] - kpts[11][1]) < (y2 - y1) * 0.35:
                    posture = "Crouching"

            gaze = self._get_gaze(kpts)

            if   behavior == "RUNNING":   limit_f = self.LIMIT_RUNNING_F
            elif posture  == "Crouching": limit_f = self.LIMIT_CROUCHING_F
            elif behavior == "Pacing":    limit_f = self.LIMIT_PACING_F
            else:                         limit_f = self.LIMIT_STATIONARY_F

            is_alert = dwell_frames > limit_f

            if is_alert:
                if tid not in self.active_writers:
                    # Only open a clip writer if the person has been alerting for
                    # more than 1 second — avoids clips from transient detections.
                    if dwell_frames < int(self.fps) + limit_f:
                        # Still within the first second of the alert — skip for now
                        active_alerts.append(f"{behavior} ID{tid}")
                        loiter_data[tid] = {
                            'in_zone':   True,
                            'is_alert':  is_alert,
                            'behavior':  behavior,
                            'posture':   posture,
                            'gaze':      gaze,
                            'speed':     speed,
                            'dwell_sec': dwell_sec,
                            'kpts':      kpts,
                            'path_pts':  path_pts,
                            'x1':x1,'y1':y1,'x2':x2,'y2':y2
                        }
                        continue
                    fn = f"ID{tid}_{behavior}_{frame_count}.avi"
                    self.active_writers[tid] = cv2.VideoWriter(
                        os.path.join(self.save_dir, fn),
                        cv2.VideoWriter_fourcc(*'XVID'),
                        self.fps, (self.W, self.H))
                    print(f"[INFO] REC loiter | ID {tid} | "
                          f"{behavior} | frame {frame_count}")
                # Write every frame while the alert is active (continuous clip)
                self.active_writers[tid].write(frame)
                active_alerts.append(f"{behavior} ID{tid}")

            loiter_data[tid] = {
                'in_zone':   True,
                'is_alert':  is_alert,
                'behavior':  behavior,
                'posture':   posture,
                'gaze':      gaze,
                'speed':     speed,
                'dwell_sec': dwell_sec,
                'kpts':      kpts,
                'path_pts':  path_pts,
                'x1':x1,'y1':y1,'x2':x2,'y2':y2
            }

        return loiter_data

    def cleanup_stale(self, current_ids: list, frame_count: int):
        stale_frames = int(3.0 * self.fps)
        for tid in list(self.zone_entry_frame.keys()):
            if tid not in current_ids:
                if frame_count - self.loiter_last_frame.get(tid, 0) > stale_frames:
                    if tid in self.active_writers:
                        self.active_writers[tid].release()
                        del self.active_writers[tid]
                    self.zone_entry_frame.pop(tid, None)
                    self.in_zone_prev.pop(tid, None)
                    self.loiter_last_frame.pop(tid, None)
                    self.movement_path.pop(tid, None)
                    self.behavior_history.pop(tid, None)
                    self.confirmed_behavior.pop(tid, None)

    def release(self):
        for w in self.active_writers.values():
            w.release()


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class DetectorOrchestrator:

    def __init__(self, fps: float):
        self.fps         = max(fps, 1.0)
        self.frame_count = 0

        # ── Device selection: CUDA → MPS → CPU ──────────────────────────────
        self.device = select_device()
        # half-precision only works on CUDA; MPS and CPU must use float32
        self.use_half = (self.device == "cuda")
        print(f"[INFO] Device          : {self.device.upper()}")
        print(f"[INFO] Half precision  : {self.use_half}")
        # ────────────────────────────────────────────────────────────────────

        self.model  = YOLO(MODEL_PATH)
        self.logger = CSVLogger("security_logs.csv")

        self.zone_points = []
        self.roi_polygon = None

        self.panic_det   = None
        self.fall_det    = None
        self.loiter_det  = None

    # ─────────────────────────────────────────────────────────────────────────
    # ZONE SELECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.zone_points.append([x, y])

    def select_zone(self, frame):
        cv2.namedWindow("Setup")
        cv2.setMouseCallback("Setup", self._mouse_callback)
        print("Draw Zone. Enter to start. 'C' to clear.")
        while True:
            img = frame.copy()
            if len(self.zone_points) > 1:
                cv2.polylines(img, [np.array(self.zone_points)], True, (0,255,255), 2)
            cv2.imshow("Setup", img)
            key = cv2.waitKey(1)
            if key == 13 and len(self.zone_points) >= 3:
                break
            if key == ord('c'):
                self.zone_points = []
            # Exit gracefully if the user closes the window
            if cv2.getWindowProperty("Setup", cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Zone selection cancelled — window closed.")
        cv2.destroyWindow("Setup")
        return np.array(self.zone_points, np.int32)

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _blur_face(frame, kpts):
        face_pts  = kpts[0:5]
        valid_pts = face_pts[face_pts[:, 0] > 0]
        if len(valid_pts) > 2:
            x_min, y_min = np.min(valid_pts, axis=0)[:2]
            x_max, y_max = np.max(valid_pts, axis=0)[:2]
            pad = 20
            x1 = max(0, int(x_min - pad));  y1 = max(0, int(y_min - pad))
            x2 = min(frame.shape[1], int(x_max + pad))
            y2 = min(frame.shape[0], int(y_max + pad))
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                    frame[y1:y2, x1:x2], (51, 51), 30)
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_all(self, frame, boxes, keypoints_xy,
                  panic_state, crowd_members,
                  fall_fight_data, loiter_data):

        if len(crowd_members) > 0:
            UI.alert_badge(frame,
                f"  CROWD ALERT  -  {len(crowd_members)} PERSONS DETECTED  ",
                14, 90, UI.CROWD, scale=0.55, thickness=2)

        if boxes is None:
            return

        for i, box in enumerate(boxes):
            tid          = int(box[4])
            x1,y1,x2,y2 = map(int, box[:4])

            p_state  = panic_state.get(tid, {})
            ff_state = fall_fight_data.get(tid, {})
            lo_state = loiter_data.get(tid, {})

            is_running_panic = p_state.get('is_running', False)
            in_crowd         = p_state.get('is_in_crowd', False)
            has_fall         = ff_state.get('has_fall', False)
            has_fight        = ff_state.get('has_fight', False)
            is_static        = ff_state.get('is_static', False)
            in_zone          = lo_state.get('in_zone', False)
            loiter_alert     = lo_state.get('is_alert', False)

            any_alert = (is_running_panic or in_crowd or
                         has_fall or has_fight or loiter_alert)

            kpts = (keypoints_xy[i]
                    if keypoints_xy is not None and i < len(keypoints_xy)
                    else None)

            if in_zone and lo_state:
                path_pts = lo_state.get('path_pts', [])
                n = len(path_pts)
                if n > 2:
                    for j in range(1, n):
                        a     = j / n
                        col   = (int(40*a), int(40*a), int(50 + 180*a))
                        thick = max(1, int(np.sqrt(64 / float(j+1)) * 1.5))
                        cv2.line(frame, path_pts[j-1], path_pts[j],
                                 col, thick, cv2.LINE_AA)

            if BLUR_FACES and not any_alert and kpts is not None:
                frame = self._blur_face(frame, kpts)

            if is_static:
                UI.corner_mark(frame, x1,y1,x2,y2, UI.STATIC, length=10, thickness=1)
                UI.label_pill(frame, "STATIC", x1, y1+20, UI.STATIC, scale=0.42)
                continue

            if not in_zone:
                UI.corner_mark(frame, x1,y1,x2,y2, UI.NORMAL, length=10, thickness=1)
                UI.label_pill(frame, f"ID {tid}  PASSING",
                              x1, y1-8, UI.NORMAL, scale=0.42)
                continue

            if has_fall:           color = UI.FALL
            elif has_fight:        color = UI.FIGHT
            elif is_running_panic: color = UI.ALERT
            elif in_crowd:         color = UI.CROWD
            elif loiter_alert:     color = UI.ALERT
            else:                  color = UI.WATCHING

            thickness = 2 if any_alert else 1
            length    = 16 if any_alert else 12
            UI.corner_mark(frame, x1,y1,x2,y2, color, length=length, thickness=thickness)
            if any_alert:
                cv2.line(frame, (x1,y1), (x2,y1), color, 2, cv2.LINE_AA)

            badge_y = y1 - 10
            if has_fall:
                UI.alert_badge(frame, "FALL DETECTED", x1, badge_y, UI.FALL)
                badge_y -= 28
            if has_fight:
                UI.alert_badge(frame, "FIGHT / AGGRESSION", x1, badge_y, UI.FIGHT)
                badge_y -= 28
            if is_running_panic:
                UI.alert_badge(frame, "ALERT  RUNNING", x1, badge_y, UI.ALERT)
                badge_y -= 28
            elif in_crowd:
                UI.alert_badge(frame, "ALERT  IN CROWD", x1, badge_y, UI.CROWD)
                badge_y -= 28
            if loiter_alert and lo_state:
                behavior  = lo_state.get('behavior', '')
                dwell_sec = lo_state.get('dwell_sec', 0)
                UI.alert_badge(frame,
                               f"ALERT  {behavior}  {dwell_sec}s",
                               x1, badge_y, UI.ALERT, scale=0.52, thickness=2)

            if lo_state:
                posture   = lo_state.get('posture', '')
                gaze      = lo_state.get('gaze', '')
                speed     = lo_state.get('speed', 0)
                dwell_sec = lo_state.get('dwell_sec', 0)
                behavior  = lo_state.get('behavior', '')
                if any_alert:
                    UI.label_pill(frame,
                                  f"ID {tid}  |  {posture}  |  "
                                  f"GAZE {gaze}  |  SPD {int(speed)}",
                                  x1, y2+18, UI.TEXT_DIM, scale=0.40)
                else:
                    UI.label_pill(frame,
                                  f"{behavior}  {dwell_sec}s",
                                  x1, y1-10, color, scale=0.48)
                    UI.label_pill(frame,
                                  f"ID {tid}  |  GAZE {gaze}  |  SPD {int(speed)}",
                                  x1, y2+16, UI.TEXT_DIM, scale=0.38)

            if loiter_alert and tid in self.loiter_det.active_writers:
                self.loiter_det.active_writers[tid].write(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def process(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
            return

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        self.fps = max(src_fps, 1.0)
        W, H     = 1280, 720

        ret, first = cap.read()
        if not ret:
            print("[ERROR] Cannot read first frame.")
            cap.release()
            return

        self.roi_polygon = self.select_zone(cv2.resize(first, (W, H)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.panic_det  = PanicCrowdDetector(self.fps, self.logger,
                                              self.roi_polygon, SAVE_DIR, W, H)
        self.fall_det   = FallFightDetector(self.fps, self.logger, FALL_SAVE_DIR)
        self.loiter_det = LoiteringDetector(self.fps, self.logger,
                                             self.roi_polygon, SAVE_DIR, W, H)

        print(f"[INFO] Source FPS        : {self.fps:.1f}")
        print(f"[INFO] Confirm window    : {BEHAVIOR_CONFIRM_FRAMES} frames")
        print(f"[INFO] Stationary limit  : {LIMIT_STATIONARY_S}s")
        print(f"[INFO] Pacing limit      : {LIMIT_PACING_S}s")
        print(f"[INFO] Crouching limit   : {LIMIT_CROUCHING_S}s")
        print(f"[INFO] Alert cooldown    : {ALERT_COOLDOWN_SECONDS}s")
        print(f"[INFO] Pre-event buffer  : {BUFFER_SECONDS}s")
        print("[INFO] Press 'Q' to quit.\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                active_alerts    = []

                frame = cv2.resize(frame, (W, H))

                # Dashed ROI border
                pts = self.roi_polygon.reshape(-1, 2)
                n   = len(pts)
                for k in range(n):
                    p1    = tuple(pts[k])
                    p2    = tuple(pts[(k+1) % n])
                    total = int(np.linalg.norm(np.array(p2) - np.array(p1)))
                    if total == 0:
                        continue
                    dash, gap = 12, 6
                    for s in range(0, total, dash + gap):
                        t1 = s / total
                        t2 = min((s + dash) / total, 1.0)
                        a1 = (int(p1[0] + t1*(p2[0]-p1[0])),
                              int(p1[1] + t1*(p2[1]-p1[1])))
                        a2 = (int(p1[0] + t2*(p2[0]-p1[0])),
                              int(p1[1] + t2*(p2[1]-p1[1])))
                        cv2.line(frame, a1, a2, (100, 110, 130), 1, cv2.LINE_AA)
                if n > 0:
                    cx = int(pts[:, 0].mean())
                    cy = int(pts[:, 1].mean())
                    UI.label_pill(frame, "MONITORING ZONE",
                                  cx-60, cy, (100, 110, 130), scale=0.38)

                # ── YOLO tracking — uses self.device and self.use_half ───────
                results = self.model.track(
                    frame, persist=True, conf=CONF_THRESHOLD,
                    verbose=False,
                    device=self.device,
                    half=self.use_half,
                    iou=0.5)
                # ─────────────────────────────────────────────────────────────

                boxes         = None
                keypoints_xy  = None
                keypoints_xyn = None
                current_ids   = []

                if (results[0].boxes is not None
                        and results[0].boxes.id is not None):
                    boxes         = results[0].boxes.data.cpu().numpy()
                    keypoints_xy  = results[0].keypoints.data.cpu().numpy()
                    keypoints_xyn = results[0].keypoints.xyn.cpu().numpy()
                    current_ids   = [int(b[4]) for b in boxes]

                panic_state, crowd_members = self.panic_det.compute(
                    frame, boxes, keypoints_xy, keypoints_xyn,
                    self.frame_count, active_alerts=active_alerts)

                fall_fight_data = self.fall_det.compute(
                    frame, boxes, keypoints_xy, keypoints_xyn,
                    self.frame_count, active_alerts=active_alerts,
                    all_box_count=len(current_ids))

                loiter_data = self.loiter_det.compute(
                    frame, boxes, keypoints_xy,
                    self.frame_count, active_alerts=active_alerts)

                for mod in (self.panic_det, self.fall_det, self.loiter_det):
                    mod.cleanup_stale(current_ids, self.frame_count)

                self._draw_all(frame, boxes, keypoints_xy,
                               panic_state, crowd_members,
                               fall_fight_data, loiter_data)

                # Pass device name so HUD can display it
                UI.draw_hud(frame, self.frame_count, self.fps,
                            active_alerts, device=self.device)

                cv2.imshow("Unified Security System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.panic_det.release()
            self.fall_det.release()
            self.loiter_det.release()
            self.logger.shutdown()
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n[INFO] Shutdown complete.")
            print(f"[INFO] Frames processed : {self.frame_count}")
            print(f"[INFO] Event log        : security_logs.csv")
            print(f"[INFO] Clips saved to   : {SAVE_DIR}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    os.makedirs(SAVE_DIR,      exist_ok=True)
    os.makedirs(FALL_SAVE_DIR, exist_ok=True)

    _cap = cv2.VideoCapture(VIDEO_PATH)
    if not _cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    else:
        _fps = _cap.get(cv2.CAP_PROP_FPS) or 30
        _cap.release()
        DetectorOrchestrator(_fps).process()