import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont, filedialog
import cv2
import threading
import numpy as np
import os
import time
from PIL import Image, ImageTk
from reid import ReIDEngine, AnalyticsEngine # Ensure AnalyticsEngine is in reid.py

# ==================== SETTINGS ====================
MODEL_PATH = 'model/osnet_x0_75_imagenet.pth'
VIDEO_1 = "./videos/cam1.1.mp4"
VIDEO_2 = "./videos/cam6.1.mp4"
MATCHES_DIR = "matches"
ANALYTICS_DIR = "analytics_output"
# ==================================================

# ==================== THEME ====================
BG_DARK      = "#0a0e1a"
BG_PANEL     = "#0f1628"
BG_CARD      = "#141c30"
BG_SURFACE   = "#1a2240"
ACCENT_BLUE  = "#00a8ff"
ACCENT_CYAN  = "#00e5ff"
ACCENT_GREEN = "#00ff88"
ACCENT_RED   = "#ff4757"
ACCENT_AMBER = "#ffa502"
TEXT_PRIMARY = "#e8edf5"
TEXT_SECONDARY = "#7a8fa8"
TEXT_DIM     = "#3d4f6b"
BORDER       = "#1e2d4a"
BORDER_BRIGHT= "#2a3f5f"
# ===============================================

class SentinelVision(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SSF VISION  ·  AI Surveillance Platform")
        self.geometry("1400x900")
        self.minsize(1200, 750)
        self.configure(bg=BG_DARK)

        self.zone_page

        # Engines
        self.engine = ReIDEngine(MODEL_PATH)
        self.analytics_engine = AnalyticsEngine() # New Analytics Engine

        # Global State
        self.is_running = False
        self.current_frame_size = (640, 480)
        
        # Re-ID State
        self.selected_tid = None
        self.latest_detections = []
        self.match_queue = []
        self.match_results = []
        self.current_result_idx = 0
        self.search_complete = False

        # Directory Setup
        os.makedirs(ANALYTICS_DIR, exist_ok=True)

        self._build_fonts()
        self._build_layout()
        self._start_clock()
        self.show_page("Home")

    def _build_fonts(self):
        self.font_title  = ("Courier New", 18, "bold")
        self.font_head   = ("Courier New", 13, "bold")
        self.font_sub     = ("Courier New", 10)
        self.font_mono   = ("Courier New", 9)
        self.font_huge   = ("Courier New", 32, "bold")
        self.font_btn    = ("Courier New", 10, "bold")

    def _start_clock(self):
        self._update_clock()

    def _update_clock(self):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        if hasattr(self, 'clock_lbl'):
            self.clock_lbl.config(text=now)
        self.after(1000, self._update_clock)

    def _build_layout(self):
        topbar = tk.Frame(self, bg=BG_PANEL, height=52)
        topbar.pack(side="top", fill="x")
        topbar.pack_propagate(False)

        logo_frame = tk.Frame(topbar, bg=BG_PANEL)
        logo_frame.pack(side="left", padx=20)
        tk.Label(logo_frame, text="⬡", font=("Courier New", 22, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side="left")
        tk.Label(logo_frame, text=" SSF", font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_PANEL).pack(side="left")
        tk.Label(logo_frame, text=" VISION", font=("Courier New", 16), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side="left")

        right = tk.Frame(topbar, bg=BG_PANEL)
        right.pack(side="right", padx=20)
        tk.Label(right, text="● LIVE", font=("Courier New", 9, "bold"), fg=ACCENT_GREEN, bg=BG_PANEL).pack(side="right", padx=10)
        self.clock_lbl = tk.Label(right, font=("Courier New", 10), fg=TEXT_SECONDARY, bg=BG_PANEL, text="")
        self.clock_lbl.pack(side="right", padx=10)

        tk.Frame(self, bg=ACCENT_CYAN, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(body, bg=BG_PANEL, width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        self.content = tk.Frame(body, bg=BG_DARK)
        self.content.pack(side="right", fill="both", expand=True)

        self.home_page      = tk.Frame(self.content, bg=BG_DARK)
        self.reid_page      = tk.Frame(self.content, bg=BG_DARK)
        self.analytics_page = tk.Frame(self.content, bg=BG_DARK) # New Page
        self.results_page   = tk.Frame(self.content, bg=BG_DARK)

        self._build_home_page()
        self._build_reid_page()
        self._build_analytics_page() # New UI Builder
        self._build_results_page()

    def _build_sidebar(self):
        tk.Frame(self.sidebar, bg=BG_PANEL, height=20).pack()
        tk.Label(self.sidebar, text="NAVIGATION", font=("Courier New", 7, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(10, 4))

        nav_items = [
            ("Home",      "⌂", "Dashboard"),
            ("ReID",      "◉", "Person Re-ID"),
            ("Analytics", "📊", "People Analytics"),
            ("Zones",     "🚫", "Restricted Zones"), # Add this
            ("Results",   "▦", "Match Results"),
        ]
        self.nav_btns = {}
        for key, icon, label in nav_items:
            btn = self._nav_button(self.sidebar, icon, label, command=lambda k=key: self.show_page(k))
            self.nav_btns[key] = btn

        # --- Session Status Section ---
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=15, pady=15)
        tk.Label(self.sidebar, text="SESSION", font=("Courier New", 7, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(0, 8))
        self.status_dot = tk.Label(self.sidebar, text="● IDLE", font=("Courier New", 9, "bold"), fg=TEXT_DIM, bg=BG_PANEL)
        self.status_dot.pack(anchor="w", padx=18)
        self.sidebar_status = tk.Label(self.sidebar, text="No active operation", font=("Courier New", 8), fg=TEXT_SECONDARY, bg=BG_PANEL, wraplength=180, justify="left")
        self.sidebar_status.pack(anchor="w", padx=18, pady=(4, 0))

        # --- MATCH COUNTER SECTION (FIXES THE ERROR) ---
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=15, pady=15)
        tk.Label(self.sidebar, text="MATCHES FOUND", font=("Courier New", 7, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(0, 4))
        
        self.match_count_lbl = tk.Label(self.sidebar, text="0", font=("Courier New", 28, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL)
        self.match_count_lbl.pack(anchor="w", padx=18)

        # --- View Results Button ---
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=15, pady=15)
        self.view_results_btn = tk.Button(self.sidebar, text="▶  VIEW RESULTS", font=self.font_btn, bg=ACCENT_CYAN, fg=BG_DARK, relief="flat", pady=8, cursor="hand2", command=lambda: self.show_page("Results"))
        # We don't pack it yet; it will be shown by _on_search_complete when matches exist

    def _nav_button(self, parent, icon, label, command):
        frame = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        frame.pack(fill="x", padx=10, pady=2)
        icon_l = tk.Label(frame, text=icon, font=("Courier New", 12), fg=TEXT_SECONDARY, bg=BG_PANEL, width=3)
        icon_l.pack(side="left")
        text_l = tk.Label(frame, text=label, font=("Courier New", 11), fg=TEXT_SECONDARY, bg=BG_PANEL, anchor="w")
        text_l.pack(side="left", fill="x", expand=True, pady=8)
        
        for w in (frame, icon_l, text_l):
            w.bind("<Button-1>", lambda e, c=command: c())
        frame._icon, frame._text = icon_l, text_l
        return frame

    def _set_active_nav(self, key):
        for k, frame in self.nav_btns.items():
            active = (k == key)
            bg = BG_SURFACE if active else BG_PANEL
            frame.config(bg=bg)
            frame._icon.config(bg=bg, fg=ACCENT_CYAN if active else TEXT_SECONDARY)
            frame._text.config(bg=bg, fg=TEXT_PRIMARY if active else TEXT_SECONDARY)

    # ------------------------------------------------------------------ #
    #  ANALYTICS PAGE UI                                                 #
    # ------------------------------------------------------------------ #
    def _build_analytics_page(self):
        p = self.analytics_page
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="PEOPLE COUNTING & HEATMAPS", font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        main_cols = tk.Frame(p, bg=BG_DARK)
        main_cols.pack(fill="both", expand=True, padx=25, pady=15)

        # --- RIGHT SIDEBAR (Control Panel) ---
        # Pack FIRST
        right_col = tk.Frame(main_cols, bg=BG_DARK, width=320)
        right_col.pack(side="right", fill="y", padx=(20, 0))
        right_col.pack_propagate(False)

        # (Existing widgets inside right_col)
        tk.Label(right_col, text="DATA SOURCE", font=self.font_head, fg=ACCENT_CYAN, bg=BG_DARK).pack(pady=(0, 10))
        tk.Button(right_col, text="📁 LOAD VIDEO FILE", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY, pady=10, command=self.select_analytics_video).pack(fill="x", pady=5)
        tk.Button(right_col, text="📷 START WEBCAM", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY, pady=10, command=lambda: self.start_analytics(0)).pack(fill="x", pady=5)

        tk.Frame(right_col, bg=BORDER, height=1).pack(fill="x", pady=20)
        self.ana_count_lbl = tk.Label(right_col, text="CURRENT: 0", font=("Courier New", 14, "bold"), fg=ACCENT_GREEN, bg=BG_PANEL, pady=15)
        self.ana_count_lbl.pack(fill="x", pady=5)
        self.ana_peak_lbl = tk.Label(right_col, text="PEAK: 0", font=("Courier New", 12), fg=TEXT_SECONDARY, bg=BG_PANEL, pady=15)
        self.ana_peak_lbl.pack(fill="x", pady=5)

        self.stop_ana_btn = tk.Button(right_col, text="■ STOP & SAVE REPORT", font=self.font_btn, bg=ACCENT_RED, fg=TEXT_PRIMARY, pady=15, state="disabled", command=self.stop_analytics)
        self.stop_ana_btn.pack(side="bottom", fill="x", pady=20)

        # --- LEFT AREA (Video Feed) ---
        # Pack SECOND
        left_col = tk.Frame(main_cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        ana_feed_wrap = tk.Frame(left_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        ana_feed_wrap.pack(fill="both", expand=True)
        self.ana_vid_label = tk.Label(ana_feed_wrap, bg="#040810", text="[ SELECT INPUT SOURCE ]", font=self.font_head, fg=TEXT_DIM)
        self.ana_vid_label.pack(fill="both", expand=True, padx=4, pady=4)
    
    def _apply_image_to_label(self, cv_frame, label):
        """Standardized image update for all video labels."""
        # Detect current size of the label widget
        lw = label.winfo_width()
        lh = label.winfo_height()
        
        # Fallback for initial state before window is rendered
        if lw < 10 or lh < 10:
            lw, lh = 800, 450
            
        img = Image.fromarray(cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)).resize((lw, lh))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk, text="")
        label._img_tk = img_tk

    # ------------------------------------------------------------------ #
    #  ANALYTICS LOGIC                                                   #
    # ------------------------------------------------------------------ #
    def select_analytics_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.start_analytics(file_path)

    def start_analytics(self, source):
        self.is_running = True
        self.stop_ana_btn.config(state="normal")
        self._set_sidebar_status("searching", "Analytics session active")
        threading.Thread(target=self.run_analytics_loop, args=(source,), daemon=True).start()

    def run_analytics_loop(self, source):
        cap = cv2.VideoCapture(source)
        ret, first_frame = cap.read()
        if not ret:
            self.is_running = False
            return
        
        self.analytics_engine.reset(first_frame.shape)
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break

            # Keep a copy of the last frame (backwards compatible name + new one)
            self.last_valid_frame = frame.copy()
            self.last_ana_frame = frame.copy()

            processed, count = self.analytics_engine.process_analytics_frame(frame)

            # Update stats on the main thread
            self.after(0, lambda c=count: self.ana_count_lbl.config(text=f"CURRENT: {c}"))
            self.after(0, lambda p=self.analytics_engine.max_count: self.ana_peak_lbl.config(text=f"PEAK: {p}"))

            # Update video display using centralized helper
            self.after(0, self._apply_image_to_label, processed, self.ana_vid_label)
        
        cap.release()

    def _update_ana_display(self, frame, count):
        self.ana_curr_lbl.config(text=f"CURRENT: {count}")
        self.ana_peak_lbl.config(text=f"PEAK: {self.analytics_engine.max_count}")
        
        lw, lh = self.ana_vid_label.winfo_width(), self.ana_vid_label.winfo_height()
        if lw < 10: lw, lh = 800, 450
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((lw, lh))
        img_tk = ImageTk.PhotoImage(img)
        self.ana_vid_label.config(image=img_tk, text="")
        self.ana_vid_label._img_tk = img_tk

    def stop_analytics(self):
        self.is_running = False
        self.stop_ana_btn.config(state="disabled")
        self._set_sidebar_status("done", "Analytics report generated")

        # Create final heatmap
        if hasattr(self, 'last_valid_frame'):
            final_heatmap = self.analytics_engine.get_final_heatmap(self.last_valid_frame)
            save_path = os.path.join(ANALYTICS_DIR, f"heatmap_{int(time.time())}.png")
            cv2.imwrite(save_path, final_heatmap)
            
            # Show final result in a new window or as a popup
            self.show_final_heatmap_popup(final_heatmap, save_path)

    def show_final_heatmap_popup(self, cv_img, path):
        top = tk.Toplevel(self)
        top.title("Session Report - People Density")
        top.configure(bg=BG_DARK)
        
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        img.thumbnail((1000, 700))
        img_tk = ImageTk.PhotoImage(img)
        
        lbl = tk.Label(top, image=img_tk, bg=BG_DARK)
        lbl.image = img_tk
        lbl.pack(padx=20, pady=20)
        
        tk.Label(top, text=f"Report saved to: {path}", fg=ACCENT_GREEN, bg=BG_DARK, font=self.font_mono).pack(pady=(0, 20))

    # ------------------------------------------------------------------ #
    #  EXISTING MODULES (RE-ID, HOME, RESULTS)                           #
    # ------------------------------------------------------------------ #
    # ... (Keep existing _build_home_page, _build_reid_page, _build_results_page) ...
    
    def _build_home_page(self):
        p = self.home_page

        # Header
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=30, pady=(30, 0))
        tk.Label(hdr, text="OPERATIONS DASHBOARD",
                 font=("Courier New", 20, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(anchor="w")
        tk.Label(hdr, text="AI-powered surveillance and person re-identification system",
                 font=("Courier New", 10), fg=TEXT_SECONDARY, bg=BG_DARK).pack(anchor="w", pady=(4, 0))
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=30, pady=15)

        # ── Stat Cards Row ────────────────────────────────────────────────
        cards_row = tk.Frame(p, bg=BG_DARK)
        cards_row.pack(fill="x", padx=30, pady=(0, 20))

        stat_defs = [
            ("DETECTION MODEL",  "YOLOv8n",    ACCENT_CYAN,  "◈"),
            ("RE-ID BACKBONE",   "OSNet AiN",  ACCENT_BLUE,  "◉"),
            ("TRACKING ENGINE",  "ByteTrack",  ACCENT_GREEN, "▲"),
            ("INFERENCE DEVICE", "GPU / CPU",  ACCENT_AMBER, "⬡"),
        ]
        for label, value, color, icon in stat_defs:
            card = tk.Frame(cards_row, bg=BG_CARD, bd=0, highlightbackground=BORDER,
                            highlightthickness=1)
            card.pack(side="left", padx=6, pady=4, fill="x", expand=True)
            inner = tk.Frame(card, bg=BG_CARD)
            inner.pack(padx=18, pady=16)
            tk.Label(inner, text=icon, font=("Courier New", 20),
                     fg=color, bg=BG_CARD).pack(anchor="w")
            tk.Label(inner, text=value, font=("Courier New", 14, "bold"),
                     fg=TEXT_PRIMARY, bg=BG_CARD).pack(anchor="w", pady=(4, 0))
            tk.Label(inner, text=label, font=("Courier New", 8),
                     fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w")

        # ── How It Works ──────────────────────────────────────────────────
        steps_frame = tk.Frame(p, bg=BG_DARK)
        steps_frame.pack(fill="x", padx=30, pady=(0, 20))
        tk.Label(steps_frame, text="WORKFLOW", font=("Courier New", 11, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).pack(anchor="w", pady=(0, 10))

        steps_inner = tk.Frame(steps_frame, bg=BG_DARK)
        steps_inner.pack(fill="x")

        steps = [
            ("01", "SELECT TARGET",    "Play reference video, click on the person of interest to lock tracking."),
            ("02", "AUTO-PROFILE",     "System captures 15 feature embeddings to build a robust appearance model."),
            ("03", "SEARCH FOOTAGE",   "Searches secondary video using cosine similarity against the gallery."),
            ("04", "REVIEW MATCHES",   "Browse timestamped match images saved to the matches/ directory."),
        ]
        for num, title, desc in steps:
            row = tk.Frame(steps_inner, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=num, font=("Courier New", 18, "bold"),
                     fg=TEXT_DIM, bg=BG_CARD, width=5).pack(side="left")
            txt_frame = tk.Frame(row, bg=BG_CARD)
            txt_frame.pack(side="left", fill="x", expand=True, pady=10)
            tk.Label(txt_frame, text=title, font=("Courier New", 11, "bold"),
                     fg=ACCENT_CYAN, bg=BG_CARD, anchor="w").pack(anchor="w")
            tk.Label(txt_frame, text=desc, font=("Courier New", 9),
                     fg=TEXT_SECONDARY, bg=BG_CARD, anchor="w").pack(anchor="w")

        # ── Launch Button ─────────────────────────────────────────────────
        btn_frame = tk.Frame(p, bg=BG_DARK)
        btn_frame.pack(fill="x", padx=30, pady=20)
        launch_btn = tk.Button(btn_frame, text="LAUNCH RE-ID MODULE  →",
                               font=("Courier New", 12, "bold"),
                               bg=ACCENT_CYAN, fg=BG_DARK,
                               relief="flat", cursor="hand2",
                               pady=14, padx=30,
                               command=lambda: self.show_page("ReID"))
        launch_btn.pack(anchor="w")

    # ------------------------------------------------------------------ #
    #  RE-ID PAGE                                                          #
    # ------------------------------------------------------------------ #
    def _build_reid_page(self):
        p = self.reid_page
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="PERSON RE-IDENTIFICATION", font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        self.phase_lbl = tk.Label(hdr, text="[ PHASE 1: TARGET SELECTION ]", font=("Courier New", 10, "bold"), fg=ACCENT_AMBER, bg=BG_DARK)
        self.phase_lbl.pack(side="right")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        cols = tk.Frame(p, bg=BG_DARK)
        cols.pack(fill="both", expand=True, padx=25, pady=15)

        # --- RIGHT SIDEBAR (Control Panel) ---
        right_col = tk.Frame(cols, bg=BG_DARK, width=320) 
        right_col.pack(side="right", fill="y", padx=(15, 0))
        right_col.pack_propagate(False) 

        # Operation Status
        status_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        status_card.pack(fill="x", pady=(0, 12))
        tk.Label(status_card, text="OPERATION STATUS", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.status_lbl = tk.Label(status_card, text="Awaiting start...", font=self.font_sub, fg=TEXT_SECONDARY, bg=BG_CARD, wraplength=280)
        self.status_lbl.pack(pady=10, padx=10)

        # Progress Bar
        prog_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        prog_card.pack(fill="x", pady=(0, 12))
        tk.Label(prog_card, text="SEARCH PROGRESS", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.prog_bar = ttk.Progressbar(prog_card, orient="horizontal", mode="determinate")
        self.prog_bar.pack(fill="x", padx=15, pady=10)
        self.prog_pct_lbl = tk.Label(prog_card, text="0%", font=self.font_mono, fg=TEXT_DIM, bg=BG_CARD)
        self.prog_pct_lbl.pack(pady=(0, 5))

        # Matches Count (THIS WAS MISSING)
        match_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        match_card.pack(fill="x", pady=(0, 12))
        tk.Label(match_card, text="MATCHES DETECTED", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.live_match_lbl = tk.Label(match_card, text="0", font=("Courier New", 26, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.live_match_lbl.pack(pady=10)

        # Tracked ID
        lock_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        lock_card.pack(fill="x", pady=(0, 12))
        tk.Label(lock_card, text="TRACKED ID", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.locked_id_lbl = tk.Label(lock_card, text="—", font=self.font_huge, fg=ACCENT_AMBER, bg=BG_CARD)
        self.locked_id_lbl.pack(pady=10)

        self.start_btn = tk.Button(right_col, text="▶ START PHASE 1", font=self.font_btn, bg=ACCENT_GREEN, fg=BG_DARK, pady=12, command=self.start_p1)
        self.start_btn.pack(fill="x", side="bottom")

        # --- LEFT AREA (Video Feed) ---
        left_col = tk.Frame(cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        feed_wrap = tk.Frame(left_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        feed_wrap.pack(fill="both", expand=True)
        self.vid_label = tk.Label(feed_wrap, bg="#040810", text="[ VIDEO FEED INACTIVE ]", font=self.font_title, fg=TEXT_DIM)
        self.vid_label.pack(fill="both", expand=True, padx=4, pady=4)
        self.vid_label.bind("<Button-1>", self.handle_click)

    def _update_gallery_bars(self):
        count = len(self.engine.target_gallery)
        for i, bar in enumerate(self.gallery_bars):
            bar.config(bg=ACCENT_CYAN if i < count else TEXT_DIM)
        self.gallery_count_lbl.config(text=f"{count} / 15 samples")
        if count > 0:
            self.gallery_count_lbl.config(fg=ACCENT_CYAN)
        self.after(200, self._update_gallery_bars)

    # ------------------------------------------------------------------ #
    #  RESULTS PAGE                                                        #
    # ------------------------------------------------------------------ #
    def _build_results_page(self):
        p = self.results_page

        # Header
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="MATCH RESULTS VIEWER",
                 font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        self.result_counter_lbl = tk.Label(hdr, text="", font=("Courier New", 10),
                                            fg=TEXT_SECONDARY, bg=BG_DARK)
        self.result_counter_lbl.pack(side="right")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        # Main content
        main = tk.Frame(p, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=25, pady=20)

        # Left: Image viewer
        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        img_card = tk.Frame(left, bg=BG_CARD, highlightbackground=BORDER_BRIGHT,
                            highlightthickness=1)
        img_card.pack(fill="both", expand=True)

        img_topbar = tk.Frame(img_card, bg=BG_SURFACE)
        img_topbar.pack(fill="x")
        tk.Label(img_topbar, text="▦ MATCH PREVIEW",
                 font=("Courier New", 9, "bold"), fg=ACCENT_CYAN,
                 bg=BG_SURFACE).pack(side="left", padx=12, pady=6)

        self.result_img_lbl = tk.Label(img_card, bg="#040810",
                                        text="[ NO RESULTS ]",
                                        font=("Courier New", 14), fg=TEXT_DIM)
        self.result_img_lbl.pack(fill="both", expand=True, padx=8, pady=8)

        # Nav bar
        nav_bar = tk.Frame(left, bg=BG_DARK)
        nav_bar.pack(fill="x", pady=(12, 0))

        self.prev_btn = tk.Button(nav_bar, text="◀  PREVIOUS",
                                   font=("Courier New", 11, "bold"),
                                   bg=BG_CARD, fg=TEXT_PRIMARY,
                                   relief="flat", cursor="hand2",
                                   pady=10, padx=20,
                                   highlightbackground=BORDER, highlightthickness=1,
                                   command=self.show_prev_result)
        self.prev_btn.pack(side="left")

        self.nav_pos_lbl = tk.Label(nav_bar, text="0 / 0",
                                     font=("Courier New", 12, "bold"),
                                     fg=TEXT_SECONDARY, bg=BG_DARK)
        self.nav_pos_lbl.pack(side="left", expand=True)

        self.next_btn = tk.Button(nav_bar, text="NEXT  ▶",
                                   font=("Courier New", 11, "bold"),
                                   bg=BG_CARD, fg=TEXT_PRIMARY,
                                   relief="flat", cursor="hand2",
                                   pady=10, padx=20,
                                   highlightbackground=BORDER, highlightthickness=1,
                                   command=self.show_next_result)
        self.next_btn.pack(side="right")

        # Right: Metadata panel
        right = tk.Frame(main, bg=BG_DARK, width=300)
        right.pack(side="right", fill="y", padx=(20, 0))
        right.pack_propagate(False)

        meta_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER,
                             highlightthickness=1)
        meta_card.pack(fill="x", pady=(0, 12))
        tk.Label(meta_card, text="MATCH METADATA",
                 font=("Courier New", 9, "bold"), fg=TEXT_SECONDARY,
                 bg=BG_SURFACE).pack(fill="x", padx=12, pady=6)

        meta_inner = tk.Frame(meta_card, bg=BG_CARD)
        meta_inner.pack(fill="x", padx=12, pady=12)

        self.meta_fields = {}
        for key, label in [("video", "SOURCE FILE"), ("timestamp", "TIMESTAMP"),
                            ("index", "MATCH INDEX"), ("saved", "SAVED TO")]:
            tk.Label(meta_inner, text=label, font=("Courier New", 8, "bold"),
                     fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", pady=(6, 0))
            val_lbl = tk.Label(meta_inner, text="—", font=("Courier New", 10),
                               fg=TEXT_PRIMARY, bg=BG_CARD, wraplength=240,
                               justify="left", anchor="w")
            val_lbl.pack(anchor="w")
            self.meta_fields[key] = val_lbl

        # Summary card
        summary_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER,
                                highlightthickness=1)
        summary_card.pack(fill="x", pady=(0, 12))
        tk.Label(summary_card, text="SESSION SUMMARY",
                 font=("Courier New", 9, "bold"), fg=TEXT_SECONDARY,
                 bg=BG_SURFACE).pack(fill="x", padx=12, pady=6)

        sum_inner = tk.Frame(summary_card, bg=BG_CARD)
        sum_inner.pack(fill="x", padx=12, pady=12)

        for key, label in [("total_matches", "TOTAL MATCHES"),
                            ("save_dir", "SAVE DIRECTORY"),
                            ("threshold", "MATCH THRESHOLD")]:
            tk.Label(sum_inner, text=label, font=("Courier New", 8, "bold"),
                     fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", pady=(6, 0))
            val = {"total_matches": "0",
                   "save_dir": f"./{MATCHES_DIR}/",
                   "threshold": "> 0.75 cosine sim"}.get(key, "—")
            lbl = tk.Label(sum_inner, text=val, font=("Courier New", 10),
                           fg=TEXT_PRIMARY, bg=BG_CARD, anchor="w")
            lbl.pack(anchor="w")
            if key == "total_matches":
                self.results_total_lbl = lbl

        # Back button
        tk.Button(right, text="◀  BACK TO TRACKING",
                  font=("Courier New", 10, "bold"),
                  bg=BG_CARD, fg=TEXT_SECONDARY,
                  relief="flat", cursor="hand2",
                  pady=10,
                  highlightbackground=BORDER, highlightthickness=1,
                  command=lambda: self.show_page("ReID")).pack(fill="x")
        
    def show_page(self, name):
        self.is_running = False # Stop all threads on page change
        self.home_page.pack_forget()
        self.reid_page.pack_forget()
        self.analytics_page.pack_forget()
        self.results_page.pack_forget()

        self._set_active_nav(name)
        if name == "Home": self.home_page.pack(fill="both", expand=True)
        elif name == "ReID": self.reid_page.pack(fill="both", expand=True)
        elif name == "Analytics": self.analytics_page.pack(fill="both", expand=True)
        elif name == "Results": self.results_page.pack(fill="both", expand=True); self._refresh_results_page()

    # ------------------------------------------------------------------ #
    #  CLICK HANDLING                                                      #
    # ------------------------------------------------------------------ #
    def handle_click(self, event):
        if not self.latest_detections:
            return

        label_w = self.vid_label.winfo_width()
        label_h = self.vid_label.winfo_height()
        orig_w, orig_h = self.current_frame_size
        real_x = event.x * (orig_w / label_w)
        real_y = event.y * (orig_h / label_h)

        for (xyxy, tid) in self.latest_detections:
            x1, y1, x2, y2 = xyxy
            if x1 <= real_x <= x2 and y1 <= real_y <= y2:
                self.selected_tid = tid
                self.engine.target_gallery = []
                self.locked_id_lbl.config(text=f"#{tid}", fg=ACCENT_AMBER)
                self._set_status(f"Locked on ID #{tid} — collecting features...", ACCENT_AMBER)
                self._set_sidebar_status("active", f"Collecting features for ID #{tid}")
                break

    # ------------------------------------------------------------------ #
    #  PHASE 1 – SELECTION                                                 #
    # ------------------------------------------------------------------ #
    def start_p1(self):
        self.is_running = True
        self.search_complete = False
        self.match_results = []
        self.match_queue = []
        self.selected_tid = None
        self.engine.target_gallery = []
        
        # Reset UI Elements
        self.locked_id_lbl.config(text="—")
        self.phase_lbl.config(text="[ PHASE 1: TARGET SELECTION ]", fg=ACCENT_AMBER)
        self.start_btn.config(state="disabled", bg=TEXT_DIM)
        
        # Safe config for labels that might have different names
        if hasattr(self, 'live_match_lbl'): self.live_match_lbl.config(text="0")
        if hasattr(self, 'match_count_lbl'): self.match_count_lbl.config(text="0")
        
        self.prog_bar["value"] = 0
        self.prog_pct_lbl.config(text="0%")
        
        self._set_status("Initializing camera... please click target person.")
        self._set_sidebar_status("active", "Phase 1: Selecting Target")
        
        # Start Thread
        threading.Thread(target=self.run_p1_loop, daemon=True).start()

    def run_p1_loop(self):
        cap = cv2.VideoCapture(VIDEO_1)
        self.current_frame_size = (int(cap.get(3)), int(cap.get(4)))

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            if len(self.engine.target_gallery) >= 15:
                break

            processed, detections = self.engine.process_frame(frame, self.selected_tid)
            self.latest_detections = detections

            if self.selected_tid is not None:
                for xyxy, tid in detections:
                    if tid == self.selected_tid:
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = frame[y1:y2, x1:x2]
                        feat = self.engine.get_features(crop)
                        if feat is not None:
                            self.engine.target_gallery.append(feat)

            # Use centralized image helper to keep sizing consistent
            self.after(0, self._apply_image_to_label, processed, self.vid_label)

        cap.release()
        if len(self.engine.target_gallery) >= 15:
            self.after(0, self.start_p2_background)
        else:
            self.after(0, lambda: self._set_status(
                "Phase 1 ended — target not fully profiled. Try again.", ACCENT_RED))
            self.after(0, lambda: self.start_btn.config(state="normal", bg=ACCENT_GREEN))

    # ------------------------------------------------------------------ #
    #  PHASE 2 – BACKGROUND SEARCH                                         #
    # ------------------------------------------------------------------ #
    def start_p2_background(self):
        self.phase_lbl.config(text="[ PHASE 2: SEARCHING FOOTAGE ]", fg=ACCENT_CYAN)
        self._set_status("Searching secondary footage...", ACCENT_CYAN)
        self._set_sidebar_status("searching", "Phase 2: Scanning footage")
        self.prog_bar['value'] = 0
        gallery_array = np.array(self.engine.target_gallery)
        threading.Thread(
            target=self.engine.search_video,
            args=(VIDEO_2, gallery_array, self.update_progress,
                  self.on_match_found, MATCHES_DIR),
            daemon=True
        ).start()
        self.check_match_queue()

    def update_progress(self, value):
        # Called from background thread — must route ALL Tk updates through after()
        self.after(0, self._apply_progress, value)

    def _apply_progress(self, value):
        """Runs on the main Tk thread."""
        self.prog_bar["value"] = value
        self.prog_pct_lbl.config(text=f"{value}%",
                                  fg=ACCENT_CYAN if value < 100 else ACCENT_GREEN)
        if value >= 100:
            self.search_complete = True
            # Give the UI queue 800ms to drain all remaining match callbacks
            self.after(800, self._on_search_complete)

    def _on_search_complete(self):
        total = len(self.match_results)
        self.phase_lbl.config(text="[ SEARCH COMPLETE ]", fg=ACCENT_GREEN)
        self._set_status(f"Search complete — {total} match(es) found", ACCENT_GREEN)
        self._set_sidebar_status("done", f"Done — {total} matches")
        self.status_dot.config(text="● DONE", fg=ACCENT_GREEN)
        self.match_count_lbl.config(text=str(total))
        self.live_match_lbl.config(text=str(total))
        self.results_total_lbl.config(text=str(total))
        self.start_btn.config(state="normal", bg=ACCENT_GREEN)

        if total > 0:
            # Show View Results button on reid page
            self.reid_results_btn.pack(fill="x", pady=(10, 0))
            # Show in sidebar
            self.view_results_btn.pack(fill="x", padx=10, pady=8)

    def on_match_found(self, filepath, video_name, timestamp):
        # Called from background thread — route to main thread via after()
        self.after(0, self.match_queue.append, (filepath, video_name, timestamp))

    def check_match_queue(self):
        while self.match_queue:
            filepath, video_name, ts = self.match_queue.pop(0)
            self.match_results.append((filepath, video_name, ts))
            count = len(self.match_results)
            self.live_match_lbl.config(text=str(count))
            self.match_count_lbl.config(text=str(count))

        # Keep polling until search is done AND queue is fully empty
        if not self.search_complete or self.match_queue:
            self.after(150, self.check_match_queue)

    # ------------------------------------------------------------------ #
    #  RESULTS VIEWER                                                      #
    # ------------------------------------------------------------------ #
    def _refresh_results_page(self):
        total = len(self.match_results)
        self.results_total_lbl.config(text=str(total))
        if total == 0:
            self.result_img_lbl.config(image="", text="[ NO MATCHES FOUND ]")
            self.nav_pos_lbl.config(text="0 / 0")
            self.result_counter_lbl.config(text="0 results")
            for f in self.meta_fields.values():
                f.config(text="—")
            return

        self.current_result_idx = 0
        self.result_counter_lbl.config(text=f"{total} result(s)")
        self._show_result(0)

    def _show_result(self, idx):
        if not self.match_results:
            return
        idx = max(0, min(idx, len(self.match_results) - 1))
        self.current_result_idx = idx

        filepath, video_name, timestamp = self.match_results[idx]

        # Load image
        try:
            img = Image.open(filepath)
            # Fit to display area
            disp_w, disp_h = 580, 480
            img.thumbnail((disp_w, disp_h), Image.LANCZOS)

            # Add padding to center
            bg = Image.new("RGB", (disp_w, disp_h), (4, 8, 16))
            offset_x = (disp_w - img.width) // 2
            offset_y = (disp_h - img.height) // 2
            bg.paste(img, (offset_x, offset_y))

            img_tk = ImageTk.PhotoImage(bg)
            self.result_img_lbl.config(image=img_tk, text="")
            self.result_img_lbl._img = img_tk
        except Exception as e:
            self.result_img_lbl.config(image="", text=f"[ Error loading image: {e} ]")

        # Update metadata
        total = len(self.match_results)
        self.meta_fields["video"].config(text=video_name)
        self.meta_fields["timestamp"].config(text=timestamp, fg=ACCENT_CYAN)
        self.meta_fields["index"].config(text=f"#{idx + 1} of {total}")
        self.meta_fields["saved"].config(text=os.path.basename(filepath), fg=ACCENT_GREEN)
        self.nav_pos_lbl.config(text=f"{idx + 1} / {total}")

        # Button states
        self.prev_btn.config(fg=TEXT_PRIMARY if idx > 0 else TEXT_DIM,
                              state="normal" if idx > 0 else "disabled")
        self.next_btn.config(fg=TEXT_PRIMARY if idx < total - 1 else TEXT_DIM,
                              state="normal" if idx < total - 1 else "disabled")

    def show_prev_result(self):
        self._show_result(self.current_result_idx - 1)

    def show_next_result(self):
        self._show_result(self.current_result_idx + 1)

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def record_zone_point(self, event):
        # Scale click to original resolution
        lw, lh = self.zone_vid_label.winfo_width(), self.zone_vid_label.winfo_height()
        ow, oh = self.current_frame_size
        rx, ry = int(event.x * (ow/lw)), int(event.y * (oh/lh))
        self.temp_points.append((rx, ry))
        self._set_status(f"Added point: {rx},{ry}. Right-click to finish zone.", ACCENT_CYAN)

    def finalize_zone(self, event):
        if len(self.temp_points) >= 3:
            self.zone_engine.add_zone(self.temp_points)
            self.temp_points = []
            messagebox.showinfo("Zone Saved", "Restricted zone added successfully.")
        else:
            messagebox.showwarning("Error", "Need at least 3 points for a zone.")

    def run_zone_monitoring(self, source):
        self.is_running = True
        cap = cv2.VideoCapture(source)
        self.current_frame_size = (int(cap.get(3)), int(cap.get(4)))
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break
            
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = f"{int(msec//60000)}:{int((msec//1000)%60):02d}"
            
            processed, new_alerts = self.zone_engine.process_frame(frame, ts, "matches")
            
            for alert in new_alerts:
                self.after(0, lambda a=alert: self.alert_box.insert(0, a))
            
            self.after(0, self._apply_image_to_label, processed, self.zone_vid_label)
        cap.release()

    def update_display(self, frame):
        # Called from Phase 1 thread — schedule on main thread
        self.after(0, self._apply_display, frame.copy())

    def _apply_display(self, frame):
        lw = self.vid_label.winfo_width()
        lh = self.vid_label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = 800, 450
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((lw, lh))
        img_tk = ImageTk.PhotoImage(img)
        self.vid_label.config(image=img_tk, text="")
        self.vid_label._img_tk = img_tk

    def _set_status(self, text, color=TEXT_SECONDARY):
        self.status_lbl.config(text=text, fg=color)

    def _set_sidebar_status(self, mode, text):
        colors = {"active": ACCENT_AMBER, "searching": ACCENT_CYAN,
                  "done": ACCENT_GREEN, "idle": TEXT_DIM}
        icons  = {"active": "● ACTIVE", "searching": "◌ SCANNING",
                  "done": "✔ COMPLETE", "idle": "● IDLE"}
        self.status_dot.config(text=icons.get(mode, "●"), fg=colors.get(mode, TEXT_DIM))
        self.sidebar_status.config(text=text)
    def _build_zone_page(self):
        p = self.zone_page # Make sure to initialize self.zone_page in __init__
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="VIRTUAL RESTRICTED ZONES", font=self.font_head, fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        
        main_cols = tk.Frame(p, bg=BG_DARK)
        main_cols.pack(fill="both", expand=True, padx=25, pady=15)

        # --- RIGHT SIDEBAR (ALERTS) ---
        right_col = tk.Frame(main_cols, bg=BG_PANEL, width=350)
        right_col.pack(side="right", fill="y", padx=(15, 0))
        right_col.pack_propagate(False)

        tk.Label(right_col, text="INTRUSION ALERTS", font=self.font_mono, fg=ACCENT_RED, bg=BG_PANEL).pack(pady=10)
        
        # Scrollable Alert List
        self.alert_box = tk.Listbox(right_col, bg=BG_DARK, fg=ACCENT_RED, font=self.font_mono, 
                                     borderwidth=0, highlightthickness=1, highlightbackground=BORDER)
        self.alert_box.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Button(right_col, text="CLEAR ZONES", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY, 
                  command=self.zone_engine.clear_zones).pack(fill="x", padx=20, pady=5)
        
        self.stop_zone_btn = tk.Button(right_col, text="■ STOP MONITORING", font=self.font_btn, 
                                       bg=ACCENT_RED, fg=TEXT_PRIMARY, pady=12, command=self.stop_processing)
        self.stop_zone_btn.pack(fill="x", padx=20, pady=10)

        # --- LEFT AREA (VIDEO) ---
        left_col = tk.Frame(main_cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        feed_wrap = tk.Frame(left_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        feed_wrap.pack(fill="both", expand=True)
        
        self.zone_vid_label = tk.Label(feed_wrap, bg="#040810", text="[ CLICK TO DRAW POLYGON ]", font=self.font_sub, fg=TEXT_DIM)
        self.zone_vid_label.pack(fill="both", expand=True, padx=4, pady=4)
        
        # Drawing Bindings
        self.zone_vid_label.bind("<Button-1>", self.record_zone_point)
        self.zone_vid_label.bind("<Button-3>", self.finalize_zone)
        
        self.temp_points = []


if __name__ == "__main__":
    app = SentinelVision()
    app.mainloop()