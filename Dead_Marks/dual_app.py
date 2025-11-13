import sys

# Are we running in headless mode?
HEADLESS = ("--headless" in sys.argv)

if not HEADLESS:
    # GUI-only imports (desktop use)
    import customtkinter as ctk
    from customtkinter import CTkImage
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
else:
    # Dummy placeholders so definitions below don't crash if referenced
    ctk = None
    CTkImage = None
    filedialog = None
    messagebox = None
    Image = None
    ImageTk = None

import cv2
import threading
import main
from main_stream import CameraStreamRunner
import time
import numpy as np
import json
import os
from main_stream import blendshape_names
from main_stream import reload_multipliers

import mediapipe as mp
from mediapipe_utils import create_face_landmarker


class DualApp:
    def __init__(self, root):
        self.root = root
        self.touch_mode = ('-touch' in sys.argv)

        # Ensure last_frame_time is always initialized
        self.last_frame_time = time.time()

        self.cap_video = None
        self.frame_count = 0
        self.frame_rate = 30
        self.landmarker_video = None
        self.current_frame_index = 0

        self.tracking_running = False
        self.tracking_thread = None
        self.tracking_stop = threading.Event()
        self._last_slider_ts = 0.0
        self._scrub_updating = False

        # Streaming state
        self.running_stream = False
        self.stream_runner = None

        if self.touch_mode:
            self.setup_touch_ui()
        else:
            self.setup_full_ui()

    def _fmt_time(self, frames, fps):
        if fps <= 0: return "00:00"
        seconds = int(frames / fps)
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"
    
    def update_mode(self):
        if self.mode.get() == "video":
            self.stream_frame.pack_forget()
            self.video_frame.pack(pady=5)
        else:
            self.video_frame.pack_forget()
            self.stream_frame.pack(pady=5)
            self.display_message("Camera stopped")
        # make sure we recreate the image on next frame
        self.canvas_image_id = None

        # update enabled/disabled controls per mode
        self.update_ui_mode()


    def setup_full_ui(self):
        self.root = root

        self.touch_mode = ('-touch' in sys.argv)
        if self.touch_mode:
            self.root.title("Touch Mode")
            self.root.geometry("340x320")  # accommodates 320x280 plus small padding
            self.root.resizable(False, False)
        self.root.title("Dual Mode Face Tracker")
        self.root.geometry("900x800")
        self.root.resizable(False, False)

        self.mode = ctk.StringVar(value="video")

        self.advanced_visible = False  # start visible

        mode_frame = ctk.CTkFrame(root, border_width=0, fg_color="transparent")
        mode_frame.pack(pady=10)



        ctk.CTkRadioButton(
            mode_frame,
            text="Video File",
            variable=self.mode,
            value="video",
            command=self.update_mode,
            fg_color="#84ecf0",
            border_color="#aaaaaa"
        ).pack(side="left", padx=10)

        ctk.CTkRadioButton(
            mode_frame,
            text="Stream Camera",
            variable=self.mode,
            value="stream",
            command=self.update_mode,
            fg_color="#84ecf0",
            border_color="#aaaaaa"
        ).pack(side="left", padx=10)

        self.fps_label = ctk.CTkLabel(root, text="FPS: 0")
        self.fps_label.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")
        self.last_frame_time = time.time()

        self.video_path = None
        self.first_frame = None

        # Video Frame
        self.video_frame = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0, border_width=0)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(1, weight=0)

        checkbox_frame = ctk.CTkFrame(self.video_frame, fg_color="transparent", border_width=0)
        checkbox_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")

        self.head_var = ctk.BooleanVar(value=True)
        self.head_check = ctk.CTkCheckBox(checkbox_frame, text="Enable Head Tracking", variable=self.head_var)
        self.head_check.pack(anchor="w", pady=5)

        self.eye_var = ctk.BooleanVar(value=False)
        self.eye_check = ctk.CTkCheckBox(checkbox_frame, text="Symmetrical Eyes", variable=self.eye_var)
        self.eye_check.pack(anchor="w", pady=5)

        load_frame = ctk.CTkFrame(self.video_frame, fg_color="transparent", border_width=0)
        load_frame.grid(row=1, column=0, columnspan=2, sticky="we", padx=5)
        load_frame.grid_columnconfigure(1, weight=1)

        self.load_button = ctk.CTkButton(load_frame, text="Load Video", command=self.load_video)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.video_label = ctk.CTkLabel(load_frame, text="No video loaded")
        self.video_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.start_video_button = ctk.CTkButton(self.video_frame, text="Start Tracking", command=self.start_tracking)
        self.start_video_button.grid(row=2, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        # --- Scrub slider row (for offline video) ---
        scrub_row = ctk.CTkFrame(self.video_frame, fg_color="transparent", border_width=0)
        scrub_row.grid(row=3, column=0, columnspan=2, sticky="we", padx=5, pady=(4, 0))

        # Give more weight to the slider cell, keep label cells fixed
        scrub_row.grid_columnconfigure(0, weight=0)  # "Timeline" label
        scrub_row.grid_columnconfigure(1, weight=1)  # slider (expandable)
        scrub_row.grid_columnconfigure(2, weight=0)  # time label

        # Label for the timeline
        scrub_title = ctk.CTkLabel(scrub_row, text="Timeline", anchor="w")
        scrub_title.grid(row=0, column=0, sticky="w", padx=(5, 10))

        self.scrub_slider = ctk.CTkSlider(
            scrub_row,
            from_=0,
            to=1,                    # default range 0→1 avoids centering
            number_of_steps=1,
            command=self.on_scrub_changed
        )
        self.scrub_slider.set(0)      # visually move handle to the left
        self.scrub_slider.grid(row=0, column=1, sticky="we", padx=5)


        # Timecode label on the right
        self.scrub_label = ctk.CTkLabel(scrub_row, text="00:00 / 00:00 (0/0)")
        self.scrub_label.grid(row=0, column=2, sticky="e", padx=5)



        # Buttons: Set Neutral (Video) + Zero (offline)
        self.neutral_from_video_btn = ctk.CTkButton(
            self.video_frame, text="Set Neutral (Video)",
            command=self.set_neutral_from_video
        )
        self.neutral_from_video_btn.grid(row=4, column=0, sticky="we", padx=(5, 2), pady=(6, 5))

        self.zero_video_neutral_btn = ctk.CTkButton(
            self.video_frame, text="Zero",
            width=80,
            command=self.zero_neutral_pose   # reuse the existing reset method
        )
        self.zero_video_neutral_btn.grid(row=4, column=1, sticky="we", padx=(2, 5), pady=(6, 5))



        # Stream Frame
        self.stream_frame = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0, border_width=0)

        # Buttons side by side
        stream_buttons_frame = ctk.CTkFrame(self.stream_frame, fg_color="transparent", corner_radius=0, border_width=0)
        stream_buttons_frame.pack(pady=5)

        self.test_cam_button = ctk.CTkButton(
            stream_buttons_frame,
            text="Test Cam",
            command=self.toggle_camera,
            width=75  # fixed width
        )
        self.test_cam_button.pack(side="left", padx=5)

        self.toggle_stream_button = ctk.CTkButton(
            stream_buttons_frame,
            text="Start Streaming",
            command=self.toggle_streaming
        )
        self.toggle_stream_button.pack(side="left", padx=5)

        # Neutral Pose Button
        self.neutral_pose_button = ctk.CTkButton(
            stream_buttons_frame,
            text="Set Neutral Pose",
            command=self.set_neutral_pose
        )
        self.neutral_pose_button.pack(side="left", padx=5)

        self.zero_pose_button = ctk.CTkButton(
            stream_buttons_frame,
            text="Zero",
            command=self.zero_neutral_pose,
            width=50  # small square-like button
            
        )
        self.zero_pose_button.pack(side="left", padx=5)

        # --- Curve Response (label + slider inline) ---
        curve_row = ctk.CTkFrame(self.stream_frame, fg_color="transparent", border_width=0)
        curve_row.pack(fill="x", padx=10, pady=(10, 0))
        self.curve_label = ctk.CTkLabel(curve_row, text="Curve Response", width=110, anchor="w")
        self.curve_label.pack(side="left")
        self.curve_slider = ctk.CTkSlider(
            curve_row,
            from_=-1,
            to=1,
            number_of_steps=200,
            command=self.update_curve_strength
        )
        self.curve_slider.set(0)
        self.curve_slider.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # --- Smoothing Filter (label + slider inline) ---
        filter_row = ctk.CTkFrame(self.stream_frame, fg_color="transparent", border_width=0)
        filter_row.pack(fill="x", padx=10, pady=(10, 0))
        self.filter_label = ctk.CTkLabel(filter_row, text="Smoothing Filter", width=110, anchor="w")
        self.filter_label.pack(side="left")
        self.filter_slider = ctk.CTkSlider(
            filter_row,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.update_filter_strength
        )
        self.filter_slider.set(0)
        self.filter_slider.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Improved Shapes toggle (below smoothing slider)
        self.improved_shapes_enabled = ctk.BooleanVar(value=False)

        self.improved_shapes_checkbox = ctk.CTkCheckBox(
            self.stream_frame,  # use same parent as slider
            text="Improved Shapes",
            variable=self.improved_shapes_enabled,
            onvalue=True,
            offvalue=False,
            command=self.update_improved_shapes
        )
        self.improved_shapes_checkbox.pack(pady=(0, 10))

        udp_config_frame = ctk.CTkFrame(self.stream_frame, fg_color="transparent", corner_radius=0, border_width=0)
        udp_config_frame.pack(pady=5, fill="x", expand=True)         # <— make frame stretch horizontally
        udp_config_frame.grid_columnconfigure(0, weight=0)
        udp_config_frame.grid_columnconfigure(1, weight=1)            # <— address column expands
        udp_config_frame.grid_columnconfigure(2, weight=0)
        udp_config_frame.grid_columnconfigure(3, weight=0)

        ctk.CTkLabel(udp_config_frame, text="UDP Address:").grid(row=0, column=0, sticky="w", padx=5)
        self.udp_entry = ctk.CTkEntry(
            udp_config_frame,
            placeholder_text="e.g. 192.168.1.50 or localhost"
        )
        self.udp_entry.insert(0, "127.0.0.1")
        self.udp_entry.grid(row=0, column=1, padx=5, sticky="we")

        ctk.CTkLabel(udp_config_frame, text="UDP Port:").grid(row=0, column=2, sticky="e", padx=(10,5))
        # Slightly wider port field
        self.port_entry = ctk.CTkEntry(udp_config_frame, width=100)
        self.port_entry.insert(0, "11111")
        self.port_entry.grid(row=0, column=3, padx=5, sticky="w")

        # === Container for canvas + advanced sliders ===
        self.content_frame = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0, border_width=0)
        self.content_frame.pack(pady=10, fill="both", expand=False)

        # Canvas on the left
        self.canvas = ctk.CTkCanvas(
            self.content_frame,
            width=640,
            height=480,
            bg="#222222",
            highlightthickness=0
        )
        self.canvas.pack(side="left", padx=10, pady=10)
        self.canvas_image_id = None       


        # === Advanced Multipliers Panel on the right ===
        self.multipliers = self.load_multipliers()

        self.advanced_container = ctk.CTkFrame(self.content_frame, width=220, height=480, border_width=0)
        self.advanced_container.pack(side="left", fill="y", padx=10, pady=10)
        self.advanced_container.pack_propagate(False)
        self.advanced_visible = False  # start hidden

        # Collapse/expand button
        self.toggle_advanced_btn = ctk.CTkButton(
            self.advanced_container,
            text="Hide Advanced",
            command=self.toggle_advanced_panel
        )
        self.toggle_advanced_btn.pack(pady=(10, 5))

        # Reset button
        self.reset_multipliers_btn = ctk.CTkButton(
            self.advanced_container,
            text="Reset Multipliers",
            command=self.reset_all_multipliers
        )
        self.reset_multipliers_btn.pack(pady=(0, 10))

        # Create scrollable frame for sliders
        self.advanced_scroll = ctk.CTkScrollableFrame(self.advanced_container, width=200, height=500)
        self.advanced_scroll.pack(fill="both", expand=True)

        # Build symmetrical mapping (Left/Right -> -sym)
        sym_mapping = {}
        processed_names = []

        for name in blendshape_names:
            base = None
            if "Left" in name:
                base = name.replace("Left", "")
            elif "Right" in name:
                base = name.replace("Right", "")

            if base:
                sym_name = base + "-sym"
                if sym_name not in sym_mapping:
                    sym_mapping[sym_name] = []
                    processed_names.append(sym_name)
                if "Left" in name:
                    sym_mapping[sym_name].append(base + "Left")
                if "Right" in name:
                    sym_mapping[sym_name].append(base + "Right")
            else:
                processed_names.append(name)

        self.sym_mapping = sym_mapping  # Store for use in update_multiplier

        # Group blendshapes (processed)
        groups = {
            "Eyes": [b for b in processed_names if "eye" in b.lower()],
            "Mouth": [b for b in processed_names if "mouth" in b.lower() or "jaw" in b.lower() or "lips" in b.lower()],
            "Brows": [b for b in processed_names if "brow" in b.lower()],
            "Other": [b for b in processed_names if b not in [*([b for b in processed_names if "eye" in b.lower()]), *([b for b in processed_names if "mouth" in b.lower() or "jaw" in b.lower() or "lips" in b.lower()]), *([b for b in processed_names if "brow" in b.lower()])]]
        }

        # Build grouped sliders
        self.slider_widgets = {}

        for group_name, shapes in groups.items():
            group_label = ctk.CTkLabel(self.advanced_scroll, text=group_name, font=("Arial", 12, "bold"))
            group_label.pack(pady=(5, 0))

            for name in shapes:
                val = self.multipliers.get(name, 1.0)

                # Label with hover info
                lbl = ctk.CTkLabel(self.advanced_scroll, text=name, font=("Arial", 10))
                lbl.pack()

                # Slider
                slider = ctk.CTkSlider(
                    self.advanced_scroll,
                    from_=0,
                    to=5,
                    number_of_steps=30,
                    command=lambda v, n=name: self.update_multiplier(n, v)
                )
                slider.set(val)
                slider.pack(pady=(0, 5))

                self.slider_widgets[name] = slider


        if os.path.exists("deadface.png"):
            # Load image once and wrap in CTkImage
            icon_img = CTkImage(
                light_image=Image.open("deadface.png"),  # same for light and dark mode
                dark_image=Image.open("deadface.png"),
                size=(100, 100)  # scale to desired size
            )
            self.icon_label = ctk.CTkLabel(root, image=icon_img, text="")
            self.icon_label.place(relx=0.0, rely=1.0, x=10, y=-10, anchor="sw")
            

        self.running_camera = False
        self.running_stream = False
        self.cap = None
        self.stream_runner = None

        if self.touch_mode:
            # Hide everything except the canvas and start button
            self.fps_label.pack_forget()
            self.video_frame.pack_forget()
            self.stream_frame.pack_forget()
            self.content_frame.pack_forget()
            # Remove advanced panel entirely
            self.advanced_container.destroy()

            # Add minimal canvas
            self.canvas = ctk.CTkCanvas(self.root, width=320, height=280,
                                        bg="#222222", highlightthickness=0)
            self.canvas.place(x=10, y=10)

            # Overlay Start/Stop Streaming button
            self.start_button = ctk.CTkButton(self.root, text="Start Stream",
                                            command=self.toggle_streaming, width=120)
            self.start_button.place(x=170, y=300, anchor="n")

            return

        self.update_mode()
        self.current_frame_bgr = None  # track last frame

        # show centered deadface.png once after layout
        self.root.after_idle(self.show_canvas_placeholder)  

        # Ensure neutral_pose.json exists
        if not os.path.exists("neutral_pose.json"):
            with open("neutral_pose.json", "w") as f:
                json.dump({"blendshapes": {}, "custom": {}, "raw": {}}, f)

    def setup_touch_ui(self):
        self.root.attributes('-fullscreen', True)  # Fullscreen
        self.root.configure(bg="black")  # Optional: black background

        # Canvas to display video
        self.canvas = ctk.CTkCanvas(self.root, width=320, height=240, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas_image_id = None

        # Start/Stop stream button overlaid at bottom center
        self.start_button = ctk.CTkButton(
            self.root, text="Start Stream", font=("Segoe UI", 14),
            command=self.toggle_streaming
        )
        self.start_button.place(relx=0.5, rely=0.9, anchor="center")

    def update_fps(self):
        now = time.time()
        fps = 1.0 / (now - self.last_frame_time)
        self.last_frame_time = now
        self.fps_label.configure(text=f"FPS: {fps:.2f}")

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        if not path:
            self.display_message("No video loaded")
            return

        self.video_path = path
        self.video_label.configure(text=path)

        # --- Release any previous capture ---
        if hasattr(self, "cap_video") and self.cap_video is not None:
            try:
                self.cap_video.release()
            except:
                pass
            self.cap_video = None

        # --- Open the new video ---
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.display_message("Error loading video")
            return

        self.cap_video = cap
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.current_frame_index = 0

        # --- Configure scrub slider ---
        if hasattr(self, "scrub_slider"):
            self.scrub_slider.configure(
                from_=0,
                to=max(self.frame_count - 1, 0),
                number_of_steps=max(self.frame_count - 1, 1)
            )
        self._update_scrub_label()

        # --- Create landmarker for video mode (if not yet created) ---
        if getattr(self, "landmarker_video", None) is None:
            from mediapipe_utils import create_face_landmarker
            self.landmarker_video = create_face_landmarker()

        # --- Read and display first frame ---
        ret, frame = cap.read()
        if not ret:
            self.display_message("Error reading first frame")
            return

        # Clear any previous items; let the renderer create a fresh image item
        self.canvas.delete("all")
        self.canvas_image_id = None
        self.update_canvas_frame(frame)


        self.display_message(f"Loaded video: {os.path.basename(path)} "
                            f"({self.frame_count} frames @ {self.frame_rate:.2f} fps)")

        # store frame for later reference if needed
        # self.first_frame = frame_rgb

    def show_canvas_placeholder(self):
        """Show deadface.png centered on the canvas once, at startup."""
        try:
            if not os.path.exists("deadface.png"):
                return

            self.root.update_idletasks()
            cw = self.canvas.winfo_width()  or (320 if self.touch_mode else 640)
            ch = self.canvas.winfo_height() or (280 if self.touch_mode else 480)

            img = Image.open("deadface.png")
            iw, ih = img.size
            max_w, max_h = int(cw * 0.7), int(ch * 0.7)
            scale = min(max_w / max(1, iw), max_h / max(1, ih), 1.0)
            img = img.resize((max(1, int(iw*scale)), max(1, int(ih*scale))), Image.BICUBIC)

            ox = (cw - img.width) // 2
            oy = (ch - img.height) // 2

            # clear and draw
            self.canvas.delete("all")
            self.canvas_image_id = None

            # keep a ref so Tk doesn't GC it
            self.photo = ImageTk.PhotoImage(img)

            self.canvas_image_id = self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        except Exception as e:
            print("Placeholder error:", e)



    def display_message(self, message):
        self.canvas.delete("all")
        # IMPORTANT: force re-create of canvas image on next frame
        self.canvas_image_id = None
        self.canvas.create_text(320, 240, text=message, fill="white", font=("Segoe UI", 20))

    def start_tracking(self):
        # If already running, this acts as "Stop"
        if self.tracking_running:
            self.tracking_stop.set()
            # prevent double-clicks during shutdown
            self.start_video_button.configure(text="Stopping...", state="disabled")
            return

        if not self.video_path:
            messagebox.showerror("Error", "No video selected.")
            return

        # Start new tracking
        self.tracking_stop.clear()
        self.tracking_running = True
        self.start_video_button.configure(text="Stop Tracking", state="normal")

        head_tracking = self.head_var.get()
        symmetrical_eyes = self.eye_var.get()

        # run in worker thread
        self.tracking_thread = threading.Thread(
            target=self._tracking_worker,
            args=(self.video_path, head_tracking, symmetrical_eyes),
            daemon=True
        )
        self.tracking_thread.start()

    def _tracking_worker(self, video_path, head_tracking, symmetrical_eyes):
        try:
            # disable manual scrubbing while tracking updates it
            try: self.scrub_slider.configure(state="disabled")
            except: pass

            def update_safe(frame_bgr, frame_index=None):
                # always update image on UI thread
                self.root.after(0, lambda: self.update_canvas_frame(frame_bgr))

                # throttle slider/time updates to ~10 Hz
                if frame_index is not None:
                    import time
                    now = time.perf_counter()
                    if now - self._last_slider_ts < 0.10:
                        return
                    self._last_slider_ts = now

                    def _move():
                        self._scrub_updating = True
                        try:
                            self.current_frame_index = int(frame_index)
                            self.scrub_slider.set(self.current_frame_index)
                            self._update_scrub_label()
                        finally:
                            self._scrub_updating = False
                    self.root.after(0, _move)

            # call into main.py with a stop event
            main.run_tracking(
                video_path=video_path,
                head_tracking=head_tracking,
                symmetrical_eyes=symmetrical_eyes,
                display_callback=update_safe,
                stop_event=self.tracking_stop
            )
        finally:
            # reset UI on finish/stop
            def _reset_ui():
                self.tracking_running = False
                self.start_video_button.configure(text="Start Tracking", state="normal")
                try: self.scrub_slider.configure(state="normal")
                except: pass
                self.tracking_stop.clear()
            self.root.after(0, _reset_ui)

    def on_scrub_changed(self, value):
        if getattr(self, "_scrub_updating", False):
            return  # ignore programmatic moves
        self._show_frame_at_index(int(value))


    def run_processing(self, video_path, head_tracking, symmetrical_eyes):
        # disable user scrubbing while tracking runs
        try:
            self.scrub_slider.configure(state="disabled")
        except Exception:
            pass

        self._last_slider_ts = 0.0
        self._scrub_updating = False

        def update_safe(frame_bgr, frame_index=None):
            # update image (always on UI thread)
            self.root.after(0, lambda: self.update_canvas_frame(frame_bgr))

            # update slider position at most ~10 Hz
            if frame_index is not None:
                import time
                now = time.perf_counter()
                if now - self._last_slider_ts < 0.1:
                    return
                self._last_slider_ts = now

                def _move():
                    self._scrub_updating = True
                    try:
                        self.current_frame_index = int(frame_index)
                        self.scrub_slider.set(self.current_frame_index)
                        self._update_scrub_label()
                    finally:
                        self._scrub_updating = False
                self.root.after(0, _move)

        main.run_tracking(
            video_path=video_path,
            head_tracking=head_tracking,
            symmetrical_eyes=symmetrical_eyes,
            display_callback=update_safe
        )

        # re-enable scrubbing when done
        try:
            self.scrub_slider.configure(state="normal")
        except Exception:
            pass


    def update_canvas_frame(self, frame_bgr):
        try:
            if frame_bgr is None:
                return

            # --- Get canvas size (fallback to configured size if not ready) ---
            self.root.update_idletasks()  # ensure geometry is current
            cw = self.canvas.winfo_width()  or (320 if self.touch_mode else 640)
            ch = self.canvas.winfo_height() or (280 if self.touch_mode else 480)

            # --- Convert to PIL image ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # --- Scale to FIT (preserve aspect ratio) ---
            iw, ih = img.size
            if iw == 0 or ih == 0:
                return
            scale = min(cw / iw, ch / ih)
            sw, sh = max(1, int(iw * scale)), max(1, int(ih * scale))
            img = img.resize((sw, sh), Image.BICUBIC)

            # --- Compute letterbox offsets (centered) ---
            ox = (cw - sw) // 2
            oy = (ch - sh) // 2

            # --- Convert to Tk image ---
            self.photo = ImageTk.PhotoImage(img)

            # --- First time: create; otherwise update image + position ---
            if not getattr(self, "canvas_image_id", None):
                self.canvas_image_id = self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
            else:
                # update bitmap
                try:
                    self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
                    # and ensure it's centered at the new offsets
                    self.canvas.coords(self.canvas_image_id, ox, oy)
                except Exception:
                    self.canvas_image_id = self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)

            # Optional: show FPS only in full mode
            if not self.touch_mode:
                self.update_fps()

        except Exception as e:
            print("Frame update error:", e)



        
       

    def set_neutral_pose(self):
        from main_stream import get_current_blendshapes

        blendshapes, landmarks = get_current_blendshapes()
        if not blendshapes or not landmarks:
            self.display_message("No live face data for neutral pose.")
            return

        blendshape_dict = {b.category_name: b.score for b in blendshapes}

        # Compute raw distances from landmarks (same as before)
        lip_distance = np.linalg.norm([
            landmarks[13].x - landmarks[14].x,
            landmarks[13].y - landmarks[14].y,
            landmarks[13].z - landmarks[14].z
        ])
        lip_width = np.linalg.norm([
            landmarks[61].x - landmarks[291].x,
            landmarks[61].y - landmarks[291].y,
            landmarks[61].z - landmarks[291].z
        ])
        nostril_distance = np.linalg.norm([
            landmarks[98].x - landmarks[327].x,
            landmarks[98].y - landmarks[327].y,
            landmarks[98].z - landmarks[327].z
        ])

        # Compute custom scores as before
        max_mouth_open_distance = 0.05
        mouth_closed_raw = 1.0 - min(lip_distance / max_mouth_open_distance, 1.0)
        jaw_open_score = blendshape_dict.get("jawOpen", 0.0)
        mouth_closed_score = mouth_closed_raw * (1.0 - jaw_open_score)

        pucker_ratio = 0.0  # leave for later
        mouth_pucker_score = 0.0
        nose_sneer_score = 1.0  # neutral

        custom_scores = {
            "mouthClosedScore": mouth_closed_score,
            "mouthPuckerScore": mouth_pucker_score,
            "noseSneerScore": nose_sneer_score
        }

        neutral_data = {
            "blendshapes": blendshape_dict,
            "custom": custom_scores,
            "raw": {
                "neutral_lip_width": lip_width,
                "neutral_nostril_distance": nostril_distance,
                "neutral_lip_distance": lip_distance
            }
        }

        # Save JSON and reload into stream
        with open("neutral_pose.json.tmp", "w") as f:
            json.dump(neutral_data, f, indent=2)
        os.replace("neutral_pose.json.tmp", "neutral_pose.json")

        from main_stream import reload_neutral_pose
        reload_neutral_pose()
        self.display_message("Neutral pose saved and applied.")

    def toggle_camera(self):
        if self.running_camera:
            self.running_camera = False
            self.test_cam_button.configure(text="Test Cam", width=75)
        else:
            self.running_camera = True
            self.test_cam_button.configure(text="Stop Cam", width=75)
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        while self.running_camera and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.root.after(0, lambda f=frame: self.update_canvas_frame(f))
            else:
                break
        self.cap.release()
        self.test_cam_button.configure(text="Test Cam", width=75)

    def zero_neutral_pose(self):
        """Reset neutral_pose.json to all zeros and reload into stream."""
        neutral_data = {
            "blendshapes": {},
            "custom": {},
            "raw": {}
        }

        # Save cleared data
        with open("neutral_pose.json.tmp", "w") as f:
            json.dump(neutral_data, f, indent=2)
        os.replace("neutral_pose.json.tmp", "neutral_pose.json")

        # Reload into stream
        from main_stream import reload_neutral_pose
        reload_neutral_pose()
        self.display_message("Neutral pose reset to zero.")

    def update_filter_strength(self, value):
        if self.stream_runner:
            self.stream_runner.set_filter_strength(float(value))

    def load_multipliers(self):
        """Load multipliers from JSON."""
        if os.path.exists("multipliers.json"):
            with open("multipliers.json", "r") as f:
                return json.load(f)
        return {}

    def save_multipliers(self):
        """Save multipliers to JSON."""
        with open("multipliers.json", "w") as f:
            json.dump(self.multipliers, f, indent=2)
    
    def reset_all_multipliers(self):
        """Reset all multipliers to 1.0."""
        for name, slider in self.slider_widgets.items():
            slider.set(1.0)
            if name.endswith("-sym") and name in self.sym_mapping:
                for real_name in self.sym_mapping[name]:
                    self.multipliers[real_name] = 1.0
            else:
                self.multipliers[name] = 1.0
        self.save_multipliers()

    def toggle_advanced_panel(self):
        if self.advanced_visible:
            self.advanced_scroll.pack_forget()
            self.reset_multipliers_btn.pack_forget()
            self.toggle_advanced_btn.configure(text="Show Advanced")
            self.advanced_visible = False
        else:
            self.reset_multipliers_btn.pack(pady=(0, 10))
            self.advanced_scroll.pack(fill="both", expand=True)
            self.toggle_advanced_btn.configure(text="Hide Advanced")
            self.advanced_visible = True
 
    def update_multiplier(self, name, value):
        if abs(value - 1.0) < 0.05:
            value = 1.0
            self.slider_widgets[name].set(1.0)
        # Symmetrical sliders apply to both Left and Right variants
        if name.endswith("-sym") and name in self.sym_mapping:
            for real_name in self.sym_mapping[name]:
                self.multipliers[real_name] = round(value, 2)
        else:
            self.multipliers[name] = round(value, 2)

        self.save_multipliers()
        reload_multipliers()

    def update_improved_shapes(self):
        """Toggle improved shapes correction in the stream."""
        if self.stream_runner is not None:
            self.stream_runner.set_improved_shapes(self.improved_shapes_enabled.get())

    def update_curve_strength(self, value):
        if self.stream_runner is not None:
            self.stream_runner.set_curve_strength(float(value))

    def toggle_streaming(self):
        if self.touch_mode:
            # Touch mode: fixed UDP
            if not self.running_stream:
                self.running_stream = True
                self.stream_runner = CameraStreamRunner("192.168.0.16", 11111)
                threading.Thread(
                    target=lambda: self.stream_runner.run(
                        display_callback=lambda f: self.update_canvas_frame(f)),
                    daemon=True
                ).start()
                self.start_button.configure(text="Stop Stream")
            else:
                if self.stream_runner:
                    self.stream_runner.stop()
                self.running_stream = False
                self.start_button.configure(text="Start Stream")
                self.canvas.delete("all")
                self.canvas_image_id = None
                self.canvas.create_text(160, 140, text="Stream stopped",
                                        fill="white", font=("Segoe UI", 16))
        else:
            # Full mode: use user‑entered UDP
            if self.running_stream:
                if self.stream_runner:
                    self.stream_runner.stop()
                self.running_stream = False
                self.toggle_stream_button.configure(text="Start Streaming")
                self.display_message("Streaming stopped")
            else:
                udp_address = self.udp_entry.get()
                udp_port = int(self.port_entry.get())
                self.running_stream = True
                self.stream_runner = CameraStreamRunner(udp_address, udp_port)

                def update_safe(frame_bgr):
                    if not self.running_stream:
                        return
                    self.root.after(0, lambda: self.update_canvas_frame(frame_bgr))

                threading.Thread(
                    target=lambda: self.stream_runner.run(display_callback=update_safe),
                    daemon=True
                ).start()
                self.toggle_stream_button.configure(text="Stop Streaming")

    def _update_scrub_label(self):
        cur = int(self.current_frame_index)
        total = int(max(self.frame_count-1, 0))
        left  = self._fmt_time(cur, self.frame_rate)
        right = self._fmt_time(total, self.frame_rate)
        self.scrub_label.configure(text=f"{left} / {right} ({cur}/{total})")

    def _show_frame_at_index(self, index:int):
        if self.cap_video is None: return
        index = max(0, min(index, self.frame_count-1))
        self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap_video.read()
        if not ret:
            self.display_message("Cannot read frame")
            return
        self.current_frame_index = index
        self._update_scrub_label()
        # display on canvas
        self.update_canvas_frame(frame)

    def on_scrub_changed(self, value):
        if getattr(self, "_scrub_updating", False):
            return  # ignore programmatic moves
        self._show_frame_at_index(int(value))


    def set_neutral_from_video(self):
        if self.cap_video is None:
            self.display_message("Load a video first.")
            return

        # Ensure we have a landmarker for video mode
        if self.landmarker_video is None:
            if create_face_landmarker is not None:
                self.landmarker_video = create_face_landmarker()
            else:
                # Inline landmarker factory (MP Tasks) if you don’t have the helper
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=1,
                    output_facial_transformation_matrixes=False,
                    output_face_blendshapes=True,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self.landmarker_video = vision.FaceLandmarker.create_from_options(options)

        # Grab the exact frame where the slider currently is
        target_index = int(self.current_frame_index)
        self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame_bgr = self.cap_video.read()
        if not ret:
            self.display_message("Failed to read current frame.")
            return

        # Convert to MediaPipe image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp from frame index + FPS (constant-rate only, as desired)
        ts_ms = int((target_index / max(self.frame_rate, 1e-6)) * 1000.0)

        result = self.landmarker_video.detect_for_video(mp_image, ts_ms)
        if not result or not result.face_blendshapes or not result.face_landmarks:
            self.display_message("No face detected on this frame.")
            return

        blendshapes = result.face_blendshapes[0]
        landmarks = result.face_landmarks[0]  # 468 FaceMesh-style landmarks

        # Build a {category_name: score} dict
        blendshape_dict = {b.category_name: float(b.score) for b in blendshapes}

        # Basic raw distances (use your indices consistent with live mode)
        def _dist(a, b):
            ax, ay, az = a.x, a.y, a.z
            bx, by, bz = b.x, b.y, b.z
            return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))

        lip_distance = _dist(landmarks[13], landmarks[14])      # upper/lower inner lip
        lip_width = _dist(landmarks[61], landmarks[291])        # mouth corners
        nostril_distance = _dist(landmarks[98], landmarks[327]) # nostrils

        # Simple custom scores aligned with your live pipeline
        max_mouth_open_distance = 0.05
        mouth_closed_raw = 1.0 - min(lip_distance / max_mouth_open_distance, 1.0)
        jaw_open_score = blendshape_dict.get("jawOpen", 0.0)
        mouth_closed_score = mouth_closed_raw * (1.0 - jaw_open_score)

        neutral_data = {
            "blendshapes": blendshape_dict,
            "custom": {
                "mouthClosedScore": mouth_closed_score,
                "mouthPuckerScore": 0.0,
                "noseSneerScore": 1.0
            },
            "raw": {
                "neutral_lip_width": lip_width,
                "neutral_nostril_distance": nostril_distance,
                "neutral_lip_distance": lip_distance
            }
        }

        # Save atomically
        tmp_path = "neutral_pose.json.tmp"
        out_path = "neutral_pose.json"
        with open(tmp_path, "w") as f:
            json.dump(neutral_data, f, indent=2)
        os.replace(tmp_path, out_path)

        # If your live stream supports hot reload, call it (safe-try)
        try:
            from main_stream import reload_neutral_pose
            reload_neutral_pose()
        except Exception:
            pass

        self.display_message("Neutral pose saved from current video frame.")

    def update_ui_mode(self):
        """
        Enable/disable controls depending on active mode.
        - Live (stream) mode: grey out 'Improved Shapes' (coming soon).
        - Offline (video) mode: grey out 'Show Advanced'.
        """
        mode = self.mode.get()  # "video" or "stream"

        # --- Improved Shapes (live panel control) ---
        if hasattr(self, "improved_shapes_checkbox"):
            if mode == "stream":
                # Visible in live, but disabled (greyed out)
                self.improved_shapes_checkbox.configure(state="disabled", text="Improved Shapes (not yet implemented)")
            else:
                # In video mode this checkbox isn't shown anyway, but reset for consistency
                self.improved_shapes_checkbox.configure(state="normal", text="Improved Shapes")

        # --- Show/Hide Advanced button (global) ---
        if hasattr(self, "toggle_advanced_btn"):
            if mode == "video":
                # Grey out in offline mode
                self.toggle_advanced_btn.configure(state="disabled")
            else:
                self.toggle_advanced_btn.configure(state="normal")

        # --- Hide advanced panel when in offline mode ---
        if hasattr(self, "advanced_container"):
            if mode == "video":
                # Only hide if currently shown
                try:
                    self.advanced_container.pack_forget()
                    self.advanced_visible = False
                except Exception:
                    pass
            else:
                # In stream mode, show advanced panel by default
                if not self.advanced_visible:
                    self.advanced_container.pack(side="left", fill="y", padx=10, pady=10)
                    self.advanced_visible = True




if __name__ == "__main__":
    if "--headless" in sys.argv:
        from main_headless import main as headless_main
        argv = [a for a in sys.argv[1:] if a != "--headless"]
        headless_main(argv)
    else:
        # ctk is already imported at top when not HEADLESS
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("sky_dark_theme.json")
        root = ctk.CTk()
        app = DualApp(root)
        root.mainloop()



