import argparse
import time
import socket
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

import cv2
import mediapipe as mp
import numpy as np

from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from blendshape_utils import BLENDSHAPE_STREAM_NAMES

from picamera2 import Picamera2
from libcamera import Transform

# Reduce verbose logging from Mediapipe / TF on low-powered devices
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Mediapipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "DeadFace.task"

neutral_pose_data = None
neutral_blendshapes_baseline = {}
neutral_custom_baseline = {}
neutral_raw_baseline = {}

# Smoothing factor for blendshapes (0..1)
SMOOTHING_ALPHA = 0.75

# Latest frame for snapshot server (BGR numpy array)
latest_frame = None


# ------------- Snapshot HTTP server (same process, shared camera) -------------

class SnapshotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame

        if self.path != "/snapshot.jpg":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        frame = latest_frame
        if frame is None or frame.size == 0:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"No frame yet")
            return
        
        # --- ROTATE SNAPSHOT 90° CCW ---
        rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        

        # Encode current frame as JPEG
        ret, jpeg = cv2.imencode(".jpg", rotated)
        if not ret:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Encode error")
            return

        data = jpeg.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Silence default HTTP logging
        return


def start_snapshot_server(port=8000):
    server = HTTPServer(("", port), SnapshotHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[HEADLESS-PI] Snapshot server on port {port} (GET /snapshot.jpg)")
    return server


# ------------------------- Neutral / multipliers utils ------------------------


def _load_multipliers():
    if os.path.exists("multipliers.json"):
        try:
            import json
            with open("multipliers.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


multipliers = _load_multipliers()


def _load_neutral_pose():
    global neutral_pose_data, neutral_blendshapes_baseline, neutral_custom_baseline, neutral_raw_baseline
    try:
        import json
        if os.path.exists("neutral_pose.json"):
            with open("neutral_pose.json", "r") as f:
                neutral_pose_data = json.load(f)
            neutral_blendshapes_baseline = neutral_pose_data.get("blendshapes", {}) or {}
            neutral_custom_baseline = neutral_pose_data.get("custom", {}) or {}
            neutral_raw_baseline = neutral_pose_data.get("raw", {}) or {}
    except Exception:
        neutral_pose_data = None
        neutral_blendshapes_baseline = {}
        neutral_custom_baseline = {}
        neutral_raw_baseline = {}


_load_neutral_pose()


# ----------------------------- Main headless loop -----------------------------


def run_headless_picam(
    udp_address: str = "127.0.0.1",
    udp_port: int = 11111,
    width: int = 320,
    height: int = 240,
    target_fps: float = 15.0,
    snapshot_port: int = 8000,
):
    """
    Pi-native headless streamer using Picamera2 directly.

    - Picamera2 @ (width x height) BGR
    - Mediapipe Face Landmarker in VIDEO mode (sync, low-latency)
    - No head rotation (helmet mode)
    - Sends blendshapes + iris over UDP
    - Serves /snapshot.jpg for head framing preview
    - Prints streaming FPS + jawOpen
    """
    global latest_frame

    print(f"[HEADLESS-PI] Starting Picamera2 stream.")
    print(f"[HEADLESS-PI] Resolution: {width}x{height}, target FPS: {target_fps:.1f}")
    print(f"[HEADLESS-PI] UDP target: {udp_address}:{udp_port}")
    print(f"[HEADLESS-PI] Snapshot HTTP port: {snapshot_port} (GET /snapshot.jpg)")
    print("[HEADLESS-PI] Press Ctrl+C to stop.\n")

    # --- Camera setup: Picamera2, rotated if needed ---
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (width, height), "format": "BGR888"},
        transform=Transform(rotation=0),  # change to 90/180 if your sensor orientation needs it
    )
    picam2.configure(cfg)
    picam2.start()

    try:
        picam2.set_controls({"FrameRate": int(target_fps)})
    except Exception:
        pass

    # Start snapshot server
    snapshot_server = start_snapshot_server(snapshot_port)

    # --- UDP + LiveLink face ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((udp_address, udp_port))
    py_face = PyLiveLinkFace()

    max_mouth_open_distance = 0.05
    neutral_lip_width = 0.05
    neutral_nostril_distance = 0.035
    neutral_captured = False

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    frame_interval = 1.0 / max(target_fps, 1e-6)
    frame_index = 0

    # FPS monitoring
    frame_times = []
    last_fps_print_time = time.time()

    # Last known values to keep streaming even when tracking is lost
    last_blendshape_dict = None
    last_eye_x, last_eye_y = 0.5, 0.5  # center by default
    last_jaw_open = 0.0

    # Smoothed blendshapes for jitter reduction
    smoothed_blendshape_dict = {}

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                loop_start = time.time()
                try:
                    # --- MAIN LOOP BODY ---

                    # Capture frame from Picamera2 (BGR, HxWx3)
                    frame = picam2.capture_array()
                    if frame is None or frame.size == 0:
                        print("[HEADLESS-PI] Empty frame from Picamera2, retrying…")
                        time.sleep(0.01)
                        continue

                    # Update latest frame for snapshot server
                    latest_frame = frame

                    # Convert BGR -> RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    # Timestamp in VIDEO mode (monotonic frame counter)
                    ts_ms = int(frame_index * (1000.0 / max(target_fps, 1e-6)))
                    frame_index += 1

                    result = landmarker.detect_for_video(mp_image, ts_ms)

                    blendshape_dict = None
                    eye_x, eye_y = last_eye_x, last_eye_y  # default to last eyes

                    if result and result.face_blendshapes:
                        # Fresh detection
                        blendshapes = result.face_blendshapes[0]

                        # Raw blendshape scores
                        raw_blendshape_dict = {
                            b.category_name: float(b.score) for b in blendshapes
                        }

                        # Subtract neutral baselines (if any)
                        blendshape_dict = {}
                        for name, score in raw_blendshape_dict.items():
                            baseline = neutral_blendshapes_baseline.get(name, 0.0)
                            adjusted = score - baseline
                            blendshape_dict[name] = max(adjusted, 0.0)

                        # --- ONE-EYE BLINK MIRROR: copy left blink → right blink ---
                        if "eyeBlinkLeft" in blendshape_dict:
                            left_blink = blendshape_dict["eyeBlinkLeft"]
                            if "eyeBlinkRight" in blendshape_dict:
                                blendshape_dict["eyeBlinkRight"] = left_blink

                        # Landmarks-driven custom metrics
                        if result.face_landmarks:
                            landmarks = result.face_landmarks[0]

                            lip_distance = np.linalg.norm([
                                landmarks[13].x - landmarks[14].x,
                                landmarks[13].y - landmarks[14].y,
                                landmarks[13].z - landmarks[14].z,
                            ])

                            lip_width = np.linalg.norm([
                                landmarks[61].x - landmarks[291].x,
                                landmarks[61].y - landmarks[291].y,
                                landmarks[61].z - landmarks[291].z,
                            ])

                            nostril_distance = np.linalg.norm([
                                landmarks[98].x - landmarks[327].x,
                                landmarks[98].y - landmarks[327].y,
                                landmarks[98].z - landmarks[327].z,
                            ])

                            # Use neutral raw baselines if present
                            if neutral_raw_baseline.get("neutral_lip_width") is not None:
                                neutral_lip_width = neutral_raw_baseline["neutral_lip_width"]
                            else:
                                if not neutral_captured:
                                    neutral_lip_width = lip_width

                            if neutral_raw_baseline.get("neutral_nostril_distance") is not None:
                                neutral_nostril_distance = neutral_raw_baseline["neutral_nostril_distance"]
                            else:
                                if not neutral_captured:
                                    neutral_nostril_distance = nostril_distance

                            neutral_captured = True

                            mouth_closed_raw = 1.0 - min(
                                lip_distance / max_mouth_open_distance, 1.0
                            )
                            jaw_open_score = raw_blendshape_dict.get("jawOpen", 0.0)
                            last_jaw_open = jaw_open_score
                            mouth_closed_score = mouth_closed_raw * (1.0 - jaw_open_score)

                            mouth_pucker_score = 1.0 - min(
                                lip_width / max(neutral_lip_width, 1e-6), 1.0
                            )
                            nose_sneer_score = min(
                                nostril_distance / max(neutral_nostril_distance, 1e-6),
                                1.0,
                            )

                            if "mouthClosed" in blendshape_dict:
                                blendshape_dict["mouthClosed"] = mouth_closed_score
                            if "mouthPucker" in blendshape_dict:
                                blendshape_dict["mouthPucker"] = mouth_pucker_score
                            if "noseSneerLeft" in blendshape_dict:
                                blendshape_dict["noseSneerLeft"] = nose_sneer_score
                            if "noseSneerRight" in blendshape_dict:
                                blendshape_dict["noseSneerRight"] = nose_sneer_score

                            # Iris (left eye), mirrored
                            left_iris = landmarks[468]
                            eye_x = left_iris.x
                            eye_y = left_iris.y
                            last_eye_x, last_eye_y = eye_x, eye_y

                        # Apply multipliers
                        if multipliers:
                            for name in list(blendshape_dict.keys()):
                                if name in multipliers:
                                    blendshape_dict[name] *= multipliers[name]

                        # Remember for when tracking is lost
                        last_blendshape_dict = blendshape_dict

                    elif last_blendshape_dict is not None:
                        # No detection this frame, reuse last valid values
                        blendshape_dict = last_blendshape_dict

                    else:
                        # No detection yet ever: send zeros
                        blendshape_dict = {name: 0.0 for name in BLENDSHAPE_STREAM_NAMES}
                        eye_x, eye_y = 0.5, 0.5
                        last_eye_x, last_eye_y = eye_x, eye_y
                        last_jaw_open = 0.0
                        last_blendshape_dict = blendshape_dict

                    # --- SMOOTHING: exponential moving average on blendshapes ---
                    if smoothed_blendshape_dict:
                        new_smoothed = {}
                        for name in BLENDSHAPE_STREAM_NAMES:
                            raw = float(blendshape_dict.get(name, 0.0))
                            prev = smoothed_blendshape_dict.get(name, raw)
                            smoothed = prev + SMOOTHING_ALPHA * (raw - prev)
                            new_smoothed[name] = smoothed
                        smoothed_blendshape_dict = new_smoothed
                    else:
                        # First frame: just take raw values
                        smoothed_blendshape_dict = dict(blendshape_dict)

                    # Replace raw with smoothed for sending
                    blendshape_dict = smoothed_blendshape_dict

                    # --- Now we ALWAYS have a blendshape_dict and eye_x/eye_y ---

                    # Head rotation disabled in helmet mode
                    pitch = 0.0
                    yaw = 0.0
                    roll = 0.0

                    py_face.set_blendshape(FaceBlendShape(51), 0.0)   # tongue (unused)
                    py_face.set_blendshape(FaceBlendShape(52), yaw)
                    py_face.set_blendshape(FaceBlendShape(53), pitch)
                    py_face.set_blendshape(FaceBlendShape(54), roll)

                    # --- ONE-EYE GAZE: left only ---
                    py_face.set_blendshape(FaceBlendShape(55), eye_x)  # eyeLeftX
                    py_face.set_blendshape(FaceBlendShape(56), eye_y)  # eyeLeftY
                    py_face.set_blendshape(FaceBlendShape(57), 0.0)    # eyeLeftZ (unused)

                    # Right eye gaze forced to neutral
                    py_face.set_blendshape(FaceBlendShape(58), 0.0)    # eyeRightX
                    py_face.set_blendshape(FaceBlendShape(59), 0.0)    # eyeRightY
                    py_face.set_blendshape(FaceBlendShape(60), 0.0)

                    # Copy blendshape_dict into PyLiveLinkFace slots
                    for i, name in enumerate(BLENDSHAPE_STREAM_NAMES):
                        score = float(blendshape_dict.get(name, 0.0))
                        py_face.set_blendshape(FaceBlendShape(i), score)

                    # Send every frame, regardless of detection
                    sock.sendall(py_face.encode())

                    # fps monitoring
                    now = time.time()
                    frame_times.append(now)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    if now - last_fps_print_time >= 1.0:
                        if len(frame_times) > 1:
                            fps_est = len(frame_times) / (frame_times[-1] - frame_times[0])
                            print(f"[HEADLESS-PI] Streaming… approx {fps_est:.1f} FPS, jawOpen={last_jaw_open:.3f}")
                        else:
                            print(f"[HEADLESS-PI] Streaming… jawOpen={last_jaw_open:.3f}")
                        last_fps_print_time = now

                    # FPS limiting
                    elapsed = time.time() - loop_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    # --- END MAIN LOOP BODY ---

                except KeyboardInterrupt:
                    # Let outer try/finally handle cleanup
                    raise
                except Exception as e:
                    print("[HEADLESS-PI][ERROR in loop]:", repr(e))
                    time.sleep(0.05)
                    continue

    except KeyboardInterrupt:
        print("\n[HEADLESS-PI] Stopping due to Ctrl+C…")
    finally:
        print("[HEADLESS-PI] Cleaning up...")
        try:
            snapshot_server.shutdown()
            snapshot_server.server_close()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
        sock.close()
        print("[HEADLESS-PI] Camera, socket, and HTTP server closed. Bye.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pi-native headless Mediapipe Face Landmarker streamer (Picamera2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--udp-address", "--udp", dest="udp_address", default="127.0.0.1")
    p.add_argument("--udp-port", "--port", dest="udp_port", type=int, default=11111)
    p.add_argument("--width", type=int, default=320, help="Camera width")
    p.add_argument("--height", type=int, default=240, help="Camera height")
    p.add_argument("--fps", type=float, default=15.0, help="Target FPS")
    p.add_argument("--snapshot-port", type=int, default=8000, help="HTTP port for /snapshot.jpg")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_headless_picam(
        udp_address=args.udp_address,
        udp_port=args.udp_port,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        snapshot_port=args.snapshot_port,
    )


if __name__ == "__main__":
    main()
