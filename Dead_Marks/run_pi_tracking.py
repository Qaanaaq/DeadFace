import argparse
import time
import socket
import os
import cv2
import mediapipe as mp
import numpy as np

from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from blendshape_utils import BLENDSHAPE_STREAM_NAMES
from picamera2 import Picamera2

# Performance optimizations
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_PATH = "DeadFace.task"

def run_headless_tracking(udp_address, udp_port, width, height, target_fps):
    # Camera Setup
    picam2 = Picamera2()
    # Change this in your camera setup section
    cfg = picam2.create_video_configuration(
        main={"size": (256, 192), "format": "BGR888"},
        buffer_count=2 # Minimal buffering to prevent "lag" buildup
    )
    picam2.configure(cfg)
    picam2.start()
    
    # Networking
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((udp_address, udp_port))
    py_face = PyLiveLinkFace()

    # Mediapipe setup
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    )

    frame_index = 0
    last_valid_dict = {name: 0.0 for name in BLENDSHAPE_STREAM_NAMES}
    
    # Monitoring variables
    frame_times = []
    last_fps_print_time = time.time()
    current_jaw_open = 0.0

    print(f"[TRACKING] RAW MODE (No Smooth, No Multipliers, No Neutral Baseline)")
    print(f"[TRACKING] Sending to {udp_address}:{udp_port}")

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                loop_start = time.time()
                frame = picam2.capture_array()
                if frame is None: continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                ts_ms = int(frame_index * (1000.0 / target_fps))
                frame_index += 1
                
                result = landmarker.detect_for_video(mp_image, ts_ms)

                if result and result.face_blendshapes:
                    # Get raw scores directly
                    blendshape_dict = {b.category_name: float(b.score) for b in result.face_blendshapes[0]}
                    current_jaw_open = blendshape_dict.get("jawOpen", 0.0)
                    
                    # One-eye Mirror Logic (still included so right eye isn't dead)
                    if "eyeBlinkLeft" in blendshape_dict:
                        blendshape_dict["eyeBlinkRight"] = blendshape_dict["eyeBlinkLeft"]

                    last_valid_dict = blendshape_dict

                # Pack and Send
                for i, name in enumerate(BLENDSHAPE_STREAM_NAMES):
                    score = float(last_valid_dict.get(name, 0.0))
                    py_face.set_blendshape(FaceBlendShape(i), score)
                
                # Zero out head rotation (Helmet Mode)
                for i in range(52, 55): 
                    py_face.set_blendshape(FaceBlendShape(i), 0.0)

                sock.sendall(py_face.encode())

                # --- FPS & STATUS OUTPUT ---
                now = time.time()
                frame_times.append(now)
                if len(frame_times) > 30: frame_times.pop(0)
                
                if now - last_fps_print_time >= 1.0:
                    fps_est = len(frame_times) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
                    print(f"[TRACKING] {fps_est:.1f} FPS | jawOpen: {current_jaw_open:.3f}")
                    last_fps_print_time = now

                # Sleep to maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / target_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping tracker...")
        finally:
            picam2.stop()
            sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--udp-address", "--udp", default="127.0.0.1")
    parser.add_argument("--udp-port", "--port", type=int, default=11111)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=15.0)
    
    args = parser.parse_args()
    run_headless_tracking(args.udp_address, args.udp_port, args.width, args.height, args.fps)