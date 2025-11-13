import socket
import time
import sys

from pylivelinkface import PyLiveLinkFace, FaceBlendShape  # <-- import FaceBlendShape directly

UDP_IP = "0.0.0.0"
UDP_PORT = 11111   # change if needed

print(f"[RECEIVER] Listening on UDP {UDP_IP}:{UDP_PORT}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

packet_count = 0
last_time = time.time()
fps_print_time = time.time()

while True:
    try:
        data, addr = sock.recvfrom(2048)

        packet_count += 1
        current_time = time.time()

        # Print FPS every second
        if current_time - fps_print_time >= 1.0:
            fps = packet_count / (current_time - last_time)
            print(f"[RECEIVER] Packets: {packet_count}, approx FPS: {fps:.1f}")
            packet_count = 0
            last_time = current_time
            fps_print_time = current_time

        # Try decode
        ok, face = PyLiveLinkFace.decode(data)
        if ok:
            # Use the global enum, not face.FaceBlendShape
            jaw_open = face.get_blendshape(FaceBlendShape.JawOpen)
            print(f"[RECEIVER] Face OK, JawOpen={jaw_open:.3f}")
        else:
            print("[RECEIVER] Received data but no face found")

    except KeyboardInterrupt:
        print("\n[RECEIVER] Shutting down.")
        sys.exit(0)

    except Exception as e:
        print("[RECEIVER] Error:", e)
