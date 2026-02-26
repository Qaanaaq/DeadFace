import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from picamera2 import Picamera2

# Global for the server
latest_frame = None

class SnapshotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame
        if self.path == "/snapshot.jpg" and latest_frame is not None:
            rotated = cv2.rotate(latest_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ret, jpeg = cv2.imencode(".jpg", rotated)
            if ret:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                return
        self.send_response(404)
        self.end_headers()

def run_setup(port=8000):
    global latest_frame
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()

    server = HTTPServer(("", port), SnapshotHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Setup Mode: View stream at http://<PI_IP>:{port}/snapshot.jpg")
    print("Press Ctrl+C to stop setup and release camera.")

    try:
        while True:
            latest_frame = picam2.capture_array()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Closing setup...")
    finally:
        server.shutdown()
        picam2.stop()

if __name__ == "__main__":
    run_setup()