import socket
import threading
import time
import io

import requests
from PIL import Image, ImageTk
import customtkinter as ctk


# ---------------- CONFIG ----------------

# Incoming streams from 3 RPis
STREAMS = [
    {
        "name": "Stream 1",
        "listen_port": 11111,
        "forward_port": 11121,
        "pi_snapshot_url": "http://192.168.0.213:8000/snapshot.jpg",
    },
    {
        "name": "Stream 2",
        "listen_port": 11112,
        "forward_port": 11122,
        "pi_snapshot_url": "http://192.168.0.102:8000/snapshot.jpg",
    },
    {
        "name": "Stream 3",
        "listen_port": 11113,
        "forward_port": 11123,
        "pi_snapshot_url": "http://192.168.0.103:8000/snapshot.jpg",
    },
]

# Default forward IP (per-stream can be changed in the UI)
FORWARD_IP_DEFAULT = "127.0.0.1"

# ----------------------------------------


class StreamRelay:
    """
    One relay:
      - listens on listen_port for UDP packets
      - optionally forwards them to forward_ip:forward_port
    """

    def __init__(self, name, listen_port, forward_port, forward_ip):
        self.name = name
        self.listen_port = listen_port
        self.forward_port = forward_port
        self.forward_ip = forward_ip

        self.forward_enabled = False
        self.running = True

        self.sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_in.bind(("", self.listen_port))
        self.sock_in.settimeout(0.5)

        self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        self.packet_count = 0
        self.last_fps_time = time.time()

    def _loop(self):
        print(
            f"[{self.name}] Listening on UDP port {self.listen_port}, "
            f"forwarding to {self.forward_ip}:{self.forward_port}"
        )
        while self.running:
            try:
                data, addr = self.sock_in.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.name}] recv error:", repr(e))
                continue

            self.packet_count += 1

            # Forward only if enabled
            if self.forward_enabled:
                try:
                    self.sock_out.sendto(data, (self.forward_ip, self.forward_port))
                except Exception as e:
                    print(f"[{self.name}] forward error:", repr(e))

            # Simple internal FPS logging (console)
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                fps = self.packet_count / max(now - self.last_fps_time, 1e-6)
                print(
                    f"[{self.name}] Incoming approx {fps:.1f} FPS "
                    f"(forwarding={'ON' if self.forward_enabled else 'OFF'})"
                )
                self.packet_count = 0
                self.last_fps_time = now

        print(f"[{self.name}] Relay loop stopped.")

    def set_forwarding(self, enabled: bool):
        self.forward_enabled = enabled
        print(f"[{self.name}] Forwarding set to {enabled}")

    def update_forward_target(self, ip: str, port: int):
        """Update where this relay forwards packets."""
        self.forward_ip = ip
        self.forward_port = port
        print(f"[{self.name}] Forward target set to {self.forward_ip}:{self.forward_port}")

    def stop(self):
        self.running = False
        try:
            self.sock_in.close()
        except Exception:
            pass
        try:
            self.sock_out.close()
        except Exception:
            pass


class ReceiverStudioApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Receiver Studio – DeadFace RPi Streams")
        # Slightly wider to fit the new fields
        self.root.geometry("1100x650")
        self.root.resizable(False, False)

        # layout: 2 columns → controls | canvas
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.relays = []
        self.controls = {}  # per-stream UI elements
        # mutable copy of configs (in case you want to tweak snapshot URLs later)
        self.stream_configs = [dict(s) for s in STREAMS]

        # Shared snapshot canvas (right side)
        self.canvas = ctk.CTkCanvas(
            self.root,
            width=640,
            height=480,
            bg="#111111",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.snapshot_photo = None  # keep reference

        # Left side controls
        control_frame = ctk.CTkFrame(self.root, corner_radius=10)
        control_frame.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)
        control_frame.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            control_frame,
            text="Streams",
            font=("Segoe UI", 18, "bold")
        )
        title_label.grid(row=0, column=0, padx=5, pady=(10, 10), sticky="w")

        for idx, stream in enumerate(self.stream_configs):
            row = idx + 1

            frame = ctk.CTkFrame(control_frame, corner_radius=8)
            frame.grid(row=row, column=0, padx=5, pady=5, sticky="ew")
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=1)

            label = ctk.CTkLabel(
                frame,
                text=stream["name"],
                font=("Segoe UI", 14, "bold"),
                anchor="w"
            )
            label.grid(row=0, column=0, columnspan=2, padx=8, pady=(6, 2), sticky="w")

            # --- Forward target controls (IP + port) ---

            ip_label = ctk.CTkLabel(frame, text="Forward IP:", anchor="w")
            ip_label.grid(row=1, column=0, padx=6, pady=(2, 0), sticky="w")

            ip_entry = ctk.CTkEntry(frame)
            ip_entry.insert(0, FORWARD_IP_DEFAULT)
            ip_entry.grid(row=1, column=1, padx=6, pady=(2, 0), sticky="ew")

            port_label = ctk.CTkLabel(frame, text="Forward port:", anchor="w")
            port_label.grid(row=2, column=0, padx=6, pady=(2, 0), sticky="w")

            port_entry = ctk.CTkEntry(frame)
            port_entry.insert(0, str(stream["forward_port"]))
            port_entry.grid(row=2, column=1, padx=6, pady=(2, 0), sticky="ew")

            # --- Start/stop + snapshot buttons ---

            start_stop_btn = ctk.CTkButton(
                frame,
                text="START FORWARD",
                command=lambda i=idx: self.toggle_forward(i),
            )
            start_stop_btn.grid(row=3, column=0, padx=6, pady=(8, 6), sticky="ew")

            snapshot_btn = ctk.CTkButton(
                frame,
                text="SNAPSHOT",
                command=lambda i=idx: self.take_snapshot(i),
            )
            snapshot_btn.grid(row=3, column=1, padx=6, pady=(8, 6), sticky="ew")

            # Apply button to update forward IP/port
            apply_btn = ctk.CTkButton(
                frame,
                text="Apply IP/Port",
                command=lambda i=idx: self.apply_forward_settings(i),
            )
            apply_btn.grid(row=4, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="ew")

            self.controls[idx] = {
                "start_stop": start_stop_btn,
                "ip_entry": ip_entry,
                "port_entry": port_entry,
            }

        # Create relays with default IP/ports
        for stream in self.stream_configs:
            relay = StreamRelay(
                name=stream["name"],
                listen_port=stream["listen_port"],
                forward_port=stream["forward_port"],
                forward_ip=FORWARD_IP_DEFAULT,
            )
            self.relays.append(relay)

        # Clean shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def apply_forward_settings(self, index: int):
        """Read IP + port from the UI and apply to the relay."""
        relay = self.relays[index]
        ctrl = self.controls[index]

        ip = ctrl["ip_entry"].get().strip() or FORWARD_IP_DEFAULT
        port_text = ctrl["port_entry"].get().strip()

        try:
            port = int(port_text)
        except ValueError:
            print(f"[{relay.name}] Invalid port: {port_text!r}")
            return

        relay.update_forward_target(ip, port)

    def toggle_forward(self, index: int):
        relay = self.relays[index]
        btn = self.controls[index]["start_stop"]

        if relay.forward_enabled:
            relay.set_forwarding(False)
            btn.configure(text="START FORWARD")
        else:
            relay.set_forwarding(True)
            btn.configure(text="STOP FORWARD")

    def take_snapshot(self, index: int):
        """
        Ask the corresponding RPi for a snapshot image and display it on the shared canvas.
        Assumes the Pi exposes an HTTP endpoint returning a JPEG, like:
            http://PI_IP:8000/snapshot.jpg
        """
        url = self.stream_configs[index]["pi_snapshot_url"]
        print(f"[SNAPSHOT] Requesting image from {url} ...")

        def worker():
            try:
                resp = requests.get(url, timeout=3)
                resp.raise_for_status()
                img_data = resp.content

                image = Image.open(io.BytesIO(img_data))
                # Resize to canvas size (simple fit)
                image = image.resize((640, 480), Image.LANCZOS)

                photo = ImageTk.PhotoImage(image)

                def update_canvas():
                    self.snapshot_photo = photo  # keep ref
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor="nw", image=self.snapshot_photo)
                    print("[SNAPSHOT] Updated canvas.")

                self.root.after(0, update_canvas)

            except Exception as e:
                print("[SNAPSHOT] Error fetching snapshot:", repr(e))

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        print("[Studio] Shutting down...")
        for relay in self.relays:
            relay.stop()
        self.root.destroy()


if __name__ == "__main__":
    # Match DualApp's theme
    ctk.set_appearance_mode("dark")
    # Make sure sky_dark_theme.json is in the same folder as this script,
    # or give an absolute/relative path if it's elsewhere.
    try:
        ctk.set_default_color_theme("sky_dark_theme.json")
    except Exception as e:
        print("Warning: could not load sky_dark_theme.json theme:", e)

    root = ctk.CTk()
    app = ReceiverStudioApp(root)
    root.mainloop()
