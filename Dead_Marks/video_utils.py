import cv2
import tkinter as tk
from tkinter import filedialog
import os

# Show file dialog and return selected video file path
def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File")
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return file_path

# Open video and get metadata
def open_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / frame_rate

    return cap, frame_rate, frame_count, duration

# Format timestamp string
def format_timecode(frame_index, frame_rate):
    frame_timestamp_ms = int(frame_index * (1000 / frame_rate))
    milliseconds = int((frame_index) % frame_rate)
    seconds = int((frame_timestamp_ms / 1000) % 60)
    minutes = int((frame_timestamp_ms / (1000 * 60)) % 60)
    hours = int(frame_timestamp_ms / (1000 * 60 * 60))
    frame_index_formatted = int(frame_index % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:02}.{frame_index_formatted:03d}"

# Generate output CSV path
def get_output_csv_path(file_path):
    base, _ = os.path.splitext(file_path)
    return base + "_blendshape_data.csv"
