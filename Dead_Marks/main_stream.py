import mediapipe as mp
import numpy as np
import cv2
import socket
import transforms3d
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from face_geometry import PCF, get_metric_landmarks
from mediapipe.framework.formats import landmark_pb2
from blendshape_utils import BLENDSHAPE_STREAM_NAMES
import json
import os


os.environ["GLOG_minloglevel"] = "2"      # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow C++ logs: 1=WARNING,2=ERROR,3=FATAL

model_path = "deadface.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

global_stream_result = None
previous_blendshapes = None

blendshape_names = BLENDSHAPE_STREAM_NAMES
neutral_pose_data = None
neutral_blendshapes_baseline = {}
neutral_custom_baseline = {}
neutral_raw_baseline = {}

def load_multipliers():
    if os.path.exists("multipliers.json"):
        with open("multipliers.json", "r") as f:
            return json.load(f)
    return {}

multipliers = load_multipliers()

def load_neutral_pose():
    global neutral_pose_data, neutral_blendshapes_baseline, neutral_custom_baseline, neutral_raw_baseline
    try:
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

load_neutral_pose()


class CameraStreamRunner:
    def __init__(self, udp_address="127.0.0.1", udp_port=11111):
        self.udp_address = udp_address
        self.udp_port = udp_port
        self.running = False
        self.filter = None   # Will hold Kalman filter
        self.filter_strength = 0.0  # 0 = no filter, 1 = max smoothing
        self.rotation_filter = None
        self.use_improved_shapes = False
        self.neutral_lip_distance = None
        self.curve_strength = 0.0



    def stop(self):
        self.running = False

    
    def set_filter_strength(self, value: float):
        if value > 0:
            alpha = 1 - value * 0.9
            try:
                from filter import BlendshapeEMAFilter, EMAFilter1D
            except ImportError:
                print("[WARN] filter module not found; smoothing disabled.")
                self.filter = None
                self.rotation_filter = None
                return
            self.filter = BlendshapeEMAFilter(alpha=alpha)
            self.rotation_filter = {
                "pitch": EMAFilter1D(alpha),
                "yaw": EMAFilter1D(alpha),
                "roll": EMAFilter1D(alpha)
            }
        else:
            self.filter = None
            self.rotation_filter = None



    def set_improved_shapes(self, enabled: bool):
        self.use_improved_shapes = enabled

    def set_curve_strength(self, value: float):
        # clamp to [-1, 1] just in case
        self.curve_strength = max(-1.0, min(1.0, float(value)))


    def run(self, display_callback=None):
        global global_stream_result
        global previous_blendshapes

        self.running = True
        cap = cv2.VideoCapture(0)

        py_face = PyLiveLinkFace()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((self.udp_address, self.udp_port))

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=stream_result_callback,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        pcf = PCF(
            near=1,
            far=10000,
            frame_height=480,
            frame_width=640,
            fy=640
        )

        # Normalization constants
        max_mouth_open_distance = 0.05
        neutral_lip_width = 0.05
        neutral_nostril_distance = 0.035
        neutral_captured = False

        with FaceLandmarker.create_from_options(options) as landmarker:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                # Process blendshapes and send
                if global_stream_result is not None and global_stream_result.face_blendshapes:
                    blendshapes = global_stream_result.face_blendshapes[0]
                    previous_blendshapes = blendshapes
                elif previous_blendshapes:
                    blendshapes = previous_blendshapes
                else:
                    blendshapes = None

                if blendshapes:
                    raw_blendshape_dict = {b.category_name: b.score for b in blendshapes}

                    # Subtract neutral baselines
                    blendshape_dict = {}
                    for name, score in raw_blendshape_dict.items():
                        baseline = neutral_blendshapes_baseline.get(name, 0.0)
                        adjusted = score - baseline
                        blendshape_dict[name] = max(adjusted, 0.0)

                    if global_stream_result and global_stream_result.face_landmarks:
                        landmarks = global_stream_result.face_landmarks[0]

                        # Compute distances
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

                        # Compute mouth corner distance
                        corner_distance = np.linalg.norm([
                            landmarks[61].x - landmarks[291].x,
                            landmarks[61].y - landmarks[291].y,
                            landmarks[61].z - landmarks[291].z
                        ])

                        # Custom scores computation
                        mouth_closed_raw = 1.0 - min(lip_distance / max_mouth_open_distance, 1.0)
                        jaw_open_score = raw_blendshape_dict.get("jawOpen", 0.0)
                        mouth_closed_score = mouth_closed_raw * (1.0 - jaw_open_score)

                        # Use neutral raw baselines
                        if neutral_raw_baseline.get("neutral_lip_width") is not None:
                            neutral_lip_width = neutral_raw_baseline["neutral_lip_width"]
                        else:
                            if not neutral_captured:
                                neutral_lip_width = lip_width
                                neutral_captured = True

                        if neutral_raw_baseline.get("neutral_nostril_distance") is not None:
                            neutral_nostril_distance = neutral_raw_baseline["neutral_nostril_distance"]
                        else:
                            if not neutral_captured:
                                neutral_nostril_distance = nostril_distance
                                neutral_captured = True   

                        # --- Iris tracking (left eye only) ---
                        left_iris = landmarks[468]  # Mediapipe Face Mesh iris landmark
                        # Map to eye slots: X/Y normalized, Z always 0
                        eye_data = [
                            left_iris.x, left_iris.y, 0,  # eyeLeftX, eyeLeftY, eyeLeftZ
                            left_iris.x, left_iris.y, 0   # eyeRightX, eyeRightY, eyeRightZ (mirrored for now)
                    ]

                        # Compute pose rotation
                        pose_matrix, _, _, _ = calculate_rotation(landmarks, pcf, mp_image)
                        eul = transforms3d.euler.mat2euler(pose_matrix)
                        pitch, yaw, roll = eul[0] + 0.3, -eul[1], eul[2]

                        # Apply smoothing to head rotation if filter is enabled
                        if self.rotation_filter:
                            pitch = self.rotation_filter["pitch"].update(pitch)
                            yaw = self.rotation_filter["yaw"].update(yaw)
                            roll = self.rotation_filter["roll"].update(roll)

                        py_face.set_blendshape(FaceBlendShape(51), 0)
                        py_face.set_blendshape(FaceBlendShape(52), yaw)
                        py_face.set_blendshape(FaceBlendShape(53), pitch)
                        py_face.set_blendshape(FaceBlendShape(54), roll)
                        py_face.set_blendshape(FaceBlendShape(55), left_iris.x)  # eyeLeftX
                        py_face.set_blendshape(FaceBlendShape(56), left_iris.y)  # eyeLeftY
                        py_face.set_blendshape(FaceBlendShape(57), 0)            # eyeLeftZ (unused)
                        py_face.set_blendshape(FaceBlendShape(58), left_iris.x)  # eyeRightX (mirrored)
                        py_face.set_blendshape(FaceBlendShape(59), left_iris.y)  # eyeRightY (mirrored)
                        py_face.set_blendshape(FaceBlendShape(60), 0)    

                                       

                    # --- Improved Shapes: ---
                    if self.use_improved_shapes:
                        # Compute mouth corner distance
                        corner_distance = np.linalg.norm([
                            landmarks[61].x - landmarks[291].x,
                            landmarks[61].y - landmarks[291].y,
                            landmarks[61].z - landmarks[291].z
                        ])

                        # Reduce pucker when mouth corners are far apart
                        if "mouthPucker" in blendshape_dict and corner_distance > 0.09:
                            factor = max(0.0, 1.0 - (corner_distance - 0.09) * 5)  # tweak multiplier as needed
                            blendshape_dict["mouthPucker"] *= factor

                        # Landmarks for upper/lower lips
                        upper_lip = landmarks[13]
                        lower_lip = landmarks[14]

                        # Euclidean distance between lips
                        lip_distance = np.linalg.norm([
                            upper_lip.x - lower_lip.x,
                            upper_lip.y - lower_lip.y,
                            upper_lip.z - lower_lip.z                            
                        ])

                        # Normalize relative to neutral distance (capture neutral_lip_distance at neutral pose)
                        if "jawOpen" in blendshape_dict:
                            jaw_open_value = blendshape_dict["jawOpen"]
                        else:
                            jaw_open_value = 0.0

                        if self.neutral_lip_distance is None:
                            self.neutral_lip_distance = lip_distance  # fallback if not set

                        lip_distance_normalized = max(0.0, 1.0 - (lip_distance / self.neutral_lip_distance))
                        mouth_closed_value = lip_distance_normalized * jaw_open_value
                        blendshape_dict["mouthClosed"] = mouth_closed_value
                        
                    # Apply curve
                    if self.curve_strength != 0:
                        for name in blendshape_dict:
                            x = blendshape_dict[name]
                            # Apply curve mapping
                            if self.curve_strength < 0:
                                # Steeper onset (fast rise)
                                factor = 1.0 / (1.0 - self.curve_strength)  # maps [-1,0)
                                blendshape_dict[name] = pow(x, factor)
                            elif self.curve_strength > 0:
                                # Steeper extreme (ease-in start)
                                factor = 1.0 + self.curve_strength
                                blendshape_dict[name] = pow(x, factor)
                            # 0 = linear, no change
                    

                    # Apply Kalman filter if enabled
                    if self.filter:
                        blendshape_dict = self.filter.apply(blendshape_dict)

                    # Apply multipliers
                    if multipliers:
                        for name in blendshape_dict:
                            if name in multipliers:
                                blendshape_dict[name] *= multipliers[name]

                    # Send blendshapes
                    for i, name in enumerate(blendshape_names):
                        score = blendshape_dict.get(name, 0.0)
                        py_face.set_blendshape(FaceBlendShape(i), score)

                    sock.sendall(py_face.encode())

                    # Draw annotated frame
                    annotated = draw_landmarks_on_image(rgb_frame, global_stream_result, override_blendshape_dict=blendshape_dict)

                    # Display callback
                    if display_callback:
                        display_callback(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        cap.release()
        sock.close()
        cv2.destroyAllWindows()


def stream_result_callback(result, output_image, timestamp_ms):
    global global_stream_result
    global_stream_result = result


def calculate_rotation(landmarks, pcf, mp_image):
    frame_width = mp_image.width
    frame_height = mp_image.height
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)

    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeff = np.zeros((4, 1))

    lm_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).T

    if lm_array.shape[1] > 468:
        lm_array = lm_array[:, :468]

    metric_landmarks, pose_transform_mat = get_metric_landmarks(lm_array.copy(), pcf)

    model_points = metric_landmarks[0:3, [1, 33, 263, 61, 291, 199]].T
    image_points = (
        lm_array[0:2, [1, 33, 263, 61, 291, 199]].T
        * np.array([frame_width, frame_height])[None, :]
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector


from mediapipe.python.solutions.drawing_utils import DrawingSpec
white_style = DrawingSpec(color=(254, 254, 254), thickness=None, circle_radius=1)
yellow_style = DrawingSpec(color=(255, 255, 0), thickness=None, circle_radius=1)


def draw_landmarks_on_image(rgb_image, detection_result, override_blendshape_dict=None):
    if not detection_result or not detection_result.face_landmarks:
        return rgb_image

    annotated = np.copy(rgb_image)

    for landmarks in detection_result.face_landmarks:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])

        # mp.solutions.drawing_utils.draw_landmarks(
        #     image=annotated,
        #     landmark_list=face_landmarks_proto,
        #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        # )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=white_style
        )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=yellow_style
        )

    if override_blendshape_dict is not None:
        items = list(override_blendshape_dict.items())
        total = len(items)
        half = total // 2

        right_texts = [f"{name}: {score:.3f}" for name, score in items[half:]]
        (text_width, _) = cv2.getTextSize(max(right_texts, key=len), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        x_right = annotated.shape[1] - text_width - 10

        for idx, (name, score) in enumerate(items):
            text = f"{name}: {score:.3f}"
            if idx < half:
                x = 10
                y = 20 + idx * 15
            else:
                x = x_right
                y = 20 + (idx - half) * 15
            cv2.putText(
                annotated,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )
    elif detection_result.face_blendshapes:
        blendshapes = detection_result.face_blendshapes[0]
        total = len(blendshapes)
        half = total // 2

        right_texts = [f"{b.category_name}: {b.score:.3f}" for b in blendshapes[half:]]
        (text_width, _) = cv2.getTextSize(max(right_texts, key=len), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        x_right = annotated.shape[1] - text_width - 10

        for idx, blendshape in enumerate(blendshapes):
            text = f"{blendshape.category_name}: {blendshape.score:.3f}"
            if idx < half:
                x = 10
                y = 20 + idx * 15
            else:
                x = x_right
                y = 20 + (idx - half) * 15
            cv2.putText(
                annotated,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    return annotated

def get_current_blendshapes():
    """Return the most recent face landmarks and blendshapes from the stream."""
    global global_stream_result
    if global_stream_result and global_stream_result.face_blendshapes:
        return global_stream_result.face_blendshapes[0], global_stream_result.face_landmarks[0]
    return None, None

def reload_neutral_pose():
    """Force reload of neutral baselines from JSON at runtime."""
    load_neutral_pose()
    print("Neutral pose calibrated.")
    # print("[INFO] Neutral pose reloaded during stream:", neutral_blendshapes_baseline)
    # if self.latest_landmarks is not None:  # ensure we have landmarks
    #     upper_lip = self.latest_landmarks[13]
    #     lower_lip = self.latest_landmarks[14]
    #     self.neutral_lip_distance = np.linalg.norm([
    #         upper_lip.x - lower_lip.x,
    #         upper_lip.y - lower_lip.y,
    #         upper_lip.z - lower_lip.z
    #     ])

def reload_multipliers():
    global multipliers
    multipliers = load_multipliers()
    # print("Multipliers reloaded.")



