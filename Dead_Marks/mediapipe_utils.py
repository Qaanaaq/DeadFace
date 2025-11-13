import mediapipe as mp
import numpy as np
import cv2
from face_geometry import PCF, get_metric_landmarks

# MediaPipe Face Landmarker setup
def create_face_landmarker(model_path="DeadFace.task"):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True
    )
    return FaceLandmarker.create_from_options(options)

# Calculate rotation vectors and metric landmarks
def calculate_rotation(face_landmarks, pcf, image_shape):
    frame_width = image_shape.width
    frame_height = image_shape.height
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double"
    )

    dist_coeff = np.zeros((4, 1))

    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_landmarks.face_landmarks[0][:468]]
    ).T

    metric_landmarks, pose_transform_mat = get_metric_landmarks(
        landmarks.copy(), pcf
    )

    # Selected landmark indices for SolvePnP
    points_idx = [1, 33, 263, 61, 291, 199]
    model_points = metric_landmarks[0:3, points_idx].T
    image_points = (
        landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None, :]
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector

# Compute head Euler angles from pose matrix
def get_head_euler_angles(pose_transform_mat, transforms3d_module):
    euler_angles = transforms3d_module.euler.mat2euler(pose_transform_mat)
    pitch = -euler_angles[0]
    yaw = euler_angles[1]
    roll = euler_angles[2]
    return pitch, yaw, roll
