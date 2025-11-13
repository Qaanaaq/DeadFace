def run_tracking(video_path, head_tracking=True, symmetrical_eyes=False, display_callback=None, stop_event=None):
    import cv2
    import numpy as np
    import mediapipe as mp
    import transforms3d
    from face_geometry import PCF

    from mediapipe_utils import create_face_landmarker, calculate_rotation, get_head_euler_angles
    from drawing_utils import draw_landmarks_on_image
    from csv_utils import initialize_csv, write_blendshape_row, close_csv
    from video_utils import open_video, format_timecode, get_output_csv_path
    from blendshape_utils import BLENDSHAPE_NAMES

    headbool = bool(head_tracking)
    eyesymmetry = not bool(symmetrical_eyes)

    cap, frame_rate, frame_count, _ = open_video(video_path)

    output_csv_path = get_output_csv_path(video_path)

    landmarker = create_face_landmarker()

    blendshape_names = BLENDSHAPE_NAMES

    csv_file, writer = initialize_csv(output_csv_path, blendshape_names)

    file_name = video_path.split("/")[-1]

    # Default normalization constants (tune as needed)
    max_mouth_open_distance = 0.05
    neutral_lip_width = 0.05
    neutral_nostril_distance = 0.035

    # Flags to auto-capture neutral widths from first frame
    neutral_captured = False

    while cap.isOpened():
        # ---- add this stop check ----
        if stop_event is not None and stop_event.is_set():
            break
        # -----------------------------

        ret, frame = cap.read()
        if not ret:
            break



        frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
        time_formatted = format_timecode(frame_index, frame_rate)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_index * (1000 / frame_rate)))

        if not face_landmarker_result.face_blendshapes:
            continue

        face_blendshapes = face_landmarker_result.face_blendshapes[0]

        blendshape_scores = [f"{b.score:.8f}" for b in face_blendshapes[1:]]
        new_order = [8,10,12,14,16,18,20,9,11,13,15,17,19,21,22,25,23,24,26,31,37,38,32,43,44,29,30,27,28,45,46,39,40,41,42,35,36,33,34,47,48,0,1,2,3,4,5,6,7,49,50]
        blendshape_scores_sorted = [blendshape_scores[i] for i in new_order]

        tongue = [0]

        pcf = PCF(
            near=1,
            far=10000,
            frame_height=mp_image.height,
            frame_width=mp_image.width,
            fy=mp_image.width
        )
        pose_transform_mat, _, _, _ = calculate_rotation(face_landmarker_result, pcf, mp_image)
        pitch, yaw, roll = get_head_euler_angles(pose_transform_mat, transforms3d)
        headrotation = [pitch, yaw, roll] if headbool else [0,0,0]

        landmarks = face_landmarker_result.face_landmarks[0]
        left_iris = landmarks[468]
        right_iris = landmarks[473]

        if eyesymmetry:
            eyes = [left_iris.x, left_iris.y, 0, right_iris.x, right_iris.y, 0]
        else:
            eyes = [left_iris.x, left_iris.y, 0] * 2

        # ========================================
        # Custom landmark-based expression scores
        # ========================================

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

        # Optionally auto-capture neutral values
        if not neutral_captured:
            neutral_lip_width = lip_width
            neutral_nostril_distance = nostril_distance
            neutral_captured = True

        # Normalize
        mouth_closed_score = 1.0 - min(lip_distance / max_mouth_open_distance, 1.0)
        mouth_pucker_score = 1.0 - min(lip_width / neutral_lip_width, 1.0)
        nose_sneer_score = min(nostril_distance / neutral_nostril_distance, 1.0)

        # Overwrite the correct indices
        blendshape_scores_sorted[14] = f"{mouth_closed_score:.8f}"
        blendshape_scores_sorted[36] = f"{mouth_pucker_score:.8f}"
        blendshape_scores_sorted[29] = f"{nose_sneer_score:.8f}"
        blendshape_scores_sorted[30] = f"{nose_sneer_score:.8f}"

        write_blendshape_row(
            writer,
            time_formatted,
            len(face_blendshapes[1:]),
            blendshape_scores_sorted,
            tongue,
            headrotation,
            eyes
        )

        annotated = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result)

        text = f"{time_formatted} {file_name}"
        cv2.putText(annotated, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1)

        if display_callback:
            try:
                display_callback(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), int(frame_index))
            except TypeError:
                # fallback for older one-arg callbacks
                display_callback(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


    close_csv(csv_file)
    cap.release()
