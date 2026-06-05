import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
import time

# =========================================================
# CONFIG
# =========================================================

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.55

# Number
EYE_FRAMES = 20
YAWN_FRAMES = 15

HEAD_DOWN_THRESHOLD = 12
HEAD_SIDE_THRESHOLD = 18

ALPHA = 0.25

ALERT_COOLDOWN = 3  # seconds

# =========================================================
# LANDMARKS
# =========================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# =========================================================
# STATE
# =========================================================

eye_counter = 0
yawn_counter = 0

pitch_smooth = 0
yaw_smooth = 0

last_alert_time = 0
sound_playing = False

# =========================================================
# AUDIO INIT
# =========================================================

pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r"D:\driver_monitoring_ai\alarm.wav")

# =========================================================
# FUNCTIONS
# =========================================================

def calculate_ear(points):
    p1, p2, p3, p4, p5, p6 = points
    v1 = distance.euclidean(p2, p6)
    v2 = distance.euclidean(p3, p5)
    h = distance.euclidean(p1, p4)
    return (v1 + v2) / (2.0 * h)


def calculate_mar(points):
    p1, p2, p3, p4, p5, p6, p7, p8 = points
    v1 = distance.euclidean(p3, p6)
    v2 = distance.euclidean(p4, p5)
    h = distance.euclidean(p1, p2)
    return (v1 + v2) / (2.0 * h)


def get_head_pose(landmarks, w, h):

    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h),
        (landmarks[199].x * w, landmarks[199].y * h)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (-60.0, 40.0, -30.0),
        (60.0, 40.0, -30.0),
        (0.0, 70.0, -40.0)
    ])

    focal = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1]
    ], dtype="double")

    _, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1))
    )

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles[0], angles[1], angles[2]


# =========================================================
# INIT MEDIAPIPE
# =========================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("      Camera       error")
    exit()

# =========================================================
# LOOP
# =========================================================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "AWAKE"

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            lm = face.landmark

            # -------------------------
            # EYE
            # -------------------------
            left = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            right = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]

            ear = (calculate_ear(left) + calculate_ear(right)) / 2

            # -------------------------
            # MOUTH
            # -------------------------
            mouth = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH]
            mar = calculate_mar(mouth)

            # -------------------------
            # HEAD POSE
            # -------------------------
            pitch, yaw, roll = get_head_pose(lm, w, h)

            pitch = pitch * (1 - ALPHA) + pitch_smooth * ALPHA
            yaw = yaw * (1 - ALPHA) + yaw_smooth * ALPHA

            pitch_smooth = pitch
            yaw_smooth = yaw

            yaw = -yaw

            # -------------------------
            # LOGIC
            # -------------------------
            eye_counter = eye_counter + 1 if ear < EAR_THRESHOLD else 0
            yawn_counter = yawn_counter + 1 if mar > MAR_THRESHOLD else 0

            drowsy = eye_counter > EYE_FRAMES
            yawning = yawn_counter > YAWN_FRAMES

            head_down = pitch > HEAD_DOWN_THRESHOLD
            distracted = abs(yaw) > HEAD_SIDE_THRESHOLD

            # -------------------------
            # FUSION SCORE
            # -------------------------
            score = 0
            if drowsy: score += 2
            if yawning: score += 1
            if head_down: score += 2
            if distracted: score += 1

            if score >= 4:
                status = "HIGH DROWSINESS"
            elif score >= 2:
                status = "DROWSY"
            elif yawning:
                status = "YAWNING"
            elif distracted:
                status = "DISTRACTED"

            # -------------------------
            # AUDIO ALERT
            # -------------------------
            if score >= 4:
                if time.time() - last_alert_time > ALERT_COOLDOWN:
                    alert_sound.play()
                    last_alert_time = time.time()
                    sound_playing = True
            else:
                sound_playing = False

            # -------------------------
            # DISPLAY
            # -------------------------
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, f"Score: {score}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Status: {status}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if status == "HIGH DROWSINESS":
                cv2.putText(frame, "!!! ALERT !!!", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Driver Monitoring AI - Step 4", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()