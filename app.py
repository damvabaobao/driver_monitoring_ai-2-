import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
import time
import os
import csv
import joblib
import pandas as pd

# CONFIG

EAR_THRESHOLD = 0.25

MAR_THRESHOLD = 0.55

# Number
EYE_FRAMES = 90
YAWN_FRAMES = 15

HEAD_DOWN_THRESHOLD = 12
HEAD_SIDE_THRESHOLD = 18

ALPHA = 0.25

ALERT_COOLDOWN = 3  # seconds

DROWSY_THRESHOLD = 0.75

# LANDMARKS

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# STATE

eye_counter = 0
yawn_counter = 0

pitch_smooth = 0
yaw_smooth = 0

recording = False
current_label = None
sample_counter = 0
SAVE_EVERY_N_FRAMES = 5


last_alert_time = 0
sound_playing = False

saved_samples = 0

drowsy_counter = 0
DROWSY_CONFIRM_FRAMES = 30

face_lost_start = None
FACE_LOST_TIME = 3

probability = ([1.0, 0.0])
confirmed_drowsy = False

color = (0, 255, 0)
prob_smooth = 0

# AUDIO INIT

pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r"D:\driver_monitoring_ai\alarm.wav")

#model = joblib.load("driver_drowsiness_rf.pkl")
model = joblib.load("driver_drowsiness_xgb.pkl")
print("AI model loaded!!!!!")

# FUNCTIONS
def calculate_ear(points):
    p1, p2, p3, p4, p5, p6 = points

    vertical_a = distance.euclidean(p2, p6)
    vertical_b = distance.euclidean(p3, p5)
    horizontal = distance.euclidean(p1, p4)

    if horizontal == 0:
        return 0





    return (vertical_a + vertical_b) / (2.0 * horizontal)

def calculate_mar(points):
    p1, p2, p3, p4, p5, p6, p7, p8 = points
    v1 = distance.euclidean(p3, p6)
    v2 = distance.euclidean(p4, p5)
    h = distance.euclidean(p1, p2)
    if h == 0:
        return 0
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

def save_sample(ear, mar, pitch, yaw, label):
    global saved_samples

    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow ([
            round(ear, 4),
            round(mar, 4),
            round(pitch, 4),
            round(yaw, 4),
            label
        ])

    saved_samples += 1

# INIT MEDIAPIPE

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# CAMERA

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera error")
    exit()

# DATASET
DATASET_FILE = "driver_dataset.csv"
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EAR",
            "MAR",
            "Pitch",
            "Yaw",
            "Label"
        ])

# LOOP

ear = 0
mar = 0
pitch = 0
yaw = 0

while True:

    face_detected = False

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "AWAKE"

    if results.multi_face_landmarks:
        face_detected = True
        face_lost_start = None

        for face in results.multi_face_landmarks:

            lm = face.landmark

            # EYE

            left = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            right = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]

            ear = (calculate_ear(left) + calculate_ear(right)) / 2

            # MOUTH
            
            mouth = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH]
            mar = calculate_mar(mouth)

            # HEAD POSE
            pitch, yaw, roll = get_head_pose(lm, w, h)

            pitch = pitch * (1 - ALPHA) + pitch_smooth * ALPHA
            yaw = yaw * (1 - ALPHA) + yaw_smooth * ALPHA

            pitch_smooth = pitch
            yaw_smooth = yaw

            yaw = -yaw

            sample = pd.DataFrame([[
                ear,
                mar,
                pitch,
                yaw
            ]], columns = ["EAR", "MAR", "Pitch", "Yaw"])
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0]

            prob_smooth = 0.8 * prob_smooth + 0.2 * probability[1]

            # LOGIC
            
            eye_counter = eye_counter + 1 if ear < EAR_THRESHOLD else 0
            yawn_counter = yawn_counter + 1 if mar > MAR_THRESHOLD else 0

            drowsy = eye_counter > EYE_FRAMES
            yawning = yawn_counter > YAWN_FRAMES

            head_down = pitch > HEAD_DOWN_THRESHOLD
            distracted = abs(yaw) > HEAD_SIDE_THRESHOLD

            if prediction == 0:
                status = "AWAKE:Tinh"
            else:
                status = "DROWSY:Buon ngu"
            
            if prob_smooth > DROWSY_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = 0
            confirmed_drowsy = (
                drowsy_counter >= DROWSY_CONFIRM_FRAMES
            )
            if confirmed_drowsy:
                if time.time() - last_alert_time > ALERT_COOLDOWN:
                    alert_sound.play()
                    last_alert_time = time.time()
            
            if prob_smooth > DROWSY_THRESHOLD:
                color = (0, 0, 255) # Red
            elif prob_smooth > 0.5:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 255, 0) # Green
    else:
        prob_smooth = 0
        drowsy_counter = 0
        confirmed_drowsy = False
        if face_lost_start is None:
            face_lost_start = time.time()
        lost_duration = time.time() - face_lost_start
        if lost_duration >= FACE_LOST_TIME:
            cv2.putText(
                frame,
                "FACE NOT DETECTED!",
                (30, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            if time.time() - last_alert_time > ALERT_COOLDOWN:
                alert_sound.play()
                last_alert_time = time.time()
        cv2.putText(frame, f"Face Lost: {lost_duration:.1f}s", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)

    # DISPLAY
    cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            
    cv2.putText(frame, f"Drowsy Probability: {prob_smooth*100:.1f}%", (30, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    cv2.putText(frame, f"Drowsy Prob: {prob_smooth*100:.1f}%", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if confirmed_drowsy:
        status = "HIGH DROWSINESS"
        cv2.putText(frame, "!!!ALERT!!!", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"Status: {status}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # DATA COLLECTION
    if recording and current_label is not None and face_detected:
        sample_counter += 1
        if sample_counter >= SAVE_EVERY_N_FRAMES:
            save_sample(
                ear,
                mar,
                pitch,
                yaw,
                current_label)
            print(
                f"Saved: EAR={ear:.2f}, "
                f"MAR={mar:.2f}, "
                f"PITCH={pitch:.2f},"
                f"YAW={yaw:.2f},"
                f"Label={current_label}"
            )
            sample_counter = 0
    record_text = "NOT RECORDING"
    if recording:
        if current_label == 0:
            record_text = "RECORDING AWAKE"
        elif current_label == 1:
            record_text = "RECORDING DROWSY"
    cv2.putText( frame, record_text, (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText( frame, "A=Awake D=Drowsy S=Stop", (30, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText( frame, f"Samples: {saved_samples}", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Face: {face_detected}", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Driver Monitoring AI - Step 4", frame)
    key = cv2.waitKey(1) & 0xFF
    # Awake
    if key == ord('a'):
        recording = True
        current_label = 0

        print("Recording AWAKE")
    # Drowsy
    elif key == ord('d'):
        recording = True
        current_label = 1

        print("Recording DROWSY")
    # Stop
    elif key == ord('s'):
        recording = False
        current_label = None

        print("Recording STOPPED")
    # ESC
    elif key == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()