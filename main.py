import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load models
drowsiness_model = load_model('drowsiness_model.h5')
gesture_model = load_model('gesture_model.h5')

gesture_labels = ['Call_Pickup', 'fist', 'stop', 'thumb_down', 'thumb_up']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load Haarcascade for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gesture_action_done = {'accept': False, 'reject': False}
drowsy_alert_done = False

# Volume control functions
def get_current_volume():
    output = os.popen("osascript -e 'output volume of (get volume settings)'").read()
    try:
        return int(output.strip())
    except:
        return 0

def volume_up_mac():
    os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 10)'")

def volume_down_mac():
    os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 10)'")

# Drowsiness detection frame counter
closed_frames = 0
closed_frames_threshold = 12  # decreased for faster detection

# Start webcam
cap = cv2.VideoCapture(0)
print("🔴 Starting Driver Monitoring System (Press ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------- DROWSINESS DETECTION ----------------------
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    eyes_closed_count = 0

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes[:2]:  # take first 2 eyes detected
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (64, 64)) / 255.0
            eye_img = np.expand_dims(eye_img, axis=-1)
            eye_img = np.expand_dims(eye_img, axis=0)

            eye_pred = drowsiness_model.predict(eye_img, verbose=0)[0][0]

            if eye_pred < 0.25:  # more sensitive now
                eyes_closed_count += 1

    if eyes_closed_count >= 2:
        closed_frames += 1
    else:
        closed_frames = 0

    # -------------------- GESTURE DETECTION ----------------------
    result = hands.process(rgb_frame)
    gesture_label = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * width) - 20
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * height) - 20
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * width) + 20
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * height) + 20

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                try:
                    resized_hand = cv2.resize(hand_img, (64, 64)) / 255.0
                    resized_hand = np.expand_dims(resized_hand, axis=0)
                    prediction = gesture_model.predict(resized_hand, verbose=0)
                    gesture_idx = np.argmax(prediction)
                    gesture_label = gesture_labels[gesture_idx]
                except:
                    gesture_label = "None"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # -------------------- PERFORM GESTURE ACTIONS ----------------------
    if gesture_label == 'Call_Pickup':
        if not gesture_action_done['accept']:
            print("✅ Accepting Call")
            gesture_action_done['accept'] = True
            gesture_action_done['reject'] = False

    elif gesture_label == 'fist':
        pass  # No action for fist

    elif gesture_label == 'stop':
        if not gesture_action_done['reject']:
            print("❌ Rejecting Call")
            gesture_action_done['reject'] = True
            gesture_action_done['accept'] = False

    elif gesture_label == 'thumb_down':
        print("🔉 Decreasing Volume")
        volume_down_mac()

    elif gesture_label == 'thumb_up':
        print("🔊 Increasing Volume")
        volume_up_mac()

    else:
        gesture_action_done['accept'] = False
        gesture_action_done['reject'] = False

    # -------------------- DISPLAY INFO ----------------------
    volume_level = get_current_volume()

    cv2.putText(frame, f"Volume: {volume_level}%", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if gesture_label != "None":
        cv2.putText(frame, f"Gesture: {gesture_label}", (width-280, height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if closed_frames >= closed_frames_threshold:
        if not drowsy_alert_done:
            print("😴 Drowsiness detected!")
            drowsy_alert_done = True

        text = "😴 DROWSINESS ALERT!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, (0, 0, 255), thickness)
    else:
        drowsy_alert_done = False

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
