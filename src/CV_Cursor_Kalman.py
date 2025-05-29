import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen and Camera Parameters
cam_width, cam_height = 640, 480  # Webcam Resolution
screen_width, screen_height = pyautogui.size()  # Screen Resolution
margin = 50  # Define margin to restrict movement area

# Kalman Filter for Cursor Smoothing
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

kf = KalmanFilter()

# Function to calculate distance between two landmarks
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Capture Video
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Finger Tips
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert to Screen Coordinates (Scaled & Restricted)
            x = np.interp(index_tip.x, [margin/w, (w-margin)/w], [0, screen_width])
            y = np.interp(index_tip.y, [margin/h, (h-margin)/h], [0, screen_height])
            index_x, index_y = kf.predict(x, y)

            # Move Cursor
            pyautogui.moveTo(index_x, index_y, duration=0.02)

            # Left Click (Index & Middle Finger Close Together)
            if distance(index_tip, middle_tip) < 0.05:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right Click (Thumb & Middle Finger Touch)
            if distance(thumb_tip, middle_tip) < 0.05:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Scroll Down (Ring, Pinky, Thumb Together)
            if distance(ring_tip, pinky_tip) < 0.05 and distance(pinky_tip, thumb_tip) < 0.05:
                pyautogui.scroll(-20)
                cv2.putText(frame, "Scroll Down", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Scroll Up (All fingers except index touch)
            if distance(middle_tip, ring_tip) < 0.05 and distance(ring_tip, pinky_tip) < 0.05 and distance(pinky_tip, thumb_tip) < 0.05:
                pyautogui.scroll(20)
                cv2.putText(frame, "Scroll Up", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Zoom In (Thumb & Index Touch + Move Right/Up)
            if distance(index_tip, thumb_tip) < 0.05:
                if index_tip.x - thumb_tip.x > 0.03 or index_tip.y - thumb_tip.y < -0.03:
                    pyautogui.hotkey('ctrl', '+')
                    cv2.putText(frame, "Zoom In", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Zoom Out (Thumb & Index Touch + Move Left/Down)
            if distance(index_tip, thumb_tip) < 0.05:
                if index_tip.x - thumb_tip.x < -0.03 or index_tip.y - thumb_tip.y > 0.03:
                    pyautogui.hotkey('ctrl', '-')
                    cv2.putText(frame, "Zoom Out", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display Frame
    cv2.imshow("CV Cursor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
