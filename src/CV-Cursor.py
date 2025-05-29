import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Define margin (adjust as needed)
margin_x, margin_y = 100, 100  # Pixels from the edges

cap = cv2.VideoCapture(0)  # Start video capture

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural hand movement
    h, w, _ = frame.shape  # Get frame dimensions

    # Convert to RGB and process with MediaPipe
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw overlay (yellow rectangle showing tracking area)
    cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (0, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get fingertip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            # Convert normalized coordinates to screen size, within margin
            index_x = int((index_tip.x * w - margin_x) / (w - 2 * margin_x) * screen_w)
            index_y = int((index_tip.y * h - margin_y) / (h - 2 * margin_y) * screen_h)

            # Move mouse pointer
            pyautogui.moveTo(index_x, index_y, duration=0.01)

            # Left Click (Index & Middle Finger Close Together)
            if abs(thumb_tip.x - ring_tip.x) < 0.05 and abs(thumb_tip.y - ring_tip.y) < 0.05:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right Click (Thumb & Middle Finger Touch)
            if abs(thumb_tip.x - middle_tip.x) < 0.05 and abs(thumb_tip.y - middle_tip.y) < 0.05:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("CV Cursor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
