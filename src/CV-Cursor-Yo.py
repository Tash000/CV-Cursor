import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import logging

# Logging setup for better debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # Limit to 1 hand for better performance
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Screen and frame setup
screen_w, screen_h = pyautogui.size()
margin = 100
pyautogui.FAILSAFE = False

# Initialize camera with DirectShow for better stability
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# State variables
cursor_active = False
yo_start_time = 0
yo_gesture_held = False
border_color = (0, 255, 255)  # Yellow by default
last_deactivation_time = 0

# Scroll variables
ring_fold_start = 0
ring_fold_held = False
pinky_fold_start = 0
pinky_fold_held = False
auto_scroll_active = False
auto_scroll_direction = 0

# Smooth cursor movement variables
prev_x, prev_y = 0, 0
smoothing_factor = 7  # Higher = smoother but slower

def is_finger_up(tip, pip):
    return tip.y < pip.y

def is_thumb_away(thumb_tip, other_fingers):
    """Check if thumb is away from other fingers"""
    min_distance = 0.1
    for finger in other_fingers:
        distance = ((thumb_tip.x - finger.x) ** 2 + (thumb_tip.y - finger.y) ** 2) ** 0.5
        if distance < min_distance:
            return False
    return True

def check_yo_gesture(landmarks):
    """Check for yo sign (index and pinky up, others down)"""
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    index_up = is_finger_up(index_tip, index_pip)
    pinky_up = is_finger_up(pinky_tip, pinky_pip)
    middle_down = not is_finger_up(middle_tip, middle_pip)
    ring_down = not is_finger_up(ring_tip, ring_pip)
    
    return index_up and pinky_up and middle_down and ring_down

def check_thumbs_up(landmarks):
    """Check for thumbs up gesture (thumb up, all other fingers down)"""
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    # Check other fingers are down
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Thumb up and all other fingers down
    thumb_up = thumb_tip.y < thumb_ip.y
    index_down = not is_finger_up(index_tip, index_pip)
    middle_down = not is_finger_up(middle_tip, middle_pip)
    ring_down = not is_finger_up(ring_tip, ring_pip)
    pinky_down = not is_finger_up(pinky_tip, pinky_pip)
    
    return thumb_up and index_down and middle_down and ring_down and pinky_down

def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    
    # Better error handling
    if not success or frame is None:
        logging.warning("⚠️ Could not read frame from camera.")
        continue
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Draw border
    cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), border_color, 2)
    
    # Display status with better formatting
    status = "ACTIVE" if cursor_active else "INACTIVE"
    color = (0, 255, 0) if cursor_active else (0, 0, 255)
    cv2.putText(frame, f"Cursor: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Auto-scroll display
    if auto_scroll_active:
        scroll_text = "Auto Scroll UP" if auto_scroll_direction > 0 else "Auto Scroll DOWN"
        cv2.putText(frame, scroll_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        pyautogui.scroll(auto_scroll_direction * 5)
        logging.info(f"⬆️ AUTO SCROLL {'UP' if auto_scroll_direction > 0 else 'DOWN'}")
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get finger positions
            fingers = {
                'thumb': hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                'index': hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                'middle': hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                'ring': hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                'pinky': hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            }
            
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
            
            other_fingers = [fingers['index'], fingers['middle'], fingers['ring'], fingers['pinky']]
            thumb_away = is_thumb_away(fingers['thumb'], other_fingers)
            
            # ACTIVATION: Yo gesture for 3 seconds
            if not cursor_active:
                if check_yo_gesture(hand_landmarks):
                    if not yo_gesture_held:
                        yo_start_time = current_time
                        yo_gesture_held = True
                    
                    held_time = current_time - yo_start_time
                    cv2.putText(frame, f"Activating: {held_time:.1f}/3.0", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if held_time >= 3.0:
                        cursor_active = True
                        border_color = (255, 0, 255)  # Purple
                        yo_gesture_held = False
                        logging.info("✅ CURSOR ACTIVATED!")
                else:
                    yo_gesture_held = False
            
            # DEACTIVATION: Simple thumbs up gesture
            else:
                if current_time - yo_start_time > 2.0:
                    if check_thumbs_up(hand_landmarks):
                        cursor_active = False
                        border_color = (0, 255, 255)  # Yellow
                        auto_scroll_active = False
                        last_deactivation_time = current_time
                        cv2.putText(frame, "DEACTIVATED! 👍", (10, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        logging.info("❌ CURSOR DEACTIVATED with THUMBS UP!")
                
                # Cursor operations when active
                if cursor_active:
                    # SMOOTH MOUSE MOVEMENT using interpolation
                    index_x_raw = np.interp(fingers['index'].x * w, (margin, w - margin), (0, screen_w))
                    index_y_raw = np.interp(fingers['index'].y * h, (margin, h - margin), (0, screen_h))
                    
                    # Apply smoothing
                    smooth_x = prev_x + (index_x_raw - prev_x) / smoothing_factor
                    smooth_y = prev_y + (index_y_raw - prev_y) / smoothing_factor
                    
                    # Keep cursor within bounds
                    smooth_x = max(0, min(screen_w - 1, smooth_x))
                    smooth_y = max(0, min(screen_h - 1, smooth_y))
                    
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    # Scroll gestures (only when thumb is away)
                    if thumb_away and not check_yo_gesture(hand_landmarks) and not check_thumbs_up(hand_landmarks):
                        # Ring finger fold for scroll up
                        ring_folded = not is_finger_up(fingers['ring'], ring_pip)
                        if ring_folded:
                            if not ring_fold_held:
                                ring_fold_start = current_time
                                ring_fold_held = True
                            
                            held_time = current_time - ring_fold_start
                            cv2.putText(frame, f"Ring Fold: {held_time:.1f}/2.0", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            if held_time >= 2.0:
                                auto_scroll_active = True
                                auto_scroll_direction = 1
                                ring_fold_held = False
                        else:
                            ring_fold_held = False
                        
                        # Pinky finger fold for scroll down
                        pinky_folded = not is_finger_up(fingers['pinky'], pinky_pip)
                        if pinky_folded:
                            if not pinky_fold_held:
                                pinky_fold_start = current_time
                                pinky_fold_held = True
                            
                            held_time = current_time - pinky_fold_start
                            cv2.putText(frame, f"Pinky Fold: {held_time:.1f}/2.0", (10, 220), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            
                            if held_time >= 2.0:
                                auto_scroll_active = True
                                auto_scroll_direction = -1
                                pinky_fold_held = False
                        else:
                            pinky_fold_held = False
                    
                    # Clicks (only if not doing special gestures and thumb close)
                    if (not check_yo_gesture(hand_landmarks) and 
                        not check_thumbs_up(hand_landmarks) and 
                        not thumb_away):
                        
                        if distance(fingers['thumb'], fingers['ring']) < 0.05:
                            pyautogui.click()
                            cv2.putText(frame, "LEFT CLICK", (10, 300), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        if distance(fingers['thumb'], fingers['middle']) < 0.05:
                            pyautogui.rightClick()
                            cv2.putText(frame, "RIGHT CLICK", (10, 330), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    else:
        auto_scroll_active = False
    
    # Instructions
    if not cursor_active:
        cv2.putText(frame, "Yo sign (3s) = Activate", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Thumbs up = Deactivate", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Hand Gesture Cursor Control", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
