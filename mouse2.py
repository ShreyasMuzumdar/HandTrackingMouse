import cv2
import mediapipe as mp
import pyautogui

# Set up MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up screen resolution for mapping hand coordinates to screen coordinates
screen_width, screen_height = pyautogui.size()

# Set up the mouse control sensitivity
sensitivity = 5

# Initialize the VideoCapture object
cap = cv2.VideoCapture(0)

# Set up mouse control flag
mouse_control_enabled = False

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        # Read frame from the camera
        success, image = cap.read()
        if not success:
            print("Failed to read frame")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image for hand tracking
        results = hands.process(image_rgb)

        # Draw landmarks on the hands (if detected)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x = int(index_finger_landmark.x * screen_width)
                index_finger_y = int(index_finger_landmark.y * screen_height)

                # Invert the mouse cursor movement
                index_finger_x = screen_width - index_finger_x
                # index_finger_y = screen_height - index_finger_y

                # Move the mouse cursor
                if mouse_control_enabled:
                    pyautogui.moveTo(index_finger_x, index_finger_y, duration=0)

        # Display the resulting image
        cv2.imshow('Hand Tracking', image)

        # Handle keyboard events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('m'):
            mouse_control_enabled = not mouse_control_enabled

# Release the VideoCapture object and close any open windows
cap.release()
cv2.destroyAllWindows()