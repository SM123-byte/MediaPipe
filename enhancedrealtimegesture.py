import cv2
import mediapipe as mp
import numpy as np


def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark

    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]

    extended = 0

    # Thumb - x
    if abs(landmarks[tip_ids[0]].x - landmarks[pip_ids[0]].x) > 0.05:
        extended += 1

    # Other fingers - y
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
            extended += 1

    if extended >= 4:
        return "Open"
    elif extended <= 1:
        return "Closed Fist"
    elif extended == 1 and landmarks[8].y < landmarks[6].y:
        return "Draw"
    else:
        return "Partial"


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    canvas = None
    prev_x, prev_y = 0, 0
    circle_x, circle_y = 300, 300

    print("Press 'q' to quit | Press 'c' to clear drawing")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if canvas is None:
                canvas = np.zeros_like(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            gesture = "No hand detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    gesture = detect_gesture(hand_landmarks)

                    # Index position 
                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)

                    # Circle follows wherever hand goes - has to open
                    if gesture == "Open":
                        circle_x, circle_y = x, y

                    # Only draws when open
                    if gesture == "Draw":
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = x, y

                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                        prev_x, prev_y = x, y
                    else:
                        prev_x, prev_y = 0, 0

            # Put frame and canvas together
            frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

            # Circle can move
            cv2.circle(frame, (circle_x, circle_y), 30, (255, 0, 0), -1)

            # Text
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Gesture Interaction System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()