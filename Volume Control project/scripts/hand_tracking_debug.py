import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_control import HandDetector


def main():
    detector = HandDetector()
    prev_time = 0.0

    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = detector.findHands(frame)
            landmarks, _ = detector.findPosition(frame, listOfHighlights=[4, 8], draw=True)
            if landmarks:
                print("Thumb tip:", landmarks[4])
                print("Index tip:", landmarks[8])

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time

            cv2.putText(
                frame,
                f"{int(fps)} FPS",
                (10, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                2,
            )

            cv2.imshow("Hand Debug", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
