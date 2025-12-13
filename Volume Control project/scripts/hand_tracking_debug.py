import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_control import HandDetector


def main():
    detector = HandDetector()
    prev_time = 0.0

    camera_index = 0

    def _open_capture():
        cap = cv2.VideoCapture(camera_index)
        return cap

    cap = _open_capture()
    try:
        while True:
            frame = None
            if cap and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.release()
                    cap = None

            if frame is None:
                if cap:
                    cap.release()
                cap = _open_capture()
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"No camera feed (index {camera_index})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Connect/enable camera; will retry...",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                )
                time.sleep(0.25)
            else:
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
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
