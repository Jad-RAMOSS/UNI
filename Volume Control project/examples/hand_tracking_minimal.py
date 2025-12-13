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
    detector = HandDetector(detectionCon=0.7)
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
                    thumb_tip = landmarks[4][1], landmarks[4][2]
                    index_tip = landmarks[8][1], landmarks[8][2]
                    print("Thumb:", thumb_tip, "Index:", index_tip)

            cv2.imshow("Minimal Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
