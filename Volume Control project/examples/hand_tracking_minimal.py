import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_control import HandDetector


def main():
    detector = HandDetector(detectionCon=0.7)
    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
