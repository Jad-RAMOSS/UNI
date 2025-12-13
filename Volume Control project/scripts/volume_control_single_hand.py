import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_control import HandDetector, get_audio_endpoint


def main():
    detector = HandDetector(detectionCon=0.7)
    volume = get_audio_endpoint()

    min_vol, max_vol, _ = volume.GetVolumeRange()
    vol_bar = 260
    vol_perc = 0
    locked_hand_index = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = detector.findHands(frame)

            if detector.results and detector.results.multi_hand_landmarks:
                if locked_hand_index is None:
                    locked_hand_index = 0

                if locked_hand_index < len(detector.results.multi_hand_landmarks):
                    landmarks, _ = detector.findPosition(frame, handNo=locked_hand_index)
                    if landmarks:
                        x1, y1 = landmarks[4][1], landmarks[4][2]
                        x2, y2 = landmarks[8][1], landmarks[8][2]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        cv2.line(frame, (x1, y1), (x2, y2), (136, 252, 3), 4)
                        cv2.circle(frame, (cx, cy), 15, (136, 252, 3), cv2.FILLED)

                        length = math.hypot(x2 - x1, y2 - y1)
                        vol_level = np.interp(length, [25, 150], [min_vol, max_vol])
                        vol_bar = int(np.interp(length, [25, 150], [260, 10]))
                        vol_perc = int(np.interp(length, [25, 150], [0, 100]))

                        volume.SetMasterVolumeLevel(vol_level, None)
                        if length < 25:
                            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

            cv2.rectangle(frame, (10, 425), (260, 460), (37, 235, 7), 3)
            cv2.rectangle(frame, (vol_bar, 425), (260, 460), (37, 235, 7), cv2.FILLED)
            cv2.putText(
                frame,
                f"{vol_perc} %",
                (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Single Hand Volume Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
