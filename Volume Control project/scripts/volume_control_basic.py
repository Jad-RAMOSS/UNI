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
    detector = HandDetector(detectionCon=0.7, maxHands=1)
    volume = get_audio_endpoint()

    min_vol, max_vol, _ = volume.GetVolumeRange()
    vol_bar = 260
    vol_perc = 0

    camera_index = 0

    def _open_capture():
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    cap = _open_capture()

    prev_time = 0.0

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
                landmarks, _ = detector.findPosition(frame)
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
                frame, f"{vol_perc} %",
                (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (122, 163, 129),
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

            cv2.imshow("Volume Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
