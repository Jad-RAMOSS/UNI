import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_control import HandDetector, get_audio_endpoint


def _parse_args():
    parser = argparse.ArgumentParser(description="Control system volume with hand gestures.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture.")
    parser.add_argument("--smoothness", type=int, default=10, help="Round volume percentage to nearest N.")
    parser.add_argument("--min-distance", type=int, default=25, help="Minimum thumb-index distance in pixels.")
    parser.add_argument("--max-distance", type=int, default=150, help="Maximum thumb-index distance in pixels.")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum hand bbox area gate.")
    parser.add_argument("--max-area", type=int, default=500, help="Maximum hand bbox area gate.")
    parser.add_argument(
        "--no-pinky-guard",
        action="store_true",
        help="Disable pinky confirmation requirement before applying volume.",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the camera feed for a selfie view.",
    )
    return parser.parse_args()


def main():
    print("App started")
    args = _parse_args()
    detector = HandDetector(detectionCon=0.7, maxHands=1, mirror=args.mirror)
    volume = get_audio_endpoint()

    vol_bar = 260
    vol_perc = 0
    color = (255, 0, 0)
    smoothness = max(1, args.smoothness)
    min_distance = max(1, args.min_distance)
    max_distance = max(min_distance + 1, args.max_distance)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = detector.findHands(frame)
            landmarks, bbox = detector.findPosition(frame, draw=True)
            if landmarks and bbox != (0, 0, 0, 0):
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
                if args.min_area < area < args.max_area:
                    length, frame, coordinates = detector.findDistance(4, 8, frame)
                    vol_bar = int(np.interp(length, [min_distance, max_distance], [260, 10]))
                    vol_perc = int(np.interp(length, [min_distance, max_distance], [0, 100]))
                    vol_perc = smoothness * round(vol_perc / smoothness)

                    fingers = detector.fingersUp()
                    apply_guard = len(fingers) == 5 and (args.no_pinky_guard or not fingers[4])
                    if apply_guard:
                        volume.SetMasterVolumeLevelScalar(vol_perc / 100, None)
                        cv2.circle(frame, (coordinates[4], coordinates[5]), 15, (0, 0, 255), cv2.FILLED)
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

            cv2.rectangle(frame, (10, 425), (260, 460), (37, 235, 7), 3)
            cv2.rectangle(frame, (vol_bar, 425), (260, 460), (37, 235, 7), cv2.FILLED)
            cv2.putText(
                frame,
                f"{int(vol_perc)} %",
                (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (122, 163, 129),
                2,
            )

            current_volume = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(
                frame,
                f"Vol Set: {current_volume}",
                (480, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color,
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
            guard_text = "Guard: OFF" if args.no_pinky_guard else "Guard: pinky"
            cv2.putText(
                frame,
                f"{guard_text} | Smooth: {smoothness}",
                (20, 45),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (100, 170, 255),
                1,
            )

            cv2.imshow("Advanced Volume Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
