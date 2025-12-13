import time
import cv2
import numpy as np

from hand_control import HandDetector, get_audio_endpoint


def main(camera_index: int = 0):
    detector = HandDetector(detectionCon=0.7, maxHands=1)
    volume = get_audio_endpoint()

    vol_bar = 260
    vol_perc = 0
    color = (255, 0, 0)
    smoothness = 10

    cap = cv2.VideoCapture(camera_index)
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
                if 30 < area < 500:
                    length, frame, coordinates = detector.findDistance(4, 8, frame)
                    vol_bar = int(np.interp(length, [25, 150], [260, 10]))
                    vol_perc = int(np.interp(length, [25, 150], [0, 100]))
                    vol_perc = smoothness * round(vol_perc / smoothness)

                    fingers = detector.fingersUp()
                    if len(fingers) == 5 and not fingers[4]:
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

            cv2.imshow("Advanced Volume Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
