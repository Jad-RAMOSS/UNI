"""Utilities for detecting and tracking hands with MediaPipe."""

from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import math

__all__ = ["HandDetector"]


class HandDetector:
    """Lightweight wrapper around MediaPipe Hands with a few helper utilities."""

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
        mirror: bool = True,
    ) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mirror = mirror

        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self._mp_draw = mp.solutions.drawing_utils

        self.results = None
        self.lmList: List[List[int]] = []
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw: bool = True):
        """Detect hands in the frame and optionally draw landmarks."""
        if self.mirror:
            img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self._hands.process(imgRGB)
        imgRGB.flags.writeable = True

        if self.results.multi_hand_landmarks and draw:
            for handLMS in self.results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    img, handLMS, mp.solutions.hands.HAND_CONNECTIONS
                )
        return img

    def findPosition(
        self,
        img,
        listOfHighlights: Optional[Sequence[int]] = None,
        handNo: int = 0,
        draw: bool = False,
    ):
        """Return landmark positions for a specific hand and its bounding box."""
        listOfHighlights = listOfHighlights or []
        xList: List[int] = []
        yList: List[int] = []
        bBox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:
            if handNo >= len(self.results.multi_hand_landmarks):
                return self.lmList, bBox

            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for idx, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([idx, cx, cy])
                if draw and idx in listOfHighlights:
                    cv2.circle(img, (cx, cy), 10, (136, 252, 3), cv2.FILLED)

            if xList and yList:
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)
                bBox = (xMin, yMin, xMax, yMax)
                if draw:
                    cv2.rectangle(
                        img,
                        (bBox[0] - 20, bBox[1] - 20),
                        (bBox[2] + 20, bBox[3] + 20),
                        (0, 255, 0),
                        2,
                    )

        return self.lmList, bBox

    def fingersUp(self):
        """Return an array indicating which fingers are raised."""
        fingers = []

        if not self.lmList:
            return fingers

        # Thumb
        fingers.append(int(self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]))

        # Other fingers
        for idx in range(1, 5):
            fingers.append(
                int(self.lmList[self.tipIds[idx]][2] < self.lmList[self.tipIds[idx] - 2][2])
            )
        return fingers

    def findDistance(self, p1: int, p2: int, img, draw: bool = True):
        """Measure pixel distance between two landmarks."""
        if not self.lmList:
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (136, 252, 3), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (136, 252, 3), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (136, 252, 3), 4)
            cv2.circle(img, (cx, cy), 15, (136, 252, 3), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
