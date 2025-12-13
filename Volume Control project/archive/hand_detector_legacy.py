import cv2 
import mediapipe as mp 
import time



class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

                # for id, lm in enumerate(handLMS.landmark):
                #     # print(id, lm)
                #     h, w, c = img.shape
                #     cx, cy, = int(lm.x*w), int(lm.y*h)
                    
                #     if id == 4:
                #         cv2.circle(img, (cx, cy), 15, (255,0,0),3)
                #         print(id,cx,cy)
                #     if id == 8:
                #         cv2.circle(img, (cx, cy), 15, (255,0,0),3)
                #         print(id,cx,cy)
    
    
def main():
    pTime = 0
    cTime = 0       
    cap = cv2.VideoCapture(0)
    # Horizontal flipping
    
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,
                    3, (255,0,255),3)
        
        cv2.imshow("Image",img)
        
        # To exit from live, press Esc key
        if cv2.waitKey(1) & 0xFF == 27: # 27 is the Esc Key
            break
        
     
    
if __name__ == '__main__':
    main()