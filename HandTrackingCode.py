import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # zero uses the default webcam (if you have one camera)
# sets width and height respectively
#cap.set(3, 640)
#cap.set(4, 480)
# cap.set(10,100) BRIGHTNESS OF CAMERA

mpHands = mp.solutions.hands
hands = mpHands.Hands()  #use ctrl+click to see the parameters in that  "Hands()" method
mpDraw = mp.solutions.drawing_utils   #mediapipe method to draw lines for hand landmarks

#To display FrameRate
pTime=  0
cTime = 0


# while loop to go through each frame
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks) # test if the landmarks ae being read

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #draws the points for the landmarks

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #show the fps on screen
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("Video", img)
    cv2.waitKey(1)

