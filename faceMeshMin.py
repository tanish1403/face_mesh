import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("videos/4.mp4")
cTime = 0
pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))


while True:
    success, frame = cap.read()
    width = int( frame.shape[1] * 0.30)
    height = int( frame.shape[0] * 0.30 )

    dimension = (width, height)

    img =  cv.resize( frame, dimension, interpolation=cv.INTER_AREA )
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLMS in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLMS, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            for id, lm  in enumerate(faceLMS.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(f"{id}--> {x}, {y}")

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText( img, f"FPS: {int( fps )}", (20, 70), cv.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2 )


    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord("d"):
        break
