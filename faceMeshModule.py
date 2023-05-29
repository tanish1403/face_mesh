import cv2 as cv
import mediapipe as mp
import time

class faceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=1,
                 refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh( self.static_image_mode, self.max_num_faces,
                 self.refine_landmarks, self.min_detection_confidence,
                 self.min_tracking_confidence )
        self.drawSpec = self.mpDraw.DrawingSpec( thickness=1, circle_radius=1, color=(0, 255, 0) )

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv.cvtColor( img, cv.COLOR_BGR2RGB )

        self.results = self.faceMesh.process( imgRGB )

        faces = []
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks( img, faceLMS, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec )
                face=[]
                for id, lm in enumerate( faceLMS.landmark ):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int( lm.x * iw ), int( lm.y * ih )
                    # print( f"{id}--> {x}, {y}" )
                    face.append([x, y])
                faces.append(face)
            return  img, faces



def main():
    cap = cv.VideoCapture( "videos/4.mp4" )
    pTime = 0
    detector = faceMeshDetector()

    while True:
        success, frame = cap.read()
        width = int( frame.shape[1] * 0.30 )
        height = int( frame.shape[0] * 0.30 )

        dimension = (width, height)

        img = cv.resize( frame, dimension, interpolation=cv.INTER_AREA )

        img, faces = detector.findFaceMesh(img,False)
        if len(faces)!=0:
            print(len(faces[0]))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText( img, f"FPS: {int( fps )}", (20, 70), cv.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2 )

        cv.imshow( "Image", img )
        if cv.waitKey( 1 ) & 0xFF == ord( "d" ):
            break

if __name__ == "__main__":
    main()