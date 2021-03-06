import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
face_mesh = mpFaceMesh.FaceMesh(min_detection_confidence=0.8)
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


while True:
    _, img = cap.read()
    # imgGray = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    imFlip = cv2.flip(img, 1)
    results = face_mesh.process(imFlip)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                image=imFlip,
                landmark_list=face_landmarks,
                connections=mpFaceMesh.FACE_CONNECTIONS,
                connection_drawing_spec=drawing_spec
            )

    cv2.imshow('Image', imFlip)
    if cv2.waitKey(1) == ord('q'):
        break