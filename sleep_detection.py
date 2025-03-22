import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame

pygame.mixer.init()
pygame.mixer.music.load("censor-beep-102309.wav")  

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

cap = cv2.VideoCapture(0)

EYE_CLOSED_THRESHOLD = 0.25  
FRAME_THRESHOLD = 20  

counter = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
      
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in LEFT_EYE])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in RIGHT_EYE])

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        if avg_EAR < EYE_CLOSED_THRESHOLD:
            counter += 1
            if counter >= FRAME_THRESHOLD:
                cv2.putText(frame, "SLEEPING!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                pygame.mixer.music.play() 
        else:
            counter = 0 

    cv2.imshow("Sleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
