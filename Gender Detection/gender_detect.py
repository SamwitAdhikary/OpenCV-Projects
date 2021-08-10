import cv2
import cvlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

cap = cv2.VideoCapture(0)
model = load_model('gender_detection.model')

classes = ['man', 'woman']

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    face, confidence = cvlib.detect_face(img)

    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        face_crop = np.copy(img[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]

        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.putText(img, label, (startX, Y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('WebCam', img)
    if cv2.waitKey(1) == ord('q'):
        break