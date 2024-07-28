import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('traffic_sign_model.h5')

def preprocess_image(img):
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess_image(frame)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    cv2.putText(frame, f'Traffic Sign: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Traffic Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
