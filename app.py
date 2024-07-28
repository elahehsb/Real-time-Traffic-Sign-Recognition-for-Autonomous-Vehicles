from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
model = load_model('traffic_sign_model.h5')

def preprocess_image(img):
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_path = 'uploads/' + file.filename
    file.save(img_path)
    
    img = cv2.imread(img_path)
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    result = f'Traffic sign class: {predicted_class}'
    
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
