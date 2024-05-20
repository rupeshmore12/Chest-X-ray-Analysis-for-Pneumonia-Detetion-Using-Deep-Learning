from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="template", static_folder="staticFiles")
model = tf.keras.models.load_model("c_xray2.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = os.path.join(app.root_path, "uploads", file.filename)
    file.save(image_path)
    
    raw_img = cv2.imread(image_path)   # importing image
    raw_img = cv2.resize(raw_img, (200,200)) 
    raw_img = np.array(raw_img) # convert image to array
    raw_img = np.expand_dims(raw_img, axis=0)
    raw_img = raw_img / 255.0  # data max normalization
    probability = model.predict(raw_img)  # probability for each class
    pred = probability.argmax(axis=1)
    prediction_text = ""
    if pred > 0.5:
        prediction_text = "Pneumonia detected: Please consult a doctor immediately for further evaluation and treatment."
    else:
        prediction_text = "No pneumonia detected: Your lungs appear healthy. Please continue to monitor your symptoms and seek medical attention if they persist or worsen."

    return render_template("index.html", prediction_text=prediction_text, path="/staticFiles/uploads/" + file.filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
