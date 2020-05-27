import flask
from flask import Flask, render_template, url_for, request
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from cv2 import cv2
from PIL import Image

# Load pre-trained model
model = load_model("model.h5")

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
    return render_template('draw.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        prediction = None
        predictions = None
        #Preprocess the image : set the image to 28x28 shape
        #Access the image
        draw = request.form['hidden-image']

        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        #Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        
        # Resizing and reshaping to keep the ratio.
        image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        image = np.asarray(image, dtype="uint8")
        
        image = image.reshape(-1, 28, 28, 1).astype('float32')
        image /= 255.0
        
        # Get the prediction
        predictions = list(map(lambda x: round(x,2), list(model.predict(image))[0]))
        prediction = np.argmax(predictions)
        predictions = dict(zip(list(range(0,10)),predictions))

    return render_template('results.html', prediction=prediction, predictions=predictions)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug = True)