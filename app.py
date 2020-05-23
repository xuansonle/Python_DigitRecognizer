import flask
from flask import Flask, render_template, url_for, request
import pickle
import base64
import numpy as np
from helpfunctions import predictNumber, dataPrep

#import os
#port = int(os.environ.get('PORT', 5000))

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True
app.debug = True

#First route : Render the initial drawing template
@app.route('/')
def home():
    return render_template('draw.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        final_pred = None
        #Preprocess the image : set the image to 28x28 shape
        #Access the image
        draw = request.form['url']
        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        #Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        
        if image.shape[0] > 3658:
            image = dataPrep(image)
            final_pred = predictNumber(image)
        #image = dataPrep(image)
        #final_pred = predictNumber(image)
        #print(final_pred)

    return render_template('results.html', prediction=final_pred)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=port, debug=True)
    app.run()