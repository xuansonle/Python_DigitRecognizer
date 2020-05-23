import numpy as np
from keras.models import load_model
from cv2 import cv2
from PIL import Image

# Use pickle to load in the pre-trained model.
model = load_model("model.h5")

def predictNumber(image):
    all_pred = list(map(lambda x: round(x,2), list(model.predict(image))[0]))
    print(all_pred)
    pred = np.argmax(all_pred)
    return pred

def dataPrep(image):
    # Decoding
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    #image2 = Image.fromarray(image)
    #image2.save("./data/draw_original.png")

    # Resizing and reshaping to keep the ratio.
    image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)

    #image2 = Image.fromarray(image)
    #image2.save("./data/draw_resized.png")

    image = np.asarray(image, dtype="uint8")
    image = image.reshape(-1, 28, 28, 1).astype('float32')
    image /= 255.0

    return image