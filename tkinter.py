# #!/usr/bin/env python
# # coding: utf-8

# # In[ ]:


# ### Load libraries 

# #%matplotlib inline
# #%config InlineBackend.figure_format = "retina"

# import gzip
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import zipfile
# from tqdm import tqdm_notebook as tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle

# # Keras
# from keras.datasets import mnist
# import keras.models as models
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, UpSampling2D
# from keras.utils import to_categorical
# import keras.backend as K
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import RMSprop, Adam

# # GUI
# from numpy import argmax
# import tkinter as tk
# import math
# from PIL import Image, ImageDraw, ImageTk

# def open_images(filename):
#     with gzip.open(filename, "rb") as file:
#         data = file.read()
#         return np.frombuffer(data, dtype=np.uint8, offset=16)            .reshape(-1, 28, 28)            .astype(np.float32)

# def open_labels(filename):
#     with gzip.open(filename, "rb") as file:
#         data = file.read()
#         return np.frombuffer(data, dtype=np.uint8, offset=8)


# # # Modelling

# # ## Read data

# # In[ ]:


# #### Read dataset
# #X_train = open_images("./data/mnist/train-images-idx3-ubyte.gz")
# #y_train = open_labels("./data/mnist/train-labels-idx1-ubyte.gz")
# #X_test = open_images("./data/mnist/t10k-images-idx3-ubyte.gz")
# #y_test = open_labels("./data/mnist/t10k-labels-idx1-ubyte.gz")

# #plt.imshow(X_train[1],cmap="gray_r")


# # ## Prepare data

# # In[ ]:


# ## One hot encoding: Multinomial: y = 0, y = 1, etc.
# #y_train_multi = to_categorical(y_train)
# #y_test_multi = to_categorical(y_test)

# ## Reshape input image (number of img, width in pixel, height in pixel, number of color layer)
# #X_train = X_train.reshape(-1, 28, 28, 1)
# #X_test = X_test.reshape(-1, 28, 28, 1)

# ## Standard the input values
# #X_train = X_train.astype('float32')
# #X_test = X_test.astype('float32')
# #X_train /= 255.0
# #X_test /= 255.0


# # ## Build model

# # In[ ]:


# ## Create the model
# #model = Sequential()

# #model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1)))
# #model.add(MaxPooling2D(pool_size=(2,2)))

# #model.add(Flatten())
# #model.add(Dense(128, activation='relu'))
# #model.add(Dense(10, activation='softmax'))

# ## Compile the model
# #model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# ## Train the model
# #model.fit(X_train, y_train_multi, batch_size=32, epochs=2, verbose=1, validation_split=0.3)

# ## Evaluate on test data
# #model.evaluate(X_test, y_test_multi)


# # ## Save model

# # In[ ]:


# #model.save("DigitRecognizer.h5")


# # # GUI

# # ## Load model

# # In[ ]:


# model = load_model("DigitRecognizer.h5")


# # ## Create GUI

# # In[ ]:


# size = (28, 28)
# white = (255, 255, 255)
# black = (0, 0, 0)
# canvas_width = 150
# canvas_height = 150
# xpoints = []
# ypoints = []  
# x2points = []
# y2points = []


# # In[ ]:


# # Define functions 
# def createPaint():

#     global w

#     w = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='red')

#     w.grid(row=1)

#     w.bind("<B1-Motion>", paint)

# def paint(event):

#     x1, y1 = (event.x - 6), (event.y - 6)
#     x2, y2 = (event.x + 6), (event.y + 6)

#     # Draw each point
#     w.create_oval(x1, y1, x2, y2, fill='black')

#     # Append the coordinates to list
#     xpoints.append(x1)
#     ypoints.append(y1)
#     x2points.append(x2)
#     y2points.append(y2)

# def dataPrep(image):        
#     image = image.resize(size)
#     image = image.convert('L')
#     image = np.array(image)
#     image = image.reshape(-1, 28, 28, 1)
#     image = image.astype('float32')
#     image /= 255.0
#     return image
    
# def displayTKImage(image):
    
#     plt.imshow(image.reshape(28, 28), cmap="gray")
#     plt.savefig('./data/gui/result.png',bbox_inches='tight')
#     plt.close()

#     img = ImageTk.PhotoImage(Image.open("./data/gui/result.png"))

#     labelImage = tk.Label(window, image=img)
#     labelImage.image = img
#     labelImage.grid(row=6, column=0)
    
# def predictNumber():

#     global xpoints
#     global ypoints
#     global x2points
#     global y2points

#     image1 = Image.new("RGB", (canvas_width, canvas_height), black)
#     draw1 = ImageDraw.Draw(image1)

#     elementos = len(xpoints)

#     # Recreate the drawn image
#     for p in range(elementos):
#         x = xpoints[p]
#         y = ypoints[p]
#         x2 = x2points[p]
#         y2 = y2points[p]
#         draw1.ellipse((x, y, x2, y2), 'white')

#     # Data preparation for modeling
#     image1 = dataPrep(image1)

#     # Display the predicted value. Zeile 4
#     all_pred = list(map(lambda x: round(x,2), list(model.predict(image1))[0]))
#     pred = argmax(model.predict(image1))
#     tk.Label(window, text=f"Prediction: {pred}",font=('Arial Bold', 20)).grid(row=4)
#     tk.Label(window, text=f"Prob: {all_pred}",font=('Arial Bold', 15)).grid(row=5)
    
#     # Display TK Image
#     displayTKImage(image1)
    
#     xpoints = []
#     ypoints = []
#     x2points = []
#     y2points = []

# def reset():
#     # Reset the global image values for the next prediction
#     global xpoints
#     global ypoints
#     global x2points
#     global y2points

#     xpoints = []
#     ypoints = []
#     x2points = []
#     y2points = []
    
#     createPaint()


# # In[ ]:


# # GUI

# # Initialize a window
# window = tk.Tk()

# # Define the title and window size
# window.title("Handwriting Calculator")
# window.geometry('800x800')

# # Add heading text to Zeile 0
# tk.Label(window,
#          text="Write digits with your mouse in the red square",
#          font=('Arial Bold', 15)).grid(row=0)

# # Start building
# createPaint()

# # Display button for predicting the drawn number
# tk.Button(window, text='Save image', width=25,command=predictNumber).grid(row=2)

# # Display button in case wrong prediction
# #tk.Button(window, text='Click here if the number is not correct', width=35, command=delete).grid(row=4)

# # Display reset button
# tk.Button(window, text='Reset', width=25, command=reset).grid(row=3)

# # Open the window
# window.mainloop()

