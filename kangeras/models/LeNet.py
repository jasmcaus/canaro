# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Surpressing Tensorflow Warnings
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Importing the necessary packages
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

def LeNet(img_size=224, channels=1, output_dim):
    """
    Adding some extra code for v0.0.14
    """
    if type(output_dim) is not int:
        raise ValueError('[ERROR] Output dimensions need to be an integer')
    if type(channels) is not int:
        raise ValueError('[ERROR] Channels needs to be an integer')

    # Initialize the Model
    model = Sequential()
    input_shape = (img_size,img_size,channels)

    # If 'channels first', update the input_shape
    if backend.image_data_format() == 'channels_first':
        input_shape = (channels, img_size,img_size)
    
    # First set
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Second set
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Flattening
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(output_dim, activation="relu")) # Softmax works too if multiple Dense nodes required

    return model