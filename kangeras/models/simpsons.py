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


def createSimpsonsModel(IMG_SIZE=224, channels=1, output_dim=1, loss='binary_crossentropy', decay=None, learning_rate=None, momentum=None, nesterov=None):
    if type(output_dim) is not int:
        raise ValueError('[ERROR] Output dimensions need to be an integer')
    if type(channels) is not int:
        raise ValueError('[ERROR] Channels needs to be an integer')

    # If 'channels first', update the input_shape
    if backend.image_data_format() == 'channels_first':
        input_shape = (channels, img_size,img_size)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE,IMG_SIZE,channels)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    
    # Output Layer
    model.add(Dense(output_dim, activation='softmax'))

    optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model