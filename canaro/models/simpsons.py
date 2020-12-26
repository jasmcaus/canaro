# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

# Surpressing Tensorflow Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.optimizers import SGD

def createSimpsonsModel(IMG_SIZE=(224,224), channels=1, output_dim=1, loss='binary_crossentropy', decay=None, learning_rate=None, momentum=None, nesterov=None):
# def createSimpsonsModel(IMG_SIZE=(224,224), channels=1, output_dim=1):
    w,h = IMG_SIZE[:2]
    input_shape = (w,h,channels)
    input_layer = Input(input_shape)
    
    model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu', name='input_node')(input_layer)
    model = Conv2D(32, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(256, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    model = Flatten()(model)
    model = Dropout(0.5)(model)
    model = Dense(1024, activation='relu')(model)
    
    # Output Layer
    output = Dense(output_dim, activation='softmax', name='output_node')(model)
    
    model = Model(inputs=input, outputs=output)

    optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# # Importing the necessary packages
# from tensorflow.keras import backend
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import SGD

# IMG_SIZE = (80,80)
# LEARNING_RATE = 0.001
# DECAY = 1e-6
# MOMENTUM = .9
# EPOCHS = 5
# BATCH_SIZE = 32

# def createSimpsonsModel(IMG_SIZE=(224,224), channels=1, output_dim=1, loss='binary_crossentropy', decay=None, learning_rate=None, momentum=None, nesterov=None):
#     if type(output_dim) is not int:
#         raise ValueError('[ERROR] Output dimensions need to be an integer')
#     if type(channels) is not int:
#         raise ValueError('[ERROR] Channels needs to be an integer')

#     # If 'channels first', update the input_shape
#     if backend.image_data_format() == 'channels_first':
#         input_shape = (channels, img_size,img_size)
        
#     w, h = IMG_SIZE[:2]
    
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w, h,channels)))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))

#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))

#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))

#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(1024, activation='relu'))
    
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w, h,channels)))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))

#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))

#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))

#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(1024, activation='relu'))
    
#     # Output Layer
#     model.add(Dense(output_dim, activation='softmax'))

#     optimizer = tensorflow.keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)

#     model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#     return model