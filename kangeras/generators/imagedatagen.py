# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Surpressing Tensorflow Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Importing the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator as imgdatgen

def imageDataGenerator():
    """
    We are not adding a 'rescale' attribute because the data has already been normalized using the 'normalize' function of this class

    Returns datagen
    """
    datagen = imgdatgen(rotation_range=10, 
                           width_shift_range=.1,
                           height_shift_range=.1,
                           shear_range=.2,
                           zoom_range=.2,
                           horizontal_flip=True,
                           fill_mode='nearest')
    # We do not augment the validation data
    # val_datagen = ImageDataGenerator()

    # return train_datagen, val_datagen
    return datagen