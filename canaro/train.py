# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

# Surpressing Tensorflow Warnings
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Importing the necessary packages
sys.path.append('..')
from .generators.imagedatagen import imageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint


def train(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=30, data_augmentation=True, datagen=None):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for data_augmentation (default:True)

    :return: history (acc, loss, val_acc, val_loss for every epoch)
    """

    if data_augmentation and datagen is None:
        datagen = imageDataGenerator()

    if data_augmentation:
        filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(lr_schedule) ,checkpoint]
        history = model.fit(datagen.flow(X_train, y_train,
                                    batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_val, y_val),
                                    callbacks=callbacks_list)  
    else:
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val),
                            shuffle=True)

    return history


def lr_schedule(epoch):
    lr = 0.01
    return lr*(0.1**int(epoch/10))