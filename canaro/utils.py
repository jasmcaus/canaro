# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Surpressing Tensorflow Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf


def adjust_bboxes(bboxes, old_height, old_width, new_height, new_width):
    """Adjusts the bboxes of an image that has been resized.
    Args:
        bboxes: Tensor with shape (num_bboxes, 5). Last element is the label.
        old_height: Float. Height of the original image.
        old_width: Float. Width of the original image.
        new_height: Float. Height of the image after resizing.
        new_width: Float. Width of the image after resizing.
    Returns:
        Tensor with shape (num_bboxes, 5), with the adjusted bboxes.
    """
    # We normalize bounding boxes points.
    bboxes_float = tf.to_float(bboxes)
    x_min, y_min, x_max, y_max, label = tf.unstack(bboxes_float, axis=1)

    x_min = x_min / old_width
    y_min = y_min / old_height
    x_max = x_max / old_width
    y_max = y_max / old_height

    # Use new size to scale back the bboxes points to absolute values.
    x_min = tf.to_int32(x_min * new_width)
    y_min = tf.to_int32(y_min * new_height)
    x_max = tf.to_int32(x_max * new_width)
    y_max = tf.to_int32(y_max * new_height)
    label = tf.to_int32(label)  # Cast back to int.

    # Concat points and label to return a [num_bboxes, 5] tensor.
    return tf.stack([x_min, y_min, x_max, y_max, label], axis=1)

