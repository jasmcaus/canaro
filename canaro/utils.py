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



    """
    Increases the image size by adding large padding around the image
    Acts as a zoom out of the image, and when the image is later resized to
    the input size the network expects, it provides smaller size object
    examples.
    Args:
        image: Tensor with image of shape (H, W, 3).
        bboxes: Optional Tensor with bounding boxes with shape (num_bboxes, 5).
            where we have (x_min, y_min, x_max, y_max, label) for each one.
    Returns:
        Dictionary containing:
            image: Tensor with zoomed out image.
            bboxes: Tensor with zoomed out bounding boxes with shape
                (num_bboxes, 5).
    """
    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]
    size_multiplier = tf.random_uniform([1], minval=min_ratio,
                                        maxval=max_ratio, seed=seed)

    # Expand image
    new_height = height * size_multiplier
    new_width = width * size_multiplier
    pad_left = tf.random_uniform([1], minval=0,
                                 maxval=new_width-width, seed=seed)
    pad_right = new_width - width - pad_left
    pad_top = tf.random_uniform([1], minval=0,
                                maxval=new_height-height, seed=seed)
    pad_bottom = new_height - height - pad_top

    # TODO: use mean instead of 0 for filling the paddings
    paddings = tf.stack([tf.concat([pad_top, pad_bottom], axis=0),
                         tf.concat([pad_left, pad_right], axis=0),
                         tf.constant([0., 0.])])
    expanded_image = tf.pad(image, tf.to_int32(paddings), constant_values=fill)

    # Adjust bboxes
    shift_bboxes_by = tf.concat([pad_left, pad_top, pad_left, pad_top], axis=0)
    bbox_labels = tf.reshape(bboxes[:, 4], (-1, 1))
    bbox_adjusted_coords = bboxes[:, :4] + tf.to_int32(shift_bboxes_by)
    bbox_adjusted = tf.concat([bbox_adjusted_coords, bbox_labels], axis=1)

    # Return results
    return_dict = {'image': expanded_image}
    if bboxes is not None:
        return_dict['bboxes'] = bbox_adjusted
    return return_dict


def resize_image(image, bboxes=None, min_size=None, max_size=None):
    """
    We need to resize image and (optionally) bounding boxes when the biggest
    side dimension is bigger than `max_size` or when the smaller side is
    smaller than `min_size`. If no max_size defined it won't scale down and if
    no min_size defined it won't scale up.
    Then, using the ratio we used, we need to properly scale the bounding
    boxes.
    Args:
        image: Tensor with image of shape (H, W, 3).
        bboxes: Optional Tensor with bounding boxes with shape (num_bboxes, 5).
            where we have (x_min, y_min, x_max, y_max, label) for each one.
        min_size: Min size of width or height.
        max_size: Max size of width or height.
    Returns:
        Dictionary containing:
            image: Tensor with scaled image.
            bboxes: Tensor with scaled (using the same factor as the image)
                bounding boxes with shape (num_bboxes, 5).
            scale_factor: Scale factor used to modify the image (1.0 means no
                change).
    """
    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]

    if min_size is not None:
        # We calculate the upscale factor, the rate we need to use to end up
        # with an image with it's lowest dimension at least `image_min_size`.
        # In case of being big enough the scale factor is 1. (no change)
        min_size = tf.to_float(min_size)
        min_dimension = tf.minimum(height, width)
        upscale_factor = tf.maximum(min_size / min_dimension, 1.)
    else:
        upscale_factor = tf.constant(1.)

    if max_size is not None:
        # We do the same calculating the downscale factor, to end up with an
        # image where the biggest dimension is less than `image_max_size`.
        # When the image is small enough the scale factor is 1. (no change)
        max_size = tf.to_float(max_size)
        max_dimension = tf.maximum(height, width)
        downscale_factor = tf.minimum(max_size / max_dimension, 1.)
    else:
        downscale_factor = tf.constant(1.)

    scale_factor = upscale_factor * downscale_factor

    # New size is calculate using the scale factor and rounding to int.
    new_height = height * scale_factor
    new_width = width * scale_factor

    # Resize image using TensorFlow's own `resize_image` utility.
    image = tf.image.resize_images(
        image, tf.stack(tf.to_int32([new_height, new_width])),
        method=tf.image.ResizeMethod.BILINEAR
    )

    if bboxes is not None:
        bboxes = adjust_bboxes(
            bboxes,
            old_height=height, old_width=width,
            new_height=new_height, new_width=new_width
        )
        return {
            'image': image,
            'bboxes': bboxes,
            'scale_factor': scale_factor,
        }

    return {
        'image': image,
        'scale_factor': scale_factor,
    }


def resize_image_fixed(image, new_height, new_width, bboxes=None):

    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]

    scale_factor_height = new_height / height
    scale_factor_width = new_width / width

    # Resize image using TensorFlow's own `resize_image` utility.
    image = tf.image.resize_images(
        image, tf.stack(tf.to_int32([new_height, new_width])),
        method=tf.image.ResizeMethod.BILINEAR
    )

    if bboxes is not None:
        bboxes = adjust_bboxes(
            bboxes,
            old_height=height, old_width=width,
            new_height=new_height, new_width=new_width
        )
        return {
            'image': image,
            'bboxes': bboxes,
            'scale_factor': (scale_factor_height, scale_factor_width),
        }

    return {
        'image': image,
        'scale_factor': (scale_factor_height, scale_factor_width),
    }

