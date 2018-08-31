# 文件名不能叫io.py

import tensorflow as tf
import os
from config import *

'''
------------------------------------------------------------------------------------------------------------------------------------------
Image process
------------------------------------------------------------------------------------------------------------------------------------------
'''


def preprocess_for_style(image, target_height, target_width):
    """
    :param image: A Tensor of type uint8.
    :param target_height:
    :param target_width:
    :return: A Tensor of type flot32, shape [1, target_height,target_width,3]
    """
    image = _aspect_preserving_resize(image, target_height, target_width)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)
    return image

def preprocess_for_train(image, output_height, output_width, resize_side_min=RESIZE_SIDE_MIN, resize_side_max=RESIZE_SIDE_MAX):
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32) # 左闭右开

    image = _aspect_preserving_resize(image, resize_side, resize_side)
    # image = _random_crop([image], output_height, output_width)[0]
    image = tf.random_crop(image, [output_height, output_width, 3])
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return image


def _smallest_size_at_least(height, width, target_height, target_width):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    target_height = tf.to_float(target_height)
    target_width = tf.to_float(target_width)

    scale = tf.cond(tf.greater(target_height / height, target_width / width),
                    lambda: target_height / height,
                    lambda: target_width / width)
    new_height = tf.to_int32(tf.round(height * scale))
    new_width = tf.to_int32(tf.round(width * scale))
    return new_height, new_width


def _aspect_preserving_resize(image, target_height, target_width):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    target_height = tf.convert_to_tensor(target_height, dtype=tf.int32)
    target_width = tf.convert_to_tensor(target_width, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, target_height, target_width)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3]) # 为啥要让H,W变None
    return resized_image


'''
------------------------------------------------------------------------------------------------------------------------------------------
Image IO
------------------------------------------------------------------------------------------------------------------------------------------
'''


def get_imgpath_batch_iterator(batch_size, epoch=2, shuffle=True):
    # Input:  epoch, batch size, using shuffle or not.
    # Output: an iterator
    filelist = os.listdir(CONTENT_IMG_DIR)  # return files tuple.String, not tensor, but it's ok.
    dataset = tf.data.Dataset.from_tensor_slices(filelist)  # return a dataset. Each element is one file name.
    # basic config for the dataset
    # no map operation because image sizes are different cannot rand_resize
    # map 在batch之前，不然map是以一整个batch作为运算
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle)
    iterator = dataset.make_one_shot_iterator()
    # by official doc:
    # The returned iterator will be initialized automatically.
    # A "one-shot" iterator does not currently support re-initialization.
    next_batch = iterator.get_next()  # get one batch.
    # when use sess.run, the iterator will go next, and we get next element(cause we use the compute graph).
    return next_batch

#
def get_train_img(file):  # map function
    # Input: file name. String
    # Output: a corresponding processed image.
    # key: Train set directory is set by settings.py
    # Attention: no zero-mean, cause in gen net there are BNs
    try:
        # Tensor: type uint8 , shape [height, width, num_channels]
        with tf.device('/cpu:0'):
            print('get raw img from', file)
            img = load_img(CONTENT_IMG_DIR + file)
            # print('get raw img done')
    except:
        img = None
        print('img is None')
    return img


def load_img(img_path):
    """
    :param img_path: one path
    :return A Tensor of type uint8.
    """
    try:
        read_byt = tf.read_file(img_path)
        img = tf.image.decode_jpeg(read_byt, channels=3)
    except:
        img = None
        print('img is None')
        exit(1)
    return img


def save_img(img, name, sess):
    with tf.device('/cpu:0'):
        print('save generated image')
        img = tf.clip_by_value(tf.cast(img,tf.uint8),0,255)
        img = tf.squeeze(img)
        out_byt = tf.image.encode_jpeg(img)
        save_op = tf.write_file(IMG_SAVE_DIR + name, out_byt)
        sess.run(save_op)
        print('save success')
