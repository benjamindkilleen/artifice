"""Transformation utils, used to build up augmentations.

Note that these transformations are expected to adjust not just the image but
also the label to match. This is easily done for translation, where the
corresponding label dimension (position) is known, but less so for other pose
dimensions. Custom code may be written here to allow for this (or we may add
more standard dimensions to the label, in addition to position, to allow for )

In each cases, background inpainting is done with random noise, with the same
mean and variance as the original image.

These will be wrapped in py_function, ensuring eager execution (fine since this
is just preparing data). Each function should take a list of tensors [image,
label, annotation] and return a [new_image, new_label] list.

Because these are wrapped in py_function, things can be turned into numpy arrays
and back.

"""

import logging
import tensorflow as tf

logger = logging.getLogger('artifice')

def _swap(t):
  return tf.gather(t, [1,0])

def identity(image, label, annotation):
  return [image, label]

def normal_translate(image, label, annotation):
  """Translate each object with a random offset, normal distributed."""
  # image, label, annotation = args

  # num_objects array

  image_std = tf.math.reduce_std(image)
  image_mean = tf.reduce_mean(image)
  new_image = tf.identity(image)
  new_label = label.numpy()
  for i in range(label.shape[0]):
    obj_image = tf.identity(image)
    mask = tf.cast(tf.equal(annotation, tf.constant(i, tf.float32)),
                   tf.float32)
    obj_mask = tf.identity(mask)

    offset = tf.random.normal([2], mean=0, stddev=5, dtype=tf.float32)
    new_label[i,:2] += offset.numpy()
    obj_image = tf.contrib.image.translate(
      obj_image, _swap(offset), interpolation='BILINEAR')
    obj_mask = tf.contrib.image.translate(
      obj_mask, _swap(offset), interpolation='NEAREST')

    new_image = tf.where(tf.cast(mask, tf.bool),
                         tf.random.normal(image.shape, mean=image_mean,
                                          stddev=image_std,
                                          dtype=tf.float32),
                         new_image)
    new_image = tf.where(tf.cast(obj_mask, tf.bool), obj_image, new_image)
  return [new_image, tf.constant(new_label)]

def uniform_rotate(image, label, annotation):
  """Rotate each object by a random angle from Unif(0,2pi)"""
  raise NotImplementedError

def normal_translate_uniform_rotate(image, label, annotation):
  """Apply random translations and rotations, as above, together."""
  raise NotImplementedError

def normal_scale(image, label, annotation):
  """Adjust the scale of each object, with scale factors from N(1,0.1)."""
  raise NotImplementedError

def normal_translate_uniform_rotate_normal_scale(image, label, annotation):
  """Do all three, potentially adjusting scale dimension of label."""
  raise NotImplementedError

transformations = {0 : identity,
                   1 : normal_translate,
                   2 : uniform_rotate,
                   3 : normal_translate_uniform_rotate,
                   4 : normal_scale,
                   5 : normal_translate_uniform_rotate_normal_scale}
