"""Transformation utils, used to build up augmentations.

Note that these transformations are expected to adjust not just the image but
also the label to match. This is easily done for translation, where the
corresponding label dimension (position) is known, but less so for other pose
dimensions. Custom code may be written here to allow for this (or we may add
more standard dimensions to the label, in addition to position, to allow for )

In each cases, background inpainting is done with random noise, with the same
mean and variance as the original image.

"""

import logging
import tensorflow as tf

logger = logging.getLogger('artifice')

def identity(image, label, annotation):
  return image, label

def normal_translate(image, label, annotation):
  """Translate each object with a random offset, normal distributed."""

  # num_objects array

  offsets = tf.random.normal([tf.shape(label)[0], 2], mean=0, stddev=4.0,
                             dtype=tf.float32)
  
  raise NotImplementedError

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
