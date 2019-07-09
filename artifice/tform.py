"""Transformation utils, used to build up augmentations.

Note that these transformations are expected to adjust not just the image but
also the label to match. This is easily done for translation, where the
corresponding label dimension (position) is known, but less so for other pose
dimensions. Custom code may be written here to allow for this (or we may add
more standard dimensions to the label, in addition to position, to allow for )


These will be wrapped in py_function, ensuring eager execution (fine since this
is just preparing data). Each function should take `image, label, annotation,
background` as arguments and return a [new_image, new_label] list.

Because these are wrapped in py_function, things can be turned into numpy arrays
and back.

These transformations are applied to entire images, not patches, so
interpolation should be limited to as small a region as possible.

"""

import logging
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from artifice import img, vis

logger = logging.getLogger('artifice')

def swap(t):
  return tf.gather(t, [1,0])

def identity(image, label, annotation, background):
  return [image, label]

def normal_translate(image, label, annotation, background):
  """Translate each object with a random offset, normal distributed."""
  # boilerplate code
  new_image = image.numpy()
  new_label = label.numpy()
  image = image.numpy()
  annotation = annotation.numpy()
  background = background.numpy()
  for i in range(label.shape[0]):
    mask = annotation == i
    if not mask.any():
      vis.plot_image(annotation)
      plt.show()
      logger.debug(f"no {i}'th object")
      continue
    top, bottom, left, right = img.compute_object_patch(mask)
    image_patch = image[top:bottom, left:right].copy()
    mask_patch = mask[top:bottom, left:right].copy()

    # todo; figure out if this is worth it.
    # replace the original object with background
    new_image[top:bottom, left:right][mask_patch] = 0 # \
      # background[top:bottom, left:right][mask_patch]

    mask_patch = mask_patch.astype(np.float32)
    
    # the meat of the transformation, what's being done on each object
    translation = np.random.normal(loc=0, scale=5, size=2).astype(np.float32)
    new_label[i,:2] += translation
    top = max(top + np.floor(translation[0]).astype(np.int64), 0)
    bottom = min(bottom + np.floor(translation[0]).astype(np.int64), image.shape[0])
    left = max(left + np.floor(translation[1]).astype(np.int64), 0)
    right = min(right + np.floor(translation[1]).astype(np.int64), image.shape[1])
    offset = swap(translation % 1)
    image_patch = tf.contrib.image.translate(
      image_patch, offset, interpolation='BILINEAR').numpy()
    mask_patch = tf.contrib.image.translate(
      mask_patch, offset, interpolation='NEAREST').numpy()
    image_patch = image_patch[:bottom - top, :right - left]
    mask_patch = mask_patch[:bottom - top, :right - left]

    # insert the transformed object
    mask_patch = mask_patch.astype(np.bool)
    # new_image[top:bottom, left:right][mask_patch] = image_patch[mask_patch]
  return [new_image, new_label]

def uniform_rotate(image, label, annotation, background):
  """Rotate each object by a random angle from Unif(0,2pi)"""
  raise NotImplementedError

def normal_scale(image, label, annotation, background):
  """Adjust the scale of each object, with scale factors from N(1,0.1)."""
  raise NotImplementedError

def combine_1_2(image, label, annotation, background):
  """Apply random translations and rotations, as above, together."""
  raise NotImplementedError

def combine_1_2_3(image, label, annotation, background):
  """Do all three, potentially adjusting scale dimension of label."""
  raise NotImplementedError

transformations = {0 : identity,
                   1 : normal_translate,
                   2 : uniform_rotate,
                   4 : normal_scale,
                   3 : combine_1_2,
                   5 : combine_1_2_3}
