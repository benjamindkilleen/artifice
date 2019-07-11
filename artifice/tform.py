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
  image = image.numpy()
  label = label.numpy()
  annotation = annotation.numpy()
  background = background.numpy()
  new_image = image.copy()
  new_label = label.copy()
  for l in range(label.shape[0]):
    mask = annotation == l
    if not mask.any():
      logger.warning(f"no {l}'th object")
      continue
    i, j, si, sj = img.compute_object_patch(mask)
    image_patch = image[i:i+si, j:j+sj].copy()
    mask_patch = mask[i:i+si, j:j+sj].copy()

    # todo; figure out if this is worth it.
    # replace the original object with background
    new_image[i:i+si, j:j+sj][mask_patch] = \
      background[i:i+si, j:j+sj][mask_patch]

    # get the translation, adjust the patch values
    mask_patch = mask_patch.astype(np.float32)
    translation = np.random.normal(loc=0, scale=5, size=2).astype(np.float32)
    offset = swap(translation % 1)
    image_patch = tf.contrib.image.translate(
      image_patch, offset, interpolation='BILINEAR').numpy()
    mask_patch = tf.contrib.image.translate(
      mask_patch, offset, interpolation='NEAREST').numpy()

    # adjust the coordinates of the patches, and their sizes
    new_label[l,:2] += translation
    i += int(np.floor(translation[0]))
    j += int(np.floor(translation[1]))
    if i + si < 0 or i >= mask.shape[0] or j + sj < 0 or j >= mask.shape[1]:
      # patch got shifted entirely outside of the frame
      continue
    if i < 0:
      si += i
      i = 0
      image_patch = image_patch[:si]
      mask_patch = mask_patch[:si]
    if i + si > mask.shape[0]:
      si = mask.shape[0] - i
      image_patch = image_patch[-si:]
      mask_patch = mask_patch[-si:]
    if j < 0:
      sj += j
      j = 0
      image_patch = image_patch[:,:sj]
      mask_patch = mask_patch[:,:sj]
    if j + sj > mask.shape[1]:
      sj = mask.shape[1] - j
      image_patch = image_patch[:,-sj:]
      mask_patch = mask_patch[:,-sj:]

    mask_patch = mask_patch.astype(np.bool)
    out = image_patch[mask_patch]
    new_image[i:i+si, j:j+sj][mask_patch] = out
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

transformations = {0 : normal_translate, 
                   1 : uniform_rotate,
                   2 : normal_scale,
                   3 : combine_1_2,
                   4 : combine_1_2_3}
