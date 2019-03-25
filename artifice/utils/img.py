"""Util functions for manipulating images in artifice.
"""

import numpy as np
from PIL import Image
from skimage import transform
import scipy
import logging

logger = logging.getLogger('artifice')

def fill_negatives(image):
  """Fill the negative values in background with gaussian noise.
  
  :param image: a numpy array with negative values to fill
  
  """
  image = image.copy()
  indices = image >= 0
  mean = image[indices].mean()
  std = image[indices].std()
  
  indices = image < 0
  image[indices] = np.random.normal(mean, std, size=image[indices].shape)
  return image

def inside(indices, shape):
  """Returns a boolean array for which indices are inside shape.
  
  :param indices: 2D array of indices. Fast axis must have same dimension as shape.
  :param shape: image shape to compare against, using first two dimensions
  :returns: 1-D boolean array
  
  """
  
  over = np.logical_and(indices[0] >= 0, indices[1] >= 0)
  under = np.logical_and(indices[0] < shape[0],
                         indices[1] < shape[1])

  return np.logical_and(over, under)


def get_inside(indices, shape):
  """Get the indices that are inside image's shape.

  :param indices: 2D array of indices. Fast axis must have same dimension as shape.
  :param shape: image shape to compare with
  :returns: a subset of indices.

  """
  which = inside(indices, shape)
  return indices[0][which], indices[1][which]


def grayscale(image):
  """Convert an n-channel, 3D image to grayscale.
  
  Use the [luminosity weighted average]
  (https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/)
  if there are three channels. Otherwise, just use the average.

  :param image: image to convert
  :returns: new grayscale image.

  """
  image = np.array(image)
  out_shape = (image.shape[0], image.shape[1], 1)
  if image.ndim == 2:
    return image.reshape(*out_shape)

  assert(image.ndim == 3)
  
  if image.shape[2] == 3:
    W = np.array([0.21, 0.72, 0.07])
    return (image * W).mean(axis=2).reshape(*out_shape).astype(np.uint8)
  else:
    return image.mean(axis=2).reshape(*out_shape).astype(np.uint8)

def open_as_array(fname):
  im = Image.open(fname)
  if im.mode == 'L':
    image = np.array(im).reshape(im.size[1], im.size[0])
  elif im.mode == 'RGB':
    image = np.array(im).reshape(im.size[1], im.size[0], 3)
  elif im.mode == 'P':
    image = np.array(im.convert('RGB')).reshape(im.size[1], im.size[0], 3)
  elif im.mode == 'RGBA':
    image = np.array(im.convert('RGB')).reshape(im.size[1], im.size[0], 3)
  else:
    raise NotImplementedError("Cannot create image mode '{}'".format(im.mode))
  return image

def save(fname, image):
  """Save the array image to png in fname."""
  im = Image.fromarray(image)
  im.save(fname)




