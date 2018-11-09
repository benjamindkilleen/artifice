"""Util functions for manipulating images in artifice.
"""

import numpy as np


def grayscale(image):
  """Convert an n-channel image to grayscale, with ndim = 3, using the luminosity
  weighted average if there are three channels. Otherwise, just use the average.
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
  
    
