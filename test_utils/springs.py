"""Analayze positions of two spheres to try to gauge the spring constant
connecting the two."""

import numpy as np
from artifice import vis


def find_constant(labels, tethered=[]):
  """

  :param labels: `(num_examples, num_objects, label_dim)` array of labeles or
  detections
  :param tethered: list of object IDs which are tethered to the center of the
  image.
  :returns: predicted spring constant
  :rtype: 

  """
  
  positions = labels[:,:,1:3]
  ls = np.linalg.norm(positions[:,1] - positions[:,0], axis=-1)
  return ls
