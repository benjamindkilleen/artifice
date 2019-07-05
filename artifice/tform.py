"""Transformation utils, used to build up augmentations.

NOT IN USE
todo: update

"""

import logging
import tensorflow as tf

logger = logging.getLogger('artifice')

def identity(image, label, annotation):
  return image, label

# todo: a bunch of transformations

transformations = {0, identity}

