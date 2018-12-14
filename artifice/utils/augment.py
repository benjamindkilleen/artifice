"""Augmentation utils."""

import numpy as np
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class Augmentation():
  """An augmentation is a "function" that takes a scene (image, annotation) and
  returns a list of scenes. By convenction, the first pair is always the
  original scene.

  Augmentations can be added together, without duplicating the original scene,
  using the "+" operator. This applies each augmentation to the original images
  separately.

  Augmentations can also be multiplied together. This applies aug1 to every
  output of aug2 (excluding the identity).

  :augmentation: the augmentation function
  :N: optional argument giving the number of scenes the augmentation outputs. 

  """
  def __init__(self, augmentation, N=None):
    self._augmentation = augmentation
    self.N

  def __call__(self, image, annotation):
    return self._augmentation(image, annotation)

  def __add__(self, aug):
    def augmentation(image, annotation):
      return self.__call__(image,annotation) + aug.__call__(image,annotation)[1:]
    N = None if (self.N is None or aug.N is None) else self.N + aug.N - 1
    
    return Augmentation(augmentation, N=N)

  def __mul__(self, aug):
    def augmentation(image, annotation):
      scenes = self.__call__(image, annotation)
      output_scenes = []
      for scene in scenes:
        output_scenes += aug.__call__(*scene)
      return output_scenes
    N = None if (self.N is None or aug.N is None) else self.N * aug.N
    return Augmentation(augmentation, N=N)
  

identity = Augmentation(lambda image, annotation : [image, annotation])

def _flip_horizontal(image, augmentation):
  flipped_image = np.fliplr(image)
  flipped_annotation = np.fliplr(annotation)
  return [(image, annotation),
          (flipped_image, flipped_annotation)]
flip_horizontal = Augmentation(_flip_horizontal, 2)

def _flip_vertical(image, augmentation):
  flipped_image = np.flipud(image)
  flipped_annotation = np.flipud(annotation)
  return [(image, annotation),
          (flipped_image, flipped_annotation)]
flip_vertical = Augmentation(_flip_vertical, 2)

# TODO: more augmentations
