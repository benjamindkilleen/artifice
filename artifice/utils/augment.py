"""Augmentation utils."""

import numpy as np
from artifice.utils import img
from collections import Counter
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
    self.N = N

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

def _flip_vertical(image, annotation):
  flipped_image = np.flipud(image)
  flipped_annotation = np.flipud(annotation)
  return [(image, annotation),
          (flipped_image, flipped_annotation)]
flip_vertical = Augmentation(_flip_vertical, 2)

def _random_thumbnail(image, annotation):
  """Zoom in on a random frame, three_quarters the size of the original image."""
  shape = (0.75*np.array(image.shape)).astype(np.int)
  xoffset = np.random.randint(image.shape[0] - shape[0])
  yoffset = np.random.randint(image.shape[1] - shape[1])

  image_thumb = image[xoffset:xoffset + shape[0], yoffset:yoffset + shape[1]]
  annotation_thumb = annotation[xoffset:xoffset + shape[0], yoffset:yoffset + shape[1]]

  # Clean up interpolation:
  counter = Counter(annotation.flatten())
  label = 1 if counter[1] > counter[2] else 2

  resized_image_thumb = img.resize(image_thumb, image.shape)
  resized_annotation_thumb = img.resize(
    annotation_thumb, annotation.shape, label=label)

  return [(image, annotation),
          (resized_image_thumb, resized_annotation_thumb)]


# TODO: more augmentations
