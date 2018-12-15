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
  def __init__(self, augmentation=lambda i, a : [(i,a)], N=None):
    self._augmentation = augmentation
    self.N = N

  def __call__(self, image, annotation):
    return self._augmentation(image, annotation)

  def __add__(self, aug):
    def augmentation(image, annotation):
      return self._augmentation(image,annotation) + aug._augmentation(image,annotation)[1:]
    N = None if (self.N is None or aug.N is None) else self.N + aug.N - 1
    
    return Augmentation(augmentation, N=N)

  def __mul__(self, aug):
    def augmentation(image, annotation):
      scenes = self._augmentation(image, annotation)
      output_scenes = []
      for scene in scenes:
        output_scenes += aug._augmentation(*scene)
      return output_scenes
    N = None if (self.N is None or aug.N is None) else self.N * aug.N
    return Augmentation(augmentation, N=N)


identity = Augmentation()

def _flip_horizontal(image, annotation):
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
random_thumbnail = Augmentation(_random_thumbnail, 2)


premade = {
  'identity'         : identity,
  'flip_horizontal'  : flip_horizontal,
  'fliph'            : flip_horizontal,
  'flip_vertical'    : flip_vertical,
  'flipv'            : flip_vertical,
  'random_thumbnail' : random_thumbnail,
  'rtn'              : random_thumbnail
}


def join(augs):
  """Accept a list of names of augmentations, or augmentation objects
  themselves, and coalesce into a single augmentation.
  """

  aug_out = Augmentation()
  for aug in augs:
    if type(aug) == str:
      assert(premade.get(aug) is not None)
      aug_out += premade[aug]
    elif type(aug) == Augmentation:
      aug_out += aug
    else:
      raise RuntimeError("unrecogignized augmentation")

  return aug_out
  
# TODO: more augmentations
