"""Augmentation utils."""

import numpy as np
import itertools
import functools
from functional import compose
from artifice.utils import img
from collections import Counter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Augmentation():
  """An augmentation is a callable that takes a scene (image, annotation) and
  returns an iterable of scenes. Mathematically, it is a set of transformations.
  Regardless of transformations, the first scene yielded is always the
  original.

  Augmentations can be added together, without duplicating the original scene,
  using the "+" operator. This applies each augmentation to the original images
  separately. Augmentations can also be "multiplied" (in a cartesian,
  compositional sense) together. This applies aug1 to every output of aug2.

  Ideally, augmentations should be built up from small augmentation functions
  through addition or multiplication. When called, scenes are then generated as
  lazily as possible.

  """
  def __init__(self, transformations=[]):
    """input:
    :transformations: iterable of transformation functions (scene -> scene). Can
      also be a single callable.
    """
    if callable(transformations):
      self._transformations = iter([transformations])
    else:
      self._transformations = iter(transformations)

  def __len__(self):
    return len(self._transformations) + 1
      
  def __call__(self, scene):
    yield scene
    for t in self._transformations:
      yield t(scene)

  def __add__(self, other):
    ts = itertools.chain(self._transformations, other._transformations)
    return Augmentation(ts)

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)

  def __mul__(self, other):
    pairs = itertools.product(self._transformations, other._transformations)
    ts = itertools.starmap(compose, pairs)
    return Augmentation(ts)


def _flip_horizontal(scene):
  image, annotation = scene
  flipped_image = np.fliplr(image)
  flipped_annotation = np.fliplr(annotation)
  return iter([(image, annotation),
               (flipped_image, flipped_annotation)])

def _flip_vertical(scene):
  image, annotation = scene
  flipped_image = np.flipud(image)
  flipped_annotation = np.flipud(annotation)
  return iter([(image, annotation),
               (flipped_image, flipped_annotation)])

def _random_thumbnail(scene):
  """Zoom in on a random frame, three_quarters the size of the original image."""
  image, annotation = scene
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

  return iter([(image, annotation),
               (resized_image_thumb, resized_annotation_thumb)])


# Premade augmentations
identity = Augmentation()
flip_horizontal = Augmentation(_flip_horizontal, 2)
flip_vertical = Augmentation(_flip_vertical, 2)
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
