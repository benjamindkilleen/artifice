"""Augmentation utils."""

import numpy as np
import itertools
import functools
from functional import compose
from artifice.utils import img
import tensorflow as tf
import logging

logger = logging.getLogger('artifice')

class Transformation():
  """A transformation is a callable that takes tensors representing an example
  (usually a scene) and returns a new pair. It should be mappable over a
  tf.data.Dataset.

  Transformations are meant to be aggressively subclassed. This allows families
  of transformations which, although not totally identitcal, to belong
  together. These can define the transform as a method rather than passing it in
  on initialization.

  Transformations can be composed (or "added together") using the "+"
  operator.

  """

  def __init__(self, transform=lambda *scene : scene):
    """
    :transforms: a transformation function or an iterable of them.
    """
    if callable(transform):
      self._transforms = [transform]
    elif hasattr(transform, '__iter__'):
      self._transforms = list(transform)
    else:
      raise ValueError()

  def __call__(self, *scene):
    return functools.reduce(compose, self._transforms)(*scene)

  def __add__(self, other):
    return Transformation(self._transforms + other._transforms)

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)

# TODO: various transformations, either taking or defining methods, with
# transform functions.

identity_transformation = Transformation()



class Augmentation():
  """An augmentation is a callable that takes a tf.data.Dataset and applies an
  augmentation to its elements, returning a larger tf.data.Dataset with the
  original elements, as well as any added ones, according to the provided
  Transformation object(s).

  The base class performs transformations by mapping them over the whole
  dataset, so the resulting dataset is an integer multiple times
  larger. Subclasses may change this behavior (e.g. BoundaryAwareAugmentation or
  SelectiveAugmentation).

  Augmentations can be added together using the "+" operator. This applies each
  augmentation to the original images separately, then concatenates
  them. Augmentations can also be "multiplied" (in a cartesian, compositional
  sense) together. This applies aug1 to every output of aug2.

  Ideally, complex augmentations should be built up from small augmentations
  through addition or multiplication.

  """

  def __init__(self, transformation=[]):
    """
    :transformation: a Transformation object, or an iterable of them.
    
    TODO: allow for no-identity augmentations?
    """
    if issubclass(type(transformation), Transformation):
      self._transformations = [identity_transformation] + [transformation]
    elif hasattr(transformation, '__iter__'):
      self._transformations = [identity_transformation] + list(transformation)
    else:
      raise ValueError()    

  def __add__(self, other):
    """Adds the transformations of OTHER, effectively combining the two
    augmentations without any composition."""
    return Augmentation(self._transformations + other._transformations[1:])
    
  def __mul__(self, other):
    """Composes each transformation in other onto the end of each transformation in
    self.

    """
    return Augmentation([ti + tj
                         for ti in self._transformations
                         for tj in other._transformations])

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)

  def __call__(self, dataset):
    """Take a tf.data.Dataset object and apply transformations to it."""

    # TODO: don't skip over identity, by convention.
    for transformation in self._transformations[1:]:
      dataset = dataset.concatenate(dataset.map(transformation))
      
    return dataset


identity = Augmentation()

