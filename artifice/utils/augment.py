"""Augmentation utils."""

import itertools
import functools
from artifice.utils import img, inpaint, tform
import tensorflow as tf
import logging

logger = logging.getLogger('artifice')


class Augmentation():
  """An augmentation is a callable that takes a tf.data.Dataset and applies
  transformations to its elements, returning a tf.data.Dataset with the
  transformed elements.

  The base class performs transformations by mapping them over the whole
  dataset, so the resulting dataset is an integer multiple times
  larger. Subclasses may change this behavior (e.g. BoundaryAwareAugmentation or
  SelectiveAugmentation).

  Augmentations can be added together using the "+" operator. This makes a new
  Augmentation with the transformations of both. Augmentations can also be
  "multiplied" (in a cartesian, compositional sense) together. This applies aug1
  to every output of aug2.

  Ideally, complex augmentations should be built up from small augmentations
  through addition or multiplication. To preserve the original dataset, build up
  augmentations by adding onto the identity, e.g.

  aug = (Augmentation()+Augmentation(t1))*(Augmentation()+Augmentation(t2+t3))

  will take a dataset d and output:

  {d, t1(d), (t2+t3)(d), (t1+t2+t3)(d)}

  """

  def __init__(self, transformation=None, **kwargs):
    """
    :param transformation: a Transformation object, or an iterable of them. Default
      (None) creates an identity augmentation.
    
    """
    if transformation is None:
      self._transformations = [tform.identity]
    elif issubclass(type(transformation), tform.Transformation):
      self._transformations = [transformation]
    elif hasattr(transformation, '__iter__'):
      self._transformations = list(transformation)
    else:
      raise ValueError()    

    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    

  def set_num_parallel_calls(self, num_parallel_calls):
    self.num_parallel_calls = num_parallel_calls
    return self

  def __add__(self, other):
    """Adds the transformations of OTHER, effectively combining the two
    augmentations without any composition.

    """
    if issubclass(type(other), Augmentation):
      return Augmentation(self._transformations + other._transformations)
    elif issubclass(type(other), tform.Transformation):
      return Augmentation(self._transformations + [other])
    else:
      raise ValueError(f"unrecognized type '{type(other)}'")
  
  def __mul__(self, other):
    """Composes each transformation in other onto the end of each transformation in
    self.

    """
    if issubclass(type(other), Augmentation):
      return Augmentation([ti + tj
                           for ti in self._transformations
                           for tj in other._transformations])
    elif issubclass(type(other), tform.Transformation):
      return Augmentation([t + other for t in self._transformations])
    else:
      raise ValueError(f"unrecognized type '{type(other)}'")

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)

  def __call__(self, dataset):
    """Apply transformations to a dataset.

    :param dataset: a tf.data.Dataset
    :returns: the transformed/augmented dataset

    """
    new_dataset = tf.data.Dataset()
    for transformation in self._transformations:
      new_dataset = new_dataset.concatenate(
        transformation.apply(dataset, self.num_parallel_calls))
      
    return new_dataset

  def __len__(self):
    return len(self._transformations)



# instantiations of simple Augmentations
identity = Augmentation()
brightness = identity + sum([Augmentation(tform.AdjustMeanBrightness(m))
                             for m in [0.2, 0.4, 0.6, 0.8]])

premade = {
  'identity': identity,
  'brightness': brightness,
}
choices = list(premade.keys())

def join(augmentation_names, num_parallel_calls=1):
  """From an iterable of augmentation_names, return their sum. If
  augmentation_names is empty, returns identity augmentation.

  """
  aug = Augmentation()
  for name in augmentation_names:
    aug += premade[name]
  aug.set_num_parallel_calls(num_parallel_calls)
  return aug

