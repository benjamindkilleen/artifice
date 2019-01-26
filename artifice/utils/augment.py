"""Augmentation utils."""

import itertools
import functools
from artifice.utils import img, inpaint, transformations
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

  def __init__(self, transformation=None, 
               num_parallel_calls=None):
    """
    :transformation: a Transformation object, or an iterable of them. Default
      (None) creates an identity augmentation.
    
    """
    if transformation is None:
      self._transformations = [identity_transformation]
    elif issubclass(type(transformation), Transformation):
      self._transformations = [transformation]
    elif hasattr(transformation, '__iter__'):
      self._transformations = list(transformation)
    else:
      raise ValueError()    

    self.num_parallel_calls = num_parallel_calls
    

  def set_num_parallel_calls(self, num_parallel_calls):
    self.num_parallel_calls = num_parallel_calls
    return self

  def __add__(self, other):
    """Adds the transformations of OTHER, effectively combining the two
    augmentations without any composition.

    """
    return Augmentation(self._transformations + other._transformations)
  
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
    """Take a tf.data.Dataset object and apply transformations to it.

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
brightness = identity + sum([Augmentation(transformations.AdjustMeanBrightness(m)) 
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


"""Brainstorm: We need to design a new type of augmentation that can use any
example in the whole dataset. Should this be done on the fly? Unclear. Perhaps
the boundary-aware augmentations simply need to

The regular Augmentation object returns a new dataset object that includes the
old one. To facilitate persistent augmentations, which are aware of the whole
dataset, we will have to do eager execution, performing these augmentation steps
before every iteration of training. Essentially, this new augmentation object
should do the same thing as above, but it should have a method that returns just
the transformed examples. (Could also just modify above augmentation so that
identity transformation is not always implied, then have an optional argument
in the call that preserves it. Yup.)

So but the new method needs to collect statistics for the labels on the whole
dataset. These should fit in memory; no more than a few items per image. This
run-through should also assemble a distribution for every pixel (more memory
intensive, perhaps only use a helpful subsample?) that constructs a reasonable
background to inpaint with. If no background can be used for a region (object is
stationary), then fill with noise (TODO: better strategy?)

The new examples will have to be based on statistics in the label space. We'll
have to look at these distributions as we build it up. For speed's sake, perhaps
the labels should be stored separately from the tfrecord as well?

All augmentation tfrecords can be stored in a data_dir directory, provided by
command line. By default, this data_dir can be the same as the tfrecord from
which it came? Or just a "tmp_data" dir in the cwd?

This type of augmentation should be used by a class in the dataset module. This
class should be related to DataInput objects, and maybe it can be made from one
of those, but it is a fundamentally different object. This is not a DataInput
object, used for feeding data into a network. This is a DataAugmenter object. It
should take one or more tfrecord files, 

"""

