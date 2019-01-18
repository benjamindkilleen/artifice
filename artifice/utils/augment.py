"""Augmentation utils."""

import numpy as np
import itertools
import functools
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

  Rather than pass in a function, subclasses may optionally define a "transform"
  method, which is taken at instantiation. This is supported but not preferred.

  """

  def __init__(self, transform=lambda *scene : scene):
    """
    :transforms: a transformation function or an iterable of them. Ignored if
      object has a "transform" method.
    """
    if hasattr(self, 'transform'):
      assert callable(self.transform)
      self._transforms = [lambda *scene : self.transform(*scene)]
    elif callable(transform):
      self._transforms = [transform]
    elif hasattr(transform, '__iter__'):
      self._transforms = list(transform)
    else:
      raise ValueError()

  def __call__(self, *scene):
    for transform in self._transforms:
      scene = transform(*scene)
    return scene

  def __add__(self, other):
    return Transformation(self._transforms + other._transforms)

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)


"""For many transformations, Simple- and ImageTransformations should be
sufficient, and they may instantiated with transform functions on their own,
depending on whether that transform should applied to both image and annotation
(SimpleTransformation) or to the image alone (ImageTransformation)

Transformations that treat image and annotation separately should inherit from
Transformation directly.
"""
class SimpleTransformation(Transformation):
  """Applies the same tensor function to both image and annotation, clipping image
  values."""
  def __init__(self, function):
    def transform(image, annnotation):
      image = function(image)
      image = tf.clip_by_value(image,
                               tf.constant(0, dtype=image.dtype),
                               tf.constant(1, dtype=image.dtype))
      return image, function(annotation)
    super().__init__(transform)


class ImageTransformation(Transformation):
  """Applies a tensor function to the image (and clips), leaving annotation."""
  def __init__(self, function):
    def transform(image, annotation):
      image = function(image)
      image = tf.clip_by_value(image,
                               tf.constant(0, dtype=image.dtype),
                               tf.constant(1, dtype=image.dtype))
      return image, annotation
    super().__init__(transform)

"""The following are transformations that must be instantiated with a
parameter separate from the arguments given to the transformation function."""
class AdjustBrightness(ImageTransformation):
  """Adjust the brightness of the image by delta."""
  def __init__(self, delta):
    def transform(image):
      return tf.image.adjust_brightness(image, delta)
    super().__init__(transform)


class AdjustMeanBrightness(ImageTransformation):
  """Adjust the mean brightness of grayscale images to mean_brightness. Afterward,
  clip values as appropriate. Thus the final mean brightness might not be the
  value passed in. Keep this in mind. 

  """
  def __init__(self, new_mean): 
    def transform(image):
      mean = tf.reduce_mean(image)
      delta = tf.constant(new_mean, dtype=mean.dtype) - mean
      return tf.image.adjust_brightness(image, delta)
    super().__init__(transform)


# Transformation instances.
identity_transformation = Transformation()
flip_left_right_transformation = SimpleTransformation(tf.image.flip_left_right)
flip_up_down_transformation = SimpleTransformation(tf.image.flip_up_down)
invert_brightness_transformation = ImageTransformation(lambda image : 1 - image)



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

  def __init__(self, transformation=None, num_parallel_calls=None):
    """
    :transformation: a Transformation object, or an iterable of them. Default
      (None) creates an identity augmentation.
    
    
    TODO: allow for no-identity augmentations?
    """
    if transformation is None:
      self._transformations = [identity_transformation]
    elif issubclass(type(transformation), Transformation):
      self._transformations = [identity_transformation] + [transformation]
    elif hasattr(transformation, '__iter__'):
      self._transformations = [identity_transformation] + list(transformation)
    else:
      raise ValueError()    

    self.num_parallel_calls = num_parallel_calls
    

  def set_num_parallel_calls(self, num_parallel_calls):
    self.num_parallel_calls = num_parallel_calls
    return self

  def __add__(self, other):
    """Adds the transformations of OTHER, effectively combining the two
    augmentations without any composition. Removes the identity from the other.

    """
    return Augmentation(self._transformations[1:] + other._transformations[1:])
  
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
    # TODO: don't skip over identity, by convention?
    for transformation in self._transformations[1:]:
      dataset = dataset.concatenate(dataset.map(transformation,
                                                self.num_parallel_calls))
      
    return dataset

  def __len__(self):
    return len(self._transformations)


# instantiations of Augmentations
identity = Augmentation()
brightness = sum([Augmentation(AdjustMeanBrightness(m)) 
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

