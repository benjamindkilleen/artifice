"""Annotator class for providing annotations of an image (and maybe its index in
a dataset).

Simplest case given the full array of labels and lsit of annotation filenames,
which it can index into and retrieve.

"""



import numpy as np
import tensorflow as tf
from glob import glob
import logging


logger = logging.getLogger('artifice')


class Oracle:
  def annotate(self, image, idx=None):
    """Annotate a single image.

    Subclasses should overwrite this example, obtaining the annotation either
    from the example index (if pre-known), or from the image itself (if actually
    querying a user).

    :param image: numpy image
    :param idx: index of example, if needed
    :returns: `((image, label), annotation)` numpy scene, which can be parsed
    into a dataset for AugmentationData

    """
    raise NotImplementedError

  def label(self, image, idx=None):
    """Label a single image

    :param image: 
    :param idx: 
    :returns: `(image, label)` example pair
    :rtype: 

    """
    
    raise NotImplementedError
  
  def __call__(self, *args, **kwargs):
    return self.label(*args, **kwargs)

  
  
class PerfectOracle(Oracle):
  """Query the existing labels (for testing purposes)"""
  def __init__(self, labels, annotation_paths=None, **kwargs):
    """A "perfect" annotator that uses premade labels/annotations.

    Note that ALL the annotations must be provided. TODO: check this.

    :param labels: numpy array of labels 
    :param annotation_paths: list of filenames, indexes corresponding to labels

    """
    super().__init__(**kwargs)
    self.labels = labels
    self.annotation_paths = annotation_paths

  def label(self, image, idx):
    return (image, self.labels[idx])

  def annotate(self, image, idx):
    if self.annotation_paths is None:
      raise RuntimeError("Must provide annotations for PerfectOracle.annotate()")
    return (image, self.labels[idx]), np.load(self.annotation_paths[idx])

  
class HumanOracle(Oracle):
  """Query a human being."""
  pass
    
