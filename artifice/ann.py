"""Annotator class for providing annotations of an image (and maybe its index in
a dataset).

Simplest case given the full array of labels and lsit of annotation filenames,
which it can index into and retrieve.

"""



import numpy as np
import tensorflow as tf
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

  def __call__(self, *args, **kwargs):
    return self.annotate(*args, **kwargs)

  
class PerfectOracle(Oracle):
  """Query the existing labels (for testing purposes)"""
  def __init__(self, labels, annotation_paths):
    """A "perfect" annotator that uses premade labels/annotations

    :param labels: numpy array of labels 
    :param annotation_paths: list of filenames, indexes corresponding to labels

    """
    self.labels = labels
    self.annotation_paths = annotation_paths

  def annotate(self, image, idx):
    return (image, self.labels[idx]), np.load(self.annotation_paths[idx])

  
  
class HumanOracle(Oracle):
  """Query a human being."""
  pass
    
