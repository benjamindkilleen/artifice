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

  def __call__(self, *args, **kwargs):
    return self.annotate(*args, **kwargs)

  
class PreparedOracle(Oracle):
  """Query the existing labels (for testing purposes)"""
  def __init__(self, labels, annotations_dir, images_dir=None, **kwargs):
    """Uses premade labels/annotations.

    If images_dir is not None, matches annotation basenames to image basenames
    in an images_dir, providing a set of "allowed" indices to choose
    from. Otherwise, assumes annotations are sequential starting at 0, sorted by
    name.

    :param labels: numpy array of labels 
    :param annotations_dir: directory to find .npy annotation files
    :param images_dir: optional directory to find .png image files

    """
    super().__init__(**kwargs)
    self.labels = labels
    self.annotations_dir = annotations_dir
    self.images_dir = images_dir
    self.annotation_paths, self.valid_indices = self._create_annotation_paths()

  def _create_annotation_paths(self):
    # Determine annotation_paths, list of path names (or None if doesn't exist),
    # and valid_indices, where annoation_paths is not None
    if self.images_dir is None:
      annotation_paths = sorted(glob(self.annotations_dir + '*.npy'))
      valid_indices = list(range(len(self.annotation_paths)))
    else:
      ann_paths = iter(sorted(set(glob(self.annotations_dir + '*.npy'))))
      image_paths = sorted(glob(self.images_dir + '*.png'))
      ann_path = next(ann_paths)
      ann_name = splitext(basename(ann_path))[0]
      annotation_paths = [None] * len(image_paths)
      logger.debug(f"searching for matching annotations in {self.annotations_dir}")
      for i, image_path in enumerate(image_paths):
        if ann_name == splitext(basename(image_path))[0]:
          annotation_paths[i] = ann_path
          valid_indices.append(i)
          ann_path = next(ann_paths)
          ann_name = splitext(basename(ann_path))[0]
    logger.debug(f"found {len(valid_indices)} annotations")
    return annotation_paths, valid_indices
  
  def annotate(self, image, idx):
    if self.annotation_paths[idx] is None:
      raise RuntimeError(f"no annotation for image {idx}")
    return (image, self.labels[idx]), np.load(self.annotation_paths[idx])

  
class HumanOracle(Oracle):
  """Query a human being."""
  pass
    
