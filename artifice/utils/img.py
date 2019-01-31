"""Util functions for manipulating images in artifice.
"""

import numpy as np
from PIL import Image
from skimage import transform
import scipy
import logging

logger = logging.getLogger('artifice')

def connected_components(annotation, num_classes=2):
  """Get connected components from an annotation.

  :param annotation: 3D annotation (as a numpy array). First channel contains
    semantic labels.
  :param num_classes: Number of classes in the annotation. Default=2.
  :returns: 
  :rtype: 

  """

  components = np.empty((annotation.shape[0], annotation.shape[1], num_classes),
                        dtype=np.int64)

  # TODO use the number of features returned by scipy.ndimage.label instead of
  # component_ids
  
  for obj_id in range(num_classes):
    components[:,:,obj_id], _ = scipy.ndimage.label(
      np.atleast_3d(annotation)[:,:,0] == obj_id)

  return components

def connected_component_indices(annotation, semantic_class, location,
                                components=None, num_classes=2, 
                                component_ids=None):
  if components is None:
    components = connected_components(annotation, num_classes=num_classes)
  semantic_class = int(semantic_class)
  x, y = location.astype(np.int64)
  component_id = components[x,y,semantic_class]
  if component_ids is not None:
    if component_id in component_ids[semantic_class]:
      return None;             # component already encountered
    else:
      component_ids[semantic_class].add(component_id)

  return np.where(components[:,:,semantic_class] == component_id)


def inpaint_image_background(image, indices, background_image=None, **kwargs):
  """Inpaint image using `background_image`.

  :param image: 3D numpy `image` to inpaint
  :param indices: array of indices into `image`
  :param background_image: image to inpaint with. If None, use mean value of `image`
  :returns: updated `image`
  :rtype: np.ndarray

  """
  if background_image is None:
    mu = image.mean()
    background_image = np.zeros_like(image)

  image[indices] = background_image[indices]
  return image

def inpaint_annotation(annotation, indices, location, semantic_label=0,
                       distance_cval=None, **kwargs):
  """Inpaint an annotation.

  If semantic_label == 0, then assume `indices` points to a background region
  and inpaint accordingly.

  If semantic_label > 0, assume `indices` points to an object region and inpaint
  with appropriate distance measure.

  :param annotation: 3D numpy `annotation` to inpaint
  :param indices: array indices into `annotation`
  :param location: object's location in index space.
    Ignored if `semantic_label = 0`.
  :param semantic_label: semantic label for `annotation[:,:,0]`. Default is 0
    (background).
  :param distance_cval: Constant value to use for background distance. Default
    is `max(annotation[:,:,1])`. Ignored if `semantic_label > 0`.
  :returns: updated `annotation`

  """
  
  if semantic_label == 0:
    if distance_cval is None:
      distance_cval = np.max(annotation[:,:,1])
    values = np.array([[semantic_label, distance_cval]])
  else:
    assert location is not None
    distances = np.linalg.norm(indices[:,:2] - location, axis=1)[:, np.newaxis]
    values = np.concatenate(
      (semantic_label * np.ones_like(distances), distances), axis=1)

  annotation[indices] = values
  return annotation


def inpaint_annotation_background(annotation, indices, distance_cval=None,
                                  **kwargs):
  """Inpaint a background region pointed to by indices.

  :param annotation: 
  :param indices: 
  :param semantic_label: 
  :param distance_cval: 
  :returns: updated `annotation`
  :rtype: 

  """
  return inpaint_annotation(annotation, indices, None,
                            semantic_label=0, distance_cval=distance_cval)


def inside(indices, image):
  """Returns a boolean array for which indices are inside image.shape.
  
  :param indices: 2D array of indices. Fast axis must have same dimension as shape.
  :param image: image to compare against
  :returns: 1-D boolean array
  
  """
  
  over = np.logical_and(indices[0] >= 0, indices[1] >= 0)
  under = np.logical_and(indices[0] < image.shape[0],
                         indices[1] < image.shape[1])

  return np.logical_and(over, under)


def get_inside(indices, image):
  """Get the indices that are inside image's shape.

  :param indices: 2D array of indices. Fast axis must have same dimension as shape.
  :param image: image to compare against 
  :returns: a subset of indices.
  
  """
  which = inside(indices, image)
  return indices[0][which], indices[1][which]


def grayscale(image):
  """Convert an n-channel, 3D image to grayscale.
  
  Use the [luminosity weighted average]
  (https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/)
  if there are three channels. Otherwise, just use the average.

  :param image: image to convert
  :returns: new grayscale image.

  """
  image = np.array(image)
  out_shape = (image.shape[0], image.shape[1], 1)
  if image.ndim == 2:
    return image.reshape(*out_shape)

  assert(image.ndim == 3)
  
  if image.shape[2] == 3:
    W = np.array([0.21, 0.72, 0.07])
    return (image * W).mean(axis=2).reshape(*out_shape).astype(np.uint8)
  else:
    return image.mean(axis=2).reshape(*out_shape).astype(np.uint8)
  

def resize(image, shape, label=None):
  resized = (255*transform.resize(image, shape, mode='reflect')).astype(np.uint8)
  if label is not None: 
    resized[resized != 0] = label
  return resized

def open_as_array(fname):
  im = Image.open(fname)
  if im.mode == 'L':
    image = np.array(im).reshape(im.size[1], im.size[0])
  elif im.mode == 'RGB':
    image = np.array(im).reshape(im.size[1], im.size[0], 3)
  elif im.mode == 'P':
    image = np.array(im.convert('RGB')).reshape(im.size[1], im.size[0], 3)
  elif im.mode == 'RGBA':
    image = np.array(im.convert('RGB')).reshape(im.size[1], im.size[0], 3)
  else:
    raise NotImplementedError("Cannot create image mode '{}'".format(im.mode))
  return image

def save(fname, image):
  """Save the array image to png in fname."""
  im = Image.fromarray(image)
  im.save(fname)




