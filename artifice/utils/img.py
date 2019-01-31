"""Util functions for manipulating images in artifice.
"""

import numpy as np
from PIL import Image
from skimage import transform
import scipy

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
  for obj_id in range(num_classes):
    components[:,:,obj_id] = scipy.ndimage.measurements.label(
      annotation[:,:,0] == obj_id)

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

def inpaint_annotation_background(annotation, indices, bg_semantic=0, bg_distance=None
                                  **kwargs):
  """Inpaint annotation.

  :param annotation: 3D numpy `annotation` to inpaint
  :param indices: array indices into `annotation`
  :param bg_semantic: semantic label for `annotation[:,:,0]`. Default is 0.
  :param bg_distance: Default is `max(annotation[:,:,1])`
  :returns: updated `annotation`
  """
  
  if bg_distance is None:
    bg_distance = np.max(annotation[:,:,1])

  value = np.array([bg_semantic, bg_distance])
  annotation[indices] = value
  return annotation

def grayscale(image):
  """Convert an n-channel image to grayscale, with ndim = 3, using the luminosity
  weighted average if there are three channels. Otherwise, just use the average.
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




