"""Utils relating to images and annotations as tf.Tensors."""

import tensorflow as tf

def connected_components(annotation, num_classes=2):
  """Get connected components in an annotation.

  :param annotation: 3D image annotation, where the first channel contains semantic
    labels.
  :param num_classes: number of classes in the annotation. Default=2.
  :returns: a tensor of connected components, with a different channel for every
  semantic class (including background).
  """
  components = []
  for obj_id in range(num_classes):
    components.append(tf.contrib.images.connected_components(
      tf.equal(annotation[:,:,0], tf.constant(obj_id, annotation.dtype))))
    
  return tf.stack(components, axis=3, name='connected_components')


def connected_component_indices(annotation, semantic_class, location,
                                 num_classes=2, components=None,
                                 component_ids=None):
  """Get the indices for the connected component at `location` with
  `semantic_class`.

  :param annotation: annotation of the image
  :param semantic_class: integer class of the desired component
  :param location: image-space location of the object's center
  :param num_classes: number of classes in the annotation
  :param components: components to use, if precalculated. Default is None.
  :param component_ids: None, or a list of sets containing previously seen
  component_ids. Behavior is undefined if components is None.
  :returns: indices of the connected component at `location`, or None if
  component previously seen.
  :rtype: 

  """
  if components is None:
    components = connected_components(annotation, num_classes=num_classes)
  semantic_class = tf.cast(semantic_class, tf.int64)
  x = tf.cast(location[0], tf.int64)
  y = tf.cast(location[1], tf.int64)
  component_id = components[x,y,semantic_class]
  if component_ids is not None:
    if component_id in component_ids[semantic_class]:
      return None;             # component already encountered
    else:
      component_ids[semantic_class].add(component_id)

  return tf.where(tf.equal(components[:,:,semantic_class], component_id))


def inside(indices, image):
  """Returns a boolean array for which indices are inside image.shape.
  
  :indices: 2D tensor of indices. Fast axis must have same dimension as shape.
  :image: image to get the 
  
  Returns: 1-D boolean tensor
  
  """
  
  over = tf.greater_equal(indices, tf.constant(0, dtype=indices.dtype))
  under = tf.less(indices, tf.shape(image))
  return tf.reduce_any(tf.logical_and(over, under), axis=1, 
                       name='inside')


def get_inside(indices, image):
  """Get the indices that are inside image's shape.
  :indices: 2D tensor of indices. Fast axis must have same dimension as shape.
  :image: image to get the 
  
  Returns: a subset of indices.
  
  """
  
  return tf.gather(indices, tf.where(inside(indices, image)), 
                   name='get_inside')

