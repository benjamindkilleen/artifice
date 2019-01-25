"""Utils relating to tf.Tensors."""

import tensorflow as tf


def get_component_indices(annotation, object_label):
  """Get the indices associated with the object at image-space point x. If the
  surrounding pixels are associated with the background, returns an empty
  tensor. If the object is connected to another object, returns a component
  with both. 

  :annotation: 4D annotation tensor for the image, containing semantic labels
  in the first channel.
  :object_label: the 1D label for the desired object.
  
  Returns: a Nx3 tensor of indices into annotation.
  
  """
  
  x, y = object_label[1:3]
  components = tf.contrib.image.connected_components(
    tf.equal(annotation[:,:,:,0], object_label[0]))
  component_id = components[tf.cast(x, tf.int64),tf.cast(y, tf.int64)]
  return tf.where(tf.equal(components, component_id), 
                  name='get_object_indices')


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

