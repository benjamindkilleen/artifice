"""Transformation utils, used to build up augmentations."""

from artifice.utils import img, vis
import tensorflow as tf
import logging
from scipy.ndimage import interpolation
import numpy as np

logger = logging.getLogger('artifice')

def transform(image, label, annotation, new_label,
              background=None, num_objects=2):
  """Transform an example to match 'new_label`.

  Transforms each object in a random order, so objects overlap in random
  order. This can be made more intelligent.

  Currently only performs translations.

  :param image: tensor image
  :param label: tensor label of shape (num_objects, >3)
  :param annotation: instance annotation, with 0 as background class
  :param new_label: desired label
  :param background: background to fill in with, same shape as image. Default is 0s.
  :param num_objects: 
  :returns: new example `(new_image, new_label)`, object ids corrected in new_label
  :rtype: (tf.Tensor, tf.Tensor)

  """
  if background is None:
    background = tf.zeros_like(image)
  
  object_order = tf.range(num_objects, dtype=tf.int64)
  object_order = tf.random.shuffle(object_order)

  # fix the instance labels for new_label
  label = label[:,:tf.shape(new_label)[1]]
  id_indices = tf.one_hot(tf.constant(0, tf.int64, (num_objects,)), tf.shape(label)[1])
  new_label = tf.where(tf.cast(id_indices, tf.bool), label, new_label,
                       name='fix_new_label')

  new_image = tf.identity(image)
  
  for i in range(num_objects):
    obj_image = tf.identity(image)
    obj_annotation = tf.identity(annotation)
    
    obj_idx = object_order[i]
    obj_label = label[obj_idx]
    new_obj_label = new_label[obj_idx]
    obj_id = obj_label[0]

    # translate the object
    translation = new_obj_label[1:3] - obj_label[1:3]
    translation = tf.gather(translation, [1,0]) # translate flips x and y
    obj_image = tf.contrib.image.translate(
      obj_image, translation, interpolation='BILINEAR')
    obj_annotation = tf.contrib.image.translate(
      obj_annotation, translation, interpolation='NEAREST')
    
    new_image = tf.where(tf.equal(annotation, obj_id), background, new_image)
    new_image = tf.where(tf.equal(obj_annotation, obj_id), obj_image, new_image)
        
  return new_image, new_label


