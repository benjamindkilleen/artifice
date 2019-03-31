"""Transformation utils, used to build up augmentations."""

from artifice.utils import img, vis
import tensorflow as tf
import logging
from scipy.ndimage import interpolation
import numpy as np

logger = logging.getLogger('artifice')

def swap(t):
  rank = tf.rank(t)
  return tf.case(
    [(lambda: tf.equal(rank, tf.constant(1, rank.dtype)), tf.gather(t, [1,0])),
     (lambda: tf.equal(rank, tf.constant(2, rank.dtype)),
      tf.concat((t[:,1], t[:,0]), axis=1))],
   exclusive=True, name='swap')

def ensure_batched_images(inputs):
  """Ensure that `inputs` is a batched tensor.

  :param inputs: the tensor input
  :param rank: optional rank of tensor
  :returns: tensor with expanded dimensions

  """
  rank = tf.rank(inputs)
  return tf.case(
    [(tf.equal(rank, tf.constant(2, rank.dtype)),
      lambda: tf.expand_dims(tf.expand_dims(inputs,0), 3)),
     (tf.equal(rank, tf.constant(3, rank.dtype)), lambda: tf.expand_dims(inputs, 0)),
     (tf.equal(rank, tf.constant(4, rank.dtype)), lambda: inputs)],
   exclusive=True, name='ensure_batched_images')

def ensure_image_rank(inputs, rank):
  """Undo the ensure_batched operation on `inputs`.

  :param inputs: tensor of rank 4
  :param rank: desired or "original" rank of `inputs`
  :returns: sliced tensor with `rank`

  """
  return tf.case(
    [(tf.equal(rank, tf.constant(2, rank.dtype)), lambda: inputs[0,:,:,0]),
     (tf.equal(rank, tf.constant(3, rank.dtype)), lambda: inputs[0,:,:,:]),
     (tf.equal(rank, tf.constant(4, rank.dtype)), lambda: inputs)],
    exclusive=True)

def ensure_batched_labels(inputs):
  rank = tf.rank(inputs)
  return tf.case(
    [(tf.equal(rank, tf.constant(2, rank.dtype)), lambda: tf.expand_dims(inputs, 0)),
     (tf.equal(rank, tf.constant(3, rank.dtype)), lambda: inputs)],
    exclusive=True, name='ensure_batched_labels')
     
def ensure_label_rank(inputs, rank):
  return tf.case(
    [(tf.equal(rank, tf.constant(2, rank.dtype)), lambda: inputs[0,:,:]),
     (tf.equal(rank, tf.constant(3, rank.dtype)), lambda: inputs)],
    exclusive=True)


def transform_objects(image, label, annotation, new_label,
                      background=None, num_objects=2):
  """Transform an example to match 'new_label`.

  Transforms each object in a random order, so objects overlap in random
  order. This can be made more intelligent.

  :param image: tensor image, possibly batched
  :param label: tensor label of shape (num_objects, >=3), possibly batched.
  :param annotation: instance annotation, with 0 as background class
  :param new_label: desired label
  :param background: background to fill in with, same shape as image. Default is 0s.
  :param num_objects: 
  :returns: new example `(new_image, new_label)`, object ids corrected in new_label
  :rtype: (tf.Tensor, tf.Tensor)

  """

  images = ensure_batched_images(image)
  labels = ensure_batched_labels(label)
  annotations = ensure_batched_images(annotation)
  new_labels = ensure_batched_labels(new_label)

  batch_size = tf.shape(images)[0]
  obj_label_size = tf.shape(labels)[2]
  
  if background is None:
    background = tf.zeros_like(images)

  # (num_objects,) array
  object_order = tf.constant(num_objects, dtype=tf.int64)
  object_order = tf.random.shuffle(object_order)
  # TODO: for batched inputs, vary object_order within each batch

  # fix the obj_id's for new_label
  labels = labels[:,:,:tf.shape(new_labels)[2]]
  id_indices = tf.tile(tf.reshape(
    tf.one_hot(tf.constant(0, tf.int64, (num_objects,)), obj_label_size),
    [1, num_objects, obj_label_size]), [batch_size, 1])
  new_labels = tf.where(tf.cast(id_indices, tf.bool), labels, new_labels)

  shape = tf.cast(tf.shape(images), tf.float32)
  center = tf.expand_dims(shape[1:3] / tf.constant(2,tf.float32), 0)
  new_images = tf.identity(images)
  
  for i in range(num_objects):
    obj_images = tf.identity(images)
    obj_annotations = tf.identity(annotations)
    
    obj_idx = object_order[i]
    obj_labels = labels[:,obj_idx]
    new_obj_labels = new_label[:,obj_idx]
    obj_ids = obj_label[:,0]

    # translate the object to center
    translations = swap(center - obj_labels[:,1:3])
    obj_images = tf.contrib.image.translate(
      obj_images, translations, interpolation='BILINEAR')
    obj_annotation = tf.contrib.image.translate(
      obj_annotations, translations, interpolation='NEAREST')

    # Rotate the objects
    angles = new_obj_labels[3] - obj_labels[3]
    obj_images = tf.contrib.image.rotate(
      obj_images, angles, interpolation='BILINEAR')
    obj_annotations = tf.contrib.image.rotate(
      obj_annotations, angles, interpolation='NEAREST')

    # translate the object to desired position
    translations = swap(new_obj_labels[:,1:3] - center)
    obj_images = tf.contrib.image.translate(
      obj_images, translations, interpolation='BILINEAR')
    obj_annotations = tf.contrib.image.translate(
      obj_annotations, translations, interpolation='NEAREST')
    
    new_images = tf.where(tf.equal(annotations, obj_ids), background, new_images)
    new_images = tf.where(tf.equal(obj_annotations, obj_ids), obj_images, new_images)

  new_image = ensure_image_rank(new_images, tf.rank(image))
  new_label = ensure_label_rank(new_labels, tf.rank(label))
  return new_image, new_label


