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
    [(tf.equal(rank, tf.constant(1, rank.dtype)),
      lambda: tf.gather(t, [1,0])),
     (tf.equal(rank, tf.constant(2, rank.dtype)),
      lambda : tf.stack((t[:,1], t[:,0]), axis=1))],
   exclusive=True, name='swap')

def ensure_batched_images(inputs):
  """Adds dims to `inputs` until 4D.

  :param inputs: a numpy or tensor image or batch of images

  """
  if inputs.get_shape().ndims is None:
    raise TypeError("rank must be statically known")
  rank = len(inputs.get_shape())
  if rank == 2:
    images = tf.expand_dims(inputs, 0)
    images = tf.expand_dims(images, 3)
  elif rank == 3:
    images = tf.expand_dims(inputs, 0)
  elif rank == 4:
    images = inputs
  else:
    raise TypeError("Images should have rank between 2 and 4.")
  return images

def restore_image_rank(images, inputs=None, rank=None):
  """Restore original rank of `inputs` to `images`."""
  if inputs is not None:
    rank = len(inputs.get_shape())
  assert rank is not None
  if rank == 2:
    return images[0, :, :, 0]
  elif rank == 3:
    return images[0, :, :, :]
  else:
    return images

def ensure_batched_labels(inputs):
  """Adds dims to inputs until 3D (batched)."""
  if inputs.get_shape().ndims is None:
    raise TypeError("rank must be statically known")
  rank = len(inputs.get_shape())
  if rank == 2:
    labels = tf.expand_dims(inputs, 0)
  elif rank == 3:
    labels = inputs
  else:
    raise TypeError("Labels should have rank 2 or 3.")
  return labels
     
def restore_label_rank(labels, inputs=None, rank=None):
  """Restore original rank of `labels`."""
  if inputs is not None:
    rank = len(inputs.get_shape())
  assert rank is not None
  if rank == 2:
    return labels[0, :, :]
  else:
    return labels


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
  else:
    background = tf.constant(background, tf.float32)

  # (num_objects,) array
  object_order = tf.range(num_objects, dtype=tf.int64)
  object_order = tf.random.shuffle(object_order)
  # TODO: for batched inputs, vary object_order within each batch

  # fix obj_ids
  labels = labels[:,:,:tf.shape(new_labels)[2]]
  new_labels = tf.concat((labels[:,:,0:1], new_labels[:,:,1:]), axis=2)

  shape = tf.cast(tf.shape(images), tf.float32)
  center = tf.expand_dims(shape[1:3] / tf.constant(2,tf.float32), 0)
  new_images = tf.identity(images)
  
  for i in range(num_objects):
    logger.debug(f"transforming object {i}...")
    obj_images = tf.identity(images)
    obj_annotations = tf.identity(annotations)

    obj_idx = object_order[i]
    obj_labels = labels[:,obj_idx]
    new_obj_labels = new_labels[:,obj_idx]
    obj_ids = tf.reshape(obj_labels[:,0], [-1,1,1,1])

    # translate the object to center
    translations = swap(center - obj_labels[:,1:3])
    obj_images = tf.contrib.image.translate(
      obj_images, translations, interpolation='BILINEAR')
    obj_annotations = tf.contrib.image.translate(
      obj_annotations, translations, interpolation='NEAREST')

    # Rotate the objects
    angles = new_obj_labels[:,3] - obj_labels[:,3]
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

  new_image = restore_image_rank(new_images, image)
  new_label = restore_label_rank(new_labels, label)
  return new_image, new_label


