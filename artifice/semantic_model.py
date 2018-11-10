"""Provides semantic segmentation capability for artifice. The specific
algorithm isn't important, as long as it returns an output in familiar form.

We implement the U-Net segmentation architecture
(http://arxiv.org/abs/1505.04597), emulating implementations at
* https://github.com/tks10/segmentation_unet/

"""

import os
import numpy as np
import logging
from collections import CollectDict

import tensorflow as tf
from artifice.utils import dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


"""A model implementing semantic segmentation.
args:
* channels: number of channels in the image (grayscale by default)
* num_classes: number of classes of objects to be detected (including default)
"""
class SemanticModel:
  def __init__(self, image_shape, num_classes=2):
    self.image_shape = image_shape
    self.num_classes = num_classes

    self.image, self.prediction, self.annotation, self.training = (
      self.create_model(self.image_shape, self.num_classes))
    
  @staticmethod
  def create_model(image_shape, num_classes, l2_reg_scale=None):
    raise NotImplementedError("SemanticModel subclass should implement create_model.")


"""Implementation of UNet."""
class UNet(SemanticModel):
  @staticmethod
  def create_model(image_shape, num_classes, l2_reg_scale=None):
    """Create the unet graph. Should only be called by init, after self.x,
    self.channels, and self.n_class have been set.

    args:
    * image_shape: shape of input image. Includes channels.
    * num_classes: number of object classes, including background.
    """
    
    logging.info("Creating unet graph...")

    image_shape = list(image_shape)
    assert(len(image_shape) == 3)
    image = tf.placeholder(tf.uint8, [None] + image_shape)
    annotation = tf.placeholder(tf.uint8, [None, image_shape[0], image_shape[1],
                                           num_classes])
    training = tf.placeholder(tf.bool)
    
    # The UNet architecture has two stages, up and down. We denote layers in the
    # down-stage with "dn" and those in the up stage with "up," even though the
    # up_conv layers are just performing regular, dimension-preserving
    # convolution. "up_deconv" layers are doing the convolution transpose or
    # "upconv-ing."

    # block level 1
    dn_conv1_1 = UNet.conv(image, filters=64, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_conv1_2 = UNet.conv(dn_conv1_1, filters=64, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_pool1 = UNet.pool(dn_conv1_2)

    # block level 2
    dn_conv2_1 = UNet.conv(dn_pool1, filters=128, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_conv2_2 = UNet.conv(dn_conv2_1, filters=128, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_pool2 = UNet.pool(dn_conv2_2)
    
    # block level 3
    dn_conv3_1 = UNet.conv(dn_pool2, filters=256, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_conv3_2 = UNet.conv(dn_conv3_1, filters=256, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_pool3 = UNet.pool(dn_conv3_2)

    # block level 4
    dn_conv4_1 = UNet.conv(dn_pool3, filters=512, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_conv4_2 = UNet.conv(dn_conv4_1, filters=512, l2_reg_scale=l2_reg_scale,
                           training=training)
    dn_pool4 = UNet.pool(dn_conv4_2)

    # block level 5 (bottom). No max pool; instead deconv and concat.
    dn_conv5_1 = UNet.conv(dn_pool4, filters=1024, l2_reg_scale=l2_reg_scale,
                        training=training)
    dn_conv5_2 = UNet.conv(dn_conv5_1, filters=1024, l2_reg_scale=l2_reg_scale,
                        training=training)
    up_deconv5 = UNet.deconv(dn_conv5_2, filters=512, l2_reg_scale=l2_reg_scale)
    up_concat5 = tf.concat([dn_conv4_2, up_deconv5], axis=3)
    
    # block level 4 (going up)
    up_conv4_1 = UNet.conv(up_concat5, filters=512, l2_reg_scale=l2_reg_scale)
    up_conv4_2 = UNet.conv(up_conv4_1, filters=512, l2_reg_scale=l2_reg_scale)
    up_deconv4 = UNet.deconv(up_conv4_2, filters=256, l2_reg_scale=l2_reg_scale)
    up_concat4 = tf.concat([dn_conv3_2, up_deconv4], axis=3)

    # block level 3
    up_conv3_1 = UNet.conv(up_concat4, filters=256, l2_reg_scale=l2_reg_scale)
    up_conv4_2 = UNet.conv(up_conv3_1, filters=256, l2_reg_scale=l2_reg_scale)
    up_deconv3 = UNet.deconv(up_conv3_2, filters=128, l2_reg_scale=l2_reg_scale)
    up_concat3 = tf.concat([dn_conv2_2, up_deconv3], axis=3)

    up_conv2_1 = UNet.conv(up_concat3, filters=128, l2_reg_scale=l2_reg_scale)
    up_conv2_2 = UNet.conv(up_conv2_1, filters=128, l2_reg_scale=l2_reg_scale)
    up_deconv2 = UNet.deconv(up_conv2_2, filters=64, l2_reg_scale=l2_reg_scale)
    up_concat2 = tf.concat([dn_conv1_2, up_deconv2], axis=3)

    up_conv1_1 = UNet.conv(up_concat2, filters=64, l2_reg_scale=l2_reg_scale)
    up_conv1_2 = UNet.conv(up_conv1_1, filters=64, l2_reg_scale=l2_reg_scale)
    prediction = UNet.conv(up_conv1_2, filters=num_classes,
                           kernel_size=[1, 1], activation=None)

    return image, prediction, annotation, training
    
  @staticmethod
  def conv(inputs, filters=64, kernel_size=[3,3], activation=tf.nn.relu,
           l2_reg_scale=None, training=None):
    """Apply a single convolutional layer with the given activation function applied
    afterword. If l2_reg_scale is not None, specifies the Lambda factor for
    weight normalization in the kernels. If training is not None, indicates that
    batch_normalization should occur, based on whether training is happening.
    """

    if l2_reg_scale is None:
      regularizer = None
    else:
      regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)

    output = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      padding="same",
      activation=activation,
      kernel_regularizer=regularizer)

    if training is not None:
      # normalize the weights in the kernel
      output = tf.layers.batch_normalization(
        inputs=output,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training=training)

    return output
                 
  @staticmethod
  def pool(inputs):
    """Apply 2x2 maxpooling."""
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)

  @staticmethod
  def deconv(inputs, filters, l2_reg_scale=None):
    """Perform "de-convolution" or "up-conv" to the inputs, increasing shape."""
    if l2_reg_scale is None:
      regularizer = None
    else:
      regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale) 

    output = tf.layers.conv2d_transpose(
      inputs=inputs,
      filters=filters,
      strides=[2, 2],
      kernel_size=[2, 2],
      padding='same',
      activation=tf.nn.relu,
      kernel_regularizer=regularizer
    )
    return output
