"""Provides semantic segmentation capability for artifice. The specific
algorithm isn't important, as long as it returns an output in familiar form.

We implement the U-Net segmentation architecture
(http://arxiv.org/abs/1505.04597), loosely inspired by implementation at:
https://github.com/tks10/segmentation_unet/

"""

import os
from shutil import rmtree
import numpy as np
import logging

import tensorflow as tf
from artifice.utils import dataset

logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s:%(asctime)s:%(message)s')


"""A model implementing semantic segmentation.
args:
* channels: number of channels in the image (grayscale by default)
* num_classes: number of classes of objects to be detected (including background)

In this context, `image` is the input to the model, and `annotation`, is the
SEMANTIC annotation (groung truth) of `image`, a [image_shape[0],
image_shape[1], num_classes] shape array which one-hot encoedes each pixel's
class.
"""

class SemanticModel:
  num_shuffle = 1000
  learning_rate = 0.001
  def __init__(self, image_shape, num_classes, model_dir=None):
    self.image_shape = list(image_shape)
    self.annotation_shape = self.image_shape[:2] + [1]
    assert(len(self.image_shape) == 3)
    self.num_classes = num_classes

    feature_columns = [tf.feature_column.numeric_column(
      'image', shape=self.image_shape, dtype=tf.uint8)]
    
    self.params = {'feature_columns' : feature_columns}

    self.model_dir = model_dir

  @staticmethod
  def create(training=True, l2_reg_scale=None):
    raise NotImplementedError("SemanticModel subclass should implement create().")

  def train(self, train_data, batch_size=4, test_data=None, overwrite=False,
            num_epochs=1):
    """Train the model with tf Dataset object train_data. If test_data is not None,
    evaluate the model with it, and log the results (at INFO level).

    """
    if overwrite and self.model_dir is None:
      logging.warning("FAIL to overwrite; model_dir is None")

    if (overwrite and self.model_dir is not None
        and os.path.exists(self.model_dir)):
      rmtree(self.model_dir)
      
    if (overwrite and self.model_dir is not None
        and not os.path.exists(self.model_dir)):
      os.mkdir(self.model_dir)

    # Configure session
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    run_config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                        save_checkpoints_steps=50,
                                        log_step_count_steps=5)

    input_train = lambda : (
      train_data.shuffle(self.num_shuffle)
      .batch(batch_size)
      .repeat(num_epochs)
      .make_one_shot_iterator()
      .get_next())

    model = tf.estimator.Estimator(model_fn=self.create(training=True),
                                   model_dir=self.model_dir,
                                   params=self.params,
                                   config=run_config)

    model.train(input_fn=input_train)
    
    if test_data is not None:
      input_test = lambda : (
        test_data.batch(batch_size)
        .make_one_shot_iterator()
        .get_next())
      eval_result = model.evaluate(input_fn=input_test)
      logging.info(eval_result)

  def predict(self, test_data):
    """Return the estimator's predictions on test_data.

    """
    if self.model_dir is None:
      logging.warning("prediction FAILED (no model_dir)")
      return None

    input_pred = lambda : (
      test_data.batch(batch_size)
      .make_one_shot_iterator()
      .get_next())
  
    model = tf.estimator.Estimator(model_fn=self.create(training=False),
                                   model_dir=self.model_dir,
                                   params=self.params)

    predictions = model.predict(input_fn=input_pred)
    return predictions


"""Implementation of UNet."""
class UNet(SemanticModel):
  def create(self, training=True, l2_reg_scale=None):
    """Create the unet model function for a custom estimator.

    """
    
    logging.info("Creating unet graph...")

    def model_function(features, labels, mode, params):
      images = tf.reshape(features, [-1] + self.image_shape,
                          name='reshape_images')
      annotations = tf.reshape(labels, [-1] + self.annotation_shape, 
                               name='reshape_annotations')
      annotations_3D = tf.reshape(
        labels,
        [-1, self.annotation_shape[0], self.annotation_shape[1]], 
        name='reshape_annotations_3D')
      annotations_one_hot = tf.one_hot(annotations_3D, self.num_classes)

      # The UNet architecture has two stages, up and down. We denote layers in the
      # down-stage with "dn" and those in the up stage with "up," even though the
      # up_conv layers are just performing regular, dimension-preserving
      # convolution. "up_deconv" layers are doing the convolution transpose or
      # "upconv-ing."

      # block level 1
      dn_conv1_1 = UNet.conv(images, filters=64, l2_reg_scale=l2_reg_scale,
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
      up_conv3_2 = UNet.conv(up_conv3_1, filters=256, l2_reg_scale=l2_reg_scale)
      up_deconv3 = UNet.deconv(up_conv3_2, filters=128, l2_reg_scale=l2_reg_scale)
      up_concat3 = tf.concat([dn_conv2_2, up_deconv3], axis=3)

      up_conv2_1 = UNet.conv(up_concat3, filters=128, l2_reg_scale=l2_reg_scale)
      up_conv2_2 = UNet.conv(up_conv2_1, filters=128, l2_reg_scale=l2_reg_scale)
      up_deconv2 = UNet.deconv(up_conv2_2, filters=64, l2_reg_scale=l2_reg_scale)
      up_concat2 = tf.concat([dn_conv1_2, up_deconv2], axis=3)

      up_conv1_1 = UNet.conv(up_concat2, filters=64, l2_reg_scale=l2_reg_scale)
      up_conv1_2 = UNet.conv(up_conv1_1, filters=64, l2_reg_scale=l2_reg_scale)
      predicted_logits = UNet.conv(up_conv1_2, filters=self.num_classes,
                                    kernel_size=[1, 1], activation=None)

      predictions = tf.argmax(predicted_logits, 3)

      # In PREDICT mode, return the output asap.
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode, predictions={'annotation' : predictions})

      # Calculate loss:
      cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels=annotations_one_hot, logits=predicted_logits)

      # Return an optimizer, if mode is TRAIN
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss=cross_entropy,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=cross_entropy, 
                                          train_op=train_op)
    
      assert mode == tf.estimator.ModeKeys.EVAL
      accuracy = tf.metrics.accuracy(labels=annotations,
                                     predictions=predictions)
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=cross_entropy)


    return model_function
    
  @staticmethod
  def conv(inputs, filters=64, kernel_size=[3,3], activation=tf.nn.relu,
           l2_reg_scale=None, training=True):
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
