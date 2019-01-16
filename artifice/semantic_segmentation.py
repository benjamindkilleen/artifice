"""Provides semantic segmentation capability for artifice. The specific
algorithm isn't important, as long as it returns an output in familiar form.

We implement the U-Net segmentation architecture
(http://arxiv.org/abs/1505.04597), loosely inspired by implementation at:
https://github.com/tks10/segmentation_unet/

Notes:
We consider two types of augmentations: standard and
boundary-aware. Boundary-aware augmentations have to be done in between training
runs, so a Trainer object could use multiple runs through this, and apply
boundary-aware augmentations in between runs, saving them in a separate
tfrecord, but standard augmentations can be done on-the-fly without being
stored, so they should be done here.

"""

import os
from shutil import rmtree
import numpy as np
import logging
import tensorflow as tf
from artifice.utils import dataset

tf.logging.set_verbosity(tf.logging.INFO)

logger = logging.getLogger('artifice')

# TODO: allow for customizing these values
batch_size = 1
prefetch_buffer_size = 1  

def compute_balanced_weights(annotations, num_classes):
  """Calculate the weight tensor given annotations, like sklearn's
  compute_class_weight().

  """

  counts = tf.bincount(tf.cast(annotations, tf.int32), 
                       minlength=num_classes, maxlength=num_classes,
                       dtype=tf.float64)
  num_samples = tf.cast(
    tf.constant(batch_size) * tf.reduce_prod(annotations.shape[1:]), 
    tf.float64)
  class_weights = num_samples / (num_classes * counts)
  class_weights = class_weights / tf.norm(class_weights, ord=1)
  class_weights = tf.Print(class_weights, [class_weights],
                           message='class weights:', first_n=1)

  weights = tf.gather(class_weights, annotations)[:,:,:,0]
  return weights


def crop(image, shape, batched = True):
    """Crop IMAGE (possibly batched) to SHAPE, centering as much as possible
    (tending toward the upper left when rounding.
    """

    logger.debug(f"cropping {image.shape} to {shape}")

    if batched:
      height = image.shape[1]
      width = image.shape[2]
    else:
      height = image.shape[0]
      width = image.shape[1]

    offset_height = (height - shape[0]) // 2
    offset_width  = (width - shape[1]) // 2

    return tf.image.crop_to_bounding_box(image,
                                         offset_height, offset_width,
                                         shape[0], shape[1])


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
  num_shuffle = 10000
  learning_rate = 0.01
  momentum = 0.99
  def __init__(self, image_shape, num_classes, model_dir=None, l2_reg_scale=None):
    self.image_shape = list(image_shape)
    self.annotation_shape = self.image_shape[:2] + [1]
    assert(len(self.image_shape) == 3)
    self.num_classes = num_classes
    self.l2_reg_scale = l2_reg_scale

    feature_columns = [tf.feature_column.numeric_column(
      'image', shape=self.image_shape, dtype=tf.uint8)]
    
    self.params = {'feature_columns' : feature_columns}

    self.model_dir = model_dir

  def train(self, train_data_input, 
            eval_data_input=None, 
            overwrite=False,
            num_epochs=1,
            eval_secs=600,
            save_steps=100,
            log_steps=5,
            cores=1):
    """Train the model with tf Dataset object train_data. If test_data is not None,
    evaluate the model with it, and log the results (at INFO level).

    :train_data_input: DataInput subclass to use for training.
    :eval_data_input: DataInput subclass to use for evaluation. If None, does no
      evaluation. 
    :overwrite: overwrite the existing model
    :eval_secs: do evaluation every EVAL_SECS. Default = 600. Set to 0 for no
      evaluation.
    :cores: number of cores to parallelize over for data processing/augmentation.

    """
    assert eval_secs >= 0

    if overwrite and self.model_dir is None:
      logger.warning("FAIL to overwrite; model_dir is None")

    if (overwrite and self.model_dir is not None
        and os.path.exists(self.model_dir)):
      rmtree(self.model_dir)
      
    if (overwrite and self.model_dir is not None
        and not os.path.exists(self.model_dir)):
      os.mkdir(self.model_dir)

    # Configure session
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    run_config = tf.estimator.RunConfig(model_dir=self.model_dir,
                                        save_checkpoints_steps=save_steps,
                                        log_step_count_steps=log_steps)
    

    # Train the model. (Might take a while.)
    model = tf.estimator.Estimator(model_fn=self.create(training=True),
                                   model_dir=self.model_dir,
                                   params=self.params,
                                   config=run_config)

    if eval_data_input is None:
      logger.info("train...")
      model.train(input_fn=train_data_input(num_epochs=num_epochs))
    else:
      logger.info("train and evaluate...")
      train_spec = tf.estimator.TrainSpec(
        input_fn=train_data_input(num_epochs=num_epochs))
      eval_spec = tf.estimator.EvalSpec(input_fn=eval_data_input(),
                                        throttle_secs=eval_secs)
      tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


  def predict(self, data_input):
    """Return the estimator's predictions on data_input.

    """
    if self.model_dir is None:
      logger.warning("prediction FAILED (no model_dir)")
      return None
  
    model = tf.estimator.Estimator(model_fn=self.create(training=False),
                                   model_dir=self.model_dir,
                                   params=self.params)

    predictions = model.predict(input_fn=data_input())
    return predictions


  def create(self, training=True):
    """Create the model function for a custom estimator."""
    
    def model_function(features, labels, mode, params):
      # TODO: images -> image (plural -> singular)
      image = tf.reshape(features, [-1] + self.image_shape)
      prediction_logits = self.infer(image, training=training)
      prediction = tf.reshape(tf.argmax(prediction_logits, axis=3),
                               [-1] + self.annotation_shape)
      prediction_one_hot = tf.one_hot(tf.reshape(
        predictions, [-1] + self.annotation_shape[:2]), self.num_classes)

      # In PREDICT mode, return the output asap.
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
          mode=mode, predictions={'image' : image,
                                  'logits' : prediction_logits,
                                  'annotation' : prediction})

      # Get "ground truth" for other modes.
      annotation, label = labels
      annotation = tf.cast(tf.reshape(annotation[:,:,:,0], [-1] +
                                      self.annotation_shape), tf.int64)
      annotation_one_hot = tf.one_hot(annotation[:,:,:,0], self.num_classes)

      logger.debug("annotation_one_hot: {}".format(annotation_one_hot.shape))
      logger.debug("prediction_logits: {}".format(prediction_logits.shape))

      # weight by class frequency
      weights = compute_balanced_weights(annotation, self.num_classes)
      logger.debug("weights: {}".format(weights.shape))

      # TODO: give boundaries greater weight (boundaries currently not encoded,
      # could also predict a second annotation level encoding position,
      # somehow.)

      # Calculate loss:
      cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels=annotation_one_hot,
        logits=prediction_logits,
        weights=weights)

      # Return an optimizer, if mode is TRAIN
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        # TODO: allow choice between these
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss=cross_entropy,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=cross_entropy, 
                                          train_op=train_op)
    
      assert mode == tf.estimator.ModeKeys.EVAL
      accuracy = tf.metrics.accuracy(labels=annotation,
                                     predictions=prediction)
      
      # TODO: somehow, we're getting very high (99.5%) accuracy for objId 1 (the
      # balls), but the prediction images are messed up somehow. Figure out why.
      # TODO: above still happening
      eval_metrics = {'accuracy' : accuracy}
      for objId in range(self.num_classes):
        weights = tf.gather(annotation_one_hot, objId, axis=3)
        weights = tf.Print(weights, [weights], message=f'obj_{objId} weights:',
                           first_n=1)
        indices = tf.where(tf.equal(tf.constant(objId, dtype=tf.int64), annotation))
        obj_annotations = tf.gather_nd(annotation, indices)
        obj_prediction = tf.gather_nd(prediction, indices)
        eval_metrics[f'class_{objId}_accuracy'] = tf.metrics.accuracy(
          labels=obj_annotation,
          predictions=obj_prediction)
    
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=cross_entropy,
                                        eval_metric_ops=eval_metrics)

    return model_function

  def infer(self, images, training=True):
    raise NotImplementedError("SemanticModel subclass should implement inference().")



"""Implementation of UNet."""
class UNet(SemanticModel):
  padding = 96 # net removes 92 pixels on each side? (only if input image has
  # shape such that pooling layers evenly divide it) 

  # TODO: adjust test data such that all these divisions happen cleanly, or
  # allow for tiling.

  def infer(self, images, training=True):  
    """The UNet architecture has two stages, up and down. We denote layers in the
    down-stage with "dn" and those in the up stage with "up," even though the
    up_conv layers are just performing regular, dimension-preserving
    convolution. "up_deconv" layers are doing the convolution transpose or
    "upconv-ing."

    """
    
    # TODO: reformat test data to be of a size that can be correctly padded, or
    # implement tiling. (Going to have to do tiling eventually, but would rather
    # stick to 512x512 inputs size, or larger? Make experiment test data
    # accordingly.

    # TODO: tile inputs

    logger.debug(f"images: {images.shape}")

    paddings = tf.constant([[0,0],
                            [self.padding, self.padding],
                            [self.padding, self.padding],
                            [0,0]])
    padded = tf.pad(images, paddings, mode='REFLECT')

    logger.debug(f"padded: {padded.shape}")

    # block level 1
    dn_conv1_1 = self.conv(padded, filters=64, training=training)
    dn_conv1_2 = self.conv(dn_conv1_1, filters=64, training=training)
    dn_pool1 = self.pool(dn_conv1_2)
    logger.debug(f"dn_pool1: {dn_pool1.shape}")    
    
    # block level 2
    dn_conv2_1 = self.conv(dn_pool1, filters=128, training=training)
    dn_conv2_2 = self.conv(dn_conv2_1, filters=128, training=training)
    dn_pool2 = self.pool(dn_conv2_2)
    logger.debug(f"dn_pool1: {dn_pool2.shape}")
    
    # block level 3
    dn_conv3_1 = self.conv(dn_pool2, filters=256, training=training)
    dn_conv3_2 = self.conv(dn_conv3_1, filters=256, training=training)
    dn_pool3 = self.pool(dn_conv3_2)
    logger.debug(f"dn_pool3: {dn_pool3.shape}")
    
    # block level 4
    dn_conv4_1 = self.conv(dn_pool3, filters=512, training=training)
    dn_conv4_2 = self.conv(dn_conv4_1, filters=512, training=training)
    dn_pool4 = self.pool(dn_conv4_2)
    logger.debug(f"dn_pool4: {dn_pool4.shape}")
    
    # block level 5 (bottom). No max pool; instead deconv and concat.
    dn_conv5_1 = self.conv(dn_pool4, filters=1024, training=training)
    dn_conv5_2 = self.conv(dn_conv5_1, filters=1024, training=training)
    up_deconv5 = self.deconv(dn_conv5_2, filters=512)
    up_concat5 = tf.concat([crop(dn_conv4_2, up_deconv5.shape[1:3]),
                            up_deconv5], axis=3)
    logger.debug(f"up_concat5: {up_concat5.shape}")
    
    # block level 4 (going up)
    up_conv4_1 = self.conv(up_concat5, filters=512)
    up_conv4_2 = self.conv(up_conv4_1, filters=512)
    up_deconv4 = self.deconv(up_conv4_2, filters=256)
    up_concat4 = tf.concat([crop(dn_conv3_2, up_deconv4.shape[1:3]),
                            up_deconv4], axis=3)
    logger.debug(f"up_concat4: {up_concat4.shape}")
    
    # block level 3
    up_conv3_1 = self.conv(up_concat4, filters=256)
    up_conv3_2 = self.conv(up_conv3_1, filters=256)
    up_deconv3 = self.deconv(up_conv3_2, filters=128)
    up_concat3 = tf.concat([crop(dn_conv2_2, up_deconv3.shape[1:3]),
                            up_deconv3], axis=3)
    logger.debug(f"up_concat3: {up_concat3.shape}")
    
    # block level 2
    up_conv2_1 = self.conv(up_concat3, filters=128)
    up_conv2_2 = self.conv(up_conv2_1, filters=128)
    up_deconv2 = self.deconv(up_conv2_2, filters=64)
    up_concat2 = tf.concat([crop(dn_conv1_2, up_deconv2.shape[1:3]),
                            up_deconv2], axis=3)
    logger.debug(f"up_concat2: {up_concat2.shape}")


    # block level 1
    up_conv1_1 = self.conv(up_concat2, filters=64)
    up_conv1_2 = self.conv(up_conv1_1, filters=64)
    logits = self.conv(up_conv1_2, filters=self.num_classes,
                       kernel_size=[1, 1], activation=None)
    logger.debug(f"logits: {logits.shape}")
    cropped_logits = crop(logits, images.shape[1:3])
    logger.debug(f"cropped_logits: {cropped_logits.shape}")
    return cropped_logits
    
  def conv(self, inputs, filters=64, kernel_size=[3,3], activation=tf.nn.relu,
           training=True, padding='valid'):
    """Apply a single convolutional layer with the given activation function applied
    afterword. If l2_reg_scale is not None, specifies the Lambda factor for
    weight normalization in the kernels. If training is not None, indicates that
    batch_normalization should occur, based on whether training is happening.
    """

    if self.l2_reg_scale is None:
      regularizer = None
    else:
      regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

    stddev = np.sqrt(2 / (np.prod(kernel_size) * filters))
    initializer = tf.initializers.random_normal(stddev=stddev)

    output = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      padding=padding,
      activation=activation,
      kernel_initializer=initializer,
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
                 
  def pool(self, inputs):
    """Apply 2x2 maxpooling."""
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)

  def deconv(self, inputs, filters, padding='valid'):
    """Perform "de-convolution" or "up-conv" to the inputs, increasing shape."""
    if self.l2_reg_scale is None:
      regularizer = None
    else:
      regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale) 

    stddev = np.sqrt(2 / (2*2*filters))
    initializer = tf.initializers.random_normal(stddev=stddev)
    
    output = tf.layers.conv2d_transpose(
      inputs=inputs,
      filters=filters,
      strides=[2, 2],
      kernel_size=[2, 2],
      padding=padding,
      activation=tf.nn.relu,
      kernel_initializer=initializer,
      kernel_regularizer=regularizer)

    return output
