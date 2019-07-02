"""Implements artifice's detection scheme from end to end.
"""

import os
import logging
import json
import time
import numpy as np
from stringcase import snakecase
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from artifice import lay, dat, utils, vis, img

logger = logging.getLogger('artifice')

def crop(inputs, shape):
  top_crop = int(np.floor(int(inputs.shape[1] - shape[1]) / 2))
  bottom_crop = int(np.ceil(int(inputs.shape[1] - shape[1]) / 2))
  left_crop = int(np.floor(int(inputs.shape[2] - shape[2]) / 2))
  right_crop = int(np.ceil(int(inputs.shape[2] - shape[2]) / 2))
  outputs = keras.layers.Cropping2D(cropping=((top_crop, bottom_crop),
                                              (left_crop, right_crop)),
                                    input_shape=inputs.shape)(inputs)
  return outputs

def conv(inputs, filters, kernel_shape=(3,3),
         activation='relu', padding='valid', norm=True):
  """Perform 3x3 convolution on the layer.

  :param inputs: input tensor
  :param filters: number of filters or kernels
  :param activation: keras activation to use. Default is 'relu'
  :param padding: 'valid' or 'same'
  :param norm: whether or not to perform batch normalization on the output
  :returns:
  :rtype:

  """
  if norm:
    inputs = keras.layers.Conv2D(
      filters, kernel_shape,
      activation=None,
      padding=padding,
      use_bias=False,
      kernel_initializer='glorot_normal')(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Activation(activation)(inputs)
  else:
    inputs = keras.layers.Conv2D(
      filters, kernel_shape,
      activation=activation,
      padding=padding,
      kernel_initializer='glorot_normal')(inputs)
  return inputs

def conv_transpose(inputs, filters, activation='relu'):
  inputs = keras.layers.Conv2DTranspose(
    filters, (2,2),
    strides=(2,2),
    padding='same',
    activation=activation)(inputs)
  return inputs

class Model():
  """A wrapper around keras models.

  If loading an existing model, this class is sufficient, since the save file
  will have the model topology and optimizer. Otherwise, a subclass should
  implement the `forward()` and `compile()` methods, which are called during
  __init__. In this case, super().__init__() should be called last in the
  subclass __init__() method.

  """
  def __init__(self, inputs, model_dir='.', learning_rate=0.1,
               overwrite=False, expect_checkpoint=False):
    """Describe a model using keras' functional API.

    Compiles model here, so all other instantiation should be finished.

    :param inputs: tensor or list of tensors to input into the model (such as
    layers.Input)
    :param model_dir: directory to save the model. Default is cwd.
    :param learning_rate:
    :param overwrite: prefer to create a new model rather than load an existing
    one in `model_dir`. Note that if a subclass uses overwrite=False, then the
    loaded architecture may differ from the stated architecture in the subclass,
    although the structure of the saved model names should prevent this.

    """
    self.overwrite = overwrite
    self.model_dir = model_dir
    self.learning_rate = learning_rate
    self.name = snakecase(type(self).__name__).lower()
    self.model_path = os.path.join(self.model_dir, f"{self.name}.hdf5")
    self.checkpoint_path = os.path.join(self.model_dir, f"{self.name}_ckpt.hdf5")
    self.history_path = os.path.join(self.model_dir, f"{self.name}_history.json")

    outputs = self.forward(inputs)
    self.model = keras.Model(inputs, outputs)
    self.compile()

    if os.path.exists(self.checkpoint_path) and not self.overwrite:
      logger.info(f"loading_weights from {self.checkpoint_path}")
      self.model.load_weights(self.checkpoint_path)
    elif expect_checkpoint:
      logger.warning(f"no checkpoing found at {self.checkpoint_path}")

  def __str__(self):
    output = f"{self.name}:\n"
    for layer in layers:
      output += "layer:{} -> {}:{}\n".format(
        layer.input_shape, layer.output_shape, layer.name)
    return output

  def forward(self, inputs):
    raise NotImplementedError("subclasses should implement")

  def compile(self):
    raise NotImplementedError("subclasses should implement")

  @property
  def callbacks(self):
    return [keras.callbacks.ModelCheckpoint(
      self.checkpoint_path, verbose=1, save_weights_only=False)]

  def save(self, filename=None, overwrite=True):
    if filename is None:
      filename = self.model_path
    return keras.models.save_model(self.model, filename, overwrite=overwrite,
                                   include_optimizer=False)
  # todo: would like to have this be True, but custom loss function can't be
  # found in keras library. Look into it during training. For now, we're fine
  # with just weights in the checkpoint file.

  def train(self, art_data, **kwargs):
    """Fits the model, saving it along the way and saving the training history

    :param art_set: ArtificeData 
    :returns: history dictionary
    :rtype: 

    """
    kwargs['callbacks'] = kwargs.get('callbacks', []) + self.callbacks
    hist = self.model.fit(art_data.training_input,
                          steps_per_epoch=art_data.steps_per_epoch, **kwargs).history
    with open(self.history_path, 'w') as f:
      f.write(json.dumps(utils.jsonable(hist))) # todo: add to existing history
    self.save()
    return hist
  
  def evaluate(self, art_data):
    """Run evaluation, reassembling tiles, with the ArtificeData object.

    Returns an iterator over the predictions. This is necessary for very large
    test sets, which we will use.

    :param art_data: ArtificeData object
    :returns: `(num_examples, num_objects, 1 + pose_dim)` predictions
    :rtype: iterator over numpy array

    """
    if tf.executing_eagerly():
      tiles = []
      proxies = []
      tile_labels = []
      errors = []
      total_num_failed = 0
      for batch_tiles, batch_labels in art_data.evaluation_input:
        tiles += list(batch_tiles)
        tile_labels += list(batch_labels)
        proxies += list(self.model.predict_on_batch(batch_tiles))
        if len(proxies) >= art_data.num_tiles:
          label = tile_labels[0]
          proxy = art_data.untile(proxies[:art_data.num_tiles])
          image = art_data.untile(tiles[:art_data.num_tiles])
          error, num_failed = dat.evaluate_proxy(label, proxy)
          total_num_failed += num_failed
          logger.debug(f"error: {error}")
          errors.append(error)
          del tiles[:art_data.num_tiles]
          del tile_labels[:art_data.num_tiles]
          del proxies[:art_data.num_tiles]
      avg_error = np.array(errors).mean(axis=[0,1])
    else:
      raise NotImplementedError

    return avg_error, total_num_failed
    
class ProxyUNet(Model):
  def __init__(self, *, base_shape, level_filters, num_channels, pose_dim,
               level_depth=2, dropout=0.5, **kwargs):
    """Create an hourglass-shaped model for object detection.

    :param base_shape: the height/width of the output of the first layer in the lower
    level. This determines input and output tile shapes. Can be a tuple,
    specifying different height/width, or a single integer.
    :param level_filters: number of filters at each level (top to bottom).
    :param level_depth: number of layers per level
    :param dropout: dropout to use for concatenations
    :param num_channels: number of channels in the input
    :param pose_dim:

    """
    self.base_shape = utils.listify(base_shape, 2)
    self.level_filters = level_filters
    self.num_channels = num_channels
    self.pose_dim = pose_dim
    self.level_depth = level_depth
    self.dropout = dropout
    self.input_tile_shape = self.compute_input_tile_shape(
      base_shape, len(self.level_filters), self.level_depth)
    self.output_tile_shape = self.compute_input_tile_shape(
      base_shape, len(self.level_filters), self.level_depth)
    super().__init__(keras.layers.Input(self.input_tile_shape + [self.num_channels]), **kwargs)

  @staticmethod
  def compute_input_tile_shape(base_shape, num_levels, level_depth):
    """Compute the shape of the input tiles.

    :param base_shape: shape of the output of the first layer in the
    lower level.
    :param num_levels: number of levels
    :param level_depth: layers per level (per side)
    :returns: shape of the input tiles

    """
    tile_shape = np.array(base_shape)
    for _ in range(num_levels - 1):
      tile_shape *= 2
      tile_shape += 2*level_depth
    return list(tile_shape)

  @staticmethod
  def compute_output_tile_shape(base_shape, num_levels, level_depth):
    tile_shape = np.array(base_shape)
    tile_shape -= 2*level_depth
    for _ in range(num_levels - 1):
      tile_shape *= 2
      tile_shape -= 2*level_depth
    return list(tile_shape)

  @staticmethod
  def loss(proxy, prediction):
    distance_term = tf.losses.mean_squared_error(proxy[:, :, :, 0], prediction[:, :, :, 0])
    pose_term = tf.losses.mean_squared_error(proxy[:, :, :, 1:], prediction[:, :, :, 1:],
                                             weights=proxy[:, :, :, :1])
    return distance_term + pose_term

  def compile(self):
    if tf.executing_eagerly():
      optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
    else:
      optimizer = keras.optimizers.Adadelta(self.learning_rate)
    self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['mae'])

  def forward(self, inputs):
    level_outputs = []

    for i, filters in enumerate(self.level_filters):
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters)
      if i < len(self.level_filters) - 1:
        level_outputs.append(inputs)
        inputs = keras.layers.MaxPool2D()(inputs)

    level_outputs = reversed(level_outputs)
    for i, filters in enumerate(reversed(self.level_filters[:-1])):
      inputs = conv_transpose(inputs, filters)
      cropped = crop(next(level_outputs), inputs.shape)
      dropped = keras.layers.Dropout(rate=self.dropout)(cropped)
      inputs = keras.layers.Concatenate()([dropped, inputs])
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters)

    inputs = conv(inputs, 1 + self.pose_dim, kernel_shape=(1,1), activation=None,
                  padding='same', norm=False)
    return inputs

  # todo: rewrite full_predict and detect from git.
