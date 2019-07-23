"""Implements artifice's detection scheme from end to end.
"""

import os
import json
from time import time
import itertools
import numpy as np
from stringcase import snakecase
import tensorflow as tf
from tensorflow import keras

from artifice.log import logger
from artifice import dat
from artifice import utils
from artifice import img
from artifice import lay

def _update_hist(a, b):
  """Concat the lists in b onto the lists in a.

  If b has elements that a does not, includes them. Behavior is undefined for
  elements that are not lists.

  :param a: 
  :param b: 
  :returns: 
  :rtype:

  """
  c = a.copy()
  for k,v in b.items():
    if type(v) is list and type(c.get(k)) is list:
      c[k] += v
    else:
      c[k] = v
  return c

def _unbatch_outputs(outputs):
  """Essentially transpose the batch dimension to the outer dimension outputs.

  :param outputs: batched outputs of the model, like
  [[pose 0, pose 1, ....], 
   [output_0 0, output_0 1, ...],
   [output_1 0, output_1 1, ...]]
  :returns: result after unbatching, like
  [[pose 0, output_0 0, output_1 0, ...], 
   [pose 1, output_0 1, output_1 1, ...], 
   ...]

  """
  outputs = [list(output) for output in outputs]
  unbatched_outputs = []
  for i in range(len(outputs[0])):
    unbatched_outputs.append([output[i] for output in outputs])
  return unbatched_outputs

def crop(inputs, shape):
  top_crop = int(np.floor(int(inputs.shape[1] - shape[1]) / 2))
  bottom_crop = int(np.ceil(int(inputs.shape[1] - shape[1]) / 2))
  left_crop = int(np.floor(int(inputs.shape[2] - shape[2]) / 2))
  right_crop = int(np.ceil(int(inputs.shape[2] - shape[2]) / 2))
  outputs = keras.layers.Cropping2D(cropping=((top_crop, bottom_crop),
                                              (left_crop, right_crop)),
                                    input_shape=inputs.shape)(inputs)
  return outputs

def conv(inputs, filters, kernel_shape=(3,3), activation='relu',
         padding='valid', norm=True, **kwargs):
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
      filters, kernel_shape, activation=None, padding=padding, use_bias=False,
      kernel_initializer='glorot_normal', **kwargs)(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Activation(activation)(inputs)
  else:
    inputs = keras.layers.Conv2D(
      filters, kernel_shape, activation=activation, padding=padding,
      kernel_initializer='glorot_normal', **kwargs)(inputs)
  return inputs

def conv_transpose(inputs, filters, activation='relu'):
  inputs = keras.layers.Conv2DTranspose(
    filters, (2,2),
    strides=(2,2),
    padding='same',
    activation=activation)(inputs)
  return inputs

class ArtificeModel():
  """A wrapper around keras models.

  If loading an existing model, this class is sufficient, since the save file
  will have the model topology and optimizer. Otherwise, a subclass should
  implement the `forward()` and `compile()` methods, which are called during
  __init__. In this case, super().__init__() should be called last in the
  subclass __init__() method.

  """
  def __init__(self, inputs, model_dir='.', learning_rate=0.1,
               overwrite=False):
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

    if not self.overwrite:
      self.load_weights()

  def __str__(self):
    output = f"{self.name}:\n"
    for layer in self.model.layers:
      output += "layer:{} -> {}:{}\n".format(
        layer.input_shape, layer.output_shape, layer.name)
    return output

  @property
  def layers(self):
    return self.model.layers

  def forward(self, inputs):
    raise NotImplementedError("subclasses should implement")

  def compile(self):
    raise NotImplementedError("subclasses should implement")
  
  @property
  def callbacks(self):
    return [keras.callbacks.ModelCheckpoint(
      self.checkpoint_path, verbose=1, save_weights_only=True)]
  
  def load_weights(self, checkpoint_path=None):
    """Update the model weights from the chekpoint file.

    :param checkpoint_path: checkpoint path to use. If not provided, uses the
    class name to construct a checkpoint path.

    """
    if checkpoint_path is None:
      checkpoint_path = self.checkpoint_path
    if os.path.exists(checkpoint_path):
      self.model.load_weights(checkpoint_path, by_name=True)
      logger.info(f"loaded model weights from {checkpoint_path}")
    else:
      logger.info(f"no checkpoint at {checkpoint_path}")

  def save(self, filename=None, overwrite=True):
    if filename is None:
      filename = self.model_path
    return keras.models.save_model(self.model, filename, overwrite=overwrite,
                                   include_optimizer=False)
  # todo: would like to have this be True, but custom loss function can't be
  # found in keras library. Look into it during training. For now, we're fine
  # with just weights in the checkpoint file.

  def fit(self, art_data, hist=None, cache=False, **kwargs):
    """Thin wrapper around model.fit(). Preferred method is `train()`.

    :param art_data: 
    :param hist: existing hist. If None, starts from scratch. Use train for
    loading from existing hist.
    :param cache: cache the dataset. Should only be used if multiple epochs will
    be run.
    :returns: 
    :rtype: 

    """
    kwargs['callbacks'] = kwargs.get('callbacks', []) + self.callbacks
    new_hist = self.model.fit(art_data.training_input(cache=cache),
                              steps_per_epoch=art_data.steps_per_epoch,
                              **kwargs).history
    new_hist = utils.jsonable(new_hist)
    if hist is not None:
      new_hist = _update_hist(hist, new_hist)
    utils.json_save(self.history_path, hist)
    return hist
  
  def train(self, art_data, initial_epoch=0, epochs=1, seconds=0,
            **kwargs):
    """Fits the model, saving it along the way, and reloads every epoch.

    :param art_data: ArtificeData set
    :param initial_epoch: epoch that training is starting from
    :param epochs: epoch number to stop at. If -1, training continues forever.
    :param seconds: seconds after which to stop reloading every epoch. If -1,
    reload is never stopped. If 0, dataset is loaded only once, at beginning.
    :returns: history dictionary

    """
    if initial_epoch > 0 and os.path.exists(self.history_path) and not self.overwrite:
      hist = utils.json_load(self.history_path)
    else:
      hist = {}
    epoch = initial_epoch
    start_time = time()

    while epoch != epochs and time() - start_time > seconds > 0:
      logger.info("reloading dataset (not cached)...")
      hist = self.fit(art_data, hist=hist, initial_epoch=epoch, epochs=epoch +
                      1, **kwargs)
      epoch += 1

    if epoch != epochs:
      hist = self.fit(art_data, hist=hist, initial_epoch=epoch,
                      epochs=epochs, **kwargs)

    self.save()
    return hist

  def predict(self, art_data):
    """Run prediction, reassembling tiles, with the Artifice data.

    :param art_data: ArtificeData object
    :returns: iterator over predictions

    """
    raise NotImplementedError("subclasses should implement.")

  def predict_visualization(self, art_data):
    """Run prediction, reassembling tiles, with the ArtificeData.

    Intended for visualization. Implementation will depend on the model.

    :param art_data: ArtificeData object
    :returns: iterator over (image, field, prediction)

    """
    raise NotImplementedError()
    
    
  def predict_outputs(self, art_data):
    """Run prediction for single tiles images with the Artifice data.

    Returns the raw outputs, with no prediction. Depends on subclass
    implementation.

    :param art_data: ArtificeData object
    :returns: iterator over (tile, prediction, model_outputs)

    """
    raise NotImplementedError("subclasses should implement")

  
  def evaluate(self, art_data):
    """Run evaluation for object detection with the ArtificeData object.

    Depends on the structure of the model.

    :param art_data: ArtificeData object
    :returns: `errors, total_num_failed` error matrix and number of objects not
    detected
    :rtype: np.ndarray, int

    """
    raise NotImplementedError('subclasses should implmement')
  
  def uncertainty_on_batch(self, images):
    """Estimate the model's uncertainty for each image.

    :param images: a batch of images
    :returns: "uncertainty" for each image. 
    :rtype: 

    """
    raise NotImplementedError("uncertainty estimates not implemented")
    
class ProxyUNet(ArtificeModel):
  def __init__(self, *, base_shape, level_filters, num_channels, pose_dim,
               level_depth=2, dropout=0.5, **kwargs):
    """Create a U-Net model for object detection.

    Regresses a distance proxy at every level for multi-scale tracking. Model
    output consists first of the `pose_dim`-channel pose image, followed by
    multi-scale fields from smallest (lowest on the U) to largest (original
    image dimension).

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

    self.num_levels = len(self.level_filters)
    self.input_tile_shape = self.compute_input_tile_shape()
    self.output_tile_shapes = self.compute_output_tile_shapes()
    super().__init__(keras.layers.Input(self.input_tile_shape +
                                        [self.num_channels]), **kwargs)

  @staticmethod
  def compute_input_tile_shape_(base_shape, num_levels, level_depth):
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
  def compute_input_tile_shape(self):
    return self.compute_input_tile_shape(
      self.base_shape, self.num_levels, self.level_depth) 

  @staticmethod
  def compute_output_tile_shape_(base_shape, num_levels, level_depth):
    tile_shape = np.array(base_shape)
    tile_shape -= 2*level_depth
    for _ in range(num_levels - 1):
      tile_shape *= 2
      tile_shape -= 2*level_depth
    return list(tile_shape)
  def compute_output_tile_shape(self):
    return self.compute_output_tile_shape(
      self.base_shape, self.num_levels, self.level_depth) 

  @staticmethod
  def compute_output_tile_shapes_(base_shape, num_levels, level_depth):
    """Compute the shape of the output tiles at every level, bottom to top."""
    shapes = []
    tile_shape = np.array(base_shape)
    tile_shape -= 2*level_depth
    shapes.append(list(tile_shape))
    for _ in range(num_levels - 1):
      tile_shape *= 2
      tile_shape -= 2*level_depth
      shapes.append(list(tile_shape))
    return shapes
  def compute_output_tile_shapes(self):
    return self.compute_output_tile_shapes(
      self.base_shape, self.num_levels, self.level_depth) 

  def convert_point_between_levels(self, point, level, new_level):
    """Convert len-2 point in the tile-space at `level` to `new_level`.

    Level 0 is the lowest level, by convention. -1 can mean the highest
    (original resolution) level.

    :param point: 
    :param level: level to which the point belongs (last layer in that level).
    :param new_level: level of the space to which the point should be converted.
    :returns: 
    :rtype: 

    """
    while level < new_level:
      point *= 2
      point += self.level_depth
      level += 1
    while level > new_level:
      point /= 2
      point -= self.level_depth
      level -= 1
    return point
  
  def convert_distance_between_levels(self, distance, level, new_level):
    return distance * 2**(new_level - level)

  @staticmethod
  def pose_loss(pose, pred):
    return tf.losses.mean_squared_error(pose[:, :, :, 1:],
                                        pred[:, :, :, 1:],
                                        weights=pose[:, :, :, :1])

  def compile(self):
    if tf.executing_eagerly():
      optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
    else:
      optimizer = keras.optimizers.Adadelta(self.learning_rate)
    self.model.compile(optimizer=optimizer, loss=[self.pose_loss] +
                       ['mse']*self.num_levels, metrics=['mae'])

  def forward(self, inputs):
    level_outputs = []
    outputs = []

    for level, filters in enumerate(self.level_filters):
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters)
      if level < self.num_levels - 1:
        level_outputs.append(inputs)
        inputs = keras.layers.MaxPool2D()(inputs)
      else:
        outputs.append(conv(inputs, 1, kernel_shape=[1,1], activation=None,
                            norm=False, name='output_0'))
        
    level_outputs = reversed(level_outputs)
    for i, filters in enumerate(reversed(self.level_filters[:-1])):
      inputs = conv_transpose(inputs, filters)
      cropped = crop(next(level_outputs), inputs.shape)
      dropped = keras.layers.Dropout(rate=self.dropout)(cropped)
      inputs = keras.layers.Concatenate()([dropped, inputs])
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters)
      outputs.append(conv(inputs, 1, kernel_shape=[1,1], activation=None,
                         norm=False, name=f'output_{i+1}'))

    pose_image = conv(inputs, 1 + self.pose_dim, kernel_shape=(1,1), activation=None,
                      padding='same', norm=False, name='pose')
    return [pose_image] + outputs

  def predict(self, art_data):
    """Run prediction, reassembling tiles, with the Artifice data."""
    if tf.executing_eagerly():
      outputs = []
      for batch in art_data.prediction_input():
        outputs += _unbatch_outputs(self.model.predict_on_batch(batch))
        while len(outputs) >= art_data.num_tiles:
          prediction = art_data.analyze_outputs(outputs)
          del outputs[:art_data.num_tiles]
          yield prediction
    else:
      raise NotImplementedError("patient prediction")

  def evaluate(self, art_data):
    """Runs evaluation for ProxyUNet."""
    raise NotImplementedError("update for outputs style")
    # if tf.executing_eagerly():
    #   proxies = []
    #   tile_labels = []
    #   errors = []
    #   total_num_failed = 0
    #   for i, (batch_tiles,batch_labels) in enumerate(art_data.evaluation_input()):
    #     if i % 10 == 0:
    #       logger.info(f"predicting batch {i} / {art_data.steps_per_epoch}")
    #     tile_labels += list(batch_labels)
    #     proxies += list(self.model.predict_on_batch(batch_tiles))
    #     while len(proxies) >= art_data.num_tiles:
    #       label = tile_labels[0]
    #       proxy = art_data.untile(proxies[:art_data.num_tiles])
    #       error, num_failed = dat.evaluate_proxy(label, proxy)
    #       total_num_failed += num_failed
    #       errors += list(error[error[:, 0] >= 0])
    #       del tile_labels[:art_data.num_tiles]
    #       del proxies[:art_data.num_tiles]
    #     if len(errors) >= art_data.size: # todo: figure out why necessary
    #       break
    #   errors = np.array(errors)
    # else:
    #   raise NotImplementedError
    # return errors, total_num_failed
    
  def predict_visualization(self, art_data):
    """Run prediction, reassembling tiles, with the Artifice data."""
    if tf.executing_eagerly():
      tiles = []
      dist_tiles = []
      outputs = []
      p = art_data.image_padding()
      for batch in art_data.prediction_input():
        tiles += [tile[p[0][0]:, p[1][0]:] for tile in list(batch)]
        outputs += _unbatch_outputs(self.model.predict_on_batch(batch))
        dist_tiles += [output[-1] for output in outputs]
        while len(outputs) >= art_data.num_tiles:
          image = art_data.untile(tiles[:art_data.num_tiles])
          dist_image = art_data.untile(dist_tiles[:art_data.num_tiles])
          prediction = art_data.analyze_outputs(outputs)
          del outputs[:art_data.num_tiles]
          del tiles[:art_data.num_tiles]
          del dist_tiles[:art_data.num_tiles]
          yield (image, dist_image, prediction)
    else:
      raise NotImplementedError("patient prediction")

  def predict_outputs(self, art_data):
    """Run prediction for single tiles images with the Artifice data."""
    num_tiles = art_data.num_tiles
    art_data.num_tiles = 1
    if tf.executing_eagerly():
      tiles = []
      outputs = []
      p = art_data.image_padding()
      for batch in art_data.prediction_input():
        tiles += [tile[p[0][0]:, p[1][0]:] for tile in list(batch)]
        outputs += _unbatch_outputs(self.model.predict_on_batch(batch))
        while outputs:
          tile = art_data.untile(tiles[:1])
          yield (tile, outputs[0]) # todo: del line
          del outputs[0]
          del tiles[0]
    else:
      raise NotImplementedError("patient prediction")
    art_data.num_tiles = num_tiles
    
  def uncertainty_on_batch(self, images):
    """Estimate the model's uncertainty for each image."""
    raise NotImplementedError("update for outputs")
    predictions = self.model.predict_on_batch(images)
    proxies = predictions[:,:,:,0]
    confidences = np.empty(proxies.shape[0], np.float32)
    for i, proxy in enumerate(proxies):
      detections = dat.detect_peaks(proxy)
      confidences[i] = np.mean([proxy[x,y] for x,y in detections])
    return 1 - confidences
    
class SparseUNet(ProxyUNet):
  def __init__(self, *, block_size=[8,8], threshold=0.1, **kwargs):
    """Create a UNet-like architecture using multi-scale tracking.

    :param block_size: width/height of the blocks used for sparsity, at the
    scale of the original resolution (resized at each level. These are rescaled
    at each level.

    :param threshold: absolute threshold value for sbnet attention.
    :returns:
    :rtype:

    """
    super().__init__(**kwargs)
    self.threshold = threshold
    self.block_size = block_size

  # todo: write this forward function. It's gonna be a mess.
  def forward(self, inputs):
    level_outputs = []
    outputs = []

    for level, filters in enumerate(self.level_filters):
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters)
      if level < self.num_levels - 1:
        level_outputs_down.append(inputs)
        inputs = keras.layers.MaxPool2D()(inputs)
      else:
        outputs.append(conv(inputs, 1, kernel_shape=[1,1], activation=None,
                            norm=False, name='output_0'))

    level_outputs = reversed(level_outputs)
    for i, filters in enumerate(reversed(self.level_filters[:-1])):
      inputs = conv_transpose(inputs, filters)
      cropped = crop(next(level_outputs), inputs.shape)
      dropped = keras.layers.Dropout(rate=self.dropout)(cropped)
      inputs = keras.layers.Concatenate()([dropped, inputs])
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters)
      outputs.append(conv(inputs, 1, kernel_shape=[1,1], activation=None,
                         norm=False, name=f'output_{i+1}'))

        
    peaks = lay.PeakDetection()(outputs[0]) # [num_objects, 4]
    raise NotImplementedError()
    # goal: for each of the peaks, need to get out a slice from the actual
    # feature image. So the shape of the inputs to the next convolutional layer
    # will have shape [num_objects, batch_size, sx, sy, filters]. Can then use
    # tf.

    # better idea: from this point on, construct sparse tensors with the desired
    # regions and perform convolution on those. 
    

    
    
