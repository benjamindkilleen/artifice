"""Implements artifice's detection scheme from end to end.
"""


import tensorflow as tf
from tensorflow import keras
from os.path import join
from artifice import lay
import numpy as np
import logging

logger = logging.getLogger('artifice')


def log_model(model):
  logger.info(f'model: {model.name}')
  log_layers(model.layers)

def log_layers(layers):
  for layer in layers:
    logger.info(
      f"layer:{layer.input_shape} -> {layer.output_shape}:{layer.name}")

def crop(inputs, shape):
  return lay.Crop(shape)(inputs)
    
def maxpool(inputs, dropout=None):
  inputs = keras.layers.MaxPool2D()(inputs)
  if dropout is not None:
    inputs = keras.layers.Dropout(dropout)(inputs)
  return inputs

def conv(inputs, filters, activation='relu', padding='valid'):
  inputs = keras.layers.Conv2D(
    filters, (3,3),
    activation=activation,
    padding=padding,
    kernel_initializer='glorot_normal')(inputs)
  return inputs

def dense(inputs, nodes, activation='relu'):
  inputs = keras.layers.Dense(nodes, activation=activation)(inputs)
  return inputs

def conv_transpose(inputs, filters, activation='relu', dropout=None):
  inputs = keras.layers.Conv2DTranspose(
    filters, (2,2),
    strides=(2,2),
    padding='same',
    activation=activation)(inputs)
  if dropout is not None:
    inputs = keras.layers.Dropout(dropout)(inputs)
  return inputs

def concat(inputs, *other_inputs, axis=-1, dropout=None):
  """Apply dropout to inputs and concat with other_inputs."""
  if dropout is not None:
    inputs = keras.layers.Dropout(dropout)(inputs)
  inputs = keras.layers.concatenate([inputs] + list(other_inputs))
  return inputs

    
class Model():
  def __init__(self, model, model_dir=None, tensorboard=None):
    """Wrapper around a keras.Model for saving, loading, and running.

    Subclasses can overwrite the loss() method to create their own loss.

    In general, these layers should include an input layer. Furthermore,
    super().__init__() should usually be called at the end of the subclass's
    __init__, after it has initialized variables necessary for
    self.create_layers().

    :param input_shape: 
    :returns: 
    :rtype: 

    """
    self.model_dir = model_dir
    self.checkpoint_path = (None if model_dir is None else
                            join(model_dir, "cp-{epoch:04d}.hdf5"))
    self.tensorboard = tensorboard
    
    self.model = model
    log_model(self.model)


  def compile(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def callbacks(self):
    callbacks = []
    if self.checkpoint_path is not None:
      callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        self.checkpoint_path, verbose=1, save_weights_only=True,
        period=1))
    if self.tensorboard:
      # Need to have an actual director in which to store the logs.
      raise NotImplementedError
    return callbacks
    
  def fit(self, *args, **kwargs):
    kwargs['callbacks'] = self.callbacks
    return self.model.fit(*args, **kwargs)

  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)

  def evaluate(self, *args, **kwargs):
    return self.model.evaluate(*args, **kwargs)

  def save(self, *args, **kwargs):
    return self.model.save(*args, **kwargs)
    
  # load the model from a most recent checkpoint
  def load(self):
    if self.model_dir is None:
      logger.warning(f"failed to load weights; no `model_dir` set")
      return
    
    latest = tf.train.latest_checkpoint(self.model_dir)
    if latest is None:
      logger.info(f"no checkpoint found in {self.model_dir}")
    else:
      self.model.load_weights(latest)
      logger.info(f"restored model from {latest}")


class FunctionalModel(Model):
  def __init__(self, input_shape,
               **kwargs):
    self.input_shape = input_shape
    inputs = keras.layers.Input(self.input_shape)
    outputs = self.forward(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    super().__init__(model, **kwargs)

  def forward(self, inputs):
    """Forward inference function for the model.

    Subclasses override this.

    :param inputs: 
    :returns: 
    :rtype: 

    """
    return inputs

  
class HourglassModel(FunctionalModel):
  def __init__(self, tile_shape,
               level_filters=[64,64,128],
               level_depth=2,
               valid=True,
               pool_dropout=0.25,
               concat_dropout=0.5,
               **kwargs):
    """Create an hourglass-shaped model for object detection.

    :param tile_shape: shape of the output tiles
    :param level_filters: 
    :param level_depth: 
    :param valid: whether to use valid padding
    :param pool_dropout: 
    :param concat_dropout: 
    :returns: 
    :rtype: 

    """
    self.tile_shape = tile_shape
    self.level_filters = level_filters
    self.levels = len(level_filters)
    self.level_depth = level_depth
    self.valid = valid
    self.padding = 'valid' if self.valid else 'same'
    self.pool_dropout = pool_dropout
    self.concat_dropout = concat_dropout
    self.pad = self.calc_pad()
    input_shape = (tile_shape[0] + 2*self.pad, tile_shape[1] + 2*self.pad, 1)
    super().__init__(input_shape, **kwargs)

  def calc_pad(self):
    if not self.valid:
      return 0
    pad = self.level_depth
    for _ in range(self.levels - 1):
      pad = 2*pad + 2*self.level_depth
    return pad
    
  def compile(self, learning_rate=0.1, **kwargs):
    kwargs['optimizer'] = kwargs.get(
      'optimizer', tf.train.AdadeltaOptimizer(learning_rate))
    kwargs['loss'] = kwargs.get('loss', 'mse')
    kwargs['metrics'] = kwargs.get('metrics', ['mae'])
    self.model.compile(**kwargs)

  def forward(self, inputs):
    level_outputs = []

    for i, filters in enumerate(self.level_filters):
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters, padding=self.padding)
      if i < len(self.level_filters) - 1:
        level_outputs.append(inputs)
        inputs = maxpool(inputs, dropout=self.pool_dropout)

    level_outputs = reversed(level_outputs)
    for i, filters in enumerate(reversed(self.level_filters[:-1])):
      inputs = conv_transpose(inputs, filters)
      outputs = next(level_outputs)
      if self.valid:
        outputs = crop(outputs, inputs.shape)
      inputs = concat(outputs, inputs, dropout=self.concat_dropout)
      for _ in range(self.level_depth):
        inputs = conv(inputs, filters, padding=self.padding)

    inputs = conv(inputs, 1, activation=None, padding='same')
    return inputs

  def full_predict(self, data, steps=20):
    """Yield reassembled fields from the data.

    Requires batch_size to be a multiple of num_tiles

    :param data: dat.Data set
    :param size: number of examples in the dataset
    :returns: 
    :rtype: 

    """
    assert data.batch_size % data.num_tiles == 0
    for _ in range(data.size // data.batch_size):
      tiles = self.predict(data.eval_input, steps=steps, verbose=2)
      for i in range(0, steps*data.batch_size // data.num_tiles, data.num_tiles):
        yield data.untile(tiles[i:i+data.num_tiles])
  
  def detect(self, data, max_iter=None):
    """Detect objects in the reassembled fields.

    :param data: dat.Data set
    :returns: generator over these elements

    """
    labels = np.zeros((data.size, data.num_objects, 3), np.float32)
    for i, field in enumerate(self.full_predict(data)):
      if max_iter is not None and i >= max_iter:
        break
      labels[i] = data.from_field(field)
    return labels
