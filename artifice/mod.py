"""Implements artifice's detection scheme from end to end.
"""


import tensorflow as tf
from tensorflow import keras
from shutil import rmtree
import os
import numpy as np
from glob import glob
import logging

logger = logging.getLogger('artifice')


def log_model(model):
  logger.info(f'model: {model.name}')
  log_layers(model.layers)

def log_layers(layers):
  for layer in layers:
    logger.info(
      f"layer:{layer.input_shape} -> {layer.output_shape}:{layer.name}")

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
                            os.path.join(model_dir, "cp-{epoch:04d}.hdf5"))
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

      
      
class HourglassModel(SequentialModel):
  def __init__(self, input_shape,
               level_filters=[32,64,128],
               level_depth=3,
               **kwargs):
    """Create an hourglass-shaped model for object detection.

    :param input_shape: 
    :param level_filters: 
    :param level_depth: 
    :returns: 
    :rtype: 

    """
    self.input_shape = input_shape
    self.level_filters = level_filters
    self.level_depth = level_depth
    model = self.create_model()
    super().__init__(model, **kwargs)

  def compile(self, learning_rate=0.1, **kwargs):
    kwargs['optimizer'] = kwargs.get(
      'optimizer', tf.train.AdadeltaOptimizer(learning_rate))
    kwargs['loss'] = kwargs.get('loss', 'mse')
    kwargs['metrics'] = kwargs.get('metrics', 'mae')
    self.model.compile(**kwargs)

  def create_model(self):
    inputs = keras.layers.Input(self.input_shape)
    
    return inputs
    
