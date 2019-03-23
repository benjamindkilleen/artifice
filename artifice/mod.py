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
  pass
