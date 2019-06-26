#!/usr/bin/env python

"""The main script for running artifice.

"""

import os
from os.path import join, exists
import logging
import argparse
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from artifice import dat, mod, learn, docs, img, vis, conversions


logger = logging.getLogger('artifice')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:artifice:%(message)s'))
logger.addHandler(handler)

logger.debug(f"Use Python{3.6} or higher.")

def _set_verbosity(verbose):
  if verbose == 0:
    logger.setLevel(logging.WARNING)
  elif verbose == 1:
    logger.setLevel(logging.INFO)
  else:
    logger.setLevel(logging.DEBUG)

def _set_eager(eager):
  if eager:
    tf.enable_eager_execution()

def _set_show(show):
  if not show:
    mpl.use('Agg')
    plt.ioff()

def _ensure_dirs_exist(dirs):
  for path in dirs:
    if not exists(path):
      logger.info(f"creating '{path}'")
      os.makedirs(path)


class Artifice:
  """Bag of state for each run of `artifice`."""
  def __init__(self, command=[], mode='default', data_root='data/default',
               model_root='models/default', overwrite=False, cache_dir='data/.cache',
               image_shape=[100,100,1], tile_shape=[100,100,1], data_size=5000,
               test_size=500, epoch_size=10000, batch_size=4, num_objects=4,
               initial_epoch=0, epochs=1, learning_rate=0.1,
               num_parallel_calls=-1, verbose=1, keras_verbose=2, eager=True,
               show=False):
    # 
    self.commands = commands
    self.mode = mod

    # file settings
    self.data_root = data_root
    self.model_root = model_root
    self.overwrite = overwrite
    self.cache_dir = cache_dir

    # data sizes
    self.image_shape = image_shape
    self.tile_shape = tile_shape
    self.data_size = data_size
    self.test_size = test_size
    self.epoch_size = epoch_size
    self.batch_size = batch_size
    self.num_objects = num_objects

    # hyperparameters
    self.initial_epoch = 0
    self.epochs = 1
    self.learning_rate = learning_rate
    self.num_parallel_calls = num_parallel_calls
    self.verbose = verbose
    self.keras_verbose = keras_verbose
    self.eager = eager
    self.show = show

    # globals
    _set_verbosity(self.verbose)
    _set_eager(self.eager)
    _set_show(self.show)
    self._set_num_parallel_calls()

    # derived sizes/shapes
    self.input_tile_shape

    # relating to tiling
    self.num_tiles = int(np.ceil(self.image_shape[0] / self.tile_shape[0]) *
                         np.ceil(self.image_shape[1] / self.tile_shape[1]))
    
    # standard model input data paths
    self.unlabeled_set_path = join(self.data_root, 'unlabeled_set.tfrecord')
    self.labeled_set_path = join(self.data_root, 'labeled_set.tfrecord')

    # ensure directories exist
    _ensure_dirs_exist([self.data_root, self.model_root, self.cache_dir])

  def __str__(self):
    return "todo: make artifice string to represent all this info"

  def __call__(self):
    for command in self.commands:
      if (command[0] == '_' or not hasattr(self, command) or
          not callable(getattr(self, command))):
        raise RuntimError(f"bad command: {command}")
      getattr(self, command)()
  
  def _set_cores(self):
    if self.cores <= 0:
      self.cores = os.cpu_count()

  def convert(self):
    conversions.conversions[self.convert_mode](
      self.data_root, num_parallel_calls=self.num_parallel_calls)

def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('commands', nargs='+', help=docs.commands)
  parser.add_argument('--mode', nargs=1, default=['augmented-active'],
                      help=docs.mode)

  # file settings
  parser.add_argument('--data-root', '--input', '-i', nargs=1,
                      default=['data/default'],
                      help=docs.data_root)
  parser.add_argument('--model-root', '--model-dir', '-m', nargs=1,
                      default=['models/default'],
                      help=docs.model_root)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite)
  parser.add_argument('--cache-dir', nargs=1,
                      default=['data/.cache'],
                      help=docs.cache_dir)

  # data conversion settings
  parser.add_argument('--convert-mode', nargs=1,
                      default=[0], type=int,
                      help=docs.convert_mode)

  # sizes relating to data
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3,
                      type=int, default=[100,100,1],
                      help=docs.image_shape)
  parser.add_argument('--data-size', '-N', nargs=1,
                      default=[3000], type=int,
                      help=docs.data_size)
  parser.add_argumnet('--test-size', '-T', nargs=1,
                      default=[100], type=int,
                      help=docs.test_size)
  parser.add_argument('--epoch-size', '--num-examples', '-n', nargs=1,
                      default=[10000], type=int,
                      help=docs.epoch_size)
  parser.add_argument('--batch-size', '-b', nargs=1,
                      default=[4], type=int,
                      help=docs.batch_size)
  parser.add_argument('--num-objects', nargs=1,
                      default=[2], type=int,
                      help=docs.num_objects)
  
  # model hyperparameters
  parser.add_argument('--base-shape', nargs='1',
                      default=[32], type=int,
                      help=docs.level_filters)
  parser.add_argument('--level-filters', nargs='+',
                      default=[32,64,128], type=int,
                      help=docs.level_filters)
  parser.add_argument('--level-depth', nargs='+',
                      default=[2], type=int,
                      help=docs.level_depth)
  parser.add_argumnet('--dropout', nargs=1,
                      default=[0.5], type=float,
                      help=docs.dropout)
  parser.add_argument('--initial-epoch', nargs=1,
                      default=[0], type=int,
                      help=docs.initial_epoch) # todo: get from ckpt
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[1], type=int,
                      help=docs.epochs)
  parser.add_argument('--learning-rate', '-l', nargs=1,
                      default=[0.1], type=float,
                      help=docs.learning_rate)

  # runtime settings
  parser.add_argument('--num-parallel-calls', '--cores', nargs=1,
                      default=[-1], type=int,
                      help=docs.cores)
  parser.add_argument('--verbose', '-v', nargs=1,
                      default=[2], type=int,
                      help=docs.verbose)
  parser.add_argument('--keras-verbose', nargs=1,
                      default=[1], type=int,
                      help=docs.keras_verbose)
  parser.add_argument('--patient', action='store_false',
                      help=docs.patient)
  parser.add_argument('--show', action='store_true',
                      help=docs.show)
  args = parser.parse_args()
  art = Artifice(args)
  logger.info(art)
  art()

  
if __name__ == "__main__":
  main()
