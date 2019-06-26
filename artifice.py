#!/usr/bin/env python

"""The main script for running artifice.

"""

from time import time, asctime
import os
from os.path import join, exists
import logging
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from artifice import dat, mod, docs, vis, conversions, utils

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
  """Bag of state that controls a single `artifice` run.

  All arguments are required keyword arguments, for the sake of
  correctness. Defaults are specified in the command-line defaults for this
  script. Run `python artifice.py -h` for more info.

  :param commands:
  :param mode:
  :param data_root:
  :param model_root:
  :param overwrite:
  :param convert_mode:
  :param image_shape:
  :param data_size:
  :param test_size:
  :param epoch_size:
  :param batch_size:
  :param num_objects:
  :param base_shape:
  :param level_filters:
  :param level_depth:
  :param dropout:
  :param initial_epoch:
  :param epochs:
  :param learning_rate:
  :param num_parallel_calls:
  :param verbose:
  :param keras_verbose:
  :param eager:
  :param show:

  # todo: copy the above from docs file

  """
  def __init__(self, *, commands, mode, data_root, model_root, overwrite,
               convert_mode, image_shape, data_size, test_size, epoch_size,
               batch_size, num_objects, base_shape, level_filters, level_depth,
               dropout, initial_epoch, epochs, learning_rate,
               num_parallel_calls, verbose, keras_verbose, eager, show, cache):
    # main
    self.commands = commands
    self.mode = mode

    # file settings
    self.data_root = data_root
    self.model_root = model_root
    self.overwrite = overwrite
    self.convert_mode = convert_mode

    # data sizes
    self.image_shape = image_shape
    self.data_size = data_size
    self.test_size = test_size
    self.epoch_size = epoch_size
    self.batch_size = batch_size
    self.num_objects = num_objects

    # model architecture
    self.base_shape = utils.listify(base_shape, 2)
    self.level_filters = level_filters
    self.level_depth = level_depth

    # hyperparameters
    self.dropout = dropout
    self.initial_epoch = initial_epoch
    self.epochs = epochs
    self.learning_rate = learning_rate

    # runtime settings
    self.num_parallel_calls = num_parallel_calls
    self.verbose = verbose
    self.keras_verbose = keras_verbose
    self.eager = eager
    self.show = show
    self.cache = cache

    # globals
    _set_verbosity(self.verbose)
    _set_eager(self.eager)
    _set_show(self.show)
    self._set_num_parallel_calls()

    # derived sizes/shapes
    self.num_levels = len(self.level_filters)
    self.input_tile_shape = mod.ProxyUNet.compute_input_tile_shape(
      self.base_shape, self.num_levels, self.level_depth)
    self.output_tile_shape = mod.ProxyUNet.compute_output_tile_shape(
      self.base_shape, self.num_levels, self.level_depth)
    self.num_tiles = dat.ArtificeData.compute_num_tiles(
      self.image_shape, self.output_tile_shape)

    # standard model input data paths
    self.unlabeled_set_path = join(self.data_root, 'unlabeled_set.tfrecord')
    self.labeled_set_path = join(self.data_root, 'labeled_set.tfrecord')

    # ensure directories exist
    _ensure_dirs_exist([self.data_root, self.model_root,
                        join(self.data_root, 'cache')])

  def __str__(self):
    return f"""{asctime()}:
num_parallel_calls: {self.num_parallel_calls}
todo: other attributes"""

  def __call__(self):
    for command in self.commands:
      if (command[0] == '_' or not hasattr(self, command) or
          not callable(getattr(self, command))):
        raise RuntimError(f"bad command: {command}")
      getattr(self, command)()

  def _set_num_parallel_calls(self):
    if self.num_parallel_calls <= 0:
      self.num_parallel_calls = os.cpu_count()

  @property
  def _data_kwargs(self):
    return {'size' : self.data_size,
            'image_shape' : self.image_shape,
            'input_tile_shape' : self.input_tile_shape,
            'output_tile_shape' : self.output_tile_shape,
            'batch_size' : self.batch_size,
            'num_parallel_calls' : self.num_parallel_calls,
            'num_shuffle' : min(self.data_size, 1000)}

  def _load_labeled(self):
    return dat.LabeledData(self.labeled_set_path, **self._data_kwargs)

  def convert(self):
    conversions.conversions[self.convert_mode](
      self.data_root, num_parallel_calls=self.num_parallel_calls)

  def vis(self):
    logger.debug(f"labeled_set: {self.labeled_set_path}")
    labeled_set = self._load_labeled()
    for i, (images, proxies) in enumerate(labeled_set.training_input):
      if i == labeled_set.steps:
        break
      if i == 0:
        t1 = time()
      image, proxy = images[0], proxies[0]
      logger.info(f"example {i}")
      logger.info(f"  image: {image.shape}")
      logger.info(f"  proxy: {proxy.shape}")
      if self.show:
        vis.plot_image(image, proxy[:,:,0], proxy[:,:,1], proxy[:,:,2])
        plt.show()
    t2 = time()
    logger.info(f"iterated over dataset in {t2 - t1} seconds")

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
                      default=['models/tmp'],
                      help=docs.model_root)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite)

  # data conversion settings
  parser.add_argument('--convert-mode', nargs=1,
                      default=[0], type=int,
                      help=docs.convert_mode)

  # sizes relating to data
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3,
                      type=int, default=[100,100,1],
                      help=docs.image_shape)
  parser.add_argument('--data-size', '-N', nargs=1,
                      default=[2000], type=int,
                      help=docs.data_size)
  parser.add_argument('--test-size', '-T', nargs=1,
                      default=[100], type=int,
                      help=docs.test_size)
  parser.add_argument('--epoch-size', nargs=1,
                      default=[10000], type=int,
                      help=docs.epoch_size)
  parser.add_argument('--batch-size', '-b', nargs=1,
                      default=[4], type=int,
                      help=docs.batch_size)
  parser.add_argument('--num-objects', '-n', nargs=1,
                      default=[4], type=int,
                      help=docs.num_objects)

  # model architecture
  parser.add_argument('--base-shape', nargs='+',
                      default=[32], type=int,
                      help=docs.base_shape)
  parser.add_argument('--level-filters', nargs='+',
                      default=[32,64,128], type=int,
                      help=docs.level_filters)
  parser.add_argument('--level-depth', nargs='+',
                      default=[2], type=int,
                      help=docs.level_depth)

  # model hyperparameters
  parser.add_argument('--dropout', nargs=1,
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
                      help=docs.num_parallel_calls)
  parser.add_argument('--verbose', '-v', nargs=1,
                      default=[2], type=int,
                      help=docs.verbose)
  parser.add_argument('--keras-verbose', nargs=1,
                      default=[1], type=int,
                      help=docs.keras_verbose)
  parser.add_argument('--patient', action='store_true',
                      help=docs.patient)
  parser.add_argument('--show', action='store_true',
                      help=docs.show)
  parser.add_argument('--cache', action='store_true',
                      help=docs.cache)

  args = parser.parse_args()
  art = Artifice(commands=args.commands, mode=args.mode[0],
                 convert_mode=args.convert_mode[0],
                 data_root=args.data_root[0], model_root=args.model_root[0],
                 overwrite=args.overwrite, image_shape=args.image_shape,
                 data_size=args.data_size[0], test_size=args.test_size[0],
                 epoch_size=args.epoch_size[0], batch_size=args.batch_size[0],
                 num_objects=args.num_objects[0], base_shape=args.base_shape,
                 level_filters=args.level_filters,
                 level_depth=args.level_depth[0], dropout=args.dropout[0],
                 initial_epoch=args.initial_epoch[0], epochs=args.epochs[0],
                 learning_rate=args.learning_rate[0],
                 num_parallel_calls=args.num_parallel_calls[0],
                 verbose=args.verbose[0], keras_verbose=args.keras_verbose[0],
                 eager=(not args.patient), show=args.show, cache=args.cache)
  logger.info(art)
  art()

if __name__ == "__main__":
  main()
