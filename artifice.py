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
from artifice import dat, mod, docs, vis, conversions, utils, img

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
  :param transformation: 
  :param image_shape: 
  :param data_size: 
  :param test_size: 
  :param batch_size: 
  :param num_objects: 
  :param pose_dim: 
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
  :param cache: 
  :returns: 
  :rtype: 

  # todo: copy the above from docs file

  """
  def __init__(self, *, commands, data_root, model_root, overwrite,
               convert_mode, transformation, identity_prob, select_mode,
               annotation_mode, image_shape, data_size, test_size, batch_size,
               num_objects, pose_dim, base_shape, level_filters, level_depth,
               dropout, initial_epoch, epochs, learning_rate,
               num_parallel_calls, verbose, keras_verbose, eager, show, cache):
    # main
    self.commands = commands

    # file settings
    self.data_root = data_root
    self.model_root = model_root
    self.overwrite = overwrite

    # data modes
    self.convert_modes = utils.listwrap(convert_mode)
    self.transformation = transformation
    self.identity_prob = identity_prob
    self.select_mode = select_mode
    self.annotation_mode = annotation_mode

    # data sizes
    self.image_shape = image_shape
    self.data_size = data_size
    self.test_size = test_size
    self.epoch_size = epoch_size
    self.batch_size = batch_size
    self.num_objects = num_objects
    self.pose_dim = pose_dim

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

    # derived model subdirs/paths
    self.cache_dir = join(self.model_root, 'cache')
    self.annotation_info_path = join(self.model_root, 'annotation_info.json')

    # ensure directories exist
    _ensure_dirs_exist([self.data_root, self.model_root, self.cache_dir,
                        join(self.data_root, "annotated")]))

  def __str__(self):
    return f"""{asctime()}:
num_parallel_calls: {self.num_parallel_calls}
todo: other attributes"""

  def __call__(self):
    for command in self.commands:
      if (command[0] == '_' or not hasattr(self, command) or
          not callable(getattr(self, command))):
        raise RuntimeError(f"bad command: {command}")
      getattr(self, command)()

  def _set_num_parallel_calls(self):
    if self.num_parallel_calls <= 0:
      self.num_parallel_calls = os.cpu_count()

  @property
  def _data_kwargs(self):
    return {'image_shape' : self.image_shape,
            'input_tile_shape' : self.input_tile_shape,
            'output_tile_shape' : self.output_tile_shape,
            'batch_size' : self.batch_size,
            'num_parallel_calls' : self.num_parallel_calls,
            'num_shuffle' : min(self.data_size, 1000),
            'cache_dir' : self.cache_dir if self.cache else None}
  def _load_labeled(self):
    return dat.LabeledData(join(self.data_root, 'labeled_set.tfrecord'),
                           size=self.data_size, **self._data_kwargs)
  def _load_unlabeled(self):
    return dat.UnlabeledData(join(self.data_root, 'unlabeled_set.tfrecord'),
                             size=self.data_size, **self._data_kwargs)
  def _load_annotated(self):
    return dat.AnnotatedData(join(self.data_root, 'annotated'),
                             transformation=tform.transformations[self.transformation],
                             size=self.data_size, **self._data_kwargs)
  def _load_test(self):
    return dat.LabeledData(join(self.data_root, 'test_set.tfrecord'),
                           size=self.test_size, **self._data_kwargs)

  def _load_model(self, expect_checkpoint=False):
    return mod.ProxyUNet(base_shape=self.base_shape,
                         level_filters=self.level_filters,
                         num_channels=self.image_shape[2],
                         pose_dim=self.pose_dim, level_depth=self.level_depth,
                         dropout=self.dropout, model_dir=self.model_root,
                         learning_rate=self.learning_rate,
                         overwrite=self.overwrite,
                         expect_checkpoint=expect_checkpoint)

  #################### Methods implementing Commands ####################
  
  def convert(self):
    for mode in self.convert_modes:
      conversions.conversions[mode](
        self.data_root, test_size=self.test_size)

  def train(self):
    labeled_set = self._load_labeled()
    model = self._load_model()
    model.train(labeled_set, epochs=self.epochs,
                initial_epoch=self.initial_epoch,
                verbose=self.keras_verbose)

  def evaluate(self):
    test_set = self._load_test()
    model = self._load_model(expect_checkpoint=True)
    errors, num_failed = model.evaluate(test_set)
    avg_error = errors.mean(axis=0)
    total_num_objects = self.test_size * self.num_objects
    num_detected = total_num_objects - num_failed
    logger.info(f"objects detected: {num_detected} / "
                f"{total_num_objects}")
    logger.info(f"avg (euclidean) detection error: {avg_error[0]}")
    logger.info(f"avg (absolute) pose error: {avg_error[1:]}")
    logger.info("note: some objects may be occluded, making detection impossible")
    logger.info(f"std: {errors.std(axis=0)}")
    logger.info(f"min: {errors.min(axis=0)}")
    logger.info(f"max: {errors.max(axis=0)}")

  def visualize(self):
    test_set = self._load_test()
    for images, proxies in test_set.training_input:
      image = images[0]
      proxy = proxies[0]
      vis.plot_image(image, proxy[:,:,0])
      plt.show()

  def select(self):
    """Run selection using an active learning or other strategy. 

    Note that this does not perform any labeling. It simply maintains a queue of
    the indices for examples most recently desired for labeling. This queue
    contains no repeats. The queue is saved to disk, and a file lock should be
    created whenever it is altered, ensuring that the annotator does not make a
    bad access.

    """

    pass
    
    # todo: pick a selector, which could require data and a model or just the
    # size of the data. Probably needs dataset to select from, in which case
    # data_size should be 
    
  def annotate(self):
    """Continually annotate new examples.

    Continually access the selection queue, pop off the most recent, and
    annotate it, either with a human annotator, or automatically using prepared
    labels (and a sleep timer). Needs to keep a list of examples already
    annotated, since they will be strewn throughout different files, as well as
    respect the file lock on the queue.

    """
    pass

def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('commands', nargs='+', help=docs.commands)

  # file settings
  parser.add_argument('--data-root', '--input', '-i', nargs=1,
                      default=['data/default'],
                      help=docs.data_root)
  parser.add_argument('--model-root', '--model-dir', '-m', nargs=1,
                      default=['models/tmp'],
                      help=docs.model_root)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite)

  # data settings
  parser.add_argument('--convert-mode', nargs='+', default=[0, 4], type=int,
                      help=docs.convert_mode)
  parser.add_argument('--transformation', '--augmentation', '-a', nargs=1,
                      default=[None], type=int, help=docs.transformation)
  parser.add_argument('--identity-prob', nargs=1, default=[0.01], type=float,
                      help=docs.identity_prob)
  parser.add_argument('--select-mode', '--select', nargs=1, default=['random'],
                      help=docs.select_mode)
  parser.add_argument('--annotation-mode', '--annotate', nargs=1,
                      default=['prelabeled'], help=docs.annotation_mode)

  # sizes relating to data
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3, type=int,
                      default=[100,100,1], help=docs.image_shape)
  parser.add_argument('--data-size', '-N', nargs=1, default=[2000], type=int,
                      help=docs.data_size)
  parser.add_argument('--test-size', '-T', nargs=1, default=[100], type=int,
                      help=docs.test_size)
  parser.add_argument('--batch-size', '-b', nargs=1, default=[4], type=int,
                      help=docs.batch_size)
  parser.add_argument('--num-objects', '-n', nargs=1, default=[4], type=int,
                      help=docs.num_objects)
  parser.add_argument('--pose-dim', '-p', nargs=1, default=[2], type=int,
                      help=docs.pose_dim)

  # model architecture
  parser.add_argument('--base-shape', nargs='+', default=[32], type=int,
                      help=docs.base_shape)
  parser.add_argument('--level-filters', nargs='+', default=[32,64,128],
                      type=int, help=docs.level_filters)
  parser.add_argument('--level-depth', nargs='+', default=[2], type=int,
                      help=docs.level_depth)

  # model hyperparameters
  parser.add_argument('--dropout', nargs=1, default=[0.5], type=float,
                      help=docs.dropout)
  parser.add_argument('--initial-epoch', nargs=1, default=[0], type=int,
                      help=docs.initial_epoch) # todo: get from ckpt
  parser.add_argument('--epochs', '-e', nargs=1, default=[1], type=int,
                      help=docs.epochs)
  parser.add_argument('--learning-rate', '-l', nargs=1, default=[0.1],
                      type=float, help=docs.learning_rate)

  # runtime settings
  parser.add_argument('--num-parallel-calls', '--cores', nargs=1, default=[-1],
                      type=int, help=docs.num_parallel_calls)
  parser.add_argument('--verbose', '-v', nargs=1, default=[2], type=int,
                      help=docs.verbose)
  parser.add_argument('--keras-verbose', nargs=1, default=[1], type=int,
                      help=docs.keras_verbose)
  parser.add_argument('--patient', action='store_true', help=docs.patient)
  parser.add_argument('--show', action='store_true', help=docs.show)
  parser.add_argument('--cache', action='store_true', help=docs.cache)

  args = parser.parse_args()
  art = Artifice(commands=args.commands, convert_mode=args.convert_mode,
                 transformation=args.transformation,
                 identity_prob=args.identity_prob[0],
                 select_mode=args.select_mode[0],
                 annotation_mode=args.annotation_mode[0],
                 data_root=args.data_root[0], model_root=args.model_root[0],
                 overwrite=args.overwrite, image_shape=args.image_shape,
                 data_size=args.data_size[0], test_size=args.test_size[0],
                 batch_size=args.batch_size[0], num_objects=args.num_objects[0],
                 pose_dim=args.pose_dim[0], base_shape=args.base_shape,
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
