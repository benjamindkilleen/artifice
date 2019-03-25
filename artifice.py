#!/usr/bin/env python

"""The main script for running artifice.

"""

import logging
logger = logging.getLogger('artifice')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:artifice:%(message)s'))
logger.addHandler(handler)

import os
from os.path import join, exists
from glob import glob
import numpy as np
import argparse
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from artifice import dat
from artifice.utils import docs, img, vis
from multiprocessing import cpu_count

logger.debug(f"Use Python{3.6} or higher.")


class Artifice:
  """Bag of state for each run of `artifice`."""
  def __init__(self, args):
    """Process the input args.

    Only absolutely essential copying/processing is stored here. Any dependent
    attributes should be created as @property functions.

    :param args: parsed args from ArgParse

    """

    # copy arguments
    self.command = args.command
    self.data_root = args.data_root[0]
    self.model_root = args.model_root[0]
    self.verbose = args.verbose[0]
    self.overwrite = args.overwrite
    self.image_shape = args.image_shape
    self.epochs = args.epochs[0]
    self.batch_size = args.batch_size[0]
    self.num_annotated = args.num_annotated[0]
    self.num_examples = args.num_examples[0]
    self.num_objects = args.num_objects[0]
    self.splits = args.splits
    self.cores = args.cores[0] if args.cores[0] > 0 else os.cpu_count()
    self.eager = args.eager
    self.show = args.show

    # runtime configurations
    if self.verbose == 0:
      logger.setLevel(logging.WARNING)
    elif self.verbose == 1:
      logger.setLevel(logging.INFO)
    else:
      logger.setLevel(logging.DEBUG)

    if self.eager:
      tf.enable_eager_execution()

    if not self.show:
      mpl.use('Agg')
      plt.ioff()
    else:
      plt.rcParams["figure.figsize"] = (10,5)
    
    # ensure directories exist
    for path in [self.data_root, self.model_root]:
      if not exists(path):
        os.makedirs(path)

    # png input format paths
    self.labels_path = join(self.data_root, 'labels.npy')
    self._image_paths = None
    self._annotation_paths = None

    # standard model input data paths
    self.annotated_set_path = join(self.data_root, 'annotated_set.tfrecord')
    self.train_set_path = join(self.data_root, 'train_set.tfrecord') # not used
    self.validation_set_path = join(self.data_root, 'validation_set.tfrecord')
    self.test_set_path = join(self.data_root, 'test_set.tfrecord')

    # training set sizes
    self.annotated_set_size = self.num_annotated
    self.train_set_size = self.num_examples
    self.unlabeled_set_size = self.splits[0]
    self.validation_set_size = self.splits[1]
    self.test_set_size = self.splits[2]
    
    
  def __str__(self):
    return f"<run '{self.command}'>"

  @property
  def image_paths(self):
    if self._image_paths is None:
      self._image_paths = sorted(glob(join(self.data_root, 'images', '*.png')))
    return self._image_paths
  
  @property
  def annotation_paths(self):
    if self._annotation_paths is None:
      self._annotation_paths = sorted(glob(join(
        self.data_root, 'annotations', '*.npy')))
    return self._annotation_paths

  def load_data(self):
    """Load the train (annotated), validation, and test sets."""
    kwargs = {'image_shape' : self.image_shape,
              'batch_size' : self.batch_size,
              'num_parallel_calls' : self.cores}
    train_set = dat.AugmentationData(
      self.annotated_set_path,
      num_examples=self.num_examples,
      num_objects=self.num_objects,
      **kwargs)
    validation_set = dat.Data(self.validation_set_path, **kwargs)
    test_set = dat.Data(self.test_set_path, **kwargs)
    return train_set, validation_set, test_set

  
def cmd_convert(art):
  """Standardize input data."""
  labels = np.load(art.labels_path)
  example_iterator = zip(art.image_paths, labels)
  annotation_iterator = iter(art.annotation_paths)

  # Over training set, taking annotated examples
  logger.info(f"writing annotated set to '{art.annotated_set_path}'...")
  writer = tf.python_io.TFRecordWriter(art.annotated_set_path)
  for i in range(art.unlabeled_set_size):
    image_path, label = next(example_iterator)
    if i < art.annotated_set_size:
      annotation_path = next(annotation_iterator)
      image = img.open_as_array(image_path)
      annotation = np.load(annotation_path)
      scene = (image, label), annotation
      writer.write(dat.proto_from_scene(scene))
  writer.close()
  logger.info("finished")
  logger.info(f"wrote {art.annotated_set_size} annotated examples")
  
  # Collect the validation set
  logger.info(f"writing validation set to '{art.validation_set_path}'...")
  writer = tf.python_io.TFRecordWriter(art.validation_set_path)
  for i in range(art.validation_set_size):
    image_path, label = next(example_iterator)
    image = img.open_as_array(image_path)
    example = (image, label)
    writer.write(dat.proto_from_example(example))
  writer.close()
  logger.info("finished")
  logger.info(f"wrote {art.validation_set_size} validation examples")

  # Collect the test set
  logger.info(f"writing validation set to '{art.test_set_path}'...")
  writer = tf.python_io.TFRecordWriter(art.test_set_path)
  for _ in range(art.validation_set_size):
    image_path, label = next(example_iterator)
    image = img.open_as_array(image_path)
    example = (image, label)
    writer.write(dat.proto_from_example(example))
  writer.close()
  logger.info("finished")
  logger.info(f"wrote {art.validation_set_size} test examples")
  
  
def cmd_train(art):
  train_set, validation_set, test_set = art.load_data()

  

def cmd_predict(art):
  pass

def cmd_evaluate(art):
  pass

def cmd_augment(art):
  pass
    
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', help=docs.command_help)
  parser.add_argument('--data-root', '--input', '-i', nargs=1,
                      default=['data/coupled_spheres'],
                      help=docs.data_dir_help)
  parser.add_argument('--output', '-o', nargs=1,
                      default=['show'],
                      help=docs.output_help)
  parser.add_argument('--model-root', '--model-dir', '-m', nargs=1,
                      default=['models/coupled_spheres'],
                      help=docs.model_dir_help)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite_help)
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3,
                      type=int, default=[388, 388, 1],
                      help=docs.image_shape_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[1], type=int,
                      help=docs.epochs_help)
  parser.add_argument('--splits', nargs=3,
                      default=[7000,1000,1000],
                      type=int,
                      help=docs.splits_help)
  parser.add_argument('--batch-size', '-b', nargs=1,
                      default=[64], type=int,
                      help=docs.batch_size_help)
  parser.add_argument('--num-annotated', nargs=1,
                      default=[10], type=int,
                      help=docs.num_annotated_help)
  parser.add_argument('--num-examples', '-n', nargs=1,
                      default=[10000], type=int,
                      help=docs.num_examples_help)
  parser.add_argument('--num-objects', nargs=1,
                      default=[2], type=int,
                      help=docs.num_objects_help)
  parser.add_argument('--l2-reg', nargs=1,
                      default=[0.0001], type=float,
                      help=docs.l2_reg_help)
  parser.add_argument('--cores', '--num-parallel-calls', nargs=1,
                      default=[-1], type=int,
                      help=docs.cores_help)
  parser.add_argument('--verbose', '-v', nargs=1,
                      default=[2], type=int,
                      help=docs.verbose_help)
  parser.add_argument('--eager', action='store_true',
                      help=docs.eager_help)
  parser.add_argument('--show', action='store_true',
                      help=docs.show_help)

  args = parser.parse_args()
  art = Artifice(args)
  logger.info(art)

  if art.command == 'convert':
    cmd_convert(art)
  elif art.command == 'train':
    cmd_train(art)
  elif art.command == 'predict':
    cmd_predict(art)
  elif art.command == 'evaluate':
    cmd_evaluate(art)
  elif art.command == 'augment':
    cmd_augment(art)
  else:
    logger.error(f"No command '{args.command}'.")

if __name__ == "__main__":
  main()
