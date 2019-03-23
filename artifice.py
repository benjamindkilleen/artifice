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
import numpy as np
import argparse
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from artifice.utils import docs
from multiprocessing import cpu_count

logger.debug(f"Use Python{3.6} or higher.")


class Artifice:
  """Bag of state for each run of `artifice`."""
  def init(self, args):
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
    
    # ensure directories exist
    for path in [self.data_root, self.model_root]:
      if not os.path.exists(path):
        os.makedirs(path)

  def __str__(self):
    return f"<run '{self.command}'>"

  def load_data(self):
    return None

def cmd_convert(art):
  pass

def cmd_train(art):
  pass

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
                      default=['data/data']
                      help=docs.input_help)
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
                      default=[10,1000,1000], # TODO: fix for coupled_spheres data
                      type=int,
                      help=docs.splits_help)
  parser.add_argument('--num-examples', '-n', nargs=1,
                      default=[1], type=int,
                      help=docs.num_examples_help)
  parser.add_argument('--num-classes', '--classes', '-c', nargs=1,
                      default=[2], type=int,
                      help=docs.num_classes_help)
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
