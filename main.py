#!/usr/bin/env python

"""The main script for running artifice.

"""

import logging
logger = logging.getLogger('artifice')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:artifice:%(message)s'))
logger.addHandler(handler)

import os
import numpy as np
import argparse
from artifice.utils import docs, dataset, augment, vis
from artifice.semantic_segmentation import UNet
from multiprocessing import cpu_count


logger.debug(f"using Python{3} sanity check.")


def cmd_experiment(args):
  logger.info(f"training from experiment '{args.input[0]}'")

  augmentation = augment.join(args.augment)
  if args.eval_data is not None:
    train_data_input = datset.load(args.input, train=True,
                                   num_parallel_calls=args.cores[0],
                                   augmentation=augmentation)
    eval_data_input = dataset.load(args.eval_data,
                                   num_parallel_calls=args.cores[0])
  else:
    eval_data_input, train_data_input = dataset.load_data_input(
      args.input,
      input_classes=[dataset.DataInput, dataset.TrainDataInput],
      input_sizes=[args.num_eval[0], -1],
      num_parallel_calls=args.cores[0],
      augmentation=augmentation)

  unet = UNet(args.image_shape, args.num_classes[0], 
              model_dir=args.model_dir[0],
              l2_reg_scale=args.l2_reg[0])
  unet.train(train_data_input, 
             eval_data_input=eval_data_input,
             overwrite=args.overwrite, 
             num_epochs=args.epochs[0],
             eval_secs=args.eval_secs[0])


def cmd_predict(args):
  logger.info("Predict")
  data_input = dataset.load(args.input, num_parallel_calls=args.cores[0])
  unet = UNet(args.image_shape, args.num_classes[0], model_dir=args.model_dir[0])
  predictions = unet.predict(data_input)
  originals = dataset.read_tfrecord(args.input[0])

  if args.output[0] == 'show':
    for i, prediction in enumerate(predictions):
      if 0 < args.num_examples[0] <= i:
        break
      image, annotation = next(originals)
      vis.show_predict(image, annotation, prediction)
  else:
    raise NotImplementedError("use show")


def cmd_evaluate(args):
  logger.info("Evaluate")
  raise NotImplementedError()
  # TODO: evaluate command


def cmd_augment(args):
  """Do a single augmentation "cycle," adding new data based on the provided
  distribution. (Currently for testing.)

  """
  logger.info("Augmentation (testing purposes)")
  auger = dataset.DataAugmenter(args.input,
                                num_parallel_calls=args.cores[0])

  auger.run()

  if args.output[0] == 'show':
    # vis.show_labels(auger.labels)
    vis.show_background(auger.background)
  else:
    raise NotImplementedError("use '-o show'")
  
  # TODO: actual augmentation?
  
    
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=docs.command_choices,
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs='+', required=True,
                      help=docs.input_help)
  parser.add_argument('--output', '-o', nargs=1,
                      default=['show'],
                      help=docs.output_help)
  parser.add_argument('--model-dir', '-m', nargs=1,
                      default=['models/experiment'],
                      help=docs.model_dir_help)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite_help)
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3,
                      type=int, default=[388, 388, 1],
                      help=docs.image_shape_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[-1], type=int,
                      help=docs.epochs_help)
  parser.add_argument('--num-examples', '-n', nargs=1,
                      default=[-1], type=int,
                      help=docs.num_examples_help)
  parser.add_argument('--num-classes', '--classes', '-c', nargs=1,
                      default=[2], type=int,
                      help=docs.num_classes_help)
  eval_time = parser.add_mutually_exclusive_group()
  eval_time.add_argument('--eval-secs', nargs=1,
                         default=[1200], type=int,
                         help=docs.eval_secs_help)
  eval_time.add_argument('--eval-mins', nargs=1,
                         default=[None], type=int,
                         help=docs.eval_mins_help)
  eval_data = parser.add_mutually_exclusive_group()
  eval_data.add_argument('--num-eval', nargs=1,
                         default=[100], type=int,
                         help=docs.eval_data_help)
  eval_data.add_argument('--eval-data', nargs=1,
                         default=None,
                         help=docs.eval_data_help)
  parser.add_argument('--l2-reg', nargs=1,
                      default=[0.0001], type=float,
                      help=docs.l2_reg_help)
  parser.add_argument('--cores', '--num-parallel-calls', nargs=1,
                      default=[-1], type=int,
                      help=docs.cores_help)
  parser.add_argument('--augment', '-a', nargs='+',
                      choices=augment.choices,
                      default = [],
                      help=docs.augment_help)

  args = parser.parse_args()

  if args.cores[0] == -1:
    args.cores[0] = cpu_count()

  if args.command == 'experiment':
    if args.eval_mins[0] is not None:
      args.eval_secs[0] = args.eval_mins[0] * 60
    cmd_experiment(args)
  elif args.command == 'predict':
    if args.output is None:
      raise ValueError("")
    cmd_predict(args)
  elif args.command == 'evaluate':
    cmd_evaluate(args)
  elif args.command:
    cmd_augment(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
