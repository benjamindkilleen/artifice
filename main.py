"""The main script for running artifice.

"""

import os
import numpy as np
import argparse
from artifice.utils import docs, dataset
from artifice.semantic_segmentation import UNet
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)





def cmd_experiment(args):
  logging.info(f"loading '{args.input[0]}'")
  data = dataset.load(args.input[0])
  unet = UNet(args.image_shape, 3, model_dir=args.output[0])
  unet.train(data, overwrite=args.overwrite, num_epochs=args.epochs[0])

  
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=['experiment'],
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs=1, required=True,
                      help=docs.input_help)
  parser.add_argument('--output', '-o', nargs=1,
                      default=['models/experiment'],
                      help=docs.output_help)
  parser.add_argument('--overwrite', '-f', action='store_true',
                      help=docs.overwrite_help)
  parser.add_argument('--image-shape', '--shape', '-s', nargs=2,
                      type=int, default=[512, 512, 1],
                      help=docs.image_shape_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[-1], type=int,
                      help=docs.epochs_help)

  args = parser.parse_args()
  
  if args.command == 'experiment':
    cmd_experiment(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
