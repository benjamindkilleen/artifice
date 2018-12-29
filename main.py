"""The main script for running artifice.

"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from artifice.utils import docs, dataset
from artifice.semantic_segmentation import UNet
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def cmd_experiment(args):
  logging.info(f"training from experiment '{args.input[0]}'")
  data = dataset.load(args.input[0])
  unet = UNet(args.image_shape, args.num_classes[0], model_dir=args.model_dir[0])
  unet.train(data, overwrite=args.overwrite, num_epochs=args.epochs[0])


def cmd_predict(args):
  logging.info("Predict")
  data = dataset.load(args.input[0])
  unet = UNet(args.image_shape, args.num_classes[0], model_dir=args.model_dir[0])
  predictions = unet.predict(data, num_examples=args.num_examples[0])

  if args.output[0] == 'show':
    prediction = next(predictions)
    fig, (image_ax, pred_ax) = plt.subplots(1, 2)
    image_ax.imshow(np.squeeze(prediction['image']))
    image_ax.set_title("Original Image")
    pred_ax.imshow(prediction['annotation'])
    pred_ax.set_title("Predicted Annotation")
    plt.show()
  else:
    raise NotImplementedError("use show")

    
def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', choices=docs.command_choices,
                      help=docs.command_help)
  parser.add_argument('--input', '-i', nargs=1, required=True,
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
                      type=int, default=[512, 512, 1],
                      help=docs.image_shape_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[-1], type=int,
                      help=docs.epochs_help)
  parser.add_argument('--num_examples', '-n', nargs=1,
                      default=[-1], type=int,
                      help=docs.num_examples_help)
  parser.add_argument('--num_classes', '--classes', '-c', nargs=1,
                      default=[3], type=int,
                      help=docs.num_classes_help)

  args = parser.parse_args()

  if args.command == 'experiment':
    cmd_experiment(args)
  elif args.command == 'predict':
    if args.output is None:
      raise ValueError("")
    if args.output[0] == 'show':
      assert 0 < args.num_examples[0] <= 10
    cmd_predict(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
