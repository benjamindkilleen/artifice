"""The main script for running artifice.

"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from artifice.utils import docs, dataset
from artifice.semantic_segmentation import UNet
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


logger.debug(f"using Python{3} sanity check.")



def cmd_experiment(args):
  logger.info(f"training from experiment '{args.input[0]}'")
  data = dataset.load(args.input[0])
  unet = UNet(args.image_shape, args.num_classes[0], model_dir=args.model_dir[0])
  unet.train(data, overwrite=args.overwrite, num_epochs=args.epochs[0], 
             eval_secs=args.eval_secs[0])


def cmd_predict(args):
  logger.info("Predict")
  data = dataset.load(args.input[0])
  unet = UNet(args.image_shape, args.num_classes[0], model_dir=args.model_dir[0])
  predictions = unet.predict(data)
  originals = dataset.read_tfrecord(args.input[0])

  if args.output[0] == 'show':
    for i, prediction in enumerate(predictions):
      if 0 < args.num_examples[0] <= i:
        break
      image, annotation = next(originals)
      fig, (image_ax, truth_ax, pred_ax) = plt.subplots(1, 3)
      image_ax.imshow(np.squeeze(image), cmap='gray')
      image_ax.set_title("Original Image")
      truth_ax.imshow(np.squeeze(annotation))
      truth_ax.set_title("Annotation")
      pred_ax.imshow(np.squeeze(prediction['annotation']))
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
                      default=[2], type=int,
                      help=docs.num_classes_help)
  eval_time = parser.add_mutually_exclusive_group()
  eval_time.add_argument('--eval_secs', nargs=1,
                         default=[1200], type=int,
                         help=docs.eval_secs_help)
  eval_time.add_argument('--eval_mins', nargs=1,
                         default=[None], type=int,
                         help=docs.eval_mins_help)

  args = parser.parse_args()

  if args.command == 'experiment':
    if args.eval_mins[0] is not None:
      args.eval_secs[0] = args.eval_mins[0] * 60
    cmd_experiment(args)
  elif args.command == 'predict':
    if args.output is None:
      raise ValueError("")
    cmd_predict(args)
  else:
    RuntimeError()


if __name__ == "__main__":
  main()
