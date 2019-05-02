#!/usr/bin/env python

"""The main script for running artifice.

"""

import os
from os.path import join, exists
import logging
from glob import glob
import argparse
import json
import numpy as np
from skimage.draw import circle
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from artifice import dat, mod, learn, oracles
from artifice.utils import docs, img, vis, vid
from test_utils import springs


logger = logging.getLogger('artifice')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:artifice:%(message)s'))
logger.addHandler(handler)

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
    self.args = args
    self.command = args.command
    self.mode = args.mode[0]
    self.data_root = args.data_root[0]
    self.model_root = args.model_root[0]
    self.verbose = args.verbose[0]
    self.keras_verbose = args.keras_verbose[0]
    self.overwrite = args.overwrite
    self.image_shape = args.image_shape
    self.tile_shape = args.tile_shape
    self.epochs = args.epochs[0]
    self.batch_size = args.batch_size[0]
    self.learning_rate = args.learning_rate[0]
    self.subset_size = args.subset_size[0]
    self.epoch_size = args.epoch_size[0]
    self.num_objects = args.num_objects[0]
    self.splits = args.splits
    self.cores = args.cores[0] if args.cores[0] > 0 else os.cpu_count()
    self.eager = args.eager
    self.show = args.show
    self.num_candidates = args.num_candidates[0]
    self.query_size = args.query_size[0]
    self.regions_path = args.regions_path[0]

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

    # relating to tiling
    self._pad = None
    self.num_tiles = int(np.ceil(self.image_shape[0] / self.tile_shape[0]) *
                         np.ceil(self.image_shape[1] / self.tile_shape[1]))

    # original input format paths
    self.labels_path = join(self.data_root, 'labels.npy')
    self.annotations_dir = join(self.data_root, 'annotations')
    self.images_dir = join(self.data_root, 'images')
    self._image_paths = None
    self._annotation_paths = None

    # standard model input data paths
    self.annotated_set_path = join(self.data_root, 'annotated_set.tfrecord')
    self.unlabeled_set_path = join(self.data_root, 'unlabeled_set.tfrecord')
    self.labeled_set_path = join(self.data_root, 'labeled_set.tfrecord')
    self.labeled_subset_path = join(
      self.data_root, f'labeled_subset_{self.subset_size}.tfrecord')
    self.validation_set_path = join(self.data_root, 'validation_set.tfrecord')
    self.test_set_path = join(self.data_root, 'test_set.tfrecord')
    self.labels_hist_path = join(self.data_root, 'labels_histogram.pdf')

    # training set sizes
    # self.labeled_subset_size already set
    self.labeled_size = self.splits[0]
    self.annotated_size = self.splits[0]
    self.unlabeled_size = self.splits[0]
    self.validation_size = self.splits[1]
    self.test_size = self.splits[2]

    # number of steps per epochs
    self.train_steps = int(np.ceil(self.epoch_size / self.batch_size))
    self.validation_steps = int(np.ceil(self.validation_size / self.batch_size))
    self.test_steps = int(np.ceil(self.test_size / self.batch_size))

    # model dirs
    self.hourglass_dir = join(self.model_root, 'hourglass/')

    # model-dependent paths, i.e. made during training
    self.model_data_root = join(self.model_root, self.data_root.split(os.sep)[-1])
    self.annotated_subset_dir = join(self.model_root, 'annotated_subsets')
    self.labeled_subset_dir = join(self.model_root, 'labeled_subsets')
    self.history_path = join(self.model_root, 'history.json')

    # model-data dependent paths, i.e. figures and predictions
    self.predicted_fields_path = join(self.model_data_root, 'predicted_fields.npy')
    self.model_detections_path = join(self.model_data_root, 'detections.npy')
    self.detections_video_path = join(self.model_data_root, 'detections.mp4')
    self.example_detection_path = join(self.model_data_root, 'example_detection.pdf')
    self.regional_errors_path = join(self.model_data_root, 'regional_errors.pdf')
    self.regional_losses_path = join(self.model_data_root, 'regional_losses.pdf')

    # ensure directories exist
    for path in [self.data_root, self.model_root,
                 self.model_data_root,
                 self.annotated_subset_dir,
                 self.labeled_subset_dir]:
      if not exists(path):
        os.makedirs(path)


  def __str__(self):
    return self.args.__str__()

  @property
  def pad(self):
    if self._pad is None:
      logger.warning(f"loading data before model")
      logger.warning(f"pad values may be incorrect")
      return 0
    else:
      return self._pad

  @property
  def image_paths(self):
    if self._image_paths is None:
      self._image_paths = sorted(glob(join(self.images_dir, '*.png')))
    return self._image_paths

  @property
  def annotation_paths(self):
    if self._annotation_paths is None:
      self._annotation_paths = sorted(glob(join(self.annotations_dir, '*.npy')))
    return self._annotation_paths

  def make_oracle(self):
    return oracles.PerfectOracle(
      np.load(self.labels_path), self.annotation_paths)

  @property
  def dat_kwargs(self):
    return {'image_shape' : self.image_shape,
            'tile_shape' : self.tile_shape,
            'batch_size' : self.batch_size,
            'num_parallel_calls' : self.cores,
            'pad' : self.pad,
            'num_objects' : self.num_objects}

  def load_unlabeled(self):
    return dat.UnlabeledData(
      self.unlabeled_set_path,
      size=self.unlabeled_size,
      **self.dat_kwargs)

  def load_data(self):
    """Load the unlabeled, annotated, validation, and test sets."""
    unlabeled_set = self.load_unlabeled()
    validation_set = dat.Data(
      self.validation_set_path,
      size=self.validation_size,
      **self.dat_kwargs)
    test_set = dat.Data(
      self.test_set_path,
      size=self.test_size,
      **self.dat_kwargs)
    return unlabeled_set, validation_set, test_set

  def load_labeled(self):
    return dat.Data(self.labeled_set_path, size=self.labeled_size, **self.dat_kwargs)

  def load_annotated(self):
    return dat.AugmentationData(self.annotated_set_path, size=self.annotated_size,
                                **self.dat_kwargs)

  def load_labeled_subset(self):
    if not exists(self.labeled_subset_path) or self.overwrite:
      oracle = self.make_oracle()
      unlabeled_set = self.load_unlabeled()
      indices = np.random.choice(self.labeled_size,
                                 size=self.subset_size, replace=False)
      sampling = np.zeros(self.labeled_size, dtype=np.int64)
      sampling[indices] = 1
      labeled_subset = unlabeled_set.sample_and_label(
        sampling, oracle, self.labeled_subset_path)
      return labeled_subset
    else:
      return dat.Data(self.labeled_subset_path, size=self.subset_size,
                      **self.dat_kwargs)

  def make_model(self):
    """Create and compile the model."""
    model = mod.HourglassModel(
      self.tile_shape,
      model_dir=self.hourglass_dir)

    self._pad = model.pad
    model.compile(learning_rate=self.learning_rate)
    return model

  def load_model(self):
    """Load the model, depending on self.overwrite."""
    model = self.make_model()
    if not self.overwrite:
      model.load()
    return model

  def load_learner(self, model=None):
    """Create a learner around a loaded model."""
    if model is None:
      model = self.load_model()
    learner = learn.ActiveLearner(
      model, self.make_oracle(),
      num_candidates=self.num_candidates,
      query_size=self.query_size,
      subset_size=self.subset_size)
    return learner

  def save_history(self, history):
    with open(self.history_path, 'w') as f:
      f.write(json.dumps(history))

def cmd_convert(art):
  """Standardize input data."""
  labels = np.load(art.labels_path)
  logger.debug(art.image_paths, labels)
  example_iterator = zip(art.image_paths, labels)
  annotation_iterator = iter(art.annotation_paths)
  more_annotations = True

  # over unlabeled set
  logger.info(f"writing unlabeled set to '{art.unlabeled_set_path}'...")
  annotated_writer = tf.python_io.TFRecordWriter(art.annotated_set_path)
  unlabeled_writer = tf.python_io.TFRecordWriter(art.unlabeled_set_path)
  labeled_writer = tf.python_io.TFRecordWriter(art.labeled_set_path)
  i = -1
  for i in range(art.unlabeled_size):
    if i % 100 == 0:
      logger.info(f"writing {i} / {art.unlabeled_size}")
    image_path, label = next(example_iterator)
    image = img.open_as_array(image_path)
    unlabeled_writer.write(dat.proto_from_image(image))
    labeled_writer.write(dat.proto_from_example((image, label)))
    if more_annotations:
      try:
        annotation_path = next(annotation_iterator)
        annotation = np.load(annotation_path)
        scene = (image, label), annotation
        annotated_writer.write(dat.proto_from_scene(scene))
      except StopIteration:
        more_annotations = False
  unlabeled_writer.close()
  labeled_writer.close()
  annotated_writer.close()
  logger.info("finished")
  logger.info(f"wrote {i+1} unlabeled images")

  # Collect the validation set
  logger.info(f"writing validation set to '{art.validation_set_path}'...")
  writer = tf.python_io.TFRecordWriter(art.validation_set_path)
  for i in range(art.validation_size):
    if i % 100 == 0:
      logger.info(f"writing {i} / {art.validation_size}")
    image_path, label = next(example_iterator)
    image = img.open_as_array(image_path)
    example = (image, label)
    writer.write(dat.proto_from_example(example))
  writer.close()
  logger.info("finished")
  logger.info(f"wrote {i+1} validation examples")

  # Collect the test set
  logger.info(f"writing test set to '{art.test_set_path}'...")
  writer = tf.python_io.TFRecordWriter(art.test_set_path)
  for i in range(art.test_size):
    if i % 100 == 0:
      logger.info(f"writing {i} / {art.test_size}")
    image_path, label = next(example_iterator)
    image = img.open_as_array(image_path)
    example = (image, label)
    writer.write(dat.proto_from_example(example))
  writer.close()
  logger.info("finished")
  logger.info(f"wrote {i+1} test examples")


def cmd_train(art):
  model = art.load_model()
  unlabeled_set, validation_set, test_set = art.load_data()
  kwargs = {'epochs' : art.epochs,
            'steps_per_epoch' : art.train_steps,
            'validation_data' : validation_set.eval_input,
            'validation_steps' : art.validation_steps,
            'verbose' : art.keras_verbose}

  if art.mode == 'full':
    # run "traditional" training on the full, labeled dataset
    labeled_set = art.load_labeled()
    hist = model.fit(labeled_set.training_input, **kwargs)

  elif art.mode == 'random':
    # Label a small, random subset of the data, as a human might
    labeled_subset = art.load_labeled_subset()
    hist = model.fit(labeled_subset.training_input, **kwargs)

  elif art.mode == 'active':
    # Label a small, actively selected subset of the data during training
    learner = art.load_learner(model)
    hist = learner.fit(unlabeled_set, art.labeled_subset_dir, **kwargs)

  elif art.mode == 'augmented-full':
    # run training with full dataset, augmented
    annotated_set = art.load_annotated()
    hist = model.fit(annotated_set.training_input, **kwargs)

  elif art.mode == 'augmented-random':
    # use a random set of original examples, augmented
    annotated_set = art.load_annotated()
    hist = model.fit(annotated_set.training_input, **kwargs)

  elif art.mode == 'augmented-active':
    # actively select examples and augment
    learner = art.load_learner(model)
    hist = learner.fit(unlabeled_set, art.annotated_subset_dir, **kwargs)
  else:
    raise RuntimeError(f"no such mode: '{art.mode}'")

  art.save_history(hist)


def cmd_predict(art):
  model = art.load_model()
  unlabeled_set, validation_set, test_set = art.load_data()
  predictions = model.full_predict(test_set, steps=1, verbose=art.keras_verbose)
  if art.show and tf.executing_eagerly():
    for i, (prediction, example) in enumerate(zip(predictions, test_set.fielded())):
      image, field = example
      vis.plot_image(image, field, prediction)
      plt.show()

def cmd_evaluate(art):
  pass

def cmd_detect(art):
  """Run detection and show some images with true/predicted positions."""
  model = art.load_model()
  unlabeled_set, validation_set, test_set = art.load_data()
  detections, fields = model.detect(test_set)
  np.save(art.model_detections_path, detections)
  logger.info(f"saved detections to {art.model_detections_path}")
  np.save(art.predicted_fields_path, fields)
  logger.info(f"saved predicted_fields to {art.predicted_fields_path}")
  labels = test_set.labels
  errors = np.linalg.norm(detections[:,:,1:3] - labels[:,:,1:3], axis=2)
  logger.info(f"average error: {errors.mean():.02f}")
  logger.info(f"error std: {errors.std():.02f}")
  logger.info(f"minimum error: {errors.min():.02f}")
  logger.info(f"maximum error: {errors.max():.02f}")

def cmd_visualize(art):
  unlabeled_set, validation_set, test_set = art.load_data()
  labels = test_set.labels
  detections = np.load(art.model_detections_path)
  fields = np.load(art.predicted_fields_path)
  errors = np.linalg.norm(detections[:,:,1:3] - labels[:,:,1:3], axis=2)
  logger.info(f"average error: {errors.mean():.02f}")
  logger.info(f"error std: {errors.std():.02f}")
  logger.info(f"minimum error: {errors.min():.02f}")
  logger.info(f"maximum error: {errors.max():.02f}")

  # visualize the color map of errors
  vis.plot_errors(labels, errors, art.image_shape)
  if art.show:
    plt.show()
  else:
    plt.savefig(art.regional_errors_path)
    logger.info(f"saved error map to {art.regional_errors_path}")

  # visualize the losses at each point
  losses = np.zeros((fields.shape[0], art.num_objects))
  for i in range(fields.shape[0]):
    if i % 100 == 0:
      logger.info(f"calculating object-wise loss at {i}/{fields.shape[0]}")
    for j in range(labels.shape[1]):
      true_field = test_set.to_numpy_field(labels[i])
      pred_field = fields[i]
      rr, cc = circle(labels[i,j,1], labels[i,j,2], 20., shape=art.image_shape[:2])
      losses[i,j] = np.square(pred_field[rr,cc] - true_field[rr,cc]).mean()
  vis.plot_errors(labels, losses, art.image_shape)
  if art.show:
    plt.show()
  else:
    plt.savefig(art.regional_losses_path)
    logger.info(f"saved losses map to {art.regional_losses_path}")

  # get_next = test_set.dataset.make_one_shot_iterator().get_next()
  # writer = vid.MP4Writer(art.detections_video_path)
  # logger.info(f"writing detections to video...")
  # with tf.Session() as sess:
  #   for i, detection in enumerate(detections):
  #     if i % 100 == 0:
  #       logger.info(f"{i} / {detections.shape[0]}")
  #     image, label = sess.run(get_next)
  #     fig, _ = vis.plot_detection(label, detection, image, fields[i])
  #     if i == 0:
  #       writer.write_fig(fig, close=False)
  #       plt.savefig(art.example_detection_path)
  #       logger.info(f"saved example detection to {art.example_detection_path}")
  #     else:
  #       writer.write_fig(fig)
  # writer.close()
  # logger.info(f"finished")
  # logger.info(f"wrote mp4 to {art.detections_video_path}")

def cmd_analyze(art):
  """Analayze the detections for a spring constant."""
  labels = np.load(art.labels_path)
  ls = springs.find_constant(labels)
  plt.plot(ls, 'b.')
  if art.show:
    plt.show()
  else:
    plt.close()

  vis.plot_labels(labels, art.image_shape)
  if art.show:
    plt.show()
  else:
    plt.close() # savefig(art.labels_hist_path)

def main():
  parser = argparse.ArgumentParser(description=docs.description)
  parser.add_argument('command', help=docs.command_help)
  parser.add_argument('--mode', nargs=1, default=['augmented-active'],
                      help=docs.mode_help)
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
                      type=int, default=[196,196,1],
                      help=docs.image_shape_help)
  parser.add_argument('--tile-shape', nargs=3,
                      type=int, default=[196,196,1],
                      help=docs.tile_shape_help)
  parser.add_argument('--epochs', '-e', nargs=1,
                      default=[1], type=int,
                      help=docs.epochs_help)
  parser.add_argument('--splits', nargs=3,
                      default=[10000,1000,1000],
                      type=int,
                      help=docs.splits_help)
  parser.add_argument('--batch-size', '-b', nargs=1,
                      default=[4], type=int,
                      help=docs.batch_size_help)
  parser.add_argument('--learning-rate', '-l', nargs=1,
                      default=[0.1], type=float,
                      help=docs.learning_rate_help)
  parser.add_argument('--subset-size', nargs=1, default=[10], type=int,
                      help=docs.subset_size_help)
  parser.add_argument('--num-candidates', nargs=1,
                      default=[1000], type=int,
                      help=docs.num_candidates_help)
  parser.add_argument('--query-size', nargs=1,
                      default=[1], type=int,
                      help=docs.query_size_help)
  parser.add_argument('--epoch-size', '--num-examples', '-n', nargs=1,
                      default=[10000], type=int,
                      help=docs.epoch_size_help)
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
                      default=[1], type=int,
                      help=docs.verbose_help)
  parser.add_argument('--keras-verbose', nargs=1,
                      default=[1], type=int,
                      help=docs.keras_verbose_help)
  parser.add_argument('--eager', action='store_true',
                      help=docs.eager_help)
  parser.add_argument('--show', action='store_true',
                      help=docs.show_help)
  parser.add_argument('--regions-path', '--regions', nargs=1,
                      default=[None], help=docs.regions_help)
  args = parser.parse_args()
  art = Artifice(args)
  logger.info(art)

  if art.command == 'convert':
    cmd_convert(art)
  elif art.command == 'train':
    cmd_train(art)
  elif art.command == 'predict':
    cmd_predict(art)
  elif art.command == 'detect':
    cmd_detect(art)
  elif art.command == 'visualize':
    cmd_visualize(art)
  elif art.command == 'analyze':
    cmd_analyze(art)
  else:
    logger.error(f"No command '{args.command}'.")

if __name__ == "__main__":
  main()
