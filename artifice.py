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
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from artifice import dat, mod, learn, oracles, docs, img, vis, vid


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
               model_root='models/default', overwrite=False,
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

    # ensure directories exist
    _ensure_dirs_exist([self.data_root, self.model_root])
    
    for path in [self.data_root, self.model_root,
                 self.model_data_root,
                 self.annotated_subset_dir,
                 self.labeled_subset_dir]:
      if not exists(path):
        logger.info(f"creating '{path}'")
        os.makedirs(path)

    # relating to tiling
    
    self.num_tiles = int(np.ceil(self.image_shape[0] / self.tile_shape[0]) *
                         np.ceil(self.image_shape[1] / self.tile_shape[1]))

    # regions
    self.regions = None if self.regions_path is None else np.load(self.regions_path)

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
    self.full_detections_path = join(self.model_data_root, 'full_detections.npy')
    self.detections_video_path = join(self.model_data_root, 'detections.mp4')
    self.example_detection_path = join(self.model_data_root, 'example_detection.png')
    self.regional_errors_path = join(self.model_data_root, 'regional_errors.pdf')
    self.regional_peaks_path = join(self.model_data_root, 'regional_peaks.pdf')

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
            'num_objects' : self.num_objects,
            'regions' : self.regions}
  
  def load_unlabeled(self):
    if self.regions_path is not None:
      return dat.RegionBasedUnlabeledData(
        self.unlabeled_set_path,
        size=self.unlabeled_size,
        **self.dat_kwargs)
    else:
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
    if self.regions_path is not None:
      return dat.RegionBasedAugmentationData(
        self.annotated_set_path,
        size=self.annotated_size,
        **self.dat_kwargs)
    else:
      return dat.AugmentationData(
        self.annotated_set_path,
        size=self.annotated_size,
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

  def load_history(self):
    with open(self.history_path, 'r') as f:
      hist = json.loads(f.read())
    return hist
      

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


def cmd_proxy(art):
  _ = art.load_model()
  _, _, test_set = art.load_data()
  get_next = test_set.eval_input.make_one_shot_iterator().get_next()
  with tf.Session() as sess:
    while True:
      images, fields = sess.run(get_next)
      img.save('docs/gyros_tile.png', images[5])
      img.save('docs/gyros_tile_proxy.png', fields[5])
      break
  
def cmd_augment(art):
  """Run augmentation of the train_set. 
  If `art.show`, then show the new examples, otherwise, save the augmented
  train_set.
  """
  annotated_set = art.load_annotated()
  get_next = annotated_set.training_input.make_one_shot_iterator().get_next()
  with tf.Session() as sess:
    while True:
      images, fields = sess.run(get_next)
      img.save('docs/augmented_spheres.png', images[5])
      img.save('docs/augmented_spheres_proxy.png', fields[5])
      break
      # for image, field in zip(images, fields):
      #   vis.plot_image(image, field)
      #   if art.show:
      #     plt.show()
      #   else:
      #     plt.close()
      #     raise RuntimeError("Use --show")
  
def cmd_train(art):
  model = art.load_model()
  unlabeled_set, validation_set, test_set = art.load_data()
  kwargs = {'epochs' : art.epochs,
            'steps_per_epoch' : art.train_steps,
            'verbose' : art.keras_verbose,
            'initial_epoch' : art.initial_epoch}
  
  if art.validation_size > 0:
    kwargs.update({'validation_data' : validation_set.eval_input,
                   'validation_steps' : art.validation_steps})
  if 'active' in art.mode:
    kwargs['previous_history'] = art.load_history()
    kwargs['history_path'] = art.history_path
    learner = art.load_learner(model)

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

def cmd_detect_full(art):
  """Run detection on the entire unlabeled dataset."""
  model = art.load_model()
  _, _, test_set = art.load_data()
  detections, _ = model.detect(test_set, steps=300, save_fields=False)
  np.save(art.model_detections_path, detections)
  logger.info(f"saved detections to {art.full_detections_path}")
  
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
  for i in range(errors.shape[1]):
    logger.info(f"object {i}:")
    logger.info(f"  average error: {errors[:,i].mean():.02f}")
    logger.info(f"  error std: {errors[:,i].std():.02f}")
    logger.info(f"  minimum error: {errors[:,i].min():.02f}")
    logger.info(f"  maximum error: {errors[:,i].max():.02f}")

    
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
  for i in range(errors.shape[1]):
    logger.info(f"object {i}:")
    logger.info(f"  average error: {errors[:,i].mean():.02f}")
    logger.info(f"  error std: {errors[:,i].std():.02f}")
    logger.info(f"  minimum error: {errors[:,i].min():.02f}")
    logger.info(f"  maximum error: {errors[:,i].max():.02f}")

  # # visualize the color map of errors
  # vis.plot_errors(labels, errors, art.image_shape, vmax=5)
  # if art.show:
  #   plt.show()
  # else:
  #   plt.savefig(art.regional_errors_path, transparent=True, pad_inches=0)
  #   logger.info(f"saved error map to {art.regional_errors_path}")

  # indices = np.floor(detections[:,:,1:3]).astype(np.int64)
  # peaks = np.zeros_like(errors)
  # for i in range(peaks.shape[0]):
  #   for j in range(peaks.shape[1]):
  #     peaks[i,j] = fields[i,indices[i,j,0],indices[i,j,1]]
  # fig, _ = vis.plot_errors(labels, peaks, art.image_shape, cmap='gray', vmax=10.)
  # fig.suptitle("Detection Peak Values")
  # if art.show:
  #   plt.show()
  # else:
  #   plt.savefig(art.regional_peaks_path, transparent=True, pad_inches=0)
  #   logger.info(f"saved peaks map to {art.regional_peaks_path}")

  get_next = test_set.dataset.make_one_shot_iterator().get_next()
  writer = vid.MP4Writer(art.detections_video_path)
  logger.info(f"writing detections to video...")
  with tf.Session() as sess:
    for i, detection in enumerate(detections):
      if i % 100 == 0:
        logger.info(f"{i} / {detections.shape[0]}")
      image, label = sess.run(get_next)
      frame = vis.frame_detection(label, detection, image, fields[i])
      if i == 0:
        img.save(art.example_detection_path, frame)
        logger.info(f"saved example detection to {art.example_detection_path}")
      writer.write(frame)
  writer.close()
  logger.info(f"finished")
  logger.info(f"wrote mp4 to {art.detections_video_path}")

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

  # sizes relating to data
  parser.add_argument('--image-shape', '--shape', '-s', nargs=3,
                      type=int, default=[100,100,1],
                      help=docs.image_shape)
  parser.add_argument('--data-size', '-N', nargs=1,
                      default=[3000], type=int,
                      help=docs.data_size)
  parser.add_argumnet('--test-size', '-T', nargs=1,
                      default=[0], type=int,
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
  art()

  
if __name__ == "__main__":
  main()
