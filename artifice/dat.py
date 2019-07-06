"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils.

Data class for feeding data into models, possibly with augmentation.

We expect labels to be of the form:
|-------|-------|--------------------------------------|
| x pos | y pos |        object pose parameters        |
|-------|-------|--------------------------------------|


"""

import os
import logging
from glob import glob
import numpy as np
import tensorflow as tf
from artifice import img, utils


logger = logging.getLogger('artifice')

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _serialize_feature(feature):
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

#################### Begin parsing/serializing functions ####################

def proto_from_image(image):
  image = img.as_float(image)
  feature = {'image' : _bytes_feature(image.tostring()),
             'image_dim0' : _int64_feature(image.shape[0]),
             'image_dim1' : _int64_feature(image.shape[1]),
             'image_dim2' : _int64_feature(image.shape[2])}
  return _serialize_feature(feature)

def image_from_proto(proto):
  feature_description = {
    'image' : tf.FixedLenFeature([], tf.string),
    'image_dim0' : tf.FixedLenFeature([], tf.int64),
    'image_dim1' : tf.FixedLenFeature([], tf.int64),
    'image_dim2' : tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(proto, feature_description)
  image = tf.decode_raw(features['image'], tf.float32)
  return tf.reshape(image, [features['image_dim0'],
                            features['image_dim1'],
                            features['image_dim2']])

def proto_from_example(example):
  image, label = example
  image = img.as_float(image)
  label = label.astype(np.float32)
  feature = {'image' : _bytes_feature(image.tostring()),
             'image_dim0' : _int64_feature(image.shape[0]),
             'image_dim1' : _int64_feature(image.shape[1]),
             'image_dim2' : _int64_feature(image.shape[2]),
             'label' : _bytes_feature(label.tostring()),
             'label_dim0' : _int64_feature(label.shape[0]),
             'label_dim1' : _int64_feature(label.shape[1])}
  return _serialize_feature(feature)

def example_from_proto(proto):
  feature_description = {
    'image' : tf.FixedLenFeature([], tf.string),
    'image_dim0' : tf.FixedLenFeature([], tf.int64),
    'image_dim1' : tf.FixedLenFeature([], tf.int64),
    'image_dim2' : tf.FixedLenFeature([], tf.int64),
    'label' : tf.FixedLenFeature([], tf.string),
    'label_dim0' : tf.FixedLenFeature([], tf.int64),
    'label_dim1' : tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(proto, feature_description)
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.reshape(image, (features['image_dim0'],
                             features['image_dim1'],
                             features['image_dim2']))
  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, [features['label_dim0'], features['label_dim1']])
  return image, label

def proto_from_annotated_example(example):
  image, label, annotation = example
  image = img.as_float(image)
  label = label.astype(np.float32)
  annotation = img.as_float(annotation)
  feature = {'image' : _bytes_feature(image.tostring()),
             'image_dim0' : _int64_feature(image.shape[0]),
             'image_dim1' : _int64_feature(image.shape[1]),
             'image_dim2' : _int64_feature(image.shape[2]),
             'label' : _bytes_feature(label.tostring()),
             'label_dim0' : _int64_feature(label.shape[0]),
             'label_dim1' : _int64_feature(label.shape[1]),
             'annotation' : _bytes_feature(annotation.tostring()),
             'annotation_dim0' : _int64_feature(annotation.shape[0]),
             'annotation_dim1' : _int64_feature(annotation.shape[1]),
             'annotation_dim2' : _int64_feature(annotation.shape[2])}
  return _serialize_feature(feature)

def annotated_example_from_proto(proto):
  feature_description = {
    'image' : tf.FixedLenFeature([], tf.string),
    'image_dim0' : tf.FixedLenFeature([], tf.int64),
    'image_dim1' : tf.FixedLenFeature([], tf.int64),
    'image_dim2' : tf.FixedLenFeature([], tf.int64),
    'label' : tf.FixedLenFeature([], tf.string),
    'label_dim0' : tf.FixedLenFeature([], tf.int64),
    'label_dim1' : tf.FixedLenFeature([], tf.int64),
    'annotation' : tf.FixedLenFeature([], tf.string),
    'annotation_dim0' : tf.FixedLenFeature([], tf.int64),
    'annotation_dim1' : tf.FixedLenFeature([], tf.int64),
    'annotation_dim2' : tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(proto, feature_description)
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.reshape(image, (features['image_dim0'],
                             features['image_dim1'],
                             features['image_dim2']))
  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, [features['label_dim0'], features['label_dim1']])
  annotation = tf.decode_raw(features['annotation'], tf.float32)
  annotation = tf.reshape(annotation, (features['annotation_dim0'],
                                       features['annotation_dim1'],
                                       features['annotation_dim2']))
  return image, label, annotation

#################### loading and saving tf datasets ####################

def load_dataset(record_name, parse, num_parallel_calls=None):
  """Load a tfrecord dataset.

  :param record_name: File name(s) to load.
  :param parse_function: function to parse each entry.
  :param num_parallel_calls: passed to map.
  :returns:
  :rtype:

  """
  dataset = tf.data.TFRecordDataset(record_name)
  return dataset.map(parse, num_parallel_calls=num_parallel_calls)


def save_dataset(record_name, dataset, serialize=None,
                 num_parallel_calls=None):
  """Write a tf.data.Dataset to a file.

  :param record_name:
  :param dataset:
  :param serialize: function to serialize examples. If None, assumes
  dataset already serialized.
  :param num_parallel_calls: only used if serialize() is not None.
  :returns:
  :rtype:

  """
  if serialize is not None:
    dataset = dataset.map(serialize, num_parallel_calls)
  writer = tf.data.experimental.TFRecordWriter(record_name)
  write_op = writer.write(dataset)
  if not tf.executing_eagerly():
    with tf.Session() as sess:
      sess.run(write_op)

#################### ArtificeData classes ####################
      
class ArtificeData(object):
  """Abstract class for data wrappers in artifice, which are distinguished by the
  type of examples they hold (unlabeled images, (image, label) pairs (examples),
  etc.).

  Subclasses should implement the process() and serialize() functions to
  complete. Serialize is used for saving the dataset.

  """

  # dataset modes:
  TRAINING = "TRAINING"         # `(image, proxy)` tensor tuple
  PREDICTION = "PREDICTION"     # single `image` tensor
  EVALUATION = "EVALUATION"     # `(image, label)` tensor tuple
  ENUMERATED_PREDICTION = "ENUMERATED_PREDICTION"
  AUGMENTED_TRAINING = "AUGMENTED_TRAINING"
  AUGMENTED_PREDICTION = "AUGMENTED_PREDICTION"
  AUGMENTED_EVALUATION = "AUGMENTED_EVALUATION"
  
  def __init__(self, record_path, *, size, image_shape, input_tile_shape,
               output_tile_shape, batch_size, num_parallel_calls=None,
               num_shuffle=10000, cache_dir=None, **kwargs):
    """Initialize the data, loading it if necessary..

    kwargs is there only to allow extraneous keyword arguments. It is not used.

    :param record_paths: path or paths containing tfrecord files. If a
    directory, then grabs all .tfrecord files in that directory *at runtime*.
    :param size: 
    :param image_shape: 
    :param input_tile_shape: 
    :param output_tile_shape: 
    :param batch_size: 
    :param num_parallel_calls: 
    :param num_shuffle: 
    :param cache_dir: 
    :returns: 
    :rtype: 

    """
    # inherent
    self.record_paths = utils.listwrap(record_path)
    self.size = size            # size of an epoch.
    self.image_shape = image_shape
    assert len(self.image_shape) == 3
    self.input_tile_shape = input_tile_shape
    self.output_tile_shape = output_tile_shape
    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls
    self.num_shuffle = num_shuffle
    self.cache_dir = cache_dir

    # derived
    self.num_tiles = self.compute_num_tiles(self.image_shape,
                                            self.output_tile_shape)
    self.prefetch_buffer_size = self.batch_size

    # private:
    self._preprocess_ops = []    # list of method names and args (as a tuple)

  @property
  def record_names(self):
    record_names = []
    for path in self.record_paths:
      if os.path.isdir(path):
        record_names += glob(os.path.join(path, "*.tfrecord"))
      else:
        record_names.append(path)
    return record_names

  @staticmethod
  def serialize(entry):
    raise NotImplementedError("subclass should implement")

  def process(self, dataset, mode):
    """Process the dataset of serialized examples into tensors ready for input.

    todo: update this documentation for modes.

    The full data processing pipeline is:
    * deserialize example
    * augment (if applicable)
    * convert to proxy
    * tile
    * shuffle (if mode is TRAINING)
    * repeat
    * batch

    process() performs the first four of these functions, and it should do so
    in a single call to map() or interleave(), for efficiency's
    sake. postprocess takes care of the last 3.

    If training is False, then return tiles paired with (possibly duplicated)
    labels for the corresponding full images.

    :param dataset:
    :param training: if this is for trainign
    :returns:
    :rtype:

    """
    raise NotImplementedError("subclasses should implement")

  def postprocess(self, dataset, mode):
    if "ENUMERATED" in mode:
      dataset = dataset.enumerate()
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    if "TRAINING" in mode:
      dataset = dataset.shuffle(self.num_shuffle)
    dataset = dataset.repeat(-1).prefetch(self.prefetch_buffer_size)
    if self.cache_dir is not None:
      dataset = dataset.cache(self.cache_dir)
    return dataset

  def get_input(self, mode):
    dataset = tf.data.TFRecordDataset(self.record_names)
    dataset = self.process(dataset, mode)
    return self.postprocess(dataset, mode)

  @property
  def training_input(self):
    return self.get_input(ArtificeData.TRAINING)
  @property
  def prediction_input(self):
    return self.get_input(ArtificeData.PREDICTION)
  @property
  def evaluation_input(self):
    return self.get_input(ArtificeData.EVALUATION)
  @property
  def enumerated_prediction_input(self):
    return self.get_input(ArtificeData.ENUMERATED_PREDICTION)
  @property
  def augmented_training_input(self):
    return self.get_input(ArtificeData.AUGMENTED_TRAINING)
  @property
  def augmented_prediction_input(self):
    return self.get_input(ArtificeData.AUGMENTED_PREDICTION)
  @property
  def augmented_evaluation_input(self):
    return self.get_input(ArtificeData.AUGMENTED_EVALUATION)

  @property
  def dataset(self):
    return load_dataset(self.record_names, self.parse, self.num_parallel_calls)
  
  @property
  def steps_per_epoch(self):
    return int(self.size // self.batch_size)

  @staticmethod
  def compute_num_tiles(image_shape, output_tile_shape):
    return int(np.ceil(image_shape[0] / output_tile_shape[0])*
               np.ceil(image_shape[1] / output_tile_shape[1]))

  def __len__(self):
    return self.size

  def save(self, record_name):
    """Save the dataset to record_name."""
    save_dataset(record_name, self.dataset, serialize=self.serialize,
                 num_parallel_calls=self.num_parallel_calls)

  def get_entry(self, i):
    """Get the i'th entry of the original dataset, in numpy form."""
    if tf.executing_eagerly():
      entry = next(self.dataset.skip(i).take(1))
    else:
      raise NotImplementedError
    return entry
    
  #################### Generic functions for proxies/tiling ####################
    
  def make_proxy(self, image, label):
    """Function for converting an (image,label) pair to (image, proxy).

    Note that this operates on the untiled images, so label is expected to have
    shape `num_objects`.

    """
    positions = tf.cast(label[:, :2], tf.float32)
    indices = tf.constant(np.array( # [H*W,2]
      [np.array([i,j]) for i in range(self.image_shape[0])
       for j in range(self.image_shape[1])], dtype=np.float32), tf.float32)
    indices = tf.expand_dims(indices, axis=1)                # [H*W,1,2]
    positions = tf.expand_dims(positions, axis=0)            # [1,num_objects,2]
    object_distances = tf.norm(indices - positions, axis=-1) # [H*W,num_objects]

    # make pose_maps
    pose = label[:,2:]                             # [num_objects,pose_dim]
    regions = tf.expand_dims(tf.argmin(object_distances, axis=-1), axis=-1) # [H*W,1]
    pose_maps = tf.reshape(
      tf.gather_nd(pose, regions),
      [self.image_shape[0], self.image_shape[1], -1]) # [H,W,pose_dim]

    # make distance proxy function: 1 / (d^2 + 1)
    distances = tf.reduce_min(object_distances, axis=-1)  # [H*W,]
    flat = tf.reciprocal(tf.square(distances) + tf.constant(1, tf.float32))
    proxy = tf.reshape(flat, [self.image_shape[0], self.image_shape[1], 1]) # [H,W,1]
    return tf.concat([proxy, pose_maps], axis=-1, name='proxy_step')

  @property
  def image_padding(self):
    diff0 = self.input_tile_shape[0] - self.output_tile_shape[0]
    diff1 = self.input_tile_shape[1] - self.output_tile_shape[1]
    rem0 = self.image_shape[0] - (self.image_shape[0] % self.output_tile_shape[0])
    rem1 = self.image_shape[1] - (self.image_shape[1] % self.output_tile_shape[1])
    pad_top = int(np.floor(diff0 / 2))
    pad_bottom = int(np.ceil(diff0 / 2)) + rem0
    pad_left = int(np.floor(diff1 / 2))
    pad_right = int(np.ceil(diff1 / 2)) + rem1
    return [[pad_top, pad_bottom], [pad_left, pad_right], [0,0]]

  @property
  def proxy_padding(self):
    rem0 = self.image_shape[0] - (self.image_shape[0] % self.output_tile_shape[0])
    rem1 = self.image_shape[1] - (self.image_shape[1] % self.output_tile_shape[1])
    return [[0, rem0], [0, rem1], [0,0]]

  def tile_image(self, image):
    image = tf.pad(image, self.image_padding, 'CONSTANT')
    tiles = []
    for i in range(0, self.image_shape[0], self.output_tile_shape[0]):
      for j in range(0, self.image_shape[1], self.output_tile_shape[1]):
        tiles.append(image[i:i + self.input_tile_shape[0],
                           j:j + self.input_tile_shape[1]])
    return tf.data.Dataset.from_tensor_slices(tiles)
  
  def tile_image_proxy(self, image, proxy):
    image = tf.pad(image, self.image_padding, 'CONSTANT')
    proxy = tf.pad(proxy, self.proxy_padding, 'CONSTANT')
    tiles = []
    proxies = []
    for i in range(0, self.image_shape[0], self.output_tile_shape[0]):
      for j in range(0, self.image_shape[1], self.output_tile_shape[1]):
        tiles.append(image[i:i + self.input_tile_shape[0],
                           j:j + self.input_tile_shape[1]])
        proxies.append(proxy[i:i + self.output_tile_shape[0],
                             j:j + self.output_tile_shape[1]])
    return tf.data.Dataset.from_tensor_slices((tiles, proxies))

  def tile_image_label(self, image, label):
    """Tile the images, copy the full image label to each tile."""
    image = tf.pad(image, self.image_padding, 'CONSTANT')
    tiles = []
    tile_labels = []
    for i in range(0, self.image_shape[0], self.output_tile_shape[0]):
      for j in range(0, self.image_shape[1], self.output_tile_shape[1]):
        tiles.append(image[i:i + self.input_tile_shape[0],
                           j:j + self.input_tile_shape[1]])
        tile_labels.append(tf.identity(label))
    return tf.data.Dataset.from_tensor_slices((tiles, tile_labels))

  def untile(self, tiles):
    """Untile num_tiles tiles into a single "image".

    `tiles` must contain the correct number of tiles.

    :param tiles: `num_tiles` length list of 3D arrays or tiles.
    :returns: reconstructed image.
    :rtype:

    """
    if len(tiles) != self.num_tiles:
      raise RuntimeError("Ensure tiles is same length as num_tiles.")
    if self.num_tiles == 1:
      return tiles[0]

    image = np.empty(self.image_shape, dtype=np.float32)
    tiles = list(tiles)
    for i in range(0, self.image_shape[0], self.output_tile_shape[0]):
      if i + self.output_tile_shape[0] < self.image_shape[0]:
        si = self.output_tile_shape[0]
      else:
        si = self.image_shape[0] % self.tile_shape[0]
      for j in range(0, self.image_shape[1], self.output_tile_shape[1]):
        if j + self.output_tile_shape[1] < self.image_shape[1]:
          sj = self.output_tile_shape[1]
        else:
          sj = self.image_shape[1] % self.output_tile_shape[1]
        image[i:i + si, j:j + sj] = tiles[i][:si,:sj]
    return image

  #################### End proxy/tiling functions ####################

  def accumulate(self, accumulator):
    """Runs the accumulators across the dataset.
      
    An accumulator function should take a `entry` and an `aggregate` object. On
    the first call, `aggregate` will be None. Afterward, each accumulator will
    be passed the output from its previous call as `aggregate`, as well as the
    next entry in the data as 'entry'. On the final call, `entry` will be None,
    allowing for post-processing.

    :param accumulator: an accumulator function OR a dictionary mapping names to
    accumulator functions
    :returns: aggregate from `accumulator` OR a dictionary of aggregates with
    the same keys as `accumulators`.
    :rtype: dict
    """
    if type(accumulator) == dict:
      accumulators = accumulator
    else:
      accumulators = {0 : accumulator}
    aggregates = dict.fromkeys(accumulators.keys())
    if tf.executing_eagerly():
      for entry in self.dataset:
        for k, acc in accumulators.items():
          aggregates[k] = acc(entry.numpy(), aggregates[k]) # todo: .numpy()?
          # if above doesn't work, use tf.nest.map_structure()
    else:
      next_entry = self.dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        while True:
          try:
            entry = sess.run(next_entry)
            for k, acc in accumulators.items():
              aggregates[k] = acc(entry, aggregates[k])
          except tf.errors.OutOfRangeError:
            break

    logger.info("finished accumulation")
    for k, acc in accumulators.items():
      aggregates[k] = acc(None, aggregates[k])
    if type(accumulator) == dict:
      return aggregates
    else:
      return aggregates[0]
    
class UnlabeledData(ArtificeData):
  @staticmethod
  def serialize(entry):
    return proto_from_image(entry)

  @staticmethod
  def parse(proto):
    return image_from_proto(proto)

  def process(self, dataset, mode):
    def map_func(proto):
      image = self.parse(proto)
      if mode == ArtificeData.PREDICTION:
        return self.tile_image(image)
      raise ValueError(f"{mode} mode invalid for UnlabeledData")
    return dataset.interleave(map_func, cycle_length=self.num_parallel_calls,
                              block_length=self.num_tiles,
                              num_parallel_calls=self.num_parallel_calls)

class LabeledData(ArtificeData):
  @staticmethod
  def serialize(entry):
    return proto_from_example(entry)

  @staticmethod
  def parse(proto):
    return example_from_proto(proto)

  @staticmethod
  def label_accumulator(entry, labels):
    if labels is None:
      labels = []
    if entry is None:
      return np.array(labels)
    labels.append(entry[1])
    return labels

  def get_labels(self):
    return self.accumulate(self.label_accumulator)

  def process(self, dataset, mode):
    def map_func(proto):
      image, label = self.parse(proto)
      if mode == ArtificeData.PREDICTION:
        return self.tile_image(image)
      if mode == ArtificeData.EVALUATION:
        return self.tile_image_label(image, label)
      if mode == ArtificeData.TRAINING:
        proxy = self.make_proxy(image, label)
        return self.tile_image_proxy(image, proxy)
      raise ValueError(f"{mode} mode invalid for LabeledData")
    return dataset.interleave(map_func, cycle_length=self.num_parallel_calls,
                              block_length=self.num_tiles,
                              num_parallel_calls=self.num_parallel_calls)

class AnnotatedData(LabeledData):
  """Class for annotated data, which can be augmented. Annotated data consists of
  an `(image, label, annotation)` tuple, which we call an "annotated example".

  AnnotatedData inherits from LabeledData, since get_labels is a perfectly legal
  operation on `(image, label, annotation)` tuples as written.

  This data can technically be used to regular prediction, evaluation, and
  training modes, but note that in this case, unless the number of examples is
  known precisely, then some examples may be repeated.

  Annotations are essentially instance segmentations with object ids given by
  that object's index in the label. Background pixels should be filled with
  -1. (This off-by-one from convention, where the background is given ID 0 and
  objects are > 0).

  """
  def __init__(self, *args, transformation=None, identity_prob=0.01, **kwargs):
    """FIXME! briefly describe function

    :param transformation: a single or list of transformations that are applied
    during augmentation. If multiple, then each augmented example has a randomly
    selected transformation applied to it.
    :param identity_prob: probability that no augmentations are applied to an example.
    :returns: 
    :rtype: 

    """
    self.transformation = transformation
    self.identity_prob = identity_prob
  
  @staticmethod
  def serialize(entry):
    return proto_from_annotated_example(entry)

  @staticmethod
  def parse(proto):
    return annotated_example_from_proto(proto)
  
  def process(self, dataset, mode):
    def map_func(proto):
      image, label, annotation = self.parse(proto)
      if mode == ArtificeData.PREDICTION:
        return self.tile_image(image)
      if mode == ArtificeData.EVALUATION:
        return self.tile_image_label(image, label)
      if mode == ArtificeData.TRAINING:
        proxy = self.make_proxy(image, label)
        return self.tile_image_proxy(image, proxy)

      # remaining modes require augmentation
      image, label = self.augment(image, label, annotation)
      if mode == ArtificeData.AUGMENTED_PREDICTION:
        return self.tile_image(image)
      if mode == ArtificeData.AUGMENTED_EVALUATION:
        return self.tile_image_label(image, label)
      if mode == ArtificeData.AUGMENTED_TRAINING:
        proxy = self.make_proxy(image, albel)
        return self.tile_image_proxy(image, proxy)
      raise ValueError(f"{mode} mode invalid for AnnotatedData")
    return dataset.interleave(map_func, cycle_length=self.num_parallel_calls,
                              block_length=self.num_tiles,
                              num_parallel_calls=self.num_parallel_calls)

  def augment(self, image, label, annotation):
    """Augment an example using self.transformation.

    A transformation takes in an image, a label, and an annotation and returns
    an `(image, label)` pair. If more than one transformation is listed, then a
    random one is selected on each call. If the pose dimensions of the label are
    affected by the transformation, then it should know how to deal with those
    as well.

    Of course, some transformations may require additional information. This
    could be encoded in the annotation, which could be a nested tensor if
    handled correctly.

    Artifice includes several builtin transformations, all of which are in the
    `tform` module. For now, only one of these may be selected, but the function
    in question could randomly apply different transformations within its body.

    :param image: 
    :param label: 
    :param annotation: 
    :returns: new `(image, label)` example

    """
    if self.transformation is None:
      return image, label
    return tf.case(
      tf.cast(tf.less(tf.random.uniform([], 0, 1, tf.float32),
                      tf.constant(self.identity_prob, tf.float32)),
              tf.int64),
      [(lambda : self.transformation(image, label)),
       (lambda : (image, label))]
    )

#################### Independant data processing functions ####################
  
def evaluate_proxy(label, proxy, distance_threshold=10):
  """Evaluage the proxy against the label and return an array of absolute errors.

  The error array has the same ordering as objects in the label. Negative values
  indicate that object was not detected.

  :param label:
  :param proxy:
  :param distance_threshold:
  :returns:
  :rtype:

  """
  peaks = img.detect_peaks(proxy[:,:,0]) # [num_peaks, 2]
  error = np.empty((label.shape[0], label.shape[1] - 1)) # [num_objects,1+pose_dim]
  num_failed = 0
  for i in range(label.shape[0]):
    distances = np.linalg.norm(peaks - label[i:i+1, :2], axis=1)
    if np.min(distances) >= distance_threshold:
      num_failed += 1
      error[i] = -1
      continue
    pidx = np.argmin(distances)
    peak = peaks[pidx].copy()
    peaks[pidx] = np.inf
    error[i, 0] = np.linalg.norm(peak - label[i, :2])
    for j in range(1, error.shape[1]):
      error[i, j] = abs(proxy[int(peak[0]), int(peak[1]), j] - label[i, j+1])
  return error, num_failed


