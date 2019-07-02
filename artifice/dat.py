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

class ArtificeData(object):
  """Abstract class for data wrappers in artifice, which are distinguished by the
  type of examples they hold (unlabeled images, (image, label) pairs (examples),
  etc.).

  Subclasses should implement the process() and serialize() functions to
  complete. Serialize is used for saving the dataset.

  """
  def __init__(self, record_names, *, size, image_shape, input_tile_shape,
               output_tile_shape, batch_size=4, num_parallel_calls=None,
               num_shuffle=10000, cache=False, **kwargs):
    """Initialize the data, loading it if necessary..

    kwargs is there only to allow extraneous keyword arguments. It is not used.

    :param record_names:
    :param size:
    :param image_shape:
    :param input_tile_shape:
    :param output_tile_shape:
    :param batch_size:
    :param num_parallel_calls:
    :param num_shuffle:
    :returns:
    :rtype:

    """
    # inherent
    self.record_names = utils.listwrap(record_names)
    self.size = size            # size of an epoch.
    self.image_shape = image_shape
    assert len(self.image_shape) == 3
    self.input_tile_shape = input_tile_shape
    self.output_tile_shape = output_tile_shape
    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls
    self.num_shuffle = num_shuffle
    self.cache = cache

    # derived
    self.steps_per_epoch = int(self.size // self.batch_size)
    self.num_tiles = self.compute_num_tiles(self.image_shape, self.output_tile_shape)
    self.prefetch_buffer_size = self.batch_size
    self.cache_dir = os.path.join(os.path.dirname(self.record_names[0]), 'cache')

    # private:
    self._preprocess_ops = []    # list of method names and args (as a tuple)

  @property
  def steps(self):
    return int(self.size // self.batch_size)

  @staticmethod
  def compute_num_tiles(image_shape, output_tile_shape):
    return int(np.ceil(image_shape[0] / output_tile_shape[0])*
               np.ceil(image_shape[1] / output_tile_shape[1]))

  @staticmethod
  def serialize(entry):
    raise NotImplementedError("subclass should implement")

  def __len__(self):
    return self.size

  def save(self, record_name):
    """Save the dataset to record_name."""
    save_dataset(record_name, self.dataset, serialize=self.serialize,
                 num_parallel_calls=self.num_parallel_calls)

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

  def process(self, dataset, training):
    """Process the dataset of serialized examples into tensors ready for input.

    The full data processing pipeline is:
    * deserialize example
    * augment (if applicable)
    * convert to proxy
    * tile
    * shuffle (if `training` is True)
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

  def postprocess(self, dataset, training):
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    if training:
      dataset = dataset.shuffle(self.num_shuffle)
    dataset = dataset.repeat(-1).prefetch(self.prefetch_buffer_size)
    if self.cache:
      dataset = dataset.cache(self.cache_dir)
    return dataset

  def get_input(self, training):
    dataset = tf.data.TFRecordDataset(self.record_names)
    dataset = self.process(dataset, training)
    return self.postprocess(dataset, training)

  @property
  def dataset(self):
    return load_dataset(self.record_names, self.parse, self.num_parallel_calls)

  @property
  def training_input(self):
    return self.get_input(True)

  @property
  def evaluation_input(self):
    return self.get_input(False)

class LabeledData(ArtificeData):
  @staticmethod
  def serialize(entry):
    return proto_from_example(entry)

  @staticmethod
  def parse(proto):
    return example_from_proto(proto)

  def make_proxy(self, image, label):
    """Map function for converting an (image,label) pair to (image, proxy).

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

  def process(self, dataset, training):
    def map_func(proto):
      image, label = self.parse(proto)
      if not training:
        return self.tile_image_label(image, label)
      proxy = self.make_proxy(image, label)
      # returns a dataset, not a nested tensor
      return self.tile_image_proxy(image, proxy)
    return dataset.interleave(map_func, cycle_length=self.num_parallel_calls,
                              block_length=self.num_tiles,
                              num_parallel_calls=self.num_parallel_calls)

class UnlabeledData(ArtificeData):
  pass


def evaluate_proxy(label, proxy, distance_threshold=5):
  """Evaluage the proxy against the label and return an array of absolute errors.

  :param label:
  :param proxy:
  :param distance_threshold:
  :returns:
  :rtype:

  """
  # todo: make this function robust to num peaks detected, whether an object was
  # actually found, etc. Can signify with -1,-1 position that object was not
  # found. 
  peaks = img.detect_peaks(proxy[:,:,0]) # [num_peaks, 2]
  error = np.empty((label.shape[0], label.shape[1] - 1)) # [num_objects,1+pose_dim]
  num_failed = 0
  for i in range(label.shape[0]):
    distances = np.linalg.norm(peaks - label[i:i+1, :2], axis=1)
    if np.min(distances) >= distance_threshold:
      num_failed += 1
      error[i] = 0
      continue
    pidx = np.argmin(distances)
    peak = peaks[pidx].copy()
    peaks[pidx] = np.inf
    error[i, 0] = np.linalg.norm(peak - label[i, :2])
    for j in range(1, error.shape[1]):
      error[i, j] = abs(proxy[int(peak[0]), int(peak[1]), j+1] - label[i, j+1])
  return error, num_failed

