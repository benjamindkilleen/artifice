"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils.

Data class for feeding data into models, possibly with augmentation.

We expect labels to be of the form:
|-------|-------|--------------------------------------|
| x pos | y pos |        object pose parameters        |
|-------|-------|--------------------------------------|


"""

import logging
import numpy as np
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from artifice import tform, img


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

def _proto_from_image(image):
  image = img.as_float(image)
  feature = {'image' : _bytes_feature(image.tostring()),
             'image_dim0' : _int64_feature(image.shape[0]),
             'image_dim1' : _int64_feature(image.shape[1]),
             'image_dim2' : _int64_feature(image.shape[2])}
  return _serialize_feature(feature)

def _image_from_proto(proto):
  feature_description = {
    'image' : tf.FixedLenFeature([], tf.string),
    'image_dim0' : tf.FixedLenFeature([], tf.int64),
    'image_dim1' : tf.FixedLenFeature([], tf.int64),
    'image_dim2' : tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(proto, feature_description)
  image = tf.decode_raw(features['image'], tf.float32)
  return tf.reshape(image, (features['image_dim0'],
                            features['image_dim1'],
                            features['image_dim2']))

def _proto_from_example(example):
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

def _example_from_proto(proto):
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
  label = tf.reshape(label, (features['label_dim0'], features['label_dim1']))
  return image, label

def load_dataset(record_name, parse_function, num_parallel_calls=None):
  """Load a tfrecord dataset.

  :param record_name: File name(s) to load.
  :param parse_function: function to parse each entry.
  :param num_parallel_calls: passed to map.
  :returns: 
  :rtype: 

  """
  dataset = tf.data.TFRecordDataset(record_name)
  return dataset.map(parse_entry, num_parallel_calls=num_parallel_calls)


def save_dataset(record_name, dataset, serialize_function=None,
                 num_parallel_calls=None):
  """Write a tf.data.Dataset to a file.

  :param record_name:
  :param dataset:
  :param serialize_function: function to serialize examples. If None, assumes
  dataset already serialized.
  :param num_parallel_calls:
  :returns:
  :rtype:

  """
  if serialize_function is not None:
    dataset = dataset.map(serialize_function, num_parallel_calls)
  writer = tf.data.experimental.TFRecordWriter(record_name)
  write_op = writer.write(dataset)
  if not tf.executing_eagerly():
    with tf.Session() as sess:
      sess.run(write_op)

class ArtificeData(object):
  """Abstract class for data wrappers in artifice, which are distinguished by the
  type of examples they hold (unlabeled images, (image, label) pairs (examples),
  etc.).

  Subclasses should implement the parse_function(), serialize_function(), and
  process() functions to complete.

  """
  def __init__(self, dataset, size, image_shape, input_tile_shape=[32,32],
               output_tile_shape=[32,32], batch_size=4, num_parallel_calls=None,
               num_shuffle=10000, label_dim=2, **kwargs):
    """Initialize the data, loading it if necessary..

    kwargs is there only to allow extraneous keyword arguments. It is not used.

    :param dataset: 
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
    self.size = size            # size of an epoch.
    self.image_shape = image_shape
    assert len(self.image_shape) == 3
    self.input_tile_shape = input_tile_shape
    self.output_tile_shape = output_tile_shape
    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls
    self.num_shuffle = num_shuffle
    self.label_dim = label_dim

    if issubclass(type(dataset), tf.data.Dataset):
      self.dataset = dataset
    elif issubclass(type(dataset), ArtificeData):
      self.dataset = dataset.dataset
    elif type(dataset) in [str, list, tuple]:
      self.dataset = load_dataset(dataset, self.parse_function,
                                  num_parallel_calls=self.num_parallel_calls)
    else:
      raise ValueError("unexpected dataset type")

    # derived
    self.num_tiles = int(
      np.ceil(self.image_shape[0] / self.output_tile_shape[0]) *
      np.ceil(self.image_shape[1] / self.output_tile_shape[1]))
    self.prefetch_buffer_size = self.batch_size
    
  @staticmethod
  def parse_function(proto):
    raise NotImplementedError("subclass should implement")

  @staticmethod
  def serialize_function(entry):
    raise NotImplementedError("subclass should implement")
  
  def __len__(self):
    return self.size

  def __iter__(self):
    return self.dataset.__iter__()
  
  def save(self, record_name):
    """Save the dataset to record_name."""
    save_dataset(record_name, self.dataset,
                 serialize_function=self.serialize_function,
                 num_parallel_calls=self.num_parallel_calls)

  def preprocess(self, dataset, training=False):
    """Perform preprocessing steps on dataset.

    Usually just repeats the dataset. Shouldn't change the nesting of tensors at
    all.

    """
    return dataset.repeat(-1)

  def process(self, dataset, training=False):
    """Process the data into tensors ready for input.
    
    The full data processing pipeline is:

    * repeat dataset (in preprocess)
    * augment (if applicable)
    * convert to proxy
    * tile
    * shuffle (in postprocess)
    * batch (in postprocess)

    :param dataset: 
    :param training: if this is for trainign
    :returns: 
    :rtype: 

    """
    raise NotImplementedError("subclasses should implement")

  def postprocess(self, dataset, training=False):
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    if training:
      dataset = dataset.shuffle(self.num_shuffle)
    return dataset.prefetch(self.prefetch_buffer_size)

  @property
  def training_input(self):
    preprocessed = self.preprocess(self.dataset, training=True)
    processed = self.process(preprocessed, training=True)
    return self.postprocess(processed, training=True)

  @property
  def evaluation_input(self):
    preprocessed = self.preprocess(self.dataset, training=False)
    processed = self.process(preprocessed, training=False)
    return self.postprocess(processed, training=False)

  def skip(self, n):
    """Wrapper around tf.data.Dataset.skip."""
    dataset = self.dataset.repeat(-1).skip(n)
    return type(self)(dataset, n, self.image_shape, **vars(self))

  def take(self, n):
    dataset = self.dataset.repeat(-1).skip(n)
    return type(self)(dataset, n, self.image_shape, **vars(self))

  def split(self, splits):
    """Split the dataset into different sets.

    Pass in the number of examples for each split. Returns a new Data object
    with datasets of the corresponding number of examples.

    :param splits: iterable of sizes to take
    :returns: list of Data objects with datasets of the corresponding sizes.
    :rtype: [Data]

    """
    datas = []
    for i, n in enumerate(splits):
      dataset = self.dataset.repeat(-1).skip(sum(splits[:i])).take(n)
      datas.append(type(self)(dataset, n, self.image_shape))
    return datas

  def _sample(self, sampling):
    """Given a `sampling` of the dataset.

    :param sampling: array or tensor of counts for each example. Typically a
    boolean index array, but can specify duplicates for examples. Sampling must
    be at least as large as the dataset.
    :returns: tf dataset of sampled entries

    """
    s = tf.constant(sampling, dtype=tf.int64)
    dataset = self.dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, entry):
      return tf.data.Dataset.from_tensors((idx, entry)).repeat(s[idx])
    return dataset.flat_map(map_func)

  def sample(self, sampling):
    """Sample a dataset with an index array.

    :param sampling: numpy sampling array.
    :returns:
    :rtype:

    """
    dataset = self._sample(sampling)
    dataset = dataset.map(lambda idx, entry : entry,
                          num_parallel_calls=self.num_parallel_calls)
    return type(self)(dataset, np.sum(sampling), self.image_shape, **vars(self))

  def accumulate(self, accumulator, take=None):
    """Runs the accumulators across the dataset.

    An accumulator function should take a `entry` and an `aggregate` object. On
    the first call, `aggregate` will be None. Afterward, each accumulator will
    be passed the output from its previous call as `aggregate`, as well as the
    next entry in the data as 'entry'. On the final call, `entry` will be None,
    allowing for post-processing.

    If executing eagerly, uses the existing session.

    :param accumulator: an accumulator function OR a dictionary mapping names to
      accumulator functions
    :param take: accumulate over at most `take` examples. None (default) or 0
      accumulates over the whole dataset.
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
      for i, entry in enumerate(self.dataset):
        if i == take:
          break
        for k, acc in accumulators.items():
          aggregates[k] = acc(self.as_numpy(entry), aggregates[k])
    else:
      next_entry = self.dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        logger.info("initialized iterator, starting accumulation...")
        for i in itertools.count():
          if i == take:
            break
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
  pass
    
class LabeledData(ArtificeData):

  

  def proxy(self, dataset):
    """Add proxies to the dataset.

    Uses the image shapes from the map, rather than the self variables.

    """
    def map_func(image, label):
      # problem, what if num_objects=0 for this tile?
      positions = tf.cast(label[:, 1:3], tf.float32)

      # indices: (H*W, 2)
      indices = tf.constant(np.array(
        [np.array([i,j]) for i in range(image.shape[0])
         for j in range(image.shape[1])], dtype=np.float32), tf.float32)

      # indices: (M*N, 1, 2), positions: (1, num_objects, 2)
      indices = tf.expand_dims(indices, axis=1)
      positions = tf.expand_dims(positions, axis=0)

      # distances: (M*N, num_objects)
      distances = tf.reduce_min(tf.norm(indices - positions, axis=-1), axis=-1)

      # proxy function: 1 / (d^2 + 1)
      flat = tf.reciprocal(tf.square(distance) + tf.constant(1, tf.float32))
      proxy = tf.reshape(flat, image.shape[:2])

      # problem: don't know how many objects are in each of the tiles, don't
      # know how to match which pixel in the final annotation to its
      # object. But we do want to do tiling before augmentation? Augmentation
      # would go between tiling and proxies, I think. Or maybe it would still go
      # before tiling. But shuffling can go after both? Problem with shuffling
      # tiles is that you may not have objects in every tile? But that's fine.
      return None
      
    return dataset.map(map_func, self.num_parallel_calls)

  def tile(self, dataset):
    """Tile the dataset. Adjust labels accordingly.

    """
    diff0 = self.input_tile_shape[0] - self.output_tile_shape[0]
    diff1 = self.input_tile_shape[1] - self.output_tile_shape[1]
    rem0 = self.image_shape[0] - (self.image_shape[0] % self.output_tile_shape[0])
    rem1 = self.image_shape[1] - (self.image_shape[1] % self.output_tile_shape[1])
    pad_top = int(np.floor(diff0 / 2))
    pad_bottom = int(np.ceil(diff0 / 2)) + rem0
    pad_left = int(np.floor(diff1 / 2))
    pad_right = int(np.ceil(diff1 / 2)) + rem1
    def map_func(image, proxy):
      images = tf.pad(image,
                      [[pad_top, pad_bottom], [pad_left, pad_right], [0,0]],
                      'CONSTANT')
      tiles = []
      proxies = []
      for i in range(0, self.image_shape[0], self.output_tile_shape[0]):
        for j in range(0, self.image_shape[1], self.output_tile_shape[1]):
          tiles.append(image[i:i + self.input_tile_shape[0],
                             j:j + self.input_tile_shape[1]])
          proxies.append(proxy[i:i + self.input_tile_shape[0],
                               j:j + self.input_tile_shape[1]])
          # indices = tf.where(tf.logical_and(
          #   tf.logical_and(labels[:,0] >= i + pad_top,
          #                  labels[:,0] < i + pad_top + self.output_tile_shape[0]),
          #   tf.logical_and(labels[:,1] >= j + pad_left,
          #                  labels[:,1] < j + pad_left + self.output_tile_shape[1])))
          tile_labels.append(labels[b,indices])
      return tf.stack(tiles), tf.stack(tile_labels)
    return dataset.map(map_func, self.num_parallel_calls)
    
  # todo: construct the proxies, not just for distance but also extraneous
  # labels, from the tiled images/labels. This needs a distance threshold to say
  # how far away pixels should include the pose info? Or maybe just have the
  # pose at exactly the pixel containing the object, and weight only those
  # pixels as significant.

  # todo: strip the rest of this file for valuables, get rid of it.
    
class Data(object):
  eps = 0.001

  def __init__(self, data, **kwargs):
    """args:
    :param data: a tf.data.Dataset OR tfrecord file name(s) OR another Data
      subclass. In this case, the other object's _dataset is adopted, allows
      kwargs to be overwritten.
    """

    # kind of required arguments
    self.size = kwargs.get('size', 0)
    self.image_shape = kwargs.get('image_shape', None) # TODO: gather with acc
    assert len(self.image_shape) == 3

    self.num_objects = kwargs.get('num_objects', 2)
    self.tile_shape = kwargs.get('tile_shape', [32, 32, 1])
    self.pad = kwargs.get('pad', 0)
    self.distance_threshold = kwargs.get('distance_threshold', 20.)
    self.batch_size = kwargs.get('batch_size', 1) # in multiples of num_tiles
    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    self.num_shuffle = kwargs.get('num_shuffle', self.size // self.batch_size)
    self.regions = kwargs.get('regions')
    self._kwargs = kwargs

    if issubclass(type(data), Data):
      self._dataset = data._dataset
    elif issubclass(type(data), tf.data.Dataset):
      self._dataset = data
    elif type(data) in [str, list, tuple]:
      # Loading tfrecord files
      self._dataset = load_dataset(
        data, parse_entry=self.parse_entry,
        num_parallel_calls=self.num_parallel_calls)
    else:
      raise ValueError(f"unrecognized data '{data}'")

    self.num_tiles = int(np.ceil(self.image_shape[0] / self.tile_shape[0]) *
                         np.ceil(self.image_shape[1] / self.tile_shape[1]))
    self.prefetch_buffer_size = self.num_tiles * self.batch_size
    self._labels = None
    self.label_shape = (self.num_objects, 4)
    self.labels_shape = (self.batch_size,) + self.label_shape

  @staticmethod
  def parse_entry(*args):
    return example_from_proto(*args)

  @staticmethod
  def encode_entry(*args):
    return proto_from_example(*args)

  def save(self, record_name):
    """Save the dataset to record_name."""
    save_dataset(record_name, self.dataset,
                 encode_entry=self.encode_entry,
                 num_parallel_calls=self.num_parallel_calls)

  def __iter__(self):
    return self._dataset.__iter__()

  def as_numpy(self, entry):
    """Convert `entry` to corresponding tuple of numpy arrys.

    :param entry: tuple of tensors
    :returns: tuple of numpy arrays
    :rtype:

    """
    assert tf.executing_eagerly()
    image, label = entry
    return np.array(image), np.array(label)

  def split(self, *splits, types=None):
    """Split the dataset into different sets.

    Pass in the number of examples for each split. Returns a new Data object
    with datasets of the corresponding number of examples.

    :param types: optional list of Data subclasses to instantiate the splits
    :returns: list of Data objects with datasets of the corresponding sizes.
    :rtype: [Data]

    """

    datas = []
    for i, n in enumerate(splits):
      dataset = self.dataset.skip(sum(splits[:i])).take(n)
      if types is None or i >= len(types):
        datas.append(type(self)(dataset, **self._kwargs))
      elif types[i] is None:
        datas.append(None)
      else:
        datas.append(types[i](dataset, **self._kwargs))

    return datas

  def skip(self, count):
    """Wrapper around tf.data.Dataset.skip."""
    dataset = self.dataset.repeat(-1).skip(count)
    kwargs = self._kwargs.copy()
    kwargs['size'] = count
    return type(self)(dataset, **kwargs)

  def take(self, count):
    dataset = self.dataset.repeat(-1).skip(count)
    kwargs = self._kwargs.copy()
    kwargs['size'] = count
    return type(self)(dataset, **kwargs)

  def _sample(self, sampling):
    """First part of any sampling, just retruns the tf.data.Dataset

    """
    s = tf.constant(sampling, dtype=tf.int64)
    dataset = self.dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, entry):
      return tf.data.Dataset.from_tensors((idx, entry)).repeat(s[idx])
    return dataset.flat_map(map_func)


  def sample(self, sampling):
    """Draw a sampling from the dataset, returning a new dataset of the same type.

    :param sampling: 1-D array indexing the dataset, e.g. a boolean array, with
    possible repetitions.
    :returns: new data object with examples selected by sampling.
    :rtype: a Data subclass, same as self

    """
    dataset = self._sample(sampling)
    dataset = dataset.map(lambda idx, entry : entry,
                          num_parallel_calls=self.num_parallel_calls)
    kwargs = self._kwargs.copy()
    kwargs['size'] = np.sum(sampling)
    return type(self)(dataset, **kwargs)

  def accumulate(self, accumulator):
    """Runs the accumulators across the dataset.

    An accumulator function should take a `entry` and an `aggregate` object. On
    the first call, `aggregate` will be None. Afterward, each accumulator will
    be passed the output from its previous call as `aggregate`, as well as the
    next entry in the data as 'entry'. On the final call, `entry` will be None,
    allowing for post-processing.

    If executing eagerly, uses the existing session.

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
          aggregates[k] = acc(self.as_numpy(entry), aggregates[k])
    else:
      next_entry = self.dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        logger.info("initialized iterator, starting accumulation...")
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

  @staticmethod
  def label_accumulator(entry, labels):
    if labels is None:
      labels = []
    if entry is None:
      return np.array(labels)
    image, label = entry
    labels.append(label)
    return labels

  def accumulate_labels(self):
    return self.accumulate(self.label_accumulator)

  @property
  def labels(self):
    if self._labels is None:
      self._labels = self.accumulate_labels()
    return self._labels

  @property
  def dataset(self):
    return self._dataset

  def preprocess(self, dataset, training=False):
    """Responsible for converting dataset to well-shuffled, repeated (image, label) form.

    Can be overwritten by subclasses to perform augmentation."""
    if training:
      dataset = dataset.shuffle(self.num_shuffle)
    return dataset.repeat(-1).batch(self.batch_size, drop_remainder=True)

  def postprocess(self, dataset, training=False):
    return (dataset.prefetch(self.prefetch_buffer_size))

  def preprocessed(self, training=False):
    return self.preprocess(self.dataset, training=training)

  def to_numpy_field(self, label):
    """Create a distance annotation with numpy `label`.

    :param label: numpy
    :returns:
    :rtype:

    """
    positions = label[:,1:3].astype(np.float32)
    indices = np.array([np.array([i,j])
                        for i in range(self.image_shape[0])
                        for j in range(self.image_shape[1])],
                       dtype=np.float32)
    positions = np.expand_dims(positions, axis=0) # (1,num_objects,2)
    indices = np.expand_dims(indices, axis=1) # (M*N, 1, 2)
    distances = np.min(np.linalg.norm(indices - positions, axis=2), axis=1)
    flat_field = np.reciprocal(np.square(distances + self.eps))
    thresh = self.distance_threshold
    flat_field = np.where(distances < thresh, flat_field, 0.)
    field = np.reshape(flat_field, self.image_shape)
    return field

  def to_field(self, label):
    """Create a tensor distance annotation with tensor `label`.

    Operates on tensors. Use `to_numpy_field` for the numpy version.

    :param label: tensor label for the example with shape
    `(num_objects,label_dim)` OR `(batch_size, num_objects, label_dim)`

    """
    labels = tform.ensure_batched_labels(label)

    # (batch_size, num_objects, 2)
    positions = tf.cast(labels[:,:,1:3], tf.float32)
    # TODO: fix position to INF for objects not present

    # indices: (M*N, 2)
    indices = np.array([np.array([i,j])
                        for i in range(self.image_shape[0])
                        for j in range(self.image_shape[1])],
                       dtype=np.float32)
    indices = tf.constant(indices, dtype=tf.float32)

    # indices: (1, M*N, 1, 2), positions: (batch_size, 1, num_objects, 2)
    indices = tf.expand_dims(indices, axis=0)
    indices = tf.expand_dims(indices, axis=2)
    positions = tf.expand_dims(positions, axis=1)

    # distances: (batch_size, M*N, num_objects)
    distances = tf.reduce_min(tf.norm(indices - positions, axis=3), axis=2)

    # take inverse distance
    eps = tf.constant(self.eps)
    flat_field = tf.square(tf.reciprocal(distances + eps))
 
    # zero the inverse distances outside of threshold
    # logger.debug(f"distance_threshold: {self.distance_threshold}")
    # logger.debug(f"flat_field: {flat_field.shape}, {flat_field}")
    thresh = tf.constant(self.distance_threshold, tf.float32)
    zeros = tf.zeros_like(flat_field)
    flat_field = tf.where(distances < thresh, flat_field, zeros)
    field = tf.reshape(flat_field, [-1,] + self.image_shape)
    field = tform.restore_image_rank(field, rank=len(label.get_shape()) + 1)
    return field

  def from_field(self, field):
    """Recreate the position label associated with field.

    Assigns instance labels in order of detection strength.

    :param field: field array, numpy
    :returns: estimated position label `(num_objects, 3)`

    """
    label = np.zeros((self.num_objects, 3), np.float32)
    if self.regions is None:
      coords = peak_local_max(
        np.squeeze(field), min_distance=self.distance_threshold,
        num_peaks=self.num_objects,
        exclude_border=False)
    else:
      logger.debug("peak_local_max using regions info")
      coords = peak_local_max(
        np.squeeze(field), min_distance=self.distance_threshold / 2,
        exclude_border=False,
        indices=True,
        labels=self.regions,
        num_peaks_per_label=1)
    label[:coords.shape[0],1:3] = coords
    label[:coords.shape[0],0] = np.arange(coords.shape[0])
    return label

  def fielded(self, training=False):
    def map_func(image, label):
      return image, self.to_field(label)
    return (self.preprocessed(training=training)
            .map(map_func, self.num_parallel_calls))

  @property
  def visualized(self):
    def map_func(image, label):
      return image, self.to_field(label), label
    return self.dataset.map(map_func, self.num_parallel_calls)

  def tile(self, images, fields):
    """Tile the batched, fielded dataset."""
    fields = tf.pad(fields, [
      [0,0],
      [0,self.image_shape[0] - (self.image_shape[0] % self.tile_shape[0])],
      [0,self.image_shape[1] - (self.image_shape[1] % self.tile_shape[1])],
      [0,0]], 'CONSTANT')
    images = tf.pad(images, [
      [0,0],
      [self.pad,
       self.pad + self.image_shape[0] - (self.image_shape[0] % self.tile_shape[0])],
      [self.pad,
       self.pad + self.image_shape[1] - (self.image_shape[1] % self.tile_shape[1])],
      [0,0]], 'CONSTANT')
    image_tiles = []
    field_tiles = []
    for b in range(self.batch_size):
      for i in range(0, self.image_shape[0], self.tile_shape[0]):
        for j in range(0, self.image_shape[1], self.tile_shape[1]):
          image_tiles.append(images[
            b, i:i + self.tile_shape[0] + 2*self.pad,
            j:j + self.tile_shape[1] + 2*self.pad])
          field_tiles.append(fields[
            b, i:i + self.tile_shape[0],
            j:j + self.tile_shape[1]])

    out = tf.stack(image_tiles), tf.stack(field_tiles)
    logger.debug(f"tiled: {out}")
    return out

  def tiled(self, training=False):
    def map_func(images, fields):
      return self.tile(images, fields)
    return (self.fielded(training=training)
            .map(map_func, num_parallel_calls=self.num_parallel_calls))

  @property
  def training_input(self):
    return self.postprocess(self.tiled(training=True), training=True)

  @property
  def eval_input(self):
    return self.postprocess(self.tiled(training=False), training=False)

  def untile_single(self, tiles):
    """Untile from `self.num_tiles` numpy tiles, corresponding to single image.

    Does not support batch-ordered inputs

    :param tiles: iterable of `self.num_tiles` numpy tiles each with
    `tile_shape`.
    :returns: an image reconstructed from `tiles`
    :rtype: ndarray

    """
    if self.num_tiles == 1:
      return tiles[0]

    image = np.empty(self.image_shape, dtype=np.float32)
    next_tile = iter(tiles)
    for i in range(0, self.image_shape[0], self.tile_shape[0]):
      if i + self.tile_shape[0] < self.image_shape[0]:
        si = self.tile_shape[0]
      else:
        si = self.image_shape[0] % self.tile_shape[0]
      for j in range(0, self.image_shape[1], self.tile_shape[1]):
        if j + self.tile_shape[1] < self.image_shape[1]:
          sj = self.tile_shape[1]
        else:
          sj = self.image_shape[1] % self.tile_shape[1]
        try:
          tile = next(next_tile)
        except StopIteration:
          break
        image[i:i + si, j:j + sj] = tile[:si,:sj]
    return image

  def untile(self, tiles):
    """Untile the tiles as output in batches

    :param tiles: array of batch-ordered tiles, outer dimension (number of
    tiles) must be a multiple of `batch_size * num_tiles`.
    :returns: array of corresponding images.
    """

    logger.debug(f"tiles: {tiles.shape}, num_tiles: {self.num_tiles}, "
                 f"batch_size: {self.batch_size}")

    assert tiles.shape[0] % (self.batch_size * self.num_tiles) == 0
    num_images = tiles.shape[0] // self.num_tiles
    images = np.zeros([num_images] + self.image_shape, tiles.dtype)
    step = self.num_tiles
    for i in range(images.shape[0]):
      images[i] = self.untile_single(tiles[step*i : step*i + step])
    return images


class UnlabeledData(Data):
  def __init__(self, *args, **kwargs):
    """Hold data with no labels.

    Stores the "mode" background image. Provides a sample_and_annotate method
    that returns a new AugmentationData object with the sampling annotated and
    labeled.

    """
    super().__init__(*args, **kwargs)
    self.background = kwargs.get('background')

    accumulators = {}

  @staticmethod
  def parse_entry(*args):
    return image_from_proto(*args)

  @staticmethod
  def encode_entry(*args):
    return proto_from_image(*args)

  def preprocess(self, dataset, training=True):
    """Responsible for converting dataset to batched `(image, dummy_label)` form.

    :param dataset: the dataset
    :param training: otherwise ignored (maintained for compatibility)

    """
    def map_func(image):
      return image, tf.zeros(self.label_shape, tf.float32, name='dummy')
    dataset = dataset.map(map_func, self.num_parallel_calls)
    dataset = dataset.repeat(-1)
    dataset = dataset.batch(self.batch_size)
    return dataset

  def _sample_and_query(self, sampling, oracle, record_name, query_name,
                        output_type):
    """Helper function to generalize sample_and_{blank}

    :param sampling:
    :param oracle:
    :param record_name:
    :param query_name: name of oracle function to use
    :param output_type: class of output
    :returns:
    :rtype:

    """
    query_function = getattr(oracle, query_name)
    dataset = self._sample(sampling)

    writer = tf.python_io.TFRecordWriter(record_name)
    if tf.executing_eagerly():
      for idx, image in dataset:
        entry = query_function(image, idx)
        # TODO: depends on scenes being parsed
        entry = (np.array(entry[0][0]), np.array(entry[0][1])), np.array(entry[1])
        writer.write(output_type.encode_entry(entry))
    else:
      get_next = dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        while True:
          try:
            idx, image = sess.run(get_next)
            entry = query_function(image, idx)
            writer.write(output_type.encode_entry(entry))
          except tf.errors.OutOfRangeError:
            break
    writer.close()

    kwargs = self._kwargs.copy()
    kwargs['size'] = np.sum(sampling)
    return output_type(record_name, **kwargs)


  def sample_and_label(self, sampling, oracle, record_name):
    """Draw a sampling from the dataset and label each example, and save.

    :param sampling: 1D boolean or "counts" array indexing the dataset.
    :param record_name: tfrecord path to save the newly annotated dataset to.
    :param oracle: an Oracle subclass with annotations
    :returns: new AugmentationData object with scenes for the sampled points
    """
    return self._sample_and_query(sampling, oracle, record_name, 'label', Data)

  def sample_and_annotate(self, sampling, oracle, record_name):
    """Draw a sampling from the dataset and annotate each example, and save

    :param sampling: 1D boolean or "counts" array indexing the dataset.
    :param record_name: tfrecord path to save the newly annotated dataset to.
    :param oracle: an Oracle subclass with annotations
    :returns: new AugmentationData object with scenes for the sampled points
    """
    return self._sample_and_query(sampling, oracle, record_name, 'annotate',
                                  AugmentationData)

  @staticmethod
  def mode_background_accumulator(image, agg):
    """Approximate a running mode of the images.

    Quantizes the image to 256 bins for this mode, makes sense because original
    images are 8-bit. Finalized output will convert this back to a float32
    image in [0,1].

    :param image: either None (last call), or numpy image, ndim == 3
    :param agg: either None (first call), or the histogram of image values
    :returns: new agg or the background ()
    :rtype:

    """
    if agg is None:
      # first call
      assert image is not None
      assert image.ndim == 3
      agg = np.zeros(image.shape + (256,), dtype=np.int64)

    if image is None:
      # last call
      assert agg is not None
      return img.as_float(np.argmax(agg, axis=-1))

    idx = (255. * image).astype(np.int64)
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        for k in range(image.shape[2]):
          agg[i,j,k,idx[i,j,k]] += 1
    return agg


class AugmentationData(Data):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.background = kwargs.get('background')

    accumulators = {}
    if self.background is None:
      accumulators['background'] = AugmentationData.mean_background_accumulator

    aggregates = self.accumulate(accumulators)
    for k,v in aggregates.items():
      setattr(self, k, v)

  @staticmethod
  def parse_entry(*args):
    return scene_from_proto(*args)

  @staticmethod
  def encode_entry(*args):
    return proto_from_scene(*args)

  def as_numpy(self, entry):
    """Convert `entry` to corresponding tuple of numpy arrys.

    :param entry: tuple of tensors
    :returns: tuple of numpy arrays
    :rtype:

    """
    assert tf.executing_eagerly()
    (image, label), annotation = entry
    return (np.array(image), np.array(label)), np.array(annotation)

  def valid(self, label):
    """Determine whether a prospective label is valid.

    Optional function (returns true by default) that may be used to verify
    position.
    
    :param label: the prospective label, numpy (num_objects, >=3)
    :returns: boolean

    """
    return True
    
  def draw(self):
    """Draw a new point.

    Subclasses may re-implement this method.

    :returns: `(n, num_objects, 4)` array of object labels, each containing
    `[obj_id, x, y, theta]`
    :rtype:

    """
    label = np.ones(self.label_shape, dtype=np.float32)
    label[:,1:3] = np.random.uniform([0,0], self.image_shape[:2],
                                      size=(label.shape[0],2))
    label[:,3] = np.random.uniform(0., 2*np.pi, size=label.shape[0])
    return label

  @property
  def label_generator(self):
    """Draw a new, valid point. Wrapper around draw()

    :returns: `(batch_size, num_objects, 4)` array of object labels, each containing
    `[obj_id, x, y, theta]`
    :rtype:

    """
    def gen():
      while True:
        label = self.draw()
        if self.valid(label):
          invalid_count = 0
          yield label
        else:
          invalid_count += 1
          if invalid_count % 10 == 0:
            logger.warning(f"drew {invalid_count} invalid examples")
    return gen

  def augment(self, dataset):
    """Generate the desired labels and then map them over the batched set.

    Dataset should already be repeating indefinitely and batched (if applicable)

    """
    new_labels = tf.data.Dataset.from_generator(
      self.label_generator, tf.float32, tf.TensorShape(self.label_shape))
    new_labels = new_labels.batch(self.batch_size, drop_remainder=True)
    zip_set = tf.data.Dataset.zip((new_labels, dataset))

    background = np.stack([self.background.copy() for _ in range(self.batch_size)])

    def map_func(new_label, scene):
      example, annotation = scene
      image, label = example
      return tform.transform_objects(image, label, annotation, new_label,
                                     num_objects=self.num_objects,
                                     background=background)

    return zip_set.map(map_func, self.num_parallel_calls)

  def preprocess(self, dataset, training=True):
    """Call the augment function.

    :param dataset: the dataset
    :param training: asserted True, maintained for compatibility

    """
    dataset = dataset.repeat(-1).batch(self.batch_size)
    return self.augment(dataset)

  @staticmethod
  def mean_background_accumulator(scene, agg):
    """Take a running average of the pixels where no objects exist.

    Fills pixels with no values at the end of the accumulation with gaussian
    noise.

    """
    if agg is None:
      assert scene is not None
      (image, label), annotation = scene
      background = -np.ones_like(image, dtype=np.float32)
      n = np.zeros_like(background, dtype=np.int64)
    else:
      background, n = agg

    if scene is None:
      return img.fill_negatives(background)

    (image, label), annotation = scene

    # Update the elements with a running average
    bg_indices = np.atleast_3d(np.equal(annotation[:,:,0], 0))
    indices = np.logical_and(background >= 0, bg_indices)
    n[indices] += 1
    background[indices] = (background[indices] +
                           (image[indices] - background[indices]) / n[indices])

    # initialize the new background elements
    indices = np.logical_and(background < 0, bg_indices)
    background[indices] = image[indices]
    n[indices] += 1
    return background, n

  @staticmethod
  def label_accumulator(entry, labels):
    if labels is None:
      labels = []
    if entry is None:
      return np.array(labels)
    image, label = entry
    labels.append(label)
    return labels


def match_detections(detections, labels):
  """Resolve the `detections` with `labels`, matching object ids/order.

  :param detections: `(N, num_objects, >3)` array of detections
  :param labels: `(N, num_objects, >3)` array of labels
  :returns: re-ordered detections, same shape
  :rtype:

  """
  matched_detections = np.empty_like(detections)
  for i in range(detections.shape[0]):
    distances = cdist(detections[i,:,1:3], labels[i,:,1:3])
    row_ind, col_ind = linear_sum_assignment(distances)
    for j in range(detections.shape[1]):
      matched_detections[i,j,0] = labels[i,j,0]
      matched_detections[i,j,1:3] = detections[i,col_ind[j],1:3]
  return matched_detections

# TODO: unify region based classes using multiple inheritance
class RegionBasedUnlabeledData(UnlabeledData):
  def sample_and_annotate(self, sampling, oracle, record_name):
    return self._sample_and_query(sampling, oracle, record_name, 'annotate',
                                  RegionBasedAugmentationData)

class RegionBasedAugmentationData(AugmentationData):
  """Assume that each object is confined to a separate region."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.regions is None:
      raise RuntimeError("RegionBasedAugmentationData requires regions.")
    self.region_indices = img.indices_from_regions(self.regions, self.num_objects)

  def draw(self):
    label = np.ones(self.label_shape, dtype=np.float32)
    label[:,0] = np.arange(1,self.num_objects+1, dtype=np.float32)
    for i in range(self.num_objects):
      obj_id = i + 1
      xs, ys = self.region_indices[obj_id]
      if len(xs) == 0:
        raise RuntimeError(f"object '{obj_id}' has no region")
      idx = np.random.randint(0, len(xs))
      X = np.array([xs[idx], ys[idx]], dtype=np.float32)
      label[i,1:3] = X + np.random.uniform(0, 1, size=2)
    label[:,3] = np.random.uniform(0., 2.*np.pi, size=label.shape[0])
    return label


  
