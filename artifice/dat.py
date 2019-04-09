"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils.

Data class for feeding data into models, possibly with augmentation.

"""

import numpy as np
import tensorflow as tf
from artifice import tform, oracles
from artifice.utils import img
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import logging


logger = logging.getLogger('artifice')


def as_float(image):
  """Return image as a grayscale float32 array at least 3d."""
  if image.dtype in [np.float32, np.float64]:
    image = image.astype(np.float32)
  elif image.dtype in [np.uint8, np.int32, np.int64]:
    image = image.astype(np.float32) / 255.
  else:
    raise ValueError(f"image dtype '{image.dtype}' not allowed")
  return np.atleast_3d(image)

def _bytes_feature(value):
  # Helper function for writing a string to a tfrecord
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  # Helper function for writing an array to a tfrecord
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def proto_from_image(image):
  """Create a tf example proto from an image.

  :param image: single numpy array
  :returns: 
  :rtype: 

  """
  image = as_float(image)
  image_string = image.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)
  feature = {"image" : _bytes_feature(image_string),
             "image_shape" : _int64_feature(image_shape)}
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()


def image_from_proto(proto):
  """Parse `proto` into tensor `image`."""
  features = tf.parse_single_example(
    proto,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64)
    })
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.reshape(image, features['image_shape'])
  return image


def proto_from_example(example):
  """Creates a tf example proto from an (image, label) pair.

  :param example: (image, label) pair
  :returns: proto string
  :rtype: str

  """
  image, label = example
  image = as_float(image)
  label = label.astype(np.float32)
  image_string = image.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)
  label_string = label.tostring()
  label_shape = np.array(label.shape, dtype=np.int64)
  feature = {"image" : _bytes_feature(image_string),
             "image_shape" : _int64_feature(image_shape),
             "label" : _bytes_feature(label_string),
             "label_shape" : _int64_feature(label_shape)}
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()

def example_from_proto(proto):
  """Parse `proto` into tensors `(image, label)`.

  """
  features = tf.parse_single_example(
    proto,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64),
      'label' : tf.FixedLenFeature([], tf.string),
      'label_shape' : tf.FixedLenFeature([2], tf.int64)
    })

  # decode strings
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.reshape(image, features['image_shape'])

  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, features['label_shape'],
                     name='reshape_label_proto')

  return (image, label)


def proto_from_scene(scene):
  """Creates a tf example from the scene, which contains an (image,label) example

  :param scene: 

  """
  example, annotation = scene
  image, label = example
  image = as_float(image)
  label = label.astype(np.float32)
  annotation = annotation.astype(np.float32)

  image_string = image.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)

  annotation_string = annotation.tostring()
  annotation_shape = np.array(annotation.shape, dtype=np.int64)

  label_string = label.tostring()
  label_shape = np.array(label.shape, dtype=np.int64)

  feature = {"image" : _bytes_feature(image_string),
             "image_shape" : _int64_feature(image_shape),
             "annotation" : _bytes_feature(annotation_string),
             "annotation_shape" : _int64_feature(annotation_shape),
             "label" : _bytes_feature(label_string),
             "label_shape" : _int64_feature(label_shape)}

  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()


def scene_from_proto(proto):
  """Parse `proto` into tensors `(image, label), annotation`.

  """
  features = tf.parse_single_example(
    proto,
    # Defaults are not specified since both keys are required.
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64),
      'annotation': tf.FixedLenFeature([], tf.string),
      'annotation_shape': tf.FixedLenFeature([3], tf.int64),
      'label' : tf.FixedLenFeature([], tf.string),
      'label_shape' : tf.FixedLenFeature([2], tf.int64)
    })

  # decode strings
  image = tf.decode_raw(features['image'], tf.float32)
  image = tf.reshape(image, features['image_shape'])

  annotation = tf.decode_raw(features['annotation'], tf.float32)
  annotation = tf.reshape(annotation, features['annotation_shape'],
                          name='reshape_annotation_proto')

  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, features['label_shape'],
                     name='reshape_label_proto')

  return (image, label), annotation


def load_dataset(record_name,
                 parse_entry=example_from_proto,
                 num_parallel_calls=None):
  """Load the record_name as a tf.data.Dataset"""
  dataset = tf.data.TFRecordDataset(record_name)
  return dataset.map(parse_entry, num_parallel_calls=num_parallel_calls)


def save_dataset(record_name, dataset,
                 encode_entry=proto_from_example,
                 num_parallel_calls=None):
  """Save the dataset in tfrecord format

  :param record_name: filename to save to
  :param dataset: tf.data.Dataset to save
  :param proto_from_example: 
  :param num_parallel_calls: 
  :returns: 
  :rtype: 

  """
  next_example = dataset.make_one_shot_iterator().get_next()

  logger.info(f"writing dataset to {record_name}...")
  writer = tf.python_io.TFRecordWriter(record_name)
  with tf.Session() as sess:
    i = 0
    while True:
      try:
        example = sess.run(next_example)
        writer.write(encode_entry(example))
      except tf.errors.OutOfRangeError:
        break
      i += 1


  writer.close()
  logger.info(f"wrote {i} examples")

  
class Data(object):
  """Wrapper around tf.data.Dataset of examples.

  Chiefly useful for feeding into models in various forms. Different properties
  should be created for each kind of feed.

  Subclasses of the Data object should be created for different types of
  datasets. The self._dataset object should never be altered.

  """

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
    assert(len(self.image_shape) == 3)

    self.num_objects = kwargs.get('num_objects', 2)
    self.tile_shape = kwargs.get('tile_shape', [32, 32, 1])
    self.pad = kwargs.get('pad', 0)
    self.distance_threshold = kwargs.get('distance_threshold', 20.)
    self.batch_size = kwargs.get('batch_size', 1) # in multiples of num_tiles
    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    self.parse_entry = kwargs.get('parse_entry', example_from_proto)
    self.encode_entry = kwargs.get('encode_entry', proto_from_example)
    self.num_shuffle = kwargs.get('num_shuffle', self.size // self.batch_size)
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
  
  def sample(self, sampling):
    """Draw a sampling from the dataset, returning a new dataset of the same type.

    :param sampling: 1-D array indexing the dataset, e.g. a boolean array, with
    possible repetitions.
    :returns: new data object with examples selected by sampling.
    :rtype: a Data subclass, same as self

    """
    s = tf.constant(sampling, dtype=tf.int64)
    dataset = self.dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, entry):
      return tf.data.Dataset.from_tensors(entry).repeat(s[idx])
    dataset = dataset.flat_map(map_func)

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
      iterator = self.dataset.make_initializable_iterator()
      next_item = iterator.get_next()
      with tf.Session() as sess:
        sess.run(iterator.initializer)
        logger.info("initialized iterator, starting accumulation...")
        while True:
          try:
            entry = sess.run(next_item)
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

  def preprocess(self, dataset):
    """Responsible for converting dataset to (image, label) form.

    Can be overwritten by subclasses to perform augmentation."""
    return dataset.batch(self.batch_size, drop_remainder=True)

  @property
  def preprocessed(self):
    return self.preprocess(self.dataset)


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
    distances = np.linalg.norm(indices - positions, axis=2)
    flat_field = np.reciprocal(np.min(distances, axis=1) + self.eps)
    inv_thresh = 1. / self.distance_threshold
    flat_field = np.where(flat_field > inv_thresh, flat_field, 0.)
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
    distances = tf.norm(indices - positions, axis=3)
    
    # take inverse distance
    eps = tf.constant(self.eps)
    flat_field = tf.reciprocal(tf.reduce_min(distances, axis=2) + eps)

    # zero the inverse distances outside of threshold
    inv_thresh = tf.constant(1. / self.distance_threshold, tf.float32,
                             flat_field.shape)
    zeros = tf.zeros_like(flat_field)
    flat_field = tf.where(flat_field > inv_thresh, flat_field, zeros)
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
    coords = peak_local_max(
      np.squeeze(field), min_distance=self.distance_threshold,
      num_peaks=self.num_objects,
      exclude_border=False)
    label[:coords.shape[0],1:3] = coords
    label[:coords.shape[0],0] = np.arange(coords.shape[0])
    return label
  
  @property
  def fielded(self):
    def map_func(image, label):
      return image, self.to_field(label)
    return self.preprocessed.map(map_func, self.num_parallel_calls)

  @property
  def visualized(self):
    def map_func(image, label):
      return image, self.to_field(label), label
    return self.dataset.map(map_func, self.num_parallel_calls)

  @property
  def tiled(self):
    """Tile the batched, fielded dataset."""
    def map_func(images, fields):
      fields = tf.pad(fields, [
        [0,0],
        [0,self.image_shape[0] - (self.image_shape[0] % self.tile_shape[0])],
        [0,self.image_shape[1] - (self.image_shape[1] % self.tile_shape[1])],
        [0,0]], 'REFLECT')
      images = tf.pad(images, [
        [0,0],
        [self.pad, self.pad + self.image_shape[0] - (self.image_shape[0] % self.tile_shape[0])],
        [self.pad, self.pad + self.image_shape[1] - (self.image_shape[1] % self.tile_shape[1])],
        [0,0]], 'REFLECT')
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
    return self.fielded.map(map_func, num_parallel_calls=self.num_parallel_calls)

  def untile_single(self, tiles):
    """Untile from `self.num_tiles` numpy tiles, corresponding to single image.

    Does not support batch-ordered inputs
    
    :param tiles: iterable of `self.num_tiles` numpy tiles each with
    `tile_shape`. 
    :returns: an image reconstructed from `tiles`
    :rtype: ndarray

    """
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

  def postprocess_for_training(self, dataset):
    return (dataset.repeat(-1)
            .prefetch(self.prefetch_buffer_size))

  def postprocess_for_evaluation(self, dataset):
    return (dataset.repeat(-1)
            .prefetch(self.prefetch_buffer_size))

  @property
  def training_input(self):
    return self.postprocess_for_training(self.tiled)

  @property
  def eval_input(self):
    return self.postprocess_for_evaluation(self.tiled)


class UnlabeledData(Data):
  def __init__(self, *args, **kwargs):
    """Hold data with no labels.

    Stores the "mode" background image. Provides a sample_and_annotate method
    that returns a new AugmentationData object with the sampling annotated and
    labeled.

    """
    kwargs['parse_entry'] = image_from_proto
    kwargs['encode_entry'] = proto_from_image
    super().__init__(*args, **kwargs)
    self.background = kwargs.get('background')    

    accumulators = {}
    # if self.background is None:
    #   accumulators['background'] = UnlabeledData.mode_background_accumulator
      
    # aggregates = self.accumulate(accumulators)
    # for k,v in aggregates.items():
    #   setattr(self, k, v)

    # self._kwargs['background'] = self.background

  def preprocess(self, dataset):
    """Responsible for converting dataset to batched `(image, dummy_label)` form."""
    def map_func(image):
      return image, tf.zeros(self.label_shape, tf.float32, name='dummy')
    dataset = dataset.map(map_func, self.num_parallel_calls)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    return dataset
    
  def sample_and_annotate(self, sampling, oracle, record_name):
    """Draw a sampling from the dataset and annotate each example, and save

    :param sampling: 1D boolean or "counts" array indexing the dataset.
    :param record_name: tfrecord path to save the newly annotated dataset to.
    :param oracle: a callable with type `(image,idx) -> scene`
    :returns: new AugmentationData object with scenes for the sampled points

    """
    logger.debug(f"sampling: {sampling}")
    s = tf.constant(sampling, dtype=tf.int64)
    dataset = self.dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, entry):
      return tf.data.Dataset.from_tensors((idx, entry)).repeat(s[idx])
    dataset = dataset.flat_map(map_func)

    writer = tf.python_io.TFRecordWriter(record_name)
    if tf.executing_eagerly():
      for idx, image in dataset:
        scene = oracle(image, idx)
        scene = (np.array(scene[0][0]), np.array(scene[0][1])), np.array(scene[1])
        writer.write(proto_from_scene(scene))
    else:
      get_next = dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        while True:
          try:
            idx, image = sess.run(get_next)
            scene = oracle(image, idx)
            writer.write(proto_from_scene(scene))
          except tf.errors.OutOfRangeError:
            break
    writer.close()

    kwargs = self._kwargs.copy()
    kwargs['size'] = np.sum(sampling)
    return AugmentationData(record_name, **kwargs)

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
      return as_float(np.argmax(agg, axis=-1))

    idx = (255. * image).astype(np.int64)
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        for k in range(image.shape[2]):
          agg[i,j,k,idx[i,j,k]] += 1
    return agg
    
      
class AugmentationData(Data):
  def __init__(self, *args, **kwargs):
    kwargs['parse_entry'] = scene_from_proto
    kwargs['encode_entry'] = proto_from_scene
    super().__init__(*args, **kwargs)
    self.background = kwargs.get('background')

    accumulators = {}
    if self.background is None:
      accumulators['background'] = AugmentationData.mean_background_accumulator
      
    aggregates = self.accumulate(accumulators)
    for k,v in aggregates.items():
      setattr(self, k, v)

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

    For now, just checks if each object position is inside the shape.

    :param label: the prospective label, numpy (num_objects, >=3)
    :returns: boolean

    """
    positions = label[:,1:3]
    present = label[:,0].astype(bool)
    indices = np.where(present, positions, np.zeros_like(positions))
    positions_good = np.all(img.inside(indices, self.image_shape))
    theta_good = True
    return positions_good and theta_good

  def draw(self):
    """Draw a new point.

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
      
  def preprocess(self, dataset):
    """Call the augment function."""
    dataset = dataset.repeat(-1).batch(self.batch_size, drop_remainder=True)
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
