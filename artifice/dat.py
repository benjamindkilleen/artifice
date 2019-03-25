"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils. A "scene" is the info needed for a single, labeled
example. It should always be a 2-tuple, usually (image, annotation),
although it could also be (image, (annotation, label)).

"""

import numpy as np
import tensorflow as tf
from artifice import tform
from artifice.utils import img
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

  TODO: allow for loading an unlabelled example. Can do this by including a
  boolean in the data? That's one possibility.

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
  
  def __init__(self, data, **kwargs):
    """args:
    :param data: a tf.data.Dataset OR tfrecord file name(s) OR another Data
      subclass. In this case, the other object's _dataset is adopted, allows
      kwargs to be overwritten.
    """

    self.image_shape = kwargs.get('image_shape', None)
    self.tile_shape = kwargs.get('tile_shape', [32, 32, 1])
    self.pad = kwargs.get('pad', 0)
    self.distance_threshold = kwargs.get('distance_threshold', 20.)
    self.batch_size = kwargs.get('batch_size', 1)
    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    self.parse_entry = kwargs.get('parse_entry', example_from_proto)
    self.encode_entry = kwargs.get('encode_entry', proto_from_example)
    self.prefetch_buffer_size = kwargs.get('prefetch_buffer_size', self.batch_size)
    self.num_shuffle = kwargs.get('num_shuffle', 10000)
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
    
  def save(self, record_name):
    """Save the dataset to record_name."""
    save_dataset(record_name, self._dataset,
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
      dataset = self._dataset.skip(sum(splits[:i])).take(n)
      if types is None or i >= len(types):
        datas.append(type(self)(dataset, **self._kwargs))
      elif types[i] is None:
        datas.append(None)
      else:
        datas.append(types[i](dataset, **self._kwargs))
      
    return datas

  def sample(self, sampling):
    """Draw a sampling from the dataset, returning a new dataset of the same type.

    :param sampling: 1-D array indexing the dataset, e.g. a boolean array, with
    possible repetitions.
    :returns: new data object with examples selected by sampling.
    :rtype: a Data subclass, same as self

    """
    sampling = tf.constant(sampling, dtype=tf.int64)
    dataset = self._dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, example):
      return tf.data.Dataset.from_tensors(example).repeat(sampling[idx])
    dataset = dataset.flat_map(map_func)
    return type(self)(dataset, **self._kwargs)
  
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
      for entry in self._dataset:
        for k, acc in accumulators.items():
          aggregates[k] = acc(self.as_numpy(entry), aggregates[k])
    else:
      iterator = self._dataset.make_initializable_iterator()
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

  def preprocess(self, dataset):
    """Responsible for converting dataset to (image, label) form.

    Can be overwritten by subclasses to perform augmentation."""
    return dataset
  
  def postprocess_for_training(self, dataset):
    return (dataset.shuffle(self.num_shuffle)
            .repeat(-1)
            .batch(self.batch_size)
            .prefetch(self.prefetch_buffer_size))

  def postprocess_for_evaluation(self, dataset):
    return (dataset.repeat(-1)
            .batch(self.batch_size)
            .prefetch(self.prefetch_buffer_size))
    
  @property
  def dataset(self):
    return self.preprocess(self._dataset)

  def to_field(self, label):
    """Create a distance annotation with from `label`.

    :param label: label for the example with shape (num_objects, label_dim)

    """
    # Create the distance for each object, then conditionally take
    positions = tf.cast(label[:,1:3], tf.float32) # (num_objects, 2)
    # TODO: fix position to INF for objects not present
    
    # indices: (M*N, 2)
    indices = np.array([np.array([i,j])
                        for i in range(self.image_shape[0])
                        for j in range(self.image_shape[1])])
    indices = tf.constant(indices, dtype=tf.float32)

    # indices: (M*N, 1, 2), positions: (1, num_objects, 2)
    indices = tf.expand_dims(indices, axis=1)
    positions = tf.expand_dims(positions, axis=0)

    # distances: (M*N, num_objects)
    distances = tf.norm(indices - positions, axis=2)
    
    # take inverse distance
    eps = tf.constant(0.001)
    flat_field = tf.reciprocal(tf.reduce_min(distances, axis=1) + eps)

    # zero the inverse distances outside of threshold
    inv_thresh = tf.constant(1. / self.distance_threshold, tf.float32,
                             flat_field.shape)
    zeros = tf.zeros_like(flat_field)
    flat_field = tf.where(flat_field > inv_thresh, flat_field, zeros)
    return tf.reshape(flat_field, self.image_shape)
  
  @property
  def fielded(self):
    def map_func(image, label):
      return image, self.to_field(label)
    return self.dataset.map(map_func, self.num_parallel_calls)

  @property
  def tiled(self):
    """Tile the fielded dataset."""
    def map_func(image, field):
      even_pad = [
        [0,self.image_shape[0] - (self.image_shape[0] % self.tile_shape[0])],
        [0,self.image_shape[1] - (self.image_shape[1] % self.tile_shape[1])],
        [0,0]]
      image = tf.pad(image, even_pad)
      field = tf.pad(field, even_pad)
      model_pad = [[self.pad,self.pad], [self.pad,self.pad], [0,0]]
      image = tf.pad(image, model_pad)
      image_tiles = []
      field_tiles = []
      for i in range(0, self.image_shape[0], self.tile_shape[0]):
        for j in range(0, self.image_shape[1], self.tile_shape[1]):
          image_tiles.append(image[
            i:i + self.tile_shape[0] + 2*self.pad,
            j:j + self.tile_shape[1] + 2*self.pad])
          field_tiles.append(field[i:i + self.tile_shape[0],
                                   j:j + self.tile_shape[1]])
      images = tf.data.Dataset.from_tensor_slices(image_tiles)
      fields = tf.data.Dataset.from_tensor_slices(field_tiles)
      out = tf.data.Dataset.zip((images, fields))
      logger.debug(f"tiled: {out}")
      return out
    return self.fielded.flat_map(map_func)

  def untile(self):
    pass

  @property
  def training_input(self):
    out = self.postprocess_for_training(self.tiled)
    return out

  @property
  def eval_input(self):
    return self.postprocess_for_evaluation(self.tiled)

  
class AugmentationData(Data):
  def __init__(self, *args, **kwargs):
    super().__init__(*args,
                     parse_entry=scene_from_proto,
                     encode_entry=proto_from_scene,
                     **kwargs)
    self.labels = None          # accumulate labels
    self.num_examples = kwargs.get('num_examples', 10000)
    self.num_objects = kwargs.get('num_objects', 2)
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

    :param label: the prospective label, numpy (num_objects, >3)
    :returns: boolean

    """
    positions = label[:,1:3]
    present = label[:,0].astype(bool)
    indices = np.where(present, positions, np.zeros_like(positions))
    return np.all(img.inside(indices, self.image_shape))

  def draw(self, n):
    """Draw `n` new points.

    :param n: number of labels to draw
    :returns: array of labels
    :rtype: 

    """
    labels = np.ones((n, self.num_objects, 3), dtype=np.float32)
    labels[:,:,1:3] = np.random.uniform([0,0], self.image_shape[:2],
                                        size=(n,self.num_objects,2))
    return labels

  def sample(self):
    """Sample `num_examples` new valid labels.

    :returns: valid labels.
    :rtype: 

    """
    n = self.num_examples
    labels = np.ones((n, self.num_objects, 3), dtype=np.float32)
    indices = np.ones(n, dtype=bool)
    while n > 0:
      logger.debug(f"sample: drawing {n} points")
      draws = self.draw(n)
      valid = np.array([self.valid(draw) for draw in draws])
      valid_draws = draws[valid]
      i = self.num_examples - n
      labels[i:i + valid_draws.shape[0]] = valid_draws
      n -= valid_draws.shape[0]
    logger.debug("done")
    return labels
      
  def preprocess(self, dataset):
    """Call the augment function."""
    return self.augment(dataset)

  def augment(self, dataset):
    """Generate the desired labels and then map them over the original set."""
    labels = self.sample()
    new_label_set = tf.data.Dataset.from_tensor_slices(labels)
    dataset = dataset.repeat(-1)
    zip_set = tf.data.Dataset.zip((new_label_set, dataset))
    
    def map_func(new_label, scene):
      example, annotation = scene
      image, label = example
      return tform.transform(image, label, annotation, new_label,
                             num_objects=self.num_objects,
                             background=self.background)
    
    return zip_set.map(map_func, self.num_parallel_calls)
  
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
